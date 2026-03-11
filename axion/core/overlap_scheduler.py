"""Axion OverlapScheduler: Chunk-based compute–communication overlap for MoE.

Overview
--------
Standard MoE execution serialises AllToAll and Expert FFN::

    ─── A2A dispatch (10–20 ms) ───► ─── Expert FFN (5–15 ms) ───►
        ─── A2A combine (10–20 ms) ───►

    Total wall time ≈ A2A_dispatch + FFN + A2A_combine

OverlapScheduler splits the token sequence into N *chunks* and pipelines
AllToAll with Expert FFN across chunks::

    Chunk 1:  ─ dispatch ──────────────────────────────────────────────►
              ╰──────── FFN ─────────────►
                        ╰───── combine ───────────────────────────────►

    Chunk 2:            ─ dispatch ──────────────────────────────────►
                                   ╰──── FFN ──────────────►
                                         ╰─── combine ─────────────►

    Total wall time ≈ A2A_dispatch/N + FFN + A2A_combine/N   (ideal)
    Speedup        ≈ (A2A + FFN) / (A2A/N + FFN)

The scheduler uses two CUDA streams — one for AllToAll communication and one
for Expert FFN compute — and synchronises them with :class:`torch.cuda.Event`
objects.  This mirrors the dual-stream design in Megatron-Core's EP overlap
implementation (arXiv 2603.07685, §5.2).

Design constraints
------------------
* Chunk splitting is along the *token dimension* (dim 0).  Expert GEMM
  semantics are per-token-independent, so chunks are numerically equivalent.
* The number of chunks ``N`` must divide evenly into the total token count.
* The theoretical sweet spot is ``N ≈ round(A2A_ms / FFN_chunk_ms)``; in
  practice ``N = 2`` or ``N = 4`` works well on MI300X intra-node A2A.

References
----------
* Feature 3 Design: OverlapScheduler (axion/feature3/design/Design.md)
* Megatron-Core MoE paper (arXiv 2603.07685), §5.2: EP Communication Overlap
* FlowMoE (NeurIPS'25): DAG-based MoE scheduling
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch
import torch.distributed as dist

from axion.core.comm_tensor import (
    CommLayout,
    CommSpec,
    CommTensor,
    RoutingTable,
    _scale_comm_spec,
)
from axion.core.moe_dispatcher import MoEDispatchConfig, MoEDispatcher


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class OverlapConfig:
    """Configuration for :class:`OverlapScheduler`.

    Parameters
    ----------
    num_chunks:
        Number of token-dimension chunks.  ``1`` disables overlap (serial
        baseline).  ``2`` or ``4`` is recommended for MI300X.
    dispatch_config:
        MoE dispatcher configuration (num_experts, topk, hidden_dim, …).
    comm_stream:
        CUDA stream for AllToAll communication.  If ``None`` a new stream is
        created automatically.
    compute_stream:
        CUDA stream for Expert FFN computation.  If ``None`` a new stream is
        created automatically.
    """

    num_chunks: int
    dispatch_config: MoEDispatchConfig
    comm_stream: Optional[torch.cuda.Stream] = None
    compute_stream: Optional[torch.cuda.Stream] = None


# ---------------------------------------------------------------------------
# OverlapScheduler
# ---------------------------------------------------------------------------

class OverlapScheduler:
    """Pipelines AllToAll dispatch/combine with Expert FFN across token chunks.

    The scheduler is stateless between forward passes.  Streams and events are
    created once at construction time and reused.

    Example
    -------
    ::

        sched = OverlapScheduler(OverlapConfig(
            num_chunks=4,
            dispatch_config=MoEDispatchConfig(
                num_experts=64, topk=2, hidden_dim=4096,
            ),
        ))

        output = sched.forward(
            hidden=attention_output,          # [S, H]
            routing_table=routing_table,      # from RoutingTable.build()
            expert_fn=my_grouped_gemm,        # [R, H] → [R, H]
        )

    Correctness guarantee
    ---------------------
    ``sched.forward(hidden, rt, fn)`` is numerically equivalent to the
    serial dispatch → fn → combine path up to floating-point rounding from
    addition order (verified by ``tests/test_overlap_scheduler.py``).
    """

    def __init__(self, config: OverlapConfig) -> None:
        self._cfg = config
        self._dispatcher = MoEDispatcher(config.dispatch_config)

        # Create CUDA streams if not provided.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._comm_stream: Optional[torch.cuda.Stream] = None
        self._compute_stream: Optional[torch.cuda.Stream] = None
        if device.type == "cuda":
            self._comm_stream = config.comm_stream or torch.cuda.Stream()
            self._compute_stream = config.compute_stream or torch.cuda.Stream()

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden: torch.Tensor,
        routing_table: RoutingTable,
        expert_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Chunked pipeline: dispatch → expert_fn → combine with overlap.

        Parameters
        ----------
        hidden:
            Shape ``[num_tokens, hidden_dim]``.  INTERLEAVED layout.
        routing_table:
            Routing decisions for this step.
        expert_fn:
            Callable that applies all local experts to a tensor of shape
            ``[recv_tokens_per_chunk, hidden_dim]`` and returns a tensor of
            the same shape.  Called once per chunk on the compute stream.

        Returns
        -------
        torch.Tensor
            Shape ``[num_tokens, hidden_dim]``.  Numerically equivalent to
            the serial (non-overlapped) result.
        """
        n = self._cfg.num_chunks
        num_tokens = hidden.shape[0]

        if n > 1 and num_tokens % n != 0:
            raise ValueError(
                f"num_tokens ({num_tokens}) must be divisible by "
                f"num_chunks ({n}) for overlap scheduling."
            )

        if n == 1 or self._comm_stream is None:
            # Fast path: no overlap, delegate to MoEDispatcher directly.
            return self._dispatcher.full_forward(hidden, routing_table, expert_fn)

        return self._chunked_pipeline(hidden, routing_table, expert_fn, n)

    # ------------------------------------------------------------------
    # Internal: chunked pipeline
    # ------------------------------------------------------------------

    def _chunked_pipeline(
        self,
        hidden: torch.Tensor,
        routing_table: RoutingTable,
        expert_fn: Callable[[torch.Tensor], torch.Tensor],
        n: int,
    ) -> torch.Tensor:
        """Execute the N-chunk pipeline with dual CUDA streams.

        Pipeline for chunk i (0-indexed)::

            comm_stream:    dispatch[i]        combine[i]
            compute_stream:            FFN[i-1]          FFN[i]  ...

        Cross-stream synchronisation via torch.cuda.Event objects ensures that
        FFN[i] only starts after dispatch[i] has completed and that combine[i]
        only starts after FFN[i] has completed.
        """
        num_tokens = hidden.shape[0]
        hidden_dim = hidden.shape[-1]
        chunk_size = num_tokens // n

        comm_stream = self._comm_stream
        compute_stream = self._compute_stream
        current_stream = torch.cuda.current_stream()

        # Build CommSpec for a single chunk (scale counts by 1/N).
        full_comm_spec = CommSpec(
            send_counts=routing_table.send_counts.tolist(),
            recv_counts=routing_table.recv_counts.tolist(),
            group=self._cfg.dispatch_config.group,
            rank=dist.get_rank(self._cfg.dispatch_config.group)
            if dist.is_initialized() else 0,
            world_size=self._cfg.dispatch_config.world_size,
        )
        chunk_spec = _scale_comm_spec(full_comm_spec, 1.0 / n)

        # Split hidden and routing permutation indices into N chunks.
        hidden_chunks = hidden.chunk(n, dim=0)  # list of [chunk_size, H]
        topk = routing_table.expert_indices.shape[1]
        slot_per_chunk = chunk_size * topk
        src_chunks = routing_table.src_indices.chunk(n)
        inv_chunks = routing_table.inverse_src_indices.chunk(n)
        weight_chunks = routing_table.routing_weights.chunk(n, dim=0)
        expert_chunks = routing_table.expert_indices.chunk(n, dim=0)

        # Storage for combined outputs (in BLOCKED_BY_DST slot order).
        combined_slots: List[Optional[torch.Tensor]] = [None] * n
        # Events for cross-stream sync.
        dispatch_done: List[Optional[torch.cuda.Event]] = [None] * n
        ffn_done: List[Optional[torch.cuda.Event]] = [None] * n

        # ----------------------------------------------------------------
        # Phase 1: launch dispatch for all chunks on comm_stream.
        # ----------------------------------------------------------------
        dispatched_cts: List[Optional[CommTensor]] = [None] * n
        comm_stream.wait_stream(current_stream)

        with torch.cuda.stream(comm_stream):
            for i in range(n):
                # Build per-chunk RoutingTable (only the fields needed for
                # from_dense and alltoall_dispatch).
                chunk_rt = _make_chunk_routing_table(
                    expert_chunks[i], weight_chunks[i],
                    routing_table.send_counts // n,
                    routing_table.recv_counts // n,
                    src_chunks[i], inv_chunks[i],
                    routing_table.tokens_per_expert // n,
                )

                # Pack and dispatch (zero-pack AllToAll on comm_stream).
                ct_send = CommTensor.from_dense(hidden_chunks[i], chunk_rt, chunk_spec)
                ct_recv = ct_send.alltoall_dispatch()
                dispatched_cts[i] = ct_recv

                # Record event so compute_stream knows dispatch[i] is done.
                ev = torch.cuda.Event()
                ev.record(comm_stream)
                dispatch_done[i] = ev

        # ----------------------------------------------------------------
        # Phase 2: Expert FFN for each chunk on compute_stream, overlapping
        #          with combine dispatches.
        # ----------------------------------------------------------------
        expert_cts: List[Optional[CommTensor]] = [None] * n
        with torch.cuda.stream(compute_stream):
            for i in range(n):
                # Wait until dispatch[i] is complete.
                compute_stream.wait_event(dispatch_done[i])

                # Expert FFN on the received tokens.
                expert_out = expert_fn(dispatched_cts[i].data)
                expert_cts[i] = CommTensor(
                    expert_out,
                    CommLayout.BLOCKED_BY_SRC,
                    dispatched_cts[i].comm_spec,
                )

                ev = torch.cuda.Event()
                ev.record(compute_stream)
                ffn_done[i] = ev

        # ----------------------------------------------------------------
        # Phase 3: combine AllToAll for each chunk on comm_stream,
        #          overlapping with remaining FFN chunks.
        # ----------------------------------------------------------------
        combined_cts: List[Optional[CommTensor]] = [None] * n
        with torch.cuda.stream(comm_stream):
            for i in range(n):
                # Wait until FFN[i] is complete before sending results back.
                comm_stream.wait_event(ffn_done[i])

                combined_ct = expert_cts[i].alltoall_combine()
                combined_cts[i] = combined_ct

        # ----------------------------------------------------------------
        # Phase 4: restore token order and merge routing weights.
        # ----------------------------------------------------------------
        current_stream.wait_stream(comm_stream)
        current_stream.wait_stream(compute_stream)

        output_chunks: List[torch.Tensor] = []
        for i in range(n):
            chunk_rt = _make_chunk_routing_table(
                expert_chunks[i], weight_chunks[i],
                routing_table.send_counts // n,
                routing_table.recv_counts // n,
                src_chunks[i], inv_chunks[i],
                routing_table.tokens_per_expert // n,
            )
            out_chunk = combined_cts[i].to_dense(chunk_rt)
            output_chunks.append(out_chunk)

        return torch.cat(output_chunks, dim=0)


# ---------------------------------------------------------------------------
# Helper: build a minimal RoutingTable for one chunk
# ---------------------------------------------------------------------------

def _make_chunk_routing_table(
    expert_indices: torch.Tensor,
    routing_weights: torch.Tensor,
    send_counts: torch.Tensor,
    recv_counts: torch.Tensor,
    src_indices: torch.Tensor,
    inverse_src_indices: torch.Tensor,
    tokens_per_expert: torch.Tensor,
) -> RoutingTable:
    """Construct a RoutingTable for a single chunk from pre-split tensors."""
    return RoutingTable(
        expert_indices=expert_indices,
        routing_weights=routing_weights,
        send_counts=send_counts,
        recv_counts=recv_counts,
        src_indices=src_indices,
        inverse_src_indices=inverse_src_indices,
        tokens_per_expert=tokens_per_expert,
    )
