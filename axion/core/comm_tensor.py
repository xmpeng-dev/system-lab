"""Axion CommTensor: Communication-native tensor with layout-aware memory management.

Design philosophy
-----------------
Standard MoE dispatch treats communication as a post-processing step::

    attention_output [S, H]  (INTERLEAVED)
         ↓  pack / sort_by_dst_rank()   ← copy #1
    sorted_buffer [S, H]    (BLOCKED_BY_DST)
         ↓  rccl_alltoall()
    dispatched [R, H]       (BLOCKED_BY_SRC)
         ↓  Expert FFN
    expert_out [R, H]       (BLOCKED_BY_SRC)
         ↓  rccl_alltoall()
    combined_sorted [S, H]  (BLOCKED_BY_DST)
         ↓  unpack / index_select()     ← copy #2
    combined [S, H]         (INTERLEAVED)

CommTensor elevates *layout* to a first-class attribute of the tensor itself.
Knowing the layout at the point of allocation:

1. Enables zero-pack dispatch when the buffer is *already* BLOCKED_BY_DST.
2. Catches layout errors at the Python layer (runtime type-check) instead of
   producing silent numerical errors.
3. Allows the OverlapScheduler to reason about layout transitions statically
   and insert CUDA events at the correct points.

This file implements the type system described in
``axion/ir/IR_Phase1_Type_System_Design.md`` as executable Python code, and
provides the low-level primitives consumed by ``moe_dispatcher.py`` and
``overlap_scheduler.py``.

References
----------
* Axion IR Phase 1 Type System Design (axion/ir/)
* Feature 4 Design: CommTensor zero-copy (axion/feature4/design/Design.md)
* Megatron-Core MoE paper (arXiv 2603.07685)
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# CommLayout — physical memory layout enum
# ---------------------------------------------------------------------------

class CommLayout(enum.Enum):
    """Physical memory layout of a CommTensor.

    The layout determines what communication or compute operation the tensor is
    *immediately ready* for — without any additional pack or unpack copy.

    Layout transition diagram for a single MoE forward pass::

        INTERLEAVED
            │  CommTensor.from_dense()   (sort / index_copy_)
            ▼
        BLOCKED_BY_DST
            │  Comm.alltoall_dispatch()  (NCCL/RCCL AllToAll, zero-pack DMA)
            ▼
        BLOCKED_BY_SRC
            │  Expert FFN               (compute, no reorder needed)
            ▼
        BLOCKED_BY_SRC
            │  Comm.alltoall_combine()  (NCCL/RCCL AllToAll, zero-pack DMA)
            ▼
        BLOCKED_BY_DST
            │  CommTensor.to_dense()    (index_copy_ to restore token order)
            ▼
        INTERLEAVED

    For FSEP expert parameter communication::

        BLOCKED_BY_EXPERT → Comm.allgather() → DenseTensor
    """

    # -----------------------------------------------------------------------
    # Primary layouts used in the forward MoE path
    # -----------------------------------------------------------------------

    INTERLEAVED = "interleaved"
    """Tokens in original sequential (batch × seq_len) order.

    This is the standard PyTorch tensor layout.  A CommTensor in this state is
    *ready for attention computation* but *not yet ready for AllToAll dispatch*
    (a sort / pack step is required first).
    """

    BLOCKED_BY_DST = "blocked_by_dst"
    """Memory grouped by *destination* GPU.

    Physical layout::

        [tokens_for_rank_0 | tokens_for_rank_1 | … | tokens_for_rank_N]

    This is the layout NCCL/RCCL requires on the *send side* of AllToAll.
    Allocating or converting to this layout before the A2A eliminates the
    ``sort_by_dst_rank`` pack copy inside the AllToAll kernel.
    """

    BLOCKED_BY_SRC = "blocked_by_src"
    """Memory grouped by *source* GPU.

    Physical layout::

        [tokens_from_rank_0 | tokens_from_rank_1 | … | tokens_from_rank_N]

    This is the layout NCCL/RCCL *produces* on the receive side of AllToAll.
    Expert FFN can process tokens directly from this layout — no unpack or
    reorder is needed before feeding the grouped GEMM.
    """

    BLOCKED_BY_EXPERT = "blocked_by_expert"
    """Expert parameters grouped by Expert ID.

    Physical layout::

        [expert_0_params | expert_1_params | … | expert_E_params]

    Used for FSEP (Fully Sharded Expert Parallel) AllGather.  Shards are
    contiguous per-expert, so the gather requires no reordering.
    """

    SPARSE_CSR = "sparse_csr"
    """Sparse Compressed Sparse Row format (values, col_indices, row_ptr).

    Used when routing sparsity exceeds ~70 %: empty experts are skipped and
    the effective communication volume is reduced proportionally.  The
    ``CommTensor.data`` field holds a ``torch.sparse_csr_tensor`` in this
    case.
    """


# ---------------------------------------------------------------------------
# CommSpec — communication parameters
# ---------------------------------------------------------------------------

@dataclass
class CommSpec:
    """All information needed to execute the AllToAll for this CommTensor.

    Computing ``send_counts`` / ``recv_counts`` requires a global barrier
    (allreduce of token counts).  By attaching these to the CommTensor at
    routing time they are computed *once* and reused for dispatch, combine, and
    overlap chunking without additional synchronisation.
    """

    # Number of elements (tokens × hidden_dim) to send to each rank.
    send_counts: List[int]

    # Number of elements (tokens × hidden_dim) to receive from each rank.
    recv_counts: List[int]

    # Process group used for AllToAll.  ``None`` means the default group.
    group: Optional[dist.ProcessGroup] = None

    # Rank of *this* process within ``group``.
    rank: int = 0

    # Total number of processes in ``group``.
    world_size: int = 1

    @property
    def total_send(self) -> int:
        """Total number of elements sent across all ranks."""
        return sum(self.send_counts)

    @property
    def total_recv(self) -> int:
        """Total number of elements received across all ranks."""
        return sum(self.recv_counts)


# ---------------------------------------------------------------------------
# RoutingTable — per-step routing decisions
# ---------------------------------------------------------------------------

@dataclass
class RoutingTable:
    """Routing decisions for one MoE forward pass.

    Computed once by the gate/router and consumed by dispatch, combine, and
    layout-conversion helpers.  All tensors live on the same device as the
    hidden states.

    Invariants
    ----------
    * ``expert_indices.shape == (num_tokens, topk)``
    * ``routing_weights.shape == (num_tokens, topk)``
    * ``src_indices.shape == (num_tokens * topk,)`` — permutation that converts
      INTERLEAVED → BLOCKED_BY_DST.
    * ``inverse_src_indices.shape == (num_tokens * topk,)`` — inverse permutation
      that converts BLOCKED_BY_DST → INTERLEAVED.
    * ``send_counts.shape == (world_size,)``
    * ``recv_counts.shape == (world_size,)``
    * ``tokens_per_expert.shape == (num_experts,)``
    """

    # Expert assignments per token.
    expert_indices: torch.Tensor  # [num_tokens, topk]

    # Softmax routing weights per token.
    routing_weights: torch.Tensor  # [num_tokens, topk]

    # Number of tokens sent to each GPU (from send_counts in CommSpec).
    send_counts: torch.Tensor  # [world_size]

    # Number of tokens received from each GPU.
    recv_counts: torch.Tensor  # [world_size]

    # Permutation: position i in BLOCKED_BY_DST ← token src_indices[i] in INTERLEAVED.
    src_indices: torch.Tensor  # [num_tokens * topk]

    # Inverse permutation: position inverse_src_indices[i] in INTERLEAVED ← slot i.
    inverse_src_indices: torch.Tensor  # [num_tokens * topk]

    # How many tokens arrived at each expert on *this* GPU.
    tokens_per_expert: torch.Tensor  # [num_local_experts]

    @staticmethod
    def build(
        expert_indices: torch.Tensor,
        routing_weights: torch.Tensor,
        num_experts: int,
        expert_to_rank: Optional[torch.Tensor] = None,
        world_size: int = 1,
        group: Optional[dist.ProcessGroup] = None,
    ) -> RoutingTable:
        """Build a RoutingTable from raw gate outputs.

        Parameters
        ----------
        expert_indices:
            Shape ``[num_tokens, topk]``.  Expert ID per token per slot,
            as returned by ``torch.topk`` on gate logits.
        routing_weights:
            Shape ``[num_tokens, topk]``.  Softmax-normalised routing scores.
        num_experts:
            Total number of experts across *all* GPUs.
        expert_to_rank:
            Optional mapping ``expert_id → rank`` (shape ``[num_experts]``).
            When ``None`` experts are distributed round-robin across
            ``world_size`` ranks.
        world_size:
            Size of the Expert Parallel process group.
        group:
            Expert Parallel process group.

        Returns
        -------
        RoutingTable
            Ready to pass to :meth:`CommTensor.from_dense`.
        """
        num_tokens, topk = expert_indices.shape
        device = expert_indices.device

        # Default expert → rank assignment: expert e lives on rank e // (num_experts // world_size)
        if expert_to_rank is None:
            experts_per_rank = max(1, num_experts // world_size)
            expert_to_rank = expert_indices.new_tensor(
                [e // experts_per_rank for e in range(num_experts)]
            )

        # Flatten: shape [num_tokens * topk]
        flat_experts = expert_indices.reshape(-1)
        flat_ranks = expert_to_rank[flat_experts]  # [S * topk]

        # Sort by destination rank to compute BLOCKED_BY_DST permutation.
        src_indices = torch.argsort(flat_ranks, stable=True)  # [S * topk]

        # Inverse permutation.
        inverse_src_indices = torch.empty_like(src_indices)
        inverse_src_indices[src_indices] = torch.arange(
            src_indices.numel(), device=device
        )

        # Count tokens sent to each rank.
        send_counts = torch.bincount(flat_ranks, minlength=world_size).to(device)

        # Exchange send_counts across the group to get recv_counts.
        if world_size > 1 and dist.is_initialized():
            recv_counts = send_counts.clone()
            dist.all_to_all_single(
                recv_counts,
                send_counts,
                group=group,
            )
        else:
            recv_counts = send_counts.clone()

        # Per-expert token counts for the *local* GPU.
        rank = dist.get_rank(group) if (world_size > 1 and dist.is_initialized()) else 0
        local_mask = (expert_to_rank == rank)
        local_experts = torch.where(local_mask)[0]
        tokens_per_expert = torch.zeros(
            local_experts.numel(), dtype=torch.long, device=device
        )
        for local_idx, eid in enumerate(local_experts.tolist()):
            tokens_per_expert[local_idx] = (flat_experts == eid).sum()

        return RoutingTable(
            expert_indices=expert_indices,
            routing_weights=routing_weights,
            send_counts=send_counts,
            recv_counts=recv_counts,
            src_indices=src_indices,
            inverse_src_indices=inverse_src_indices,
            tokens_per_expert=tokens_per_expert,
        )


# ---------------------------------------------------------------------------
# CommTensor — the core type
# ---------------------------------------------------------------------------

class CommTensor:
    """A tensor whose *physical memory layout* is part of its type.

    CommTensor wraps a :class:`torch.Tensor` and attaches a
    :class:`CommLayout` describing how tokens are arranged in memory.
    This makes layout mismatches into explicit Python-level errors rather than
    silent numerical bugs.

    Typical usage
    -------------
    ::

        # 1. Convert attention output to dispatch-ready layout.
        ct = CommTensor.from_dense(hidden, routing_table)
        assert ct.layout == CommLayout.BLOCKED_BY_DST

        # 2. AllToAll dispatch (zero-pack: data is already grouped by rank).
        dispatched = ct.alltoall_dispatch(comm_spec)
        assert dispatched.layout == CommLayout.BLOCKED_BY_SRC

        # 3. Expert FFN (no reorder needed).
        expert_out = my_expert_ffn(dispatched.data)
        expert_ct = CommTensor(expert_out, CommLayout.BLOCKED_BY_SRC, comm_spec)

        # 4. AllToAll combine (zero-pack: expert_out is already grouped by rank).
        combined_ct = expert_ct.alltoall_combine(comm_spec_reverse)
        assert combined_ct.layout == CommLayout.BLOCKED_BY_DST

        # 5. Restore original token order.
        combined = combined_ct.to_dense(routing_table)
        assert combined.shape == hidden.shape

    Notes
    -----
    CommTensor is intentionally **not** a :class:`torch.Tensor` subclass.
    Subclassing ``Tensor`` interacts poorly with ``torch.compile`` and
    ``autograd`` in subtle ways.  The ``.data`` attribute gives direct access
    to the underlying storage for use with standard PyTorch ops.
    """

    def __init__(
        self,
        data: torch.Tensor,
        layout: CommLayout,
        comm_spec: Optional[CommSpec] = None,
    ) -> None:
        """
        Parameters
        ----------
        data:
            Underlying storage tensor.  Its physical layout *must* match
            ``layout``; the caller is responsible for this invariant.
        layout:
            Physical memory layout of ``data``.
        comm_spec:
            Communication parameters (send/recv counts, process group).
            Required for AllToAll operations; optional for compute-only use.
        """
        self._data = data
        self._layout = layout
        self._comm_spec = comm_spec

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def data(self) -> torch.Tensor:
        """The underlying :class:`torch.Tensor` storage."""
        return self._data

    @property
    def layout(self) -> CommLayout:
        """Physical memory layout of this CommTensor."""
        return self._layout

    @property
    def comm_spec(self) -> Optional[CommSpec]:
        """Communication parameters attached at routing time."""
        return self._comm_spec

    @property
    def shape(self) -> torch.Size:
        return self._data.shape

    @property
    def dtype(self) -> torch.dtype:
        return self._data.dtype

    @property
    def device(self) -> torch.device:
        return self._data.device

    def __repr__(self) -> str:
        return (
            f"CommTensor(shape={tuple(self.shape)}, dtype={self.dtype}, "
            f"layout={self.layout.value}, device={self.device})"
        )

    # ------------------------------------------------------------------
    # Factory: INTERLEAVED → BLOCKED_BY_DST
    # ------------------------------------------------------------------

    @staticmethod
    def from_dense(
        hidden: torch.Tensor,
        routing_table: RoutingTable,
        comm_spec: Optional[CommSpec] = None,
    ) -> CommTensor:
        """Convert a dense INTERLEAVED tensor to BLOCKED_BY_DST layout.

        This is the "pack" step, but implemented with
        :func:`torch.Tensor.index_copy_` which generates *coalesced writes*
        rather than random reads, improving HBM throughput on MI300X / H100.

        Parameters
        ----------
        hidden:
            Shape ``[num_tokens, topk, hidden_dim]`` or ``[num_tokens, hidden_dim]``.
            When ``topk > 1`` the tensor is first expanded so each
            (token, expert) pair occupies a separate row.
        routing_table:
            Pre-computed routing decisions; supplies ``src_indices``.
        comm_spec:
            Optional communication spec to attach to the result.

        Returns
        -------
        CommTensor
            Layout ``BLOCKED_BY_DST``, ready for zero-pack AllToAll dispatch.
        """
        num_tokens = hidden.shape[0]
        hidden_dim = hidden.shape[-1]

        # If hidden is [S, H], expand to [S * topk, H] using routing weights.
        # Each token appears topk times (once per expert assignment).
        topk = routing_table.expert_indices.shape[1]
        if hidden.ndim == 2:
            # Repeat each token row topk times: [S, H] → [S * topk, H]
            expanded = hidden.unsqueeze(1).expand(-1, topk, -1).reshape(-1, hidden_dim)
        else:
            expanded = hidden.reshape(-1, hidden_dim)

        # Pre-allocate BLOCKED_BY_DST buffer (coalesced writes beat random reads).
        sorted_buf = torch.empty_like(expanded)
        sorted_buf.index_copy_(0, routing_table.src_indices, expanded)

        return CommTensor(sorted_buf, CommLayout.BLOCKED_BY_DST, comm_spec)

    # ------------------------------------------------------------------
    # Factory: BLOCKED_BY_DST → INTERLEAVED
    # ------------------------------------------------------------------

    def to_dense(self, routing_table: RoutingTable) -> torch.Tensor:
        """Restore INTERLEAVED order from a BLOCKED_BY_DST CommTensor.

        This is the "unpack" / combine step.  Like :meth:`from_dense` it uses
        ``index_copy_`` for coalesced writes.

        Parameters
        ----------
        routing_table:
            Supplies ``inverse_src_indices`` and ``routing_weights``.

        Returns
        -------
        torch.Tensor
            Shape ``[num_tokens, hidden_dim]`` in the original token order,
            with routing weights applied (weighted sum over topk experts).

        Raises
        ------
        ValueError
            If the current layout is not ``BLOCKED_BY_DST``.
        """
        if self._layout not in (CommLayout.BLOCKED_BY_DST, CommLayout.BLOCKED_BY_SRC):
            raise ValueError(
                f"to_dense() expects BLOCKED_BY_DST or BLOCKED_BY_SRC layout, "
                f"got {self._layout}"
            )

        num_slots, hidden_dim = self._data.shape
        topk = routing_table.expert_indices.shape[1]
        num_tokens = num_slots // topk

        # Restore INTERLEAVED order: coalesced writes.
        restored = torch.zeros(num_slots, hidden_dim, dtype=self._data.dtype,
                               device=self._data.device)
        restored.index_copy_(0, routing_table.inverse_src_indices, self._data)

        # Reshape to [S, topk, H] and apply routing weights.
        restored = restored.reshape(num_tokens, topk, hidden_dim)
        weights = routing_table.routing_weights.unsqueeze(-1)  # [S, topk, 1]
        return (restored * weights).sum(dim=1)  # [S, H]

    # ------------------------------------------------------------------
    # AllToAll: BLOCKED_BY_DST → BLOCKED_BY_SRC  (dispatch)
    # ------------------------------------------------------------------

    def alltoall_dispatch(
        self,
        comm_spec: Optional[CommSpec] = None,
    ) -> CommTensor:
        """Execute dispatch AllToAll: BLOCKED_BY_DST → BLOCKED_BY_SRC.

        Because ``self.data`` is already in BLOCKED_BY_DST layout, NCCL/RCCL
        can DMA directly without an additional pack copy.

        Parameters
        ----------
        comm_spec:
            Communication spec to use.  Falls back to ``self.comm_spec`` if
            not provided.

        Returns
        -------
        CommTensor
            Layout ``BLOCKED_BY_SRC``, ready for Expert FFN.

        Raises
        ------
        ValueError
            If layout is not ``BLOCKED_BY_DST``.
        RuntimeError
            If no ``CommSpec`` is available.
        """
        if self._layout != CommLayout.BLOCKED_BY_DST:
            raise ValueError(
                f"alltoall_dispatch() requires BLOCKED_BY_DST layout, "
                f"got {self._layout}"
            )

        spec = comm_spec or self._comm_spec
        if spec is None:
            raise RuntimeError(
                "alltoall_dispatch() requires a CommSpec (set comm_spec on the "
                "CommTensor or pass it explicitly)."
            )

        recv_buf = _alltoall(self._data, spec)
        # Build reverse CommSpec (for combine step): swap send/recv counts.
        reverse_spec = CommSpec(
            send_counts=spec.recv_counts,
            recv_counts=spec.send_counts,
            group=spec.group,
            rank=spec.rank,
            world_size=spec.world_size,
        )
        return CommTensor(recv_buf, CommLayout.BLOCKED_BY_SRC, reverse_spec)

    # ------------------------------------------------------------------
    # AllToAll: BLOCKED_BY_SRC → BLOCKED_BY_DST  (combine)
    # ------------------------------------------------------------------

    def alltoall_combine(
        self,
        comm_spec: Optional[CommSpec] = None,
    ) -> CommTensor:
        """Execute combine AllToAll: BLOCKED_BY_SRC → BLOCKED_BY_DST.

        Because ``self.data`` is already in BLOCKED_BY_SRC layout (output of
        Expert FFN), NCCL/RCCL can DMA directly without an additional pack
        copy.

        Parameters
        ----------
        comm_spec:
            Communication spec to use.  Falls back to ``self.comm_spec`` if
            not provided.

        Returns
        -------
        CommTensor
            Layout ``BLOCKED_BY_DST``; call :meth:`to_dense` to restore token
            order.

        Raises
        ------
        ValueError
            If layout is not ``BLOCKED_BY_SRC``.
        """
        if self._layout != CommLayout.BLOCKED_BY_SRC:
            raise ValueError(
                f"alltoall_combine() requires BLOCKED_BY_SRC layout, "
                f"got {self._layout}"
            )

        spec = comm_spec or self._comm_spec
        if spec is None:
            raise RuntimeError(
                "alltoall_combine() requires a CommSpec."
            )

        recv_buf = _alltoall(self._data, spec)
        return CommTensor(recv_buf, CommLayout.BLOCKED_BY_DST, None)

    # ------------------------------------------------------------------
    # Chunk splitting (for overlap scheduling)
    # ------------------------------------------------------------------

    def chunk(self, num_chunks: int) -> List[CommTensor]:
        """Split along dim 0 into ``num_chunks`` CommTensors.

        Used by :class:`~axion.core.overlap_scheduler.OverlapScheduler` to
        pipeline AllToAll with Expert FFN computation.

        Parameters
        ----------
        num_chunks:
            Number of chunks.  Must divide evenly into ``self.shape[0]``.

        Returns
        -------
        list of CommTensor
            Each chunk has the same layout and a proportionally scaled
            CommSpec.
        """
        if self._data.shape[0] % num_chunks != 0:
            raise ValueError(
                f"Cannot split {self._data.shape[0]} tokens into {num_chunks} "
                f"equal chunks."
            )

        data_chunks = self._data.chunk(num_chunks, dim=0)
        result: List[CommTensor] = []
        for chunk_data in data_chunks:
            # Scale CommSpec counts proportionally.
            chunk_spec: Optional[CommSpec] = None
            if self._comm_spec is not None:
                chunk_spec = _scale_comm_spec(self._comm_spec, 1.0 / num_chunks)
            result.append(CommTensor(chunk_data, self._layout, chunk_spec))
        return result


# ---------------------------------------------------------------------------
# Internal AllToAll helper
# ---------------------------------------------------------------------------

def _alltoall(
    send_tensor: torch.Tensor,
    spec: CommSpec,
) -> torch.Tensor:
    """Low-level AllToAll using variable-count split/concat."""
    if spec.world_size == 1 or not dist.is_initialized():
        # Single-process fast-path: identity.
        return send_tensor.clone()

    hidden_dim = send_tensor.shape[-1]

    # Split send tensor by rank.
    send_list = list(send_tensor.split(
        [c * hidden_dim for c in spec.send_counts], dim=0
    )) if sum(spec.send_counts) > 0 else [
        send_tensor.new_empty(0, hidden_dim) for _ in range(spec.world_size)
    ]

    # Allocate receive buffers.
    recv_list = [
        send_tensor.new_empty(rc * hidden_dim)
        .reshape(rc, hidden_dim)
        for rc in spec.recv_counts
    ]

    dist.all_to_all(recv_list, send_list, group=spec.group)
    return torch.cat(recv_list, dim=0)


def _scale_comm_spec(spec: CommSpec, scale: float) -> CommSpec:
    """Return a new CommSpec with counts scaled by ``scale``."""
    return CommSpec(
        send_counts=[max(0, int(c * scale)) for c in spec.send_counts],
        recv_counts=[max(0, int(c * scale)) for c in spec.recv_counts],
        group=spec.group,
        rank=spec.rank,
        world_size=spec.world_size,
    )
