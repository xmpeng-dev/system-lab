"""
EP communication–computation overlap scheduler.

[Megatron] Dual CUDA-stream strategy: dispatch runs on a dedicated comm
stream while the shared expert executes on the compute stream, then
combine overlaps similarly.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from simple_moe.moe_layer import MoELayer


class EPOverlapScheduler:
    """
    Overlaps MoE dispatch/combine communication with shared-expert compute.

    Timeline (forward):

        Compute Stream: [Router] ──────────── [Expert GEMM] ── [+Shared]
        Comm Stream:              [Dispatch]                  [Combine]
                                   └── overlap ─┘
    """

    def __init__(self) -> None:
        self._comm_stream: Optional[torch.cuda.Stream] = None

    @property
    def comm_stream(self) -> torch.cuda.Stream:
        if self._comm_stream is None:
            self._comm_stream = torch.cuda.Stream()
        return self._comm_stream

    # ------------------------------------------------------------------

    def forward_with_overlap(
        self,
        moe_layer: MoELayer,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run the four MoE stages with dispatch/combine on a separate stream.
        """
        compute_stream = torch.cuda.current_stream()

        # Stage 1: Route (compute stream).
        probs, topk_idx, rmap, aux_loss = moe_layer.router(hidden_states)

        # Stage 2: Dispatch (comm stream).
        record = torch.cuda.Event()
        compute_stream.record_event(record)
        self.comm_stream.wait_event(record)

        with torch.cuda.stream(self.comm_stream):
            dispatched, meta = moe_layer.dispatcher.dispatch(
                hidden_states, probs, topk_idx, rmap
            )
        dispatch_done = torch.cuda.Event()
        self.comm_stream.record_event(dispatch_done)

        # Shared expert on compute stream (overlaps with dispatch).
        shared_out: torch.Tensor | None = None
        if moe_layer.shared_expert is not None:
            shared_out = moe_layer.shared_expert(hidden_states)

        # Wait for dispatch to finish before expert compute.
        compute_stream.wait_event(dispatch_done)

        # Stage 3: Expert compute (compute stream).
        expert_out = moe_layer.experts(dispatched, meta.tokens_per_expert)

        # Stage 4: Combine (comm stream).
        record2 = torch.cuda.Event()
        compute_stream.record_event(record2)
        self.comm_stream.wait_event(record2)

        with torch.cuda.stream(self.comm_stream):
            combined = moe_layer.dispatcher.combine(expert_out, meta)
        combine_done = torch.cuda.Event()
        self.comm_stream.record_event(combine_done)
        compute_stream.wait_event(combine_done)

        if shared_out is not None:
            combined = combined + shared_out

        return combined, aux_loss
