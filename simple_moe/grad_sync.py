"""
Overlapped gradient synchronization for FSDP + EP.

Registers backward hooks on expert parameters so that gradient AllReduce
starts immediately when each gradient is ready, running on a dedicated
comm stream while backward continues through attention / dispatch layers.

Timeline (backward of one transformer block):

  Compute stream: [MoE combine bwd] → [MoE expert bwd] → [MoE dispatch bwd] → [Attn bwd]
  Comm stream:                          [grad AR expert]  ← overlap →           [still running]
                                                                                ↓
                                        grad AR finishes before optimizer.step()

This is the same principle as PyTorch DDP's bucketed gradient AllReduce.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn


class GradSyncOverlap:
    """
    Async gradient AllReduce that overlaps with backward computation.

    Usage:
        sync = GradSyncOverlap(model, fsdp_group, params_to_sync='expert')
        # training loop:
        loss.backward()       # hooks fire → async AllReduce starts
        sync.finish()         # wait for all AllReduce to complete
        optimizer.step()
    """

    def __init__(
        self,
        model: nn.Module,
        fsdp_group: dist.ProcessGroup,
        sync_filter: str = "expert",
    ) -> None:
        self.group = fsdp_group
        self.world_size = dist.get_world_size(fsdp_group)
        self._comm_stream = torch.cuda.Stream()
        self._pending: List[torch.cuda.Event] = []
        self._handles: List[dist.Work] = []
        self._hooks: List[torch.utils.hooks.RemovableHook] = []

        # Register hooks on matching parameters.
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if self._should_sync(name, sync_filter):
                hook = param.register_post_accumulate_grad_hook(
                    self._make_hook(param)
                )
                self._hooks.append(hook)

    def _should_sync(self, name: str, filter_str: str) -> bool:
        if filter_str == "expert":
            return any(k in name for k in ("expert", "w_gate_up", "w_down", "router"))
        elif filter_str == "all":
            return True
        return filter_str in name

    def _make_hook(self, param: nn.Parameter):
        """Create a hook that fires async AllReduce when grad is ready."""
        group = self.group
        ws = self.world_size
        comm_stream = self._comm_stream
        pending = self._pending

        def hook(p):
            if p.grad is None:
                return
            # Record compute stream event so comm stream waits for grad.
            event = torch.cuda.Event()
            torch.cuda.current_stream().record_event(event)
            comm_stream.wait_event(event)

            with torch.cuda.stream(comm_stream):
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=group)
                p.grad.div_(ws)

            # Record comm completion event.
            done = torch.cuda.Event()
            comm_stream.record_event(done)
            pending.append(done)

        return hook

    def finish(self) -> None:
        """Wait for all async AllReduce ops to complete. Call before optimizer.step()."""
        for event in self._pending:
            torch.cuda.current_stream().wait_event(event)
        self._pending.clear()

    def remove_hooks(self) -> None:
        """Clean up registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
