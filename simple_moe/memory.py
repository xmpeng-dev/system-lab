"""
Memory optimisation utilities.

[Megatron] Fine-grained selective recomputation and activation offloading.
[veScale] Buffer-pool based memory reuse (see buffer_pool.py).
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Selective recomputation wrapper
# ---------------------------------------------------------------------------

class SelectiveRecompute(torch.autograd.Function):
    """
    [Megatron] Fine-grained recomputation.

    Wraps a *recomputable_fn* so that its intermediate activations are freed
    after the forward pass and recomputed during backward.  The caller
    decides which sub-computation to wrap (e.g. SwiGLU activation, LayerNorm)
    while keeping expensive parts (SDPA, expert GEMM) fully materialised.
    """

    @staticmethod
    def forward(ctx, run_fn: Callable, preserve: torch.Tensor, *args):
        ctx.run_fn = run_fn
        ctx.save_for_backward(preserve, *args)
        with torch.no_grad():
            return run_fn(preserve, *args)

    @staticmethod
    def backward(ctx, *grad_outputs):
        inputs = ctx.saved_tensors
        preserve = inputs[0]
        rest = inputs[1:]
        with torch.enable_grad():
            preserve = preserve.detach().requires_grad_(True)
            detached = tuple(t.detach().requires_grad_(t.requires_grad) for t in rest)
            output = ctx.run_fn(preserve, *detached)
        grads = torch.autograd.grad(
            output, (preserve,) + detached, grad_outputs,
            allow_unused=True,
        )
        return (None,) + grads


def recompute_wrapper(fn: Callable, preserve: torch.Tensor, *args):
    """Convenience: apply selective recomputation to *fn*."""
    return SelectiveRecompute.apply(fn, preserve, *args)


# ---------------------------------------------------------------------------
# Activation offloading helpers
# ---------------------------------------------------------------------------

class ActivationOffloader:
    """
    [Megatron] Fine-grained activation offloading to CPU.

    Uses a dedicated D2H stream so that the copy overlaps with the next
    module's forward computation.  Reload uses a separate H2D stream
    triggered one layer ahead (Layer-Staggered Reload).
    """

    def __init__(self) -> None:
        self._d2h_stream: Optional[torch.cuda.Stream] = None
        self._h2d_stream: Optional[torch.cuda.Stream] = None
        self._cpu_stash: dict[int, torch.Tensor] = {}
        self._events: dict[int, torch.cuda.Event] = {}

    @property
    def d2h_stream(self) -> torch.cuda.Stream:
        if self._d2h_stream is None:
            self._d2h_stream = torch.cuda.Stream()
        return self._d2h_stream

    @property
    def h2d_stream(self) -> torch.cuda.Stream:
        if self._h2d_stream is None:
            self._h2d_stream = torch.cuda.Stream()
        return self._h2d_stream

    def offload(self, layer_id: int, activation: torch.Tensor) -> None:
        """Asynchronously copy *activation* to CPU pinned memory."""
        cpu_buf = torch.empty(
            activation.shape, dtype=activation.dtype,
            device="cpu", pin_memory=True,
        )
        event = torch.cuda.Event()
        torch.cuda.current_stream().record_event(event)
        self.d2h_stream.wait_event(event)
        with torch.cuda.stream(self.d2h_stream):
            cpu_buf.copy_(activation, non_blocking=True)
        self._cpu_stash[layer_id] = cpu_buf
        done = torch.cuda.Event()
        self.d2h_stream.record_event(done)
        self._events[layer_id] = done

    def reload(self, layer_id: int, device: torch.device) -> torch.Tensor:
        """Reload previously offloaded activation back to GPU."""
        cpu_buf = self._cpu_stash.pop(layer_id)
        event = self._events.pop(layer_id, None)
        if event is not None:
            self.h2d_stream.wait_event(event)
        gpu_buf = torch.empty(
            cpu_buf.shape, dtype=cpu_buf.dtype, device=device,
        )
        with torch.cuda.stream(self.h2d_stream):
            gpu_buf.copy_(cpu_buf, non_blocking=True)
        reload_done = torch.cuda.Event()
        self.h2d_stream.record_event(reload_done)
        torch.cuda.current_stream().wait_event(reload_done)
        return gpu_buf


# ---------------------------------------------------------------------------
# High-level configuration helpers
# ---------------------------------------------------------------------------

def configure_recomputation(model: nn.Module) -> None:
    """
    [Megatron] Mark lightweight modules for selective recomputation.

    Never recompute:
      - Expert GEMM (would re-trigger All-to-All)
      - Attention SDPA (O(s²) compute)

    Safe to recompute:
      - SwiGLU activation functions
      - LayerNorm
      - Attention up-projection
    """
    # This is a configuration marker; actual wrapping happens in the
    # trainer's forward loop via recompute_wrapper.
    for module in model.modules():
        if isinstance(module, nn.LayerNorm) or isinstance(module, nn.RMSNorm):
            module._simple_moe_recompute = True


def configure_offloading(
    model: nn.Module,
    offloader: ActivationOffloader,
    modules: Sequence[str] = ("attention", "expert_ffn"),
) -> None:
    """
    [Megatron] Register modules for fine-grained activation offloading.
    """
    for name, module in model.named_modules():
        for tag in modules:
            if tag in name:
                module._simple_moe_offloader = offloader
                break
