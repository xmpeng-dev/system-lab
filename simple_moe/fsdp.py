"""
RaggedFSDP: Fully Sharded Data Parallel with autograd support.

[veScale] Parameters are sharded across the FSDP group. Forward does
All-Gather to reconstruct full params, backward does Reduce-Scatter to
distribute gradients back to shards.

Uses custom autograd Functions so the AG/RS are properly differentiable.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn


# ---------------------------------------------------------------------------
# Autograd-safe All-Gather / Reduce-Scatter
# ---------------------------------------------------------------------------

class _AllGatherFunc(torch.autograd.Function):
    """Forward: All-Gather shards → full tensor. Backward: Reduce-Scatter."""

    @staticmethod
    def forward(ctx, shard, group, world_size, full_numel, full_shape):
        ctx.group = group
        ctx.world_size = world_size
        ctx.shard_numel = shard.numel()

        gather_list = [torch.empty_like(shard) for _ in range(world_size)]
        dist.all_gather(gather_list, shard.contiguous(), group=group)
        full = torch.cat(gather_list, dim=0)[:full_numel]
        ctx.full_numel = full_numel
        ctx.full_shape = full_shape
        return full.view(full_shape)

    @staticmethod
    def backward(ctx, grad_output):
        grad_flat = grad_output.contiguous().flatten()
        # Pad to be divisible by world_size
        padded_len = ctx.shard_numel * ctx.world_size
        if grad_flat.numel() < padded_len:
            grad_flat = torch.nn.functional.pad(
                grad_flat, (0, padded_len - grad_flat.numel())
            )
        shard_grad = torch.empty(
            ctx.shard_numel, dtype=grad_flat.dtype, device=grad_flat.device
        )
        dist.reduce_scatter_tensor(shard_grad, grad_flat, group=ctx.group)
        return shard_grad, None, None, None, None


def fsdp_all_gather(shard, group, world_size, full_numel, full_shape):
    """Autograd-safe All-Gather: shard → full parameter."""
    return _AllGatherFunc.apply(shard, group, world_size, full_numel, full_shape)


# ---------------------------------------------------------------------------
# FSDP Module Wrapper
# ---------------------------------------------------------------------------

class ShardedFSDP(nn.Module):
    """
    Wraps a module so its parameters are sharded across the FSDP group.

    On construction, each parameter is replaced with its local shard
    (1/world_size of the original). During forward, All-Gather reconstructs
    full parameters; during backward, Reduce-Scatter distributes gradients.

    This is ZeRO-3 style sharding. Compatible with any optimizer (Adam,
    Shampoo, Muon) since the optimizer only sees shard-sized parameters.
    """

    def __init__(
        self,
        module: nn.Module,
        fsdp_group: dist.ProcessGroup,
    ) -> None:
        super().__init__()
        self.module = module
        self.group = fsdp_group
        self.rank = dist.get_rank(fsdp_group)
        self.world_size = dist.get_world_size(fsdp_group)

        self._param_info: Dict[str, _ParamInfo] = {}
        self._shard_params()

    def _shard_params(self) -> None:
        """Replace each parameter with its local shard."""
        for name, param in list(self.module.named_parameters()):
            full_numel = param.numel()
            full_shape = param.shape
            chunk = (full_numel + self.world_size - 1) // self.world_size
            start = self.rank * chunk
            end = min(start + chunk, full_numel)

            shard = param.data.flatten()[start:end].contiguous().clone()
            new_param = nn.Parameter(shard, requires_grad=param.requires_grad)
            _set_param(self.module, name, new_param)

            self._param_info[name] = _ParamInfo(
                full_numel=full_numel,
                full_shape=full_shape,
                shard_numel=end - start,
            )

    def forward(self, *args, **kwargs):
        """All-Gather params → compute → (RS happens in backward)."""
        originals: dict[str, torch.Tensor] = {}

        for name, info in self._param_info.items():
            param = _get_param(self.module, name)
            originals[name] = param

            full = fsdp_all_gather(
                param, self.group, self.world_size,
                info.full_numel, info.full_shape,
            )
            # Temporarily replace shard with full param for compute.
            # We use a trick: create a "view" parameter that wraps the AG output.
            _set_attr(self.module, name, full)

        output = self.module(*args, **kwargs)

        # Restore shards so optimizer sees the right parameter objects.
        for name, orig in originals.items():
            _set_attr(self.module, name, orig)

        return output

    @property
    def shard_memory_bytes(self) -> int:
        total = 0
        for name, info in self._param_info.items():
            total += info.shard_numel * _get_param(self.module, name).element_size()
        return total

    @property
    def full_memory_bytes(self) -> int:
        total = 0
        for name, info in self._param_info.items():
            total += info.full_numel * _get_param(self.module, name).element_size()
        return total


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _ParamInfo:
    __slots__ = ("full_numel", "full_shape", "shard_numel")

    def __init__(self, full_numel: int, full_shape: Tuple[int, ...], shard_numel: int):
        self.full_numel = full_numel
        self.full_shape = full_shape
        self.shard_numel = shard_numel


def _get_param(module: nn.Module, name: str):
    parts = name.split(".")
    for p in parts[:-1]:
        module = getattr(module, p)
    return getattr(module, parts[-1])


def _set_param(module: nn.Module, name: str, value: nn.Parameter) -> None:
    parts = name.split(".")
    for p in parts[:-1]:
        module = getattr(module, p)
    setattr(module, parts[-1], value)


def _set_attr(module: nn.Module, name: str, value) -> None:
    """
    Set attribute on a nested module, bypassing nn.Module type checks.
    
    During FSDP forward we temporarily replace a Parameter (shard) with
    a plain Tensor (the All-Gathered full param).  nn.Module.__setattr__
    rejects plain tensors for registered parameters, so we go through
    the _parameters dict directly.
    """
    parts = name.split(".")
    for p in parts[:-1]:
        module = getattr(module, p)
    key = parts[-1]
    if key in module._parameters:
        module._parameters[key] = value
    else:
        setattr(module, key, value)
