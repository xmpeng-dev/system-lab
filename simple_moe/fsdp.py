"""
RaggedFSDP: block-aware Fully Sharded Data Parallel wrapper.

[veScale] Core contribution – shards dense parameters along semantic block
boundaries so that block-wise quantisation and second-order optimisers
(Shampoo / Muon) can operate correctly without extra communication.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from simple_moe.buffer_pool import BufferPool
from simple_moe.ragged_shard import (
    BlockDescriptor,
    FlatShardMeta,
    RaggedShardMeta,
    RaggedShardSpec,
    extract_blocks,
    place_blocks,
    ragged_all_gather,
    ragged_reduce_scatter,
)


class RaggedFSDP(nn.Module):
    """
    Wrap a dense sub-module (e.g. Attention, LayerNorm) with RaggedShard FSDP.

    On construction the original parameters are replaced with their local
    shards.  ``all_gather_params`` / ``reduce_scatter_grads`` must be called
    around the actual computation.
    """

    def __init__(
        self,
        module: nn.Module,
        fsdp_group: dist.ProcessGroup,
        shard_plan: Dict[str, RaggedShardSpec],
    ) -> None:
        super().__init__()
        self.module = module
        self.group = fsdp_group
        self.rank = dist.get_rank(fsdp_group)
        self.world_size = dist.get_world_size(fsdp_group)

        # Shard every parameter that appears in the plan.
        self._shard_metas: Dict[str, RaggedShardMeta | FlatShardMeta] = {}
        self._all_metas: Dict[str, List[RaggedShardMeta | FlatShardMeta]] = {}

        for name, param in list(module.named_parameters()):
            spec = shard_plan.get(name)
            if spec is None:
                continue
            meta, all_m = self._shard_param(name, param, spec)
            self._shard_metas[name] = meta
            self._all_metas[name] = all_m

        # [veScale] Buffer Pool for All-Gather output reuse.
        max_numel = max(
            (m.total_numel for m in self._shard_metas.values()), default=1
        )
        self.ag_pool = BufferPool(max_numel, num_buffers=2,
                                  dtype=self._param_dtype())

    # ------------------------------------------------------------------
    # Sharding
    # ------------------------------------------------------------------

    def _shard_param(
        self,
        name: str,
        param: nn.Parameter,
        spec: RaggedShardSpec,
    ) -> Tuple:
        if spec.shard_type == "block_wise" and spec.block_assignments is not None:
            my_blocks = spec.block_assignments[self.rank]
            shard_data = extract_blocks(param.data, my_blocks)
            meta = RaggedShardMeta(
                blocks=my_blocks,
                full_shape=tuple(param.shape),
                total_numel=param.numel(),
            )
            all_metas = [
                RaggedShardMeta(
                    blocks=spec.block_assignments[r],
                    full_shape=tuple(param.shape),
                    total_numel=param.numel(),
                )
                for r in range(self.world_size)
            ]
        else:
            chunk = (param.numel() + self.world_size - 1) // self.world_size
            start = self.rank * chunk
            end = min(start + chunk, param.numel())
            shard_data = param.data.flatten()[start:end].contiguous()
            meta = FlatShardMeta(
                offset=start, length=end - start,
                total_numel=param.numel(),
                full_shape=tuple(param.shape),
            )
            all_metas = [
                FlatShardMeta(
                    offset=r * chunk,
                    length=min(chunk, param.numel() - r * chunk),
                    total_numel=param.numel(),
                    full_shape=tuple(param.shape),
                )
                for r in range(self.world_size)
            ]

        # Replace the original parameter with the shard.
        new_param = nn.Parameter(shard_data, requires_grad=param.requires_grad)
        _set_param(self.module, name, new_param)
        return meta, all_metas

    # ------------------------------------------------------------------
    # Collective helpers
    # ------------------------------------------------------------------

    def all_gather_params(self) -> Dict[str, torch.Tensor]:
        """
        [veScale] Lazy All-Gather: reconstruct full parameters from shards.
        """
        full: dict[str, torch.Tensor] = {}
        for name, meta in self._shard_metas.items():
            param = _get_param(self.module, name)
            gathered = ragged_all_gather(
                local_shard=param.data,
                all_metas=self._all_metas[name],
                full_shape=meta.full_shape,
                group=self.group,
            )
            full[name] = gathered
        return full

    def reduce_scatter_grads(
        self, full_grads: Dict[str, torch.Tensor]
    ) -> None:
        """
        [veScale] Hierarchical Reduce-Scatter: aggregate gradients and keep
        only the local shard.
        """
        for name, grad in full_grads.items():
            meta = self._shard_metas[name]
            shard_grad = ragged_reduce_scatter(
                full_grad=grad,
                local_meta=meta,
                all_metas=self._all_metas[name],
                group=self.group,
            )
            param = _get_param(self.module, name)
            if param.grad is None:
                param.grad = shard_grad
            else:
                param.grad.add_(shard_grad)

    def release_full_params(self, full: Dict[str, torch.Tensor]) -> None:
        """[veScale] Return AG buffers early."""
        for v in full.values():
            self.ag_pool.release(v)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, *args, **kwargs):
        """All-Gather → compute → (grads handled separately)."""
        full_params = self.all_gather_params()
        # Temporarily install full parameters for the forward pass.
        originals: dict[str, torch.Tensor] = {}
        for name, fp in full_params.items():
            originals[name] = _get_param(self.module, name).data
            _get_param(self.module, name).data = fp

        output = self.module(*args, **kwargs)

        # Restore shards.
        for name, orig in originals.items():
            _get_param(self.module, name).data = orig

        self.release_full_params(full_params)
        return output

    # ------------------------------------------------------------------
    def _param_dtype(self) -> torch.dtype:
        for p in self.module.parameters():
            return p.dtype
        return torch.bfloat16


# ---------------------------------------------------------------------------
# Helpers for nested parameter access
# ---------------------------------------------------------------------------

def _get_param(module: nn.Module, name: str) -> nn.Parameter:
    parts = name.split(".")
    for p in parts[:-1]:
        module = getattr(module, p)
    return getattr(module, parts[-1])


def _set_param(module: nn.Module, name: str, value: nn.Parameter) -> None:
    parts = name.split(".")
    for p in parts[:-1]:
        module = getattr(module, p)
    setattr(module, parts[-1], value)
