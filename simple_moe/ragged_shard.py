"""
RaggedShard: flexible parameter sharding that respects block boundaries.

[veScale] Core data structures and collective operations for block-aware FSDP.

Key idea: the minimum indivisible sharding unit is a *semantic block*
(defined by quantisation or optimiser requirements), not an element or row.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Block descriptor
# ---------------------------------------------------------------------------

@dataclass
class BlockDescriptor:
    """A rectangular block inside a 2-D parameter matrix."""

    row_offset: int
    col_offset: int
    row_size: int
    col_size: int

    @property
    def offset(self) -> Tuple[int, int]:
        return (self.row_offset, self.col_offset)

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.row_size, self.col_size)

    @property
    def numel(self) -> int:
        return self.row_size * self.col_size


# ---------------------------------------------------------------------------
# Shard metadata variants
# ---------------------------------------------------------------------------

@dataclass
class RaggedShardMeta:
    """Metadata for a block-wise shard (ragged)."""

    blocks: List[BlockDescriptor]
    full_shape: Tuple[int, ...]
    total_numel: int

    @property
    def shard_numel(self) -> int:
        return sum(b.numel for b in self.blocks)


@dataclass
class FlatShardMeta:
    """Metadata for a traditional flat (row-wise) shard."""

    offset: int
    length: int
    total_numel: int
    full_shape: Tuple[int, ...]


# ---------------------------------------------------------------------------
# Shard spec produced by the planner
# ---------------------------------------------------------------------------

@dataclass
class RaggedShardSpec:
    """Per-parameter sharding specification."""

    shard_type: str  # 'block_wise' | 'row_wise' | 'flat'
    num_shards: int
    full_shape: Optional[Tuple[int, ...]] = None

    # Only for block_wise: mapping rank -> list[BlockDescriptor]
    block_assignments: Optional[List[List[BlockDescriptor]]] = None


# ---------------------------------------------------------------------------
# Block-level parameter extraction / placement
# ---------------------------------------------------------------------------

def extract_blocks(
    tensor: torch.Tensor,
    blocks: Sequence[BlockDescriptor],
) -> torch.Tensor:
    """Extract blocks from a 2-D tensor and flatten them into a 1-D shard."""
    parts: list[torch.Tensor] = []
    for b in blocks:
        part = tensor[
            b.row_offset : b.row_offset + b.row_size,
            b.col_offset : b.col_offset + b.col_size,
        ]
        parts.append(part.contiguous().flatten())
    if not parts:
        return tensor.new_empty(0)
    return torch.cat(parts)


def place_blocks(
    shard: torch.Tensor,
    blocks: Sequence[BlockDescriptor],
    full_shape: Tuple[int, int],
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Scatter a 1-D shard back into a 2-D tensor at block positions."""
    if output is None:
        output = shard.new_zeros(full_shape)
    offset = 0
    for b in blocks:
        numel = b.numel
        block_data = shard[offset : offset + numel].view(b.row_size, b.col_size)
        output[b.row_offset : b.row_offset + b.row_size,
               b.col_offset : b.col_offset + b.col_size] = block_data
        offset += numel
    return output


# ---------------------------------------------------------------------------
# Ragged collective operations
# ---------------------------------------------------------------------------

def _gather_shard_sizes(local_size: int, group: dist.ProcessGroup) -> List[int]:
    """All-gather scalar shard sizes within a process group."""
    world = dist.get_world_size(group)
    local_t = torch.tensor([local_size], dtype=torch.long, device="cuda")
    all_sizes = [torch.empty(1, dtype=torch.long, device="cuda") for _ in range(world)]
    dist.all_gather(all_sizes, local_t, group=group)
    return [t.item() for t in all_sizes]


def ragged_all_gather(
    local_shard: torch.Tensor,
    all_metas: List[RaggedShardMeta | FlatShardMeta],
    full_shape: Tuple[int, ...],
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """
    [veScale] Ragged All-Gather.

    Each rank contributes a differently-sized shard.  We use point-to-point
    sends/recvs under an NCCL group to implement the variable-length gather
    and then scatter blocks back to their original positions.
    """
    world = dist.get_world_size(group)
    rank = dist.get_rank(group)
    output = local_shard.new_zeros(full_shape)

    shard_sizes = _gather_shard_sizes(local_shard.numel(), group)

    recv_bufs: list[torch.Tensor] = []
    for r in range(world):
        recv_bufs.append(
            local_shard.new_empty(shard_sizes[r])
        )

    # Exchange shards via point-to-point inside a group call.
    dist.barrier(group=group)
    ops: list[dist.Work] = []
    for r in range(world):
        if r == rank:
            recv_bufs[r].copy_(local_shard)
        else:
            ops.append(dist.isend(local_shard, dst=dist.get_global_rank(group, r), group=group))
            ops.append(dist.irecv(recv_bufs[r], src=dist.get_global_rank(group, r), group=group))
    for op in ops:
        op.wait()

    # Scatter received shards into the full tensor.
    for r, meta in enumerate(all_metas):
        if isinstance(meta, RaggedShardMeta) and len(full_shape) == 2:
            place_blocks(recv_bufs[r], meta.blocks, full_shape, output=output)
        else:
            off = meta.offset if isinstance(meta, FlatShardMeta) else 0
            length = recv_bufs[r].numel()
            output.flatten()[off : off + length] = recv_bufs[r]

    return output


def ragged_reduce_scatter(
    full_grad: torch.Tensor,
    local_meta: RaggedShardMeta | FlatShardMeta,
    all_metas: List[RaggedShardMeta | FlatShardMeta],
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """
    [veScale] Ragged Reduce-Scatter.

    Reduce the full gradient, then each rank keeps only its own shard.
    Implemented as all-reduce followed by local extraction (simple but correct).
    A production implementation would use hierarchical RS for efficiency.
    """
    dist.all_reduce(full_grad, op=dist.ReduceOp.SUM, group=group)

    if isinstance(local_meta, RaggedShardMeta) and full_grad.dim() == 2:
        return extract_blocks(full_grad, local_meta.blocks)
    else:
        off = local_meta.offset if isinstance(local_meta, FlatShardMeta) else 0
        length = local_meta.length if isinstance(local_meta, FlatShardMeta) else local_meta.shard_numel
        return full_grad.flatten()[off : off + length].clone()
