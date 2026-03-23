"""
Structure-Aware Planner: unified planning for FSDP + EP + PP.

[veScale] Core algorithm – analyse model structure, identify co-location
constraints (quantisation blocks, Kronecker factors), and produce a
RaggedShard allocation that balances memory across FSDP ranks.

[Megatron] PP stage partitioning – cost-aware greedy split that accounts
for MoE layers being much heavier than dense layers.

[Megatron] Expert placement – uniform assignment across EP ranks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from simple_moe.config import ClusterConfig, MoEModelConfig
from simple_moe.ragged_shard import BlockDescriptor, RaggedShardSpec


# ---------------------------------------------------------------------------
# Union-Find for co-location constraint merging
# ---------------------------------------------------------------------------

class UnionFind:
    """Weighted union-find with path compression."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


# ---------------------------------------------------------------------------
# Distributed plan output
# ---------------------------------------------------------------------------

@dataclass
class DistributedPlan:
    pp_partition: List[List[int]]
    fsdp_plan: Dict[str, RaggedShardSpec]
    ep_placement: Dict[int, int]  # expert_id -> ep_rank


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class StructureAwarePlanner:
    """
    Produce a complete FSDP + EP + PP plan for an MoE model.
    """

    def __init__(
        self,
        model_config: MoEModelConfig,
        cluster: ClusterConfig,
        optimizer_type: str = "adam",
    ) -> None:
        self.model_cfg = model_config
        self.cluster = cluster
        self.optimizer_type = optimizer_type

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def plan(self) -> DistributedPlan:
        pp_partition = self._plan_pp_stages()
        fsdp_plan = self._plan_fsdp_sharding()
        ep_placement = self._plan_expert_placement()
        return DistributedPlan(pp_partition, fsdp_plan, ep_placement)

    # ------------------------------------------------------------------
    # PP stage partitioning  [Megatron]
    # ------------------------------------------------------------------

    def _plan_pp_stages(self) -> List[List[int]]:
        """
        Cost-aware greedy split.  MoE layers carry much more weight than
        dense layers; use parameter count as proxy for cost.
        """
        cfg = self.model_cfg
        num_stages = self.cluster.pp_size * max(self.cluster.vpp_size, 1)

        costs: list[float] = []
        for i in range(cfg.num_layers):
            is_moe = i >= cfg.num_dense_layers
            if is_moe:
                local_experts = cfg.num_experts // self.cluster.ep_size
                expert_cost = local_experts * cfg.expert_ffn_dim * cfg.hidden_dim * 3
                attn_cost = 4 * cfg.hidden_dim * cfg.hidden_dim
                costs.append(float(attn_cost + expert_cost))
            else:
                costs.append(float(4 * cfg.hidden_dim * cfg.hidden_dim + 3 * cfg.hidden_dim * cfg.dense_ffn_dim))

        target = sum(costs) / num_stages
        stages: list[list[int]] = []
        cur: list[int] = []
        cur_cost = 0.0
        for i, c in enumerate(costs):
            cur.append(i)
            cur_cost += c
            if cur_cost >= target and len(stages) < num_stages - 1:
                stages.append(cur)
                cur, cur_cost = [], 0.0
        stages.append(cur)
        return stages

    # ------------------------------------------------------------------
    # FSDP sharding  [veScale]
    # ------------------------------------------------------------------

    def _plan_fsdp_sharding(self) -> Dict[str, RaggedShardSpec]:
        """
        [veScale] Structure-Aware Planning Algorithm

        1. Determine block boundaries from optimiser / quantisation needs.
        2. Build co-location constraints via Union-Find.
        3. Greedy balanced assignment of super-blocks to FSDP ranks.
        """
        cfg = self.model_cfg
        plan: dict[str, RaggedShardSpec] = {}

        # Iterate over the *types* of dense parameters that will exist.
        dense_param_shapes = self._enumerate_dense_param_shapes()

        for name, shape in dense_param_shapes.items():
            if len(shape) < 2:
                plan[name] = RaggedShardSpec(
                    shard_type="flat",
                    num_shards=self.cluster.fsdp_size,
                    full_shape=shape,
                )
                continue

            block_size = self._infer_block_size(shape)
            if block_size is None:
                plan[name] = RaggedShardSpec(
                    shard_type="row_wise",
                    num_shards=self.cluster.fsdp_size,
                    full_shape=shape,
                )
                continue

            blocks = self._partition_into_blocks(shape, block_size)
            groups = self._build_co_location_groups(blocks, shape)
            assignments = self._greedy_balanced_assign(groups)

            plan[name] = RaggedShardSpec(
                shard_type="block_wise",
                num_shards=self.cluster.fsdp_size,
                full_shape=shape,
                block_assignments=assignments,
            )

        return plan

    # ------------------------------------------------------------------
    # EP placement  [Megatron]
    # ------------------------------------------------------------------

    def _plan_expert_placement(self) -> Dict[int, int]:
        E = self.model_cfg.num_experts
        ep = self.cluster.ep_size
        placement: dict[int, int] = {}
        per_gpu = E // ep
        for rank in range(ep):
            for j in range(per_gpu):
                placement[rank * per_gpu + j] = rank
        return placement

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _enumerate_dense_param_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Return representative (name, shape) pairs for dense parameters."""
        H = self.model_cfg.hidden_dim
        heads = self.model_cfg.num_heads
        hd = self.model_cfg.head_dim
        F_dense = self.model_cfg.dense_ffn_dim
        V = self.model_cfg.vocab_size

        shapes: dict[str, tuple[int, ...]] = {}
        shapes["embed_tokens.weight"] = (V, H)
        for i in range(self.model_cfg.num_layers):
            pfx = f"layers.{i}"
            shapes[f"{pfx}.norm1.weight"] = (H,)
            shapes[f"{pfx}.norm2.weight"] = (H,)
            shapes[f"{pfx}.attn.q_proj.weight"] = (heads * hd, H)
            shapes[f"{pfx}.attn.k_proj.weight"] = (heads * hd, H)
            shapes[f"{pfx}.attn.v_proj.weight"] = (heads * hd, H)
            shapes[f"{pfx}.attn.o_proj.weight"] = (H, heads * hd)
            is_moe = i >= self.model_cfg.num_dense_layers
            if not is_moe:
                shapes[f"{pfx}.ffn.gate_proj.weight"] = (F_dense, H)
                shapes[f"{pfx}.ffn.up_proj.weight"] = (F_dense, H)
                shapes[f"{pfx}.ffn.down_proj.weight"] = (H, F_dense)
        shapes["output_head.weight"] = (V, H)
        return shapes

    def _infer_block_size(self, shape: Tuple[int, ...]) -> Optional[int]:
        if self.optimizer_type == "shampoo":
            return min(128, *shape)
        elif self.optimizer_type == "muon":
            return min(256, shape[0])
        elif self.cluster.quant_block_size is not None:
            return self.cluster.quant_block_size
        return None

    @staticmethod
    def _partition_into_blocks(
        shape: Tuple[int, int], block_size: int
    ) -> List[BlockDescriptor]:
        rows, cols = shape
        blocks: list[BlockDescriptor] = []
        for r in range(0, rows, block_size):
            rh = min(block_size, rows - r)
            for c in range(0, cols, block_size):
                cw = min(block_size, cols - c)
                blocks.append(BlockDescriptor(r, c, rh, cw))
        return blocks

    def _build_co_location_groups(
        self,
        blocks: List[BlockDescriptor],
        shape: Tuple[int, int],
    ) -> List[List[BlockDescriptor]]:
        """
        Apply Union-Find to merge blocks that share co-location constraints.

        For Shampoo / Muon every block within the same Kronecker partition
        row or column must stay together.  For quantisation each block is
        already atomic.
        """
        n = len(blocks)
        uf = UnionFind(n)

        if self.optimizer_type in ("shampoo", "muon"):
            row_map: dict[int, list[int]] = {}
            for idx, b in enumerate(blocks):
                row_map.setdefault(b.row_offset, []).append(idx)
            for indices in row_map.values():
                for j in range(1, len(indices)):
                    uf.union(indices[0], indices[j])

        groups_map: dict[int, list[int]] = {}
        for i in range(n):
            root = uf.find(i)
            groups_map.setdefault(root, []).append(i)

        return [[blocks[i] for i in idxs] for idxs in groups_map.values()]

    def _greedy_balanced_assign(
        self,
        groups: List[List[BlockDescriptor]],
    ) -> List[List[BlockDescriptor]]:
        """
        [veScale] Greedy balanced assignment.

        Sort super-blocks by descending size, then assign each to the rank
        with the smallest current load.
        """
        num_shards = self.cluster.fsdp_size
        assignments: list[list[BlockDescriptor]] = [[] for _ in range(num_shards)]
        loads = [0] * num_shards

        group_sizes = [(sum(b.numel for b in g), g) for g in groups]
        group_sizes.sort(key=lambda x: -x[0])

        for size, group in group_sizes:
            target = loads.index(min(loads))
            assignments[target].extend(group)
            loads[target] += size

        return assignments
