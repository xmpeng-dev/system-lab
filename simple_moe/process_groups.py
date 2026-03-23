"""
Process group management for the three parallel dimensions: PP × EP × FSDP.

[Megatron] Inspired by Parallel Folding / Dual DeviceMesh.
Simplified: no TP or CP dimensions.

GPU layout (example 128 GPUs = 16 nodes × 8 GPUs/node, PP=4, EP=8, FSDP=4):

  Stage 0 (32 GPUs, nodes 0-3):
    Node 0:  ep_rank 0-7  fsdp_rank=0      ← EP within node (NVLink)
    Node 1:  ep_rank 0-7  fsdp_rank=1      ← FSDP across nodes (IB)
    Node 2:  ep_rank 0-7  fsdp_rank=2
    Node 3:  ep_rank 0-7  fsdp_rank=3
"""

from __future__ import annotations

from typing import Optional

import torch.distributed as dist

from simple_moe.config import ClusterConfig


class ProcessGroups:
    """
    Create and hold the three families of NCCL process groups.

    Attributes
    ----------
    pp_rank, ep_rank, fsdp_rank : int
        This rank's coordinate in each dimension.
    fsdp_group : ProcessGroup
        Dense parameter All-Gather / Reduce-Scatter.
    ep_group : ProcessGroup
        MoE All-to-All dispatch / combine.
    edp_group : ProcessGroup
        Expert Data Parallel gradient AllReduce (== fsdp_group).
    pp_group : ProcessGroup
        Pipeline parallel send / recv.
    """

    def __init__(self, cluster: ClusterConfig) -> None:
        self.cluster = cluster
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        pp, ep, fsdp = cluster.pp_size, cluster.ep_size, cluster.fsdp_size
        stage_size = ep * fsdp

        self.pp_rank = rank // stage_size
        local = rank % stage_size
        # Within a stage the layout is [node=fsdp_rank][local_gpu=ep_rank]
        # so that EP stays inside a node (NVLink).
        self.fsdp_rank = local // ep
        self.ep_rank = local % ep

        # --- FSDP groups (same PP stage, same EP rank, varying FSDP rank) ---
        self.fsdp_group: dist.ProcessGroup = self._make_groups_fsdp(rank)

        # --- EP groups (same PP stage, same FSDP rank, varying EP rank) ---
        self.ep_group: dist.ProcessGroup = self._make_groups_ep(rank)

        # --- EDP groups (expert data parallel == FSDP groups) ---
        self.edp_group = self.fsdp_group

        # --- PP groups (same EP rank & FSDP rank, varying PP stage) ---
        self.pp_group: dist.ProcessGroup = self._make_groups_pp(rank)

    # ------------------------------------------------------------------
    # Group construction helpers
    # ------------------------------------------------------------------

    def _global_rank(self, pp: int, ep: int, fsdp: int) -> int:
        c = self.cluster
        return pp * (c.ep_size * c.fsdp_size) + fsdp * c.ep_size + ep

    def _make_groups_fsdp(self, rank: int) -> dist.ProcessGroup:
        c = self.cluster
        my_group: Optional[dist.ProcessGroup] = None
        for pp in range(c.pp_size):
            for e in range(c.ep_size):
                ranks = [self._global_rank(pp, e, f) for f in range(c.fsdp_size)]
                g = dist.new_group(ranks)
                if rank in ranks:
                    my_group = g
        assert my_group is not None
        return my_group

    def _make_groups_ep(self, rank: int) -> dist.ProcessGroup:
        c = self.cluster
        my_group: Optional[dist.ProcessGroup] = None
        for pp in range(c.pp_size):
            for f in range(c.fsdp_size):
                ranks = [self._global_rank(pp, e, f) for e in range(c.ep_size)]
                g = dist.new_group(ranks)
                if rank in ranks:
                    my_group = g
        assert my_group is not None
        return my_group

    def _make_groups_pp(self, rank: int) -> dist.ProcessGroup:
        c = self.cluster
        my_group: Optional[dist.ProcessGroup] = None
        for e in range(c.ep_size):
            for f in range(c.fsdp_size):
                ranks = [self._global_rank(pp, e, f) for pp in range(c.pp_size)]
                g = dist.new_group(ranks)
                if rank in ranks:
                    my_group = g
        assert my_group is not None
        return my_group
