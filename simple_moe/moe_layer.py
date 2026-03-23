"""
MoE layer: four-stage forward pipeline.

[Megatron] Route → Dispatch → Expert Compute → Combine.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from simple_moe.config import MoEModelConfig
from simple_moe.dispatcher import AllToAllDispatcher
from simple_moe.experts import GroupedMLP, MLP
from simple_moe.router import TopKRouter


class MoELayer(nn.Module):
    """
    Single MoE feed-forward layer.

    Stages:
        1. **Route** – TopKRouter selects K experts per token.
        2. **Dispatch** – All-to-All sends tokens to expert-owning GPUs.
        3. **Compute** – GroupedMLP processes tokens locally.
        4. **Combine** – All-to-All returns results; weighted sum + shared expert.
    """

    def __init__(
        self,
        config: MoEModelConfig,
        ep_group: dist.ProcessGroup,
        edp_group: dist.ProcessGroup,
    ) -> None:
        super().__init__()
        self.config = config
        self.ep_group = ep_group
        self.edp_group = edp_group
        ep_size = dist.get_world_size(ep_group)
        self.num_local_experts = config.num_experts // ep_size

        # Stage 1: Router
        self.router = TopKRouter(
            hidden_dim=config.hidden_dim,
            num_experts=config.num_experts,
            top_k=config.top_k,
            score_func=config.score_func,
            load_balance=config.load_balance,
            aux_loss_coeff=config.aux_loss_coeff,
        )

        # Stage 2 & 4: Dispatcher
        self.dispatcher = AllToAllDispatcher(
            ep_group=ep_group,
            num_local_experts=self.num_local_experts,
        )

        # Stage 3: Local experts (Grouped GEMM)
        self.experts = GroupedMLP(
            num_experts=self.num_local_experts,
            hidden_dim=config.hidden_dim,
            ffn_dim=config.expert_ffn_dim,
        )

        # Optional shared expert [Megatron / DeepSeek-V3 style]
        self.shared_expert: Optional[MLP] = None
        if config.num_shared_experts > 0:
            self.shared_expert = MLP(
                hidden_dim=config.hidden_dim,
                ffn_dim=config.shared_expert_ffn_dim,
            )

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        hidden_states : [T, H]

        Returns
        -------
        output : [T, H]
        aux_loss : scalar
        """
        # ---- Stage 1: Route ----
        probs, topk_indices, routing_map, aux_loss = self.router(hidden_states)

        # ---- Stage 2: Dispatch ----
        dispatched, meta = self.dispatcher.dispatch(
            hidden_states, probs, topk_indices, routing_map
        )

        # ---- Stage 3: Expert Compute ----
        expert_out = self.experts(dispatched, meta.tokens_per_expert)

        # Shared expert (parallel with expert compute in overlap mode).
        shared_out: Optional[torch.Tensor] = None
        if self.shared_expert is not None:
            shared_out = self.shared_expert(hidden_states)

        # ---- Stage 4: Combine ----
        combined = self.dispatcher.combine(expert_out, meta)

        if shared_out is not None:
            combined = combined + shared_out

        return combined, aux_loss
