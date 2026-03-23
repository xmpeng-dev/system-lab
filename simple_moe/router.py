"""
TopK Router with load-balancing support.

[Megatron] Implements gating, top-k selection, and three load-balancing
strategies: auxiliary loss, expert-choice, and aux-loss-free (learnable bias).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKRouter(nn.Module):
    """
    [Megatron] Top-K router that selects K experts per token.

    Parameters
    ----------
    hidden_dim : int
        Model hidden dimension.
    num_experts : int
        Total number of experts.
    top_k : int
        Number of experts activated per token.
    score_func : str
        ``"softmax"`` or ``"sigmoid"`` scoring.
    load_balance : str
        ``"aux_loss"`` | ``"aux_loss_free"`` | ``"none"``.
    aux_loss_coeff : float
        Coefficient for the auxiliary load-balancing loss.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        top_k: int = 2,
        score_func: str = "softmax",
        load_balance: str = "aux_loss_free",
        aux_loss_coeff: float = 0.01,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.score_func = score_func
        self.load_balance = load_balance
        self.aux_loss_coeff = aux_loss_coeff

        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)

        # [Megatron] Aux-Loss-Free: learnable bias for expert selection.
        if load_balance == "aux_loss_free":
            self.expert_bias = nn.Parameter(torch.zeros(num_experts))
        else:
            self.expert_bias = None

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        probs : [T, K]
            Routing weights for selected experts.
        topk_indices : [T, K]
            Expert indices selected per token.
        routing_map : [T, E]
            Boolean mask indicating selected experts.
        aux_loss : scalar
            Load-balancing auxiliary loss.
        """
        T = hidden_states.shape[0]
        logits = self.gate(hidden_states)  # [T, E]

        if self.score_func == "softmax":
            scores = F.softmax(logits, dim=-1)
        else:
            scores = torch.sigmoid(logits)

        # Top-K selection (bias only affects selection, not weights).
        selection_scores = scores
        if self.expert_bias is not None:
            selection_scores = scores + self.expert_bias.unsqueeze(0)
        topk_values, topk_indices = torch.topk(
            selection_scores, self.top_k, dim=-1
        )

        # Actual routing weights from unbiased scores.
        probs = torch.gather(scores, dim=-1, index=topk_indices)
        if self.score_func == "softmax":
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-9)

        # Boolean routing map [T, E].
        routing_map = torch.zeros(
            T, self.num_experts, dtype=torch.bool, device=hidden_states.device
        )
        routing_map.scatter_(1, topk_indices, True)

        aux_loss = self._compute_aux_loss(scores, routing_map)
        return probs, topk_indices, routing_map, aux_loss

    # ------------------------------------------------------------------
    def _compute_aux_loss(
        self, scores: torch.Tensor, routing_map: torch.Tensor
    ) -> torch.Tensor:
        if self.load_balance == "none":
            return scores.new_tensor(0.0)

        if self.load_balance == "aux_loss":
            # Standard auxiliary loss: encourage uniform expert utilisation.
            # f_i = fraction of tokens routed to expert i
            # p_i = mean routing probability for expert i
            # loss = E * sum(f_i * p_i)
            f = routing_map.float().mean(dim=0)
            p = scores.mean(dim=0)
            return self.aux_loss_coeff * self.num_experts * (f * p).sum()

        if self.load_balance == "aux_loss_free":
            # Bias is updated externally; no differentiable loss required.
            return scores.new_tensor(0.0)

        return scores.new_tensor(0.0)
