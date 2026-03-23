"""
Expert MLP implementations.

[Megatron] GroupedMLP – all local experts in a single batched GEMM call.
Also provides a plain MLP for shared experts and dense FFN layers.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupedMLP(nn.Module):
    """
    [Megatron] Grouped GEMM expert computation.

    Instead of looping over experts, all local experts' tokens are processed
    in one batched matrix multiply (or sequential fallback when no grouped
    GEMM kernel is available).

    Architecture per expert: SwiGLU
        out = W_down · (SiLU(W_gate · x) ⊙ (W_up · x))
    """

    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        ffn_dim: int,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim

        # [num_experts, ffn_dim, hidden_dim]
        self.w_gate = nn.Parameter(torch.empty(num_experts, ffn_dim, hidden_dim))
        self.w_up = nn.Parameter(torch.empty(num_experts, ffn_dim, hidden_dim))
        self.w_down = nn.Parameter(torch.empty(num_experts, hidden_dim, ffn_dim))
        self._init_weights()

    def _init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.hidden_dim)
        for p in [self.w_gate, self.w_up, self.w_down]:
            nn.init.trunc_normal_(p, std=std)

    def forward(
        self,
        x: torch.Tensor,
        tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [total_tokens, hidden_dim]
            Already sorted by expert.
        tokens_per_expert : [num_local_experts]
        """
        return _grouped_swiglu(
            x, self.w_gate, self.w_up, self.w_down, tokens_per_expert
        )


class MLP(nn.Module):
    """Standard SwiGLU MLP (used for shared experts and dense FFN layers)."""

    def __init__(self, hidden_dim: int, ffn_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Grouped SwiGLU implementation (sequential fallback)
# ---------------------------------------------------------------------------

def _grouped_swiglu(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    """
    Fallback grouped SwiGLU – loops over experts.

    A production system would use CUTLASS Grouped GEMM or cuBLASLt
    grouped GEMM for a single-kernel launch.
    """
    outputs: list[torch.Tensor] = []
    offset = 0
    for i, count in enumerate(tokens_per_expert.tolist()):
        count = int(count)
        if count == 0:
            continue
        xi = x[offset : offset + count]  # [count, H]
        gate_out = xi @ w_gate[i].t()  # [count, F]
        up_out = xi @ w_up[i].t()  # [count, F]
        hidden = F.silu(gate_out) * up_out
        out = hidden @ w_down[i].t()  # [count, H]
        outputs.append(out)
        offset += count
    if not outputs:
        return x.new_empty(0, x.shape[-1])
    return torch.cat(outputs, dim=0)
