"""
Expert MLP implementations.

[Megatron] GroupedMLP – all local experts in a single batched GEMM call
using the ``grouped_gemm`` library (gmm kernel).

Falls back to sequential per-expert loops if the library is unavailable.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Detect grouped_gemm availability
# ---------------------------------------------------------------------------

_HAS_GROUPED_GEMM = False
_gmm = None

try:
    from grouped_gemm.ops import gmm as _gmm_impl
    _gmm = _gmm_impl
    _HAS_GROUPED_GEMM = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# GroupedMLP
# ---------------------------------------------------------------------------

class GroupedMLP(nn.Module):
    """
    [Megatron] Grouped GEMM expert computation.

    All local experts' tokens are processed via ``grouped_gemm.ops.gmm``
    which issues a single fused kernel call instead of looping over experts.

    Architecture per expert (SwiGLU):
        out = W_down · (SiLU(W_gate · x) ⊙ (W_up · x))

    Weight layout: [num_experts, out_features, in_features]
    gmm convention: gmm(a=[T,K], b=[E,K,N], batch_sizes) → [T,N]
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

        # Stored as [E, out, in] so gmm(x, w) computes x @ w^T per expert.
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
        x : [total_tokens, hidden_dim]  – already sorted by expert.
        tokens_per_expert : [num_local_experts] int64
        """
        if _HAS_GROUPED_GEMM and x.numel() > 0 and x.dtype == torch.bfloat16:
            return _grouped_swiglu_gmm(
                x, self.w_gate, self.w_up, self.w_down, tokens_per_expert
            )
        return _grouped_swiglu_sequential(
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
# Grouped SwiGLU: gmm kernel (preferred)
# ---------------------------------------------------------------------------

def _grouped_swiglu_gmm(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    """
    Grouped SwiGLU using ``grouped_gemm.ops.gmm``.

    gmm(a, b, batch_sizes, trans_b) computes per-expert a_i @ b_i (or a_i @ b_i^T).
    Weights are stored [E, out, in], so we use trans_b=True → a_i @ w_i^T.
    batch_sizes must be a CPU int64 tensor.
    """
    assert _gmm is not None
    bs = tokens_per_expert.to(dtype=torch.int64, device="cpu")

    # x: [T, H]  w_gate: [E, F, H] → gmm(x, w_gate, bs, trans_b=True) → [T, F]
    gate_out = _gmm(x, w_gate, bs, trans_b=True)
    up_out = _gmm(x, w_up, bs, trans_b=True)

    hidden = F.silu(gate_out) * up_out  # [T, F]

    # hidden: [T, F]  w_down: [E, H, F] → gmm(hidden, w_down, bs, trans_b=True) → [T, H]
    return _gmm(hidden, w_down, bs, trans_b=True)


# ---------------------------------------------------------------------------
# Grouped SwiGLU: sequential fallback
# ---------------------------------------------------------------------------

def _grouped_swiglu_sequential(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    """Fallback: loops over experts one-by-one."""
    outputs: list[torch.Tensor] = []
    offset = 0
    for i, count in enumerate(tokens_per_expert.tolist()):
        count = int(count)
        if count == 0:
            continue
        xi = x[offset : offset + count]
        gate_out = xi @ w_gate[i].t()
        up_out = xi @ w_up[i].t()
        hidden = F.silu(gate_out) * up_out
        out = hidden @ w_down[i].t()
        outputs.append(out)
        offset += count
    if not outputs:
        return x.new_empty(0, x.shape[-1])
    return torch.cat(outputs, dim=0)
