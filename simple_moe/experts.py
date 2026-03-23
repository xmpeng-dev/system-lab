"""
Expert MLP implementations.

[Megatron] GroupedMLP – all local experts in a single batched GEMM call
using the ``grouped_gemm`` library (gmm kernel).

Gate and Up projections are fused into a single weight matrix (w_gate_up)
so that the first GEMM produces both outputs in one kernel, matching
Megatron's ColumnParallelLinear fusion. TP=1 so no split across GPUs.
"""

from __future__ import annotations

import math

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
    Grouped GEMM expert computation with fused gate+up projection.

    Architecture per expert (SwiGLU):
        gate_up = x @ W_gate_up^T          ← single fused GEMM
        gate, up = split(gate_up, 2)
        out = (SiLU(gate) * up) @ W_down^T ← second GEMM

    Weight layout (TP=1, no split):
        w_gate_up : [E, 2*F, H]   — gate and up concatenated along dim 0 of output
        w_down    : [E, H, F]

    This matches Megatron's ColumnParallelLinear fusion: 2 GEMMs → 1 GEMM
    for the first projection, halving kernel launch overhead and improving
    GPU utilisation via larger matrix dimensions.
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

        # Fused gate+up: [E, 2*F, H]
        self.w_gate_up = nn.Parameter(torch.empty(num_experts, 2 * ffn_dim, hidden_dim))
        # Down projection: [E, H, F]
        self.w_down = nn.Parameter(torch.empty(num_experts, hidden_dim, ffn_dim))
        self._init_weights()

    def _init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.hidden_dim)
        nn.init.trunc_normal_(self.w_gate_up, std=std)
        nn.init.trunc_normal_(self.w_down, std=std)

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
            return _fused_swiglu_gmm(
                x, self.w_gate_up, self.w_down, self.ffn_dim, tokens_per_expert
            )
        return _fused_swiglu_sequential(
            x, self.w_gate_up, self.w_down, self.ffn_dim, tokens_per_expert
        )


class MLP(nn.Module):
    """Standard SwiGLU MLP (used for shared experts and dense FFN layers)."""

    def __init__(self, hidden_dim: int, ffn_dim: int) -> None:
        super().__init__()
        self.gate_up_proj = nn.Linear(hidden_dim, 2 * ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


# ---------------------------------------------------------------------------
# Fused SwiGLU: gmm kernel (preferred)
# ---------------------------------------------------------------------------

def _fused_swiglu_gmm(
    x: torch.Tensor,
    w_gate_up: torch.Tensor,
    w_down: torch.Tensor,
    ffn_dim: int,
    tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    """
    Fused gate+up SwiGLU via grouped_gemm.

    Step 1: gate_up = gmm(x, w_gate_up, trans_b=True)   → [T, 2*F]  (1 GEMM)
    Step 2: gate, up = split(gate_up, F)
    Step 3: hidden = SiLU(gate) * up                     → [T, F]
    Step 4: out = gmm(hidden, w_down, trans_b=True)      → [T, H]   (1 GEMM)

    Total: 2 GEMMs instead of 3.
    """
    assert _gmm is not None
    bs = tokens_per_expert.to(dtype=torch.int64, device="cpu")

    gate_up = _gmm(x, w_gate_up, bs, trans_b=True)        # [T, 2*F]
    gate, up = gate_up.split(ffn_dim, dim=-1)              # [T, F] each
    hidden = F.silu(gate) * up                             # [T, F]
    return _gmm(hidden, w_down, bs, trans_b=True)          # [T, H]


# ---------------------------------------------------------------------------
# Fused SwiGLU: sequential fallback
# ---------------------------------------------------------------------------

def _fused_swiglu_sequential(
    x: torch.Tensor,
    w_gate_up: torch.Tensor,
    w_down: torch.Tensor,
    ffn_dim: int,
    tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    """Fallback: loops over experts one-by-one with fused gate+up."""
    outputs: list[torch.Tensor] = []
    offset = 0
    for i, count in enumerate(tokens_per_expert.tolist()):
        count = int(count)
        if count == 0:
            continue
        xi = x[offset : offset + count]                   # [count, H]
        gate_up = xi @ w_gate_up[i].t()                   # [count, 2*F]
        gate, up = gate_up.split(ffn_dim, dim=-1)          # [count, F]
        hidden = F.silu(gate) * up
        out = hidden @ w_down[i].t()                       # [count, H]
        outputs.append(out)
        offset += count
    if not outputs:
        return x.new_empty(0, x.shape[-1])
    return torch.cat(outputs, dim=0)
