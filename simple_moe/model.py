"""
MoE Transformer language model.

A decoder-only Transformer where the last *num_moe_layers* layers use MoE
feed-forward blocks and the remaining layers use a standard dense MLP.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from simple_moe.config import MoEModelConfig
from simple_moe.experts import MLP
from simple_moe.moe_layer import MoELayer


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).to(x.dtype) * self.weight


class Attention(nn.Module):
    def __init__(self, config: MoEModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        total_head_dim = config.num_heads * config.head_dim

        self.q_proj = nn.Linear(config.hidden_dim, total_head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, total_head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, total_head_dim, bias=False)
        self.o_proj = nn.Linear(total_head_dim, config.hidden_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, S, H = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=mask is None)
        attn = attn.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(attn)


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Single transformer block; either dense FFN or MoE FFN."""

    def __init__(
        self,
        config: MoEModelConfig,
        is_moe: bool = False,
        ep_group: Optional[dist.ProcessGroup] = None,
        edp_group: Optional[dist.ProcessGroup] = None,
    ) -> None:
        super().__init__()
        self.is_moe = is_moe
        self.norm1 = RMSNorm(config.hidden_dim)
        self.attn = Attention(config)
        self.norm2 = RMSNorm(config.hidden_dim)

        if is_moe:
            assert ep_group is not None and edp_group is not None
            self.moe = MoELayer(config, ep_group, edp_group)
            self.ffn = None
        else:
            self.moe = None
            self.ffn = MLP(config.hidden_dim, config.dense_ffn_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (hidden_states, aux_loss).
        """
        residual = x
        x = self.norm1(x)
        x = residual + self.attn(x, mask)

        residual = x
        x = self.norm2(x)
        aux_loss = x.new_tensor(0.0)

        if self.is_moe:
            B, S, H = x.shape
            moe_in = x.view(B * S, H)
            moe_out, aux_loss = self.moe(moe_in)
            x = residual + moe_out.view(B, S, H)
        else:
            x = residual + self.ffn(x)

        return x, aux_loss


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class MoETransformerLM(nn.Module):
    """
    Decoder-only MoE Transformer language model.

    The first ``num_dense_layers`` layers use a dense MLP; the remaining
    ``num_moe_layers`` layers use MoE.
    """

    def __init__(
        self,
        config: MoEModelConfig,
        ep_group: Optional[dist.ProcessGroup] = None,
        edp_group: Optional[dist.ProcessGroup] = None,
    ) -> None:
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_dim)

        self.layers = nn.ModuleList()
        for i in range(config.num_layers):
            is_moe = i >= config.num_dense_layers
            self.layers.append(
                TransformerBlock(
                    config,
                    is_moe=is_moe,
                    ep_group=ep_group,
                    edp_group=edp_group,
                )
            )

        self.norm_f = RMSNorm(config.hidden_dim)
        self.output_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.config.hidden_dim)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.trunc_normal_(module.weight, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        input_ids : [B, S] int64
        labels : [B, S] int64, optional (for loss computation)
        mask : attention mask, optional

        Returns
        -------
        logits : [B, S, V]
        total_aux_loss : scalar
        """
        x = self.embed_tokens(input_ids)

        total_aux = x.new_tensor(0.0)
        for layer in self.layers:
            x, aux = layer(x, mask)
            total_aux = total_aux + aux

        x = self.norm_f(x)
        logits = self.output_head(x)

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            return loss + total_aux, total_aux

        return logits, total_aux
