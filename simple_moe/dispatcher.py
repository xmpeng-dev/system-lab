"""
All-to-All Token Dispatcher for Expert Parallel.

[Megatron] Implements the Permute → All-to-All → … → All-to-All → Unpermute
data path with Memory-Efficient Permutation (routing weights applied before
the second expert linear layer instead of at combine time).

Includes custom autograd Functions so gradients flow correctly through
All-to-All in both directions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Autograd-safe All-to-All primitives
# ---------------------------------------------------------------------------

class _AllToAllForward(torch.autograd.Function):
    """All-to-All with correct backward (reverse All-to-All)."""

    @staticmethod
    def forward(ctx, input, send_splits, recv_splits, group):
        ctx.group = group
        ctx.send_splits = send_splits
        ctx.recv_splits = recv_splits

        H = input.shape[-1]
        recv_buf = input.new_empty(sum(recv_splits), H)
        input_list = list(input.split(send_splits, dim=0))
        output_list = list(recv_buf.split(recv_splits, dim=0))
        dist.all_to_all(output_list, input_list, group=group)
        return torch.cat(output_list, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        # Backward of All-to-All is All-to-All with swapped splits.
        H = grad_output.shape[-1]
        grad_input = grad_output.new_empty(sum(ctx.send_splits), H)
        input_list = list(grad_output.split(ctx.recv_splits, dim=0))
        output_list = list(grad_input.split(ctx.send_splits, dim=0))
        dist.all_to_all(output_list, input_list, group=ctx.group)
        return torch.cat(output_list, dim=0), None, None, None


class _AllToAllBackward(torch.autograd.Function):
    """Reverse All-to-All (combine direction) with correct backward."""

    @staticmethod
    def forward(ctx, input, send_splits, recv_splits, group):
        ctx.group = group
        ctx.send_splits = send_splits
        ctx.recv_splits = recv_splits

        H = input.shape[-1]
        recv_buf = input.new_empty(sum(recv_splits), H)
        input_list = list(input.split(send_splits, dim=0))
        output_list = list(recv_buf.split(recv_splits, dim=0))
        dist.all_to_all(output_list, input_list, group=group)
        return torch.cat(output_list, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        H = grad_output.shape[-1]
        grad_input = grad_output.new_empty(sum(ctx.send_splits), H)
        input_list = list(grad_output.split(ctx.recv_splits, dim=0))
        output_list = list(grad_input.split(ctx.send_splits, dim=0))
        dist.all_to_all(output_list, input_list, group=ctx.group)
        return torch.cat(output_list, dim=0), None, None, None


def all_to_all_fwd(input, send_splits, recv_splits, group):
    """Dispatch direction: autograd-aware All-to-All."""
    return _AllToAllForward.apply(input, send_splits, recv_splits, group)


def all_to_all_bwd(input, send_splits, recv_splits, group):
    """Combine direction: autograd-aware All-to-All."""
    return _AllToAllBackward.apply(input, send_splits, recv_splits, group)


# ---------------------------------------------------------------------------
# DispatchMeta
# ---------------------------------------------------------------------------

@dataclass
class DispatchMeta:
    """Book-keeping produced by ``dispatch`` and consumed by ``combine``."""

    permute_indices: torch.Tensor  # [T*K] int64 – original token index
    tokens_per_expert: torch.Tensor  # [num_local_experts] int64
    send_counts: torch.Tensor  # [ep_size] int64
    recv_counts: torch.Tensor  # [ep_size] int64
    original_shape: Tuple[int, int]  # (T, H)
    num_local_experts: int


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

class AllToAllDispatcher:
    """
    [Megatron] Token dispatcher using NCCL All-to-All.

    Uses custom autograd Functions so that gradients flow correctly:
      Forward:  tokens → All-to-All → experts
      Backward: grad   → reverse All-to-All → back to source GPUs
    """

    def __init__(
        self,
        ep_group: dist.ProcessGroup,
        num_local_experts: int,
    ) -> None:
        self.ep_group = ep_group
        self.ep_size = dist.get_world_size(ep_group)
        self.ep_rank = dist.get_rank(ep_group)
        self.num_local_experts = num_local_experts

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        probs: torch.Tensor,
        topk_indices: torch.Tensor,
        routing_map: torch.Tensor,
    ) -> Tuple[torch.Tensor, DispatchMeta]:
        T, H = hidden_states.shape
        K = topk_indices.shape[1]
        E = routing_map.shape[1]

        # ---- Permute: sort tokens by expert id ----
        flat_expert_ids = topk_indices.flatten()  # [T*K]
        flat_probs = probs.flatten()  # [T*K]
        token_indices = (
            torch.arange(T, device=hidden_states.device)
            .unsqueeze(1)
            .expand(T, K)
            .flatten()
        )  # [T*K]

        sort_order = torch.argsort(flat_expert_ids, stable=True)
        sorted_expert_ids = flat_expert_ids[sort_order]
        sorted_token_idx = token_indices[sort_order]
        sorted_probs = flat_probs[sort_order]

        permuted_tokens = hidden_states[sorted_token_idx]  # [T*K, H]

        # [Megatron] Memory-Efficient Permutation: apply routing weights now.
        permuted_tokens = permuted_tokens * sorted_probs.unsqueeze(-1)

        # tokens_per_expert: how many tokens go to each expert globally.
        tokens_per_expert = torch.zeros(E, dtype=torch.long, device=hidden_states.device)
        tokens_per_expert.scatter_add_(
            0, sorted_expert_ids, torch.ones_like(sorted_expert_ids)
        )

        # ---- All-to-All (autograd-safe) ----
        send_counts = self._compute_send_counts(tokens_per_expert)
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=self.ep_group)

        send_splits = send_counts.tolist()
        recv_splits = recv_counts.tolist()

        recv_tokens = all_to_all_fwd(
            permuted_tokens, send_splits, recv_splits, self.ep_group
        )

        # Per-expert token counts for received data:
        # All-reduce the local tpe to get global tpe, then extract our experts.
        global_tpe = tokens_per_expert.clone()
        dist.all_reduce(global_tpe, group=self.ep_group)
        start_e = self.ep_rank * self.num_local_experts
        end_e = start_e + self.num_local_experts
        local_tpe = global_tpe[start_e:end_e]

        meta = DispatchMeta(
            permute_indices=sorted_token_idx,
            tokens_per_expert=local_tpe,
            send_counts=send_counts,
            recv_counts=recv_counts,
            original_shape=(T, H),
            num_local_experts=self.num_local_experts,
        )
        return recv_tokens, meta

    # ------------------------------------------------------------------
    # Combine
    # ------------------------------------------------------------------

    def combine(
        self,
        expert_output: torch.Tensor,
        meta: DispatchMeta,
    ) -> torch.Tensor:
        T, H = meta.original_shape

        # All-to-All reverse (autograd-safe).
        send_splits = meta.recv_counts.tolist()
        recv_splits = meta.send_counts.tolist()

        returned = all_to_all_bwd(
            expert_output, send_splits, recv_splits, self.ep_group
        )

        # Unpermute and reduce: each original token may have K contributions.
        output = torch.zeros(T, H, dtype=expert_output.dtype,
                             device=expert_output.device)
        output.scatter_add_(
            0,
            meta.permute_indices.unsqueeze(-1).expand_as(returned),
            returned,
        )
        return output

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_send_counts(self, tokens_per_expert: torch.Tensor) -> torch.Tensor:
        send_counts = torch.zeros(
            self.ep_size, dtype=torch.long, device=tokens_per_expert.device
        )
        for r in range(self.ep_size):
            start_e = r * self.num_local_experts
            end_e = start_e + self.num_local_experts
            send_counts[r] = tokens_per_expert[start_e:end_e].sum()
        return send_counts

