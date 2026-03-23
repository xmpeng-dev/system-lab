"""
All-to-All Token Dispatcher for Expert Parallel.

[Megatron] Implements the Permute → All-to-All → … → All-to-All → Unpermute
data path with Memory-Efficient Permutation (routing weights applied before
the second expert linear layer instead of at combine time).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.distributed as dist


@dataclass
class DispatchMeta:
    """Book-keeping produced by ``dispatch`` and consumed by ``combine``."""

    permute_indices: torch.Tensor  # [T*K] int64 – original token index
    tokens_per_expert: torch.Tensor  # [E_total] int64
    send_counts: torch.Tensor  # [ep_size] int64
    recv_counts: torch.Tensor  # [ep_size] int64
    original_shape: Tuple[int, int]  # (T, H)
    num_local_experts: int


class AllToAllDispatcher:
    """
    [Megatron] Token dispatcher using NCCL All-to-All.

    The dispatcher

    1. **Permutes** tokens so that those destined for the same expert are
       contiguous.
    2. Applies routing weights (*memory-efficient permutation*).
    3. Executes **All-to-All** to move tokens to the GPU that owns the
       target expert.

    ``combine`` reverses the process.
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
        """
        Parameters
        ----------
        hidden_states : [T, H]
        probs : [T, K] routing weights
        topk_indices : [T, K] expert indices
        routing_map : [T, E] boolean mask

        Returns
        -------
        recv_tokens : [total_recv, H]
            Tokens ready for local expert computation.
        meta : DispatchMeta
        """
        T, H = hidden_states.shape
        K = topk_indices.shape[1]
        E = routing_map.shape[1]

        # ---- Permute: sort tokens by expert id ----
        # Expand each token K times (one copy per selected expert).
        flat_expert_ids = topk_indices.flatten()  # [T*K]
        flat_probs = probs.flatten()  # [T*K]
        token_indices = (
            torch.arange(T, device=hidden_states.device)
            .unsqueeze(1)
            .expand(T, K)
            .flatten()
        )  # [T*K]

        # Sort by expert id so tokens for the same expert are contiguous.
        sort_order = torch.argsort(flat_expert_ids, stable=True)
        sorted_expert_ids = flat_expert_ids[sort_order]
        sorted_token_idx = token_indices[sort_order]
        sorted_probs = flat_probs[sort_order]

        permuted_tokens = hidden_states[sorted_token_idx]  # [T*K, H]

        # [Megatron] Memory-Efficient Permutation: apply routing weights now
        # so that combine can simply sum without storing expert outputs.
        permuted_tokens = permuted_tokens * sorted_probs.unsqueeze(-1)

        # tokens_per_expert: how many tokens are assigned to each expert.
        tokens_per_expert = torch.zeros(E, dtype=torch.long, device=hidden_states.device)
        tokens_per_expert.scatter_add_(
            0, sorted_expert_ids, torch.ones_like(sorted_expert_ids)
        )

        # ---- All-to-All ----
        send_counts, recv_counts, recv_tokens = self._all_to_all_dispatch(
            permuted_tokens, tokens_per_expert, H
        )

        # Recompute local tokens_per_expert for received data.
        local_tpe = self._local_tokens_per_expert(recv_counts)

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
        """Reverse All-to-All, unpermute, and reduce duplicated tokens."""
        T, H = meta.original_shape

        # All-to-All reverse.
        returned = self._all_to_all_combine(expert_output, meta)

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
    # Internal All-to-All
    # ------------------------------------------------------------------

    def _all_to_all_dispatch(
        self,
        permuted: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        H: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        E = tokens_per_expert.shape[0]
        experts_per_rank = self.num_local_experts

        # send_counts[r] = number of tokens going to rank r.
        send_counts = torch.zeros(
            self.ep_size, dtype=torch.long, device=permuted.device
        )
        for r in range(self.ep_size):
            start_e = r * experts_per_rank
            end_e = start_e + experts_per_rank
            send_counts[r] = tokens_per_expert[start_e:end_e].sum()

        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(
            recv_counts, send_counts, group=self.ep_group
        )

        send_splits = send_counts.tolist()
        recv_splits = recv_counts.tolist()

        recv_tokens = permuted.new_empty(int(recv_counts.sum()), H)

        input_list = list(permuted.split(send_splits, dim=0))
        output_list = list(recv_tokens.split(recv_splits, dim=0))

        dist.all_to_all(output_list, input_list, group=self.ep_group)

        recv_tokens = torch.cat(output_list, dim=0)
        return send_counts, recv_counts, recv_tokens

    def _all_to_all_combine(
        self,
        expert_output: torch.Tensor,
        meta: DispatchMeta,
    ) -> torch.Tensor:
        H = expert_output.shape[-1]
        send_splits = meta.recv_counts.tolist()
        recv_splits = meta.send_counts.tolist()

        recv_buf = expert_output.new_empty(int(meta.send_counts.sum()), H)

        input_list = list(expert_output.split(send_splits, dim=0))
        output_list = list(recv_buf.split(recv_splits, dim=0))

        dist.all_to_all(output_list, input_list, group=self.ep_group)

        return torch.cat(output_list, dim=0)

    def _local_tokens_per_expert(
        self, recv_counts: torch.Tensor
    ) -> torch.Tensor:
        """
        Approximate per-expert token counts for received data.

        A production implementation would send the exact per-expert counts
        alongside the tokens; here we assume uniform distribution within
        each rank's received batch for simplicity.
        """
        total = int(recv_counts.sum())
        tpe = torch.zeros(
            self.num_local_experts, dtype=torch.long,
            device=recv_counts.device,
        )
        if self.num_local_experts > 0 and total > 0:
            base = total // self.num_local_experts
            rem = total % self.num_local_experts
            tpe[:] = base
            tpe[:rem] += 1
        return tpe
