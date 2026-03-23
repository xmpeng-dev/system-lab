"""
SimpleMoE Trainer: orchestrates FSDP + EP + PP training.

[Megatron] Pipeline schedule (1F1B), EP overlap, memory optimisation.
[veScale] RaggedFSDP for dense layers, Lazy AG, Hierarchical RS.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from simple_moe.config import ClusterConfig, MoEModelConfig, TrainConfig
from simple_moe.fsdp import RaggedFSDP
from simple_moe.memory import (
    ActivationOffloader,
    configure_offloading,
    configure_recomputation,
)
from simple_moe.model import MoETransformerLM, TransformerBlock
from simple_moe.moe_layer import MoELayer
from simple_moe.overlap import EPOverlapScheduler
from simple_moe.pipeline import (
    ActionType,
    OneFOneBSchedule,
    pp_recv,
    pp_send,
)
from simple_moe.planner import DistributedPlan, StructureAwarePlanner
from simple_moe.process_groups import ProcessGroups


class SimpleMoETrainer:
    """
    End-to-end distributed trainer for MoE Transformer models.

    Combines:
      - [veScale] RaggedFSDP for dense parameters
      - [Megatron] Expert Parallel for MoE layers
      - [Megatron] 1F1B pipeline schedule
      - [Megatron] EP comm/compute overlap
    """

    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.device = torch.device("cuda", torch.cuda.current_device())

        # ---- Process groups ----
        self.pg = ProcessGroups(config.cluster)

        # ---- Planning ----
        self.planner = StructureAwarePlanner(
            config.model, config.cluster, config.optimizer_type,
        )
        self.plan: DistributedPlan = self.planner.plan()

        # ---- Model ----
        self.model = self._build_model()

        # ---- FSDP for dense layers [veScale] ----
        self._wrap_fsdp()

        # ---- Memory optimisation [Megatron] ----
        if config.enable_recomputation:
            configure_recomputation(self.model)
        self.offloader: Optional[ActivationOffloader] = None
        if config.enable_offloading:
            self.offloader = ActivationOffloader()
            configure_offloading(self.model, self.offloader)

        # ---- EP overlap [Megatron] ----
        self.ep_overlap = EPOverlapScheduler()

        # ---- Optimiser ----
        self.optimizer = self._build_optimizer()

        # ---- Pipeline schedule [Megatron] ----
        self.schedule = OneFOneBSchedule(
            num_stages=config.cluster.pp_size,
            num_micro_batches=config.num_micro_batches,
            stage_id=self.pg.pp_rank,
            pp_group=self.pg.pp_group,
        )

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _build_model(self) -> nn.Module:
        """Build the model and keep only this PP stage's layers."""
        full_model = MoETransformerLM(
            self.config.model,
            ep_group=self.pg.ep_group,
            edp_group=self.pg.edp_group,
        ).to(self.device)

        my_layer_ids = self.plan.pp_partition[self.pg.pp_rank]

        kept_layers = nn.ModuleList(
            [full_model.layers[i] for i in my_layer_ids]
        )

        # Keep embedding / output head only on first / last stage.
        is_first = self.pg.pp_rank == 0
        is_last = self.pg.pp_rank == self.config.cluster.pp_size - 1

        stage_model = _StageModel(
            layers=kept_layers,
            embed=full_model.embed_tokens if is_first else None,
            norm_f=full_model.norm_f if is_last else None,
            output_head=full_model.output_head if is_last else None,
        )
        return stage_model

    def _wrap_fsdp(self) -> None:
        """[veScale] Wrap dense sub-modules with RaggedFSDP."""
        fsdp_plan = self.plan.fsdp_plan
        for layer in self.model.layers:
            if not isinstance(layer, TransformerBlock):
                continue
            layer.attn = RaggedFSDP(layer.attn, self.pg.fsdp_group, fsdp_plan)
            # Norms are small; wrap together for one AG call.
            layer.norm1 = RaggedFSDP(layer.norm1, self.pg.fsdp_group, fsdp_plan)
            layer.norm2 = RaggedFSDP(layer.norm2, self.pg.fsdp_group, fsdp_plan)

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------

    def _build_optimizer(self) -> torch.optim.Optimizer:
        dense_params: list[nn.Parameter] = []
        expert_params: list[nn.Parameter] = []

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if any(tag in name for tag in ("experts", "w_gate", "w_up", "w_down")):
                expert_params.append(p)
            else:
                dense_params.append(p)

        cfg = self.config
        param_groups = []
        if dense_params:
            param_groups.append({"params": dense_params, "lr": cfg.lr})
        if expert_params:
            param_groups.append({"params": expert_params, "lr": cfg.expert_lr})

        return torch.optim.AdamW(
            param_groups,
            weight_decay=cfg.weight_decay,
        )

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Execute one training iteration (all micro-batches).

        Parameters
        ----------
        batch : dict
            Must contain ``"input_ids"`` [B, S] and ``"labels"`` [B, S].

        Returns
        -------
        loss : float
            Average loss across micro-batches.
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        micro_batches = self._split_micro_batches(batch)
        actions = self.schedule.steps()

        saved_activations: dict[int, torch.Tensor] = {}
        saved_inputs: dict[int, torch.Tensor] = {}
        total_loss = torch.tensor(0.0, device=self.device)

        for act in actions:
            mb = micro_batches[act.mb_id] if act.mb_id < len(micro_batches) else None

            if act.action == ActionType.RECV_FWD:
                shape = self._activation_shape()
                saved_inputs[act.mb_id] = pp_recv(
                    self.pg.pp_rank - 1,
                    shape, torch.bfloat16, self.device,
                    self.pg.pp_group,
                )

            elif act.action == ActionType.FORWARD:
                inp = saved_inputs.get(act.mb_id)
                out, loss = self._forward_chunk(mb, inp)
                saved_activations[act.mb_id] = out
                if loss is not None:
                    total_loss += loss.detach()

            elif act.action == ActionType.SEND_FWD:
                pp_send(
                    saved_activations[act.mb_id],
                    self.pg.pp_rank + 1,
                    self.pg.pp_group,
                )

            elif act.action == ActionType.RECV_BWD:
                shape = self._activation_shape()
                grad = pp_recv(
                    self.pg.pp_rank + 1,
                    shape, torch.bfloat16, self.device,
                    self.pg.pp_group,
                )
                saved_activations[act.mb_id].backward(grad)

            elif act.action == ActionType.BACKWARD:
                act_tensor = saved_activations.pop(act.mb_id, None)
                # For the last stage, backward was already triggered via
                # loss.backward() inside _forward_chunk.
                if act_tensor is not None and act_tensor.grad_fn is not None:
                    act_tensor.backward(torch.ones_like(act_tensor))

            elif act.action == ActionType.SEND_BWD:
                inp = saved_inputs.pop(act.mb_id, None)
                if inp is not None and inp.grad is not None:
                    pp_send(inp.grad, self.pg.pp_rank - 1, self.pg.pp_group)

        # Gradient clipping.
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )

        self.optimizer.step()
        return total_loss.item() / max(len(micro_batches), 1)

    # ------------------------------------------------------------------
    # Forward for one micro-batch
    # ------------------------------------------------------------------

    def _forward_chunk(
        self,
        micro_batch: Optional[Dict[str, torch.Tensor]],
        recv_input: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run the local layers for one micro-batch."""
        model: _StageModel = self.model  # type: ignore[assignment]

        # First stage: embed tokens.
        if model.embed is not None and micro_batch is not None:
            x = model.embed(micro_batch["input_ids"])
        elif recv_input is not None:
            x = recv_input.requires_grad_(True)
        else:
            raise RuntimeError("No input available for forward chunk")

        total_aux = x.new_tensor(0.0)
        for layer in model.layers:
            x, aux = layer(x)
            total_aux = total_aux + aux

        loss: Optional[torch.Tensor] = None
        if model.output_head is not None:
            x = model.norm_f(x)
            logits = model.output_head(x)
            if micro_batch is not None and "labels" in micro_batch:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = micro_batch["labels"][..., 1:].contiguous()
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
                loss = loss + total_aux
                loss.backward()

        return x, loss

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _split_micro_batches(
        self, batch: Dict[str, torch.Tensor]
    ) -> List[Dict[str, torch.Tensor]]:
        M = self.config.num_micro_batches
        result: list[dict[str, torch.Tensor]] = []
        B = next(iter(batch.values())).shape[0]
        mb_size = max(B // M, 1)
        for i in range(M):
            start = i * mb_size
            end = min(start + mb_size, B)
            if start >= B:
                break
            result.append({k: v[start:end] for k, v in batch.items()})
        return result

    def _activation_shape(self) -> torch.Size:
        cfg = self.config
        mb = max(cfg.batch_size // cfg.num_micro_batches, 1)
        return torch.Size([mb, cfg.seq_len, cfg.model.hidden_dim])


# ---------------------------------------------------------------------------
# Stage model wrapper
# ---------------------------------------------------------------------------

class _StageModel(nn.Module):
    """Holds only the layers belonging to this PP stage."""

    def __init__(
        self,
        layers: nn.ModuleList,
        embed: Optional[nn.Embedding],
        norm_f: Optional[nn.Module],
        output_head: Optional[nn.Linear],
    ) -> None:
        super().__init__()
        self.layers = layers
        self.embed = embed
        self.norm_f = norm_f
        self.output_head = output_head

    def forward(self, *args, **kwargs):
        raise RuntimeError("Use trainer._forward_chunk instead")
