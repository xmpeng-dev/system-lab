"""
Pipeline-parallel 1F1B schedule.

[Megatron] One-Forward-One-Backward schedule with point-to-point activation
transfers between adjacent pipeline stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Iterator, List, Optional

import torch
import torch.distributed as dist


class ActionType(Enum):
    FORWARD = auto()
    BACKWARD = auto()
    SEND_FWD = auto()
    RECV_FWD = auto()
    SEND_BWD = auto()
    RECV_BWD = auto()


@dataclass
class PipelineAction:
    action: ActionType
    mb_id: int = 0


class OneFOneBSchedule:
    """
    [Megatron] 1F1B pipeline schedule generator.

    For *num_stages* stages and *num_micro_batches* micro-batches, this
    produces the canonical warm-up / steady-state / cool-down action list
    for each stage.
    """

    def __init__(
        self,
        num_stages: int,
        num_micro_batches: int,
        stage_id: int,
        pp_group: dist.ProcessGroup,
    ) -> None:
        self.num_stages = num_stages
        self.num_micro_batches = num_micro_batches
        self.stage_id = stage_id
        self.pp_group = pp_group

    def steps(self) -> List[PipelineAction]:
        """Return the ordered list of actions for this stage."""
        S = self.num_stages
        M = self.num_micro_batches
        stage = self.stage_id
        is_first = stage == 0
        is_last = stage == S - 1

        actions: list[PipelineAction] = []

        # Number of warm-up forwards = distance to last stage.
        warmup = S - stage - 1

        fwd_mb = 0
        bwd_mb = 0

        # Warm-up: only forwards.
        for _ in range(min(warmup, M)):
            if not is_first:
                actions.append(PipelineAction(ActionType.RECV_FWD, fwd_mb))
            actions.append(PipelineAction(ActionType.FORWARD, fwd_mb))
            if not is_last:
                actions.append(PipelineAction(ActionType.SEND_FWD, fwd_mb))
            fwd_mb += 1

        # Steady state: interleaved 1F1B.
        steady = M - warmup
        for _ in range(max(steady, 0)):
            if not is_first:
                actions.append(PipelineAction(ActionType.RECV_FWD, fwd_mb))
            actions.append(PipelineAction(ActionType.FORWARD, fwd_mb))
            if not is_last:
                actions.append(PipelineAction(ActionType.SEND_FWD, fwd_mb))
            fwd_mb += 1

            if not is_last:
                actions.append(PipelineAction(ActionType.RECV_BWD, bwd_mb))
            actions.append(PipelineAction(ActionType.BACKWARD, bwd_mb))
            if not is_first:
                actions.append(PipelineAction(ActionType.SEND_BWD, bwd_mb))
            bwd_mb += 1

        # Cool-down: only backwards.
        while bwd_mb < M:
            if not is_last:
                actions.append(PipelineAction(ActionType.RECV_BWD, bwd_mb))
            actions.append(PipelineAction(ActionType.BACKWARD, bwd_mb))
            if not is_first:
                actions.append(PipelineAction(ActionType.SEND_BWD, bwd_mb))
            bwd_mb += 1

        return actions


# ---------------------------------------------------------------------------
# P2P send / recv helpers
# ---------------------------------------------------------------------------

def pp_send(
    tensor: torch.Tensor,
    dest_stage: int,
    pp_group: dist.ProcessGroup,
) -> None:
    dest_rank = dist.get_global_rank(pp_group, dest_stage)
    dist.send(tensor.contiguous(), dst=dest_rank, group=pp_group)


def pp_recv(
    src_stage: int,
    shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    pp_group: dist.ProcessGroup,
) -> torch.Tensor:
    src_rank = dist.get_global_rank(pp_group, src_stage)
    buf = torch.empty(shape, dtype=dtype, device=device)
    dist.recv(buf, src=src_rank, group=pp_group)
    return buf
