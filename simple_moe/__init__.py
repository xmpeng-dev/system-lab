"""
SimpleMoE: A lightweight MoE training framework.

Built on veScale-FSDP sharding ideas + Megatron-Core MoE implementation logic.
Parallel strategy: FSDP + EP + PP only (no TP / CP).
"""

from simple_moe.config import (
    ClusterConfig,
    MoEModelConfig,
    TrainConfig,
)
from simple_moe.trainer import SimpleMoETrainer

__all__ = [
    "ClusterConfig",
    "MoEModelConfig",
    "TrainConfig",
    "SimpleMoETrainer",
]
