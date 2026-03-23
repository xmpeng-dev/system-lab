"""
Configuration dataclasses for SimpleMoE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ClusterConfig:
    """Hardware topology and parallel-dimension sizes."""

    total_gpus: int = 128
    gpus_per_node: int = 8

    # Three parallel dimensions: PP × EP × FSDP == total_gpus
    pp_size: int = 4
    ep_size: int = 8
    fsdp_size: int = 4

    vpp_size: int = 1  # virtual pipeline parallel chunks

    # [veScale] Block-wise quantisation block size.
    # When set, the Structure-Aware Planner aligns RaggedShard blocks to this.
    quant_block_size: Optional[int] = None

    def __post_init__(self) -> None:
        assert self.pp_size * self.ep_size * self.fsdp_size == self.total_gpus, (
            f"pp({self.pp_size}) * ep({self.ep_size}) * fsdp({self.fsdp_size}) "
            f"!= total_gpus({self.total_gpus})"
        )


@dataclass
class MoEModelConfig:
    """Model architecture hyper-parameters."""

    vocab_size: int = 128256
    hidden_dim: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    head_dim: int = 128

    # MoE-specific
    num_moe_layers: int = 30  # last N layers are MoE, rest are dense
    num_experts: int = 64
    top_k: int = 4
    expert_ffn_dim: int = 2048
    dense_ffn_dim: int = 11008
    score_func: str = "sigmoid"  # 'softmax' | 'sigmoid'

    # [Megatron] Shared expert (DeepSeek-V3 style)
    num_shared_experts: int = 0
    shared_expert_ffn_dim: int = 2048

    # [Megatron] Load-balancing strategy
    load_balance: str = "aux_loss_free"  # 'aux_loss' | 'aux_loss_free'
    aux_loss_coeff: float = 0.01

    max_seq_len: int = 4096

    @property
    def num_dense_layers(self) -> int:
        return self.num_layers - self.num_moe_layers


@dataclass
class TrainConfig:
    """End-to-end training configuration."""

    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    model: MoEModelConfig = field(default_factory=MoEModelConfig)

    # Optimiser
    optimizer_type: str = "adam"  # 'adam' | 'shampoo' | 'muon'
    lr: float = 3e-4
    expert_lr: float = 1e-4
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0

    # Training
    batch_size: int = 1024
    seq_len: int = 4096
    num_micro_batches: int = 8

    # Memory
    enable_recomputation: bool = True
    enable_offloading: bool = False
    memory_budget_gb: float = 70.0

    # FP8
    fp8_enabled: bool = False
    fp8_recipe: str = "blockwise_fp8"

    # Misc
    seed: int = 42
    log_interval: int = 10
