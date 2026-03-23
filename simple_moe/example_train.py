"""
Example: launch a SimpleMoE training run.

Usage (single-node 8-GPU):
  torchrun --nproc_per_node=8 simple_moe/example_train.py

Usage (multi-node):
  torchrun --nnodes=16 --nproc_per_node=8 \
           --rdzv_backend=c10d --rdzv_endpoint=$MASTER:29500 \
           simple_moe/example_train.py
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from simple_moe.config import ClusterConfig, MoEModelConfig, TrainConfig
from simple_moe.trainer import SimpleMoETrainer


def main() -> None:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    world_size = dist.get_world_size()

    # --- Configuration (adjust to your hardware) ---
    # The product pp_size * ep_size * fsdp_size must equal world_size.
    if world_size >= 128:
        cluster = ClusterConfig(
            total_gpus=world_size, gpus_per_node=8,
            pp_size=4, ep_size=8, fsdp_size=world_size // 32,
        )
    elif world_size >= 8:
        cluster = ClusterConfig(
            total_gpus=world_size, gpus_per_node=min(8, world_size),
            pp_size=1, ep_size=min(8, world_size), fsdp_size=max(1, world_size // 8),
        )
    else:
        cluster = ClusterConfig(
            total_gpus=world_size, gpus_per_node=world_size,
            pp_size=1, ep_size=world_size, fsdp_size=1,
        )

    model_cfg = MoEModelConfig(
        vocab_size=32000,
        hidden_dim=1024,
        num_layers=8,
        num_heads=16,
        head_dim=64,
        num_moe_layers=6,
        num_experts=cluster.ep_size * 2,  # 2 experts per GPU
        top_k=2,
        expert_ffn_dim=512,
        dense_ffn_dim=2048,
        score_func="sigmoid",
        num_shared_experts=0,
        load_balance="aux_loss",
        max_seq_len=512,
    )

    config = TrainConfig(
        cluster=cluster,
        model=model_cfg,
        optimizer_type="adam",
        lr=3e-4,
        expert_lr=1e-4,
        batch_size=max(16, world_size * 2),
        seq_len=512,
        num_micro_batches=max(1, cluster.pp_size * 2),
        enable_recomputation=False,
        enable_offloading=False,
    )

    if dist.get_rank() == 0:
        print(f"[SimpleMoE] world_size={world_size}  "
              f"pp={cluster.pp_size} ep={cluster.ep_size} fsdp={cluster.fsdp_size}")
        print(f"[SimpleMoE] model: {model_cfg.num_layers} layers, "
              f"{model_cfg.num_experts} experts, top-{model_cfg.top_k}")

    trainer = SimpleMoETrainer(config)

    # --- Dummy training loop ---
    num_steps = 20
    for step in range(num_steps):
        batch = {
            "input_ids": torch.randint(
                0, model_cfg.vocab_size,
                (config.batch_size, config.seq_len),
                device="cuda",
            ),
            "labels": torch.randint(
                0, model_cfg.vocab_size,
                (config.batch_size, config.seq_len),
                device="cuda",
            ),
        }
        loss = trainer.train_step(batch)
        if dist.get_rank() == 0 and step % config.log_interval == 0:
            print(f"  step {step:>4d}  loss={loss:.4f}")

    if dist.get_rank() == 0:
        print("[SimpleMoE] Training complete.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
