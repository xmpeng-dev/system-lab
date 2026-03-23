"""
Single-node 8-GPU smoke test for SimpleMoE.

Tests the core MoE data path: Router → All-to-All Dispatch → GroupedMLP → Combine.
No Pipeline Parallel, no FSDP – pure EP on 8 GPUs to validate correctness and
measure baseline throughput.

Usage:
  torchrun --nproc_per_node=8 simple_moe/test_single_node.py
"""

from __future__ import annotations

import os
import time

import torch
import torch.distributed as dist

from simple_moe.config import MoEModelConfig
from simple_moe.model import MoETransformerLM


def main() -> None:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # ---- Config: small model for smoke test ----
    cfg = MoEModelConfig(
        vocab_size=32000,
        hidden_dim=2048,
        num_layers=8,
        num_heads=16,
        head_dim=128,
        num_moe_layers=6,
        num_experts=world_size * 2,  # 16 experts, 2 per GPU
        top_k=2,
        expert_ffn_dim=1024,
        dense_ffn_dim=4096,
        score_func="sigmoid",
        num_shared_experts=0,
        load_balance="aux_loss",
        aux_loss_coeff=0.01,
        max_seq_len=512,
    )

    batch_size = 8
    seq_len = 512
    num_warmup = 3
    num_steps = 10

    if rank == 0:
        print("=" * 60)
        print("SimpleMoE Single-Node Test")
        print("=" * 60)
        print(f"  GPUs        : {world_size}x {torch.cuda.get_device_name(0)}")
        print(f"  Model       : {cfg.num_layers} layers, hidden={cfg.hidden_dim}")
        print(f"  MoE layers  : {cfg.num_moe_layers}, {cfg.num_experts} experts, top-{cfg.top_k}")
        print(f"  Expert FFN  : {cfg.expert_ffn_dim}")
        print(f"  Batch       : {batch_size} x {seq_len}")
        print(f"  dtype       : bfloat16")
        print("=" * 60)

    # ---- EP group = all GPUs ----
    ep_group = dist.new_group(list(range(world_size)))

    # ---- Build model ----
    model = MoETransformerLM(
        cfg, ep_group=ep_group, edp_group=ep_group,
    ).to(torch.bfloat16).cuda()

    total_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"\n  Total params (this GPU): {total_params / 1e6:.1f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # ---- Warm-up ----
    if rank == 0:
        print(f"\n  Warming up ({num_warmup} steps)...")
    for _ in range(num_warmup):
        ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device="cuda")
        labels = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device="cuda")
        loss, aux = model(ids, labels=labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dist.barrier()

    # ---- Benchmark ----
    if rank == 0:
        print(f"  Benchmarking ({num_steps} steps)...\n")

    torch.cuda.synchronize()
    t_start = time.perf_counter()
    total_tokens = 0
    losses = []

    for step in range(num_steps):
        ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device="cuda")
        labels = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device="cuda")

        loss, aux = model(ids, labels=labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        total_tokens += batch_size * seq_len * world_size
        losses.append(loss.item())

        if rank == 0:
            print(f"    step {step:>3d}  loss={loss.item():.4f}  aux={aux.item():.6f}")

    torch.cuda.synchronize()
    dist.barrier()
    elapsed = time.perf_counter() - t_start

    # ---- Report ----
    if rank == 0:
        tokens_per_sec = total_tokens / elapsed
        tokens_per_sec_per_gpu = tokens_per_sec / world_size
        ms_per_step = elapsed / num_steps * 1000

        # Rough FLOP estimate: ~6 * active_params * tokens (forward+backward)
        active_params_per_token = (
            cfg.hidden_dim * cfg.hidden_dim * 4  # attention
            + cfg.hidden_dim * cfg.expert_ffn_dim * 3 * cfg.top_k  # MoE active
        ) * cfg.num_moe_layers + (
            cfg.hidden_dim * cfg.hidden_dim * 4
            + cfg.hidden_dim * cfg.dense_ffn_dim * 3
        ) * cfg.num_dense_layers
        flops_per_step = 6 * active_params_per_token * batch_size * seq_len
        tflops = flops_per_step / (ms_per_step / 1000) / 1e12

        peak_mem = torch.cuda.max_memory_allocated() / 1e9

        print("\n" + "=" * 60)
        print("Results")
        print("=" * 60)
        print(f"  Elapsed          : {elapsed:.2f} s")
        print(f"  ms/step          : {ms_per_step:.1f} ms")
        print(f"  Tokens/s (total) : {tokens_per_sec:,.0f}")
        print(f"  Tokens/s/GPU     : {tokens_per_sec_per_gpu:,.0f}")
        print(f"  ~TFLOPS/GPU      : {tflops / world_size:.1f} (rough estimate)")
        print(f"  Peak GPU mem     : {peak_mem:.2f} GB")
        print(f"  Avg loss         : {sum(losses)/len(losses):.4f}")
        print("=" * 60)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
