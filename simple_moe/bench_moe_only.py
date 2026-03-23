"""
SimpleMoE MoE-layers-only benchmark (fair comparison with Megatron).
Tests only the MoE layer stack, no attention/embedding.

Usage: torchrun --nproc_per_node=8 simple_moe/bench_moe_only.py
"""
from __future__ import annotations

import os
import time

import torch
import torch.distributed as dist

from simple_moe.config import MoEModelConfig
from simple_moe.moe_layer import MoELayer


def run_config(tag, hidden, num_experts, top_k, expert_ffn, num_layers,
               batch_size, seq_len, ws, rank):
    cfg = MoEModelConfig(
        hidden_dim=hidden, num_experts=num_experts, top_k=top_k,
        expert_ffn_dim=expert_ffn, score_func="sigmoid",
        load_balance="aux_loss", aux_loss_coeff=0.01,
    )
    ep = dist.new_group(list(range(ws)))

    layers = torch.nn.ModuleList([
        MoELayer(cfg, ep_group=ep, edp_group=ep) for _ in range(num_layers)
    ]).to(torch.bfloat16).cuda()

    total_params = sum(p.numel() for p in layers.parameters())
    optimizer = torch.optim.AdamW(layers.parameters(), lr=1e-4)

    if rank == 0:
        print(f"  Config: H={hidden} E={num_experts} K={top_k} FFN={expert_ffn} "
              f"layers={num_layers} BS={batch_size} SL={seq_len}")
        print(f"  Params/GPU (MoE layers): {total_params / 1e6:.1f}M")

    T = batch_size * seq_len
    num_warmup = 3
    num_steps = 10

    if rank == 0:
        print(f"  Warming up ({num_warmup} steps)...")
    for _ in range(num_warmup):
        x = torch.randn(T, hidden, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        for layer in layers:
            out, aux = layer(x)
            x = out
        loss = x.float().sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dist.barrier()

    if rank == 0:
        print(f"  Benchmarking ({num_steps} steps)...")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for step in range(num_steps):
        x = torch.randn(T, hidden, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        for layer in layers:
            out, aux = layer(x)
            x = out
        loss = x.float().sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if rank == 0:
            print(f"    step {step} loss={loss.item():.4f}")

    torch.cuda.synchronize()
    dist.barrier()
    elapsed = time.perf_counter() - t0

    if rank == 0:
        tps = T * ws * num_steps / elapsed
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"\n  --- SimpleMoE {tag} Results ---")
        print(f"  ms/step        : {elapsed / num_steps * 1000:.1f}")
        print(f"  Tokens/s total : {tps:,.0f}")
        print(f"  Tokens/s/GPU   : {tps / ws:,.0f}")
        print(f"  Peak mem/GPU   : {mem:.2f} GB")


def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    ws = dist.get_world_size()

    if rank == 0:
        print("=" * 60)
        print("SimpleMoE MoE-Layers-Only Benchmark")
        print(f"  GPUs: {ws}x {torch.cuda.get_device_name(0)}")
        print("=" * 60)

    if rank == 0:
        print("\n[small config]")
    run_config("small", hidden=2048, num_experts=ws * 2, top_k=2,
               expert_ffn=1024, num_layers=6,
               batch_size=8, seq_len=512, ws=ws, rank=rank)

    torch.cuda.reset_peak_memory_stats()
    dist.barrier()

    if rank == 0:
        print("\n\n[large config]")
    run_config("large", hidden=4096, num_experts=ws * 4, top_k=2,
               expert_ffn=2048, num_layers=10,
               batch_size=4, seq_len=1024, ws=ws, rank=rank)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
