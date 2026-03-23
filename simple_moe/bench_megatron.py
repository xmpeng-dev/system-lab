"""
Megatron-Core MoE benchmark for comparison with SimpleMoE.

Tests MoE layers only (Router → Dispatch → Expert Compute → Combine)
with matching configs on 8x MI355X.

Usage: torchrun --nproc_per_node=8 simple_moe/bench_megatron.py
"""
from __future__ import annotations

import os
import sys
import time
import traceback

import torch
import torch.distributed as dist

MEGATRON_PATH = "/home/xiaompen/Megatron-LM-v13.0"
if MEGATRON_PATH not in sys.path:
    sys.path.insert(0, MEGATRON_PATH)


def build_moe_layer(config, layer_number):
    """Build a single MoELayer with proper ModuleSpec submodules."""
    from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
    from megatron.core.transformer.moe.experts import SequentialMLP
    from megatron.core.transformer.mlp import MLPSubmodules
    from megatron.core.transformer.spec_utils import ModuleSpec

    # SequentialMLP needs MLPSubmodules with linear specs.
    # Use plain nn.Linear based column/row parallel stubs since TP=1.
    from megatron.core.models.gpt.gpt_layer_specs import LocalSpecProvider
    backend = LocalSpecProvider()
    linear_fc1 = backend.column_parallel_linear()
    linear_fc2 = backend.row_parallel_linear()
    mlp_submodules = MLPSubmodules(linear_fc1=linear_fc1, linear_fc2=linear_fc2)

    experts_spec = ModuleSpec(module=SequentialMLP, submodules=mlp_submodules)
    submodules = MoESubmodules(experts=experts_spec)

    return MoELayer(config, submodules=submodules, layer_number=layer_number)


def run_config(tag, hidden, num_experts, top_k, expert_ffn, num_layers,
               batch_size, seq_len, ws, rank):
    from megatron.core.transformer.transformer_config import TransformerConfig

    config = TransformerConfig(
        num_layers=num_layers,
        hidden_size=hidden,
        num_attention_heads=hidden // 128,
        ffn_hidden_size=expert_ffn,
        num_moe_experts=num_experts,
        moe_router_topk=top_k,
        moe_ffn_hidden_size=expert_ffn,
        moe_router_load_balancing_type="aux_loss",
        moe_aux_loss_coeff=0.01,
        moe_router_score_function="sigmoid",
        moe_token_dispatcher_type="alltoall",
        moe_grouped_gemm=False,
        add_bias_linear=False,
        bias_activation_fusion=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=ws,
    )

    layers = torch.nn.ModuleList([
        build_moe_layer(config, layer_number=i) for i in range(num_layers)
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
        x = torch.randn(T, hidden, dtype=torch.bfloat16, device="cuda")
        for layer in layers:
            out, bias = layer(x)
            x = out
        loss = x.sum()
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
        x = torch.randn(T, hidden, dtype=torch.bfloat16, device="cuda")
        for layer in layers:
            out, bias = layer(x)
            x = out
        loss = x.sum()
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
        print(f"\n  --- Megatron {tag} Results ---")
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

    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    if rank == 0:
        print("=" * 60)
        print("Megatron-Core v0.13 MoE Benchmark")
        print(f"  GPUs: {ws}x {torch.cuda.get_device_name(0)}")
        print("=" * 60)

    # ----------- small config -----------
    if rank == 0:
        print("\n[small config]")
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=ws,
    )
    model_parallel_cuda_manual_seed(42)
    try:
        run_config(
            tag="small", hidden=2048, num_experts=ws * 2, top_k=2,
            expert_ffn=1024, num_layers=6,
            batch_size=8, seq_len=512, ws=ws, rank=rank,
        )
    except Exception:
        if rank == 0:
            traceback.print_exc()
    parallel_state.destroy_model_parallel()
    dist.barrier()

    # ----------- large config -----------
    if rank == 0:
        print("\n\n[large config]")
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=ws,
    )
    model_parallel_cuda_manual_seed(42)
    try:
        run_config(
            tag="large", hidden=4096, num_experts=ws * 4, top_k=2,
            expert_ffn=2048, num_layers=10,
            batch_size=4, seq_len=1024, ws=ws, rank=rank,
        )
    except Exception:
        if rank == 0:
            traceback.print_exc()
    parallel_state.destroy_model_parallel()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
