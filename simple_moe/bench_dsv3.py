"""
DeepSeek-V3-like MoE benchmark on 8x MI355X.

DSv3 key specs:
  hidden_dim     = 7168
  num_experts    = 256 (routed) + 1 shared
  top_k          = 8
  expert_ffn_dim = 2048  (each expert is small)
  num_heads      = 128  (MLA in real DSv3, standard MHA here)
  head_dim       = 128

We test 1 and 2 full transformer layers with EP=8 (32 experts/GPU).

Usage: torchrun --nproc_per_node=8 simple_moe/bench_dsv3.py
"""
from __future__ import annotations

import os
import sys
import time
import traceback

import torch
import torch.distributed as dist


def bench_simple_moe(tag, num_layers, batch_size, seq_len, ws, rank):
    """Benchmark SimpleMoE with DSv3-like config."""
    from simple_moe.config import MoEModelConfig
    from simple_moe.model import MoETransformerLM

    cfg = MoEModelConfig(
        vocab_size=32000,       # smaller vocab for test
        hidden_dim=7168,
        num_layers=num_layers,
        num_heads=56,           # 56 heads × 128 dim = 7168
        head_dim=128,
        num_moe_layers=num_layers,  # all layers are MoE
        num_experts=256,
        top_k=8,
        expert_ffn_dim=2048,
        dense_ffn_dim=18432,    # not used since all layers are MoE
        score_func="sigmoid",
        num_shared_experts=0,   # skip shared expert for simplicity
        load_balance="aux_loss",
        aux_loss_coeff=0.01,
        max_seq_len=seq_len,
    )

    ep = dist.new_group(list(range(ws)))
    model = MoETransformerLM(cfg, ep_group=ep, edp_group=ep).to(torch.bfloat16).cuda()

    total_params = sum(p.numel() for p in model.parameters())
    expert_params = sum(p.numel() for n, p in model.named_parameters() if "expert" in n or "w_gate" in n or "w_up" in n or "w_down" in n)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    if rank == 0:
        print(f"\n  [SimpleMoE] DSv3-like {num_layers}L, BS={batch_size} SL={seq_len}")
        print(f"    Total params/GPU : {total_params / 1e9:.2f}B")
        print(f"    Expert params/GPU: {expert_params / 1e9:.2f}B")
        peak_before = torch.cuda.max_memory_allocated() / 1e9
        print(f"    Mem after init   : {peak_before:.2f} GB")

    num_warmup = 2
    num_steps = 5

    if rank == 0:
        print(f"    Warming up ({num_warmup} steps)...")
    for _ in range(num_warmup):
        ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device="cuda")
        loss, aux = model(ids, labels=ids)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dist.barrier()

    if rank == 0:
        print(f"    Benchmarking ({num_steps} steps)...")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for step in range(num_steps):
        ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device="cuda")
        loss, aux = model(ids, labels=ids)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if rank == 0:
            print(f"      step {step} loss={loss.item():.4f}")

    torch.cuda.synchronize()
    dist.barrier()
    elapsed = time.perf_counter() - t0

    if rank == 0:
        T = batch_size * seq_len
        tps = T * ws * num_steps / elapsed
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"    --- SimpleMoE DSv3 {tag} ---")
        print(f"    ms/step        : {elapsed / num_steps * 1000:.1f}")
        print(f"    Tokens/s total : {tps:,.0f}")
        print(f"    Tokens/s/GPU   : {tps / ws:,.0f}")
        print(f"    Peak mem/GPU   : {mem:.2f} GB")

    del model, optimizer
    torch.cuda.empty_cache()


def bench_megatron(tag, num_layers, batch_size, seq_len, ws, rank):
    """Benchmark Megatron-Core MoE with DSv3-like config."""
    MEGATRON_PATH = "/home/xiaompen/Megatron-LM-v13.0"
    if MEGATRON_PATH not in sys.path:
        sys.path.insert(0, MEGATRON_PATH)

    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
    from megatron.core.transformer.moe.experts import SequentialMLP
    from megatron.core.transformer.mlp import MLPSubmodules
    from megatron.core.transformer.spec_utils import ModuleSpec
    from megatron.core.models.gpt.gpt_layer_specs import LocalSpecProvider

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=ws,
    )
    model_parallel_cuda_manual_seed(42)

    config = TransformerConfig(
        num_layers=num_layers,
        hidden_size=7168,
        num_attention_heads=56,
        ffn_hidden_size=2048,
        num_moe_experts=256,
        moe_router_topk=8,
        moe_ffn_hidden_size=2048,
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

    backend = LocalSpecProvider()
    linear_fc1 = backend.column_parallel_linear()
    linear_fc2 = backend.row_parallel_linear()
    mlp_sub = MLPSubmodules(linear_fc1=linear_fc1, linear_fc2=linear_fc2)
    experts_spec = ModuleSpec(module=SequentialMLP, submodules=mlp_sub)
    submodules = MoESubmodules(experts=experts_spec)

    layers = torch.nn.ModuleList([
        MoELayer(config, submodules=submodules, layer_number=i)
        for i in range(num_layers)
    ]).to(torch.bfloat16).cuda()

    total_params = sum(p.numel() for p in layers.parameters())
    optimizer = torch.optim.AdamW(layers.parameters(), lr=1e-4)

    if rank == 0:
        print(f"\n  [Megatron] DSv3-like {num_layers}L MoE-only, BS={batch_size} SL={seq_len}")
        print(f"    Params/GPU (MoE): {total_params / 1e9:.2f}B")
        peak_before = torch.cuda.max_memory_allocated() / 1e9
        print(f"    Mem after init  : {peak_before:.2f} GB")

    T = batch_size * seq_len
    num_warmup = 2
    num_steps = 5

    if rank == 0:
        print(f"    Warming up ({num_warmup} steps)...")
    for _ in range(num_warmup):
        x = torch.randn(T, 7168, dtype=torch.bfloat16, device="cuda")
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
        print(f"    Benchmarking ({num_steps} steps)...")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for step in range(num_steps):
        x = torch.randn(T, 7168, dtype=torch.bfloat16, device="cuda")
        for layer in layers:
            out, bias = layer(x)
            x = out
        loss = x.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if rank == 0:
            print(f"      step {step} loss={loss.item():.4f}")

    torch.cuda.synchronize()
    dist.barrier()
    elapsed = time.perf_counter() - t0

    if rank == 0:
        tps = T * ws * num_steps / elapsed
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"    --- Megatron DSv3 {tag} ---")
        print(f"    ms/step        : {elapsed / num_steps * 1000:.1f}")
        print(f"    Tokens/s total : {tps:,.0f}")
        print(f"    Tokens/s/GPU   : {tps / ws:,.0f}")
        print(f"    Peak mem/GPU   : {mem:.2f} GB")

    del layers, optimizer
    torch.cuda.empty_cache()
    parallel_state.destroy_model_parallel()


def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    ws = dist.get_world_size()

    if rank == 0:
        print("=" * 65)
        print("  DeepSeek-V3-like MoE Benchmark")
        print(f"  GPUs: {ws}x {torch.cuda.get_device_name(0)}")
        print(f"  DSv3 config: H=7168, E=256, K=8, FFN=2048, heads=56")
        print("=" * 65)

    BS, SL = 2, 4096

    # ===== 1 Layer =====
    if rank == 0:
        print("\n" + "=" * 65)
        print("  1 LAYER")
        print("=" * 65)

    bench_simple_moe("1L", num_layers=1, batch_size=BS, seq_len=SL, ws=ws, rank=rank)
    torch.cuda.empty_cache()
    dist.barrier()

    try:
        bench_megatron("1L", num_layers=1, batch_size=BS, seq_len=SL, ws=ws, rank=rank)
    except Exception:
        if rank == 0:
            traceback.print_exc()
    torch.cuda.empty_cache()
    dist.barrier()

    # ===== 2 Layers =====
    if rank == 0:
        print("\n" + "=" * 65)
        print("  2 LAYERS")
        print("=" * 65)

    bench_simple_moe("2L", num_layers=2, batch_size=BS, seq_len=SL, ws=ws, rank=rank)
    torch.cuda.empty_cache()
    dist.barrier()

    try:
        bench_megatron("2L", num_layers=2, batch_size=BS, seq_len=SL, ws=ws, rank=rank)
    except Exception:
        if rank == 0:
            traceback.print_exc()
    torch.cuda.empty_cache()
    dist.barrier()

    if rank == 0:
        print("\n" + "=" * 65)
        print("  DONE")
        print("=" * 65)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
