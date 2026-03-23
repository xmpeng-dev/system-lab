"""
DSv3-like pure MoE layer benchmark: SimpleMoE vs Megatron-Core.
No attention, no embedding — only MoE layers for fair comparison.

Usage: torchrun --nproc_per_node=8 simple_moe/bench_dsv3_moe_only.py
"""
from __future__ import annotations
import os, sys, time, traceback, torch, torch.distributed as dist

MEGATRON_PATH = "/home/xiaompen/Megatron-LM-v13.0"
H, E, K, FFN, HEADS = 7168, 256, 8, 2048, 56
BS, SL = 2, 4096
WARMUP, STEPS = 3, 10


def bench_simple_moe(tag, num_layers, ws, rank, ep_group):
    from simple_moe.config import MoEModelConfig
    from simple_moe.moe_layer import MoELayer

    cfg = MoEModelConfig(
        hidden_dim=H, num_experts=E, top_k=K, expert_ffn_dim=FFN,
        score_func="sigmoid", load_balance="aux_loss", aux_loss_coeff=0.01,
    )
    layers = torch.nn.ModuleList([
        MoELayer(cfg, ep_group=ep_group, edp_group=ep_group)
        for _ in range(num_layers)
    ]).to(torch.bfloat16).cuda()

    tp = sum(p.numel() for p in layers.parameters())
    opt = torch.optim.AdamW(layers.parameters(), lr=1e-4)
    T = BS * SL

    if rank == 0:
        print(f"  [SimpleMoE] {num_layers}L MoE-only  Params/GPU: {tp/1e9:.2f}B")

    for _ in range(WARMUP):
        x = torch.randn(T, H, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        for layer in layers:
            x, _ = layer(x)
        x.float().sum().backward()
        opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); dist.barrier()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for s in range(STEPS):
        x = torch.randn(T, H, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        for layer in layers:
            x, _ = layer(x)
        x.float().sum().backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        if rank == 0:
            print(f"    step {s}")
    torch.cuda.synchronize(); dist.barrier()
    el = time.perf_counter() - t0

    if rank == 0:
        tps = T * ws * STEPS / el
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"  SimpleMoE {tag}: {el/STEPS*1000:.1f} ms/step  "
              f"{tps/ws:,.0f} tok/s/GPU  {mem:.1f} GB")

    del layers, opt; torch.cuda.empty_cache()
    return el / STEPS * 1000 if rank == 0 else 0.0


def bench_megatron(tag, num_layers, ws, rank):
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
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
        expert_model_parallel_size=ws,
    )
    model_parallel_cuda_manual_seed(42)

    mcfg = TransformerConfig(
        num_layers=num_layers, hidden_size=H, num_attention_heads=HEADS,
        ffn_hidden_size=FFN, num_moe_experts=E, moe_router_topk=K,
        moe_ffn_hidden_size=FFN, moe_router_load_balancing_type="aux_loss",
        moe_aux_loss_coeff=0.01, moe_router_score_function="sigmoid",
        moe_token_dispatcher_type="alltoall", moe_grouped_gemm=False,
        add_bias_linear=False, bias_activation_fusion=False,
        bf16=True, params_dtype=torch.bfloat16,
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
        expert_model_parallel_size=ws,
    )
    backend = LocalSpecProvider()
    mlp_sub = MLPSubmodules(linear_fc1=backend.column_parallel_linear(),
                            linear_fc2=backend.row_parallel_linear())
    submodules = MoESubmodules(experts=ModuleSpec(module=SequentialMLP, submodules=mlp_sub))

    layers = torch.nn.ModuleList([
        MoELayer(mcfg, submodules=submodules, layer_number=i)
        for i in range(num_layers)
    ]).to(torch.bfloat16).cuda()

    tp = sum(p.numel() for p in layers.parameters())
    opt = torch.optim.AdamW(layers.parameters(), lr=1e-4)
    T = BS * SL

    if rank == 0:
        print(f"  [Megatron]  {num_layers}L MoE-only  Params/GPU: {tp/1e9:.2f}B")

    for _ in range(WARMUP):
        x = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
        for layer in layers:
            x, _ = layer(x)
        x.sum().backward()
        opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); dist.barrier()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for s in range(STEPS):
        x = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
        for layer in layers:
            x, _ = layer(x)
        x.sum().backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        if rank == 0:
            print(f"    step {s}")
    torch.cuda.synchronize(); dist.barrier()
    el = time.perf_counter() - t0

    if rank == 0:
        tps = T * ws * STEPS / el
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Megatron  {tag}: {el/STEPS*1000:.1f} ms/step  "
              f"{tps/ws:,.0f} tok/s/GPU  {mem:.1f} GB")

    del layers, opt; torch.cuda.empty_cache()
    parallel_state.destroy_model_parallel()
    return el / STEPS * 1000 if rank == 0 else 0.0


def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    rank, ws = dist.get_rank(), dist.get_world_size()
    ep_group = dist.new_group(list(range(ws)))

    if rank == 0:
        print("=" * 65)
        print(f"  DSv3-like Pure MoE Benchmark  |  {ws}x {torch.cuda.get_device_name(0)}")
        print(f"  H={H} E={E} K={K} FFN={FFN}  BS={BS} SL={SL}")
        print("=" * 65)

    results = {}
    for NL in [1, 2, 4]:
        if rank == 0:
            print(f"\n{'='*65}")
            print(f"  {NL} LAYER(S)")
            print(f"{'='*65}")

        s_ms = bench_simple_moe(f"{NL}L", NL, ws, rank, ep_group)
        dist.barrier(); torch.cuda.empty_cache()

        try:
            m_ms = bench_megatron(f"{NL}L", NL, ws, rank)
        except Exception:
            if rank == 0: traceback.print_exc()
            m_ms = 0.0
        dist.barrier(); torch.cuda.empty_cache()

        if rank == 0:
            results[NL] = (s_ms, m_ms)

    if rank == 0:
        print(f"\n{'='*65}")
        print(f"  SUMMARY: Pure MoE Layer Comparison (DSv3-like)")
        print(f"{'='*65}")
        print(f"  {'Layers':<8} {'SimpleMoE':>12} {'Megatron':>12} {'Ratio':>10}")
        print(f"  {'-'*42}")
        for nl, (s, m) in results.items():
            ratio = f"{m/s:.2f}x" if s > 0 and m > 0 else "N/A"
            winner = "S" if s < m else "M"
            print(f"  {nl}L       {s:>9.1f}ms  {m:>9.1f}ms  {ratio:>8} ← {winner}")
        print(f"{'='*65}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
