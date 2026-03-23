"""
EP=4 head-to-head: SimpleMoE vs Megatron-Core.

Both use EP=4 with FSDP=2 (gradient sync). Same expert count, same tokens.
Tests pure MoE layers for fair comparison.

Usage: torchrun --nproc_per_node=8 simple_moe/bench_ep4_compare.py
"""
from __future__ import annotations
import os, sys, time, traceback, torch, torch.distributed as dist

H, K, FFN = 7168, 8, 2048
BS, SL = 2, 4096
WARMUP, STEPS = 3, 10
MEGATRON_PATH = "/home/xiaompen/Megatron-LM-v13.0"


def make_groups(ws):
    rank = dist.get_rank()
    # EP=4 groups: {0,1,2,3}, {4,5,6,7}
    my_ep4 = None
    for s in range(0, ws, 4):
        g = dist.new_group(list(range(s, s + 4)))
        if s <= rank < s + 4:
            my_ep4 = g
    # FSDP=2 groups: {0,4}, {1,5}, {2,6}, {3,7}
    my_fsdp2 = None
    for b in range(4):
        g = dist.new_group([b, b + 4])
        if rank in (b, b + 4):
            my_fsdp2 = g
    # EP=8 (baseline)
    ep8 = dist.new_group(list(range(ws)))
    return ep8, my_ep4, my_fsdp2


def fsdp_sync(model, group):
    ws = dist.get_world_size(group)
    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, group=group)
            p.grad.div_(ws)


def bench(tag, layers, opt, fsdp_group, T, ws, rank):
    for _ in range(WARMUP):
        x = torch.randn(T, H, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        for layer in layers:
            out = layer(x)
            if isinstance(out, tuple):
                x = out[0]
            else:
                x = out
        x.float().sum().backward()
        if fsdp_group:
            fsdp_sync(layers, fsdp_group)
        opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); dist.barrier()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for s in range(STEPS):
        x = torch.randn(T, H, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        for layer in layers:
            out = layer(x)
            if isinstance(out, tuple):
                x = out[0]
            else:
                x = out
        x.float().sum().backward()
        if fsdp_group:
            fsdp_sync(layers, fsdp_group)
        opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); dist.barrier()
    el = time.perf_counter() - t0

    if rank == 0:
        ms = el / STEPS * 1000
        tps = T * ws * STEPS / el
        mem = torch.cuda.max_memory_allocated() / 1e9
        tp = sum(p.numel() for p in layers.parameters())
        print(f"  {tag}")
        print(f"    Params/GPU : {tp/1e9:.2f}B")
        print(f"    ms/step    : {ms:.1f}")
        print(f"    tok/s/GPU  : {tps/ws:,.0f}")
        print(f"    Peak mem   : {mem:.1f} GB")
        return ms
    return 0.0


def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    rank, ws = dist.get_rank(), dist.get_world_size()
    T = BS * SL

    ep8, ep4, fsdp2 = make_groups(ws)

    if rank == 0:
        print("=" * 65)
        print(f"  EP=4 Head-to-Head: SimpleMoE vs Megatron-Core")
        print(f"  {ws}x {torch.cuda.get_device_name(0)}")
        print(f"  H={H} K={K} FFN={FFN} BS={BS} SL={SL}")
        print("=" * 65)

    results = {}

    for NL in [1, 2, 4]:
        E = 128  # EP=4 → 32 experts/GPU
        if rank == 0:
            print(f"\n{'='*65}")
            print(f"  {NL} LAYER(S)  |  E={E} EP=4 FSDP=2")
            print(f"{'='*65}")

        # ---- SimpleMoE EP=4 + FSDP=2 ----
        from simple_moe.config import MoEModelConfig
        from simple_moe.moe_layer import MoELayer
        cfg = MoEModelConfig(
            hidden_dim=H, num_experts=E, top_k=K, expert_ffn_dim=FFN,
            score_func="sigmoid", load_balance="aux_loss", aux_loss_coeff=0.01,
        )
        s_layers = torch.nn.ModuleList([
            MoELayer(cfg, ep_group=ep4, edp_group=fsdp2) for _ in range(NL)
        ]).to(torch.bfloat16).cuda()
        s_opt = torch.optim.AdamW(s_layers.parameters(), lr=1e-4)
        s_ms = bench(f"SimpleMoE EP=4 FSDP=2 {NL}L", s_layers, s_opt, fsdp2, T, ws, rank)
        del s_layers, s_opt; torch.cuda.empty_cache(); dist.barrier()

        # ---- SimpleMoE EP=8 (baseline) ----
        cfg8 = MoEModelConfig(
            hidden_dim=H, num_experts=256, top_k=K, expert_ffn_dim=FFN,
            score_func="sigmoid", load_balance="aux_loss", aux_loss_coeff=0.01,
        )
        s8_layers = torch.nn.ModuleList([
            MoELayer(cfg8, ep_group=ep8, edp_group=ep8) for _ in range(NL)
        ]).to(torch.bfloat16).cuda()
        s8_opt = torch.optim.AdamW(s8_layers.parameters(), lr=1e-4)
        s8_ms = bench(f"SimpleMoE EP=8 (baseline) {NL}L", s8_layers, s8_opt, None, T, ws, rank)
        del s8_layers, s8_opt; torch.cuda.empty_cache(); dist.barrier()

        # ---- Megatron EP=4 + EDP=2 ----
        if MEGATRON_PATH not in sys.path:
            sys.path.insert(0, MEGATRON_PATH)
        from megatron.core import parallel_state
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
        from megatron.core.transformer.transformer_config import TransformerConfig
        from megatron.core.transformer.moe.moe_layer import MoELayer as MegMoE, MoESubmodules
        from megatron.core.transformer.moe.experts import SequentialMLP
        from megatron.core.transformer.mlp import MLPSubmodules
        from megatron.core.transformer.spec_utils import ModuleSpec
        from megatron.core.models.gpt.gpt_layer_specs import LocalSpecProvider

        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
            expert_model_parallel_size=4,
        )
        model_parallel_cuda_manual_seed(42)

        mcfg = TransformerConfig(
            num_layers=NL, hidden_size=H, num_attention_heads=56,
            ffn_hidden_size=FFN, num_moe_experts=E, moe_router_topk=K,
            moe_ffn_hidden_size=FFN, moe_router_load_balancing_type="aux_loss",
            moe_aux_loss_coeff=0.01, moe_router_score_function="sigmoid",
            moe_token_dispatcher_type="alltoall", moe_grouped_gemm=False,
            add_bias_linear=False, bias_activation_fusion=False,
            bf16=True, params_dtype=torch.bfloat16,
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
            expert_model_parallel_size=4,
        )
        backend = LocalSpecProvider()
        mlp_sub = MLPSubmodules(linear_fc1=backend.column_parallel_linear(),
                                linear_fc2=backend.row_parallel_linear())
        submod = MoESubmodules(experts=ModuleSpec(module=SequentialMLP, submodules=mlp_sub))
        m_layers = torch.nn.ModuleList([
            MegMoE(mcfg, submodules=submod, layer_number=i) for i in range(NL)
        ]).to(torch.bfloat16).cuda()
        m_opt = torch.optim.AdamW(m_layers.parameters(), lr=1e-4)
        m_ms = bench(f"Megatron EP=4 EDP=2 {NL}L", m_layers, m_opt, None, T, ws, rank)
        del m_layers, m_opt; torch.cuda.empty_cache()

        # Megatron EP=8 baseline
        parallel_state.destroy_model_parallel()
        dist.barrier()
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
            expert_model_parallel_size=8,
        )
        model_parallel_cuda_manual_seed(42)
        mcfg8 = TransformerConfig(
            num_layers=NL, hidden_size=H, num_attention_heads=56,
            ffn_hidden_size=FFN, num_moe_experts=256, moe_router_topk=K,
            moe_ffn_hidden_size=FFN, moe_router_load_balancing_type="aux_loss",
            moe_aux_loss_coeff=0.01, moe_router_score_function="sigmoid",
            moe_token_dispatcher_type="alltoall", moe_grouped_gemm=False,
            add_bias_linear=False, bias_activation_fusion=False,
            bf16=True, params_dtype=torch.bfloat16,
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
            expert_model_parallel_size=8,
        )
        m8_layers = torch.nn.ModuleList([
            MegMoE(mcfg8, submodules=submod, layer_number=i) for i in range(NL)
        ]).to(torch.bfloat16).cuda()
        m8_opt = torch.optim.AdamW(m8_layers.parameters(), lr=1e-4)
        m8_ms = bench(f"Megatron EP=8 (baseline) {NL}L", m8_layers, m8_opt, None, T, ws, rank)
        del m8_layers, m8_opt; torch.cuda.empty_cache()
        parallel_state.destroy_model_parallel()
        dist.barrier()

        if rank == 0:
            results[NL] = (s_ms, s8_ms, m_ms, m8_ms)

    if rank == 0:
        print(f"\n{'='*75}")
        print(f"  SUMMARY")
        print(f"{'='*75}")
        print(f"  {'L':>2}  {'S-EP4':>10} {'S-EP8':>10} {'M-EP4':>10} {'M-EP8':>10}  {'S4/M4':>7} {'S8/M8':>7}")
        print(f"  {'-'*68}")
        for nl, (s4, s8, m4, m8) in results.items():
            r4 = f"{m4/s4:.2f}x" if s4 > 0 and m4 > 0 else "N/A"
            r8 = f"{m8/s8:.2f}x" if s8 > 0 and m8 > 0 else "N/A"
            print(f"  {nl:>2}  {s4:>8.1f}ms {s8:>8.1f}ms {m4:>8.1f}ms {m8:>8.1f}ms  {r4:>7} {r8:>7}")
        print(f"\n  S4/M4 > 1 = SimpleMoE EP=4 faster than Megatron EP=4")
        print(f"  S8/M8 > 1 = SimpleMoE EP=8 faster than Megatron EP=8")
        print(f"{'='*75}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
