"""
EP=4 + FSDP=2 benchmark: blocking vs overlapped gradient sync.

Compares three modes:
  A: SimpleMoE EP=4, blocking grad AllReduce (current)
  B: SimpleMoE EP=4, overlapped grad AllReduce (new)
  C: Megatron EP=4, EDP=2 (reference)

Usage: torchrun --nproc_per_node=8 simple_moe/bench_ep4_overlap.py
"""
from __future__ import annotations
import os, sys, time, torch, torch.distributed as dist

from simple_moe.config import MoEModelConfig
from simple_moe.moe_layer import MoELayer
from simple_moe.grad_sync import GradSyncOverlap

H, K, FFN = 7168, 8, 2048
BS, SL, E = 2, 4096, 128
WARMUP, STEPS = 3, 10
MEGATRON_PATH = "/home/xiaompen/Megatron-LM-v13.0"


def make_groups(ws):
    rank = dist.get_rank()
    ep8 = dist.new_group(list(range(ws)))
    my_ep4 = None
    for s in range(0, ws, 4):
        g = dist.new_group(list(range(s, s + 4)))
        if s <= rank < s + 4:
            my_ep4 = g
    my_f2 = None
    for b in range(4):
        g = dist.new_group([b, b + 4])
        if rank in (b, b + 4):
            my_f2 = g
    return ep8, my_ep4, my_f2


def blocking_sync(model, group):
    ws = dist.get_world_size(group)
    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, group=group)
            p.grad.div_(ws)


def run_bench(tag, layers, opt, grad_mode, fsdp_group, T, ws, rank, grad_sync_obj=None):
    for _ in range(WARMUP):
        x = torch.randn(T, H, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        for layer in layers:
            out = layer(x)
            x = out[0] if isinstance(out, tuple) else out
        x.float().sum().backward()
        if grad_mode == "blocking":
            blocking_sync(layers, fsdp_group)
        elif grad_mode == "overlap" and grad_sync_obj:
            grad_sync_obj.finish()
        opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); dist.barrier()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for s in range(STEPS):
        x = torch.randn(T, H, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        for layer in layers:
            out = layer(x)
            x = out[0] if isinstance(out, tuple) else out
        x.float().sum().backward()
        if grad_mode == "blocking":
            blocking_sync(layers, fsdp_group)
        elif grad_mode == "overlap" and grad_sync_obj:
            grad_sync_obj.finish()
        opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); dist.barrier()
    el = time.perf_counter() - t0
    ms = el / STEPS * 1000
    if rank == 0:
        mem = torch.cuda.max_memory_allocated() / 1e9
        tps = T * ws * STEPS / el
        print(f"  {tag}: {ms:.1f} ms/step  {tps/ws:,.0f} tok/s/GPU  {mem:.1f}GB")
    return ms if rank == 0 else 0.0


def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    rank, ws = dist.get_rank(), dist.get_world_size()
    T = BS * SL
    ep8, ep4, fsdp2 = make_groups(ws)

    if rank == 0:
        print("=" * 65)
        print(f"  Blocking vs Overlapped Grad Sync (EP=4 FSDP=2)")
        print(f"  {ws}x {torch.cuda.get_device_name(0)}")
        print(f"  H={H} E={E} K={K} FFN={FFN} BS={BS} SL={SL}")
        print("=" * 65)

    results = {}
    for NL in [1, 2, 4]:
        if rank == 0:
            print(f"\n{'='*65}")
            print(f"  {NL} LAYER(S)")
            print(f"{'='*65}")

        cfg = MoEModelConfig(
            hidden_dim=H, num_experts=E, top_k=K, expert_ffn_dim=FFN,
            score_func="sigmoid", load_balance="aux_loss", aux_loss_coeff=0.01,
        )

        # ---- A: Blocking ----
        layers_a = torch.nn.ModuleList([
            MoELayer(cfg, ep_group=ep4, edp_group=fsdp2) for _ in range(NL)
        ]).to(torch.bfloat16).cuda()
        opt_a = torch.optim.AdamW(layers_a.parameters(), lr=1e-4)
        ms_a = run_bench(f"Blocking  {NL}L", layers_a, opt_a, "blocking", fsdp2, T, ws, rank)
        del layers_a, opt_a; torch.cuda.empty_cache(); dist.barrier()

        # ---- B: Overlapped ----
        layers_b = torch.nn.ModuleList([
            MoELayer(cfg, ep_group=ep4, edp_group=fsdp2) for _ in range(NL)
        ]).to(torch.bfloat16).cuda()
        opt_b = torch.optim.AdamW(layers_b.parameters(), lr=1e-4)
        sync = GradSyncOverlap(layers_b, fsdp2, sync_filter="expert")
        ms_b = run_bench(f"Overlap   {NL}L", layers_b, opt_b, "overlap", fsdp2, T, ws, rank, sync)
        sync.remove_hooks()
        del layers_b, opt_b; torch.cuda.empty_cache(); dist.barrier()

        # ---- C: Megatron EP=4 ----
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
        layers_c = torch.nn.ModuleList([
            MegMoE(mcfg, submodules=submod, layer_number=i) for i in range(NL)
        ]).to(torch.bfloat16).cuda()
        opt_c = torch.optim.AdamW(layers_c.parameters(), lr=1e-4)
        ms_c = run_bench(f"Megatron  {NL}L", layers_c, opt_c, "none", None, T, ws, rank)
        del layers_c, opt_c; torch.cuda.empty_cache()
        parallel_state.destroy_model_parallel(); dist.barrier()

        if rank == 0:
            results[NL] = (ms_a, ms_b, ms_c)

    if rank == 0:
        print(f"\n{'='*70}")
        print(f"  SUMMARY: EP=4 Grad Sync Comparison")
        print(f"{'='*70}")
        print(f"  {'L':>2}  {'Blocking':>10} {'Overlap':>10} {'Megatron':>10}  {'Speedup':>8} {'vs Mega':>8}")
        print(f"  {'-'*58}")
        for nl, (a, b, c) in results.items():
            speedup = f"{a/b:.2f}x" if b > 0 else "N/A"
            vs_mega = f"{c/b:.2f}x" if b > 0 and c > 0 else "N/A"
            print(f"  {nl:>2}  {a:>8.1f}ms {b:>8.1f}ms {c:>8.1f}ms  {speedup:>8} {vs_mega:>8}")
        print(f"\n  Speedup = Blocking / Overlap (> 1 = overlap wins)")
        print(f"  vs Mega = Megatron / Overlap (> 1 = SimpleMoE overlap wins)")
        print(f"{'='*70}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
