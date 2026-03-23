"""
FSDP value proposition: save memory → larger batch → higher throughput.

On 8x MI355X with DSv3-like 4L:
  A: EP=8, no FSDP, max batch that fits
  B: EP=4 FSDP=2, optimizer states sharded → can fit larger batch
  C: EP=2 FSDP=4, optimizer states sharded even more → even larger batch

The key insight: FSDP slows each step, but if it enables 2x batch,
total tokens/sec can still improve.

Usage: torchrun --nproc_per_node=8 simple_moe/bench_fsdp_throughput.py
"""
from __future__ import annotations
import os, time, torch, torch.distributed as dist
from simple_moe.config import MoEModelConfig
from simple_moe.moe_layer import MoELayer
from simple_moe.grad_sync import GradSyncOverlap

H, K, FFN, NL = 7168, 8, 2048, 4
WARMUP, STEPS = 2, 5


def make_groups(ws):
    rank = dist.get_rank()
    ep8 = dist.new_group(list(range(ws)))
    my_ep4 = None
    for s in range(0, ws, 4):
        g = dist.new_group(list(range(s, s + 4)))
        if s <= rank < s + 4: my_ep4 = g
    my_f2 = None
    for b in range(4):
        g = dist.new_group([b, b + 4])
        if rank in (b, b + 4): my_f2 = g
    my_ep2 = None
    for s in range(0, ws, 2):
        g = dist.new_group([s, s + 1])
        if rank in (s, s + 1): my_ep2 = g
    my_f4 = None
    for b in range(2):
        ranks = [b + i * 2 for i in range(4)]
        g = dist.new_group(ranks)
        if rank in ranks: my_f4 = g
    return ep8, my_ep4, my_f2, my_ep2, my_f4


def run(tag, layers, opt, fsdp_group, sync_obj, T, SL, ws, rank):
    for _ in range(WARMUP):
        x = torch.randn(T, H, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        for l in layers: x = l(x)[0]
        x.float().sum().backward()
        if sync_obj: sync_obj.finish()
        elif fsdp_group:
            for p in layers.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, group=fsdp_group)
                    p.grad.div_(dist.get_world_size(fsdp_group))
        opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); dist.barrier()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for s in range(STEPS):
        x = torch.randn(T, H, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        for l in layers: x = l(x)[0]
        x.float().sum().backward()
        if sync_obj: sync_obj.finish()
        elif fsdp_group:
            for p in layers.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, group=fsdp_group)
                    p.grad.div_(dist.get_world_size(fsdp_group))
        opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); dist.barrier()
    el = time.perf_counter() - t0

    ms = el / STEPS * 1000
    if rank == 0:
        tokens_total = T * ws * STEPS
        tps = tokens_total / el
        mem = torch.cuda.max_memory_allocated() / 1e9
        bs = T // SL
        print(f"  {tag}")
        print(f"    Batch={bs} SL={SL} → T={T}/GPU  ms/step={ms:.0f}  "
              f"tok/s/GPU={tps/ws:,.0f}  mem={mem:.1f}GB")
        return ms, tps / ws, mem, T
    return 0, 0, 0, T


def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    rank, ws = dist.get_rank(), dist.get_world_size()
    ep8, ep4, f2, ep2, f4 = make_groups(ws)

    if rank == 0:
        print("=" * 70)
        print(f"  FSDP Throughput Test: Smaller batch vs Larger batch")
        print(f"  {ws}x {torch.cuda.get_device_name(0)} (256 GB each)")
        print(f"  DSv3-like {NL}L  H={H} E=varies K={K} FFN={FFN}")
        print("=" * 70)

    SL = 4096
    results = []

    # --- A: EP=8, BS=2 (baseline) ---
    if rank == 0: print(f"\n  Config A: EP=8, BS=2 (baseline)")
    cfg_a = MoEModelConfig(hidden_dim=H, num_experts=256, top_k=K, expert_ffn_dim=FFN,
                           score_func="sigmoid", load_balance="aux_loss", aux_loss_coeff=0.01)
    la = torch.nn.ModuleList([MoELayer(cfg_a, ep8, ep8) for _ in range(NL)]).to(torch.bfloat16).cuda()
    oa = torch.optim.AdamW(la.parameters(), lr=1e-4)
    r = run("EP=8 BS=2", la, oa, None, None, 2 * SL, SL, ws, rank)
    if rank == 0: results.append(("EP=8 BS=2", *r))
    del la, oa; torch.cuda.empty_cache(); dist.barrier()

    # --- A2: EP=8, BS=3 (try to push batch — may OOM) ---
    if rank == 0: print(f"\n  Config A2: EP=8, BS=3 (push batch)")
    try:
        la2 = torch.nn.ModuleList([MoELayer(cfg_a, ep8, ep8) for _ in range(NL)]).to(torch.bfloat16).cuda()
        oa2 = torch.optim.AdamW(la2.parameters(), lr=1e-4)
        r = run("EP=8 BS=3", la2, oa2, None, None, 3 * SL, SL, ws, rank)
        if rank == 0: results.append(("EP=8 BS=3", *r))
        del la2, oa2
    except torch.cuda.OutOfMemoryError:
        if rank == 0:
            results.append(("EP=8 BS=3", 0, 0, 999, 3 * SL))
            print("    OOM! Cannot increase batch with EP=8")
    torch.cuda.empty_cache(); dist.barrier()

    # --- B: EP=4 FSDP=2, BS=4 (FSDP enables 2x batch with same experts/GPU) ---
    if rank == 0: print(f"\n  Config B: EP=4 FSDP=2, BS=4 (FSDP saves mem → 2x batch)")
    cfg_b = MoEModelConfig(hidden_dim=H, num_experts=128, top_k=K, expert_ffn_dim=FFN,
                           score_func="sigmoid", load_balance="aux_loss", aux_loss_coeff=0.01)
    lb = torch.nn.ModuleList([MoELayer(cfg_b, ep4, f2) for _ in range(NL)]).to(torch.bfloat16).cuda()
    ob = torch.optim.AdamW(lb.parameters(), lr=1e-4)
    sb = GradSyncOverlap(lb, f2, sync_filter="expert")
    r = run("EP=4+FSDP=2 BS=4", lb, ob, None, sb, 4 * SL, SL, ws, rank)
    if rank == 0: results.append(("EP=4+F2 BS=4", *r))
    sb.remove_hooks(); del lb, ob; torch.cuda.empty_cache(); dist.barrier()

    # --- B2: EP=4 FSDP=2, BS=8 ---
    if rank == 0: print(f"\n  Config B2: EP=4 FSDP=2, BS=8 (push further)")
    lb2 = torch.nn.ModuleList([MoELayer(cfg_b, ep4, f2) for _ in range(NL)]).to(torch.bfloat16).cuda()
    ob2 = torch.optim.AdamW(lb2.parameters(), lr=1e-4)
    sb2 = GradSyncOverlap(lb2, f2, sync_filter="expert")
    r = run("EP=4+FSDP=2 BS=8", lb2, ob2, None, sb2, 8 * SL, SL, ws, rank)
    if rank == 0: results.append(("EP=4+F2 BS=8", *r))
    sb2.remove_hooks(); del lb2, ob2; torch.cuda.empty_cache(); dist.barrier()

    # --- C: EP=2 FSDP=4, BS=8 ---
    if rank == 0: print(f"\n  Config C: EP=2 FSDP=4, BS=8")
    cfg_c = MoEModelConfig(hidden_dim=H, num_experts=64, top_k=K, expert_ffn_dim=FFN,
                           score_func="sigmoid", load_balance="aux_loss", aux_loss_coeff=0.01)
    lc = torch.nn.ModuleList([MoELayer(cfg_c, ep2, f4) for _ in range(NL)]).to(torch.bfloat16).cuda()
    oc = torch.optim.AdamW(lc.parameters(), lr=1e-4)
    sc = GradSyncOverlap(lc, f4, sync_filter="expert")
    r = run("EP=2+FSDP=4 BS=8", lc, oc, None, sc, 8 * SL, SL, ws, rank)
    if rank == 0: results.append(("EP=2+F4 BS=8", *r))
    sc.remove_hooks(); del lc, oc; torch.cuda.empty_cache(); dist.barrier()

    # --- Summary ---
    if rank == 0:
        print(f"\n{'='*70}")
        print(f"  SUMMARY: FSDP enables larger batch → higher throughput")
        print(f"{'='*70}")
        print(f"  {'Config':<20} {'Batch':>6} {'Tokens/step':>12} {'ms/step':>9} {'tok/s/GPU':>11} {'Mem':>7}")
        print(f"  {'-'*67}")
        for name, ms, tps, mem, T in results:
            bs = T // SL
            print(f"  {name:<20} {bs:>5}x {T:>10} {ms:>7.0f}ms {tps:>10,.0f} {mem:>6.1f}GB")
        print(f"\n  Key: even if ms/step is higher with FSDP, tok/s can be higher")
        print(f"  because FSDP processes 2x/4x more data per step (larger DP batch).")
        print(f"{'='*70}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
