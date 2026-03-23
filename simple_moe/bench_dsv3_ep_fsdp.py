"""
DSv3-like benchmark with EP=4 + FSDP=2 on 8 GPUs.

This tests the actual FSDP + EP combination:
  - 8 GPUs split into 2 FSDP replicas × 4 EP ranks
  - GPUs {0,1,2,3} = EP group 0 (replica 0)
  - GPUs {4,5,6,7} = EP group 1 (replica 1)
  - FSDP groups: {0,4}, {1,5}, {2,6}, {3,7}
  - After backward: AllReduce gradients across FSDP groups

Compared with pure EP=8 to show FSDP overhead and scaling.

Usage: torchrun --nproc_per_node=8 simple_moe/bench_dsv3_ep_fsdp.py
"""
from __future__ import annotations
import os, time, torch, torch.distributed as dist

from simple_moe.config import MoEModelConfig
from simple_moe.moe_layer import MoELayer


H, E_BASE, K, FFN, HEADS = 7168, 64, 8, 2048, 56
BS, SL = 2, 4096
WARMUP, STEPS = 3, 10
NL = 2


def make_all_groups(ws):
    """
    Create ALL process groups upfront. Every rank must participate in
    every new_group call, even if it's not a member.
    """
    assert ws == 8
    rank = dist.get_rank()

    # EP=8 group (all ranks)
    ep8_group = dist.new_group(list(range(ws)))

    # EP=4 groups: {0,1,2,3} and {4,5,6,7}
    my_ep4 = None
    for start in range(0, ws, 4):
        ranks = list(range(start, start + 4))
        g = dist.new_group(ranks)
        if rank in ranks:
            my_ep4 = g

    # FSDP=2 groups: {0,4}, {1,5}, {2,6}, {3,7}
    my_fsdp2 = None
    for base in range(4):
        ranks = [base, base + 4]
        g = dist.new_group(ranks)
        if rank in ranks:
            my_fsdp2 = g

    # EP=2 groups: {0,1}, {2,3}, {4,5}, {6,7}
    my_ep2 = None
    for start in range(0, ws, 2):
        ranks = [start, start + 1]
        g = dist.new_group(ranks)
        if rank in ranks:
            my_ep2 = g

    # FSDP=4 groups: {0,2,4,6}, {1,3,5,7}
    my_fsdp4 = None
    for base in range(2):
        ranks = [base + i * 2 for i in range(4)]
        g = dist.new_group(ranks)
        if rank in ranks:
            my_fsdp4 = g

    return {
        'ep8': ep8_group,
        'ep4': my_ep4, 'fsdp2': my_fsdp2,
        'ep2': my_ep2, 'fsdp4': my_fsdp4,
    }


def fsdp_allreduce_grads(model, fsdp_group):
    """AllReduce all gradients across the FSDP group."""
    fsdp_size = dist.get_world_size(fsdp_group)
    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=fsdp_group)
            p.grad.div_(fsdp_size)


def run_bench(tag, layers, optimizer, fsdp_group, num_experts, ws, rank):
    T = BS * SL
    tp = sum(p.numel() for p in layers.parameters())

    if rank == 0:
        print(f"\n  [{tag}] {NL}L MoE  E={num_experts}  Params/GPU: {tp/1e9:.2f}B")
        print(f"    Warming up...")

    for _ in range(WARMUP):
        x = torch.randn(T, H, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        for layer in layers:
            x, _ = layer(x)
        x.float().sum().backward()
        if fsdp_group is not None:
            fsdp_allreduce_grads(layers, fsdp_group)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); dist.barrier()

    if rank == 0:
        print(f"    Benchmarking ({STEPS} steps)...")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for s in range(STEPS):
        x = torch.randn(T, H, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        for layer in layers:
            x, _ = layer(x)
        x.float().sum().backward()
        if fsdp_group is not None:
            fsdp_allreduce_grads(layers, fsdp_group)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if rank == 0:
            print(f"      step {s}")

    torch.cuda.synchronize(); dist.barrier()
    el = time.perf_counter() - t0

    if rank == 0:
        tps = T * ws * STEPS / el
        mem = torch.cuda.max_memory_allocated() / 1e9
        ms = el / STEPS * 1000
        print(f"    {tag}: {ms:.1f} ms/step  {tps/ws:,.0f} tok/s/GPU  {mem:.1f} GB")
        return ms
    return 0.0


def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    rank, ws = dist.get_rank(), dist.get_world_size()
    assert ws == 8

    if rank == 0:
        print("=" * 65)
        print(f"  DSv3-like EP+FSDP Benchmark  |  {ws}x {torch.cuda.get_device_name(0)}")
        print(f"  H={H} K={K} FFN={FFN}  BS={BS} SL={SL}  Layers={NL}")
        print("=" * 65)

    groups = make_all_groups(ws)

    # ===== Config 1: EP=8, FSDP=1 (baseline) =====
    if rank == 0:
        print(f"\n{'='*65}")
        print(f"  Config A: EP=8, FSDP=1 (baseline, no grad sync)")
        print(f"{'='*65}")
    num_experts_a = 256  # 32 per GPU
    cfg_a = MoEModelConfig(
        hidden_dim=H, num_experts=num_experts_a, top_k=K, expert_ffn_dim=FFN,
        score_func="sigmoid", load_balance="aux_loss", aux_loss_coeff=0.01,
    )
    layers_a = torch.nn.ModuleList([
        MoELayer(cfg_a, ep_group=groups['ep8'], edp_group=groups['ep8']) for _ in range(NL)
    ]).to(torch.bfloat16).cuda()
    opt_a = torch.optim.AdamW(layers_a.parameters(), lr=1e-4)
    ms_a = run_bench("EP=8 FSDP=1", layers_a, opt_a, None, num_experts_a, ws, rank)
    del layers_a, opt_a; torch.cuda.empty_cache(); dist.barrier()

    # ===== Config 2: EP=4, FSDP=2 =====
    if rank == 0:
        print(f"\n{'='*65}")
        print(f"  Config B: EP=4, FSDP=2 (grad AllReduce across 2 replicas)")
        print(f"{'='*65}")
    num_experts_b = 128  # EP=4, so 32 per GPU (same local load)
    cfg_b = MoEModelConfig(
        hidden_dim=H, num_experts=num_experts_b, top_k=K, expert_ffn_dim=FFN,
        score_func="sigmoid", load_balance="aux_loss", aux_loss_coeff=0.01,
    )
    layers_b = torch.nn.ModuleList([
        MoELayer(cfg_b, ep_group=groups['ep4'], edp_group=groups['fsdp2']) for _ in range(NL)
    ]).to(torch.bfloat16).cuda()
    opt_b = torch.optim.AdamW(layers_b.parameters(), lr=1e-4)
    ms_b = run_bench("EP=4 FSDP=2", layers_b, opt_b, groups['fsdp2'], num_experts_b, ws, rank)
    del layers_b, opt_b; torch.cuda.empty_cache(); dist.barrier()

    # ===== Config 3: EP=2, FSDP=4 =====
    if rank == 0:
        print(f"\n{'='*65}")
        print(f"  Config C: EP=2, FSDP=4 (grad AllReduce across 4 replicas)")
        print(f"{'='*65}")
    num_experts_c = 64  # EP=2, 32 per GPU
    cfg_c = MoEModelConfig(
        hidden_dim=H, num_experts=num_experts_c, top_k=K, expert_ffn_dim=FFN,
        score_func="sigmoid", load_balance="aux_loss", aux_loss_coeff=0.01,
    )
    layers_c = torch.nn.ModuleList([
        MoELayer(cfg_c, ep_group=groups['ep2'], edp_group=groups['fsdp4']) for _ in range(NL)
    ]).to(torch.bfloat16).cuda()
    opt_c = torch.optim.AdamW(layers_c.parameters(), lr=1e-4)
    ms_c = run_bench("EP=2 FSDP=4", layers_c, opt_c, groups['fsdp4'], num_experts_c, ws, rank)
    del layers_c, opt_c; torch.cuda.empty_cache(); dist.barrier()

    # ===== Summary =====
    if rank == 0:
        print(f"\n{'='*65}")
        print(f"  SUMMARY: EP vs FSDP trade-off (DSv3-like, {NL}L)")
        print(f"{'='*65}")
        print(f"  {'Config':<20} {'EP':>4} {'FSDP':>5} {'E_total':>8} {'ms/step':>10}")
        print(f"  {'-'*50}")
        print(f"  {'A: baseline':<20} {'8':>4} {'1':>5} {'256':>8} {ms_a:>8.1f}ms")
        print(f"  {'B: EP4+FSDP2':<20} {'4':>4} {'2':>5} {'128':>8} {ms_b:>8.1f}ms")
        print(f"  {'C: EP2+FSDP4':<20} {'2':>4} {'4':>5} {'64':>8} {ms_c:>8.1f}ms")
        print(f"\n  FSDP overhead (B vs A): {(ms_b/ms_a - 1)*100:+.1f}%")
        print(f"  FSDP overhead (C vs A): {(ms_c/ms_a - 1)*100:+.1f}%")
        print(f"  (B,C process 2x/4x more data per expert → better convergence)")
        print(f"{'='*65}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
