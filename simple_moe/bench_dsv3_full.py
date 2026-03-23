"""
DSv3-like full transformer benchmark: Attention(FSDP) + MoE(EP).

Tests three configs on 8 GPUs:
  A: EP=8, FSDP=1 — baseline, no parameter sharding
  B: EP=4, FSDP=2 — attention params sharded 2-way, MoE grad sync 2-way
  C: EP=2, FSDP=4 — attention params sharded 4-way, MoE grad sync 4-way

Attention layers use ShardedFSDP (All-Gather fwd / Reduce-Scatter bwd).
MoE layers use EP (All-to-All) + gradient AllReduce across FSDP replicas.

Usage: torchrun --nproc_per_node=8 simple_moe/bench_dsv3_full.py
"""
from __future__ import annotations
import os, time, torch, torch.distributed as dist, torch.nn as nn

from simple_moe.config import MoEModelConfig
from simple_moe.model import MoETransformerLM, TransformerBlock
from simple_moe.fsdp import ShardedFSDP

H, K, FFN = 7168, 8, 2048
BS, SL, NL = 2, 4096, 2
WARMUP, STEPS = 3, 8


def make_all_groups(ws):
    rank = dist.get_rank()
    groups = {}
    groups['ep8'] = dist.new_group(list(range(ws)))
    # EP=4 groups
    my_ep4 = None
    for s in range(0, ws, 4):
        g = dist.new_group(list(range(s, s + 4)))
        if s <= rank < s + 4:
            my_ep4 = g
    groups['ep4'] = my_ep4
    # FSDP=2 groups
    my_f2 = None
    for b in range(4):
        g = dist.new_group([b, b + 4])
        if rank in (b, b + 4):
            my_f2 = g
    groups['fsdp2'] = my_f2
    # EP=2 groups
    my_ep2 = None
    for s in range(0, ws, 2):
        g = dist.new_group([s, s + 1])
        if rank in (s, s + 1):
            my_ep2 = g
    groups['ep2'] = my_ep2
    # FSDP=4 groups
    my_f4 = None
    for b in range(2):
        ranks = [b + i * 2 for i in range(4)]
        g = dist.new_group(ranks)
        if rank in ranks:
            my_f4 = g
    groups['fsdp4'] = my_f4
    return groups


def build_model(num_experts, ep_group, edp_group, fsdp_group):
    """Build model, wrap attention with FSDP, keep MoE on EP."""
    cfg = MoEModelConfig(
        vocab_size=32000, hidden_dim=H, num_layers=NL, num_heads=56,
        head_dim=128, num_moe_layers=NL, num_experts=num_experts,
        top_k=K, expert_ffn_dim=FFN, dense_ffn_dim=18432,
        score_func="sigmoid", num_shared_experts=0,
        load_balance="aux_loss", aux_loss_coeff=0.01, max_seq_len=SL,
    )
    model = MoETransformerLM(cfg, ep_group=ep_group, edp_group=edp_group)
    model = model.to(torch.bfloat16).cuda()

    # Wrap attention sub-modules with ShardedFSDP
    if fsdp_group is not None and dist.get_world_size(fsdp_group) > 1:
        for layer in model.layers:
            if isinstance(layer, TransformerBlock):
                layer.attn = ShardedFSDP(layer.attn, fsdp_group)
                layer.norm1 = ShardedFSDP(layer.norm1, fsdp_group)
                layer.norm2 = ShardedFSDP(layer.norm2, fsdp_group)
        # Also shard embedding and output head
        model.embed_tokens = ShardedFSDP(model.embed_tokens, fsdp_group)
        model.output_head = ShardedFSDP(model.output_head, fsdp_group)
        model.norm_f = ShardedFSDP(model.norm_f, fsdp_group)

    return model, cfg


def fsdp_sync_expert_grads(model, fsdp_group):
    """AllReduce expert gradients across the FSDP group."""
    if fsdp_group is None:
        return
    ws = dist.get_world_size(fsdp_group)
    if ws <= 1:
        return
    for name, p in model.named_parameters():
        if p.grad is not None and ("expert" in name or "w_gate_up" in name or "w_down" in name or "router" in name):
            dist.all_reduce(p.grad, group=fsdp_group)
            p.grad.div_(ws)


def run_bench(tag, model, cfg, fsdp_group, ws, rank):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    tp = sum(p.numel() for p in model.parameters())

    if rank == 0:
        print(f"\n  [{tag}] Params/GPU: {tp/1e9:.2f}B (sharded)")

    for _ in range(WARMUP):
        ids = torch.randint(0, cfg.vocab_size, (BS, SL), device="cuda")
        loss, _ = model(ids, labels=ids)
        loss.backward()
        fsdp_sync_expert_grads(model, fsdp_group)
        opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); dist.barrier()

    if rank == 0:
        print(f"    Benchmarking ({STEPS} steps)...")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for s in range(STEPS):
        ids = torch.randint(0, cfg.vocab_size, (BS, SL), device="cuda")
        loss, aux = model(ids, labels=ids)
        loss.backward()
        fsdp_sync_expert_grads(model, fsdp_group)
        opt.step(); opt.zero_grad(set_to_none=True)
        if rank == 0:
            print(f"      step {s} loss={loss.item():.4f}")

    torch.cuda.synchronize(); dist.barrier()
    el = time.perf_counter() - t0

    ms = el / STEPS * 1000
    if rank == 0:
        tps = BS * SL * ws * STEPS / el
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"    {tag}: {ms:.1f} ms/step  {tps/ws:,.0f} tok/s/GPU  {mem:.1f} GB")

    del opt
    return ms if rank == 0 else 0.0


def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    rank, ws = dist.get_rank(), dist.get_world_size()
    assert ws == 8

    if rank == 0:
        print("=" * 65)
        print(f"  DSv3-like Full Transformer: Attention(FSDP) + MoE(EP)")
        print(f"  {ws}x {torch.cuda.get_device_name(0)}")
        print(f"  H={H} K={K} FFN={FFN} BS={BS} SL={SL} Layers={NL}")
        print("=" * 65)

    groups = make_all_groups(ws)

    # ===== A: EP=8, FSDP=1 (baseline) =====
    if rank == 0:
        print(f"\n{'='*65}\n  Config A: EP=8, no FSDP\n{'='*65}")
    model_a, cfg_a = build_model(256, groups['ep8'], groups['ep8'], None)
    ms_a = run_bench("EP=8", model_a, cfg_a, None, ws, rank)
    del model_a; torch.cuda.empty_cache(); dist.barrier()

    # ===== B: EP=4, FSDP=2 =====
    if rank == 0:
        print(f"\n{'='*65}\n  Config B: EP=4, FSDP=2 (Attn sharded 2-way)\n{'='*65}")
    model_b, cfg_b = build_model(128, groups['ep4'], groups['fsdp2'], groups['fsdp2'])
    ms_b = run_bench("EP=4+FSDP=2", model_b, cfg_b, groups['fsdp2'], ws, rank)
    del model_b; torch.cuda.empty_cache(); dist.barrier()

    # ===== C: EP=2, FSDP=4 =====
    if rank == 0:
        print(f"\n{'='*65}\n  Config C: EP=2, FSDP=4 (Attn sharded 4-way)\n{'='*65}")
    model_c, cfg_c = build_model(64, groups['ep2'], groups['fsdp4'], groups['fsdp4'])
    ms_c = run_bench("EP=2+FSDP=4", model_c, cfg_c, groups['fsdp4'], ws, rank)
    del model_c; torch.cuda.empty_cache(); dist.barrier()

    # ===== Summary =====
    if rank == 0:
        print(f"\n{'='*65}")
        print(f"  SUMMARY: Full Transformer (Attn+MoE) with FSDP sharding")
        print(f"{'='*65}")
        print(f"  {'Config':<22} {'EP':>3} {'FSDP':>5} {'E':>5} {'ms/step':>10} {'Attn sharding':>15}")
        print(f"  {'-'*62}")
        print(f"  {'A: baseline':<22} {'8':>3} {'1':>5} {'256':>5} {ms_a:>8.1f}ms {'none':>15}")
        print(f"  {'B: EP4+FSDP2':<22} {'4':>3} {'2':>5} {'128':>5} {ms_b:>8.1f}ms {'2-way shard':>15}")
        print(f"  {'C: EP2+FSDP4':<22} {'2':>3} {'4':>5} {'64':>5} {ms_c:>8.1f}ms {'4-way shard':>15}")
        print()
        if ms_a > 0:
            print(f"  FSDP=2 overhead: {(ms_b/ms_a-1)*100:+.1f}%")
            print(f"  FSDP=4 overhead: {(ms_c/ms_a-1)*100:+.1f}%")
            print(f"  FSDP=2 memory saving: Attn params stored at 1/2 → ~50% less dense param mem")
            print(f"  FSDP=4 memory saving: Attn params stored at 1/4 → ~75% less dense param mem")
        print(f"{'='*65}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
