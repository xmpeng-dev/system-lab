"""
Larger-config benchmark for 8x MI355X.
Usage: torchrun --nproc_per_node=8 simple_moe/bench_large.py
"""
from __future__ import annotations
import os, time, torch, torch.distributed as dist

def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    ws = dist.get_world_size()

    from simple_moe.config import MoEModelConfig
    from simple_moe.model import MoETransformerLM

    cfg = MoEModelConfig(
        vocab_size=32000, hidden_dim=4096, num_layers=12, num_heads=32,
        head_dim=128, num_moe_layers=10, num_experts=ws * 4, top_k=2,
        expert_ffn_dim=2048, dense_ffn_dim=8192, score_func="sigmoid",
        num_shared_experts=0, load_balance="aux_loss", aux_loss_coeff=0.01,
        max_seq_len=1024,
    )
    BS, SL = 4, 1024

    ep = dist.new_group(list(range(ws)))
    model = MoETransformerLM(cfg, ep_group=ep, edp_group=ep).to(torch.bfloat16).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    tp = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"Params/GPU: {tp/1e6:.1f}M | H={cfg.hidden_dim} L={cfg.num_layers} "
              f"E={cfg.num_experts} K={cfg.top_k} BS={BS} SL={SL}")

    # Warmup
    for _ in range(3):
        ids = torch.randint(0, cfg.vocab_size, (BS, SL), device="cuda")
        loss, _ = model(ids, labels=ids)
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); dist.barrier()

    # Benchmark
    N = 10
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for s in range(N):
        ids = torch.randint(0, cfg.vocab_size, (BS, SL), device="cuda")
        loss, aux = model(ids, labels=ids)
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        if rank == 0:
            print(f"  step {s} loss={loss.item():.4f}")
    torch.cuda.synchronize(); dist.barrier()
    el = time.perf_counter() - t0

    if rank == 0:
        tps = BS * SL * ws * N / el
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"\n{'='*50}")
        print(f"  ms/step        : {el/N*1000:.1f}")
        print(f"  Tokens/s total : {tps:,.0f}")
        print(f"  Tokens/s/GPU   : {tps/ws:,.0f}")
        print(f"  Peak mem/GPU   : {mem:.2f} GB")
        print(f"{'='*50}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
