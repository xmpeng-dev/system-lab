"""
Correctness verification for SimpleMoE.

Tests:
  1. Expert gradients: do expert parameters receive non-zero gradients?
  2. Standalone MoE layer: does the autograd chain survive All-to-All?
  3. Convergence: does loss decrease over training steps?
  4. Router validity: are routing outputs well-formed?
  5. Dispatch round-trip: does dispatch→combine preserve shape?
  6. Gradient magnitudes: are all gradients finite and non-zero?

Usage: torchrun --nproc_per_node=8 simple_moe/test_correctness.py
"""
from __future__ import annotations

import os
import torch
import torch.distributed as dist
from simple_moe.config import MoEModelConfig
from simple_moe.moe_layer import MoELayer
from simple_moe.model import MoETransformerLM
from simple_moe.router import TopKRouter
from simple_moe.dispatcher import AllToAllDispatcher

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"

results = []

def record(name, ok, msg=""):
    results.append((name, ok, msg))

def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    ws = dist.get_world_size()

    ep_group = dist.new_group(list(range(ws)))

    if rank == 0:
        print("=" * 60)
        print("  SimpleMoE Correctness Tests")
        print(f"  GPUs: {ws}")
        print("=" * 60)

    # Use enough tokens so every expert gets at least a few
    BS, SL = 16, 128  # 2048 tokens, top-2 → 4096 copies across 16 experts = ~256/expert

    # ==============================================================
    # Test 1: Full model expert gradients
    # ==============================================================
    if rank == 0:
        print("\n[Test 1] Full model expert gradient check...")

    cfg = MoEModelConfig(
        vocab_size=1000, hidden_dim=256, num_layers=2, num_heads=4,
        head_dim=64, num_moe_layers=1, num_experts=ws * 2, top_k=2,
        expert_ffn_dim=128, dense_ffn_dim=512, score_func="sigmoid",
        load_balance="aux_loss", aux_loss_coeff=0.01,
    )
    model = MoETransformerLM(cfg, ep_group=ep_group, edp_group=ep_group).to(torch.float32).cuda()
    ids = torch.randint(0, 1000, (BS, SL), device="cuda")
    loss, aux = model(ids, labels=ids)
    loss.backward()

    expert_names = [n for n, _ in model.named_parameters()
                    if "w_gate" in n or "w_up" in n or "w_down" in n]
    expert_with_grad = [n for n in expert_names
                        if model.get_parameter(n).grad is not None
                        and model.get_parameter(n).grad.norm().item() > 0]
    router_names = [n for n, _ in model.named_parameters() if "router" in n]
    router_with_grad = [n for n in router_names
                        if model.get_parameter(n).grad is not None
                        and model.get_parameter(n).grad.norm().item() > 0]
    attn_names = [n for n, _ in model.named_parameters() if "attn" in n]
    attn_with_grad = [n for n in attn_names
                      if model.get_parameter(n).grad is not None
                      and model.get_parameter(n).grad.norm().item() > 0]

    if rank == 0:
        print(f"  Expert params: {len(expert_with_grad)}/{len(expert_names)} have grad")
        print(f"  Router params: {len(router_with_grad)}/{len(router_names)} have grad")
        print(f"  Attn params  : {len(attn_with_grad)}/{len(attn_names)} have grad")
        ok = len(expert_with_grad) == len(expert_names)
        record("Expert gradients", ok,
               f"{len(expert_with_grad)}/{len(expert_names)}")
        record("Router gradients", len(router_with_grad) == len(router_names),
               f"{len(router_with_grad)}/{len(router_names)}")

    del model; torch.cuda.empty_cache()
    dist.barrier()

    # ==============================================================
    # Test 2: Standalone MoE layer autograd
    # ==============================================================
    if rank == 0:
        print("\n[Test 2] Standalone MoE layer autograd chain...")

    cfg2 = MoEModelConfig(
        hidden_dim=256, num_experts=ws * 2, top_k=2,
        expert_ffn_dim=128, score_func="sigmoid",
        load_balance="aux_loss", aux_loss_coeff=0.01,
    )
    moe = MoELayer(cfg2, ep_group=ep_group, edp_group=ep_group).to(torch.float32).cuda()
    x = torch.randn(2048, 256, dtype=torch.float32, device="cuda", requires_grad=True)
    out, aux_l = moe(x)

    has_grad_fn = out.grad_fn is not None

    if rank == 0:
        print(f"  Output has grad_fn: {has_grad_fn}")
        record("MoE autograd chain", has_grad_fn,
               "All-to-All breaks chain" if not has_grad_fn else "OK")
        if not has_grad_fn:
            print(f"  → {WARN}: All-to-All breaks autograd. Expert grads flow via")
            print(f"    embedding→attn→MoE path in full model, NOT through combine A2A.")
            print(f"    This is a known limitation; needs custom autograd Function.")

    del moe; torch.cuda.empty_cache()
    dist.barrier()

    # ==============================================================
    # Test 3: Convergence
    # ==============================================================
    if rank == 0:
        print("\n[Test 3] Training convergence (fixed data, 20 steps)...")

    cfg3 = MoEModelConfig(
        vocab_size=1000, hidden_dim=256, num_layers=2, num_heads=4,
        head_dim=64, num_moe_layers=1, num_experts=ws * 2, top_k=2,
        expert_ffn_dim=128, dense_ffn_dim=512, score_func="sigmoid",
        load_balance="aux_loss", aux_loss_coeff=0.01,
    )
    model3 = MoETransformerLM(cfg3, ep_group=ep_group, edp_group=ep_group).to(torch.bfloat16).cuda()
    opt3 = torch.optim.AdamW(model3.parameters(), lr=1e-3)
    torch.manual_seed(42 + rank)
    fixed_ids = torch.randint(0, 1000, (BS, SL), device="cuda")

    losses = []
    for step in range(20):
        loss, _ = model3(fixed_ids, labels=fixed_ids)
        loss.backward()
        opt3.step(); opt3.zero_grad(set_to_none=True)
        losses.append(loss.item())

    if rank == 0:
        print(f"  Loss[0]={losses[0]:.4f}  Loss[9]={losses[9]:.4f}  Loss[19]={losses[19]:.4f}")
        decreased = losses[19] < losses[0]
        pct = (losses[0] - losses[19]) / losses[0] * 100
        print(f"  Decreased: {decreased}  ({pct:.1f}% reduction)")
        record("Convergence", decreased, f"{pct:.1f}% loss reduction")

    del model3, opt3; torch.cuda.empty_cache()
    dist.barrier()

    # ==============================================================
    # Test 4: Router output validity
    # ==============================================================
    if rank == 0:
        print("\n[Test 4] Router output validity...")

    H, E, K = 256, ws * 2, 2
    torch.manual_seed(42)
    router = TopKRouter(H, E, K, score_func="sigmoid", load_balance="aux_loss").to(torch.bfloat16).cuda()
    rx = torch.randn(2048, H, dtype=torch.bfloat16, device="cuda")
    probs, topk_idx, routing_map, aux_r = router(rx)

    if rank == 0:
        c1 = (probs > 0).all().item()
        c2 = (topk_idx >= 0).all().item() and (topk_idx < E).all().item()
        c3 = (routing_map.sum(dim=-1) == K).all().item()
        c4 = routing_map.any(dim=0).sum().item()

        print(f"  Probs > 0        : {c1}")
        print(f"  Indices in [0,E) : {c2}")
        print(f"  K per token      : {c3}")
        print(f"  Experts used     : {c4}/{E}")
        record("Router probs>0", c1)
        record("Router indices valid", c2)
        record("Router K per token", c3)
        record("Router expert coverage", c4 >= E // 2, f"{c4}/{E}")

    del router; torch.cuda.empty_cache()
    dist.barrier()

    # ==============================================================
    # Test 5: Dispatch-Combine round trip
    # ==============================================================
    if rank == 0:
        print("\n[Test 5] Dispatch-Combine round trip...")

    dispatcher = AllToAllDispatcher(ep_group=ep_group, num_local_experts=E // ws)
    router5 = TopKRouter(H, E, K, score_func="sigmoid").to(torch.bfloat16).cuda()
    T5 = 2048
    x5 = torch.randn(T5, H, dtype=torch.bfloat16, device="cuda")
    probs5, topk5, rmap5, _ = router5(x5)
    dispatched5, meta5 = dispatcher.dispatch(x5, probs5, topk5, rmap5)
    combined5 = dispatcher.combine(dispatched5, meta5)

    if rank == 0:
        shape_ok = combined5.shape == (T5, H)
        no_nan = not torch.isnan(combined5).any().item()
        no_inf = not torch.isinf(combined5).any().item()
        print(f"  Shape {tuple(combined5.shape)} == ({T5},{H}): {shape_ok}")
        print(f"  No NaN: {no_nan}  No Inf: {no_inf}")
        record("Dispatch round-trip shape", shape_ok)
        record("Dispatch round-trip finite", no_nan and no_inf)

    del dispatcher, router5; torch.cuda.empty_cache()
    dist.barrier()

    # ==============================================================
    # Test 6: Gradient magnitudes
    # ==============================================================
    if rank == 0:
        print("\n[Test 6] Gradient magnitude analysis...")

    cfg6 = MoEModelConfig(
        vocab_size=1000, hidden_dim=256, num_layers=2, num_heads=4,
        head_dim=64, num_moe_layers=1, num_experts=ws * 2, top_k=2,
        expert_ffn_dim=128, dense_ffn_dim=512, score_func="sigmoid",
        load_balance="aux_loss", aux_loss_coeff=0.01,
    )
    model6 = MoETransformerLM(cfg6, ep_group=ep_group, edp_group=ep_group).to(torch.float32).cuda()
    ids6 = torch.randint(0, 1000, (BS, SL), device="cuda")
    loss6, _ = model6(ids6, labels=ids6)
    loss6.backward()

    if rank == 0:
        total_p = sum(1 for _, p in model6.named_parameters() if p.requires_grad)
        with_grad = sum(1 for _, p in model6.named_parameters()
                        if p.grad is not None and p.grad.norm().item() > 0)
        zero_grad = sum(1 for _, p in model6.named_parameters()
                        if p.grad is not None and p.grad.norm().item() == 0)
        no_grad = sum(1 for _, p in model6.named_parameters()
                      if p.requires_grad and p.grad is None)
        nan_grad = sum(1 for _, p in model6.named_parameters()
                       if p.grad is not None and torch.isnan(p.grad).any().item())

        print(f"  Params requiring grad : {total_p}")
        print(f"  With non-zero grad    : {with_grad}")
        print(f"  With zero grad        : {zero_grad}")
        print(f"  With no grad at all   : {no_grad}")
        print(f"  With NaN grad         : {nan_grad}")

        # Sample expert grad norms
        for n, p in model6.named_parameters():
            if ("w_gate" in n or "w_down" in n or "router" in n) and p.grad is not None:
                print(f"    {n}: grad_norm={p.grad.norm().item():.6f}")
                break

        record("All params get grad", with_grad == total_p,
               f"{with_grad}/{total_p}")
        record("No NaN grads", nan_grad == 0, f"{nan_grad} NaN")

    del model6; torch.cuda.empty_cache()
    dist.barrier()

    # ==============================================================
    # Summary
    # ==============================================================
    if rank == 0:
        print("\n" + "=" * 60)
        print("  SUMMARY")
        print("=" * 60)
        passed = failed = warned = 0
        for name, ok, msg in results:
            if ok is True:
                status = PASS; passed += 1
            elif ok is False:
                status = FAIL; failed += 1
            else:
                status = WARN; warned += 1
            extra = f" ({msg})" if msg else ""
            print(f"  [{status}] {name}{extra}")

        print(f"\n  Total: {passed} passed, {failed} failed, {warned} warnings")
        print("=" * 60)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
