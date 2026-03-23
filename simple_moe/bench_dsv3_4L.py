"""
DeepSeek-V3-like 4-layer benchmark: SimpleMoE vs Megatron-Core.
Usage: torchrun --nproc_per_node=8 simple_moe/bench_dsv3_4L.py
"""
from __future__ import annotations
import os, sys, time, traceback, torch, torch.distributed as dist

MEGATRON_PATH = "/home/xiaompen/Megatron-LM-v13.0"

def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    rank, ws = dist.get_rank(), dist.get_world_size()
    BS, SL, NL = 2, 4096, 4

    if rank == 0:
        print("=" * 65)
        print(f"  DSv3-like {NL}-Layer Benchmark  |  {ws}x {torch.cuda.get_device_name(0)}")
        print(f"  H=7168 E=256 K=8 FFN=2048 heads=56  BS={BS} SL={SL}")
        print("=" * 65)

    # ========== SimpleMoE ==========
    if rank == 0:
        print("\n--- SimpleMoE (full transformer) ---")
    from simple_moe.config import MoEModelConfig
    from simple_moe.model import MoETransformerLM

    cfg = MoEModelConfig(
        vocab_size=32000, hidden_dim=7168, num_layers=NL, num_heads=56,
        head_dim=128, num_moe_layers=NL, num_experts=256, top_k=8,
        expert_ffn_dim=2048, dense_ffn_dim=18432, score_func="sigmoid",
        num_shared_experts=0, load_balance="aux_loss", aux_loss_coeff=0.01,
        max_seq_len=SL,
    )
    ep = dist.new_group(list(range(ws)))
    model = MoETransformerLM(cfg, ep_group=ep, edp_group=ep).to(torch.bfloat16).cuda()
    tp = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    if rank == 0:
        print(f"  Params/GPU: {tp/1e9:.2f}B  Mem: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
        print("  Warmup...")
    for _ in range(2):
        ids = torch.randint(0, 32000, (BS, SL), device="cuda")
        loss, _ = model(ids, labels=ids); loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); dist.barrier()
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    N = 5
    for s in range(N):
        ids = torch.randint(0, 32000, (BS, SL), device="cuda")
        loss, aux = model(ids, labels=ids); loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        if rank == 0: print(f"    step {s} loss={loss.item():.4f}")
    torch.cuda.synchronize(); dist.barrier()
    el = time.perf_counter() - t0
    if rank == 0:
        tps = BS*SL*ws*N/el; mem = torch.cuda.max_memory_allocated()/1e9
        print(f"  ms/step      : {el/N*1000:.1f}")
        print(f"  Tokens/s/GPU : {tps/ws:,.0f}")
        print(f"  Peak mem/GPU : {mem:.2f} GB")
    del model, opt; torch.cuda.empty_cache(); dist.barrier()

    # ========== Megatron ==========
    if rank == 0:
        print("\n--- Megatron-Core v0.13 (MoE layers only) ---")
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
        num_layers=NL, hidden_size=7168, num_attention_heads=56,
        ffn_hidden_size=2048, num_moe_experts=256, moe_router_topk=8,
        moe_ffn_hidden_size=2048, moe_router_load_balancing_type="aux_loss",
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
        MoELayer(mcfg, submodules=submodules, layer_number=i) for i in range(NL)
    ]).to(torch.bfloat16).cuda()
    tp2 = sum(p.numel() for p in layers.parameters())
    opt2 = torch.optim.AdamW(layers.parameters(), lr=1e-4)
    if rank == 0:
        print(f"  Params/GPU: {tp2/1e9:.2f}B  Mem: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
        print("  Warmup...")

    T = BS * SL
    for _ in range(2):
        x = torch.randn(T, 7168, dtype=torch.bfloat16, device="cuda")
        for layer in layers: x, _ = layer(x)
        x.sum().backward(); opt2.step(); opt2.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); dist.barrier()
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for s in range(N):
        x = torch.randn(T, 7168, dtype=torch.bfloat16, device="cuda")
        for layer in layers: x, _ = layer(x)
        x.sum().backward(); opt2.step(); opt2.zero_grad(set_to_none=True)
        if rank == 0: print(f"    step {s} loss={x.sum().item():.4f}")
    torch.cuda.synchronize(); dist.barrier()
    el2 = time.perf_counter() - t0
    if rank == 0:
        tps2 = T*ws*N/el2; mem2 = torch.cuda.max_memory_allocated()/1e9
        print(f"  ms/step      : {el2/N*1000:.1f}")
        print(f"  Tokens/s/GPU : {tps2/ws:,.0f}")
        print(f"  Peak mem/GPU : {mem2:.2f} GB")
        print(f"\n{'='*65}")
        print(f"  SUMMARY ({NL} layers, DSv3-like)")
        print(f"{'='*65}")
        ratio = (el2/N)/(el/N)
        print(f"  SimpleMoE : {el/N*1000:.1f} ms/step  {BS*SL*ws*N/el/ws:,.0f} tok/s/GPU  {torch.cuda.max_memory_allocated()/1e9:.1f}GB")
        smoe_tps = BS*SL*ws*N/el/ws
        mega_tps = T*ws*N/el2/ws
        print(f"  Megatron  : {el2/N*1000:.1f} ms/step  {mega_tps:,.0f} tok/s/GPU  {mem2:.1f}GB")
        if ratio > 1:
            print(f"  SimpleMoE is {ratio:.2f}x faster (and includes Attention + Embedding)")
        else:
            print(f"  Megatron is {1/ratio:.2f}x faster (but MoE layers only)")
        print(f"{'='*65}")

    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
