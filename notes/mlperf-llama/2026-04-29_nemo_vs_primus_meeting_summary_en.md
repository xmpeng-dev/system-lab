# NeMo vs Primus — Llama-2-70B LoRA SFT — meeting summary (EN)

> Short, spoken-style. ~1.5-2 min if you read it straight through.
> Two points: where the **perf** gap comes from, and where the **HBM** gap comes from.
>
> Hardware: 8 × MI355X. Workload: Llama-2-70B LoRA SFT, GBS=8, seq=8K (packed), DP=8.
> NeMo step = 1490 ms. Primus step = 1626 ms. NeMo wins by 8.4 %.

---

## 1. Where the perf gap comes from — it's the DataLoader

> "The 8 % gap is **almost entirely DataLoader**. Out of the 136 ms gap, **about 184 ms is Primus's GPU just sitting idle** waiting for the DataLoader at the start of every step."
>
> "Primus today runs `num_workers = 0` — single process. So every step the main thread has to read the packed batch, build the attention mask on CPU (`tril` over 8K × 8K), and then HtoD-copy it. That takes about 190 ms before the GPU does anything."
>
> "NeMo runs **8 persistent workers** with prefetch. The next batch is already on GPU when the step starts. Idle is essentially zero."
>
> "**We tried to fix this on Primus today and it didn't go through.** I added `num_workers=2, persistent_workers=True` via env vars in the recipe. Two ways failed:"
>
> "- **Fork-after-CUDA deadlock**: default fork context, workers hang as soon as they touch CUDA, because the parent already initialized CUDA context."
>
> "- **`return_cu_seqlen=True` path**: would skip the CPU mask gen entirely, but it's incompatible with `fused_single_qkv_rope` in this recipe — RoPE assertion fails."
>
> "So we reverted to baseline. **The fix is real but not a one-line YAML change** — needs `multiprocessing_context=spawn` plumbing in the Bridge `dataset_provider`, or worker-side CUDA init."
>
> "**Bottom line for perf**: once DataLoader is properly fixed, Primus's compute is already on par with NeMo (same FP8 GEMM, same CK V3 attention) — Primus should beat NeMo. The 1.2 GB HtoD memcpy and 1193 ms transpose kernel you see in the NeMo trace are NeMo running unfused TE ops, fully overlapped behind compute — they don't hurt wall-clock."

---

## 2. Where the HBM gap comes from — it's config

> "Primus uses **285.84 GB allocated / 295.52 GB reserved** out of 288 GB on each MI355X. NeMo uses about **200 GB / 211 GB**. Gap is ~85 GB. **Less than 3 GB headroom on Primus — risky.**"
>
> "This gap is **almost entirely driven by configuration**, not by Primus being inefficient. The configs are also **not visible in the YAML** — we only found them from the `Overwrote` lines in `run.log.429:1502-1517`, which show the runtime values after Primus's `precision_config` logic kicks in."

### Main configs that drive the HBM gap

| # | Config | Primus runtime | NeMo | HBM impact |
| --- | --- | --- | --- | --- |
| 1 | `LlamaModelProvider.fp8_param` | **True** (← False) | **False** | **+70 GB on Primus**: stores FP8 weight (1 B) + bf16 master (2 B) = 3 B/param. NeMo bf16 only = 2 B/param. On 70 B params this is the biggest single line item. |
| 2 | `DDPConfig.fp8_param_gather` | **True** (← False) | False | Couples with #1 — Primus AG sharded params in FP8 from DistOpt; NeMo bf16 (but uses LoRA-A2A path which bypasses DDP, so it doesn't pay this BW). |
| 3 | `DDPConfig.grad_reduce_in_fp32` | **True** (← False) | False | Primus RS grads in FP32 (4 B), NeMo in bf16 (2 B). Mostly RCCL bandwidth, but also costs an FP32 reduce buffer in HBM. |
| 4 | DDP overlap stack: `overlap_grad_reduce`, `overlap_param_gather`, `overlap_param_gather_with_optimizer_step`, `average_in_collective`, `gradient_reduce_div_fusion`, `pad_buckets_for_high_nccl_busbw` | **All True** | **All False** | Bucket padding + pre-staged AG buffers cost ~10-15 GB HBM on Primus. NeMo doesn't pay this because all knobs are off and LoRA-A2A bypasses DDP buckets entirely. |
| 5 | `enable_primus_turbo` + `use_transformer_engine_op_fuser` | **True** | False | Primus fuses cast+transpose into the FP8 GEMM. Trade: ~few GB of tile-staging buffer in HBM, but saves ~1200 ms of standalone transpose kernel + 1.2 GB/step HtoD. NeMo pays the time cost (overlapped) instead of the HBM cost. |

### Why we **can't just flip a flag to verify**

> "You can't just go to the Primus YAML and set `fp8_param: False` to do an A/B test. **NeMo and Primus run completely different training workflows** even though both end up calling Megatron-Core under the hood:"
>
> "- **NeMo** path is `train.py` → `MegatronStrategy` (PTL) → `MegatronMixedPrecision(fp8='hybrid', fp8_param=False)` — `fp8_param` is a config knob the strategy reads on top of `te.fp8_model_init`."
>
> "- **Primus** path is `pretrain.py` → Megatron-Bridge `bf16_with_fp8_hybrid` precision_config → bridge's `_apply_precision_overrides` → mcore `TransformerConfig`. The `fp8_param=True` is **set as a runtime override by the bridge precision recipe**, not by the YAML."
>
> "**To do a clean A/B on `fp8_param`, you have to patch Primus's bridge precision recipe** so it sets `fp8_param=False` instead of True. Same for `fp8_param_gather` and `grad_reduce_in_fp32` — they're all set inside `_apply_precision_overrides`."
>
> "Same story for the **6 DDP overlap knobs** — they're written directly in the recipe at `Primus-dev/primus/recipes/llama2_custom.py:585-594` as `True`, not pulled from YAML. So flipping them off requires editing the recipe code."
>
> "**Bottom line for HBM**: we have a clear hypothesis (`fp8_param=True` accounts for ~70 GB out of the 85 GB gap), but **proving it requires code changes** — patching the Primus precision recipe and re-running. Not a one-shot YAML A/B."

---

## 3. What we'll do next

> "Two parallel tracks:"
>
> "**Track 1 — Perf**: properly fix Primus DataLoader prefetch with spawn context (or worker-side CUDA init). Validate step time drops to ~1440 ms. Then run end-to-end to target accuracy and quote a real wall-clock against NeMo's 10.79 min."
>
> "**Track 2 — HBM**: patch Primus's bridge precision recipe to A/B `fp8_param`, `fp8_param_gather`, `grad_reduce_in_fp32` independently. Quantify how much HBM each saves and what it costs in step time. Decide what's safe to disable."
>
> "Both tracks need code changes inside Primus, not just config changes."
