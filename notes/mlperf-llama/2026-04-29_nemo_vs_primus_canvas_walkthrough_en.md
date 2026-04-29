# Canvas walk-through script (EN)

> **Use with**: `nemo-vs-primus-llama2-70b-lora-trace.canvas.tsx`
>
> **Audience**: dev/perf review meeting. ~5-7 min if you read straight through; ~3 min if you skip the optional sections marked `[skip-if-tight]`.
>
> **How to use**: open the canvas next to the chat, scroll top → bottom. Each section header below maps to what's on screen. Read the quoted line as you point.

---

## §0 Title bar — pills at the top

> "OK, this is the NeMo vs Primus comparison for **Llama-2-70B LoRA SFT** on **8 MI355X**. Both runs use the **same model**, **same parallelism — TP, PP, CP all 1, pure DP=8**, **same packed seq 8K**, **same global batch 8**. So this is a clean apples-to-apples test."

---

## §1 Green "Headline" callout

> "Top line: **NeMo step is 1490 ms, Primus is 1626 ms — NeMo wins by 8.4 %**."
>
> "Almost the entire gap is the **191 ms DataLoader idle hole** at the start of every Primus step. NeMo eliminates it with 8 prefetching workers."
>
> "NeMo also saves about **50 ms of RCCL** on the LoRA-A2A path, and pays **1.2 GB per step of HtoD memcpy** because it runs unfused TE ops — but that memcpy is fully overlapped behind compute, so it's free."
>
> "Net result: **NeMo's compute stream is 99.5 % busy, Primus's is 84.9 %**. NeMo also already converges to target eval-acc in **10.79 minutes** wall-clock."

---

## §2 Five Stat tiles (1626 / 1490 / 1.091× / 11.8 % / 0.5 %)

> "These are the headline numbers. Primus 1626 ms, NeMo 1490 ms, speed-up 1.091×, **Primus idle 11.8 %, NeMo idle 0.5 %** — that 11-point idle delta is the whole story."

---

## §3 NeMo end-to-end card (5 green tiles)

> "For reference: NeMo this run hits **eval-acc 0.9244** — above the 0.925 target — at **step 384**, 3072 samples, in **10.79 minutes** wall-clock at **5.40 samples per second**. Production step time per the MLLOG is 1502 ms, matches the 1490 ms ProfilerStep within 1 %."
>
> "**Primus has not been run end-to-end yet** — today we only have a profiling run. So we cannot quote a Primus wall-clock to compare against this 10.79 min. That's action item A3."

---

## §4 Per-stream timeline — Primus pipeline diagram

> "Now the actual trace. **Primus pipeline first**."
>
> "Notice the **gray idle block at the very start, taking up 12 % of the step** — that's **191 ms of GPU waiting** for the DataLoader. After that, compute fills the rest: FP8 GEMMs in blue, attention forward and backward in purple, with a small RCCL all-reduce tail at the end."
>
> "Two streams visible: **stream 0 is compute, stream 33 is RCCL DDP**. The RCCL tail is 54 ms and only shows up at the end because everything else was overlapped during backward."

---

## §5 Per-stream timeline — NeMo pipeline diagram

> "Now NeMo. Look — **no idle block at the start**. Compute is packed wall-to-wall."
>
> "The **second lane on stream 3** is the memcpy lane — that orange band running from 0 to 80 % is the **1.2 GB HtoD prefetch**. It's running in parallel with compute on the same stream, using a separate hardware queue."
>
> "Stream 45 is RCCL — just two tiny pulses, total 5 ms. That's because NeMo uses LoRA all-to-all instead of DDP reduce-scatter."

---

## §6 DataLoader configuration cards (the headline gap)

> "Why is Primus's 191 ms idle hole there? **It's the DataLoader**."
>
> "Primus runs `num_workers = 0` — single process, synchronous. Every step, the main thread reads the batch, calls `tril` on an 8K × 8K mask, then HtoD-copies it. The GPU just waits."
>
> "NeMo runs **8 persistent workers with prefetch** — next batch is on GPU before the step starts."
>
> "**We tried to enable workers on Primus today and it didn't go through.** Fork-after-CUDA deadlock — workers hang the moment they touch CUDA because the parent already initialized the CUDA context. Real fix needs `multiprocessing_context=spawn` plumbed into the Bridge dataset_provider, or worker-side CUDA init. Not a one-line YAML change."
>
> "**`[skip-if-tight]`** Trade-off note: 8 workers × 8 ranks = 64 dataloader processes per node, costs ~6-8 GB host RAM."

---

## §7 GPU work decomposition table + bar chart `[skip-if-tight]`

> "Per-kernel-category breakdown across all streams. A few rows worth pointing out:"
>
> "**FP8 weight GEMM**: 773 ms Primus, 806 ms NeMo. Almost identical — this is the actual matmul compute, hipBLASLt autotuner picks the same kernels."
>
> "**Attention** (CK V3 fwd+bwd): 315 vs 329 ms. Same kernel — `aiter::fmha_*_psskddv` — within step jitter."
>
> "**FP8 cast/transpose standalone kernel**: **21 ms on Primus vs 1232 ms on NeMo**. Huge gap. This is the TE op-fuser difference — Primus fuses cast+transpose into the GEMM, NeMo runs it as a separate kernel."
>
> "**Memcpy HtoD**: 4.5 vs 1199.8 ms. Same root cause — NeMo's FP8 transpose has to be staged from pinned host every step. Both rows are warning-colored because they look bad but are actually overlapped."
>
> "**RCCL gradient sync**: 54 vs 5 ms. Two different communication paths, not NCCL tuning."
>
> "The bar chart on the right shows the same thing as % of step. Notice NeMo's chart goes over 100 % — that's the stream oversubscription from memcpy + compute running in parallel."

---

## §8 Top GPU kernels (two tables) `[skip-if-tight]`

> "Top-10 kernels per side. Primus's top is FP8 GEMM. **NeMo's top two are Memcpy HtoD at 1195 ms and `transpose_optimized_kernel` at 1193 ms** — both warning-colored. From kernel #3 onward both lists look basically the same: same FP8 GEMMs, same `aiter::fmha`, same elementwise."
>
> "**Same takeaway as before**: the scary memcpy / transpose pair is just NeMo running unfused TE ops. The **actual compute kernels are the same on both stacks**."

---

## §9 Stream occupancy + 4 stat tiles `[skip-if-tight]`

> "Stream-level numbers. Primus has 2 active streams because `GPU_MAX_HW_QUEUES=2` is set in its config. NeMo has 3, default."
>
> "**Stream oversubscription on NeMo is 257 %** — that's how the 3830 ms of busy time fits into a 1490 ms step. It's HBM bandwidth + DMA running in parallel with compute."
>
> "Compute-busy is **84.9 % Primus, 99.5 % NeMo** — and again, that 14-point gap is the DataLoader idle hole."

---

## §10 Configuration deltas table — **this is the meaty section**

> "OK, this is the most important table. **All 12 NVTE flags are identical between the two configs** — same CK V3 attention, same FP8 hybrid, same hipBLASLt. So I'm only listing the knobs that actually differ."
>
> "**Top of the table — DataLoader rows**: Primus 0 workers, NeMo 8 persistent. That's the perf gap."
>
> "**Middle — TE op-fuser rows**: Primus has `enable_primus_turbo` and `use_transformer_engine_op_fuser` on, NeMo doesn't. That fuses cast+transpose into the GEMM. This explains the 21 ms vs 1232 ms transpose kernel difference."
>
> "**`[skip-if-tight]`** Cross-entropy and gradient-accumulation-fusion rows — Primus uses fused, NeMo unfused. Small effect."
>
> "Now the rows I want to land on — these are the runtime overrides we found today by reading the `Overwrote` lines in `run.log.429`:"
>
> "- **`grad_reduce_in_fp32 = True`** on Primus, False on NeMo. Twice the RCCL bandwidth."
>
> "- **`fp8_param_gather = True`** on Primus, False on NeMo. Half the AG bandwidth."
>
> "- **`fp8_param = True`** on Primus, False on NeMo. **This single flag explains 70 GB of HBM** — Primus stores FP8 weight + bf16 master = 3 B/param, NeMo stores bf16 only = 2 B/param. On 70B params that's 70 GB."
>
> "- **All six DDP overlap and bucket-tuning knobs** — `overlap_grad_reduce`, `overlap_param_gather`, `overlap_param_gather_with_optimizer_step`, `average_in_collective`, `gradient_reduce_div_fusion`, `pad_buckets_for_high_nccl_busbw` — Primus all True, NeMo all False."
>
> "**Methodology point**: none of this last group is visible in YAML. We only found them by reading `Overwrote` lines in the run log. **Don't trust YAML for FP8 / DDP analysis on Primus**."

---

## §11 Action items — four cards

> "Four buckets:"
>
> "**A1 — high impact, eliminate the 191 ms idle**: get DataLoader workers running with spawn context. Expected to drop step to ~1440 ms."
>
> "**A2 — medium impact, FP8 amax history**: align with NeMo's 4 / most-recent. Saves a bit of HBM and helps scale-out later."
>
> "**A3 — confirmed-good, keep TE op-fuser ON**: don't turn it off to chase NeMo. The 21 ms vs 1232 ms transpose gap proves it's strictly better. HBM gap should be solved by **DistOpt + activation memory audit**, not by disabling op-fuser."
>
> "**A4 — low priority, RCCL channel tuning**: copy NeMo's `NCCL_MIN_CTAS=32` etc. for free win at scale-out. Keep `TORCH_NCCL_HIGH_PRIORITY=1` — that's a Primus advantage."

---

## §12 Methodology callout

> "Both traces produced by torch.profiler with `record_shapes=False`, one active step per rank. Numbers extracted by `full_breakdown.py` on rank 4 NeMo / rank 2 Primus. RCCL kernels categorized via `SPLIT_NCCL_BY_CPU=1`."

---

## Closing

> "To wrap up: **the 8 % step-time gap is DataLoader, fixable in code; the 85 GB HBM gap is config-driven, mostly `fp8_param=True`, also fixable but needs a code change in Primus's bridge precision recipe** — both stacks use different training workflows, so it's not a YAML-level A/B."
>
> "Track 1 — fix DataLoader, run end-to-end, compare wall-clock. Track 2 — patch bridge precision recipe, A/B `fp8_param`, quantify HBM vs step-time trade-off."
>
> "Both need code changes inside Primus. Questions?"

---

## Speaker tips

- **Don't read every row** of the kernel breakdown / top-kernels / configuration deltas — point at the highlighted (warning/danger toned) rows and skip the rest.
- **Critical landings** (slow down, repeat numbers):
  - "184 ms is GPU idle waiting for DataLoader" (§1, §4)
  - "21 ms vs 1232 ms transpose" (§7, §10)
  - "fp8_param=True explains 70 GB HBM" (§10, closing)
  - "Don't trust YAML — read the `Overwrote` log" (§10)
- **If audience asks "why didn't you fix DataLoader today?"**: pivot to §6 — fork-after-CUDA + return_cu_seqlen path both failed; needs Bridge dataset_provider patch.
- **If audience asks "can we just turn off `fp8_param` to free HBM?"**: pivot to §11 A3 — yes in theory, no in YAML; needs patch to bridge `_apply_precision_overrides`.
