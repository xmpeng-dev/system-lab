---
name: trace-vram-canvas
description: >-
  Builds a single-rank "trace + VRAM" canvas (`<runname>-trace.canvas.tsx`)
  that fuses a Kineto kernel breakdown and a VRAM-health report onto one
  page. Produces (a) a Cursor canvas with headline stats, per-stream
  pipeline diagram, kernel-category table, top-N kernel list, capacity-vs-
  peak bar, stacked bucket bar, and a 3-finding analysis card; and (b) a
  paired Markdown note for archiving. Works for any single-node training
  run regardless of fine-tuning mode (LoRA SFT, full SFT, continued
  pretraining, dense or MoE) and regardless of vendor (AMD MI300X / MI325X
  / MI350 / MI355X / ROCm / RCCL, NVIDIA H100 / B200 / CUDA / NCCL). Use
  when the user asks to characterise / analyze / visualize a training run,
  wants both kernel-level and VRAM-level views in one canvas, mentions
  trace + memory together, or refers to a `pt.trace.json` + a `run.log`
  with allocator stats. Builds on `gpu-trace-analysis` (kernel breakdown)
  and `analyze-trace-vram` (memory health); use those directly if only one
  of the two views is needed.
---

# Single-rank trace + VRAM canvas

Companion to `gpu-trace-analysis` (kernel breakdown) and `analyze-trace-vram`
(memory health). This skill is the recipe for **fusing both into one canvas**
for any single-node training run on AMD or NVIDIA, in any fine-tuning mode
(LoRA, full FT, pretraining, MoE). The reference output is
`llama2-70b-lora-baseline-trace.canvas.tsx`; `templates/single-gpu-trace-vram.canvas.tsx`
is its parameterized copy.

## When to apply

Trigger when **all** of the following are true:

- A single-node training run needs a write-up (any DP=N, with or without TP/PP/CP/EP).
- A `*.pt.trace.json` from `torch.profiler` is available (or can be produced).
- A `run.log` with allocator stats is available (or can be produced).
- The user wants **one canvas** showing both kernel breakdown AND VRAM, not two.

Use `gpu-trace-analysis` directly if only the kernel view is wanted.
Use `analyze-trace-vram` directly if only memory diagnosis is wanted.

## Workflow

```
- [ ] Step 1: Capture pt.trace.json + run.log (Primus or generic)
- [ ] Step 2: Run full_breakdown.py on a steady-state ProfilerStep
- [ ] Step 3: Re-categorize FP8 GEMM kernels (analyzer puts them in "other")
- [ ] Step 4: Extract VRAM stats from run.log (last allocator line on rank 0)
- [ ] Step 5: Calibrate VRAM bucket decomposition to Pmax (sum ≈ Pmax ± 5%)
- [ ] Step 6: Fill the canvas template and write the paired note
- [ ] Step 7: Verify lints, headline stats, and lane-segment widths sum to 100
```

### Step 1 — Capture trace + log

**Primus (this codebase)** — `run.sh` has a `TRACE=1` branch:

```bash
docker exec <container> bash -c '
cd <repo>/llama2_sft/primus &&
TRACE=1 PRIMUS_TRAIN_ITERS=100 PRIMUS_EVAL_INTERVAL=9999 \
PRIMUS_PROFILE_STEP_START=80 PRIMUS_PROFILE_STEP_END=85 \
RUN_LOG_FILE=/results/<runname>/<runname>.log \
bash run.sh'
```

**Generic torch.profiler** — wrap the train loop:

```python
prof = torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(
        skip_first=80, wait=0, warmup=1, active=4, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("/results/torchprof"),
    record_shapes=False, with_stack=False, profile_memory=False,
)
with prof:
    for it, batch in enumerate(loader):
        train_step(batch); prof.step()
```

Why these knobs (apply to both):

| knob | typical | reason |
| --- | --- | --- |
| `skip_first` / `PROFILE_STEP_START` | 80 | well past CUDA-graph capture / inductor compile / first-iter init |
| `active` window | 4-5 | enough for ≥3 clean middle ProfilerSteps |
| `record_shapes=False`, `with_stack=False` | required at 8-rank | true OOMs on 70B-class models |
| `profile_memory=False` | default | trace stays small; rely on allocator log instead |
| eval disabled / `PRIMUS_EVAL_INTERVAL` huge | yes | eval expands activation buffer → would skew memory peak |

Outputs typically `~80-650 MB` per rank.

### Step 2 — Run the analyzer

```bash
python3 .cursor/skills/gpu-trace-analysis/scripts/full_breakdown.py \
    /path/to/<trace>.pt.trace.json \
    ProfilerStep#82 \
    | tee /results/<runname>/breakdown_step82.txt
```

Pick a step **not** equal to the first or last in the active window (both
have profiler entry/exit overhead). Use a non-rank-0 rank if available
(rank 0 has extra logging).

Capture (these are the canvas inputs):

- `step wall time` (from the `[info] window` line)
- per-stream rows (compute stream + collective stream)
- per-category ms (the "GPU kernel category breakdown" block)
- "Compute / NCCL overlap" block: compute-only, nccl-only, overlap, idle, hidden %
- "Top-25 GPU kernel names by dur" — keep the first 10 for the canvas
- stream-oversubscription % from the breakdown header

### Step 3 — Re-categorize FP8 GEMM (CRITICAL on FP8 runs)

The analyzer's `cat_kernel` heuristic misses several modern FP8 GEMM
families and dumps them into `other`. On any FP8-hybrid run this single
mistake makes "other" look like 60% of the step when in reality FP8 GEMM is
the dominant bucket. The exact pattern depends on the backend:

| backend | FP8 GEMM kernel name pattern (sample) | bucket label |
| --- | --- | --- |
| AMD hipBLASLt (Tensile) | `Custom_Cijk_Alik_Bljk_F8B8BS_*`, `Custom_Cijk_Alik_Bljk_F8BS_*` | `FP8 GEMM (Custom_Cijk_*_F8*)` |
| AMD CK-tile (newer) | `ck_tile::*fp8*`, `ck::tensor_operation::device::*F8*` | `FP8 GEMM (CK-tile)` |
| NVIDIA cuBLASLt (Hopper/Blackwell) | `ampere_fp8_gemm_*`, `hopper_fp8_gemm_*`, `sm90_xmma_*_e4m3*`, `sm100_*_fp8_*` | `FP8 GEMM (cuBLASLt)` |
| NVIDIA Transformer Engine cutlass | `cutlass::Kernel<…fp8…>`, `cutlass_*sm90*fp8*` | `FP8 GEMM (TE cutlass)` |

**How to find them**: from the analyzer's "Top-25" list, anything with a
duration in the top-10 that *isn't* already in the categories
(`gemm`/`grouped_gemm`/`attn_kernel`/`norm`/`moe_*`/`nccl_*`/`elementwise`/
`fp8_cast`/`memcpy`/`optimizer`) is almost always FP8 GEMM mis-categorized
as `other`. Sum their `ms` and add a row:

```ts
{ category: "FP8 GEMM (<backend>)", ms: <sum>, tone: "info" }
```

Subtract the same `ms` from the analyzer's `other` value.

**Sanity check** — after re-categorization:

- `sum(KERNEL_ROWS) ≈ sum(STREAM_ROWS busy)` within ~1%
- both should approach `step_wall × stream_oversubscription`
- if `other` is still > 5% of step, you missed a kernel family — go back

### Step 4 — Extract VRAM stats

**Megatron-Bridge / Primus** — `train_utils.py:671` (compact form):

```bash
grep "mem-max-" <runname>.log | tail -3
```

```text
mem-allocated-gigabytes: 126.44 | mem-max-allocated-gigabytes: 285.84 |
mem-max-reserved-gigabytes: 295.52 | mem-alloc-retires: 0
```

**Generic PyTorch** — sample from the run loop:

```python
torch.cuda.memory_allocated()      # → mem-allocated
torch.cuda.max_memory_allocated()  # → Pmax (working-set peak)
torch.cuda.max_memory_reserved()   # → Rmax (driver-side peak)
torch.cuda.memory_stats()["num_alloc_retries"]  # → retires
torch.cuda.get_device_properties(0).total_memory  # → cap (alternative)
```

**Cap from trace** (works for both backends): read
`deviceProperties[*].totalGlobalMem` (bytes) from the trace JSON header.
Don't iterate `traceEvents` — the file can be > 600 MB; see
`analyze-trace-vram/scripts/probe_trace.py` for the safe peek pattern.

Map into the canvas's `VRAM` const:

| canvas field | source | meaning |
| --- | --- | --- |
| `current` | `mem-allocated-gigabytes` | working set at the sample iter |
| `pmax` | `mem-max-allocated-gigabytes` | peak working set across the run |
| `rmax` | `mem-max-reserved-gigabytes` | driver-side cached peak (= what `nvidia-smi` / `rocm-smi` report) |
| `retires` | `mem-alloc-retires` | > 0 is a red flag (allocator pressure) |
| `capGB` | `totalGlobalMem / 1e9` | per-GPU HBM cap |

The canvas computes `RESERVED_PCT`, `ALLOC_PCT`, `FRAG_PCT`, `HEADROOM`
automatically. Apply the verdict thresholds from `analyze-trace-vram`
(95-98% reserved = TIGHT, ≥98% = CRITICAL); the `Stat` tone in the canvas
already implements them.

### Step 5 — Calibrate VRAM bucket decomposition

Goal: a 6-row table whose `gb` values sum to ≈ `Pmax`. The exact buckets
depend on the training mode. Pick the row set that matches:

**Mode A — LoRA SFT** (`Pt ≪ P`; e.g. Llama-70B + LoRA r=16/alpha=32):

| bucket | estimate | typical (70B, bf16+fp8) |
| --- | --- | --- |
| Weights (mixed precision) | bf16 share × 2B + fp8 share × 1B per param | ~120 GB |
| Activations | `bs × seq × hidden × n_layers × ~2-3 B × overhead` | ~145 GB at seq=8192, layers=80 |
| LoRA grads + Adam state | `Pt × (2 + 4 + 4) B / DP` | < 1 GB |
| TE FP8 caches / cuBLAS workspace / collective bufs | empirical | ~10-15 GB |
| Allocator slack (Rmax − Pmax) | computed | ~5-15 GB |
| Other / unaccounted | `Pmax − sum(above)` | ≤ 10 GB sanity gap |

**Mode B — Full SFT / continued pretraining** (`Pt = P`):

| bucket | estimate | typical (8B full FT, bf16) |
| --- | --- | --- |
| Weights (`W = P × dtype_bytes`) | 2B/param for bf16 | 16 GB for 8B |
| Master copy (fp32, mixed-precision) | `P × 4B / DP` if distopt else `P × 4B` | 32 GB / 8 GB shared |
| Adam state (m + v, fp32) | `P × 8B / DP` if distopt | 64 GB / 8 GB shared |
| Gradients | `P × dtype_bytes` (bf16/fp32) | 16-32 GB |
| Activations | `bs × seq × hidden × n_layers × k_recompute` (k_full=2, k_selective≈10, k_none≈30) | dominant on long seq |
| TE FP8 / cuBLAS / NCCL bufs | empirical | ~10-15 GB |

(Combine Master+Adam+Grad into one "optimizer & grads" row if it keeps the
table to 6 rows.)

**Mode C — MoE pretraining** (e.g. gpt-oss-20B, DeepSeek):

| bucket | estimate | notes |
| --- | --- | --- |
| Dense weights | as Mode B | |
| MoE expert weights | `n_experts × W_expert / EP` | dominates on EP < n_experts |
| Activations + token-permute buffers | bs × seq × hidden × (1 + capacity_factor × top_k) | a2a buffers add ~5-10 GB |
| Optimizer state (sharded) | `(2 + 4 + 4) × P_total / DP` | DistOpt mandatory at this scale |
| MoE dispatch / a2a buffers | empirical ~5-10 GB | |
| Allocator slack | computed | |

In all modes: reconcile the sum to `Pmax`. **If `unaccounted > 10% of Pmax`,
the estimate is wrong** — investigate before publishing. Common causes:
(a) extra TE delayed-scaling history buffers; (b) optimizer state on master
rank only; (c) eval batch held in memory; (d) FSDP all-gather buffer
oversized.

### Step 6 — Fill the canvas + note

**Canvas path**:

```text
~/.cursor/projects/<workspace-slug>/canvases/<runname>-trace.canvas.tsx
```

Find `<workspace-slug>` by listing `~/.cursor/projects/`. **Do not invent
the slug from the repo name** — Cursor often strips suffixes (e.g. the repo
`mlperf-training-llama` lives under the workspace
`home-xiaompen-mlperf-training`, not `home-xiaompen-mlperf-training-llama`).
A canvas written to a non-existent workspace dir shows up on disk but the
canvas pane shows "Failed to open file".

Copy `templates/single-gpu-trace-vram.canvas.tsx`, then replace these
top-of-file constants only:

| const | content |
| --- | --- |
| `RUN.hardware` / `model` / `parallelism` / `batch` / `step` / `stepMs` / `samplesPerStep` / `tflopsPerGPU` / `trainable` | run identity strip |
| `KERNEL_ROWS` | Step 3 output (FP8 GEMM re-categorized) |
| `TOP_KERNELS` | Top-10 from analyzer, with a short `bucket` label |
| `STREAM_ROWS` | per-stream busy from analyzer |
| `LANES` | layout from the time-binned mix; see template comments |
| `COMPUTE_BUSY_PCT`, `NCCL_HIDDEN_PCT`, `STREAM_OVERSUB_PCT`, `IDLE_MS` | overlap block |
| `VRAM` | Step 4 output |
| `VRAM_BUCKETS` | Step 5 estimate (use the row set for the run's mode) |

Then rewrite the closing analysis card (3 numbered findings + 1 callout) to
reflect this run's actual story. Defaults for the canonical baseline:
(1) compute-bound on FP8 GEMM, (2) NCCL fully serial, (3) idle gap at step
head. The callout should always be VRAM headroom (TIGHT / CRITICAL /
HEALTHY).

For **LoRA**, drop the `trainable` pill from the run identity strip if it's
already 100% (full FT). Don't keep the LoRA-specific `LoRA grads + Adam
state` bucket row in the VRAM table for full-FT runs — replace with the
Mode B set.

Also write the paired markdown note at:

```text
slab/notes/<topic>/<YYYY-MM-DD>_<runname>_trace_breakdown.md
```

Use `templates/note.md` as the structure. Put the same data into 11
numbered sections so the note is grep-friendly for follow-up optimizations
(`@notes/.../<file>.md:<line>` references).

### Step 7 — Verify

- `ReadLints` on the canvas — must be 0 errors. Do **not** use
  `tone="critical"` on `Stat`; the valid `StatTone` values are
  `success | warning | danger | info | neutral`. Use `danger`, not
  `critical`.
- Headline `<Stat>` numbers must match the analyzer output exactly.
- Each lane in `LANES`: `segs` rendered chronologically; `t + w` of last
  segment must equal 100.
- `VRAM_BUCKETS` summed over the first 4-5 rows + the slack row should
  equal `Pmax + (Rmax - Pmax)` within ±5 GB. If wider, the bucket estimate
  is wrong, not the data.
- Tell the user the canvas path so they can `Ctrl+P` open it.

## What goes into the closing analysis card

Always 3 numbered findings + 1 highlighted callout. Format:

```
1. <one-line headline in bold>. <2-3 sentences with specific numbers from
   the analyzer — kernel ms, % of step, kernel names>.

2. <one-line headline>. <numbers explaining the symptom + a one-line
   suggested next experiment>.

3. <one-line headline>. <numbers + suggested investigation tool>.

Callout (warning if VRAM is TIGHT/CRITICAL):
   "VRAM headroom is <verdict>": Reserved Rmax / cap, fragmentation %, retires,
   the dominant bucket and one mitigation.
```

Be specific. "FP8 GEMM is heavy" is useless; "FP8 GEMM = 773 ms = 47.6% of
step, dominated by `Custom_Cijk_Alik_Bljk_F8B8BS_*shortname1` at 350 ms" is
the bar.

## Common pitfalls

- **Wrong workspace dir for canvas**: writing to a slug that doesn't exist
  in `~/.cursor/projects/` → file shows up on disk but the canvas pane
  shows "Failed to open file". Always `ls ~/.cursor/projects/` first.
- **`tone="critical"` on `Stat`**: not a valid `StatTone`. Use
  `tone="danger"`.
- **Skipping FP8 GEMM re-categorization**: `other` looks like 60% of step,
  finding #1 ends up wrong, the whole canvas reads as "mystery overhead".
- **Profiling step 0 or the last step**: inflated by profiler entry / exit
  overhead. Always pick a middle step.
- **Sampling VRAM at iter 1**: peak hasn't been reached yet. Use the last
  `mem-max-*` line in the run.log, not the first.
- **Reading the trace JSON whole**: it can be > 600 MB. Use `ijson` (the
  analyzer already does this); see `gpu-trace-analysis/SKILL.md` for the
  metadata-peek snippet.
- **Sum of bucket GB ≠ Pmax**: if the gap > 10 GB, the estimate is wrong.
  Dig before publishing — the canvas should be defensible.
- **Using the LoRA bucket set for a full-FT run** (or vice versa): wildly
  wrong percentages. Pick the right Mode (A / B / C) in Step 5.

## Files in this skill

- `templates/single-gpu-trace-vram.canvas.tsx` — the canvas template
  (verbatim copy of the canonical baseline canvas, all run-specific
  constants pulled to the top).
- `templates/note.md` — the paired markdown note template (11 sections,
  matches the canvas structure 1:1).

## Related skills

- `.cursor/skills/gpu-trace-analysis/` — primary kernel-breakdown analyzer
  (`scripts/full_breakdown.py`) and trace-collection how-to. Always use its
  script for Step 2.
- `~/.cursor/skills/analyze-trace-vram/` — VRAM health thresholds, bucket
  estimation playbook, and mitigation table. Always use its thresholds for
  Step 4 / Step 5.
- `~/.cursor/skills-cursor/canvas/` — canvas SDK (`cursor/canvas`)
  component reference. Read it before adding any new component beyond what
  the template uses.
