---
name: gpu-trace-analysis
description: >-
  Analyzes PyTorch / Kineto profiler traces from AMD GPUs (MI300X / MI325X /
  MI350 / MI355X, ROCm, RCCL) and NVIDIA GPUs (H100 / B200, CUDA, NCCL),
  producing a per-stream, per-kernel-category breakdown and rendering the
  result as a Cursor canvas. Supports two modes: (1) single-GPU deep dive
  (used to characterise an AMD or NVIDIA run on its own), and (2) head-to-head
  comparison between an AMD and an NVIDIA run on the same workload. Use when
  the user asks to analyze, profile, breakdown, compare or visualize a GPU
  trace, mentions Kineto / pt.trace.json / torch.profiler / RCCL / NCCL /
  ProfilerStep / overlap / pipeline / per-stream busy time, or wants an
  AMD-vs-NVIDIA performance comparison rendered as charts and a multi-stream
  pipeline diagram.
---

# GPU trace analysis (AMD & AMD-vs-NVIDIA)

This skill turns a raw PyTorch / Kineto trace into a structured canvas with:

- per-stream busy time
- per-kernel-category breakdown (GEMM, attention, norms, MoE dispatch, NCCL/RCCL collectives, ...)
- compute / collective overlap (NCCL hidden-behind-compute %)
- a multi-stream pipeline diagram (SVG lanes with colored kernel segments)

It works for **single-GPU analysis** (typically an AMD MI355X / MI300X run) and for **AMD ↔ NVIDIA comparisons** (e.g. MI355X vs B200 on the same model).

## When to apply

Trigger this skill when the user asks for any of:

- "analyze this trace / profile / pt.trace.json / Kineto json"
- "MI300/MI325/MI350/MI355X performance breakdown"
- "compare MI355X vs B200" / "AMD vs NVIDIA trace"
- "per-stream / multi-stream pipeline diagram", "overlap analysis"
- "RCCL/NCCL all-to-all / reduce-scatter breakdown"

## Workflow

```
- [ ] Step 1: Collect / locate the trace(s)
- [ ] Step 2: Run scripts/full_breakdown.py on each rank you care about
- [ ] Step 3: Extract the numbers needed by the canvas template
- [ ] Step 4: Build the canvas (single-gpu or comparison template)
- [ ] Step 5: Sanity-check the canvas builds and the headline numbers
```

### Step 1 — Locate the trace

A PyTorch profiler trace is usually named `*.pt.trace.json` (50–500 MB per rank). Confirm the file is **complete** before parsing — truncated traces often miss the GPU kernel events at the tail.

If you need to *generate* an AMD trace for a Megatron / Primus run, use the snippet in [Enabling the profiler on AMD](#enabling-the-profiler-on-amd) below — there are several gotchas (yaml gets overwritten by `run.sh`, `record_shapes/with_stack=true` OOMs at 8 ranks, `checkpoint_patches.py` may need an `args` dict patch).

### Step 2 — Run the analyzer

```bash
python .cursor/skills/gpu-trace-analysis/scripts/full_breakdown.py \
    /path/to/rank2.pt.trace.json  ProfilerStep#17
```

Defaults: `ProfilerStep#17` is a stable mid-run step (after warmup/CUDA graph capture). Pick a different step if 17 isn't present.

Environment knobs:

| var                  | default | meaning                                                                           |
| -------------------- | ------- | --------------------------------------------------------------------------------- |
| `SPLIT_NCCL_BY_CPU`  | `1`     | re-label opaque RCCL `nccl_generic` kernels by overlapping `c10d::*` CPU ops      |

The script prints, in order:

1. List of all `ProfilerStep#N` events in the trace (durations).
2. **Per-stream busy time** with bar plot.
3. **GPU kernel category breakdown** — `gemm`, `grouped_gemm`, `attn_kernel`, `norm`, `moe_dispatch`, `nccl_a2a`, `nccl_rs`, `nccl_ag`, `nccl_ar`, `elementwise`, `fp8_cast`, `memcpy`, `optimizer`, `other`.
4. Top-25 GPU kernel names by duration.
5. Per-stream category breakdown for the top streams.
6. **Compute / NCCL overlap** — compute-only ms, nccl-only ms, overlap ms, idle ms, NCCL-hidden-behind-compute %.
7. Time-binned kernel mix (80 bins) — useful for spotting bubbles.

Always run on the **same `ProfilerStep#N`** for both sides of a comparison and on a non-rank-0 rank (rank 0 has extra logging overhead). Other ranks should be within ±3%.

### Step 3 — Extract numbers for the canvas

For each side (AMD / NVIDIA / single-GPU), record:

- step wall time (ms) — from the `[info] window` line
- top streams: name, role, busy ms, share % — from the **per-stream busy time** block
- per-category ms — from the **GPU kernel category breakdown** block
- compute-busy %, NCCL-hidden % — from the **overlap** block
- top kernel names (for the "what's expensive" callout)

Map kernel categories to the segment kinds used by the pipeline diagram:

| breakdown category             | pipeline segment kind |
| ------------------------------ | --------------------- |
| `gemm`, `grouped_gemm`         | `gemm`                |
| `attn_kernel`                  | `attn`                |
| `norm`                         | `norm`                |
| `moe_dispatch`, `fp8_cast`     | `moe`                 |
| `elementwise`                  | `elem`                |
| `optimizer`                    | `opt`                 |
| `nccl_a2a`                     | `a2a`                 |
| `nccl_rs`                      | `rs`                  |
| `nccl_ag`                      | `ag`                  |
| `nccl_ar`                      | `ar`                  |
| (gap)                          | `idle`                |

### Step 4 — Build the canvas

Canvases live at `~/.cursor/projects/<workspace>/canvases/<name>.canvas.tsx`. For this workspace that is `/home/xiaompen/.cursor/projects/home-xiaompen-mlperf-training/canvases/`.

Pick the right template:

- **Single GPU** → copy `templates/single-gpu.canvas.tsx`. Use this when only one trace is being analyzed (typical for AMD-only profiling). The canvas shows: headline stats, one pipeline diagram, kernel breakdown table + stacked-share bar chart, per-stream table, and an analysis card.
- **AMD vs NVIDIA comparison** → copy `templates/comparison.canvas.tsx`. Use this when you have one AMD trace and one NVIDIA trace on (roughly) the same workload. The canvas shows side-by-side everything plus a per-sample throughput stat and two pipeline diagrams stacked vertically.

Both templates already contain:

- the `useSegPalette` color map (chart-aligned, 10 segment kinds + idle)
- a `PipelineDiagram` SVG component (lanes, time axis, segment labels with `<title>` tooltips)
- a `Legend` row matching the palette
- the canvas SDK `import` line — **only** import from `cursor/canvas`, no other modules

To adapt:

1. Replace the `KERNEL_ROWS`, `STREAM_ROWS_*`, `STEP_MS`, `SAMPLES`, `*_LANES` constants with your numbers.
2. Update the headline `<H1>` and the `<Pill>` strip with model / TP / PP / EP / GBS / MBS.
3. Rewrite the closing analysis card(s) — that text should reflect the actual finding (e.g. "communication-bound on a2a", "dominated by FMHA bwd", "optimizer step exposes 12 ms bubble"), not boilerplate.

The `*_LANES` arrays describe lanes for one ProfilerStep, normalized to 100 (`t` and `w` are percentages of the step). Aim for 4–6 lanes per side, ordered by busy time. Segment widths don't need to be exact — they should reflect the **structure** of the timeline (where the big GEMM blocks land, where the collectives interleave). Use the time-binned output of step 2 to lay these out.

### Step 5 — Verify

After writing the canvas:

- Read its lints (no errors expected).
- The headline numbers in the `Stat` grid must match the analyzer output exactly.
- Lane segments for any one stream must sum to 100 in `t + w` order (segments cover the lane).
- No hardcoded hex outside the `SEG_COLORS` map; everything else uses `useHostTheme()` tokens.
- Tell the user the canvas path so they can open it.

## Enabling the profiler on AMD

For Megatron-LM / Primus on MI355X (gpt-oss-20B FP8 etc.):

1. **Edit the model yaml** (e.g. `gpt_oss_20B-pretrain-fp8.yaml`):
   ```yaml
   profile: true
   profile_step_start: 16
   profile_step_end: 20
   profile_ranks: [0,1,2,3,4,5,6,7]
   torch_profiler_with_stack: false       # required — true OOMs at 8 ranks
   torch_profiler_record_shapes: false    # required — same reason
   torch_profiler_use_gzip: false
   train_iters: 24
   exit_interval: 24
   tensorboard_dir: <abs path>/torchprof
   ```
2. **Beware the run.sh overwrite**: `primus/run.sh` copies the *original* yaml from the host into the container at launch, overwriting any in-place edits made inside the container. Edit the host-side file before starting (or back up + replace before re-launching).
3. **`checkpoint_patches.py` AttributeError**: when training exits at `train_iters` with profiler on, the checkpoint patch may try `args.disable_last_saving` while `args` is a dict. Patch it to be defensive:
   ```python
   _dls = args.disable_last_saving if hasattr(args, "disable_last_saving") \
       else (isinstance(args, dict) and args.get("disable_last_saving", False))
   _ti = args.train_iters if hasattr(args, "train_iters") \
       else (isinstance(args, dict) and args.get("train_iters", -1))
   if _dls and iteration == _ti:
       ...
   ```
4. **Trace size**: with `with_stack=False record_shapes=False`, expect ~80–90 MB per rank for a 4-step active window on gpt-oss-20B-class models.

## Reading trace files efficiently

Trace JSONs are large. **Never** `cat` or `Read` them whole. Patterns:

- Stream-parse with `ijson` (the analyzer already does this).
- For a quick metadata peek:
  ```bash
  python -c "import json,sys; \
    f=open(sys.argv[1]); \
    d=json.loads(f.read(2_000_000)+'\"]}'); \
    print([e['name'] for e in d['traceEvents'][:50]])" trace.json
  ```
- To list all `ProfilerStep#N` durations without loading the whole file, use `rg -o 'ProfilerStep#\d+.{0,80}\"dur\":\s*\d+' trace.json | head`.

## Key terminology (kept consistent throughout the skill)

- **Step** — one `ProfilerStep#N` window (one optimizer iteration).
- **Stream** — a CUDA / HIP execution stream. Identified by integer id in `args.stream`.
- **Compute stream** — the stream carrying GEMM / attention / norm / elementwise. Usually stream 0 on AMD; whichever stream has the biggest non-NCCL share on NVIDIA.
- **NCCL stream** — a stream carrying only NCCL/RCCL kernels. May be more than one.
- **Hidden NCCL** — fraction of NCCL time that overlaps a compute kernel on another stream.
- **Stream oversubscription** — `sum(busy across streams) / step_wall` × 100%. Above 100% means real parallelism across streams.

## Templates and scripts

- `scripts/full_breakdown.py` — analyzer (NVIDIA + AMD aware). Bundled here; original lives at `b200/full_breakdown.py`.
- `templates/single-gpu.canvas.tsx` — canvas template for one-GPU deep dive.
- `templates/comparison.canvas.tsx` — canvas template for AMD-vs-NVIDIA side-by-side.
