#!/usr/bin/env python3
"""
Full GPU + CPU breakdown of a Kineto trace (PyTorch profiler).

Designed for the B200 GPT-OSS-20B run.

Outputs
-------
- ProfilerStep window list
- Per-stream busy time (GPU)
- Per-category kernel breakdown
- Top kernel names by total dur
- Overlap matrix between {compute, nccl, memcpy} streams
- Phase decomposition: forward / backward / optimizer (heuristic from CPU annotations)

Designed to run on a 300-400 MB trace JSON in <2 min.
"""
import sys, time, ijson, re
from collections import defaultdict, Counter

PATH = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/xiaompen/mlperf-training/b200/" \
    "primus-megatron-exp[gpt_oss_20b_nvidia]-rank[2].1775624570807343871.pt.trace.json"
WINDOW_NAME = sys.argv[2] if len(sys.argv) > 2 else "ProfilerStep#17"

# When set, also try to label nccl_generic kernels by aligning timestamps with c10d cpu_op
# annotations (only useful for AMD/RCCL traces where the GPU kernel name is opaque).
SPLIT_NCCL_BY_CPU = bool(int(__import__("os").environ.get("SPLIT_NCCL_BY_CPU", "1")))

t0 = time.time()
print(f"[info] parsing {PATH}")

# ----------------------------------------------------------------------------
# Categorisation helpers
# ----------------------------------------------------------------------------
def cat_kernel(name: str) -> str:
    n = name.lower()
    # NCCL/RCCL collectives  (ncclDevKernel_*, ncclKernel_*, rccl_*)
    if "nccldevkernel" in n or "ncclkernel" in n or n.startswith("nccl_") \
       or n.startswith("rccl_") or "ncclbackend" in n:
        if "reducescatter" in n or "reduce_scatter" in n: return "nccl_rs"
        if "allreduce" in n or "all_reduce" in n:        return "nccl_ar"
        if "allgather" in n or "all_gather" in n:        return "nccl_ag"
        if "alltoall" in n or "all_to_all" in n or "a2a" in n: return "nccl_a2a"
        if "sendrecv" in n or "send_recv" in n or "send" in n or "recv" in n or "p2p" in n:
            return "nccl_a2a"   # SendRecv on B200 = MoE all-to-all
        if "broadcast" in n: return "nccl_bcast"
        # RCCL "Generic" wraps multiple coll types in one kernel name; categorise by stream
        return "nccl_generic"
    # MoE grouped GEMM (TransformerEngine, primus_turbo)
    if "_grouped" in n or "groupedgemm" in n or "grouped_gemm" in n \
       or "groupedlinear" in n or "te_grouped" in n:
        return "grouped_gemm"
    # Attention kernels (B200: cudnn flash; AMD: ck/ck_v3/aiter::fmha)
    if "fmha" in n or "fmhca" in n or "flash" in n or "scaled_dot_product" in n \
       or "_sdpa_" in n or "_attn_" in n.lower() or "attention" in n \
       or "ck_attn" in n or "ck_v3" in n or "aiter_attn" in n \
       or "aiter::fmha" in n or "fmhafwdkernel" in n or "fmhabwd" in n \
       or "ck_tile" in n and "fmha" in n:
        return "attn_kernel"
    # Norms
    if "rmsnorm" in n or "layernorm" in n or "ln_fwd" in n or "ln_bwd" in n \
       or "norm_kernel" in n or "normalization::" in n \
       or "_rmsnorm_bwd_kernel" in n or "_rmsnorm_fwd_kernel" in n:
        return "norm"
    # GEMMs (cuBLAS, CUTLASS, NVJet (TE-B200), hipBLASLt/Tensile (Cijk_*), rocBLAS)
    if (any(k in n for k in (
            "cutlass", "sm100_xmma", "sm100_gemm", "sm100_warpspecialized",
            "blackwell_", "ampere_fp", "cublaslt", "cublas",
            "fp8_gemm", "_qmm", "wgrad_kernel", "dgrad_kernel",
            "_8bit_gemm", "matmul_kernel", "nvjet", "hgemm",
            "hipblaslt", "ck_gemm", "ck_kernel", "rocblas"))):
        return "gemm"
    # AMD Tensile / hipBLASLt GEMM mangled name: "Cijk_Ailk_Bjlk_*" or "Cijk_Alik_Bljk_*"
    if n.startswith("cijk_") or "_userargs_mt" in n:
        return "gemm"
    # MoE dispatch / sort / permute
    if any(k in n for k in ("permute", "unpermute", "_moe_", "moechunk",
                             "topk", "_sort", "histogram", "indices",
                             "dispatch", "_compute_routing")):
        return "moe_dispatch"
    # FP8 cast
    if any(k in n for k in ("amax", "to_fp8", "fp8cast", "_quantize", "dequant",
                             "cast_to_fp8", "scale_kernel", "cast_fp8")):
        return "fp8_cast"
    # Elementwise / activation
    if any(k in n for k in ("silu", "swiglu", "gelu", "relu", "sigmoid",
                             "elementwise", "vectorized")):
        return "elementwise"
    if "softmax" in n or "logsumexp" in n: return "softmax"
    if "memcpy" in n or "memset" in n: return "memcpy"
    if "adam" in n or "optimizer" in n: return "optimizer"
    if "reduce" in n: return "reduction"
    return "other"

# ----------------------------------------------------------------------------
# Pass 1: locate window
# ----------------------------------------------------------------------------
ts0 = ts1 = None
with open(PATH, "rb") as f:
    it = ijson.items(f, "traceEvents.item")
    while True:
        try: ev = next(it)
        except StopIteration: break
        except Exception: break
        if ev.get("name") == WINDOW_NAME:
            ts0 = float(ev.get("ts", 0)); dur = float(ev.get("dur", 0))
            ts1 = ts0 + dur
            break
print(f"[info] window {WINDOW_NAME}: ts0={ts0:.0f}  dur={dur/1000:.2f} ms")

# ----------------------------------------------------------------------------
# Pass 2: aggregate
# ----------------------------------------------------------------------------
GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
gpu_kern = []                    # (cat, name, ts, dur, stream, pid)
cpu_dur = defaultdict(float)
cpu_top = defaultdict(float)
cpu_cnt = defaultdict(int)
gpu_cat = defaultdict(float)
gpu_cnt = defaultdict(int)
gpu_top = defaultdict(float)
stream_busy = defaultdict(float)
stream_pid = {}                  # stream -> pid
stream_kernels = defaultdict(int)
profiler_steps = []              # all ProfilerStep events
ann_seq = []                     # user_annotation in window
gpu_cat_per_stream = defaultdict(lambda: defaultdict(float))
c10d_intervals = []              # (op_name, ts0, ts1) for c10d::* cpu_op events

n=0
with open(PATH, "rb") as f:
    it = ijson.items(f, "traceEvents.item")
    while True:
        try: ev = next(it)
        except StopIteration: break
        except Exception as e:
            print(f"[warn] tail truncated: {e}")
            break
        n+=1
        if n%1000000==0:
            print(f"  scanned {n:,} ({time.time()-t0:.1f}s)")
        nm  = ev.get("name", "")
        if nm.startswith("ProfilerStep"):
            profiler_steps.append((nm, float(ev.get("ts",0)), float(ev.get("dur",0))))
        c   = ev.get("cat", "")
        ts  = float(ev.get("ts", 0) or 0)
        d   = float(ev.get("dur", 0) or 0)
        if ts < ts0 or ts > ts1: continue
        if c == "cpu_op":
            cc = cat_kernel(nm)
            cpu_dur[cc] += d
            cpu_top[nm] += d
            cpu_cnt[cc] += 1
            if nm.startswith("c10d::") or nm.startswith("nccl:"):
                c10d_intervals.append((nm, ts, ts+d))
        elif c == "user_annotation":
            ann_seq.append((nm, ts, d))
        elif c in GPU_CATS:
            args = ev.get("args", {}) or {}
            stream = args.get("stream") or args.get("stream id") or 0
            pid    = ev.get("pid", -1)
            cc = cat_kernel(nm)
            gpu_cat[cc] += d
            gpu_cnt[cc] += 1
            gpu_top[nm] += d
            stream_busy[stream] += d
            stream_pid[stream] = pid
            stream_kernels[stream] += 1
            gpu_cat_per_stream[stream][cc] += d
            gpu_kern.append((cc, nm, ts, d, stream, pid))

print(f"[info] events scanned={n:,}  gpu_kern={len(gpu_kern):,}  c10d_intervals={len(c10d_intervals)}  ({time.time()-t0:.1f}s)")

# --- Re-classify nccl_generic kernels using overlapping c10d cpu_op intervals ---
def _c10d_to_cat(opname: str) -> str:
    o = opname.lower()
    if "reduce_scatter" in o or "reducescatter" in o: return "nccl_rs"
    if "all_reduce" in o or "allreduce" in o:        return "nccl_ar"
    if "all_gather" in o or "allgather" in o:        return "nccl_ag"
    if "all_to_all" in o or "alltoall" in o:         return "nccl_a2a"
    if "send" in o or "recv" in o:                   return "nccl_a2a"
    if "broadcast" in o:                              return "nccl_bcast"
    if "barrier" in o:                                return "nccl_other"
    return "nccl_other"

if SPLIT_NCCL_BY_CPU and c10d_intervals:
    # Sort intervals for binary search
    c10d_intervals.sort(key=lambda x: x[1])
    starts = [iv[1] for iv in c10d_intervals]
    import bisect
    relabelled = 0
    for i, (cc, name, ts, d, stream, pid) in enumerate(gpu_kern):
        if cc != "nccl_generic": continue
        # Find any cpu c10d interval that overlaps the kernel launch
        # We pick the LATEST starting c10d interval whose start <= ts+d (kernel mid)
        kmid = ts + d/2
        idx = bisect.bisect_right(starts, kmid) - 1
        # Walk back to find any covering interval (cpu interval can be older than kernel ts)
        chosen = None
        j = idx
        while j >= 0 and j > idx - 6:  # look back at most 6 intervals
            iv_name, iv_s, iv_e = c10d_intervals[j]
            # Allow up to 5ms slack (cpu launch -> gpu execute)
            if iv_s <= ts + 5000 and iv_e + 5000 >= ts:
                chosen = iv_name
                break
            j -= 1
        if chosen:
            new_cc = _c10d_to_cat(chosen)
            if new_cc != cc:
                # update aggregates
                gpu_cat[cc] -= d; gpu_cnt[cc] -= 1
                gpu_cat[new_cc] += d; gpu_cnt[new_cc] += 1
                gpu_cat_per_stream[stream][cc] -= d
                gpu_cat_per_stream[stream][new_cc] += d
                gpu_kern[i] = (new_cc, name, ts, d, stream, pid)
                relabelled += 1
    print(f"[info] re-labelled {relabelled} nccl_generic kernels via c10d cpu_op overlap")
print(f"[info] all profiler steps in trace:")
for s in profiler_steps:
    print(f"    {s[0]:18s}  dur={s[2]/1000:7.2f} ms")

# ----------------------------------------------------------------------------
# Reports
# ----------------------------------------------------------------------------
print(f"\n=== GPU per-stream busy time inside {WINDOW_NAME} ({dur/1000:.2f} ms) ===")
items = sorted(stream_busy.items(), key=lambda x: -x[1])
for s, v in items:
    pid = stream_pid.get(s, -1)
    pct = v/dur*100
    bar = "#"*int(pct/2)
    print(f"  pid={pid!s:>5} stream={s!s:>4}  {v/1000:8.2f} ms  {pct:5.1f}%  n={stream_kernels[s]:6d}  {bar}")

gpu_total = sum(gpu_cat.values())
print(f"\n=== GPU kernel category breakdown (sum across streams = {gpu_total/1000:.2f} ms; "
      f"step wall = {dur/1000:.2f} ms; oversub = {gpu_total/dur*100:.1f}%) ===")
for c, v in sorted(gpu_cat.items(), key=lambda x: -x[1]):
    print(f"  {c:18s} {v/1000:9.2f} ms  {v/gpu_total*100:5.1f}%   n={gpu_cnt[c]:7d}")

print(f"\n=== Top-25 GPU kernel names by dur ===")
for nm, v in sorted(gpu_top.items(), key=lambda x: -x[1])[:25]:
    print(f"  {v/1000:9.2f} ms  ({v/dur*100:5.1f}% of step)  {nm[:130]}")

print(f"\n=== Per-stream category breakdown (top 4 streams) ===")
top_streams = [s for s, _ in items[:6]]
for s in top_streams:
    pid = stream_pid.get(s, -1)
    print(f"  --- pid={pid} stream={s} (busy={stream_busy[s]/1000:.2f} ms) ---")
    cs = gpu_cat_per_stream[s]
    for c, v in sorted(cs.items(), key=lambda x: -x[1])[:8]:
        print(f"      {c:18s} {v/1000:8.2f} ms")

# ----------------------------------------------------------------------------
# Overlap analysis (slot=50us)
# ----------------------------------------------------------------------------
SLOT = 50  # microseconds
nslots = int(dur//SLOT) + 2
busy_compute = bytearray(nslots)   # any non-NCCL non-memcpy kernel
busy_nccl    = bytearray(nslots)
busy_memcpy  = bytearray(nslots)
for cc, _, ts, d, _, _ in gpu_kern:
    s0 = max(0, int((ts-ts0)//SLOT))
    s1 = min(nslots, int((ts+d-ts0)//SLOT)+1)
    if cc.startswith("nccl"):
        for k in range(s0,s1): busy_nccl[k] = 1
    elif cc == "memcpy":
        for k in range(s0,s1): busy_memcpy[k] = 1
    else:
        for k in range(s0,s1): busy_compute[k] = 1

c_only = sum(1 for i in range(nslots) if busy_compute[i] and not busy_nccl[i])
n_only = sum(1 for i in range(nslots) if busy_nccl[i] and not busy_compute[i])
both   = sum(1 for i in range(nslots) if busy_nccl[i] and busy_compute[i])
idle   = sum(1 for i in range(nslots) if not busy_nccl[i] and not busy_compute[i])

print(f"\n=== Compute / NCCL overlap (slot={SLOT}us) ===")
print(f"  compute-only             {c_only*SLOT/1000:8.2f} ms")
print(f"  nccl-only                {n_only*SLOT/1000:8.2f} ms")
print(f"  overlap(compute & nccl)  {both*SLOT/1000:8.2f} ms")
print(f"  idle                     {idle*SLOT/1000:8.2f} ms")
nccl_t = both + n_only
if nccl_t:
    print(f"  -> NCCL hidden behind compute: {both/nccl_t*100:.1f}% "
          f"({both*SLOT/1000:.1f}/{nccl_t*SLOT/1000:.1f} ms)")
print(f"  -> Compute kernels were busy {(c_only+both)*SLOT/1000:.2f} ms "
      f"({(c_only+both)*100/nslots:.1f}% of step)")

# ----------------------------------------------------------------------------
# Phase split (forward / backward / optimizer) using CPU annotations
# ----------------------------------------------------------------------------
fwd_start = bwd_start = opt_start = None
fwd_end = bwd_end = opt_end = None
ann_seq.sort(key=lambda x:x[1])
# Heuristic: in Megatron we have user annotations like
#   "forward", "backward", "optimizer", or based on the order of layers.
# As a fallback we mark
#   forward = first half of ProfilerStep
#   backward = second half until last NCCL_RS finishes
#   optimizer = final tail
# We will instead use the sequence of annotations + cpu_op patterns.

# Simpler: bin GPU kernels into 100 equal-time bins and show stacked area
NBINS = 80
bins = [defaultdict(float) for _ in range(NBINS)]
binw = dur/NBINS
for cc, _, ts, d, _, _ in gpu_kern:
    b0 = int(max(0, ts-ts0)/binw)
    b1 = min(NBINS-1, int(max(0, ts+d-ts0)/binw))
    for b in range(b0, b1+1):
        bs = b*binw + ts0
        be = bs + binw
        ovl = max(0, min(be, ts+d) - max(bs, ts))
        bins[b][cc] += ovl

print(f"\n=== Time-binned kernel mix (each bin = {binw/1000:.1f} ms) ===")
print("    bin   t(ms)    gemm  gGEMM   attn   norm  moeD   nccl  memcpy  elemw  other  total")
for i, b in enumerate(bins):
    g = b.get("gemm",0); gg = b.get("grouped_gemm",0); at = b.get("attn_kernel",0)
    no = b.get("norm",0); mo = b.get("moe_dispatch",0)
    nc = sum(v for k,v in b.items() if k.startswith("nccl"))
    mc = b.get("memcpy",0); el = b.get("elementwise",0)
    ot = sum(b.values()) - g-gg-at-no-mo-nc-mc-el
    tt = sum(b.values())
    print(f"    {i:3d}  {i*binw/1000:6.1f}  {g/1000:6.2f} {gg/1000:6.2f} {at/1000:6.2f} {no/1000:6.2f} "
          f"{mo/1000:6.2f} {nc/1000:6.2f} {mc/1000:6.2f} {el/1000:6.2f} {ot/1000:6.2f}  {tt/1000:6.2f}")

print(f"\n[done] in {time.time()-t0:.1f}s")
