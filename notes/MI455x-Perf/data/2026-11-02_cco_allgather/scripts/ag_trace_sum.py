"""python3 ag_trace_sum.py [TRACE_DIR]: summarize TRACE_DIR/r*/k_{kernel,memory_copy}_trace.csv from ag_trace.sh.

Blocks (split at the spin kernels) follow ag_trace.py's defaults: the allgather alone at 1024 .. 16 MiB, then
gemm + 1024 / 64 MiB and gmm_flydsl + 1024 / 64 MiB launched in order (compute first).
  alone: the allgather's kernels (WGs x threads, LDS, VGPR), device time per call, copies per call
  pair:  per episode, from the compute kernel's start: compute end, collective start / end (median over ranks and
         episodes)
"""
import csv
import glob
import statistics
import sys

SIZES = [1024, 512, 256, 128, 64, 16]
PAIRS = [("gemm", 1024), ("gemm", 64), ("gmm_flydsl", 1024), ("gmm_flydsl", 64)]
KNAME = {"gemm": "Cijk_", "gmm_flydsl": "kernel_grouped_nt"}
COMPUTE = tuple(KNAME.values())
d = sys.argv[1] if len(sys.argv) > 1 else "trace"


def wgs(r):
    g = int(r["Grid_Size_X"]) * int(r["Grid_Size_Y"]) * int(r["Grid_Size_Z"])
    return g // (int(r["Workgroup_Size_X"]) * int(r["Workgroup_Size_Y"]) * int(r["Workgroup_Size_Z"]))


def short(name):
    for k in ("ncclDevKernel_Generic", "ddaAllGatherFabricLL128", "__amd_rocclr_copyBufferBatch",
              "__amd_rocclr_copyBuffer", "__amd_rocclr_batchMemOp"):
        if k in name:
            return k
    return name[:40]


def is_comm(r):
    """The allgather's own kernels; the barrier's allreduce is the 1-WG ncclDevKernel."""
    n = r["Kernel_Name"]
    return ("ncclDevKernel" in n and wgs(r) > 1) or "dda::" in n or "__amd_rocclr_copyBuffer" in n


ranks = {}
for f in sorted(glob.glob(f"{d}/r*/k_kernel_trace.csv")):
    rows = sorted(csv.DictReader(open(f)), key=lambda r: int(r["Start_Timestamp"]))
    cps = sorted(csv.DictReader(open(f.replace("kernel_trace", "memory_copy_trace"))),
                 key=lambda r: int(r["Start_Timestamp"]))
    spins = [i for i, r in enumerate(rows) if "spin" in r["Kernel_Name"]]
    bounds = spins + [len(rows)]
    blocks = []
    for b in range(len(spins)):
        blk = rows[bounds[b] + 1:bounds[b + 1]]
        t0 = int(rows[bounds[b]]["End_Timestamp"])
        t1 = int(rows[bounds[b + 1]]["Start_Timestamp"]) if b + 1 < len(spins) else 1 << 62
        blocks.append((blk, [c for c in cps if t0 <= int(c["Start_Timestamp"]) < t1]))
    ranks[f.split("/")[-2]] = blocks

print("allgather alone (rank 0 kernels; times: median over ranks and calls)\n")
print("| 每 rank 输出 | kernel（每次调用） | WG × 线程 | LDS / WG | VGPR | 每次调用 device 时间 | copy / 调用 |")
print("|---|---|---|---|---|---|---|")
for i, mib in enumerate(SIZES):
    blk0 = ranks["r0"][i][0]
    kinds = {}
    for r in blk0:
        if is_comm(r) or "batchMemOp" in r["Kernel_Name"]:
            kinds.setdefault(short(r["Kernel_Name"]), r)
    ncalls = sum(1 for r in blk0 if is_comm(r) and short(r["Kernel_Name"]) != "__amd_rocclr_copyBufferBatch")
    per_call, copies = [], []
    for blocks in ranks.values():
        blk, cps = blocks[i]
        comm = [r for r in blk if is_comm(r)]
        # one call = the comm kernels between two barrier allreduces
        calls, cur = [], []
        for r in blk:
            if "ncclDevKernel" in r["Kernel_Name"] and wgs(r) == 1:
                if cur:
                    calls.append(cur)
                cur = []
            elif r in comm:
                cur.append(r)
        if cur:
            calls.append(cur)
        for c in calls:
            per_call.append((max(int(r["End_Timestamp"]) for r in c) - int(c[0]["Start_Timestamp"])) / 1e3)
        copies.append(len(cps) / max(1, len(calls)))
    ks = "、".join(f"`{k}`" for k in kinds)
    shape = " / ".join(f"{wgs(r)} × {r['Workgroup_Size_X']}" for r in kinds.values())
    lds = " / ".join(f"{int(r['LDS_Block_Size']) / 1024:.1f} KB" for r in kinds.values())
    vgpr = " / ".join(r["VGPR_Count"] for r in kinds.values())
    print(f"| {mib} MiB | {ks} | {shape} | {lds} | {vgpr} | {statistics.median(per_call):.0f} µs | "
          f"{statistics.median(copies):.0f} |")

print("\ncompute + allgather, compute launched first (µs from the compute kernel's start; median over ranks x episodes)\n")
print("| 计算 | 通信 | 计算 kernel：WG × 线程，LDS，VGPR | 计算单独 | 计算结束 | 通信 kernel 开始 | 通信结束 |")
print("|---|---|---|---|---|---|---|")
for j, (k, mib) in enumerate(PAIRS):
    i = len(SIZES) + j
    vals, info, alone = [], None, []
    for blocks in ranks.values():
        # the warmup calls, back to back before the first pair block of this kernel (the last two of them)
        first = len(SIZES) + next(x for x, p in enumerate(PAIRS) if p[0] == k)
        warm = [r for r in blocks[first - 1][0] if r["Kernel_Name"].startswith(KNAME[k])][-2:]
        alone += [(int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e3 for r in warm]
        blk, _ = blocks[i]
        comp = [r for r in blk if r["Kernel_Name"].startswith(COMPUTE)][-6:]
        comm = [r for r in blk if is_comm(r)][-6 * (1 if mib < 128 or mib > 256 else 2):]
        info = info or comp[0]
        for c in comp:
            cs = int(c["Start_Timestamp"])
            mine = [r for r in comm if 0 <= int(r["Start_Timestamp"]) - cs < 3_000_000]
            if not mine:
                continue
            vals.append(((int(c["End_Timestamp"]) - cs) / 1e3, (int(mine[0]["Start_Timestamp"]) - cs) / 1e3,
                         (int(mine[0]["End_Timestamp"]) - cs) / 1e3))
    m = [statistics.median(v[x] for v in vals) for x in range(3)]
    print(f"| `{k}` | {mib} MiB | {wgs(info)} × {info['Workgroup_Size_X']}，{int(info['LDS_Block_Size']) / 1024:.0f} KB，"
          f"{info['VGPR_Count']} | {statistics.median(alone):.0f} | {m[0]:.0f} | {m[1]:.0f} | {m[2]:.0f} |")
