"""Telemetry and M1 rates of a cco_bench.py results directory: python3 tel_table.py RESULTS_DIR [variant]

Kernels in run.sh order (gemm, gmm_flydsl, attn_fwd, ...). Median gfx clock (mean over XCDs) and socket power over every
local GPU's samples inside the timed windows.
  M1 table  compute alone / collective alone / both; TFLOPS and busbw alone -> both
  M2 table  the pairing closest to N * T_comp = T_comm: compute alone vs both, and the compute's time alone -> in overlap
"""
import glob
import json
import os
import sys

ORDER = ["gemm", "gmm_flydsl", "gmm", "attn_fwd", "attn_bwd", "wgrad", "gmm_wgrad"]
d = sys.argv[1]
variant = sys.argv[2] if len(sys.argv) > 2 else "default"
cases = []
for f in sorted(glob.glob(os.path.join(d, f"{variant}_*.json"))):
    cases += [c for c in json.load(open(f))["cases"] if c.get("summary")]
cases.sort(key=lambda c: (ORDER.index(c["compute"]) if c["compute"] in ORDER else len(ORDER), -c["size_mib"]))


def tel(r):
    mhz, w = r.get("gfx_mhz"), r.get("power_w")
    return f"{mhz:.0f} MHz / {w:.0f} W" if mhz and w else "—"


print("| 计算 | 通信 | 纯计算 | 纯通信 | 同时（M1） | TFLOPS：单独 → 同时 | busbw：单独 → 同时 |")
print("|---|---|---|---|---|---|---|")
for c in cases:
    r, m = c["results"], c["summary"].get("m1")
    if not m:
        continue
    print(f"| `{c['compute']}` | {c['size_mib']} MiB | {tel(r['m1 comp alone'])} | {tel(r['m1 comm alone'])} | "
          f"{tel(r['m1 both'])} | {m['tflops']:.0f} → {m['tflops_under']:.0f} | "
          f"{m['busbw']:.0f} → {m['busbw_under']:.0f} GB/s |")

print("\n| 计算 | 通信 | N | 纯计算（M2） | 同时（M2） | kernel：单独 → overlap 中 |")
print("|---|---|---|---|---|---|")
for c in cases:
    m2 = c["summary"].get("m2") or []
    if not m2:
        continue
    m = min(m2, key=lambda m: abs(m["ratio"] - 1))
    n, r = m["n_comp"], c["results"]
    print(f"| `{c['compute']}` | {c['size_mib']} MiB | {n} | {tel(r[f'm2 x{n} comp'])} | {tel(r[f'm2 x{n} both'])} | "
          f"{m['t_comp']:.3f} → {m['t_comp_in']:.3f} ms（{m['t_comp_in'] / m['t_comp']:.2f}×） |")
