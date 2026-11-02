"""Markdown table of dp_shapes.sh results: python3 dp_table.py OUT_DIR"""
import glob
import json
import os
import sys

d = sys.argv[1]
res = {}
for f in glob.glob(os.path.join(d, "*.json")):
    variant, shape = os.path.basename(f)[:-5].split("_", 1)
    for c in json.load(open(f))["cases"]:
        if c.get("summary") and c["summary"].get("m2"):
            m = min(c["summary"]["m2"], key=lambda m: abs(m["ratio"] - 1))
            res[(shape, c["size_mib"], variant)] = m

print("| 计算 | 通信 | 纯 kernel（默认 → dp） | 纯通信 | 串行（默认） | overlap：默认 → dp | E：默认 → dp | dp 相对默认串行 |")
print("|---|---|---|---|---|---|---|---|")
order = ["gmm-up912", "gmm-down912", "gmm-up456", "gmm-down456", "gemm-up8192", "gemm-up4096", "gemm-down8192"]
for shape in order:
    for size in (768, 384, 256, 128, 64):
        a, b = res.get((shape, size, "default")), res.get((shape, size, "dp"))
        if not a or not b:
            continue
        n = f"{a['n_comp']}" if a["n_comp"] == b["n_comp"] else f"{a['n_comp']} / {b['n_comp']}"
        serial = a["t_comp"] + a["t_comm"]
        print(f"| `{shape}` ×{n} | {size} MiB | {a['t_comp']:.3f} → {b['t_comp']:.3f} | {a['t_comm']:.3f} | {serial:.3f} | "
              f"{a['t_ovl']:.3f} → **{b['t_ovl']:.3f}** | {a['efficiency']:.2f} → {b['efficiency']:.2f} | "
              f"{b['t_ovl'] / serial:.2f} |")
