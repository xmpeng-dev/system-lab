"""python3 trace_sum.py [DIR]: summarize DIR/<case>/r*/k_kernel_trace.csv (DIR defaults to trace): resources of each kernel, and per episode the comm vs compute timing."""
import csv, glob, statistics, sys
for case in sorted(glob.glob(f"{sys.argv[1] if len(sys.argv) > 1 else 'trace'}/*")):
    print(f"== {case}")
    seen, eps = {}, []
    for f in sorted(glob.glob(f"{case}/r*/k_kernel_trace.csv")):
        rows = sorted(csv.DictReader(open(f)), key=lambda r: int(r["Start_Timestamp"]))
        for r in rows:
            nm = r["Kernel_Name"]
            if "Fill" in nm or "spin" in nm: continue
            key = nm[:60]
            if key not in seen:
                wgs = int(r["Grid_Size_X"]) * int(r["Grid_Size_Y"]) * int(r["Grid_Size_Z"]) // (int(r["Workgroup_Size_X"]) * int(r["Workgroup_Size_Y"]) * int(r["Workgroup_Size_Z"]))
                seen[key] = f"WG {wgs:6d} x {r['Workgroup_Size_X']:>4} thr  LDS {int(r['LDS_Block_Size'])/1024:6.1f} KB  VGPR {r['VGPR_Count']:>4}  {key}"
        spins = [i for i, r in enumerate(rows) if "spin" in r["Kernel_Name"]]
        for i in spins[-5:]:
            nxt = next((j for j in spins if j > i), len(rows))
            ep = [r for r in rows[i + 1:nxt] if int(r["Start_Timestamp"]) < int(rows[i]["End_Timestamp"]) + 5_000_000]
            ep = [r for r in ep if int(r["Start_Timestamp"]) - int(rows[i]["End_Timestamp"]) < 3_000_000]
            nc = [r for r in ep if "nccl" in r["Kernel_Name"]][:2]
            cp = [r for r in ep if "nccl" not in r["Kernel_Name"] and not any(t in r["Kernel_Name"] for t in ("elementwise", "distribution", "spin", "rocclr"))]
            if not nc or not cp: continue
            g = int(rows[i]["End_Timestamp"])
            cs, ce = int(cp[0]["Start_Timestamp"]), max(int(r["End_Timestamp"]) for r in cp)
            ns, ne = int(nc[0]["Start_Timestamp"]), int(nc[-1]["End_Timestamp"])
            if f.endswith("r0/k_kernel_trace.csv") and i == spins[-1]:
                for r in cp + nc:
                    print(f"    last episode r0: {(int(r['Start_Timestamp'])-g)/1e3:7.0f} -> {(int(r['End_Timestamp'])-g)/1e3:7.0f} us  WG {int(r['Grid_Size_X'])//int(r['Workgroup_Size_X']):5d}  {r['Kernel_Name'][:50]}")
            eps.append(((cs - g) / 1e3, (ce - g) / 1e3, (ns - g) / 1e3, (ne - g) / 1e3))
    for v in seen.values(): print("   ", v)
    m = [statistics.median(e[j] for e in eps) for j in range(4)]
    print(f"    from the gate (us, median of {len(eps)} rank-episodes): compute {m[0]:.0f} -> {m[1]:.0f}, rccl {m[2]:.0f} -> {m[3]:.0f}")
