import csv, collections, sys
D = "/perf_apps/xiaoming/megaattn_m0/rocprof"  # rocprofv3 -d output dir
def stage(name):
    if "paged_mqa_logits" in name: return "indexer"
    if "topk" in name: return "topk"
    if "metadata" in name: return None
    if "convert_req_index" in name or "mla" in name: return "sparse_mla"
    return None
cells = [(1, 8192), (1, 131072), (16, 131072), (64, 131072)]
counters = ["FETCH_SIZE", "WRITE_SIZE", "TCC_HIT_sum", "TCC_MISS_sum", "TCP_TCC_READ_REQ_sum"]
for B, N in cells:
    vals = collections.defaultdict(lambda: collections.defaultdict(list))
    for tag in ("FETCH_SIZE", "WRITE_SIZE", "TCC_HIT_sum"):
        path = f"{D}/B{B}_N{N}_{tag}_counter_collection.csv"
        per_disp = collections.defaultdict(dict)
        for r in csv.DictReader(open(path)):
            s = stage(r["Kernel_Name"])
            if s is None: continue
            per_disp[int(r["Dispatch_Id"])].update({"stage": s, r["Counter_Name"]: float(r["Counter_Value"])})
        # split into iterations: an indexer dispatch starts each iteration
        its, cur = [], None
        for d in sorted(per_disp):
            e = per_disp[d]
            if e["stage"] == "indexer":
                cur = collections.defaultdict(float); its.append(cur)
            if cur is None: continue
            for c, v in e.items():
                if c != "stage": cur[(e["stage"], c)] += v
        last = its[-1]
        for (s, c), v in last.items(): vals[s][c] = v
    logits = B * N * 4
    print(f"B={B} N={N} logits={logits/1e6:.2f} MB")
    for s in ("indexer", "topk", "sparse_mla"):
        v = vals[s]
        hit = v["TCC_HIT_sum"] / max(1, v["TCC_HIT_sum"] + v["TCC_MISS_sum"])
        print(f"  {s:10s} HBM/MALL fetch {v['FETCH_SIZE']*1024/1e6:8.2f} MB  write {v['WRITE_SIZE']*1024/1e6:8.2f} MB  "
              f"L2 req {v['TCP_TCC_READ_REQ_sum']*128/1e6:8.2f} MB(x128B)  L2 hit {100*hit:5.1f}%  "
              f"fetch/logits {v['FETCH_SIZE']*1024/logits:5.2f}  L2req/logits {v['TCP_TCC_READ_REQ_sum']*128/logits:5.2f}")
