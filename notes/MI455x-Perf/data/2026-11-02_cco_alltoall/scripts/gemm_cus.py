"""One GPU: time of a compute kernel from cco_bench, e.g. under TENSILE_PERSISTENT_MAX_CUS=<n>.
   TENSILE_PERSISTENT_MAX_CUS=192 python3 gemm_cus.py gmm"""
import os
import sys
import torch
sys.path.insert(0, os.environ.get("CCO_COMMON", "/home/xiaompen/minimax-poc/examples-phase2/training/common"))
import cco_bench

kernel = sys.argv[1]
sys.argv = [sys.argv[0]]
dev = torch.device("cuda", 0)
call = cco_bench.KERNELS[kernel](cco_bench.parse_args(), dev, torch.Generator(device=dev).manual_seed(0))["call"]
for _ in range(20):
    call()
torch.cuda.synchronize()
ts = []
for _ in range(50):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record(); call(); b.record()
    torch.cuda.synchronize()
    ts.append(a.elapsed_time(b))
ts.sort()
print(f"{kernel} TENSILE_PERSISTENT_MAX_CUS={os.environ.get('TENSILE_PERSISTENT_MAX_CUS', '-')}: {ts[len(ts) // 2] * 1e3:.0f} us")
