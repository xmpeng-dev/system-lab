"""One GPU: gmm (torch.bmm, hipBLASLt default / data-parallel grid) vs gmm_flydsl, time + FP32 check."""
import os, sys, torch
sys.path.insert(0, os.environ.get("CCO_COMMON", "/home/xiaompen/minimax-poc/examples-phase2/training/common"))
import cco_bench
kernel = sys.argv[1]; sys.argv = [sys.argv[0]]
dev = torch.device("cuda", 0)
k = cco_bench.KERNELS[kernel](cco_bench.parse_args(), dev, torch.Generator(device=dev).manual_seed(0))
for _ in range(20): k["call"]()
torch.cuda.synchronize()
ts = []
for _ in range(50):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record(); k["call"](); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
ts.sort(); t = ts[len(ts) // 2]
got, ref = k["check"]()
err = ((got.float() - ref).norm() / ref.norm()).item()
print(f"{kernel} grid_env={os.environ.get('TENSILE_PERSISTENT_FIXED_GRID', '-')}: {t * 1e3:.0f} us, {k['flops'] / t / 1e9:.0f} TF, rel err {err:.2e}  {k['label']}")
