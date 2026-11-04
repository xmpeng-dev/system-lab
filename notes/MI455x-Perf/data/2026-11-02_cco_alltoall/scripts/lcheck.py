"""One GPU: the layer_* segments of a model preset; time (median of 30), TF, check against FP32, host syncs."""
import sys
import warnings

import torch

sys.path.insert(0, "/home/xiaompen/minimax-poc/examples-phase2/training/common")
import cco_bench as c

model, kernels = sys.argv[1], sys.argv[2].split(",")
sys.argv = [sys.argv[0], "--model", model]
a = c.parse_args()
dev = torch.device("cuda", 0)
for name in kernels:
    k = c.KERNELS[name](a, dev, torch.Generator(device=dev).manual_seed(0))
    for _ in range(5):
        k["call"]()
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("warn")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        k["call"]()
    torch.cuda.set_sync_debug_mode(0)
    syncs = sum("synchroniz" in str(x.message) for x in w)
    torch.cuda.synchronize()
    ts = []
    for _ in range(30):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); k["call"](); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    t = sorted(ts)[15]
    got, ref = k["check"]()
    print(f"{model:12s} {name:15s} {t*1e3:7.0f} us {k['flops']/t/1e9:6.0f} TF err {c._err(got, ref):.1e} "
          f"host syncs {syncs}  {k['label']}", flush=True)
