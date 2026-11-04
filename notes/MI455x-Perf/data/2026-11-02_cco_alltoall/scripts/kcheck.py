"""One GPU: every kernel of a model preset; time (median of 30), TF, check against FP32."""
import os, sys, torch
sys.path.insert(0, "/home/xiaompen/minimax-poc/examples-phase2/training/common")
import cco_bench as c
model, kernels = sys.argv[1], sys.argv[2].split(",")
sys.argv = [sys.argv[0], "--model", model]
a = c.parse_args()
dev = torch.device("cuda", 0)
for name in kernels:
    try:
        k = c.KERNELS[name](a, dev, torch.Generator(device=dev).manual_seed(0))
        for _ in range(10): k["call"]()
        torch.cuda.synchronize()
        ts = []
        for _ in range(30):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(); k["call"](); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
        t = sorted(ts)[15]
        err = "-"
        if k.get("check"):
            got, ref = k["check"]()
            err = f"{c._err(got, ref):.1e}"
        print(f"{model:15s} {name:16s} {t*1e3:7.0f} us {k['flops']/t/1e9:6.0f} TF err {err:8s} {k['label']}", flush=True)
    except BaseException as ex:
        print(f"{model:15s} {name:16s} FAILED {type(ex).__name__}: {str(ex)[:160]}", flush=True)
