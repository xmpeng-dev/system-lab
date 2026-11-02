"""One GPU: launch a tiny kernel on a second stream while the gmm / gemm / attn_fwd kernel runs; when does it finish?"""
import os
import sys
import torch
sys.path.insert(0, os.environ.get("CCO_COMMON", "/home/xiaompen/minimax-poc/examples-phase2/training/common"))
import cco_bench

kernel = sys.argv[1]
sys.argv = [sys.argv[0]]
dev = torch.device("cuda", 0)
call = cco_bench.KERNELS[kernel](cco_bench.parse_args(), dev, torch.Generator(device=dev).manual_seed(0))["call"]
s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
tiny = torch.zeros(1024, device=dev)
ev = lambda: torch.cuda.Event(enable_timing=True)
res = []
for it in range(12):
    torch.cuda.synchronize()
    torch.cuda._sleep(200 * 2200)
    g = ev(); g.record()
    s1.wait_event(g); s2.wait_event(g)
    with torch.cuda.stream(s1):
        call(); e1 = ev(); e1.record()
    with torch.cuda.stream(s2):
        torch.cuda._sleep(50 * 2200)                     # the probe kernel arrives 50 us after the compute kernel
        tiny.add_(1); e2 = ev(); e2.record()
    torch.cuda.synchronize()
    if it >= 2:
        res.append((g.elapsed_time(e1), g.elapsed_time(e2)))
res.sort()
c, p = res[len(res) // 2]
print(f"{kernel}: compute ends {c * 1e3:.0f} us, tiny kernel (issued at ~50 us) ends {p * 1e3:.0f} us")
