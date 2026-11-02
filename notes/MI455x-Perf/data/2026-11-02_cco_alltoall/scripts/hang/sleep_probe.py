"""Does torch.cuda._sleep (the episode gate) ever fail to finish? One process per GPU, no RCCL.
   python3 sleep_probe.py GPU N_ITER [streams]"""
import sys
import time
import torch

gpu, n_iter = int(sys.argv[1]), int(sys.argv[2])
extra = int(sys.argv[3]) if len(sys.argv) > 3 else 0
torch.cuda.set_device(gpu)
others = [torch.cuda.Stream() for _ in range(extra)]     # idle streams, to take up hardware queues
s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
s.record(); torch.cuda._sleep(2_000_000); e.record(); e.synchronize()
cyc = 2_000_000 / (s.elapsed_time(e) * 1e3)
stalls, worst = 0, 0.0
t_start = time.time()
for i in range(n_iter):
    torch.cuda._sleep(int(230 * cyc))
    g = torch.cuda.Event()
    g.record()
    for o in others:
        o.wait_event(g)
    t0 = time.time()
    while not g.query():
        if time.time() - t0 > 5:
            stalls += 1
            print(f"gpu {gpu} iter {i}: gate not done after 5 s", flush=True)
            break
    worst = max(worst, time.time() - t0)
torch.cuda.synchronize()
print(f"gpu {gpu}: {n_iter} gates in {time.time() - t_start:.0f} s, cycles/us {cyc:.0f}, stalls {stalls}, worst wait {worst * 1e3:.1f} ms", flush=True)
