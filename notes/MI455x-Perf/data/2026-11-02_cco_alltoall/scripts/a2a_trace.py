"""Compute kernel + alltoall for a kernel trace: one gated release of the kernel on one stream and the alltoall on another.
   torchrun --nproc_per_node=4 a2a_trace.py [--kernel gmm|gemm|attn_fwd|attn_bwd] [--mib 768] [--comm-first-us 0]
   Run under rocprofv3 with trace.sh."""
import argparse
import os
import sys
import torch
import torch.distributed as dist

p = argparse.ArgumentParser()
p.add_argument("--mib", type=int, default=768)
p.add_argument("--comm-first-us", type=float, default=0.0)
p.add_argument("--kernel", default="gmm")
p.add_argument("--episodes", type=int, default=8)
a = p.parse_args()

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
local = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local)
dev = torch.device("cuda", local)
comp = torch.cuda.Stream(device=dev)
comm = torch.cuda.Stream(device=dev)

sys.path.insert(0, os.environ.get("CCO_COMMON", "/home/xiaompen/minimax-poc/examples-phase2/training/common"))
import cco_bench
sys.argv = [sys.argv[0]]
ka = cco_bench.parse_args()
call = cco_bench.KERNELS[a.kernel](ka, dev, torch.Generator(device=dev).manual_seed(rank))["call"]

n = (a.mib << 20) // 2
inp = torch.randn(n, device=dev, dtype=torch.bfloat16)
out = torch.empty_like(inp)

cyc = 2200                                  # ~cycles per us of torch.cuda._sleep on gfx1250
ev = lambda: torch.cuda.Event(enable_timing=True)
res = []
for ep in range(a.episodes + 3):
    dist.barrier()
    torch.cuda.synchronize()
    torch.cuda._sleep(int(300 * cyc))
    g_comm = ev(); g_comm.record()
    if a.comm_first_us > 0:
        torch.cuda._sleep(int(a.comm_first_us * cyc))
    g_comp = ev(); g_comp.record()
    comm.wait_event(g_comm)
    comp.wait_event(g_comp)
    with torch.cuda.stream(comm):
        dist.all_to_all_single(out, inp, async_op=True).wait()
        e_comm = ev(); e_comm.record()
    with torch.cuda.stream(comp):
        call()
        e_comp = ev(); e_comp.record()
    torch.cuda.synchronize()
    if ep >= 3:
        res.append((g_comm.elapsed_time(e_comp), g_comm.elapsed_time(e_comm)))
if rank == 0:
    import statistics
    print(f"{a.kernel} + alltoall {a.mib} MiB, comm first {a.comm_first_us} us: comp end {statistics.median(r[0] for r in res):.3f} ms,"
          f" comm end {statistics.median(r[1] for r in res):.3f} ms (from the comm release)", flush=True)
dist.barrier()
torch.cuda.synchronize()
