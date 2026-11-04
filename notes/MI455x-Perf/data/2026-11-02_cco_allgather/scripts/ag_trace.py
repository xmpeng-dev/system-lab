"""allgather alone at each size, then compute + allgather launched in order (compute first, as cco_bench m2), for
a rocprofv3 kernel trace (ag_trace.sh). A spin kernel (torch.cuda._sleep) marks the start of every block:
blocks 0..len(sizes)-1 are the allgather alone, then one block per (kernel, pair size).

   torchrun --nproc_per_node=4 ag_trace.py [--sizes-mib 1024,512,256,128,64,16] [--kernels gemm,gmm_flydsl]
                                           [--pair-mib 1024,64] [--calls 6]
"""
import argparse
import os
import sys

import torch
import torch.distributed as dist

p = argparse.ArgumentParser()
p.add_argument("--sizes-mib", default="1024,512,256,128,64,16")
p.add_argument("--kernels", default="gemm,gmm_flydsl")
p.add_argument("--pair-mib", default="1024,64")
p.add_argument("--calls", type=int, default=6)
a = p.parse_args()

sys.path.insert(0, os.environ.get("CCO_COMMON", "/home/xiaompen/minimax-poc/examples-phase2/training/common"))
import cco_bench  # noqa: E402

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
local = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local)
dev = torch.device("cuda", local)
comp, comm = torch.cuda.Stream(device=dev), torch.cuda.Stream(device=dev)
sys.argv = [sys.argv[0]]
ka = cco_bench.parse_args()


def mark():
    dist.barrier()
    torch.cuda.synchronize()
    torch.cuda._sleep(20000)
    torch.cuda.synchronize()


colls = {m: cco_bench.Collective("allgather", m << 20, torch.bfloat16, rank, world, dev)
         for m in sorted({int(s) for s in (a.sizes_mib + "," + a.pair_mib).split(",") if s})}
for c in colls.values():
    for _ in range(3):
        c.op().wait()
torch.cuda.synchronize()

for m in [int(s) for s in a.sizes_mib.split(",")]:
    mark()
    for _ in range(a.calls):
        dist.barrier()
        with torch.cuda.stream(comm):
            colls[m].op().wait()
        torch.cuda.synchronize()

ev = lambda: torch.cuda.Event(enable_timing=True)
for k in a.kernels.split(","):
    call = cco_bench.KERNELS[k](ka, dev, torch.Generator(device=dev).manual_seed(1729 + rank))["call"]
    for _ in range(3):
        call()
    for m in [int(s) for s in a.pair_mib.split(",")]:
        mark()
        res = []
        for _ in range(a.calls):
            dist.barrier()
            torch.cuda.synchronize()
            g = ev()
            g.record()
            comp.wait_event(g)
            comm.wait_event(g)
            with torch.cuda.stream(comp):
                call()
                ec = ev()
                ec.record()
            with torch.cuda.stream(comm):
                colls[m].op().wait()
                em = ev()
                em.record()
            torch.cuda.synchronize()
            res.append((g.elapsed_time(ec), g.elapsed_time(em)))
        if rank == 0:
            r = sorted(res)[len(res) // 2]
            print(f"{k} + allgather {m} MiB: comp end {r[0]:.3f} ms, comm end {r[1]:.3f} ms", flush=True)
dist.barrier()
torch.cuda.synchronize()
dist.destroy_process_group()
