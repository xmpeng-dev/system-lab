"""Cross-node all_to_all_single vs node-local groups, timed as cco_bench.py times a collective alone: async calls on
a side stream behind one start event, back to back (steady state, mean per call) and one call per episode (median).

torchrun --nnodes N --node_rank R --nproc_per_node L --master_addr A --master_port P a2a_xnode.py out.json
env: SIZES (MiB per rank), ITERS (back-to-back calls), EPS (single-call episodes)
"""
import json
import os
import socket
import statistics
import sys

import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
local, lw = int(os.environ["LOCAL_RANK"]), int(os.environ["LOCAL_WORLD_SIZE"])
torch.cuda.set_device(local)
node = rank // lw
groups = [dist.new_group(list(range(n * lw, (n + 1) * lw))) for n in range(world // lw)]
sizes = [int(s) for s in os.environ.get("SIZES", "12,48,96,192,384").split(",")]
iters, eps = int(os.environ.get("ITERS", "20")), int(os.environ.get("EPS", "10"))
side = torch.cuda.Stream()
ev = lambda: torch.cuda.Event(enable_timing=True)


def window(y, x, group, n):
    dist.barrier()
    torch.cuda.synchronize()
    g1, end = ev(), ev()
    g1.record()
    side.wait_event(g1)
    with torch.cuda.stream(side):
        for _ in range(n):
            dist.all_to_all_single(y, x, group=group, async_op=True).wait()
        end.record(side)
    torch.cuda.synchronize()
    return g1.elapsed_time(end) / n


def all_max(v):
    t = torch.tensor([float(v)], dtype=torch.float64, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def bench(mib, group, gsize, grank):
    n = (mib << 20) // 2 // gsize * gsize
    chunk = n // gsize
    x = torch.empty(n, dtype=torch.bfloat16, device="cuda")
    for d in range(gsize):
        x[d * chunk:(d + 1) * chunk] = grank * gsize + d
    y = torch.empty_like(x)
    dist.all_to_all_single(y, x, group=group)
    want = torch.cat([torch.full((chunk,), s * gsize + grank, dtype=torch.bfloat16, device="cuda")
                      for s in range(gsize)])
    bad = all_max(0 if torch.equal(y, want) else 1)
    for _ in range(3):
        window(y, x, group, 1)
    steady = all_max(window(y, x, group, iters))
    single = all_max(statistics.median(window(y, x, group, 1) for _ in range(eps)))
    return steady, single, bad == 0, n * 2


rows = []
for mib in sizes:
    for scope in ("global", "local"):
        group, gsize, grank = (None, world, rank) if scope == "global" else (groups[node], lw, local)
        steady, single, ok, b = bench(mib, group, gsize, grank)
        bus = lambda t: round(b * (gsize - 1) / gsize / (t * 1e-3) / 1e9, 1)
        row = dict(scope=scope, ranks=gsize, mib_per_rank=mib, steady_ms=round(steady, 4), single_ms=round(single, 4),
                   busbw_GBps=bus(steady), correct=ok)
        if scope == "global":
            # bytes one node sends to the other nodes per call, over the steady time
            row["xnode_GBps_per_node"] = round(b * (world - lw) / world * lw / (steady * 1e-3) / 1e9, 1)
        rows.append(row)
        if rank == 0:
            print(json.dumps(row), flush=True)

hosts = [None] * world
dist.all_gather_object(hosts, (socket.gethostname(), os.environ.get("HIP_VISIBLE_DEVICES")))
if rank == 0:
    print("hosts", hosts, flush=True)
    if len(sys.argv) > 1:
        meta = dict(hosts=hosts, iters=iters, eps=eps, rccl=".".join(map(str, torch.cuda.nccl.version())),
                    torch=torch.__version__,
                    env={k: v for k, v in os.environ.items() if k.startswith(("NCCL_", "RCCL_", "HIP_VIS"))})
        json.dump(dict(meta=meta, rows=rows), open(sys.argv[1], "w"), indent=1)
dist.destroy_process_group()
