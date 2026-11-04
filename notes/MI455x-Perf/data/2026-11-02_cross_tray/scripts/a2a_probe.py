"""Cross-node alltoall probe: 8 ranks (2 nodes x 4) vs node-local 4-rank groups, same bytes per rank.

torchrun --nnodes 2 --node_rank R --nproc_per_node 4 --master_addr A --master_port P a2a_probe.py out.json
"""
import json
import os
import socket
import sys

import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
local, lw = int(os.environ["LOCAL_RANK"]), int(os.environ["LOCAL_WORLD_SIZE"])
torch.cuda.set_device(local)
node = rank // lw
groups = [dist.new_group(list(range(n * lw, (n + 1) * lw))) for n in range(world // lw)]
local_group = groups[node]

sizes = [int(s) for s in os.environ.get("SIZES", "12,48,96,192,384").split(",")]
iters = int(os.environ.get("ITERS", "20"))


def bench(nbytes, group, gsize, grank):
    n = nbytes // 2 // gsize * gsize
    chunk = n // gsize
    x = torch.empty(n, dtype=torch.bfloat16, device="cuda")
    for d in range(gsize):
        x[d * chunk:(d + 1) * chunk] = grank * gsize + d
    y = torch.empty_like(x)
    dist.all_to_all_single(y, x, group=group)
    want = torch.cat([torch.full((chunk,), s * gsize + grank, dtype=torch.bfloat16, device="cuda") for s in range(gsize)])
    ok = bool(torch.equal(y, want))
    for _ in range(3):
        dist.all_to_all_single(y, x, group=group)
    torch.cuda.synchronize()
    dist.barrier()
    evs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    for s, e in evs:
        s.record()
        dist.all_to_all_single(y, x, group=group)
        e.record()
    torch.cuda.synchronize()
    ts = sorted(s.elapsed_time(e) for s, e in evs)
    return ts[len(ts) // 2], ok, n * 2


def gather_max(v):
    t = torch.tensor([v], dtype=torch.float64, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def gather_all_ok(ok):
    t = torch.tensor([0 if ok else 1], device="cuda")
    dist.all_reduce(t)
    return t.item() == 0


rows = []
for mib in sizes:
    for scope in ("global", "local"):
        if scope == "global":
            t, ok, b = bench(mib << 20, None, world, rank)
            gsize = world
        else:
            t, ok, b = bench(mib << 20, local_group, lw, local)
            gsize = lw
        t, ok = gather_max(t), gather_all_ok(ok)
        busbw = b * (gsize - 1) / gsize / (t * 1e-3) / 1e9
        row = dict(scope=scope, ranks=gsize, mib_per_rank=mib, ms=round(t, 4), busbw_GBps=round(busbw, 1), correct=ok)
        if scope == "global":
            row["cross_node_GBps_per_node"] = round(b * (world - lw) / world * lw / (t * 1e-3) / 1e9, 1)
        rows.append(row)
        if rank == 0:
            print(json.dumps(row), flush=True)

if rank == 0 and len(sys.argv) > 1:
    meta = dict(hosts=None, env={k: v for k, v in os.environ.items() if k.startswith(("NCCL_", "RCCL_"))},
                rccl=".".join(map(str, torch.cuda.nccl.version())), torch=torch.__version__)
    json.dump(dict(meta=meta, rows=rows), open(sys.argv[1], "w"), indent=1)
hosts = [None] * world
dist.all_gather_object(hosts, socket.gethostname())
if rank == 0:
    print("hosts", sorted(set(hosts)), flush=True)
dist.destroy_process_group()
