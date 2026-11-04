import os, socket, torch, torch.distributed as dist
dist.init_process_group("nccl")
r, w = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
n = (192 << 20) // 2 // w * w
c = n // w
x = torch.cat([torch.full((c,), r * w + d, dtype=torch.bfloat16, device="cuda") for d in range(w)])
y = torch.empty_like(x)
dist.all_to_all_single(y, x)
want = torch.cat([torch.full((c,), s * w + r, dtype=torch.bfloat16, device="cuda") for s in range(w)])
ok = torch.tensor([int(torch.equal(y, want))], device="cuda")
s, e = torch.cuda.Event(True), torch.cuda.Event(True)
N = int(os.environ.get("ITERS", "200"))
for _ in range(50): dist.all_to_all_single(y, x)
torch.cuda.synchronize(); s.record()
for _ in range(N): dist.all_to_all_single(y, x)
e.record(); torch.cuda.synchronize()
t = torch.tensor([s.elapsed_time(e) / N], device="cuda")
dist.all_reduce(ok, op=dist.ReduceOp.MIN); dist.all_reduce(t, op=dist.ReduceOp.MAX)
if r == 0:
    print(f"{socket.gethostname()} alltoall 192MiB x{w}: correct={bool(ok.item())} {t.item():.3f} ms busbw {n*2*(w-1)/w/t.item()/1e6:.0f} GB/s")
dist.destroy_process_group()
