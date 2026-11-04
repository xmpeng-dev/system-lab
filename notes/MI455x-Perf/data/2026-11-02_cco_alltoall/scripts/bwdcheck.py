"""attn_bwd of a preset against FP32 autograd on sequence 0, kv head 0 and its q heads."""
import sys, torch
sys.path.insert(0, "/home/xiaompen/minimax-poc/examples-phase2/training/common")
import cco_bench as c
impl = sys.argv[2] if len(sys.argv) > 2 else "aiter"
sys.argv = [sys.argv[0], "--model", sys.argv[1]]
a = c.parse_args(); dev = torch.device("cuda", 0)
at = c._Attn(a, dev, torch.Generator(device=dev).manual_seed(0), impl)
at.bwd(); torch.cuda.synchronize()
B, L, hq, hkv, d, _ = at.shape
g = hq // hkv
q = at.q[0, :, :g].float().transpose(0, 1).requires_grad_()      # [g, L, d]
k = at.k[0, :, 0].float().requires_grad_()                       # [L, d]
v = at.v[0, :, 0].float().requires_grad_()
s = q @ k.t() * at.scale
if at.causal:
    s = s.masked_fill(torch.ones(L, L, device=dev, dtype=torch.bool).triu(1), float("-inf"))
o = torch.softmax(s, -1) @ v
o.backward(at.do[0, :, :g].float().transpose(0, 1))
for name, got, ref in (("dq", at.dq[0, :, :g].transpose(0, 1), q.grad), ("dk", at.dk[0, :, 0], k.grad), ("dv", at.dv[0, :, 0], v.grad)):
    print(a.model, impl, name, f"{c._err(got, ref):.1e}")
