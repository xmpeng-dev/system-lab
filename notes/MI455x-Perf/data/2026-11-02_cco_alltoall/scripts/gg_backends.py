"""One GPU: the gmm shape (72 x 912 x 6144 -> 2048) on every grouped GEMM path; time + FP32 check."""
import sys, torch
from primus_turbo.pytorch.kernels.grouped_gemm.grouped_gemm_impl import grouped_gemm_impl
from primus_turbo.pytorch.core.backend import BackendType
from primus_turbo.flydsl.grouped_gemm.grouped_gemm_bf16_kernel_gfx1250 import build_m_tile_map, grouped_gemm_bf16_nt_flydsl_kernel
E, R, K, N = 72, 912, 6144, 2048
dev = "cuda"
g = torch.Generator(device=dev).manual_seed(0)
x = torch.randn(E * R, K, device=dev, dtype=torch.bfloat16, generator=g)
w_nt = torch.randn(E, N, K, device=dev, dtype=torch.bfloat16, generator=g)
w_nn = w_nt.transpose(1, 2).contiguous()
lens = torch.full((E,), R, device=dev, dtype=torch.int64)
offs = torch.arange(E + 1, device=dev, dtype=torch.int64) * R
offs32 = (torch.arange(1, E + 1, device=dev, dtype=torch.int32) * R)
out = torch.empty(E * R, N, device=dev, dtype=torch.bfloat16)
mt = build_m_tile_map(offs, 256)
paths = {
    "bmm(hipBLASLt)": lambda: torch.bmm(x.view(E, R, K), w_nn, out=out.view(E, R, N)),
    "turbo HIPBLASLT": lambda: grouped_gemm_impl(x, w_nt, lens, offs, trans_a=False, trans_b=True, num_cu=None, default_backend=BackendType.HIPBLASLT.value),
    "turbo HIPBLASLT presync": lambda: grouped_gemm_impl(x, w_nt, lens, offs, trans_a=False, trans_b=True, num_cu=None, default_backend=BackendType.HIPBLASLT.value, maybe_pre_sync=True),
    "flydsl": lambda: grouped_gemm_bf16_nt_flydsl_kernel(x, w_nt, offs, BLOCK_M=256, m_tiles=mt, out=out),
    "torch._grouped_mm": lambda: torch._grouped_mm(x, w_nt.transpose(1, 2), offs=offs32),
}
only = sys.argv[1:] or list(paths)
ref = (x[:2 * R].float().view(2, R, K) @ w_nn[:2].float())
for name in only:
    f = paths[name]
    try:
        for _ in range(10): r = f()
        torch.cuda.synchronize()
        ts = []
        for _ in range(30):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(); r = f(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
        r = out if r is None else r
        got = r.reshape(E * R, N)[:2 * R].float().view(2, R, N)
        err = ((got - ref).norm() / ref.norm()).item()
        t = sorted(ts)[15]
        print(f"{name:26s} {t*1e3:6.0f} us {2*E*R*N*K/t/1e9:5.0f} TF  rel err {err:.1e}", flush=True)
    except Exception as e:
        print(f"{name:26s} FAILED {type(e).__name__}: {str(e)[:150]}", flush=True)
