"""Ulysses benches with the GLM-4.7 / MiniMax-M2.5 shapes and env overrides for the backward fused kernels.

TURBO=/tmp/turbo_<c> PYTHONPATH=/tmp/turbo_<c> torchrun --nproc_per_node 8 uly_bench.py overlap --model glm4.7 --seq 16384
TURBO=/tmp/turbo_<c> PYTHONPATH=/tmp/turbo_<c> torchrun --nproc_per_node 8 uly_bench.py layer --model glm4.7 --seq 16384 --fused

ULY_QKV_PUSH / ULY_QKV_BM / ULY_QKV_WGRAD_AT  bwd QKV A2AGemm launch parameters (only when it carries the wgrad)
ULY_O_WGRAD_AT                                 bwd o GemmA2A wgrad position (only when it carries the wgrad)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.environ["TURBO"], "benchmark/ops/training"))

import primus_turbo.pytorch.ops.attention.ulysses_fused_a2a as uly  # noqa: E402

if os.environ.get("RANK", "0") == "0":
    print(f"[uly_bench] {uly.__file__}", flush=True)

_a2a_gemm, _gemm_a2a = uly.ulysses_a2a_gemm, uly.ulysses_gemm_a2a


def _knob(name):
    value = os.environ.get(name)
    return None if value is None else int(value)


def a2a_gemm(*args, **kwargs):
    if kwargs.get("wgrad") is not None:
        for key, env in (("num_push", "ULY_QKV_PUSH"), ("BM", "ULY_QKV_BM"), ("wgrad_at", "ULY_QKV_WGRAD_AT")):
            if _knob(env) is not None:
                kwargs[key] = _knob(env)
    return _a2a_gemm(*args, **kwargs)


def gemm_a2a(*args, **kwargs):
    if kwargs.get("wgrad") is not None and _knob("ULY_O_WGRAD_AT") is not None:
        kwargs["wgrad_at"] = _knob("ULY_O_WGRAD_AT")
    return _gemm_a2a(*args, **kwargs)


uly.ulysses_a2a_gemm, uly.ulysses_gemm_a2a = a2a_gemm, gemm_a2a

GLM = dict(hidden=5120, hq=96, hkv=8, d=128)
MINIMAX = dict(hidden=3072, hq=48, hkv=8, d=128)

if __name__ == "__main__":
    which = sys.argv.pop(1)
    if which == "overlap":
        import bench_ulysses_overlap as bench

        bench.MODELS.update({"glm4.7": GLM, "minimax-m2.5": MINIMAX})
    else:
        import bench_ulysses_a2a as bench

        bench.MODEL_PARAMS_TABLE["glm4.7"] = dict(GLM, ffn=9 * 1536, glu=True, causal=True)
        bench.MODEL_PARAMS_TABLE["minimax-m2.5"] = dict(MINIMAX, ffn=8 * 1536, glu=True, causal=True)
    bench.main()
