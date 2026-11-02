"""Compute + communication overlap without data dependency: a compute kernel on one stream, a collective on
another, run alone and at the same time; how much each slows the other and how much of the shorter one hides.

Cases (examples-phase2/training/README_CN.md section 3), one directory with a run.sh per collective:

  training/cco-alltoall       alltoall      + gmm | attn_fwd | attn_bwd | gemm   (EP, steady state of TBO)
  training/cco-allgather      allgather     + gemm | gmm | attn_fwd              (FSDP parameter prefetch)
  training/cco-reducescatter  reducescatter + wgrad | gmm_wgrad | attn_bwd       (gradient reduce-scatter)

Every measurement is an episode: a GPU sleep on the default stream holds the compute and communication streams
until all their work is enqueued, then releases both at the same moment (--stagger-us releases the collective
later, or earlier if negative). Two modes, schedules interleaved window by window as in opbench:

  m1  steady-state interference: the victim (compute or collective) runs back to back for one window while the
      other runs back to back underneath it for longer; victim time per call against its time alone.
  m2  paired: one collective and N compute calls, N such that N * T_comp ~ ratio * T_comm; time from release
      until both are done (T_ovl) against each alone; overlap efficiency
      E = (T_comp + T_comm - T_ovl) / min(T_comp, T_comm).

    torchrun --nproc_per_node=4 cco_bench.py --comm alltoall --compute gmm,attn_fwd -o e1.json
"""

import argparse
import ctypes
import datetime
import json
import math
import os
import statistics
import sys
import threading
import time

import torch
import torch.distributed as dist

BUSBW = {"alltoall": lambda n: (n - 1) / n, "allgather": lambda n: (n - 1) / n,
         "reducescatter": lambda n: (n - 1) / n}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--comm", default="alltoall", help="comma list of: " + ", ".join(BUSBW))
    p.add_argument("--sizes-mib", default="768,384,256,128,64",
                   help="per-rank size: alltoall send buffer, allgather output, reducescatter input "
                        "(768 MiB = an EP4 dispatch of 8192 tokens x top-8 x 6144 x BF16)")
    p.add_argument("--comm-dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--compute", default="gmm,attn_fwd,attn_bwd,gemm", help="comma list of: " + ", ".join(KERNELS))
    p.add_argument("--modes", default="m1,m2")
    p.add_argument("--ratios", default="0.5,1,2", help="m2: N * T_comp / T_comm")
    p.add_argument("--stagger-us", type=float, default=0.0,
                   help="release the collective this long after the compute (negative: before)")
    p.add_argument("--cu-reserve", type=int, default=0,
                   help="CUs masked off the compute stream (hipExtStreamCreateWithCUMask), left to the collective")
    p.add_argument("--comm-priority", choices=["default", "high"], default="default",
                   help="priority of the RCCL stream (ProcessGroupNCCL is_high_priority_stream)")
    p.add_argument("--gemm", default="8192,6144,24576", help="gemm / wgrad M,K,N (opbench T8192_K6144_N24576)")
    p.add_argument("--gmm", default="72,912,6144,2048",
                   help="gmm / gmm_wgrad experts,rows,K,N (EP4 local gate+up: 288 experts / 4, 8192 x 8 / 72 rows)")
    p.add_argument("--attn", default="2,4096,64,128", help="attn batch,seqlen,heads,head_dim (non-causal)")
    p.add_argument("--warmup-ms", type=float, default=1000.0, help="per case, before sizing the windows")
    p.add_argument("--target-ms", type=float, default=300.0, help="length of one timed window")
    p.add_argument("--rounds", type=int, default=10, help="timed windows per schedule")
    p.add_argument("--pg-timeout-s", type=float, default=180.0,
                   help="process group watchdog: a rank whose GPU work stalls this long aborts the run")
    p.add_argument("--no-telemetry", action="store_true")
    p.add_argument("-o", "--output", default=None, help="JSON result file (rank 0)")
    return p.parse_args()


# ------------------------------------------------------------------------------------ compute


def _randn(*shape, dev, gen):
    """randn, as in opbench (all-zero inputs draw less power and clock higher)."""
    return torch.randn(*shape, device=dev, dtype=torch.bfloat16, generator=gen)


def k_gemm(a, dev, gen):
    M, K, N = map(int, a.gemm.split(","))
    x, w = _randn(M, K, dev=dev, gen=gen), _randn(N, K, dev=dev, gen=gen)
    out = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
    ref = lambda: x[:64].float() @ w.float().t()
    return dict(call=lambda: torch.mm(x, w.t(), out=out), out=out, flops=2 * M * N * K,
                check=lambda: (out[:64], ref()), label=f"x[{M},{K}] @ W[{N},{K}]^T")


def k_wgrad(a, dev, gen):
    M, K, N = map(int, a.gemm.split(","))
    dy, x = _randn(M, N, dev=dev, gen=gen), _randn(M, K, dev=dev, gen=gen)
    out = torch.empty(N, K, device=dev, dtype=torch.bfloat16)
    ref = lambda: dy[:, :64].float().t() @ x.float()
    return dict(call=lambda: torch.mm(dy.t(), x, out=out), out=out, flops=2 * M * N * K,
                check=lambda: (out[:64], ref()), label=f"dY[{M},{N}]^T @ x[{M},{K}]")


def k_gmm(a, dev, gen):
    E, R, K, N = map(int, a.gmm.split(","))
    x, w = _randn(E, R, K, dev=dev, gen=gen), _randn(E, K, N, dev=dev, gen=gen)
    out = torch.empty(E, R, N, device=dev, dtype=torch.bfloat16)
    ref = lambda: x[:2].float() @ w[:2].float()
    return dict(call=lambda: torch.bmm(x, w, out=out), out=out, flops=2 * E * R * N * K,
                check=lambda: (out[:2], ref()), label=f"bmm x[{E},{R},{K}] @ W[{E},{K},{N}]")


def k_gmm_wgrad(a, dev, gen):
    E, R, K, N = map(int, a.gmm.split(","))
    x, dh = _randn(E, R, K, dev=dev, gen=gen), _randn(E, R, N, dev=dev, gen=gen)
    dw = torch.zeros(E, K, N, device=dev, dtype=torch.float32)
    xt = x.transpose(1, 2)
    call = lambda: torch.baddbmm(dw, xt, dh, out_dtype=torch.float32, out=dw)

    def check():  # one product from zero, against FP32
        dw.zero_()
        call()
        return dw[:2], x[:2].float().transpose(1, 2) @ dh[:2].float()

    return dict(call=call, out=None, flops=2 * E * R * N * K, check=check,
                label=f"baddbmm FP32 dW[{E},{K},{N}] += x^T dH (R={R})")


class _Attn:
    def __init__(self, a, dev, gen):
        import aiter

        self.aiter = aiter
        B, L, h, d = map(int, a.attn.split(","))
        if d != 128:
            raise SystemExit("attn: the aiter asm kernels used here are hd128 only")
        self.q, self.k, self.v, self.do = (_randn(B, L, h, d, dev=dev, gen=gen) for _ in range(4))
        self.scale, self.shape = d ** -0.5, (B, L, h, d)
        self.dq, self.dk, self.dv = (torch.empty_like(self.q) for _ in range(3))
        self.rng = torch.empty(2, dtype=torch.int64, device=dev)
        self.fwd()
        self.fwd_out = [t.clone() for t in self.res]

    def fwd(self):
        self.res = self.aiter.fmha_fwd_with_sink_asm(self.q, self.k, self.v, self.scale, False, True, None, None)

    def bwd(self):
        self.dq.zero_()  # the asm kernel accumulates dq with atomics
        out, lse = self.fwd_out
        self.aiter.fmha_v3_bwd(self.do, self.q, self.k, self.v, out, lse, 0.0, self.scale, False, -1, -1, False,
                               True, 1, self.dq, self.dk, self.dv, None, self.rng, None)

    def ref(self, b=0, rows=64):
        """FP32 attention of the first `rows` queries of sequence b, all heads."""
        q, k, v = (t[b].float().transpose(0, 1) for t in (self.q, self.k, self.v))   # [h, L, d]
        p = torch.softmax(q[:, :rows] @ k.transpose(1, 2) * self.scale, -1)
        return (p @ v).transpose(0, 1)                                                 # [rows, h, d]


def k_attn_fwd(a, dev, gen):
    at = _Attn(a, dev, gen)
    B, L, h, d = at.shape
    return dict(call=at.fwd, out=None, out_fn=lambda: at.res[0], flops=4 * B * L * L * h * d,
                check=lambda: (at.res[0][0, :64], at.ref()), label=f"aiter asm fwd B{B} L{L} h{h} d{d}")


def k_attn_bwd(a, dev, gen):
    at = _Attn(a, dev, gen)
    B, L, h, d = at.shape
    return dict(call=at.bwd, out=None, out_fn=lambda: at.dk, flops=10 * B * L * L * h * d, check=None,
                atomic=True, label=f"aiter asm bwd B{B} L{L} h{h} d{d}")


KERNELS = dict(gemm=k_gemm, wgrad=k_wgrad, gmm=k_gmm, gmm_wgrad=k_gmm_wgrad, attn_fwd=k_attn_fwd,
               attn_bwd=k_attn_bwd)


# -------------------------------------------------------------------------------- collectives


class Collective:
    """One out-of-place collective with a data pattern that every call reproduces exactly."""

    def __init__(self, kind, size_bytes, dtype, rank, world, dev):
        self.kind, self.rank, self.world = kind, rank, world
        es = torch.tensor([], dtype=dtype).element_size()
        n = size_bytes // es // world * world                   # elements, divisible into world chunks
        self.bytes = n * es
        chunk = n // world
        r = rank
        if kind == "alltoall":
            self.inp = torch.cat([torch.full((chunk,), r * world + j + 1, dtype=dtype, device=dev)
                                  for j in range(world)])
            self.out = torch.empty(n, dtype=dtype, device=dev)
            self.want = torch.cat([torch.full((chunk,), j * world + r + 1, dtype=dtype, device=dev)
                                   for j in range(world)])
            self.op = lambda: dist.all_to_all_single(self.out, self.inp, async_op=True)
        elif kind == "allgather":
            self.inp = torch.full((chunk,), r + 1, dtype=dtype, device=dev)
            self.out = torch.empty(n, dtype=dtype, device=dev)
            self.want = torch.cat([torch.full((chunk,), j + 1, dtype=dtype, device=dev) for j in range(world)])
            self.op = lambda: dist.all_gather_into_tensor(self.out, self.inp, async_op=True)
        elif kind == "reducescatter":
            self.inp = torch.cat([torch.full((chunk,), (r + 1) * (j + 1), dtype=dtype, device=dev)
                                  for j in range(world)])
            self.out = torch.empty(chunk, dtype=dtype, device=dev)
            self.want = torch.full((chunk,), (r + 1) * world * (world + 1) // 2, dtype=dtype, device=dev)
            self.op = lambda: dist.reduce_scatter_tensor(self.out, self.inp, async_op=True)
        else:
            raise SystemExit(f"unknown collective {kind}")

    def check(self):
        return bool(torch.equal(self.out, self.want))


# ----------------------------------------------------------------------------------- streams


def _hip():
    with open("/proc/self/maps") as f:
        path = next(line.split()[-1] for line in f if "libamdhip64.so" in line)
    return ctypes.CDLL(path)


def cu_masked_stream(dev, reserve):
    """A stream whose kernels may run only on the lowest-numbered (CU count - reserve) CUs."""
    n = torch.cuda.get_device_properties(dev).multi_processor_count
    keep = n - reserve
    if not 0 < keep <= n:
        raise SystemExit(f"--cu-reserve {reserve}: the GPU has {n} CUs")
    words = [0] * math.ceil(n / 32)
    for i in range(keep):
        words[i // 32] |= 1 << (i % 32)
    hip = _hip()
    s = ctypes.c_void_p()
    rc = hip.hipExtStreamCreateWithCUMask(ctypes.byref(s), ctypes.c_uint32(len(words)),
                                          (ctypes.c_uint32 * len(words))(*words))
    if rc != 0:
        raise SystemExit(f"hipExtStreamCreateWithCUMask failed: {rc}")
    return torch.cuda.ExternalStream(s.value, device=dev), keep


class Episodes:
    GATE = 40

    def __init__(self, a, dev):
        self.dev = dev
        self.comm = torch.cuda.Stream(device=dev)
        if a.cu_reserve:
            self.comp, self.comp_cus = cu_masked_stream(dev, a.cu_reserve)
        else:
            self.comp, self.comp_cus = torch.cuda.Stream(device=dev), None
        self.stagger_us = a.stagger_us
        torch.cuda._sleep(1000)
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        torch.cuda._sleep(2_000_000)
        e.record()
        e.synchronize()
        self.cycles_per_us = 2_000_000 / (s.elapsed_time(e) * 1e3)
        self.us_per_launch = 30.0     # host enqueue time per call, grown whenever an episode is released early
        self.late = 0

    def _sleep_us(self, us):
        if us > 0:
            torch.cuda._sleep(int(us * self.cycles_per_us))

    def run(self, comp_call, n_comp, comm, n_comm, per_call=False, t_comp=1.0, t_comm=1.0):
        """Enqueue one episode; returns its events (read them after a synchronize with `times`).
        Calls are enqueued in the order they are expected to start (t_comp, t_comm: ms per call alone),
        so both streams have work queued all along; the release waits for the first GATE calls only,
        after that the host stays ahead of the GPU. per_call: an event after every call, for the m1 rates."""
        ev = lambda: torch.cuda.Event(enable_timing=True)
        order = sorted([(i * t_comp, 0) for i in range(n_comp)] + [(j * t_comm, 1) for j in range(n_comm)])
        self._sleep_us(200 + self.us_per_launch * min(len(order), self.GATE))
        g1 = ev()
        g1.record()
        g2 = g1
        if self.stagger_us and n_comp and n_comm:
            self._sleep_us(abs(self.stagger_us))
            g2 = ev()
            g2.record()
        gc, gm = (g1, g2) if self.stagger_us >= 0 else (g2, g1)
        cur = torch.cuda.current_stream()
        e = dict(g1=g1, gc=gc, gm=gm, comp_calls=[], comm_calls=[], late=False)
        self.comp.wait_event(gc)
        self.comm.wait_event(gm)
        for i, (_, which) in enumerate(order):
            stream = self.comm if which else self.comp
            with torch.cuda.stream(stream):
                if which:
                    comm.op().wait()          # the RCCL stream waits on this stream when the call is enqueued
                else:
                    comp_call()
                d = torch.cuda.Event()
                d.record()
                e.setdefault("dbg", []).append(("comm" if which else "comp", d))
                if per_call:
                    e["comm_calls" if which else "comp_calls"].append(ev())
                    e["comm_calls" if which else "comp_calls"][-1].record()
            if i == min(len(order), self.GATE) - 1 and g1.query():   # released before the first calls were in
                e["late"] = True
        for s, stream, n in (("comp", self.comp, n_comp), ("comm", self.comm, n_comm)):
            if n:
                e[s] = ev()
                e[s].record(stream)
                cur.wait_event(e[s])
        e["done"] = ev()
        e["done"].record()
        if e["late"]:
            self.late += 1
        return e

    def grow_gate(self):
        """Lengthen the gate after a window with early releases; called on every rank together, because a rank
        released earlier than the others waits inside the collective and that wait counts as communication."""
        self.us_per_launch = min(2000.0, self.us_per_launch * 1.5)

    @staticmethod
    def times(e):
        rel = lambda x: e["g1"].elapsed_time(x)
        t = dict(ovl=rel(e["done"]))
        for s, g in (("comp", e["gc"]), ("comm", e["gm"])):
            if s in e:
                t[s] = g.elapsed_time(e[s])
        if e["comp_calls"] and e["comm_calls"]:
            # rates while both streams are busy: the calls that start after both are released and end
            # before the first stream runs out of work
            marks = {s: [rel(e[g])] + [rel(x) for x in e[f"{s}_calls"]] for s, g in (("comp", "gc"), ("comm", "gm"))}
            lo, hi = max(m[0] for m in marks.values()), min(m[-1] for m in marks.values())
            for s, m in marks.items():
                d = [b - a for a, b in zip(m, m[1:]) if a >= lo and b <= hi]
                t[f"{s}_under"] = statistics.mean(d) if d else float("nan")
                t[f"{s}_under_calls"] = len(d)
        return t


# ------------------------------------------------------------------------------------ timing


def _all_max(vals):
    t = torch.tensor([float(v) for v in vals], device="cuda", dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.tolist()


class Case:
    """One compute kernel x one collective: sizes its schedules from the times alone, then measures
    window by window. Every window returns {metric: ms}, the max over ranks."""

    def __init__(self, a, ep, kern, coll):
        self.a, self.ep, self.k, self.c = a, ep, kern, coll
        self.t_comp = self.t_comm = 1.0

    def window(self, n_comp, n_comm, episodes=1, per_call=False):
        """Median over the episodes that were fully enqueued before their release, max over ranks."""
        for _ in range(10):
            dist.barrier()
            torch.cuda.synchronize()
            evs = [self.ep.run(self.k["call"], n_comp, self.c, n_comm, per_call, self.t_comp, self.t_comm)
                   for _ in range(episodes)]
            t_wait, dumped = time.time(), False
            while not evs[-1]["done"].query():
                time.sleep(0.0005)
                if not dumped and time.time() - t_wait > 60:
                    dumped = True
                    r = dist.get_rank()
                    for i, e in enumerate(evs):
                        if e["done"].query():
                            continue
                        st = {}
                        for s_, d in e.get("dbg", []):
                            st.setdefault(s_, []).append(d.query())
                        summ = {k: f"{sum(v)}/{len(v)} done, first pending #{v.index(False) if False in v else '-'}" for k, v in st.items()}
                        print(f"HANGDBG rank {r} window ({n_comp},{n_comm},{episodes}) episode {i}: gate {e['g1'].query()} "
                              f"comp_end {e['comp'].query() if 'comp' in e else '-'} comm_end {e['comm'].query() if 'comm' in e else '-'} "
                              f"{summ}", flush=True)
                        break
            torch.cuda.synchronize()
            ts = [Episodes.times(e) for e in evs if not e["late"]]
            none_left, any_late = _all_max([0 if ts else 1, any(e["late"] for e in evs)])
            if any_late:
                self.ep.grow_gate()
            if none_left == 0:
                break
        else:
            raise RuntimeError("every episode was released before it was enqueued")
        keys = sorted(ts[0])
        vals = [statistics.median(t[k] for t in ts) for k in keys]
        return dict(zip(keys, _all_max(vals)))

    def plan(self):
        a = self.a
        t0 = time.time()
        cal = []
        while time.time() - t0 < a.warmup_ms / 1e3 or len(cal) < 3:
            cal.append((self.window(10, 0)["comp"] / 10, self.window(0, 10)["comm"] / 10))
        # median over the second half: one stalled collective must not size every schedule
        cal = cal[len(cal) // 2:]
        t_comp = statistics.median(c[0] for c in cal)
        t_comm = statistics.median(c[1] for c in cal)
        self.t_comp, self.t_comm = t_comp, t_comm
        nv_comp = max(1, math.ceil(a.target_ms / t_comp))
        nv_comm = max(1, math.ceil(a.target_ms / t_comm))
        sched = {}
        if "m1" in a.modes:
            sched["m1 comp alone"] = (nv_comp, 0, 1, False)
            sched["m1 comm alone"] = (0, nv_comm, 1, False)
            sched["m1 both"] = (nv_comp, nv_comm, 1, True)
        if "m2" in a.modes:
            for r in [float(x) for x in a.ratios.split(",")]:
                n = max(1, round(r * t_comm / t_comp))
                eps = max(3, math.ceil(a.target_ms / (max(n * t_comp, t_comm) + 0.3)))
                sched[f"m2 x{n} comp"] = (n, 0, eps, False)
                sched[f"m2 x{n} both"] = (n, 1, eps, False)
                sched["m2 comm"] = (0, 1, max(3, math.ceil(a.target_ms / (t_comm + 0.3))), False)
        return sched, t_comp, t_comm

    def measure(self):
        sched, t_comp, t_comm = self.plan()
        rows = {name: [] for name in sched}
        spans = {name: [] for name in sched}
        for _ in range(self.a.rounds):
            for name, (nc, nm, eps, per_call) in sched.items():
                t0 = time.time()
                rows[name].append(self.window(nc, nm, eps, per_call))
                spans[name].append((t0, time.time()))
        res = {}
        for name, ws in rows.items():
            nc, nm, eps, _ = sched[name]
            r = dict(n_comp=nc, n_comm=nm, episodes=eps,
                     **{k: statistics.median(w[k] for w in ws) for k in ws[0]})
            r["windows"] = spans[name]
            res[name] = r
        return res, t_comp, t_comm


def summarize(res, a, kern, coll, world):
    """Per-call times alone and in m1, m2 per ratio: T_comp, T_comm, T_ovl, E, S, slowdowns."""
    out = {}
    if "m1 comp alone" in res:
        tc = res["m1 comp alone"]["comp"] / res["m1 comp alone"]["n_comp"]
        tm = res["m1 comm alone"]["comm"] / res["m1 comm alone"]["n_comm"]
        both = res["m1 both"]
        tc2, tm2 = both["comp_under"], both["comm_under"]
        out["m1"] = dict(t_comp=tc, t_comm=tm, t_comp_under=tc2, t_comm_under=tm2,
                         comp_slowdown=tc2 / tc, comm_slowdown=tm2 / tm,
                         calls_under=(int(both["comp_under_calls"]), int(both["comm_under_calls"])))
        out["m1"].update(_rates(tc, tm, tc2, tm2, kern, coll, world))
    m2 = {}
    for name, r in res.items():
        if name.startswith("m2 x") and name.endswith(" both"):
            n = r["n_comp"]
            Tc, Tm, To = res[f"m2 x{n} comp"]["comp"], res["m2 comm"]["comm"], r["ovl"]
            m2[n] = dict(n_comp=n, ratio=Tc / Tm, t_comp=Tc, t_comm=Tm, t_ovl=To, t_comp_in=r["comp"],
                         t_comm_in=r["comm"], efficiency=(Tc + Tm - To) / min(Tc, Tm), speedup=(Tc + Tm) / To,
                         ideal_speedup=(Tc + Tm) / max(Tc, Tm))
    if m2:
        out["m2"] = [m2[n] for n in sorted(m2)]
    return out


def _rates(tc, tm, tc2, tm2, kern, coll, world):
    r = {}
    if "flops" in kern:
        r.update(tflops=kern["flops"] / tc / 1e9, tflops_under=kern["flops"] / tc2 / 1e9)
    else:
        r.update(tbps=kern["bytes"] / tc / 1e9, tbps_under=kern["bytes"] / tc2 / 1e9)
    f = BUSBW[coll.kind](world)
    r.update(busbw=coll.bytes / tm / 1e6 * f, busbw_under=coll.bytes / tm2 / 1e6 * f)
    return r


# -------------------------------------------------------------------------------- telemetry


def _amdsmi():
    try:
        import rocm_sdk

        sys.path.insert(0, str(rocm_sdk.find_libraries("amd_smi")[0].parents[1] / "share" / "amd_smi"))
    except (ImportError, IndexError):
        sys.path.insert(0, os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "share", "amd_smi"))
    import amdsmi

    amdsmi.amdsmi_init()
    return amdsmi


class Telemetry:
    """gfx clock (mean over XCDs) and socket power of every local rank's GPU at ~20 Hz, sampled by
    local rank 0 of each node."""

    def __init__(self, bdfs):
        self.rows, self.stop = [], threading.Event()
        try:
            smi = _amdsmi()
            self.handles = {r: h for h in smi.amdsmi_get_processor_handles()
                            for r, b in bdfs.items() if smi.amdsmi_get_gpu_device_bdf(h).startswith(b)}
            self.smi = smi
        except Exception as exc:  # noqa: BLE001 -- telemetry is optional
            print(f"telemetry unavailable ({type(exc).__name__}: {exc})", flush=True)
            self.handles = {}
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.handles and not self.stop.is_set():
            for r, h in self.handles.items():
                try:
                    m = self.smi.amdsmi_get_gpu_metrics_info(h)
                    clks = [c for c in m.get("current_gfxclks", []) if isinstance(c, int)]
                    self.rows.append((time.time(), r, sum(clks) / len(clks) if clks else None,
                                      m.get("current_socket_power")))
                except Exception:  # noqa: BLE001
                    pass
            self.stop.wait(0.05)

    def close(self):
        """Stop sampling and shut amdsmi down; left running at interpreter exit it segfaults."""
        self.stop.set()
        self.thread.join()
        if self.handles:
            self.smi.amdsmi_shut_down()


def in_windows(rows, windows):
    mhz, watts = [], []
    for t, _, c, p in rows:
        if any(a <= t <= b for a, b in windows):
            if isinstance(c, (int, float)):
                mhz.append(c)
            if isinstance(p, (int, float)):
                watts.append(p)
    med = lambda v: statistics.median(v) if v else None
    return dict(gfx_mhz=med(mhz), power_w=med(watts))


# ------------------------------------------------------------------------------------- main


def environment():
    knobs = {k: v for k, v in sorted(os.environ.items())
             if k.startswith(("NCCL_", "RCCL_", "HSA_", "HIP_", "TORCH_NCCL", "GPU_"))}
    return dict(torch=torch.__version__, hip=torch.version.hip, rccl=".".join(map(str, torch.cuda.nccl.version())),
                image=os.environ.get("DOCKER_IMAGE"), env=knobs)


def _err(got, ref):
    got, ref = got.float(), ref.float()
    rms = ref.square().mean().sqrt().clamp_min(1e-12)
    return ((got - ref).square().mean().sqrt() / rms).item()


def main():
    a = parse_args()
    opts = dist.ProcessGroupNCCL.Options()
    opts.is_high_priority_stream = a.comm_priority == "high"
    dist.init_process_group("nccl", pg_options=opts, timeout=datetime.timedelta(seconds=a.pg_timeout_s))
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    # before the first collective: a CU-masked stream created after RCCL sets up its communicator adds ~0.9 ms to
    # every collective released by an event that the masked stream also waits on
    ep = Episodes(a, dev)
    p = torch.cuda.get_device_properties(dev)
    me = dict(host=os.uname().nodename, local=local, bdf=f"{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}")
    ranks = [None] * world
    dist.all_gather_object(ranks, me)
    dtype = torch.bfloat16 if a.comm_dtype == "bf16" else torch.float32
    comms = [c for c in a.comm.split(",") if c]
    sizes = [int(s) for s in a.sizes_mib.split(",") if s]
    computes = [c for c in a.compute.split(",") if c]
    for c in computes:
        if c not in KERNELS:
            raise SystemExit(f"unknown compute {c}; choose from {', '.join(KERNELS)}")

    tel = None
    if local == 0 and not a.no_telemetry:
        tel = Telemetry({r: i["bdf"] for r, i in enumerate(ranks) if i["host"] == me["host"]})
    cases, failed = [], []
    for cname in computes:
        gen = torch.Generator(device=dev).manual_seed(1729 + rank)
        kern = KERNELS[cname](a, dev, gen)
        kern["call"]()
        torch.cuda.synchronize()
        if kern.get("check"):
            got, ref = kern["check"]()
            e = _err(got, ref)
            if not (e <= 0.01 and torch.isfinite(got).all()):
                failed.append((rank, f"{cname} vs fp32: relative L2 {e:.3g}"))
        out_fn = kern.get("out_fn") or (lambda: kern["out"])
        alone = out_fn().clone() if out_fn() is not None else None
        for kind in comms:
            for mib in sizes:
                coll = Collective(kind, mib << 20, dtype, rank, world, dev)
                coll.op().wait()
                torch.cuda.synchronize()
                if not coll.check():
                    failed.append((rank, f"{kind} {mib} MiB: wrong data alone"))
                ok = _all_max([len(failed)])[0] == 0
                res, summary = {}, {}
                if ok:
                    ep.late = 0
                    res, _, _ = Case(a, ep, kern, coll).measure()
                    summary = summarize(res, a, kern, coll, world)
                    if not coll.check():
                        failed.append((rank, f"{kind} {mib} MiB + {cname}: wrong data after the overlap runs"))
                    if alone is not None:
                        now = out_fn()
                        same = torch.equal(now, alone)
                        e = _err(now, alone)
                        tol = 1e-3 if kern.get("atomic") else 0.0
                        if not same and e > tol:
                            failed.append((rank, f"{cname} with {kind} {mib} MiB: output changed (rel L2 {e:.3g})"))
                late = int(_all_max([ep.late])[0])
                cases.append(dict(compute=cname, compute_label=kern["label"], comm=kind, size_mib=mib,
                                  comm_bytes=coll.bytes, results=res, summary=summary, late_releases=late))
                if rank == 0 and summary:
                    print_case(cases[-1], flush=True)
                del coll
        del kern
        torch.cuda.empty_cache()

    if tel:
        tel.close()
    tel_rows = [None] * world
    dist.all_gather_object(tel_rows, tel.rows if tel else [])
    all_failed = [None] * world
    dist.all_gather_object(all_failed, failed)
    failed = [f for fs in all_failed for f in fs]
    if rank == 0:
        merged = [row for rs in tel_rows for row in rs]
        for c in cases:
            for r in c["results"].values():
                r.update(in_windows(merged, r.pop("windows")))
        out = dict(config=vars(a), world=world, ranks=ranks, environment=environment(),
                   comp_stream_cus=ep.comp_cus, failed=failed, cases=cases)
        print(f"\nvalidation: {'FAILED ' + str(failed) if failed else 'all passed'}")
        if a.output:
            os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
            with open(a.output, "w") as f:
                json.dump(out, f, indent=1)
            print(f"wrote {a.output}")
    dist.barrier()
    torch.cuda.synchronize()
    if os.environ.get("CCO_DESTROY_PG", "0") == "1":
        dist.destroy_process_group()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1 if failed else 0)


def print_case(c, flush=False):
    s = c["summary"]
    head = f"{c['compute']:9s} + {c['comm']:13s} {c['size_mib']:5d} MiB"
    if "m1" in s:
        m = s["m1"]
        rate = (f"{m['tflops']:5.0f}->{m['tflops_under']:5.0f} TF" if "tflops" in m
                else f"{m['tbps']:5.2f}->{m['tbps_under']:5.2f} TB/s")
        print(f"{head}  m1  comp {m['t_comp']:7.3f}->{m['t_comp_under']:7.3f} ms ({m['comp_slowdown']:.2f}x, {rate})"
              f"  comm {m['t_comm']:7.3f}->{m['t_comm_under']:7.3f} ms ({m['comm_slowdown']:.2f}x, busbw "
              f"{m['busbw']:5.0f}->{m['busbw_under']:5.0f} GB/s)  [{m['calls_under'][0]}, {m['calls_under'][1]} calls]",
              flush=flush)
    for m in s.get("m2", []):
        print(f"{head}  m2  x{m['n_comp']:<3d} comp {m['t_comp']:7.3f}  comm {m['t_comm']:7.3f}  ovl {m['t_ovl']:7.3f} ms"
              f"  (in ovl: comp {m['t_comp_in']:7.3f}, comm {m['t_comm_in']:7.3f})  E {m['efficiency']:5.2f}"
              f"  S {m['speedup']:.2f} of {m['ideal_speedup']:.2f}", flush=flush)
    if c["late_releases"]:
        print(f"{head}  ({c['late_releases']} episodes released early and dropped)", flush=flush)


if __name__ == "__main__":
    main()
