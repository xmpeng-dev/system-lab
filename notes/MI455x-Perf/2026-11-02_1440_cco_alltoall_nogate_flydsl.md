# 卡死来自 cco_bench 的 `_sleep` 开始门；去掉后正式数据：FlyDSL grouped GEMM E 0.58–0.76，hipBLASLt dense GEMM 0.30–0.56

> 时间: 2026-11-02 14:40 (Asia/Shanghai)
> 项目: MI455x-Perf
> 硬件: 4 × AMD Instinct MI455X（gfx1250，256 CU），`ctheliosr-rck-g02-e15-01`
> 容器: `xiaompen_ep_overlap`（镜像 `amdprimus/amdprimus:tas-20261019`），primus_turbo 0.6.1.dev2（容器自带）
> 软件: torch 2.14.0+rocm10.2.0a20261011，HIP 7.17，RCCL 2.32.3，hipBLASLt（`_rocm_sdk_devel`），aiter
> 代码: `minimax-poc` main @ d8a1169 + 未提交改动（`cco_bench.py` 去掉开始门、新增 `gmm_flydsl`；`cco-alltoall/run.sh`）
> 数据: [`data/2026-11-02_cco_alltoall/results/`](./data/2026-11-02_cco_alltoall/results/)

## 1. 上下文

- 上一篇（[12:30](./2026-11-02_1230_cco_alltoall_gemm_occupies_all_cus.md)）：hipBLASLt persistent grid 占满 CU，data-parallel
  grid 把 `gmm` + 768 MiB 的 E 从 0 提到 0.56，但 data-parallel 下测试频繁卡死。
- 本篇：卡死定位、FlyDSL grouped GEMM、hipBLASLt grouped GEMM、正式数据、M1 稳态瓶颈分析。

## 2. 卡死

| # | 现象 / 实验 | 结论 |
|---|---|---|
| 1 | 两种表现：一个 rank 停在 `torch.cuda.synchronize`、其它在 `_all_max` allreduce；或 4 个 rank 停在同一个 alltoall | — |
| 2 | 加埋点的 `cco_bench_dbg.py`（每次调用后记 event，窗口 60 s 未完成就打印）：卡在**纯通信窗口**（0 comp, 1 comm, 269 episodes）第 207 个 episode；rank 2 的开始门事件 `g1`（默认 stream 上 `_sleep` 之后）一直不完成，其它 rank 的 alltoall 在等它 | 和计算 kernel 无关 |
| 3 | 4 卡 × 10 万次只跑开始门（`sleep_probe.py`，不用 RCCL） | 0 次卡住 |
| 4 | FlyDSL grouped GEMM + 开始门 | 5 次启动卡 3 次（`gated/flydsl/progress.txt`），所以换掉 hipBLASLt 没用 |
| 5 | `GPU_MAX_HW_QUEUES=16` + 开始门 | 只跑完 1 次（不卡）就停了，未下结论 |
| 6 | **去掉开始门**，顺序 launch（同 `moe-deepep/ep_overlap.py`） | `gmm`（data-parallel）、`gmm_flydsl` 各 4 次 + 正式 4 kernel × 5 大小：**0 次卡死**（`nogate_hang_check/`、`formal_nogate/`） |

- 开始门的做法：默认 stream 上 `torch.cuda._sleep` 一段（按每次 launch 30 µs 估），记 `g1`，计算 / 通信 stream 都
  `wait_event(g1)` 后排入 episode，一个窗口一次排几十到几百个 episode。目的是两条 stream 同时放行、且 host 已经排好队。
- 推测（未证实）：默认 4 个 HW queue，默认 / 计算 / 通信 / RCCL 内部 stream 共用 queue；开始门 + 跨 stream 等待 +
  RCCL kernel 等对端，在某种 queue 分配下形成环。真实训练没有这种跨 stream 门。
- 开始门不是必须的：一次调用 0.1–1.7 ms，launch 几十 µs，顺序 launch 时 host 很快领先 GPU；开头几十 µs 的错位相对
  300 ms 窗口可以忽略。`examples/13-elementwise/silu-and-mul/timing.py` 的 `_sleep` 是单 kernel 微基准去 launch 开销，
  和 overlap 无关。

## 3. grouped GEMM 实现对比（单卡，72 × 912 × 6144 → 2048，`scripts/gg_backends.py`、`gmm_impls.py`）

| 实现 | 时间 | TF | 说明 |
|---|---|---|---|
| `torch.bmm`（hipBLASLt batched） | 869 µs | 1901 | 各 expert 行数相同才能用 |
| Turbo `HIPBLASLT` 后端（`hipblaslt_grouped_gemm`） | 3086 µs | 536 | 每 expert 一次 GEMM（72 次），每次 split-K 8（`MT256x240x64` 256 WG、193 KB LDS + `PostGSU8` 归约） |
| `torch._grouped_mm` | 3206 µs | 516 | 同上，也是 hipBLASLt 逐 expert |
| FlyDSL gfx1250 grouped（`grouped_gemm_bf16_nt_flydsl_kernel`） | 734–755 µs | 2188–2251 | 2632 WG × 128 线程，416 VGPR，每 tile 一个 WG |

- 误差都是相对 FP32 1.7e-3。FlyDSL 调用时预先建好 `m_tiles`（`build_m_tile_map(offs, 256)`），`BLOCK_M=256`。
- hipBLASLt 在 gfx1250 上没有可用的 grouped kernel；变长 expert 时 FlyDSL 是可用选择。

## 4. 正式数据（顺序 launch，RCCL 默认，300 ms × 10 轮，`formal_nogate/`）

README（客户 repo）只放 M2 表。M1 稳态干扰：

| 计算 | 通信 | kernel 单次：单独 → 同时 | 通信单次：单独 → 同时 | 通信 busbw：单独 → 同时 |
|---|---|---|---|---|
| `gmm_flydsl` | 768 MiB | 0.958 → 1.313 ms（1.37×） | 0.811 → 1.113 ms（1.37×） | 745 → 543 GB/s |
| `gmm_flydsl` | 384 MiB | 0.934 → 1.238 ms（1.33×） | 0.436 → 0.695 ms（1.60×） | 693 → 434 GB/s |
| `gmm_flydsl` | 256 MiB | 0.939 → 1.103 ms（1.17×） | 0.243 → 0.369 ms（1.52×） | 830 → 545 GB/s |
| `gmm_flydsl` | 128 MiB | 0.942 → 1.067 ms（1.13×） | 0.144 → 0.217 ms（1.51×） | 701 → 465 GB/s |
| `gmm_flydsl` | 64 MiB | 0.942 → 1.029 ms（1.09×） | 0.0935 → 0.140 ms（1.49×） | 538 → 360 GB/s |
| `attn_fwd` | 768 MiB | 0.585 → 0.830 ms（1.42×） | 0.813 → 1.072 ms（1.32×） | 743 → 563 GB/s |
| `attn_fwd` | 384 MiB | 0.584 → 0.793 ms（1.36×） | 0.437 → 0.657 ms（1.50×） | 691 → 459 GB/s |
| `attn_fwd` | 256 MiB | 0.573 → 0.682 ms（1.19×） | 0.243 → 0.471 ms（1.94×） | 829 → 428 GB/s |
| `attn_fwd` | 128 MiB | 0.572 → 0.651 ms（1.14×） | 0.144 → 0.332 ms（2.31×） | 701 → 303 GB/s |
| `attn_fwd` | 64 MiB | 0.571 → 0.627 ms（1.10×） | 0.0960 → 0.255 ms（2.66×） | 525 → 197 GB/s |
| `attn_bwd` | 768 MiB | 1.644 → 2.247 ms（1.37×） | 0.811 → 1.148 ms（1.42×） | 745 → 526 GB/s |
| `attn_bwd` | 384 MiB | 1.643 → 2.140 ms（1.30×） | 0.437 → 0.716 ms（1.64×） | 692 → 422 GB/s |
| `attn_bwd` | 256 MiB | 1.646 → 1.936 ms（1.18×） | 0.243 → 0.482 ms（1.99×） | 830 → 418 GB/s |
| `attn_bwd` | 128 MiB | 1.643 → 1.842 ms（1.12×） | 0.143 → 0.353 ms（2.47×） | 703 → 285 GB/s |
| `attn_bwd` | 64 MiB | 1.643 → 1.767 ms（1.08×） | 0.0932 → 0.284 ms（3.05×） | 540 → 177 GB/s |
| `gemm` | 768 MiB | 1.266 → 1.787 ms（1.41×） | 0.812 → 1.115 ms（1.37×） | 744 → 542 GB/s |
| `gemm` | 384 MiB | 1.247 → 1.680 ms（1.35×） | 0.436 → 0.707 ms（1.62×） | 692 → 427 GB/s |
| `gemm` | 256 MiB | 1.243 → 1.445 ms（1.16×） | 0.242 → 0.499 ms（2.06×） | 830 → 403 GB/s |
| `gemm` | 128 MiB | 1.242 → 1.401 ms（1.13×） | 0.143 → 0.351 ms（2.45×） | 702 → 287 GB/s |
| `gemm` | 64 MiB | 1.238 → 1.354 ms（1.09×） | 0.0958 → 0.317 ms（3.31×） | 525 → 159 GB/s |

瓶颈分析：

- 768 MiB：两边都慢约 1.37–1.42×，四个 kernel 一样，限制在共享资源（HBM 带宽 / CU），换 kernel 没用。
- 小包：kernel 只慢 1.1×，alltoall 在 attention / dense GEMM 下慢 2.5–3.3×，GPU 偏向计算，RCCL 抢不到 CU；
  FlyDSL grouped GEMM 下只慢 1.5×（WG 128 线程、多轮调度，空出 CU 的机会更多，未验证）。方向：给通信资源，
  不是优化 GEMM。
- 用法：TBO 稳态每层 ≈ max(T_comp × kernel 倍数, T_comm × 通信倍数)。

## 5. 4 卡下 kernel 单独跑比单卡慢：功耗墙

| kernel | 单卡短测（~40 ms） | 4 卡 300 ms 窗口 | 开始门版本 |
|---|---|---|---|
| hipBLASLt `gmm`（bmm，data-parallel） | 861 µs | 1.02–1.03 ms | 0.877 ms |
| FlyDSL `gmm_flydsl` | 734 µs | 0.96 ms | — |

telemetry（`formal_nogate/*.json`，每卡 socket power / gfx 频率中位数）：纯计算 2450–2500 W、1200–1420 MHz；纯通信
1740–1860 W、1910–1980 MHz；同时跑 2340–2460 W、1440–1630 MHz。持续计算顶到功耗墙降频；单卡短测没到功耗墙，开始门版本
episode 之间有空转、平均功耗低，所以更快。推测，未单独验证（可以单卡跑 300 ms 窗口 + 采样确认）。

## 6. 结论

- 卡死是测试方法（开始门）引入的，去掉后 20+ 次运行 0 卡死；`cco_bench.py` 已改为顺序 launch。
- 客户 README：grouped GEMM 用 FlyDSL，dense GEMM 用 hipBLASLt data-parallel grid；E：FlyDSL 0.58–0.76，attention
  0.50–0.66，dense GEMM 0.30–0.56。
- 旧数据（开始门）都作废：`gated/`（prelim persistent、formal 中断、FlyDSL 开始门）。

## 7. 下一步

| 优先级 | 方向 |
|---|---|
| P0 | allgather / reducescatter 用顺序 launch 重跑 |
| P1 | 小包 alltoall 被计算拖慢 2.5–3.3×：RCCL channel 数 / CU 预留 / SDMA |
| P1 | 功耗墙确认（单卡 300 ms 窗口 + telemetry） |
| P2 | 开始门卡死的最小复现（`GPU_MAX_HW_QUEUES=16` 对照），可报 RCCL / HIP |
