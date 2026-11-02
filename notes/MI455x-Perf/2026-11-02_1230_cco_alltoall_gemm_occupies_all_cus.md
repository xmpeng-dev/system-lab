# GEMM + alltoall 没有 overlap：hipBLASLt persistent grid 占满全部 CU；data-parallel grid 把 E 从 0 提到 0.56

> 时间: 2026-11-02 12:30 (Asia/Shanghai)
> 项目: MI455x-Perf
> 硬件: 4 × AMD Instinct MI455X（gfx1250，256 CU，每 CU 4 SIMD、320 KB LDS），`ctheliosr-rck-g02-e15-01`
> 容器: `xiaompen_ep_overlap`（镜像 `amdprimus/amdprimus:tas-20261019`）
> 软件: torch 2.14.0+rocm10.2.0a20261011，HIP 7.17.26405，RCCL 2.32.3，hipBLASLt（`_rocm_sdk_devel`），rocprofv3
> 代码: `minimax-poc` main @ d8a1169 + 未提交改动（`examples-phase2/training/common/cco_bench.py`）；
>   本 note 的脚本和数据：[`data/2026-11-02_cco_alltoall/`](./data/2026-11-02_cco_alltoall/)

## 1. 时间点 / 上下文

- `examples-phase2/training/cco-alltoall`（计算和 alltoall 没有数据依赖，两条 stream 同时跑）的初步和正式扫描：
  `gmm` / `gemm` + alltoall 基本没有 overlap，attention + alltoall 有一半。
- 方法和指标同 `cco_bench.py`：E = (T_comp + T_comm − T_ovl) / min(T_comp, T_comm)；M1 为背靠背稳态干扰，M2 为一次通信配
  N 次计算。

## 2. 问题

- 现状（默认配置，768 MiB / rank）：`gmm`（bmm 72 × 912 × 6144 → 2048）0.88 ms，alltoall 0.82 ms，overlap 1.72 ms，
  **E = −0.03**；`gemm`（8192 × 6144 → 24576）overlap 1.93 ms，E = −0.01。稳态下 alltoall 慢 3–20×，GEMM 只慢 1.1–1.4×。
  attention 的 E 约 0.5。
- `NCCL_MAX_NCHANNELS`、`NCCL_CTA_POLICY=2`、`RCCL_FORCE_CE=1`、RCCL 高优先级都没有改善。

## 3. 做了什么

| # | 动作 | 脚本 | 结果 |
|---|---|---|---|
| 1 | rocprofv3 kernel trace，4 卡，一次放行计算 + alltoall | `scripts/trace.sh`、`a2a_trace.py`、`trace_sum.py` | `trace/`（22 MB，不进 git） |
| 2 | 单卡：计算 kernel 跑时，另一条 stream 上的小 kernel 什么时候结束 | `scripts/cu_probe.py` | 下文表 2 |
| 3 | 查 hipBLASLt 的 grid 开关：`TENSILE_PERSISTENT_MAX_CUS`、`TENSILE_PERSISTENT_FIXED_GRID` | `scripts/gemm_cus.py` | 下文表 4 |
| 4 | 给 RCCL 留 R 个 CU（GEMM grid = 256 − R），4 卡 cco_bench | `scripts/cu_split.sh` | `results/cusplit/r<R>.json` |
| 5 | data-parallel grid（`TENSILE_PERSISTENT_FIXED_GRID=100000`） | 同上 | `results/cusplit/fg100000.json` |
| 6 | data-parallel grid + CU mask 留 32 个 CU（`GPU_MAX_HW_QUEUES=16`） | 同上 | `results/cusplit/fgcu32.log`（卡死） |
| 7 | data-parallel grid × 7 个 GEMM shape × 4 个通信大小 | `scripts/dp_shapes.sh`、`dp_table.py` | 进行中，见后续 note |

## 4. 效果

### 4.1 根因：GEMM 在运行期间占满全部 256 个 CU

各 kernel 的资源（trace）：

| kernel | workgroup 数 × 线程 | 每 workgroup LDS | VGPR | 在 256 个 CU 上 |
|---|---|---|---|---|
| `gmm`：hipBLASLt `MT256x240x128` | 256 × 128 | 263.5 KB | 512 | **一轮**，每 CU 一个，循环算完全部 tile |
| `gemm`：hipBLASLt `MT256x256x128` | 256 × 128 | 260 KB | 512 | **一轮**，循环算完 3072 个 tile |
| `attn_fwd`：aiter `fmha_bf16_..._128x256` | 2048 × 128 | 320 KB | 512 | 8 轮 |
| `attn_bwd`：aiter `fmha_bwd_hd128_bf16_a32_pssk` | 4096 × 128 | 320 KB | 512 | 16 轮 |
| RCCL alltoall 768 MiB：`ncclDevKernel_Generic` | 先 192 × 256，再 64 × 256 | 83.5 KB | 256 | 两个 kernel 接着跑 |

GEMM 和 attention 的每个 workgroup 都占满一个 CU（4 个 wave × 512 VGPR，LDS 260–320 KB）。差别在于 GEMM 是 persistent
grid（workgroup 数 = CU 数，循环算 tile），整个 kernel 期间一个 CU 都不空；attention 每轮 workgroup 退出时空出 CU，RCCL
的 workgroup 能插进去。

**表 2：计算 kernel 跑时，另一条 stream 上的小 kernel（约 50 µs 时发起）什么时候结束**（单卡）

| 计算 | 计算结束 | 小 kernel 结束 |
|---|---|---|
| `gmm` | 864 µs | 871 µs |
| `gemm` | 1048 µs | 1082 µs |
| `attn_fwd` | 488 µs | 188 µs |
| `attn_bwd` | 1336 µs | 68 µs |

64 MiB 的 alltoall 不走 nccl kernel，走 `__amd_rocclr_batchMemOp`（1 个线程、不用 LDS）+ `__amd_rocclr_copyBufferBatch`
/ `copyBuffer`：这个 1 线程的 kernel 15 µs 发起，840 µs（GEMM 结束）才结束。所以 `RCCL_FORCE_CE=1` 也没用 —— HIP 的 copy
是 blit kernel，同样要 CU。

**表 3：从放行起各 kernel 的起止（trace 中位数，µs；alltoall 单独约 820）**

| 情况 | 计算 | alltoall |
|---|---|---|
| `gmm` + 768 MiB，同时 | 10 → 873 | 14 → 1654 |
| `gemm` + 768 MiB，同时 | 10 → 1037 | 14 → 1841 |
| `attn_fwd` + 768 MiB，同时 | 10 → 747 | 14 → 993 |
| `attn_bwd` + 768 MiB，同时 | 186 → 1773 | 14 → 1019 |
| `gmm` + 768 MiB，通信提前 100 µs | 26 → 1100 | −101 → 1203 |
| `gemm` + 768 MiB，通信提前 100 µs | 28 → 1297 | −102 → 1411 |

同时发起时 alltoall 结束 ≈ GEMM 结束 + 0.8 ms。通信先发起时 RCCL 第一个 kernel 先占 192 个 CU，GEMM 对应的 workgroup 等它
退出（GEMM 拖长约 0.25 ms），RCCL 第二个 kernel 再等 GEMM 结束 —— 还是轮流用 CU。

### 4.2 hipBLASLt 的 grid 开关

**表 4：单卡单独时间（µs）**

| 设置 | `gmm` | `gmm` grid | `gemm` | `gemm` grid |
|---|---|---|---|---|
| 默认 | 865 | 256 | 1048 | 256 |
| `TENSILE_PERSISTENT_MAX_CUS=240` | 930 | 240 | 1062 | — |
| `=224` | 954 | 219 | 1053 | **3072**（退回 data-parallel） |
| `=192` | 987 | 192 | 1271 | 192 |
| `=160` | 1141 | 159 | 1059 | **3072** |
| `=128` | 1330 | — | 1682 | — |
| `TENSILE_PERSISTENT_FIXED_GRID=2592`（≥ tile 数） | 863 | 2304 | 1058 | 3072 |
| `TENSILE_PERSISTENT_FIXED_GRID=512` | 861 | 2304 | 1060 | 512 |

- `MAX_CUS=n`：persistent grid 只开 n 个 workgroup，剩下的 CU 空着。`gemm` 在 3072 不能被 n 整除时自己退回 data-parallel。
- `FIXED_GRID ≥ tile 数`：每个 tile 一个 workgroup（data-parallel），速度和 persistent 一样。
- `TENSILE_STREAMK_DATA_PARALLEL=1`、`TENSILE_PERSISTENT_DYNAMIC_GRID=0/1`、`FIXED_GRID=0` 不改变 grid。

### 4.3 给 RCCL 留 CU vs data-parallel grid（4 卡，M2 一次 GEMM 配一次通信，ms）

`gmm` + 768 MiB（单独 0.88 / 0.82，串行 1.70）：

| 配置 | `gmm` 单独 | overlap | E | M1：GEMM / 通信变慢 |
|---|---|---|---|---|
| 默认 | 0.881 | 1.724 | −0.03 | 1.27× / 2.65× |
| 留 16 / 32 / 48 个 CU | 0.94–1.01 | 1.77–1.81 | ≈ 0 | — |
| 留 64 个 CU | 1.026 | 1.398 | 0.55 | 1.15× / 1.54× |
| 留 80 个 CU | 1.120 | 1.384 | 0.68 | — |
| 留 96 个 CU | 1.168 | 1.301 | 0.84 | 1.12× / 1.57× |
| 留 112 个 CU | 1.242 | 1.372 | 0.84 | — |
| 留 128 个 CU | — | 卡死（`gemm` 768 MiB 时 4 个 rank 一起卡在 alltoall） | | |
| **data-parallel grid** | **0.877** | **1.235** | 0.56 | 1.39× / 1.36× |

`gemm` + 768 MiB（串行 1.92）：默认 1.929，data-parallel 1.504（E 0.52），留 CU 各配比 1.47–1.52。

64 MiB：通信只有 0.1 ms，各配置 overlap 0.91–1.07 ms，默认（0.914）已经最好或差不多；data-parallel `gmm` 0.909。

- ✅ data-parallel grid 是目前最好的：`gmm` overlap −28%，`gemm` −22%，GEMM 本身不变慢。
- ✅ 留 CU 有门槛：≤ 48 个 CU 没用（RCCL 192 个 workgroup 放不下），64–96 个才有效，代价是 GEMM 慢 15–30%。
- ❌ 离理想（只剩 GEMM，0.88 ms）还差约 0.35 ms：data-parallel 下 RCCL 拿到的 CU 是零散的，两边互相拖慢。
- ❌ data-parallel + CU mask 留 32 个 CU 更差（`gmm` 768 MiB overlap 1.877，E −0.09），随后 64 MiB 卡死。

### 4.4 卡死

这轮又卡了 4 次：留 128 个 CU 的 `gemm` 768 MiB（4 个 rank 都卡在 alltoall）；`FIXED_GRID=512` 的 `gmm` 64 MiB（1 个
rank）；data-parallel + CU mask 的 `gmm` 64 MiB；data-parallel 的 `gmm` 64 MiB（`results/dp_partial/dp_gmm-up912.log`）。
之前 3 次都是 attention + alltoall。共同点：计算的 workgroup 和 RCCL 的 workgroup 交错占 CU。怀疑 RCCL 的 workgroup 没能
全部同时驻留时，跨 rank 的 channel 互相等待；64 MiB 的 batchMemOp / copy 路径更容易触发。未验证。

## 5. 可持续方向

| 优先级 | 方向 | 预期收益 | 风险 / 前置 |
|---|---|---|---|
| P0 | data-parallel grid × 更多 GEMM shape（gate+up / down，8192 / 4096 token，dense up / down） | 确认 data-parallel 普遍不慢、E 普遍 0.5+ | 进行中（`scripts/dp_shapes.sh`，先不测 64 MiB） |
| P0 | 卡死定位：卡住时 dump 各 rank 的 RCCL 状态（`NCCL_DEBUG`、rocgdb / umr 看 wave） | 64 MiB + data-parallel 能稳定复现，正好查 | 可能是 RCCL / HIP 的 bug |
| P1 | data-parallel grid + RCCL channel 数 / `NCCL_CTA_POLICY` | 让 RCCL 少占、早拿到 CU，逼近理想 | — |
| P1 | SDMA 做 alltoall（逐 peer `hipMemcpyAsync` 或 HIP 的 blit / SDMA 开关 `GPU_BLIT_ENGINE_TYPE` 等） | 通信完全不占 CU | SDMA 单引擎 P2P 1.6 TB/s（phase 1 TransferBench），多引擎并发未测 |
| P2 | persistent GEMM 留 CU（`MAX_CUS`）+ 调 RCCL channel 数匹配 | 可控配比 | GEMM 慢 15–30% |

## 相关文件

- 脚本：[`data/2026-11-02_cco_alltoall/scripts/`](./data/2026-11-02_cco_alltoall/scripts/)（`CCO_COMMON` 指向
  `minimax-poc/examples-phase2/training/common`；容器里看不到 slab，跑的时候拷到 `minimax-poc/.scratch/cco-analysis/`）
- 结果：[`data/2026-11-02_cco_alltoall/results/`](./data/2026-11-02_cco_alltoall/results/)
- 原始 trace：`data/2026-11-02_cco_alltoall/trace/`（不进 git）
- 测试用例：`minimax-poc/examples-phase2/training/cco-alltoall/`
