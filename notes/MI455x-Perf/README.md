# MI455x-Perf — MI455X 性能分析（计算 + 通信 overlap）

> **目标**: 分析 MI455X 上计算和集合通信同时跑（没有数据依赖）时的 overlap 性能，找原因和调法
> **平台**: 4 × MI455X（gfx1250，256 CU），`ctheliosr-rck-g02-e15-01`
> **栈**: ROCm 10.2（pip）/ HIP 7.17 / RCCL 2.32 / hipBLASLt / aiter
> **测试用例**: `~/minimax-poc/examples-phase2/training/`（`cco-alltoall` 等）；分析数据只放这里，不进那个 repo

## 状态

| 维度 | 值 |
|---|---|
| 当前阶段 | 根因（persistent grid 占满 CU）和卡死（cco_bench 开始门）都已定位；cco-alltoall 正式数据已出 |
| 当前最佳 | grouped GEMM 用 FlyDSL：+ 768 MiB overlap 1.279 ms（E 0.61），64 MiB E 0.76；dense GEMM 用 hipBLASLt data-parallel grid：E 0.30–0.56 |
| 已知问题 | 小包 alltoall 被计算拖慢 2.5–3.3×（FlyDSL 下 1.5×）；4 卡持续计算顶功耗墙降频 |
| 上次更新 | 2026-11-02 14:40 |

## 进展时间线

| 日期 | 里程碑 | 关键数字 | note |
|---|---|---|---|
| 2026-11-02 12:30 | GEMM 占满 CU 的根因 + CU 配比 / data-parallel grid | 留 96 CU：1.301 ms；data-parallel：1.235 ms（默认 1.724） | [note](./2026-11-02_1230_cco_alltoall_gemm_occupies_all_cus.md) |
| 2026-11-02 14:40 | 卡死 = 开始门；顺序 launch 正式数据；FlyDSL / hipBLASLt grouped GEMM | 0 卡死；FlyDSL E 0.58–0.76；hipBLASLt grouped 3.1 ms（bmm 0.87） | [note](./2026-11-02_1440_cco_alltoall_nogate_flydsl.md) |

## 下一步（按 ROI）

| 优先级 | 方向 |
|---|---|
| P0 | allgather / reducescatter 用顺序 launch 重跑 |
| P1 | 小包 alltoall 被拖慢：RCCL channel 数 / CU 预留 / SDMA |
| P1 | 功耗墙确认（单卡 300 ms 窗口 + telemetry） |
| P2 | 开始门卡死最小复现，报 RCCL / HIP |

## 文件索引

| 文件 | 说明 |
|---|---|
| `2026-11-02_1230_cco_alltoall_gemm_occupies_all_cus.md` | 根因、hipBLASLt grid 开关、CU 配比、data-parallel grid |
| `2026-11-02_1440_cco_alltoall_nogate_flydsl.md` | 卡死定位、grouped GEMM 实现对比、正式数据 + M1 瓶颈分析、功耗墙 |
| `data/2026-11-02_cco_alltoall/` | 脚本、结果 JSON / log、trace（trace 不进 git） |

## 维护约定

- 进展写新 note（`YYYY-MM-DD_HHMM_<topic>.md`），本 README 是当前状态的唯一来源。
- 数据按日期放 `data/<日期>_<主题>/`。
