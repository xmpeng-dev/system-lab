# MI455x-Perf — MI455X 性能分析（计算 + 通信 overlap）

> **目标**: 分析 MI455X 上计算和集合通信同时跑（没有数据依赖）时的 overlap 性能，找原因和调法
> **平台**: 4 × MI455X（gfx1250，256 CU），`ctheliosr-rck-g02-e15-01`
> **栈**: ROCm 10.2（pip）/ HIP 7.17 / RCCL 2.32 / hipBLASLt / aiter
> **测试用例**: `~/minimax-poc/examples-phase2/training/`（`cco-alltoall` 等）；分析数据只放这里，不进那个 repo

## 状态

| 维度 | 值 |
|---|---|
| 当前阶段 | 单 tray 4 卡：cco-alltoall / allgather 正式数据已出；跨 tray（e15-01+e15-02）摸清了 vPOD 挡 scale-up、前端 ionic_0 RoCE 约 27.5 GB/s/节点。**跨 tray / EP8 优先级已降低，暂不继续投入**（等 AFM 配同 vPOD 或后端 Vulcano 接线后再开） |
| 当前最佳 | alltoall：FlyDSL + 768 MiB overlap 1.279 ms（E 0.61），dense GEMM data-parallel E 0.30–0.56；allgather：CE 段（128–256 MiB）FlyDSL E 0.75–0.81、attention 0.60–0.66 |
| 已知问题 | allgather ≥ 512 MiB 走 Ring 256 channel，和计算按 CU 切分，E 只有 0.22–0.32（计算慢 2–2.75×）；小包（alltoall、allgather DDA）被计算拖慢约 3×；小包 allgather + dense GEMM 因降频 E 为负；4 卡持续计算顶功耗墙降频；跨 tray 无同 vPOD 时只能走前端 400G |
| 上次更新 | 2026-11-04 |

## 进展时间线

| 日期 | 里程碑 | 关键数字 | note |
|---|---|---|---|
| 2026-11-02 12:30 | GEMM 占满 CU 的根因 + CU 配比 / data-parallel grid | 留 96 CU：1.301 ms；data-parallel：1.235 ms（默认 1.724） | [note](./2026-11-02_1230_cco_alltoall_gemm_occupies_all_cus.md) |
| 2026-11-02 14:40 | 卡死 = 开始门；顺序 launch 正式数据；FlyDSL / hipBLASLt grouped GEMM | 0 卡死；FlyDSL E 0.58–0.76；hipBLASLt grouped 3.1 ms（bmm 0.87） | [note](./2026-11-02_1440_cco_alltoall_nogate_flydsl.md) |
| 2026-11-02 15:15 | cco-allgather 正式数据；RCCL 按大小换实现（Ring / CE / DDA） | E：Ring 0.22–0.32，CE 0.28–0.81，DDA −0.36–0.30；Ring 下计算慢 2–2.75× | [note](./2026-11-02_1515_cco_allgather_rccl_paths.md) |
| 2026-11-02 16:00 | cco-alltoall 多模型形状（MiniMax / M3 T4k / M3 T8k / DSv3）+ down GEMM；校准循环跨 rank 卡死 | 完整 dispatch 时 E：M3 T4k 0.54–0.77 最高，DSv3 0.52–0.61；down GEMM 拖慢 alltoall 最少（1.10×） | [note](./2026-11-02_1600_cco_alltoall_models.md) |
| 2026-11-02 20:50 | cco-alltoall 改成 Megatron `overlap_moe_expert_parallel_comm` 的真实配对（M3 一层 4 段 × 192 MiB）；DSv3 FlyDSL MLA attention | 整层 1.08×（理想 1.12×），E 0.67，各段 0.59–0.81；FlyDSL MLA fwd E 0.68（aiter 0.52），bwd 大包把 alltoall 压到 178 GB/s | [note](./2026-11-02_2050_cco_alltoall_megatron_layer.md) |
| 2026-11-02 22:30 | Helios e15-01 + e15-02 做 8 卡：跨 tray scale-up 被 vpod 分区挡住 | 同一 72 卡 ppod、链路全 UP，但每 tray 一个 4 卡 vpod；驱动侧可改，IFoE 网卡侧只能 PROVIDER 阶段改（主机切不回），需机柜 fabric 负责人配；已恢复原状 | [note](./2026-11-02_2230_helios_cross_tray_vpod.md) |
| 2026-11-03 09:40 | 跨 tray alltoall 2×2 走前端 ionic_0：RoCE vs TCP | RoCE ~27.5 GB/s/节点（192 MiB/卡 7.34 ms），TCP 3–5 GB/s；RCCL 默认跑不起来，要 shim（max_recv_wr 0→1）+ 强制 GDR + 关 RMA + 降 channel 防丢包 | [note](./2026-11-03_0940_cross_tray_a2a_2x2_frontend.md) |

## 下一步（按 ROI）

| 优先级 | 方向 |
|---|---|
| P0 | reducescatter 用顺序 launch 重跑 |
| P1 | allgather ≥ 512 MiB：减 RCCL channel 数（`nch64` / 32）或让 CE 路径覆盖大包（scratch 上限 / 注册缓冲区） |
| P1 | CE 路径通信总要等计算结束：trace 128 / 256 MiB 配对 |
| P1 | 小包 alltoall / allgather DDA 被拖慢：RCCL channel 数 / CU 预留 / SDMA |
| P1 | 功耗墙确认（单卡 300 ms 窗口 + telemetry） |
| P2 | 开始门卡死最小复现，报 RCCL / HIP |
| 暂停 | 跨 tray：等 AFM 把 e15-01/02 配进同一 vPOD，或接上 `ionic_1..8` 后端 Vulcano；前端 `ionic_0` 通路已验证，不再追加调参 |

## 文件索引

| 文件 | 说明 |
|---|---|
| `2026-11-02_1230_cco_alltoall_gemm_occupies_all_cus.md` | 根因、hipBLASLt grid 开关、CU 配比、data-parallel grid |
| `2026-11-02_1440_cco_alltoall_nogate_flydsl.md` | 卡死定位、grouped GEMM 实现对比、正式数据 + M1 瓶颈分析、功耗墙 |
| `2026-11-02_1515_cco_allgather_rccl_paths.md` | allgather 正式数据 + M1、RCCL 三种实现的资源和瓶颈、telemetry |
| `2026-11-02_1600_cco_alltoall_models.md` | 4 个模型形状的正确性、单卡速度、M2 E、M1 互相拖慢、校准循环卡死 |
| `2026-11-02_2050_cco_alltoall_megatron_layer.md` | TBO 不是训练配置、通信量只有 192 MiB、Megatron 1F1B 每层 4 对、单层结果和 M1、DSv3 FlyDSL MLA |
| `2026-11-02_2230_helios_cross_tray_vpod.md` | 两个 tray 的 ppod / vpod / IFoE 状态、驱动侧改 vpod 的步骤和结果、恢复、副作用 |
| `2026-11-03_0940_cross_tray_a2a_2x2_frontend.md` | 前端 ionic_0 上 RCCL RoCE 跑不起来的 5 个原因和绕法、2×2 RoCE / TCP alltoall 结果、ibv_rc_pingpong 基线 |
| `data/2026-11-02_cco_alltoall/` | 脚本、结果 JSON / log、trace（trace 不进 git） |
| `data/2026-11-02_cco_allgather/` | 正式结果、RCCL 路径日志、trace 脚本和汇总、telemetry 脚本（trace 不进 git） |
| `data/2026-11-02_cross_tray/` | vPOD / AFM 快照、探测脚本（大 NCCL diag log 不进 git） |
| `data/2026-11-03_cross_tray_a2a/` | 2×2 RoCE/TCP JSON、launch/pair/shim 脚本 |

## 维护约定

- 进展写新 note（`YYYY-MM-DD_HHMM_<topic>.md`），本 README 是当前状态的唯一来源。
- 数据按日期放 `data/<日期>_<主题>/`。
