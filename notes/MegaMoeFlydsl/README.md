# MegaMoeFlydsl — Mega MoE 通信-计算融合算子（FlyDSL / MI355X）

Primus-Turbo 里 **Mega MoE** 两个融合算子（`dispatch_grouped_gemm` / `grouped_gemm_combine`）
在 AMD MI355X（gfx950）上的正确性、性能与优化工作日志。EP intra-node，DeepSeek-V3 口径。

## 状态

| 项 | 值 |
|---|---|
| 项目状态 | ⚠️ **blocked on regression** |
| 上次更新 | 2026-09-07 10:34 |
| 当前最佳（bf16 dispatch，7-16 基准栈） | `grouped/dense` 91.9–100%，overlap 87.6–92.1%，speedup vs 串行 **1.46–1.61×**，XGMI dispatch **368–381 GB/s** |
| 当前实测（2026-09-07 新栈） | `grouped/dense` **8.7%**，overlap 57.1–74.3%，speedup **0.81–0.99×**，XGMI dispatch **50.8 GB/s** |
| 当前最佳（MXFP8 fwd+bwd，8-01） | e2e 14.64 ms，bf16/fp8 = **1.40×** |
| 头号阻塞 | 新软件栈上的性能回归未定位（见 2026-09-07 note 的 P0） |

**基准配置**：DeepSeek-V3，EP8，H=7168，I=2048，E=256，topk=8，T=8192（tokens/rank），跨 8 rank 取最慢。

## 环境

- 硬件：8× MI355X（gfx950），单节点 XGMI 全互联
- 容器：`xiaoming-dev`。7–8 月为 `rocm/primus:v26.3`（torch 2.10.0+git94c6e04 / ROCm 7.2.1 / triton 3.6.0）；
  9 月起为 **`tasimage/primus:pr-1048`**（torch 2.12.0+rocm7.15 / HIP 7.15.0 / triton 3.7.0）
- flydsl：7-16 起为 **0.2.4**（更早为 egg 装的 `0.1.1.dev409`，会遮蔽 pip 版）
- 代码：`/perf_apps/xiaoming/MegaMoE`（树内 `.so` ABI 正常）、`/perf_apps/xiaoming/MegaMoE-dev`、
  系统安装版 `primus-turbo 0.4.1.dev33`

## 进展时间线

| 日期 | 里程碑 | 关键数字 | note |
|---|---|---|---|
| 2026-07-16 11:04 | **基准建立** — 两融合算子 EP8 正确性 + 性能 | 5/5 阶段 cos 全绿；grouped/dense **91.9–100%**；overlap **87.6–92.1%**；speedup **1.46–1.61×**；XGMI dispatch 368–381 / combine ~357 GB/s | [link](./2026-07-16_1104_BASELINE_mega_flydsl_ep8_dsv3_bench_mi355x.md) |
| 2026-07-17 14:11 | 从源码 build（含 CK/HIP `csrc`）复现 | overlap 86.5–88.2%；grouped/dense 90–96%；speedup 1.37–1.75× ⇒ 与安装方式无关 | [link](./2026-07-17_1411_mega_flydsl_ep8_dsv3_n0125_srcbuild_mi355x.md) |
| 2026-07-20 15:20 | **MXFP8 前向**跑通 | fp8 vs bf16 SNR **20.8–22.3 dB** / cos 0.996–0.997；L2 overlap 效率 ~71%（combine 与 GEMM 争 CU） | [link](./2026-07-20_1520_mega_fp8_forward_perf_mi355x.md) |
| 2026-07-21 10:10 | MXFP8 反向性能追踪 | — | [link](./2026-07-21_1010_mega_fp8_backward_perf_t8192_mi355x.md) |
| 2026-07-23 14:52 | 通信机制 **epoch 化** + 全 stage 快照 | 双 bank i64 flag（parity 选 bank）+ 设备端 `epoch_bump`，不消费复位 | [link](./2026-07-23_1452_mega_fp8_epoch_comm_stage_perf_t8192_mi355x.md) |
| 2026-07-24 09:56 | CU 配比重扫 + wgrad variable-K GEMM campaign | — | [link](./2026-07-24_0956_mega_fp8_cu_split_retune_and_wgrad_gemm_t8192_mi355x.md) |
| 2026-07-24 14:32 | `fc1_dgrad` 提前 + (`fc1_wgrad` ∥ combine) 单 kernel CU 分角色 overlap **设计** | `fc1_dgrad_combine` 3.03 ms 为 fp8 反向最弱腿，仅 1.26× vs bf16 | [link](./2026-07-24_1432_mega_fp8_fc1_dgrad_wgrad_combine_overlap_plan.md) |
| 2026-08-01 14:40 | 换机 + flydsl 0.2.4 后各阶段回归 | fp8 fwd+bwd **14.64 ms** / fwd 5.52 / bwd 9.08；bf16/fp8 = **1.40×**（bwd 1.53×） | [link](./2026-08-01_1440_mega_fp8_bwd_only_regression_flydsl024_t8192_mi355x.md) |
| 2026-08-01 15:30 | 反向瓶颈拆解 + combine top-k reduce 向量化 | — | [link](./2026-08-01_1530_mega_fp8_bwd_bottleneck_and_combine_reduce_b128_mi355x.md) |
| 2026-09-03 15:55 | **FlyDSL vs Triton** 算子对比（6 类 / 126 shape） | — | [link](./2026-09-03_1555_flydsl_vs_triton_op_perf_mi355x.md) |
| 2026-09-07 10:34 | ⚠️ **新栈上 dispatch 大幅回归** | grouped/dense **8.7%**（基准 91.9–100%）；XGMI **50.8 GB/s**（基准 368–381，**7.3×**）；speedup **0.81×**（基准 1.46–1.61×）。正确性仍全绿；已排除 autotune 缓存与 token 规模。**同日同机朴素 RCCL a2a 实测 329.6 GB/s ⇒ 融合 dispatch 比朴素集合通信还慢 6.5×，硬件无辜** | [link](./2026-09-07_1034_mega_bf16_dispatch_regression_vs_0716_baseline_mi355x.md) |

## 下一步（按 ROI）

| 优先级 | 方向 | 前置 / 风险 |
|---|---|---|
| **P0** | 二分回归点：固定 flydsl 0.2.4，分别只换 turbo 版本、只换容器 torch | 需能取到 `rocm/primus:v26.3` 镜像 |
| **P0** | 用 `/perf_apps/xiaoming/MegaMoE` 树内构建在新容器复跑同工况 | 区分「turbo 版本」与「容器 torch」两个嫌疑 |
| P1 | 单独定位 XGMI dispatch 50.8 vs 368 GB/s | 用 `bench_deepep_intranode.py` / `bench_symmetric_memory.py` 交叉印证 |
| P1 | 修 `wgrad` 的 `dispatch_only` 计时不稳（会产生 159.5% 这种非物理 roofline） | 读 bench 计时实现 |
| P2 | 补跑 `--mode grouped_gemm_combine`；给 T=16384 提高 PG timeout | `timeout=600000` 在脚本内硬编码 |

## 文件索引

- **滚动工作日志**：[`mxfp8_moe_bwd_perf_summary.md`](./mxfp8_moe_bwd_perf_summary.md) — MXFP8 反向逐 stage 汇总、
  dual-quant 优化 loop、`fc1_dgrad_combine` rocprof 拆解、dW1/dW2 优化、CU 配比 sweep 等（按主题追加，非按时间）
- [`HANDOFF_step3_deadlock.md`](./HANDOFF_step3_deadlock.md) — step3 死锁交接
- `_repro/`、`logs/`、`assets/` — 复现脚本、原始日志、图表

## 维护约定

- 每次实验后按 `YYYY-MM-DD_HHMM_<topic>.md` 追加 note，并回写本 README 的时间线与状态。
- 旧 note 不改；"当前最佳"以本 README 为准。
- 所有性能数字必须标注**软件栈版本**——本项目已两次因换栈出现读数不可比（8-01、9-07）。
