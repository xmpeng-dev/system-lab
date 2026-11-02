# Mega MoE (FlyDSL) dispatch 融合算子在新栈上大幅回归 — grouped GEMM 8.7%、XGMI 50.8 GB/s、融合慢于串行

> 时间: 2026-09-07 10:34 (Asia/Shanghai)
> 项目: MegaMoeFlydsl
> 硬件: 8× AMD Instinct MI355X (gfx950)，单节点 XGMI 全互联，节点 `smci355-ccs-aus-n04-25`（8 卡实测空闲）
> 容器: `xiaoming-dev`，镜像 **`tasimage/primus:pr-1048`**（docker，非 podman）
> 软件: torch **`2.12.0+rocm7.15.0a20260720`**，HIP **7.15.0**，triton **3.7.0**，flydsl **0.2.4**
> 代码: **系统安装版** `primus-turbo 0.4.1.dev33`（`/opt/venv/lib/python3.12/site-packages/primus_turbo`），非仓库树
> 原始日志: 容器内 `/tmp/bench_disp.log`（T=4096）、`/tmp/bench_disp8k.log`（T=8192 热缓存）、`/tmp/bench_disp8k_cold.log`（T=8192 冷缓存）、`/tmp/bench_disp16k.log`（T=16384 崩溃）

## 1. 时间点 / 上下文

- 上一次同 bench 同工况的读数：[`2026-07-16_1104_BASELINE_mega_flydsl_ep8_dsv3_bench_mi355x.md`](./2026-07-16_1104_BASELINE_mega_flydsl_ep8_dsv3_bench_mi355x.md)（flydsl 也是 0.2.4）。
- 触发事件：在做 [MoE 组件移植选型](../primus-moe/2026-09-07_0900_moonep-vs-ultraep-port-selection.md) 时发现 **MORI 与 Triton-distributed 在 AMD 上的 EP 性能公开数据均为零**，于是决定用自家 MegaMoE 在 MI355X 上先立一个自测基线。
- 结果与预期相反：**没有立起基线，而是撞上一个可复现的回归。**

## 2. 问题

- 目标：在 MI355X 上取得 MegaMoE EP8 dispatch 路径的当前实测水位，用于和 MORI-EP（公布 MI355X EP8 dispatch 345 GB/s）、Triton-distributed AMD fused EP 对标。
- 现状（本次实测）：`grouped/dense` = **8.7%**，XGMI dispatch = **50.8 GB/s**，融合 speedup vs 串行 = **0.83×**。
- 对照（7-16 baseline，**同 bench / 同模式 / 同模型 / 同 EP / 同 T / 同 dtype / 同 flydsl 版本**）：`grouped/dense` **91.9–100%**，XGMI dispatch **368–381 GB/s**，speedup **1.46–1.61×**。
- 卡点：**回归点在软件栈，不在硬件也不在 autotune 缓存**（见 §4 的两条排除性证据）。

## 3. 做了什么

统一配置：DeepSeek-V3，EP8，H=7168，I=2048，E=256，topk=8，**bf16**，`iters=30`，跨 8 rank 取最慢。

| # | 动作 | 命令 / 路径 | 结果 |
|---|---|---|---|
| 1 | 确认 GPU 空闲 | `rocm-smi --showuse --showmemuse` | 8 卡全 0%，容器内仅 `sleep infinity`。⚠ 节点被他人 SLURM 作业（`yihou` 29313）占着 allocation，但 GPU 实际空闲 |
| 2 | 绕开脚本自插路径 | 把 `benchmark/ops/training/*.py` 拷到 `/tmp/mmbench` | 见 §3.1，这是能用系统 turbo 的关键 |
| 3 | T=4096 热缓存 | `python bench_mega_moe.py --mode dispatch_grouped_gemm --num-processes 8 --num-tokens 4096 --iters 30 --models DeepSeek-V3` | 完成 |
| 4 | T=16384 | 同上 `--num-tokens 16384` | **崩溃**，600 s NCCL watchdog 超时 |
| 5 | T=8192 热缓存 | 同上 `--num-tokens 8192` | 完成 |
| 6 | T=8192 **冷** autotune 缓存 | `mv /root/.flydsl /root/.flydsl.bak_20260907` 后重跑 | 完成，**与热缓存几乎一致** |

### 3.1 为什么必须拷到 `/tmp/mmbench`

`bench_mega_moe.py` 第 38 行把仓库根 `os.path.join(_HERE,"..","..","..")` 插到 `sys.path[0]`，于是**源码树会遮蔽系统安装版**：

- `/workspace/Primus-Turbo`：commit `edc8d2cc`（2026-07-21），**树内无 `_C*.so`** → `ModuleNotFoundError: primus_turbo.pytorch._C`
- `/perf_apps/xiaoming/Primus-Turbo`、`/perf_apps/xiaoming/MegaMoE-dev`：树内有 `.so`，但**是对着别的 libtorch 编的** → `ImportError: undefined symbol: _ZNR5torch7Library4_def...`
- `/perf_apps/xiaoming/MegaMoE`：树内 `.so` **ABI 正常**（可作为需要树内构建时的退路）

拷到 `/tmp/mmbench` 后，`_HERE/../../..` 解析成 `/`，其下没有 `primus_turbo` 包，于是自然落到 `/opt/venv` 的系统安装版。

### 3.2 系统安装版是够用的（一处自我更正）

一开始判断"系统 turbo 没有 mega_moe"是**错的**，原因是按旧名字找：

| | 源码树（旧） | 系统安装版 0.4.1.dev33（新） |
|---|---|---|
| 模块名 | `ops/moe/mega_moe_fused.py` | **`ops/moe/fused_mega_moe.py`**（另有 `fused_mega_moe_fp8.py`） |
| 入口 | `mega_moe_fused` | `fused_mega_moe` / `fused_mega_moe_stage1` / `fused_mega_moe_stage2` |
| flydsl | — | `primus_turbo/flydsl/mega/` **齐全**（`ep_intranode` / `dispatch_grouped_gemm_bf16_kernel` / `grouped_gemm_combine_bf16_kernel` / `fp8/` …） |

## 4. 效果

### 4.1 三次运行的读数

| 指标 | T=4096 热 | T=8192 热 | T=8192 **冷缓存** | **7-16 baseline (T=8192)** |
|---|---|---|---|---|
| `dense_gemm` (TFLOPS) | 1206.0 | 1229.3 | 1232.6 | **1230–1283** ✅ 一致 |
| `gemm_only` (TFLOPS) | 97.9 | 107.2 | 107.6 | — |
| **`grouped/dense`** | **8.1%** | **8.7%** | **8.7%** | **91.9–100%** ❌ |
| **XGMI dispatch** | **50.9 GB/s** | 50.8 | 50.8 | **368–381 GB/s** ❌ |
| `fused` (ms) | 33.836 | 66.521 | 65.051 | — |
| overlap roofline | 65.5% | 57.1% | 58.2% | **87.6–92.1%** ❌ |
| **speedup vs 串行** | **0.89×** | **0.81×** | **0.83×** | **1.46–1.61×** ❌ |

反向两段（T=8192 冷缓存）：`bwd dgrad` grouped/dense 16.4%、roofline 69.3%、**0.93×**；`bwd wgrad dW1` grouped/dense **100.1%**、fused 116.315 ms、**融合比串行慢约 5.8×**。

### 4.2 定性

- ✅ **正确性全绿**：三次运行、每次 3 项检查（fwd / bwd dgrad / wgrad dW1）全部 `cos=1.00000 rel=0.0000 PASS`。**功能没坏，坏的是性能。**
- ✅ **硬件与 dense 参考健康**：`dense_gemm` 1206–1233 TFLOPS，与 7-16 的 1230–1283 吻合 ⇒ **GPU、XGMI 物理链路、dense GEMM 路径都正常**。
- ❌ **两处塌陷**：grouped GEMM 掉到 dense 的 8.7%（baseline 91.9–100%），XGMI dispatch 掉到 50.8 GB/s（baseline 368–381，**7.3×**）。
- ❌ **融合全面劣于串行**：0.81×–0.99×，而 baseline 是 1.46×–1.61×。`README_Mega_MoE.md` 宣称的"overlap 85%+、多数 90%+"在当前栈上**不成立**（实测 57.1–74.3%）。
- ⚠️ **排除 autotune 缓存**：`mv ~/.flydsl` 后冷跑，8.7% / 50.8 GB/s / 0.83× 与热跑逐项一致 ⇒ **不是缓存重放的锅**。
- ⚠️ **排除 token 规模区间**：T 从 4096 翻到 8192（每专家 token 128 → 256，跨过 TEMPO 标定的 156–168 分界），`grouped/dense` 只从 8.1% 动到 8.7%，overlap 反而更差。**我先前"是 TEMPO memory-bound 区间所致"的解释不成立，特此更正。**
- ⚠️ `wgrad` 的 `dispatch_only` 读数不稳（8.673 / 16.714 / 185.469 ms 三次差异巨大，冷跑那次导致 roofline 出现 159.5% 这种非物理值）⇒ **该行 comm 计时本身可疑**，但 `wgrad fused` 稳定在 ~116 ms，慢于串行这一结论不受影响。
- ❌ T=16384 跑不完：某 rank 的 JIT 编译超过 600 s，卡在正确性检查的 `all_gather_object`（`bench_mega_moe.py:112`）上被 NCCL watchdog 打断。**不是功能 bug，是超时。**

### 4.2b 同一小时、同一节点的 RCCL all-to-all 参照 —— 独立坐实回归

为排除"是不是今天这台机器的 fabric 本身就慢"，用**同字节量的朴素 RCCL `all_to_all_single`** 做参照
（T=8192、H=7168、topk=8、bf16、EP8，每 rank 对外 egress 0.822 GB；脚本
`/perf_apps/xiaoming/ep-bench/bench_rccl_a2a_dsv3_ep8.py`）：

```
rank 0..7 : 2.492–2.494 ms | 329.6–329.8 GB/s   (8 rank 高度一致)
WORST-RANK: 2.494 ms | 329.6 GB/s
```

| 口径（同节点 MI355X EP8，DSv3 字节量） | XGMI GB/s |
|---|---|
| MegaMoE dispatch，2026-07-16 基准 | **368–381** |
| MORI-EP 公布值（FP8） | 345 |
| **朴素 RCCL all-to-all，2026-09-07 当日实测** | **329.6** |
| **MegaMoE dispatch，2026-09-07 当日实测** | **50.8** |

**结论**：今天这台机器的 XGMI + RCCL 能跑到 329.6 GB/s，**而 MegaMoE 的融合 dispatch 只有 50.8 GB/s
——比朴素 all-to-all 还慢 6.5×**。融合算子的全部意义就是要快过朴素集合通信；现在它慢一个数量级。
硬件与 fabric 当日健康，**回归 100% 在软件侧**，这一条不再需要依赖与 7-16 的跨时间对比。

### 4.3 与外部对标的意义

baseline 的 **368–381 GB/s 高于 MORI 公布的 MI355X EP8 dispatch 345 GB/s**。也就是说 MegaMoE 在正确的栈上是有竞争力的；当前栈上的 50.8 GB/s 不能拿去和任何人对标。**在回归修复前，本节所有数字都不可用于选型结论。**

## 4.4 顺带查明的容器现状（横向对比暂时做不了）

原计划把 MegaMoE 与 DeepEP / MORI-EP / Triton-distributed 在同工况横比，实测发现**做不了**：

| 库 | 容器内状态 |
|---|---|
| **DeepEP** | 装了 `deep_ep 0.1.0`。**kernel 符号是 `uccl::intranode::*`，即 UCCL 集成版**（`initialize_uccl` / `destroy_uccl`）。**dispatch 可用**（T=1024 实测 `recv_x=(5395, 7168)`），**combine 坏**：单次、不循环、T=1024 也挂在 `DeepEP timeout for combine receivers, responsible_channel = 8, expect = 62/-62/57`，随后 `HSA_STATUS_ERROR_EXCEPTION code 0x1016` in `uccl::intranode::combine<hip_bfloat16, 8, 1024, 4096>`。已用最小复现 `smoke_deepep_combine.py` 确认**不是 harness 的问题** |
| **MORI / mori_shmem** | **未安装** |
| **Triton-distributed** | **未安装** |

两条对任何人再跑 DeepEP 都要知道的环境坑：

1. **容器 `ulimit -l` 只有 8 MB**，RDMA 注册直接 `ibv_reg_mr failed: Invalid argument`。
   必须 `docker exec --privileged` + `ulimit -l unlimited`（实测可提到 unlimited）。
2. **不能传 `num_rdma_bytes=0`**：该构建在 `Buffer.__init__` 里**无条件**调用 `initialize_uccl`，
   0 字节会去注册零长 MR 并报 EINVAL。给个非零值（本次用 256 MB）即可，intranode 流量仍走 XGMI。

## 5. 可持续方向

| 优先级 | 方向 | 预期收益 | 风险 / 前置 |
|---|---|---|---|
| **P0** | **二分回归点**：baseline 栈是 torch `2.10.0+git94c6e04` / ROCm 7.2.1 / triton 3.6.0 / 仓库 `#412`；当前是 torch `2.12.0+rocm7.15` / HIP 7.15 / triton 3.7.0 / `0.4.1.dev33`。先固定 flydsl 0.2.4 不动，只换 turbo 版本，再只换 torch 镜像 | 定位到单一变量 | 需要能取到 `rocm/primus:v26.3` 镜像 |
| **P0** | 用 `/perf_apps/xiaoming/MegaMoE`（树内 `.so` ABI 正常）在本容器复跑同工况 | 区分「turbo 版本」与「容器 torch」两个嫌疑 | 该树 `.so` 与本容器 torch 的兼容性已验证可 import，但未验证数值 |
| P1 | 单独定位 XGMI dispatch 50.8 vs 368 GB/s | 通信塌陷可能独立于 GEMM 塌陷，也可能同源 | 需要 `bench_deepep_intranode.py` / `bench_symmetric_memory.py` 做交叉印证 |
| P1 | 修 `wgrad` 的 `dispatch_only` 计时不稳 | 让 roofline 指标恢复可信（现在会出 159.5%） | 读 bench 的计时实现 |
| P2 | 给 T=16384 提高 PG timeout 或预热编译 | 打开更大规模的观测窗口 | `timeout=600000` 在脚本内硬编码，需改 |
| P2 | 补跑 `--mode grouped_gemm_combine` | 本次只覆盖了 dispatch 一侧，combine 侧未知 | — |

## 相关文件

- 复现脚本（容器内）：`/tmp/mmbench/bench_mega_moe.py`（拷自 `/perf_apps/xiaoming/Primus-Turbo/benchmark/ops/training/`）
- 对照脚本（共享盘，容器可见）：`/perf_apps/xiaoming/ep-bench/bench_rccl_a2a_dsv3_ep8.py`（RCCL a2a 参照）、
  `bench_deepep_dsv3_ep8.py`（DeepEP，combine 受阻，`--skip-combine` 可只测 dispatch）、
  `smoke_deepep_combine.py`（combine 挂死的最小复现）
- 结果 CSV（容器内）：`dispatch_grouped_gemm_20260907_MI355X.csv`
- autotune 缓存备份（容器内，可还原）：`/root/.flydsl.bak_20260907`（844 MB）
- 对照 baseline：[`2026-07-16_1104_BASELINE_mega_flydsl_ep8_dsv3_bench_mi355x.md`](./2026-07-16_1104_BASELINE_mega_flydsl_ep8_dsv3_bench_mi355x.md)
- 触发本次工作的选型文档：[`../primus-moe/2026-09-07_0900_moonep-vs-ultraep-port-selection.md`](../primus-moe/2026-09-07_0900_moonep-vs-ultraep-port-selection.md)
