# Ulysses GemmA2A / A2AGemm on MI355X — 融合 kernel 工作总结

用 FlyDSL 把 Ulysses attention 前后两次 all-to-all 与投影 GEMM 融合成 GemmA2A / A2AGemm 两个 kernel，覆盖前向和反向四个位置。代码在 Primus-Turbo `feat/ulysses-gemm-a2a`（最终提交 `50d6c80a`），训练接入在 Primus `feat/ulysses-fused-a2a`（`turbo_ulysses_fused_a2a: 1`，-1 为串行对照；数据采集时用的是当时的环境变量 `PRIMUS_ULYSSES_FUSED_A2A=1`）。

主线模型 Qwen3-30B-A3B，CP4 / EP8，单节点 8× MI355X：

| 长度 | kernel（四个位置之和，不融合 → 融合 µs） | 端到端 base → fused（ms/iter） | 端到端变化 |
|---|---|---|---|
| 16K | 1,675 → 1,034（1.62×） | 3,157 → 2,815 | −10.8% |
| 32K | 3,103 → 1,807（1.72×） | 6,581 → 5,979 | −9.2% |
| 64K | 5,927 → 3,330（1.78×） | 17,350 → 16,004 | −7.8% |
| 128K | 11,489 → 6,598（1.74×） | 55,942 → 53,005 | −5.3% |

其它模型（同一份最终代码，16K / 32K）：Qwen3-235B-A22B −5.9% / −6.6%，GLM-4.7 CP4 −6.7% / −6.2%，MiniMax-M2.5 −4.8% / −5.2%（均为 4 层 proxy），Llama3.1-8B −4.3% / −3.6%（attention 统一用 AITER），GLM-4.7 CP8 −1.9% / −3.7%。

## 文件

| 路径 | 内容 |
|---|---|
| `summary_figure.html` / `.png` | 1920×1080 总结图：出发点 → kernel → 端到端 → 其它模型 |
| `work_summary.html` | 长文页：出发点、实现、优化（13 轮）、最后的效果、提交清单 |
| `data/kernel/overlap_round-{1,3,4,9,10}.txt` | `bench_ulysses_overlap.py` 结果（Qwen3-30B-A3B n=4，16K / 32K），各保留轮次 |
| `data/kernel/overlap_final_64k_128k.txt` | 最终代码 64K / 128K 的同一 bench |
| `data/kernel/performance_trend.md`、`optimize.md` | 逐轮优化记录（含回退轮次） |
| `data/e2e/*_fix_*.log` | 最终代码端到端运行，只保留配置、patch 是否生效、每步耗时 / loss / 显存 |
| `data/e2e/parsed_all.txt` | 上面各运行的汇总（第 4–12 步中位数、第 12 步 loss、patch 状态） |
| `data/scripts/` | 端到端驱动（`run_e2e.sh`、`e2e_*.sh`、`long_runs.sh`）、解析（`parse_e2e.py`）、kernel bench 包装（`run_bench.sh`、`uly_bench.py`） |

## 口径

- kernel：每个长度的不融合和融合在同一次运行里测，每次计时前 barrier 对齐，取中位数再取各 rank 最大值；16K / 32K 为 R10 代码（`50d6c80a` 对 Qwen3-30B-A3B 无改动），64K / 128K 为最终代码。
- 端到端：TP1 / CP4 / EP8，MBS 1，GBS 16，mock 数据，12 步，取第 4–12 步中位数；base 与 fused 在同一节点上串行跑，不用 CUDA graph。
- 端到端收益里含 patch 路径换用 FlyDSL attention 的一部分（Qwen3 形状上 FlyDSL 比 AITER 快）；GLM-4.7、MiniMax-M2.5 两侧都用 AITER，收益全部来自融合。
- 文件名里的 `fix` 指 Turbo `50d6c80a`。
