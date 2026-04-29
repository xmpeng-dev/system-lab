# <ModelName> — <RunFlavor> Trace 分析

| 字段 | 值 |
| --- | --- |
| 日期 | <YYYY-MM-DD> |
| 模型 | <e.g. Llama-2-70B · LoRA SFT · `bf16_with_fp8_hybrid`> |
| 硬件 | <e.g. 8 × MI355X (288 GiB HBM, gfx950)> |
| 并行 | <e.g. TP1 · PP1 · CP1 · EP1 · DP=8（纯 DDP + DistOpt）> |
| Batch | <e.g. GBS 8 · MBS 1 · seq 8192 (packed)> |
| 训练规模 | <e.g. 总参 69.0 B / 可训参 44.5 M (LoRA, 0.06%)> |
| 100-iter 实测 | step **<X.XX> s** · **<TFLOPS>/GPU** · wall <Y.Y> s |
| 分析窗口 | `ProfilerStep#<N>` (warmup 后稳态), <stepMs> ms |
| 关键结论 | (1) <FP8 GEMM share>; (2) <NCCL overlap finding>; (3) <idle gap finding>; (4) **VRAM <pct> reserved (<verdict>)**, <risk>。 |

---

## 1. 目标 / 背景

- 这次跑的 motivation（什么问题，为什么要 trace）。
- 上一次（baseline / 上一版优化）的状态。
- 决定 trace 的版本范围 / 提交 / 时间。

## 2. 跑 baseline + 抓 trace

```bash
docker exec <container> bash -c '
cd <repo>/llama2_sft/primus &&
TRACE=1 PRIMUS_TRAIN_ITERS=100 PRIMUS_EVAL_INTERVAL=9999 \
PRIMUS_PROFILE_STEP_START=80 PRIMUS_PROFILE_STEP_END=85 \
RUN_LOG_FILE=/results/<runname>/<runname>.log \
bash run.sh'
```

产物：

| 文件 | 内容 |
| --- | --- |
| `/results/torch_profiler_traces/<...>.pt.trace.json` | <size> Kineto trace, rank 0 |
| `/results/<runname>/<runname>.log` | 完整训练日志 (`train_utils.py:671` 显存) |
| `/results/<runname>/breakdown_step<N>.txt` | `full_breakdown.py` 完整输出 |

## 3. Per-stream busy time

| pid | stream | 角色 | busy | share |
| --- | --- | --- | ---: | ---: |
| <pid> | 0  | 计算（GEMM / FMHA / norm / elem / activation） | <ms> ms | **<pct>%** |
| <pid> | <id> | RCCL DDP 梯度同步 (`Generic_1`) | <ms> ms | <pct>% |

- 流 oversubscription = <pct>%。
- Idle gap <ms> ms (~<pct>%)。

## 4. Kernel 类别分解（重新归类后）

> 默认 `full_breakdown.py` 的 `cat_kernel` 把 `Custom_Cijk_*` 全划进 `other`。
> 手动把 4 个 FP8 GEMM kernel 重新归到 GEMM。

| 类别 | ms | % step | 备注 |
| --- | ---: | ---: | --- |
| **FP8 GEMM** (`Custom_Cijk_*_F8*BS_*`) | **<ms>** | **<pct>%** | 4 个 shortname 变体 |
| **FlashAttention** (`aiter::fmha_*`) | **<ms>** | **<pct>%** | bwd <ms> + fwd <ms> |
| Elementwise / dropout | <ms> | <pct>% | |
| TE SwiGLU / unary | <ms> | <pct>% | |
| **RCCL grad-sync** | <ms> | <pct>% | overlap <pct>% |
| bf16 GEMM | <ms> | <pct>% | LoRA-A/B 等小矩阵 |
| QKV-RoPE / cast / norm / reduction / memcpy / idle | … | … | |

## 5. Top-10 单 kernel

| # | ms | % step | kernel | 桶 |
| ---: | ---: | ---: | --- | --- |
| 1 | <ms> | <pct>% | <name> | GEMM fwd |
| … | | | | |

## 6. Compute / NCCL Overlap

| 指标 | 值 |
| --- | ---: |
| compute-only 时长 | <ms> |
| nccl-only 时长 | <ms> |
| overlap (compute & nccl) | **<ms>** |
| idle | <ms> |
| **NCCL hidden behind compute** | **<pct>%** |

## 7. VRAM (HBM)

> rank 0，`train_utils.py:671` after iter <N>. trace
> `deviceProperties[*].totalGlobalMem = <bytes> = <GB> = <GiB>`.

| 指标 | 值 | 阈值判定 |
| --- | ---: | --- |
| Reserved peak (Rmax) | <GB> → **<pct>%** | <verdict> |
| Allocated peak (Pmax) | <GB> → **<pct>%** | <verdict> |
| Headroom to OOM | <GB> | <verdict> |
| Fragmentation | <pct>% | <verdict> |
| Allocator retires | <N> | <verdict> |

### 7.1 Bucket 分解（估算，校准到 Pmax = <GB>）

| Bucket | GB | % of Pmax | 备注 |
| --- | ---: | ---: | --- |
| Weights | <GB> | <pct>% | dtype |
| Activations | <GB> | <pct>% | seq × layers, recompute=… |
| LoRA grads + Adam state | <GB> | <pct>% | trainable=… |
| TE FP8 caches / cuBLAS / NCCL bufs | <GB> | <pct>% | |
| Allocator slack (Rmax − Pmax) | <GB> | n/a | 不计入 Pmax |
| 未归类 | <GB> | <pct>% | sanity gap |

## 8. 可立即 actionable 的改动

| # | 改动 | 预期收益 | 风险 / 代价 | 实测结果 |
| ---: | --- | --- | --- | --- |
| 1 | <e.g. DDP overlap> | <e.g. ~3% step time> | <risk> | <未测 / 实测结果> |
| 2 | <e.g. selective recompute> | <e.g. ~30-50 GB 显存> | <e.g. 5-15% 吞吐> | <未测> |
| 3 | <e.g. dataloader workers> | <e.g. ~5-10%> | <risk> | <未测> |

### 8.1 <如果做了 A/B 实验，把负面结果也保留作为反例>

## 9. 复现与产物

```bash
# 1) 跑 baseline + 抓 trace
…

# 2) 跑 kernel 分析
…

# 3) 抓显存
grep "mem-max-" /results/<runname>/<runname>.log | tail -3
```

## 10. 相关文件

- canvas: `~/.cursor/projects/home-xiaompen-mlperf-training/canvases/<runname>.canvas.tsx`
- trace: `<path>`
- log: `<path>`
- breakdown: `<path>`

## 11. 后续 TODO

- [ ] <下一个 A/B 实验>
- [ ] <需要复测的环境/版本>
