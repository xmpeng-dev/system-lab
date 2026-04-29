# Llama2-70B LoRA SFT — Baseline (DDP) Trace 分析

| 字段 | 值 |
| --- | --- |
| 日期 | 2026-04-29 |
| 模型 | Llama-2-70B · LoRA SFT · `bf16_with_fp8_hybrid` |
| 硬件 | 8 × MI355X (288 GiB HBM, gfx950) |
| 并行 | TP1 · PP1 · CP1 · EP1 · DP=8（纯 DDP + DistOpt） |
| Batch | GBS 8 · MBS 1 · seq 8192 (packed) |
| 训练规模 | 总参 69.0 B / 可训参 44.5 M (LoRA, 0.06%) |
| 100-iter 实测 | step **1.62 s** · **2251 TFLOP/s/GPU** · wall 220.6 s |
| 分析窗口 | `ProfilerStep#82` (warmup 后稳态), 1626.45 ms |
| 关键结论 | (1) FP8 GEMM 占 **49.9%**, FlashAttention 占 **19.7%**，~70% 已是大 kernel；(2) **NCCL 完全串行** — 54 ms 梯度同步在 backward 之后串跑，0.4% 与 compute 重叠；(3) Step 起始 **191 ms idle (~12%)**，疑似 dataloader/dispatcher gap；(4) **VRAM 95.6% reserved (TIGHT)**，eval / ckpt save / seq 抖动会 OOM。 |

---

## 1. 目标 / 背景

- 之前 FSDP2 路线连续踩了 5 层 patch (FP8 attrs / DeviceMesh dim names / trivial DTensor unwrap / DistOpt 不兼容 / LoRA forward shape)，最终因为 LoRA + FP8 + FSDP2 在 PyTorch 内部 sharding 路径上的根本性不兼容而搁置。
- 改走 **baseline (DDP)** 路线，先把现状摸清楚：每个 step 时间花在哪、有没有明显瓶颈、显存逼近 cap 是不是真的健康。
- 后续优化（开 `overlap_grad_reduce`、selective recompute、调 dataloader）都基于本 trace 的数字。

## 2. 跑 baseline + 抓 trace

```bash
docker exec xiaoming-mlperf-llama bash -c '
cd /home/xiaompen/mlperf-training-llama/llama2_sft/primus &&
TRACE=1 PRIMUS_TRAIN_ITERS=100 PRIMUS_EVAL_INTERVAL=9999 \
PRIMUS_PROFILE_STEP_START=80 PRIMUS_PROFILE_STEP_END=85 \
RUN_LOG_FILE=/results/baseline_trace_run/baseline_trace_run.log \
bash run.sh'
```

`run.sh` 里的 `TRACE=1` 分支会自动注入：

```yaml
profiling.use_pytorch_profiler: true
logger.tensorboard_dir: /results/torch_profiler_traces
PRIMUS_PROFILE_STEP_START: 80   # 跳过 cuda graph capture / 编译 warmup
PRIMUS_PROFILE_STEP_END:   85
PRIMUS_TRAIN_ITERS:        100  # 抓完 trace 再跑 ~15 iter 自然结束
```

产物：

| 文件 | 内容 |
| --- | --- |
| `/results/torch_profiler_traces/smci355-..._145319.....pt.trace.json` | 650 MB Kineto trace, rank 0 |
| `/results/baseline_trace_run/baseline_trace_run.log` | 完整训练日志 (`train_utils.py:671` 显存) |
| `/results/baseline_trace_run/breakdown_step82.txt` | `full_breakdown.py` 完整输出 |

## 3. Per-stream busy time

| pid | stream | 角色 | busy | share |
| --- | --- | --- | ---: | ---: |
| 2 | 0  | 计算（GEMM / FMHA / norm / elem / activation） | 1377.7 ms | **84.7%** |
| 2 | 33 | RCCL DDP 梯度同步 (`Generic_1`) | 54.4 ms | 3.3% |

- 流 oversubscription = 88% — **没跨流并行**（compute 只跑一条流）。
- Idle gap 191 ms (~12%)，集中在 step 头 ~180 ms（dataloader/dispatcher）和尾端的几个 ms。

## 4. Kernel 类别分解（重新归类后）

> 默认 `full_breakdown.py` 的 `cat_kernel` 把 `Custom_Cijk_*` 全划进 `other`（启发式只匹配 `cijk_*`，没匹 `custom_cijk_*`）。手动把 4 个 FP8 GEMM kernel 重新归到 GEMM。

| 类别 | ms | % step | 备注 |
| --- | ---: | ---: | --- |
| **FP8 GEMM** (`Custom_Cijk_Alik_Bljk_F8*BS_*`, hipBLASLt Tensile) | **773.5** | **47.6%** | 4 个 shortname 变体覆盖 q/k/v/o + gate/up/down |
| **FlashAttention** (`aiter::fmha_fwd/bwd_hd128_bf16_causal_*`) | **320.2** | **19.7%** | bwd 213ms（含 `dk_dv_reduce` 8.6 ms） + fwd 91 ms + odo 8 ms |
| Elementwise / dropout (`vectorized_elementwise_kernel`, `fused_dropout_kernel_vec`) | 122.8 | 7.6% | |
| TE SwiGLU / dgated_act / unary (`gated_act_kernel<silu>`, `unary_kernel`) | 64.1 | 3.9% | |
| **RCCL grad-sync** (`ncclDevKernel_Generic_1`) | 54.4 | 3.3% | **完全串行** |
| bf16 GEMM (`Cijk_Ailk_Bljk_BBS / BSS`) | 38.5 | 2.4% | LoRA-A/B 等小矩阵 |
| Fused QKV-RoPE (fwd+bwd) | 23.3 | 1.4% | |
| FP8 cast / transpose (`cast_transpose_triton`, `transpose_optimized_kernel`) | 21.2 | 1.3% | |
| RMSNorm (triton fwd+bwd) | 20.3 | 1.2% | |
| Reduction | 8.6 | 0.5% | |
| MemCopy / D2D | 4.5 | 0.3% | |
| **Idle / 未归类** | **~175** | **~10.8%** | dataloader / 启动延迟 |

## 5. Top-10 单 kernel

| # | ms | % step | kernel | 桶 |
| ---: | ---: | ---: | --- | --- |
| 1 | 350.4 | 21.5% | `Custom_Cijk_Alik_Bljk_F8B8BS_BH_SAB_NTD_UserArgs_shortname1_gfx950` | GEMM fwd |
| 2 | 279.9 | 17.2% | `Custom_Cijk_Alik_Bljk_F8BS_BH_SAB_NTD_UserArgs_shortname1_gfx950` | GEMM bwd |
| 3 | 212.5 | 13.1% | `aiter::fmha_bwd_hd128_bf16_causal_a16_psskddv` | Attention bwd |
| 4 | 104.2 |  6.4% | `Custom_Cijk_Alik_Bljk_F8BS_BH_SAB_NTD_UserArgs_shortname0_gfx950` | GEMM |
| 5 |  90.6 |  5.6% | `aiter::fmha_fwd_hd128_bf16_causal` | Attention fwd |
| 6 |  54.4 |  3.3% | `ncclDevKernel_Generic_1` | Collective |
| 7 |  43.4 |  2.7% | `vectorized_elementwise_kernel<CUDAFunctor_add bf16>` | Elementwise |
| 8 |  39.0 |  2.4% | `Custom_Cijk_Alik_Bljk_F8B8BS_BH_SAB_NTD_UserArgs_shortname0_gfx950` | GEMM |
| 9 |  34.1 |  2.1% | `transformer_engine::gated_act_kernel<silu>` | Activation |
| 10 |  30.5 |  1.9% | `transformer_engine::dgated_act_kernel<silu>` | Activation |

> Top-5 合计 ≈ 1038 ms = **63.8% of step**。前 4 个 FP8 GEMM 合计 773.5 ms = 47.6%。
> 4 个 GEMM kernel 各对应不同 shortname（0/1）和 fp8 type (`F8B8BS`=fwd 双向 fp8 + bf16 scale, `F8BS`=fwd 单向 fp8 + bf16 scale），覆盖 fwd/bwd 与 q/k/v/o vs gate/up/down 的不同 shape。

## 6. Compute / NCCL Overlap

| 指标 | 值 |
| --- | ---: |
| compute-only 时长 | 1380.4 ms |
| nccl-only 时长 | 54.4 ms |
| overlap (compute & nccl) | **0.2 ms** |
| idle | 191.6 ms |
| **NCCL hidden behind compute** | **0.4%** (0.2 / 54.6 ms) |

- 时间分箱（每 bin = 20.3 ms）显示：bin 77-79（step 末 60 ms）几乎是纯 NCCL（17-20 ms NCCL / bin），前面 76 个 bin 完全没 NCCL。
- 这是典型的 **DDP 默认未开 grad-reduce overlap** 行为：所有 grad reduce 排到 backward 完成之后串行跑。
- 可立即拿回的收益：~54 ms / 1626 ms = **~3.3% step time**。

## 7. VRAM (HBM)

> rank 0，`train_utils.py:671` after iter 10。trace `deviceProperties[*].totalGlobalMem = 309 220 868 096 B = 309.22 GB = 287.99 GiB`。

| 指标 | 值 | 阈值判定 |
| --- | ---: | --- |
| Reserved peak (Rmax, driver-side) | 295.52 GB → **95.6%** | 🟡 **TIGHT** (95-98%) |
| Allocated peak (Pmax, working set) | 285.84 GB → **92.4%** | 🟡 |
| Headroom to OOM | **13.7 GB** | 🟡 |
| Fragmentation (Rmax−Pmax)/Rmax | 3.3% | 🟢 优秀 (<5%) |
| Allocator retires | 0 | 🟢 |

判定 = **TIGHT**：静态 shape 跑得动，但 eval（更长 batch）、checkpoint save（多一份 host buffer）、或 packed-seq 抖动都会 OOM。

### 7.1 Bucket 分解（估算，校准到 Pmax = 285.84 GB）

| Bucket | GB | % of Pmax | 备注 |
| --- | ---: | ---: | --- |
| Weights (bf16 + fp8 hybrid) | ~120 | ~42% | 70B 参数，部分以 fp8 存 |
| Activations (无 recompute) | ~145 | ~51% | seq=8192 × layers=80 × bf16，主导 |
| LoRA grads + Adam state | 0.8 | 0.3% | 44.5M trainable，DistOpt 进一步切 8 |
| TE FP8 caches / cuBLAS workspace / NCCL bufs | ~12 | ~4% | |
| Allocator slack (Rmax − Pmax) | 9.7 | n/a | 不计入 Pmax，但占 cap |
| 未归类 | ~8 | ~3% | sanity gap |

> Activation 占比 ~51%，在「无 recompute + seq=8192 + packed」的 LoRA 训练里属于正常。

## 8. 可立即 actionable 的改动

| # | 改动 | 预期收益 | 风险 / 代价 | 实测结果 |
| ---: | --- | --- | --- | --- |
| 1 | ~~打开 DDP grad-reduce overlap~~ → 已默认开启，且实测**无效**（详见 §8.1） | ~~~3% step 时间~~ | — | **+0.4% slower** |
| 2 | Selective activation recompute (`recompute_granularity: selective`) | ~30-50 GB 显存空间（开 eval / 升 mbs / 升 seq） | 5-15% 吞吐损失 | 未测 |
| 3 | 排查 step 起始 ~170-190 ms idle | 最多再省 ~5-10% | 调 `num_workers` / `prefetch_factor` / `persistent_workers`，可能改 dataloader | 未测 |
| 4 | Sequence parallelism (`sequence_parallel: true`) | LayerNorm/dropout activation -10-20%，几乎免费 | 需要 TP>1 才能开（当前 TP=1，不适用） | n/a |

### 8.1 DDP overlap 实测：无效（保留作为反例）

跑了 `OVERLAP=1`（`ddp.bucket_size=4 MB` + `ddp.overlap_grad_reduce=true` + `ddp.overlap_param_gather=true` + `optimizer.overlap_param_gather=true` + `optimizer.overlap_param_gather_with_optimizer_step=true`）后做了 1:1 100-iter A/B：

| 指标 | baseline (128 MB bucket) | overlap-tuned (4 MB + with_step) | Δ |
| --- | ---: | ---: | ---: |
| ProfilerStep#82 wall | **1626.45 ms** | **1633.72 ms** | **+7.3 ms (+0.4%)** |
| 稳态 iter (10 个采样) | ~1620 ms | ~1627 ms | +7 ms |
| Grad reduce-scatter kernel | 54.12 ms @ t+1568 | **53.78 ms @ t+1576** | ≈0 |
| Param all-gather kernel | 0.24 ms @ t+187 | **30.03 ms @ t+164** | **+30 ms（变慢）** |
| NCCL hidden behind compute | 0.4% | **0.1%** | 更差 |
| 编译/计算流 busy | 84.7% | 84.4% | ≈ |
| Idle (主要在 step 起始) | 191.6 ms | 168.9 ms | -22.7 ms |

为什么不灵：

1. `comm_overlap.setup()` **本来就**已经 push `overlap_grad_reduce=True` / `overlap_param_gather=True` 到 DDPConfig，原始建议描述本身就是误读 —— flag 已开。
2. LoRA 只有 ~89 MB 可训练梯度；切成 4 MB × 22 桶后，RCCL 在这台 MI355X 上仍然把它们融合成单个 `ncclDevKernel_Generic_1`，duration 几乎不变（54 ms）。
3. **AMD/RCCL 与 compute 抢 CU**（不像 NVIDIA NCCL 走专用 SM）→ 即便提前发起 reduce-scatter，backward GEMM/attention 占满 CU 时 NCCL 实际跑不动 → trace 里只剩 0.1% 真正的 wall-clock overlap。
4. `overlap_param_gather_with_optimizer_step=true` 把原本 0.24 ms 的合并 all-gather **拆成 22 次小 gather**（30 ms），反而拖慢 step 起始。

> 结论：在「LoRA + DistOpt + AMD RCCL」这一组合下，DDP/optimizer 端的 overlap 旋钮**没有可榨取的空间**，54 ms NCCL 尾巴属于 RCCL launch + Generic_1 固定开销，真正想砍只能换 NCCL 拓扑/算法（`NCCL_ALGO`、`NCCL_PROTO`、IB tuning）或进一步缩 LoRA grad volume。
>
> 该实验用 `OVERLAP=1` env 触发 `run.sh` 中可选分支，**默认关闭**；保留代码方便后续在不同硬件/拓扑上复测。

> 不建议短期再碰 FSDP2 — Bridge + LoRA + TE-FP8 在 PyTorch FSDP2 内部的 DTensor sharding 路径还有未解的根本兼容性问题，详见 `2026-04-29_primus_fsdp2_patch_layers.md`（如果之后归档）。

## 9. 复现与产物

```bash
# 1) 跑 baseline + 抓 trace（容器内）
docker exec xiaoming-mlperf-llama bash -c \
  'cd /home/xiaompen/mlperf-training-llama/llama2_sft/primus && \
   TRACE=1 PRIMUS_TRAIN_ITERS=100 PRIMUS_EVAL_INTERVAL=9999 \
   PRIMUS_PROFILE_STEP_START=80 PRIMUS_PROFILE_STEP_END=85 \
   RUN_LOG_FILE=/results/baseline_trace_run/baseline_trace_run.log \
   bash run.sh'

# 2) 跑 kernel 分析
docker exec xiaoming-mlperf-llama bash -c \
  'pip install ijson && \
   cd /home/xiaompen/mlperf-training-llama && \
   python3 .cursor/skills/gpu-trace-analysis/scripts/full_breakdown.py \
     /results/torch_profiler_traces/smci355-..._145319.....pt.trace.json \
     ProfilerStep#82'

# 3) 抓显存（直接 grep 训练日志）
grep "mem-max-" /results/baseline_trace_run/baseline_trace_run.log | tail -3
```

## 10. 相关文件

- 训练日志：`/results/baseline_trace_run/baseline_trace_run.log`（baseline）、`/results/overlap_trace_run/overlap_trace_run.log`（overlap A/B）
- 完整 trace（650 MB）：
  - baseline：`/results/torch_profiler_traces/smci355-..._145319.1777433707462544573.pt.trace.json`
  - overlap A/B：`/results/torch_profiler_traces/smci355-..._157204.1777440043277941766.pt.trace.json`
- Kernel breakdown 文本输出：`/results/baseline_trace_run/breakdown_step82.txt`、`/results/overlap_trace_run/breakdown_step82_overlap.txt`
- 可视化 canvas：`~/.cursor/projects/home-xiaompen-mlperf-training/canvases/llama2-70b-lora-baseline-trace.canvas.tsx`
- 分析脚本：`.cursor/skills/gpu-trace-analysis/scripts/full_breakdown.py`
- 启动脚本：`llama2_sft/primus/run.sh`（`TRACE=1` 分支）
- Yaml 配置：`llama2_sft/primus/llama2_70b_lora_posttrain.yaml`

## 11. 备注 — FSDP2 路线现状（仅记录，不展开）

今天前半天试图打通 PyTorch FSDP2 + Megatron-Bridge + LoRA + TE-FP8，沿着错误链层层 patch：

1. `pg_collection` 参数名不匹配 → `BridgeTorchFullyShardedDataParallel` shim ✅
2. TE Float8 缺 `_fp8_attrs` → `_patch_primus_fsdp2_fp8_attrs_guard` ✅
3. `DeviceMesh.from_group` 不带 `mesh_dim_names` → `_patch_device_mesh_from_group_default_dim_name` ✅
4. `_concatenate` 拒绝重叠 mesh（DTensor mesh == DP world）→ `_patch_fsdp2_unwrap_trivial_dtensor` ✅
5. `DistributedOptimizer` 拿到 `nn.Module.buffers` 方法 → `run.sh` 加 `optimizer.use_distributed_optimizer=false` ✅
6. **LoRA-A 被切到 rank/8=2，LoRA-B 仍是 16 → forward `(8192,2)@(16,1280)` shape 不匹配 ❌**

第 6 层是 PyTorch FSDP2 在 `(8,1)` mesh + `(StridedShard, Replicate)` placement 下的 all-gather 行为问题，超出 patch 能修的范围。所有 patch 已留在 `Primus-dev/primus/backends/megatron_bridge/patches/torch_fsdp2_patches.py`，待 FSDP2 上游或 Bridge 修复 LoRA 路径后可直接复用。
