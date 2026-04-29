# NeMo vs Primus — Llama-2-70B LoRA SFT 配置 / 性能对比

> 8 × MI355X · GBS=8 · MBS=1 · seq=8192 (packed) · TP=PP=CP=1 · DP=8
>
> 配套产物
> - Primus baseline trace 分析: [`2026-04-29_llama2_70b_lora_baseline_trace_breakdown.md`](./2026-04-29_llama2_70b_lora_baseline_trace_breakdown.md)
> - NeMo 详细配置档: [`2026-04-29_nemo_llama2_70b_lora_config.md`](./2026-04-29_nemo_llama2_70b_lora_config.md)
> - 191 ms idle 根因: [`2026-04-29_idle_191ms_layer1_dataloader_root_cause.md`](./2026-04-29_idle_191ms_layer1_dataloader_root_cause.md)
> - 可视化 canvas: `~/.cursor/projects/home-xiaompen-mlperf-training-llama/canvases/nemo-vs-primus-llama2-70b-lora-trace.canvas.tsx`

---

## 0. TL;DR

| 维度 | Primus | NeMo | 差距 |
| --- | ---: | ---: | ---: |
| Step time (ProfilerStep) | **1626 ms** | **1490 ms** | NeMo 快 **8.4 %** |
| Step time (production MLLOG) | — | 1.502 s / 5.40 sps | — |
| Compute-stream busy | 84.9 % | 99.5 % | +14.6 pp |
| Idle (DataLoader) | **191 ms / 11.8 %** | 7 ms / 0.5 % | **−184 ms** |
| Standalone FP8 transpose | 21 ms | 1232 ms | NeMo 多 ~58× |
| HtoD memcpy / step | 4 ms | 1195 ms | NeMo 多 ~280× |
| RCCL on critical path | 54 ms (DDP overlap on, tail-RS 暴露) | 5 ms (LoRA-A2A path, no DDP RS) | −49 ms |
| 收敛 | 未跑到 target | **0.9244 @ step 384, 10.79 min** | NeMo 已端到端达标 |
| Peak VRAM (Pmax) | 285.84 GB | ~200 GB | NeMo 省 ~85 GB |

> **一句话**：NeMo 在 step time 上赢 8 %，几乎全部来自 **DataLoader 单进程 vs 8 worker 持久化 prefetch**（−191 ms 的 idle 洞）；NeMo 在 kernel mix 上输（不开 TE op-fuser → 1.2 GB/step 的 HtoD + 1193 ms 独立 transpose），但因为完全 overlap 在 compute 后面所以不影响 wall-clock。所有 NVTE / CK V3 attention / FP8 hybrid 设置两边完全一致。

---

## 1. 完全相同的部分（先确认对比公平）

跑同一个 model / dataset / parallelism / numerics — `diff` 完两侧的 sh-style config + recipe 后剩下下面这些是**严格相等**的：

### 1.1 Model architecture & numerics

| 字段 | 值 |
| --- | --- |
| `num_layers` | 80 |
| `hidden_size` | 8192 |
| `num_attention_heads` | 64 |
| `num_query_groups` (GQA) | 8 |
| `kv_channels` | 128 |
| `ffn_hidden_size` | 28672 (SwiGLU) |
| `seq_length` | 8192 |
| `normalization` | RMSNorm |
| `position_embedding_type` | RoPE (`apply_rope_fusion=True`) |
| `precision` | bf16-mixed + FP8 hybrid (E4M3 fwd / E5M2 bwd) |
| `fp8_recipe` | DelayedScaling |
| `fp8_dot_product_attention` | **False (两侧都关)**，尽管 `NVTE_FP8_DPA_BWD=1` 都设了 |
| `bias_dropout_fusion` | True |
| `apply_rope_fusion` | True |
| `recompute_granularity` | None |
| `cpu_offloading` / `enable_cuda_graph` | False |

### 1.2 LoRA & optimizer & schedule

| 字段 | 值 |
| --- | --- |
| LoRA `r` (`dim`) | 16 |
| LoRA `alpha` | 32 |
| LoRA `dropout` | 0.1 |
| LoRA `target_modules` | `[linear_qkv, linear_proj]` |
| LoRA `a2a_experimental` | **True (两侧都开)** — Bridge 走 `peft/lora.py`，NeMo 走 env `LORA_A2A=1` |
| Optimizer | AdamW (mcore) |
| `lr` / `wd` / `clip_grad` | 4e-4 / 1e-4 / 0.3 |
| `betas` / `eps` | (0.9, 0.999) / 1e-8 |
| Scheduler | Cosine, max_steps=1024, warmup=0, min_lr=0 |
| `use_distributed_optimizer` | True (mcore ZeRO-1) |

### 1.3 Parallelism & DDP

| 字段 | 值 |
| --- | --- |
| TP / PP / CP / EP | 1 / 1 / 1 / 1 |
| Sequence parallel | False |
| DP | 8 |
| FSDP | False |
| `nccl_ub` | False (两侧都关) |
| `gradient_as_bucket_view` | True (两侧都开) |
| `use_distributed_optimizer` | True (两侧都开，mcore ZeRO-1) |

### 1.4 NVTE / TE flags（**12 个完全一致**，是这次对比最重要的 invariant）

```
NVTE_FUSED_ATTN=1
NVTE_FUSED_ATTN_CK=1
NVTE_FUSED_ATTN_AOTRITON=1
NVTE_CK_USES_FWD_V3=1            # 两侧都跑 CK V3 attention
NVTE_CK_USES_BWD_V3=1
NVTE_CK_IS_V3_ATOMIC_FP32=0
NVTE_RS_STRIDED_ATOMIC=2
NVTE_FP8_DPA_BWD=1               # 但 fp8_dot_product_attention=False，没启
NVTE_USE_HIPBLASLT=1
NVTE_USE_CAST_TRANSPOSE_TRITON=1
NVTE_USE_OPTIMIZED_HIPIFIED_CAST_TRANSPOSE=0
NVTE_USE_RMSNORM_TRITON=1
```

实测 trace 里的 attention kernel 也完全一样：`aiter::fmha_*_hd128_bf16_causal_a16_psskddv` (CK V3)。

### 1.5 FP8 weight transpose cache（容易误读，记一笔）

注意有**两个**不同的 cache flag，名字相像：

| 层级 | 字段 | Primus | NeMo |
| --- | --- | --- | --- |
| TE per-Linear | `keep_fp8_weight_transpose_cache` (TELinear / TELayerNormColumnParallelLinear init 参数) | **False**（yaml `no_fp8_weight_transpose_cache: true` 触发 patch `patch_te_linear_fp8_cache` / `patch_te_layernorm_linear_fp8_cache` 在 init 期把它设 False）| **False**（env `ENABLE_TRANSPOSE_CACHE=0`）|
| Mcore DDP bucket | `keep_fp8_transpose_cache` (DistributedDataParallelConfig) | **True**（recipe `llama2_custom.py:592`）| False（mcore 默认）|

> 两侧 TE per-Linear 的 FP8 weight transpose 都**不缓存**（每 step 都要重做 cast+transpose）—— **per-Linear cache flag 不是差异点**。差异在 §2.2 的 TE op-fuser：Primus 的 op-fuser 把 cast+transpose 融进 GEMM，所以"重做"几乎免费（21 ms），NeMo 没融，所以"重做"= 1232 ms 独立 transpose kernel + 1199 ms HtoD memcpy。
>
> Mcore DDP 那个 `keep_fp8_transpose_cache=True` 是另一回事：它说的是 DDP RS bucket 内部要不要复用 FP8 transpose 表（影响 grad 通信路径），和 TE per-Linear 的 fwd/bwd 重算无关。Primus 开，NeMo 默认关。

---

## 2. 真正不同的部分（按"影响什么"分组）

### 2.1 DataLoader（**唯一一个对 step time 有大影响的差异**）

| 字段 | Primus baseline | NeMo |
| --- | --- | --- |
| `num_workers` | **0** (single-process) | **8** |
| `persistent_workers` | False (默认) | **True** |
| `prefetch_factor` | — | 2 (默认) |
| Mode | 主线程同步 collate + mask gen | Worker 进程异步 prefetch + page cache 热 |

**症状**：Primus 每个 step 主进程要做 (a) 读 packed 序列 → (b) `torch.tril(torch.ones(8192, 8192))` + `.lt(0.5)` + `stack` 生成 BSHD attention mask → (c) HtoD copy。这 ~191 ms 全在 GPU launcher 的同一个 Python 线程上，导致每个 step 开头 GPU 等 191 ms。详情见 `2026-04-29_idle_191ms_layer1_dataloader_root_cause.md`。

**为什么 Primus 现在没开**：之前试过，碰到 fork-after-CUDA 死锁（70B 大模型，CUDA context 在父进程已建好，fork 出来的 worker 进程一访问 CUDA 就挂）。需要：
- 用 `multiprocessing_context="spawn"`，或者
- 在 Bridge `dataset_provider` 里把 dataset 实例化到 worker_init_fn 里（CUDA 只在 worker 内初始化）

### 2.1.5 DDP overlap & 桶级优化（Primus 全开，NeMo 全关）

来源：`Primus-dev/primus/recipes/llama2_custom.py:485-594`（recipe 直接写在 `DistributedDataParallelConfig` 里，**不**受 yaml `comm_overlap_config: null` 影响 —— 后者只是 SymmetricMemory / TP-comm-overlap 的开关）。

| 字段 | Primus | NeMo | 解释 |
| --- | --- | --- | --- |
| `overlap_grad_reduce` | **True** | False | Primus 在 backward 期间持续 reduce-scatter grads（DistOpt path），最后只剩 tail-RS 暴露 |
| `overlap_param_gather` | **True** | False | Primus 在下个 fwd 开始时 all-gather 参数 shard |
| `overlap_param_gather_with_optimizer_step` | **True** | False | Primus 把下一 step 的 param AG overlap 到当前 optim step 上 |
| `average_in_collective` | **True** | False | Primus 把 `/world_size` 折进 RS kernel（少一次 div） |
| `gradient_reduce_div_fusion` | **True** | — | Primus-only：grad 累加 + RS div 融合 |
| `pad_buckets_for_high_nccl_busbw` | **True** | — | Primus-only：bucket 按 RCCL 友好对齐补齐，提升 busBW |
| `keep_fp8_transpose_cache` (DDP) | **True** | False (env `ENABLE_TRANSPOSE_CACHE=0` 影响 TE 而非 DDP) | DDP-level 的 FP8 transpose cache 跨 RS 复用 |
| `fp8_param_gather` | False | False | 两侧都关 |

> ⚠️ 这是之前 v1 note 写错的地方。Primus baseline yaml 的 `comm_overlap_config: null` **不是** "DDP overlap 关掉"，那只关掉了 SymmetricMemory/TP-overlap。Primus 的 DDP overlap 是 recipe 硬写的 True。

**为什么 trace 里 Primus 还是有 54 ms 的 RCCL 在 stream 33 的尾部？**

不是 overlap 失效。`overlap_grad_reduce=True` 只能 hide 掉 backward 进行中的 RS bucket，**最后一个 bucket 后面没 kernel 可 hide**，所以 final RS + 下一 step 的 first AG 必然会暴露。同时 LoRA 的 grad 量本来就小（只有 `linear_qkv` + `linear_proj` 的 r=16 LoRA adapter，每层 ~5 MB），但桶按 mcore 默认对齐到比这个大得多的 size，造成"等满桶才 launch"的尾巴。

**为什么 NeMo 关了 overlap 反而 RCCL 更小（5 ms）？**

因为 NeMo `LORA_A2A=1` 走的是 LoRA-only all-to-all 路径（`a2a_experimental=True`），只 reduce LoRA adapter 的 grad，根本不走 mcore DDP 的全权重 RS。本质上 NeMo 的"DDP overlap=False" 是无所谓的，因为它压根没用 DDP 的 bucket-AR 路径。

**结论**：DDP overlap 在 Primus 上**是有用的**（54 ms tail vs 估计 100+ ms 没 overlap 的全 AR），不要去关。

---

### 2.1.6 FP8 参数存储 & grad/AG 通信精度（runtime override，**直接解释 HBM 差距**）

来源：`run.log.429:1502-1517` 的 7 行 `Overwrote` 日志（recipe 默认值 → runtime 实际值）。

| 字段 | Primus runtime | NeMo | 对齐？ | 影响 |
| --- | --- | --- | --- | --- |
| `LlamaModelProvider.fp8_param` | **True** (← False) | False | ❌ | Primus 把 weight 真的存成 FP8 + bf16 master = 3 B/param；NeMo bf16 only = 2 B/param |
| `DistributedDataParallelConfig.fp8_param_gather` | **True** (← False) | False | ❌ | Primus DistOpt AG 用 FP8（½ BW）；NeMo bf16（但走 LoRA-A2A 绕开了 DDP AG） |
| `DistributedDataParallelConfig.grad_reduce_in_fp32` | **True** (← False) | False | ❌ | Primus RS grad 用 FP32（2× BW）；NeMo bf16 |
| `LlamaModelProvider.autocast_dtype` | **None** (← bf16) | bf16 (`autocast_enabled=True`) | ❌ | Primus 关 mcore autocast wrapper（TE 显式处理 FP8）；ε 数值影响 |
| `LlamaModelProvider.num_layers_at_start_in_bf16` | 0 (← 1) | 0 | ✅ | 80 层全 FP8 hybrid |
| `LlamaModelProvider.num_layers_at_end_in_bf16` | 0 (← 1) | 0 | ✅ | 同上 |
| `OptimizerConfig.fp8_recipe` | delayed (← None) | delayed | ✅ | 都是 DelayedScaling |

**HBM 账单**（70B param）：

```
Primus: FP8 weight (1 B) + bf16 master (2 B)             = 3 B/param × 70 B = 210 GB
NeMo:   bf16 weight only                                  = 2 B/param × 70 B = 140 GB
                                                                   ↑差 70 GB
+ 双方共享: FP32 main_grad (DistOpt sharded) + FP32 m/v (DistOpt sharded)
+ activation / KV cache / hipBLASLt workspace / Primus op-fuser tile staging
≈ 总和: Primus 285 GB, NeMo 200 GB (差 85 GB，70 GB 来自 fp8_param + 15 GB 来自 op-fuser/DDP buffer)
```

**RCCL 账单**（trace 实测）：

```
Primus AG params:   不在 critical path（DistOpt overlap + LoRA 只更新 adapter，全权重不需要 AG）
Primus RS grads:    54 ms tail (FP32 reduce, 2× BW vs bf16)
NeMo AG:            走 LoRA-A2A path, 绕开 DDP
NeMo RS:            走 LoRA-A2A path, 绕开 DDP
NeMo 唯一可见 RCCL:  5 ms (LoRA adapter grad a2a)
```

> **这是这次对比里最容易漏掉的真正差异点**。yaml 看不到，recipe 写的是默认值（False），但 runtime 被 Primus 内部的 precision_config 设置 logic 覆盖成 True。**只能从 run.log 的 `Overwrote` 行看出来**。

---

### 2.2 TE op-fuser（**kernel mix 的关键差异**，但 step time 影响小因为 overlap 掉了）

| 字段 | Primus | NeMo |
| --- | --- | --- |
| `enable_primus_turbo` | **True** | — |
| `use_transformer_engine_op_fuser` | **True** | False (NeMo 走 unfused TE) |
| `stable_lora_with_te_op_fuser` | **True** | — |

**Trace 上的体现**（一个 step）：

| 项 | Primus | NeMo | 解释 |
| --- | ---: | ---: | --- |
| 独立 `transpose_*` kernel | 21 ms | **1232 ms** (`transpose_optimized_kernel<bf16→fp8_e4m3>`) | NeMo 的 cast+transpose 是独立 kernel；Primus 把它**融进了 hipBLASLt FP8 GEMM** |
| HtoD memcpy 时长 | 4.5 ms (input prefetch) | **1199.8 ms** (FP8 transpose 流式拉) | NeMo 每 step 把 FP8 transpose 从 pinned host 拉到 device |
| FP8 GEMM 时长（4 个主 kernel 之和） | 773.5 ms | 806.3 ms | 几乎一样，差在 hipBLASLt autotuner |

> 对 wall-clock：因为 NeMo 把 1.2 GB HtoD + 1193 ms transpose 完全 overlap 在 compute stream 之后（实际上是同一个 stream 的 compute + memcpy hardware queue 并行），它的 step 仍然比 Primus 短。但是 HBM 带宽被吃掉一截（~270 GB/s 占用），并且 stream "occupancy" 从单看变成 257 % oversubscription。

### 2.3 Cross-entropy loss

| 字段 | Primus | NeMo |
| --- | --- | --- |
| `cross_entropy_fusion_impl` | `"te"` | `"native"` |
| `cross_entropy_loss_fusion` | True (mcore default) | **False (NeMo 显式关)** |

Primus 用 TE fused CE，NeMo 走 PyTorch native（unfused）。step time 上 ε 影响（CE 总共也就 2-3 ms），但内存占用 NeMo 略高（intermediate）。

### 2.4 mcore fusion knob

| 字段 | Primus | NeMo |
| --- | --- | --- |
| `gradient_accumulation_fusion` | True (mcore default) | **False (NeMo 显式关)** |

NeMo 显式关闭 wgrad 累加融合。Primus 走 mcore default，wgrad 直接累加进 main_grad bucket，省一遍 elementwise add。微影响。

### 2.5 FP8 amax buffer（DelayedScaling 的滑窗）

| 字段 | Primus (默认) | NeMo |
| --- | --- | --- |
| `fp8_amax_history_len` | 跟 TE/bridge default（128 / 1024）| **4** |
| `fp8_amax_compute_algo` | default (`max`) | **`most_recent`** |
| `tp_only_amax_red` | default | False（TP=1，no-op） |

NeMo 用更短的 amax 历史 + 取最新值的算法，省一点 HBM 和 DtoH 同步。本 run 看不到 step time 影响，但是 scale-out 时 amax-AR 上critical path 时会显出来。

### 2.6 RCCL & NCCL

| 字段 | Primus | NeMo |
| --- | --- | --- |
| RCCL on critical path (实测) | 54 ms (non-overlapped tail-AR) | 5 ms (LoRA-only) |
| `NCCL_MIN_CTAS` | — | 32 |
| `NCCL_MIN_P2P_NCHANNELS` | — | 32 |
| `NCCL_NCHANNELS_PER_NET_PEER` | — | 32 |
| `NCCL_NVLS_ENABLE` | — | 0 (no-op AMD) |
| `TORCH_NCCL_HIGH_PRIORITY` | **1** | — |

> **观测到的 49 ms RCCL 差异**主要不来自 NCCL flag tuning，来自 **LoRA-A2A path** + **更小的 amax 历史**导致 NeMo 的 reduce 数据量本来就小。把 NCCL channel/CTA 数对齐到 NeMo 是 free win，不会变慢；`TORCH_NCCL_HIGH_PRIORITY=1` 是 Primus 优势，**别去掉**。

### 2.7 HSA / HW queue（解释为什么 trace 里 stream 数不一样）

| 字段 | Primus | NeMo |
| --- | --- | --- |
| `GPU_MAX_HW_QUEUES` | **2** | default |
| `HSA_NO_SCRATCH_RECLAIM` | 1 | — |
| `HSA_ENABLE_SDMA` | 1 | — |
| `HSA_ENABLE_INTERRUPT` | 0 | — |

Primus 把 HW queue 卡到 2，所以 trace 看到 2 streams（compute + RCCL）；NeMo 默认走出 3 streams。`HSA_NO_SCRATCH_RECLAIM=1` 让 HSA scratch 不回收，对 memcpy 吞吐 ε 增益。

### 2.8 杂项（正交于性能）

| 字段 | Primus | NeMo |
| --- | --- | --- |
| `use_te_rng_tracker` | default (False) | **True (NeMo 显式开)** |
| `gradient_as_bucket_view` | True | True |
| `seed` | 1234 | 1 |
| Profiler schedule | `wait=0, warmup=2, active=5` | `wait=0, warmup=3, active=2` |

---

## 3. Step time 分解（一一对应到上面的差异）

```
Primus 1626 ms   = 191 ms idle (DataLoader) + 1380 ms compute + 54 ms RCCL tail
                                              + ε FP8 transpose (21 ms, fused)

NeMo  1490 ms   = 7 ms idle  + 1483 ms compute (compute + memcpy 完全 overlap)
                              + 5 ms RCCL (LoRA-only)
                  underlay: 1199 ms HtoD + 1193 ms transpose 在 stream 3 上和 compute 并跑
```

| 时间预算 | Primus | NeMo | Δ | 来源 |
| --- | ---: | ---: | ---: | --- |
| GPU idle | 191 ms | 7 ms | **−184 ms** | DataLoader 差异（§2.1） |
| FP8 GEMM (4 主 kernel) | 773.5 ms | 806.3 ms | +32.8 ms | autotuner / 输入差异 |
| Attention (CK V3 fwd+bwd) | 315.5 ms | 329.5 ms | +14.0 ms | 内核相同；step jitter |
| TE activation (gated/dgated/silu) | 64.1 ms | 49.5 ms | −14.6 ms | NeMo `bias_activation_fusion=True` 略有效 |
| Elementwise / cast / dropout | 122.8 ms | 104.0 ms | −18.8 ms | NeMo CE unfused 抵消 |
| RMSNorm (triton) | 20.3 ms | 24.0 ms | +3.7 ms | jitter |
| Fused QKV-RoPE | 23.3 ms | 14.9 ms | −8.4 ms | jitter |
| FP8 cast/transpose 独立 kernel | 21.2 ms | **1232.3 ms** | **+1211 ms** | TE op-fuser 差异（§2.2），但 overlap 掉 |
| Reduction (rms / amax / sum) | 8.6 ms | 8.9 ms | +0.3 ms | 一致 |
| Memcpy HtoD | 4.5 ms | **1199.8 ms** | **+1195 ms** | TE op-fuser 差异（§2.2），但 overlap 掉 |
| RCCL gradient sync | 54.4 ms | 5.2 ms | **−49.2 ms** | LoRA-A2A + amax 历史（§2.5/§2.6） |
| Optimizer / misc | 0.4 ms | 0.1 ms | −0.3 ms | jitter |

---

## 4. VRAM 对比

| 指标 | Primus baseline | NeMo |
| --- | ---: | ---: |
| Allocated peak (`Pmax`) | 285.84 GB | ~200 GB（估）|
| Reserved peak (`Rmax`, rocm-smi) | 295.52 GB | 211.5 GB |
| HBM headroom | < 3 GB（接近 OOM） | ~76 GB |

**为什么 NeMo 省 ~85 GB**（按贡献排序，更新版）：

1. **`fp8_param: True` (Primus) vs False (NeMo) → ~70 GB**：Primus 同时维护 FP8 weight + bf16 master = 3 B/param，NeMo 只有 bf16 weight = 2 B/param。70 B × 1 B = **70 GB**。这是大头。
2. **Primus op-fuser tile-staging buffer + DDP `keep_fp8_transpose_cache=True` 内部表 + `pad_buckets_for_high_nccl_busbw` 对齐 padding → ~10-15 GB**。
3. NeMo 不开 TE op-fuser，FP8 weight transpose 每 step 流式从 pinned host 拉 ~1.2 GB → device。换句话说 transpose 缓冲只用 host RAM，不占 HBM；Primus 的 op-fuser tile staging 在 HBM。

> 注意：把 `fp8_param=True` 关掉理论能省 70 GB，但会让 weight 在 fwd 之前再做一次 bf16 → FP8 cast（每 step 多 1+ GB transpose），可能反而 step time 变慢。**HBM vs step time 的本质 trade-off**。建议先做 activation / DistOpt audit（大概率能挤出 30-50 GB）再考虑动 fp8_param。

---

## 5. NeMo 端到端实测（Primus 还没跑到目标）

| 指标 | NeMo (本 run) |
| --- | ---: |
| Throughput | 5.40 samples / s |
| Eval interval | 每 48 step |
| 收敛步数 | step 384 / 3072 samples |
| 达标 eval_acc | **0.9244** ≥ target 0.925 |
| Wall-clock (run_stop) | **647.31 s = 10.79 min** |

收敛轨迹：

| step | samples | eval_acc | step_time |
| ---: | ---: | ---: | ---: |
| 192 | 1536 | 0.9415 | 1.5237 s |
| 240 | 1920 | 0.9361 | 1.5020 s |
| 288 | 2304 | 0.9302 | 1.5025 s |
| 336 | 2688 | 0.9315 | 1.5025 s |
| **384** | **3072** | **0.9244** ✅ | 1.5024 s |

> Primus 还没跑到 target accuracy；当前 baseline 跑的是 trace + 短 iter（80–85），所以下一步 Primus 也要跑端到端到 target，才能比 wall-clock。

---

## 6. Primus 后续 action items（按 ROI 排序）

### 6.1 高价值

**A1. DataLoader prefetch（−180 ms / step，期望 step → ~1440 ms）**

```yaml
# llama2_70b_lora_posttrain.yaml overrides
dataset:
  num_workers: 2          # NeMo 用 8，先稳一点
  persistent_workers: true
  prefetch_factor: 2
  multiprocessing_context: spawn   # 关键，避免 fork-after-CUDA
```

或者在 `Primus-dev/third_party/Megatron-Bridge/src/megatron/bridge/recipes/primus/recipes/llama2_custom.py` 里通过环境变量注入（之前的 path：`PRIMUS_DL_NUM_WORKERS`、`PRIMUS_DL_PERSISTENT_WORKERS`、`PRIMUS_DL_PREFETCH_FACTOR`、`PRIMUS_DL_MP_CONTEXT`）。

### 6.2 中等价值（HBM）

**A2. DistOpt + activation memory 审计**（−40 ~ −80 GB Pmax，不动 step time）

- `activation_memory_audit.py` 跑一遍 L80 transformer 看是否有 unfreed checkpoint
- 检查 `main_grad` bucket 对齐是否过大（`bucket_size` 可设小）
- 看 LoRA 的 grad accumulator 是否还在 fp32（可以转 bf16）

**A3. 跑端到端到 target accuracy** — 在 A1 修完后必须做，否则没法和 NeMo 的 10.79 min 直接比 wall-clock。

### 6.3 低价值（对齐用）

**A4. 把 NeMo 的 NCCL channel/CTA 调参 copy 过来**（free win，scale-out 有用）

```bash
export NCCL_MIN_CTAS=32
export NCCL_MIN_P2P_NCHANNELS=32
export NCCL_NCHANNELS_PER_NET_PEER=32
# 保留 TORCH_NCCL_HIGH_PRIORITY=1（这是 Primus 优势）
```

**A5. FP8 amax history → 4, algo → most_recent**（HBM 微省）

```yaml
precision_config:
  fp8_amax_history_len: 4
  fp8_amax_compute_algo: most_recent
```

### 6.4 不建议做

- ❌ **关 TE op-fuser** —— 看似对齐 NeMo，但 21 ms vs 1232 ms 的 transpose 差距说明 op-fuser 是严格更优的 kernel mix；step time 会退化几十 ms。
- ❌ **关 DDP overlap (`overlap_grad_reduce` / `overlap_param_gather`)** —— Primus 现在已经开着，且 trace 上的 54 ms tail-RS 是必然暴露的最后一桶（不是 overlap 失效）。关掉 overlap 反而会让 RCCL 直接膨胀到 100+ ms。NeMo 的 5 ms 不是因为关了 overlap，而是因为 LoRA-A2A 路径完全绕开了 DDP bucket-RS。
- ❌ **`fp8_dot_product_attention=True`** —— 两侧都关；CK V3 attention bf16 已经够快（315 ms），FP8 attention 在 8K seq + GQA-8 上没有加速空间。

---

## 7. 一句话总结

> Primus 和 NeMo 跑同一个 model + numerics + parallelism + LoRA + optimizer + RoPE + RMSNorm + CK V3 attention，**唯一让 NeMo 整体快 8 % 的差异是 DataLoader（0 vs 8 worker）**，导致 Primus 每 step 多 191 ms 同步等待。NeMo 的 1.2 GB / step HtoD + 1193 ms 独立 FP8 transpose 看起来吓人，但它是 Primus `enable_primus_turbo + use_transformer_engine_op_fuser` 把 cast-transpose 融进 GEMM 的代价 —— Primus 这边只花 21 ms。DDP 这边 Primus 全开 overlap + bucket 优化（`overlap_grad_reduce/param_gather=True`、`average_in_collective`、`gradient_reduce_div_fusion`、`pad_buckets_for_high_nccl_busbw`），NeMo 全关 —— 但 NeMo 走 LoRA-A2A path 绕开了 DDP RS，所以反而 RCCL 更小。Primus 后续只要修好 DataLoader prefetch，就应该能在 step time 上反超；HBM 285 GB 的问题需要单独做 DistOpt + activation memory 审计，**不要去动 op-fuser，也不要去关 DDP overlap**。

---

## 8. 口述总结（开会用，按顺序念就行）

> 适合 5-7 分钟汇报。每段 1 个要点，可以直接念，括号里是数据点 / 必要时口述出来。

### 8.1 开场 — 这次比的是什么 (~30 秒)

> "我们对比的是 Llama-2-70B LoRA SFT 在 8 张 MI355X 上的两个 baseline：Primus（基于 Megatron-Bridge）和 NeMo（用 Megatron-LM 后端）。两边模型一模一样、并行配置一模一样（TP/PP/CP 都是 1，纯 DP=8，packed sequence 8K），LoRA 配置一模一样（rank 16、alpha 32），优化器和 LR schedule 也一模一样（AdamW、4e-4、cosine、grad clip 0.3）。FP8 hybrid 都是 DelayedScaling，attention 都跑 CK V3 内核。所以这是一个**控制变量很干净的同基础对比**。"

### 8.2 主结果 — 8 % 的差距来自哪里 (~1 分钟)

> "NeMo 一个 step 是 1490 毫秒，Primus 是 1626 毫秒，NeMo 快 **8.4 %**。但是这 136 ms 里面，**184 ms 是 Primus 的 DataLoader 在等**。"
>
> "为什么？Primus baseline 的 `num_workers=0`，所有数据加载和 attention mask 生成都跑在主进程，每个 step 一开始 GPU 要等大概 191 毫秒；NeMo 用了 8 个 persistent worker 在异步 prefetch，所以 GPU 几乎不等。"
>
> "把 DataLoader 这个洞填上之后，Primus 的 compute 时间其实和 NeMo 是**同一个量级**——FP8 GEMM 主体 770~810 ms，attention 315~330 ms，RMSNorm/激活/RoPE 都在几十 ms。所以**这次对比真正的 takeaway 是：让 Primus 把 DataLoader 跑起来，理论上能反超 NeMo**。"

### 8.3 Trace 上看起来吓人但其实不重要的差异 — TE op-fuser (~1 分钟)

> "Trace 一打开会看到 NeMo 居然有 **1199 ms 的 HtoD memcpy** 和 **1193 ms 的 transpose 内核**，第一反应肯定是 NeMo 是不是哪里坏了。但其实正好相反——这是 NeMo **没开 TE op-fuser** 的代价：FP8 weight 每 step 都要从 pinned host 流式拉到 GPU 再 cast 一次。"
>
> "Primus 这边开了 `enable_primus_turbo + use_transformer_engine_op_fuser`，把这个 cast+transpose 直接**融进了 hipBLASLt FP8 GEMM**，所以独立 transpose kernel 只占 21 ms，HtoD 只占 4.5 ms。"
>
> "为什么这一坨在 NeMo 上没拖慢 step time？因为 NeMo 完全 overlap 在 compute stream 之后了——你看 NeMo 的 stream 3 实际 busy 时间是 **3830 ms 但 step 只有 1490 ms**，是 257 % 的 oversubscription，HBM 带宽和 DMA 引擎在并行干活。"
>
> "**结论：千万别为了对齐 NeMo 关掉 op-fuser**。"

### 8.4 这次发现的真正配置差异（容易漏） (~1.5 分钟)

> "这是这次最值得记住的部分。我们最早只看 yaml 和 recipe 的默认值，结论是两边 DDP 配置很相似。但今天读了 run.log 里的 `Overwrote` 行才发现 Primus runtime **悄悄改了** 7 个东西，有 4 个其实和 NeMo 不一致："
>
> 1. **`fp8_param: True`** —— Primus 真的把 weight 存成 FP8，再单独维护一份 bf16 master，每个 param 占 3 字节；NeMo 只存 bf16，每个 param 2 字节。**这就是 Primus 比 NeMo 多用 70 GB HBM 的主要原因**（70B × 1 B 多余 master）。
> 2. **`grad_reduce_in_fp32: True`** —— Primus reduce-scatter grad 用 FP32，比 NeMo 的 bf16 多 2 倍带宽。
> 3. **`fp8_param_gather: True`** —— Primus DistOpt all-gather 用 FP8，比 NeMo 的 bf16 省一半带宽。
> 4. **DDP overlap 全开** —— `overlap_grad_reduce`、`overlap_param_gather`、`overlap_param_gather_with_optimizer_step`、`average_in_collective`、`gradient_reduce_div_fusion`、`pad_buckets_for_high_nccl_busbw` 这 6 个 Primus 全是 True，NeMo 全是 False。
>
> "**所以 Primus 不是配置粗糙，而是被 Primus 的 precision_config = `bf16_with_fp8_hybrid` 这条路径帮忙调好了，但是这些设置都是在 runtime 才被覆盖的，看 yaml 完全看不出来**。"

### 8.5 RCCL 看起来差很多但其实是误读 (~30 秒)

> "Trace 里 Primus RCCL 占 54 ms，NeMo 只占 5 ms，乍看 NeMo 通信调得好。但其实**两边走的根本不是同一条路径**："
>
> - "Primus 走的是 mcore DDP 的标准 reduce-scatter，每 step 必然有一个 tail-RS 暴露在最后一个 bucket 后面（因为 overlap 只能 hide 中间的）。"
> - "NeMo 关了 DDP overlap，但是它走 `LORA_A2A=1` 的 a2a 路径——只 reduce LoRA adapter（44M 参数）的 grad，根本没用 DDP bucket 路径。"
>
> "所以 NeMo 的 5 ms 不是来自 NCCL 调参，是**因为通信量本来就只有 LoRA 那么大**。"

### 8.6 HBM 285 GB 怎么办 (~45 秒)

> "Primus 现在 HBM 用到 285.84 GB / 295.52 GB reserved，离 OOM 不到 3 GB，确实危险。但**不能简单地关 op-fuser 或者 fp8_param 来省**："
>
> - "关 op-fuser → step time 退化几十 ms"
> - "关 `fp8_param=True` → 省 70 GB，但每 step 要多做一次 bf16 → FP8 cast，可能拖慢"
>
> "正确的方向是先做 **activation memory audit**（看 80 层 transformer 哪里有没释放的 checkpoint）+ **DistOpt bucket 调小**，预计能挤出 30-50 GB headroom。如果还不够再考虑 `fp8_param` trade-off。"

### 8.7 NeMo 的端到端成绩 (~30 秒)

> "顺便提一下 NeMo 的端到端成绩：384 step / 3072 samples 跑到 eval_acc **0.9244**（target 0.925），wall-clock **10.79 分钟**，throughput 5.40 samples/s。Primus 这边还没跑端到端到 target，所以 wall-clock 比不了。**A1 修完后必须跑一次端到端**才能正式对比。"

### 8.8 收尾 — Action items (~30 秒)

> "总结成 3 件该做的、3 件不该做的："
>
> **该做**:
> 1. **A1**：DataLoader prefetch（spawn context 解决 fork-after-CUDA），预期 step → 1440 ms 反超 NeMo
> 2. **A2**：activation + DistOpt 内存审计，预期 −30~50 GB
> 3. **A3**：跑端到端到 target，正式比 wall-clock
>
> **不该做**:
> 1. ❌ 关 TE op-fuser（21 ms vs 1232 ms 的 transpose 差距证明它严格更优）
> 2. ❌ 关 DDP overlap（54 ms tail-RS 不是 overlap 失效，关了反而更糟）
> 3. ❌ 开 FP8 DPA（CK V3 bf16 attention 已经够快）
>
> "完。"

---

> **使用提示**：每段独立成块，按 8.1 → 8.8 念大概 5-6 分钟。如果时间紧，可以跳过 8.3（op-fuser）和 8.5（RCCL 误读），保留 8.1 → 8.2 → 8.4 → 8.6 → 8.7 → 8.8 也是完整故事。

---

## 9. English meeting script — short version (~2 min)

> Just the core points. Plain English, short sentences. Read straight through.

### Setup (~15 sec)

> "Today I compared **Primus** and **NeMo** on **Llama-2-70B LoRA SFT**, **8 MI355X**, same model, same parallelism, same optimizer, same FP8 recipe. So any gap is system, not math."

### Main result (~30 sec)

> "**NeMo step is 1490 ms, Primus step is 1626 ms — NeMo wins by 8.4 %.**"
>
> "But out of that 136 ms gap, **184 ms is just Primus's DataLoader**. Primus runs `num_workers = 0`, so the GPU waits ~190 ms every step for the CPU to build the attention mask. NeMo runs 8 persistent prefetching workers, so it waits ~zero."
>
> "**Fix the DataLoader on Primus and Primus should beat NeMo.** The compute itself is already on par."

### Two things in the trace that LOOK wrong but are NOT (~30 sec)

> "**One — NeMo has 1.2 GB host-to-device memcpy and a 1193 ms transpose kernel per step.** That is just NeMo not enabling the TE op-fuser. Primus fuses cast+transpose into the FP8 GEMM, so it only spends 21 ms there. NeMo overlaps the big transpose behind compute, so it doesn't hurt wall-clock. **Don't turn the op-fuser off on Primus.**"
>
> "**Two — Primus has 54 ms of RCCL, NeMo has 5 ms.** That's not NCCL tuning. NeMo uses LoRA all-to-all, which bypasses DDP entirely and only reduces the 44 M LoRA adapter params. **Don't turn DDP overlap off on Primus.**"

### Today's biggest finding — runtime overrides hide real config (~30 sec)

> "Reading the YAML and recipe, the two stacks look almost identical. **But the Primus run log has 7 `Overwrote` lines that show the runtime values are different.** The big one: **`fp8_param = True` on Primus**, but `False` on NeMo. Primus stores weights as FP8 plus a bf16 master copy — that's 3 bytes per param vs NeMo's 2 bytes. On 70 B params **that's the 70 GB HBM gap** between Primus's 285 GB and NeMo's 200 GB."
>
> "**Lesson: don't trust YAML for FP8 / DDP analysis. Always read the `Overwrote` lines in the run log.** We rewrote our analysis twice today because of this."

### Action items (~20 sec)

> "Three things to do:"
>
> "**A1**: turn on DataLoader workers on Primus with spawn context — expected to drop step to ~1440 ms, beating NeMo."
>
> "**A2**: activation and DistOpt memory audit — expected to free 30-50 GB of HBM headroom."
>
> "**A3**: run Primus end-to-end to target accuracy so we can compare against NeMo's 10.79 min."
>
> "Three things **not** to do: don't disable the op-fuser, don't disable DDP overlap, don't enable FP8 attention. All three make things worse."
>
> "That's it. Questions?"
