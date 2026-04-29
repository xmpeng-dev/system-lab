# NeMo Llama-2-70B LoRA SFT — 详细配置档

> Run: `config_MI355X_1x8x1` @ 2026-04-29 05:47:19 — 容器 `xm-nemo` (`rocm/amd-mlperf:llama2_70b_sft_nemo_6.0`)
> Trace dir: `llama2_sft/nemo/run_traces/config_MI355X_1x8x1_20260429_054719/`
>
> 本档把 NeMo 这次 run 实际传给 megatron-core 的所有参数（`Llama2Config70B` → `TransformerConfig`、`MegatronStrategy`、`DistributedDataParallelConfig`、`OptimizerConfig`、`MegatronMixedPrecision`、LoRA、env、profiler）整理成一份完整的配置档，对标 [Primus baseline note](./2026-04-29_llama2_70b_lora_baseline_trace_breakdown.md) 的颗粒度方便 diff。

## 0. 概览

| 字段 | 值 |
| --- | --- |
| 日期 | 2026-04-29 |
| 模型 | Llama-2-70B · LoRA SFT · `bf16-mixed + fp8 hybrid (DelayedScaling)` |
| 硬件 | 8 × MI355X (288 GiB HBM, gfx950, MI355X_1x8x1) |
| 并行 | TP1 · PP1 · CP1 · EP1 · DP=8 · SP=False |
| Batch | GBS **8** · MBS 1 · seq 8192 (packed) · grad-accum 1 |
| 训练规模 | 总参 ≈ 69.0 B · 可训参 ≈ 44.5 M (LoRA r=16, ≈ 0.06%) |
| 实测 | step time **1.502 s** · throughput **5.40 samples/s** |
| 收敛 | 384 step / 3072 samples (`eval_acc=0.9244` ≥ target 0.925) |
| Wall-clock | `INTERVAL_END run_stop` = 647.31 s = **10.79 min** |
| 入口 | `bash run.sh`（`config_MI355X_1x8x1.sh` + `TORCH_PROFILE=1`）|
| Hydra 模板 | `llama2_sft/nemo/src/conf/megatron_gpt_peft_tuning_config.yaml` |
| 训练入口 py | `llama2_sft/nemo/src/train.py` |

---

## 1. Mcore `TransformerConfig`（从 `Llama2Config70B` 实例化）

NeMo 调用：`train.py:351-374`，类源自 `nemo.collections.llm.Llama2Config70B`，最终落到 mcore `TransformerConfig`。

### 1.1 模型架构
| 字段 | 值 | 说明 |
| --- | --- | --- |
| `num_layers` | **80** | env `OVERWRITTEN_NUM_LAYERS` 默认 |
| `hidden_size` | **8192** | |
| `num_attention_heads` | **64** | |
| `num_query_groups` | **8** | GQA 8 |
| `kv_channels` | **128** | head_dim |
| `ffn_hidden_size` | **28672** | SwiGLU |
| `seq_length` / `max_position_embeddings` / `encoder_seq_length` | **8192** | |
| `vocab_size` (hf) | 32000 | tokenizer 决定 |
| `share_embeddings_and_output_weights` | False | |

### 1.2 Norm / Activation / RoPE
| 字段 | 值 |
| --- | --- |
| `normalization` | `RMSNorm` |
| `layernorm_epsilon` | 1e-5 |
| `layernorm_zero_centered_gamma` | False |
| `persist_layer_norm` | True |
| `memory_efficient_layer_norm` | False |
| `activation_func` | `silu`（`gated_linear_unit=True` 即 SwiGLU）|
| `position_embedding_type` | `rope` |
| `rotary_base` | 10000 |
| `rotary_percent` | 1.0 |
| `rotary_interleaved` | False |
| `apply_rope_fusion` | **True** |
| `fused_single_qkv_rope` | False（env `FUSED_SINGLE_QKV_ROPE=` 默认）|
| `apply_query_key_layer_scaling` | False |
| `attention_softmax_in_fp32` | False |

### 1.3 Bias / Dropout / Fusion
| 字段 | 值 |
| --- | --- |
| `add_bias_linear` | False |
| `add_qkv_bias` | False |
| `attention_dropout` | 0.0 |
| `hidden_dropout` | 0.0 |
| `bias_activation_fusion` | True |
| `bias_dropout_fusion` | **True**（NeMo 显式开）|
| `masked_softmax_fusion` | True |
| `cross_entropy_loss_fusion` | **False**（NeMo 显式关）|
| `cross_entropy_fusion_impl` | `'native'` |
| `gradient_accumulation_fusion` | **False**（NeMo 显式关）|
| `use_fused_weighted_squared_relu` | False |

### 1.4 Attention
| 字段 | 值 | 说明 |
| --- | --- | --- |
| `attention_backend` | `AttnBackend.auto` | 由 `NVTE_FUSED_ATTN=1 NVTE_FUSED_ATTN_CK=1 NVTE_FUSED_ATTN_AOTRITON=1` 走 TE FA via aiter CK v3 |
| `softmax_type` | `'vanilla'` | |
| `softmax_scale` | None（= 1/√d_k）| |
| `qk_layernorm` | False | |
| `window_size` | None | full attention |
| `window_attn_skip_freq` | None | |

### 1.5 Recompute / Activation Checkpoint（**全关**）
| 字段 | 值 |
| --- | --- |
| `recompute_granularity` | **None** |
| `recompute_method` | None |
| `recompute_num_layers` | None |
| `distribute_saved_activations` | None |
| `recompute_modules` | `['core_attn']`（类默认值，但 `granularity=None` 不生效）|
| `dropout_recompute` (NeMo 层) | False |

### 1.6 CPU Offloading（**全关**）
| 字段 | 值 |
| --- | --- |
| `cpu_offloading` | **False** |
| `cpu_offloading_num_layers` | 20（不生效）|
| `cpu_offloading_activations` | True |
| `cpu_offloading_weights` | False |
| `cpu_offloading_double_buffering` | False |

### 1.7 CUDA Graph（**全关**）
| 字段 | 值 |
| --- | --- |
| `enable_cuda_graph` (mcore) | **False**（`MCORE_CUDA_GRAPH` 未设）|
| `external_cuda_graph` (per-layer) | **False**（`LAYER_CUDA_GRAPH=False`）|
| `cuda_graph_impl` | `'none'` |
| `cuda_graph_scope` | `'full'` |
| `cuda_graph_warmup_steps` | 3（未生效）|

### 1.8 TP-Comm-Overlap（**未启用，仅记录默认**）
> `cfg.model.ub_tp_comm_overlap=False`，下面的字段都不生效。

| 字段 | 值 |
| --- | --- |
| `tp_comm_overlap` | False |
| `tp_comm_overlap_disable_qkv` | **True**（NeMo 显式）|
| `tp_comm_overlap_disable_fc1` | False |
| `tp_comm_bulk_wgrad` / `tp_comm_bulk_dgrad` | True / True |
| `tp_comm_overlap_ag` / `tp_comm_overlap_rs` / `tp_comm_overlap_rs_dgrad` | True / True / False |
| `tp_comm_split_ag` / `tp_comm_atomic_ag` | True / False |
| `tp_comm_split_rs` / `tp_comm_atomic_rs` | True / False |
| `tp_comm_bootstrap_backend` | `'nccl'` |

### 1.9 FP8 / FP4 / 精度位
| 字段 | 值 | 说明 |
| --- | --- | --- |
| `fp8` (TransformerConfig 字段) | None | 实际由 `MegatronMixedPrecision.fp8="hybrid"` 驱动（见 §4）|
| `fp8_recipe` | `'delayed'` | TE `DelayedScaling` |
| `fp8_param` | False | |
| `fp8_margin` | 0 | |
| `fp8_interval` | 1 | |
| `fp8_amax_history_len` | **4** | env `FP8_AMAX_HISTORY=4`（覆盖 cfg 默认 128）|
| `fp8_amax_compute_algo` | `'most_recent'` | |
| `fp8_wgrad` | True | |
| `fp8_dot_product_attention` | **False** | FP8 DPA off |
| `fp8_multi_head_attention` | False | |
| `tp_only_amax_red` | False | |
| `first_last_layers_bf16` | False | env `FIRST_LAST_LAYERS_BF16=False` |
| `num_layers_at_start_in_bf16` | 0 | env override |
| `num_layers_at_end_in_bf16` | 0 | env override |
| `disable_bf16_reduced_precision_matmul` | False | |
| `keep_fp8_weight_transpose_cache` | **False** | ⚠️ **关键点**：NeMo 默认不缓存 FP8 weight transpose，每 step 重算 → 解释了 trace 里 1195 ms HtoD memcpy + 1193 ms transpose 的来源（见 trace breakdown note）|
| `disable_parameter_transpose_cache` | False | （这是 bf16 param transpose，与上面那个不同）|
| `fp4` | None | env `FP4=False` |
| `fp4_recipe` | `'nvfp4'` | （未启用）|
| `fp4_param` | False | |
| `use_kitchen` | False | |

### 1.10 CP / MoE / MTP / Mamba（**与本 run 无关，全默认**）
| 字段 | 值 |
| --- | --- |
| `cp_comm_type` | `'a2a'`（CP=1，未生效）|
| `hierarchical_context_parallel_sizes` | None |
| `expert_model_parallel_size` | 1 |
| `num_moe_experts` | None |
| `moe_*` | 全默认（unused）|
| `multi_latent_attention` | False |
| `is_hybrid_model` | False（mamba）|
| `mtp_num_layers` | None |
| `flash_decode` | False |
| `use_fsdp2` | False |
| `use_te_rng_tracker` (TransformerConfig) | False（类默认；NeMo 在 `MegatronStrategy` 里覆盖为 True，见 §2）|

---

## 2. NeMo `MegatronStrategy`（mcore 并行 + DDP wrapper）

来源：`train.py:414-440`

```python
strategy = nl.MegatronStrategy(
    tensor_model_parallel_size=cfg.model.tensor_model_parallel_size,
    pipeline_model_parallel_size=cfg.model.pipeline_model_parallel_size,
    context_parallel_size=cfg.model.context_parallel_size,
    sequence_parallel=cfg.model.sequence_parallel,
    pipeline_dtype=torch.bfloat16,
    cluster_environment=cluster_env,
    ckpt_load_directly_on_device=False,
    ckpt_parallel_load=False,
    ckpt_load_optimizer=False,
    ckpt_load_main_params=False,
    ckpt_load_strictness="log_all",
    gradient_as_bucket_view=True,
    use_te_rng_tracker=cfg.model.use_te_rng_tracker,
    fsdp=cfg.model.fsdp,
    ddp=DistributedDataParallelConfig(...),
)
```

| 字段 | 值 |
| --- | --- |
| `tensor_model_parallel_size` | **1** |
| `pipeline_model_parallel_size` | **1** |
| `context_parallel_size` | **1** |
| `sequence_parallel` | **False** |
| `pipeline_dtype` | bfloat16 |
| `gradient_as_bucket_view` | True |
| `use_te_rng_tracker` | **True**（NeMo 显式开）|
| `fsdp` | None |
| `ckpt_load_directly_on_device` | False |
| `ckpt_parallel_load` | False |
| `ckpt_load_optimizer` | False |
| `ckpt_load_main_params` | False |
| `ckpt_load_strictness` | `"log_all"` |
| 派生 DP | 8 |

> 没装 `MegatronCommOverlapCallback`（`ub_tp_comm_overlap=False` → `train.py:470` 走 else）。

---

## 3. Mcore `DistributedDataParallelConfig`（DDP / DistOpt）

| 字段 | 值 | 说明 |
| --- | --- | --- |
| `use_distributed_optimizer` | **True** | mcore DistributedOptimizer (ZeRO-1) |
| `overlap_grad_reduce` | **False** | env `DDP_OVERLAP_GRAD_REDUCE=` 默认 |
| `overlap_param_gather` | **False** | env `DDP_OVERLAP_PARAM_GATHER=` 默认 |
| `fp8_param_gather` | **False** | env `FP8_PARAM_GATHER=False` |
| `average_in_collective` | **False** | env `DDP_AVERAGE_IN_COLLECTIVE=` 默认 |
| `use_custom_fsdp` | False | (`fsdp != "megatron"`) |
| `data_parallel_sharding_strategy` | `"no_shard"` | |
| `nccl_ub` | **False** | NCCL user-buffer reg 关 |
| `fsdp_double_buffer` | False | |

---

## 4. NeMo `MegatronMixedPrecision`（驱动 TE/FP8）

来源：`train.py:449-466`

| 字段 | 值 |
| --- | --- |
| `precision` | **`"bf16-mixed"`** |
| `params_dtype` | bfloat16 |
| `pipeline_dtype` | bfloat16 |
| `autocast_enabled` | True |
| `grad_reduce_in_fp32` | **False** |
| `first_last_layers_bf16` | False |
| `num_layers_at_start_in_bf16` | 0 |
| `num_layers_at_end_in_bf16` | 0 |
| `fp8` | **`"hybrid"`**（E4M3 fwd / E5M2 bwd）|
| `fp8_recipe` | **`"delayed"`** |
| `fp8_amax_history_len` | **4** |
| `fp8_amax_compute_algo` | `"most_recent"` |
| `fp8_param_gather` | False |
| `fp8_dot_product_attention` | False |
| `fp4` | None |
| `fp4_recipe` | `"nvfp4"`（unused）|

FP8 模型实例化在 `custom_llama.py:96-99`：

```python
recipe = te_recipe.DelayedScaling()
with te.fp8_model_init(recipe=recipe):
    super().configure_model()
```

---

## 5. Mcore `OptimizerConfig` + LR Schedule

来源：`train.py:308-326`

```python
optimizer_config = OptimizerConfig(
    optimizer="adam",
    lr=lr,
    clip_grad=0.3,
    weight_decay=0.0001,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_eps=1e-08,
    bf16=True,
    params_dtype=torch.bfloat16,
    use_distributed_optimizer=use_distributed_optimizer,
    overlap_param_gather_with_optimizer_step=overlap_param_gather_with_optimizer_step,
)
scheduler = CosineAnnealingScheduler(
    max_steps=max_steps,
    warmup_steps=warmup_steps,
    constant_steps=0,
    min_lr=0,
)
```

| 字段 | 值 |
| --- | --- |
| `optimizer` | adam → AdamW（`weight_decay>0`）|
| `lr` | **4e-4** |
| `clip_grad` | **0.3** |
| `weight_decay` | **1e-4** |
| `adam_beta1` / `adam_beta2` | 0.9 / 0.999 |
| `adam_eps` | 1e-8 |
| `bf16` | True |
| `params_dtype` | bfloat16 |
| `use_distributed_optimizer` | **True** |
| `overlap_param_gather_with_optimizer_step` | **False** |
| Scheduler | `CosineAnnealingScheduler(max_steps=1024, warmup_steps=0, constant_steps=0, min_lr=0)` |

MLLOG 输出确认：

| key | value |
| --- | --- |
| `opt_base_learning_rate` | 0.0004 |
| `opt_adamw_weight_decay` | 0.0001 |
| `opt_gradient_clip_norm` | 0.3 |
| `opt_learning_rate_warmup_factor` | 0.0 |
| `opt_learning_rate_training_steps` | 1024 |

---

## 6. PEFT — LoRA

来源：`train.py:340-349`

| 字段 | 值 |
| --- | --- |
| `dim` (rank) | **16** |
| `alpha` | **32** |
| `dropout` | 0.1 |
| `dropout_position` | `"pre"` |
| `lora_A_init_method` | `"kaiming"` |
| `target_modules` | `["linear_proj", "linear_qkv"]` |
| `a2a_experimental` | **True**（env `LORA_A2A=1` 路径）|
| `dropout_recompute` | False |

MLLOG 确认：`lora_rank=16`、`lora_alpha=32`。

---

## 7. 数据栈

| 字段 | 值 |
| --- | --- |
| Module | `nemo.collections.llm.FineTuningDataModule` |
| `dataset_root` | `/data/mlperf_llama2` |
| `seq_length` | 8192 |
| `micro_batch_size` | 1 |
| `global_batch_size` | 8 |
| `persistent_workers` | True |
| `num_workers` | 8（env `NUM_WORKERS=8` 默认）|
| `seed` | 1 |
| Packed sequence | enabled |
| `dataset_kwargs.return_cu_seqlen` | False |
| Tokenizer | `meta-llama/Llama-2-70b-hf` |
| `train_samples` | 3901（MLLOG）|
| `eval_samples` | 173（MLLOG）|
| `gradient_accumulation_steps` | 1 |
| `val_check_interval` | 384/8 = 48 step（每 48 step eval 一次，跳前 3 次）|

---

## 8. 训练 Schedule / Runtime

| 字段 | 值 |
| --- | --- |
| `MAX_STEPS` | 1024 |
| `WARMUP`（synthetic warmup）| True |
| `WARMUP_TRAIN_STEPS` / `WARMUP_VALIDATION_STEPS` | 5 / 5（默认）|
| `RESET_FP8_STATS_AFTER_WARMUP` | 1 |
| `LIMIT_VAL_BATCHES` | 1.0 |
| `SKIP_EVALS` | 3 |
| `LOAD_CKPT` | True (`/data/model`)|
| Resume | `AutoResume(restore_config={path=/data/model, load_model_state=True, load_optim_state=False, load_artifacts=False})` |
| `WALLTIME_MINUTES` | 50 |
| `LOGGING_INTERVAL` | 5000 |
| `MLPERF_VERBOSE_LOGS` | 0 |
| `target_accuracy` | 0.925 |

实际收敛轨迹（MLLOG）：

| step | samples | eval_acc | train_step_time |
| ---: | ---: | ---: | ---: |
| 192 | 1536 | 0.9415 | 1.5237 s |
| 240 | 1920 | 0.9361 | 1.5020 s |
| 288 | 2304 | 0.9302 | 1.5025 s |
| 336 | 2688 | 0.9315 | 1.5025 s |
| 384 | 3072 | **0.9244** ✅ | 1.5024 s |

`run_stop` @ step 384 / 3072 samples，duration **647.31 s = 10.79 min**。

---

## 9. Profiler / Tracing

| 字段 | 值 |
| --- | --- |
| `TORCH_PROFILE` | 1 |
| 输出目录 | `run_traces/config_MI355X_1x8x1_20260429_054719/torchprof/` |
| `PROF_WARMUP_STEPS` | 3 |
| `PROF_ACTIVE_STEPS` | 2 |
| `PROF_REPITIONS` | 1 |
| Schedule | `skip_first=1, wait=0, warmup=3, active=2, repeat=1` |
| `PROFILE_RPD` | 0 |
| `ENABLE_MEMORY_PROFILING` | 0 |
| 选用的 step | `ProfilerStep#5`（1490.4 ms）|
| Trace 文件 | `torchprof/trace_6_067c4d34-e473-4a5e-a69f-a6d97278ab23.json` |
| key_avg 文件 | `torchprof/key_avg_6_*.txt`（3 份 rank）|

接入点：`src/callbacks/custom_callbacks.py` 的 `on_train_start / on_train_batch_end / on_train_end`，调 `src/prof_handler.py:get_profiler()`。

---

## 10. 影响 Mcore / NCCL / TE 的关键 ENV

### 10.1 NCCL / 通信
| ENV | 值 |
| --- | --- |
| `CUDA_DEVICE_MAX_CONNECTIONS` | **1**（mcore 必需）|
| `NCCL_NVLS_ENABLE` | 0 |
| `NCCL_MIN_P2P_NCHANNELS` | 32 |
| `NCCL_MIN_CTAS` | 32 |
| `NCCL_NCHANNELS_PER_NET_PEER` | 32 |
| `MC_TP_OVERLAP_AG` / `_RS` / `_RS_DGRAD` | False / False / False |

### 10.2 TransformerEngine FP8 / Attention
| ENV | 值 | 说明 |
| --- | --- | --- |
| `NVTE_FUSED_ATTN` | 1 | 走 TE FA |
| `NVTE_FUSED_ATTN_CK` | 1 | AMD CK kernel |
| `NVTE_FUSED_ATTN_AOTRITON` | 1 | aotriton fallback |
| `NVTE_FP8_DPA_BWD` | 1 | （但 `fp8_dot_product_attention=0` 没启）|
| `NVTE_RS_STRIDED_ATOMIC` | 2 | reduce-scatter 路径 |
| `NVTE_USE_HIPBLASLT` | 1 | hipBLASLt FP8 GEMM |
| `NVTE_USE_CAST_TRANSPOSE_TRITON` | 1 | triton cast+transpose |
| `NVTE_USE_RMSNORM_TRITON` | 1 | triton RMSNorm |
| `NVTE_USE_OPTIMIZED_HIPIFIED_CAST_TRANSPOSE` | 0 | |
| `NVTE_CK_USES_BWD_V3` / `_FWD_V3` | 1 / 1 | CK FA v3 backend |
| `NVTE_CK_IS_V3_ATOMIC_FP32` | 0 | |
| `NVTE_DEBUG` / `_DEBUG_LEVEL` | 0 / 0 | |
| `USE_TE_SWIGLU` | 1 | TE SwiGLU |
| `LORA_A2A` | 1 | LoRA all-to-all |
| `ENABLE_TRANSPOSE_CACHE` | **0** | ⚠️ 与 `keep_fp8_weight_transpose_cache=False` 一致，证实 NeMo 每 step 重做 FP8 transpose |
| `FUSED_SOFTMAX` | 0 | (NVTE_MASKED_SOFTMAX 路径) |
| `RMSNORM_CAST` | 0 | |
| `PT_TENSOR_VALIDATION` | 0 | |

### 10.3 hipBLAS / cuDNN / 其它
| ENV | 值 |
| --- | --- |
| `USE_HIPBLASLT` | 1 |
| `TORCH_BLAS_PREFER_HIPBLASLT` | 1 |
| `CUBLAS_FORCE_XMMA_KERNEL_INIT` | DEVICE |
| `CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT` | 0 |
| `OMP_NUM_THREADS` | 1 |
| `VBOOST_VALUE` | 1 |
| `AITER_LOG_LEVEL` | ERROR |
| `AITER_LOG_MORE` | 0 |
| `POSSIBLE_USER_WARNINGS` | 0 |
| `HYDRA_FULL_ERROR` | 1 |

### 10.4 Python / 路径
- `PYTHONPATH=/workspace/deps/aiter::/workspace/deps/megatron_lm`
- `HOME=/root`（in-container）
- venv: `/opt/venv/bin`
- ROCm: `/opt/rocm/{bin,lib}`

---

## 11. 与 Primus baseline 关键差异

| 维度 | NeMo（本 run）| Primus baseline | 备注 |
| --- | --- | --- | --- |
| Step time | **1.502 s** | 1.626 s | NeMo 快 8.4% |
| Throughput | 5.40 sps | ≈ 4.92 sps | |
| VRAM Pmax | ~200 GB（估）| 285.84 GB | NeMo 省 ~85 GB |
| VRAM Rmax | 211.5 GB（rocm-smi）| 295.52 GB | 同上 |
| `keep_fp8_weight_transpose_cache` | **False** | True（看起来）| NeMo 流式拉 FP8 transpose，PCIe 换 HBM |
| `gradient_accumulation_fusion` | False | True（mcore 默认）| |
| `cross_entropy_loss_fusion` | False | True（mcore 默认）| |
| `disable_parameter_transpose_cache` | False | False | 一致 |
| `bias_dropout_fusion` | True | True | |
| `apply_rope_fusion` | True | True | |
| DDP `overlap_grad_reduce` | False | True（Primus comm_overlap.setup）| NeMo 不开 |
| DDP `overlap_param_gather` | False | True（同上）| NeMo 不开 |
| `use_distributed_optimizer` | True | True | |
| LoRA `a2a_experimental` | True | True | |
| FP8 amax_history | 4 | 1024（Primus 默认）| |
| Optimizer | AdamW（mcore native）| AdamW（mcore via Bridge）| |
| AdamW betas | 0.9 / 0.999 | 0.9 / 0.999 | |
| AdamW wd | 1e-4 | 1e-4 | |
| Grad clip | 0.3 | 0.3 | |
| LR | 4e-4 | 4e-4 | |
| LR warmup | 0 | 0 | |
| Cosine 总步数 | 1024 | 1024 | |

---

## 12. 关键文件 / 复现

```bash
# 跑 NeMo + 抓 trace（host 上执行，run.sh 自己 docker exec）
cd /home/xiaompen/mlperf-training-6-0/llama2_sft/nemo
TORCH_PROFILE=1 bash run.sh
# 产物：run_traces/config_MI355X_1x8x1_<TS>/{
#   env_diff.txt, hparams_summary.txt, hydra_resolved/,
#   torchprof/{trace_*.json, key_avg_*.txt},
#   run_and_time.log, run_and_time.xtrace
# }

# 跑 kernel 分析（容器内，host 上 ijson 可能没装）
docker exec xm-nemo bash -lc \
  'cd /home/xiaompen/mlperf-training-6-0 && \
   python3 .cursor/skills/gpu-trace-analysis/scripts/full_breakdown.py \
     llama2_sft/nemo/run_traces/config_MI355X_1x8x1_20260429_054719/torchprof/trace_6_067c4d34-e473-4a5e-a69f-a6d97278ab23.json \
     ProfilerStep#5'
```

| 文件 | 说明 |
| --- | --- |
| `llama2_sft/nemo/run.sh` | 入口（有 `TORCH_PROFILE=1` 分支 + trace dir 自动 dump）|
| `llama2_sft/nemo/config_MI355X_1x8x1.sh` | 本 run 用的 sh-style config |
| `llama2_sft/nemo/src/conf/megatron_gpt_peft_tuning_config.yaml` | Hydra 配置模板 |
| `llama2_sft/nemo/src/train.py` | mcore config 实例化（`OptimizerConfig` / `DistributedDataParallelConfig` / `MegatronStrategy` / `MegatronMixedPrecision` 都在这）|
| `llama2_sft/nemo/src/custom_llama.py` | `Llama2Config70B` 包装 + `te.fp8_model_init` |
| `llama2_sft/nemo/src/callbacks/custom_callbacks.py` | torch.profiler 接入点 + MLLOG |
| `llama2_sft/nemo/src/prof_handler.py` | profiler factory |
| `run_traces/.../hydra_resolved/config.yaml` | 解析后的 hydra cfg |
| `run_traces/.../env_diff.txt` | 实际生效的训练 env |
| `run_traces/.../hparams_summary.txt` | 人眼可读 summary |
| `run_traces/.../torchprof/trace_6_*.json` | Kineto trace |
| `run_traces/.../torchprof/key_avg_6_*.txt` | per-rank `key_averages` 表 |
| `slab/notes/mlperf-llama/2026-04-29_nemo_llama2_70b_lora_trace_breakdown.md` | trace 分析 note |
| `slab/notes/mlperf-llama/2026-04-29_llama2_70b_lora_baseline_trace_breakdown.md` | Primus baseline trace 分析 note |
| `~/.cursor/projects/.../canvases/nemo-llama2-70b-lora-trace.canvas.tsx` | 可视化 canvas |

---

## 13. 一句话总结

> **TP/PP/CP/SP = 1/1/1/False，DP=8，MBS=1，GBS=8，FP8 hybrid + DelayedScaling(amax_history=4)，DistributedOptimizer 开但所有 overlap (`grad_reduce` / `param_gather` / `with_optimizer_step` / `nccl_ub` / `tp_comm_overlap`) 全关，CUDA graph / activation recompute / CPU offload 全关，`keep_fp8_weight_transpose_cache=False` 让 NeMo 每 step 流式拉 FP8 transpose（用 PCIe 带宽换 ~85 GB VRAM），LoRA r=16 α=32 仅作用于 `linear_qkv` + `linear_proj`，AdamW lr=4e-4 wd=1e-4 clip=0.3 cosine 0→1024 warmup=0。**
