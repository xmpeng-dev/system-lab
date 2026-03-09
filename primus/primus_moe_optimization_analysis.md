# Primus MoE 优化分析与规划

> **定位：** 分析 Primus Megatron backend 已有 MoE 优化，对标最新论文，输出可落地的优化建议  
> **覆盖：** 通信墙 · 负载均衡 · 显存压力 · 并行策略  
> **参考论文：** Comet · FlowMoE · LAER-MoE · MoEBlaze · MemFine · MoE Parallel Folding · MegaScale-MoE · SwiftMoE  
> **更新：** 2026-03-09

---

## 第一部分：Primus 已有 MoE 优化现状

### 1.1 通信层（A2A Dispatch / Gather）

| 组件 | 实现位置 | 功能说明 |
|------|----------|----------|
| **DeepEP dispatcher** | `primus_turbo.py::PrimusTurboDeepEPTokenDispatcher` | 基于 DeepEP 的融合 Permute + A2A；支持 `async_finish=True` + `allocate_on_comm_stream=True` |
| **Sync-free MoE** | `turbo_sync_free_moe_stage` 参数（stage > 1/2/3） | 消除 DeepEP CPU busy-wait；stage=3 实现 fully sync-free，预分配最坏情况 buffer |
| **跨层 comm-comp overlap** | `model_chunk_schedule_plan.py::execute_overlapped_1f1b()` | comm_stream 运行 `combine_bwd / dispatch_fwd`，comp_stream 运行 `attn_fwd / mlp_bwd`，层间 forward-backward 交错重叠 |
| **TP overlap** | `te_patches/tp_overlap_patches.py` | Attention 层 TP All-Reduce 与 GEMM overlap（覆盖非 MoE 计算部分）|

**关键限制：**
- DeepEP dispatcher 要求 `tensor_model_parallel_size == 1`
- 使用 `deprecated_20241209` 路径时 A2A 为**串行**（无 overlap）
- 现有 overlap 为粗粒度 op 级，不是 tile 级流水（Comet 的方向）
- 无 Tensor chunk 切分，大 Tensor 一次性 A2A（FlowMoE 的方向）

```
现有时间线（DeepEP async 模式）：

comm_stream: ──── A2A Dispatch ────                    ──── A2A Combine ────
comp_stream:                    ──── Expert GEMM ────

vs 理想目标（Comet 方向，tile 级）：

comp_stream: [GEMM_T1] [GEMM_T2] [GEMM_T3] [GEMM_T4] ...
comm_stream:  [RDMA_T1]  [RDMA_T2]  [RDMA_T3]  [RDMA_T4]  ...
              ↑ 完全流水，重叠率 90%+
```

---

### 1.2 路由 / 负载均衡层

| 组件 | 实现位置 | 功能说明 |
|------|----------|----------|
| **PrimusTopKRouter** | `core/transformer/moe/router.py` | 支持 `fused_group_topk_routing_with_aux_score`（turbo fused kernel）、softcap、group-topk、seq aux loss、global aux loss |
| **Expert Bias** | `router.py::local_tokens_per_expert` | 动态 expert bias 更新，类 DeepSeek-V3 的轻量负载均衡 |
| **Capacity Factor / Token Drop** | `apply_router_token_dropping()` | 固定 capacity factor 截断过载 Expert token，防 OOM |
| **Force Load Balancing** | `moe_router_force_load_balancing` flag | round-robin 强制均匀路由（仅用于 profiling/benchmark）|

**关键限制：**
- 负载均衡靠 aux loss 软约束，无法彻底消除热点（幂律分布下 max/avg 仍可达 3~5x）
- capacity factor 方案丢 token 损精度
- 无动态调整 Expert 分片度的机制（LAER-MoE FSEP 的核心思想完全缺失）

---

### 1.3 Expert 计算层

| 组件 | 实现位置 | 功能说明 |
|------|----------|----------|
| **PrimusTurboGroupedMLP** | `primus_turbo.py` | `pt.ops.grouped_gemm` / `grouped_gemm_fp8`；`geglu_with_probs` / `swiglu_with_probs` 融合 activation kernel |
| **Activation Recompute** | `PrimusTurboGroupedMLP.forward()` | 仅对 activation function（SiLU/GELU）部分做 checkpoint，fc1/fc2 GEMM 不保存激活 |
| **Permute Fusion** | `moe_patches/permute_fusion_patches.py` | 替换 TE 的 `fused_permute` / `fused_unpermute` / `fused_sort_chunks_by_index` |
| **Shared Expert Overlap** | `DeprecatedMoEAlltoAllTokenDispatcher` | shared expert 在 EP A2A 期间运行（支持 DeepSeek-V2/V3 架构）|

**关键限制：**
- Expert GEMM 全量 token 一次执行，激活峰值 = T × F_intermediate（MemFine 方向未覆盖）
- Smart AC 未实现（MoEBlaze 方向）：Gate 输出等轻量激活仍全量保存
- routing index buffer 有中间张量，Gate-Dispatch 未完全融合为单 kernel

---

### 1.4 并行策略层

| 组件 | 实现位置 | 功能说明 |
|------|----------|----------|
| **EP + TP 分离** | `expert_tensor_parallel_size` 配置 | Expert TP 与 Attention TP 用不同 process group |
| **DP-EP 解耦** | `LayerWiseDistributedOptimizer.shard_params()` | Expert 参数按 `expt_dp` 分组，非 Expert 参数按 `dp_cp` 分组 |
| **5D 并行** | TP × EP × DP × PP × CP | 通过 Megatron-Core 基础框架支持完整组合 |
| **Parallel Folding** | ❌ **尚未实现** | 同一 Block 内 Attn/MoE 使用不同并行策略的自动切换 |

---

## 第二部分：论文 vs Primus 现状差距分析

### 2.1 通信墙优化

#### Comet（MLSys '25）—— Tile 级 GEMM-RDMA Overlap

- **论文核心：** 在同一 CUDA Kernel 内用 Warp Specialization 让 GEMM tile 和 RDMA 并发，重叠率 90%+，MoE 层吞吐 2.3x
- **Primus 现状：** DeepEP 通过 `async_finish + comm_stream` 实现 op 级 A2A 异步，`sync_free_moe_stage` 消除 CPU 等待。但这是**粗粒度 op 级** overlap，A2A 完成前 Expert GEMM 无法开始
- **差距：** 无 tile 粒度的 warp-specialized 融合 kernel；ROCm 环境下 Warp Specialization API 与 CUDA 不同，需深度定制 HIP kernel

#### FlowMoE（NeurIPS '25）—— 跨层 DAG 调度

- **论文核心：** 统一 DAG 调度器；层 i 的 A2A 与层 i+1 的 Attn 重叠；Tensor chunk 切分提升细粒度 overlap
- **Primus 现状：** `execute_overlapped_1f1b()` 已实现**跨层微批次间** dispatch/combine 与 attn/mlp 的重叠，方向一致
- **差距：**
  - chunk 切分未覆盖（大 tensor 一次性做 A2A，而非多 chunk 流水）
  - 跨层调度仅在 1f1b 调度框架内，non-interleaved PP 不覆盖

#### MegaScale-MoE（EuroSys '26）—— 拓扑感知路由

- **论文核心：** 优先路由到同节点 Expert，减少跨 Rack 流量；分层 A2A（节点内高带宽先做，节点间按需）；A2A 延迟降低 45%，万卡 MFU ~42%
- **Primus 现状：** `collective_model.py::hierarchical_alltoall()` 有拓扑分层建模，但**实际运行时 A2A 路由不感知拓扑**
- **差距：** 缺少运行时拓扑感知 Expert 放置策略；缺少分层 EP（intra-node EP group + inter-node EP group 的两段式 A2A）

---

### 2.2 负载均衡优化

#### LAER-MoE（ASPLOS '26）—— FSEP + 动态 Re-layout

- **论文核心：** Expert 参数沿 FFN 中间维度分片到多卡（FSEP），分片度可动态调整；热点 Expert 自动扩分片度，冷点缩减；端到端 1.69x，GPU 利用率 25% → 95%
- **Primus 现状：** 只有静态 EP（每 GPU 持有固定完整 Expert）；负载均衡只靠 aux loss 软约束 + capacity factor 截断
- **差距：** **完全没有 FSEP 机制**；无动态参数分片/迁移能力；是当前端到端收益最大的空白点

```
传统 EP（Primus 现状）：          FSEP（LAER-MoE 目标）：
GPU0: [Expert_0 完整]             GPU0: [E0_shard_0/4][E1_shard_0/4]
GPU1: [Expert_1 完整]             GPU1: [E0_shard_1/4][E1_shard_1/4]
GPU2: [Expert_2 完整]     →       GPU2: [E0_shard_2/4][E1_shard_2/4]
GPU3: [Expert_3 完整]             GPU3: [E0_shard_3/4][E1_shard_3/4]

E0 过载时：                        E0 过载时：
GPU0 算 400 token ← 瓶颈 ❌         4 GPU 各算 100 token ← 均衡 ✅
GPU1/2/3 空等
```

#### SwiftMoE（arXiv '25）—— 参数-优化器解耦

- **论文核心：** Expert 参数和 Adam m/v 状态解耦存储位置；冷 Expert 的优化器状态可 offload；收敛速度 +30.5% vs DeepSpeed
- **Primus 现状：** `LayerWiseDistributedOptimizer` 有 expert 专属 `expt_dp` 分组，但优化器状态与参数**绑定在同一 GPU**
- **差距：** 无优化器状态与参数的分离存储；无基于 Expert 冷热度的 offload 策略

---

### 2.3 显存优化

#### MoEBlaze（arXiv '26）—— 数据结构 + Kernel 融合 + Smart AC

- **论文核心：** 消除 token routing 中间 buffer；Gate+Dispatch+Expert GEMM 三合一 kernel；Smart AC 对不同激活选择性保存/重算；速度 4x，显存 -50%
- **Primus 现状：** permute fusion 已合并多个 permute ops；activation function 部分有 checkpoint；但 routing index buffer 仍有中间张量
- **差距：**
  - 无 Smart AC：Gate 输出等轻量激活（重算代价低）仍全量保存
  - routing buffer 未做 compact index 表示
  - Gate+Dispatch+GEMM 端到端融合未实现

#### MemFine（arXiv '25）—— Chunk 激活调度 + 选择性重计算

- **论文核心：** Expert GEMM 以 chunk 串行执行，每次只有 T/C token 的激活存在显存中；激活显存降低 48%，吞吐 +4.42%
- **Primus 现状：** Expert GEMM 全量 token 一次执行，激活峰值 = T × F；无 chunked Expert 执行路径
- **差距：** Expert GEMM 无分 chunk 执行；高显存压力场景（大 batch/长序列）OOM 风险高

---

### 2.4 并行策略优化

#### MoE Parallel Folding（arXiv '25，NVIDIA）—— 同 Block 内 Attn/MoE 异构并行

- **论文核心：** 在 Attn 和 MoE 之间插入 All-to-All GPU 分组重映射（Folding 操作），使同一 Block 内 Attn 用 TP=4/EP=1，MoE 用 TP=1/EP=N；H100 上 Mixtral 8x22B 达 49.3% MFU
- **Primus 现状：** TP 和 EP 分别可配置，但同一 Block 内 Attn 和 MoE **使用同一套并行策略**
- **差距：** 无 Parallel Folding 机制（All-to-All GPU 分组重映射）；大规模集群上 MFU 因策略统一而受损

```
现状（Primus）：                  目标（Parallel Folding）：
Block_i:                          Block_i:
  Attn: TP=4, EP=1                  Attn: TP=4, EP=1, GPU={0,1,2,3}
  MoE:  TP=4, EP=1 ← 低效            ↓ Folding（A2A 重映射）
                                    MoE:  TP=1, EP=8, GPU={0,1,2,3,4,5,6,7}
```

---

## 第三部分：优化建议（按优先级排序）

### 3.1 🔴 高优先级（ROI 高，工程可行）

#### 建议 1：Expert GEMM Chunked 执行（对标 MemFine）

- **做什么：** 在 `PrimusTurboGroupedMLP.forward()` 中，将 `tokens_per_expert` 按 `chunk_size` 切分，循环执行 fc1→act→fc2，每次只处理 T/C token
- **接入点：** `primus/backends/megatron/core/extensions/primus_turbo.py::PrimusTurboGroupedMLP`，新增 `moe_expert_chunk_size` 参数
- **收益：** 峰值激活显存降低 C 倍，解锁更大 batch；不改通信路径，实现风险低
- **兼容性：** 需兼容 `activation_recompute`（`CheckpointWithoutOutput`）；需兼容 `use_turbo_fused_act_with_probs` 的 fused kernel

```python
# 目标代码示意（PrimusTurboGroupedMLP.forward 内）
if chunk_size > 0:
    outputs = []
    token_offset = 0
    for expert_tokens in split_by_chunk(tokens_per_expert, chunk_size):
        chunk_input = permuted_local_hidden_states[token_offset:token_offset+chunk_tokens]
        chunk_output = self._expert_forward_chunk(chunk_input, expert_tokens, w1, w2)
        outputs.append(chunk_output)
        token_offset += chunk_tokens
    fc2_output = torch.cat(outputs, dim=0)
```

---

#### 建议 2：Smart Activation Checkpointing（对标 MoEBlaze Smart AC）

- **做什么：** 扩展现有 activation_recompute 策略——对 fc1 输出（Gate proj）因重算代价低不保存；对 fc2 输入（activation 后中间值）选择性保存
- **接入点：** `PrimusTurboGroupedMLP.forward()` 中 `activation_checkpoint.checkpoint()` 扩展，增加 `smart_ac_policy` 选项
- **收益：** 在不增加重算开销的前提下再减少 15~30% 激活显存
- **注意：** 与 `swiglu_with_probs` fused kernel 需对齐（fused kernel 内部无法单独 checkpoint）

```
Smart AC 决策矩阵：
激活 tensor              重算代价    存储代价    策略
─────────────────────────────────────────────────────
fc1 输出（Gate proj）      低          高        → 不保存，反向重算
activation 函数输出        中          中        → 按现有 activation_recompute 处理
fc2 输入（act 后中间值）    高          高        → 保存（当前已保存）
routing index / probs     极低         低        → 不保存，反向重算
```

---

#### 建议 3：A2A Tensor Chunk 切分（对标 FlowMoE chunk 调度）

- **做什么：** 在 `PrimusTurboDeepEPTokenDispatcher` 中，将 dispatch 后的 token 分为多个 chunk，chunk_k 的 Expert GEMM 与 chunk_{k+1} 的 A2A（下一层 dispatch）在双流上重叠
- **接入点：** `dispatch_postprocess()` 和 `token_dispatch()` 之间插入 chunk 切分；结合现有 comm_stream/comp_stream 双流机制
- **收益：** 在 EP 规模不大（节点内 A2A 短、GEMM 长）时细化 overlap 粒度
- **前置条件：** DeepEP 需支持 partial/chunked dispatch；或在上层逻辑切分后多次调用 dispatch

---

### 3.2 🟡 中优先级（核心收益大，实现复杂度较高）

#### 建议 4：FSEP 静态分片（对标 LAER-MoE，阶段一）

- **做什么：**
  - 新增 `FSEPGroupedMLP`：Expert 参数沿 `ffn_intermediate_dim` 切分到 S 张 GPU，计算时每卡做 partial GEMM，最后 ReduceScatter 合并
  - 新增 `fsep_sharding_degree` 配置参数（静态，全局统一）
  - 在 `PrimusTopKRouter` 中利用已有 `local_tokens_per_expert` 统计暴露负载分布指标
- **接入点：** 新增 `primus/backends/megatron/core/transformer/moe/fsep_experts.py`
- **收益预期：** 在负载不均衡严重（max/avg > 3x）场景下，GPU 利用率从 ~25% 提升到 ~75%；通信额外增加 50%（+ReduceScatter），但计算效率提升 3~4x，净收益正向

```
FSEP 通信模式（S=4 分片度）：
传统 EP:  A2A Dispatch → Expert GEMM (串行) → A2A Gather
          通信量 = 2 × T × H

FSEP:     A2A Dispatch → Parallel GEMM (4 GPU 并行) → ReduceScatter → A2A Gather
          通信量 = 3 × T × H（+50%），但热点 Expert 计算从 4x avg 降到 1x avg
```

#### 建议 5：FSEP 动态 Re-layout（对标 LAER-MoE，阶段二）

- **做什么：** 新增 `LoadPlanner` 模块，每 K 步检测 expert_load 分布，对热点 Expert 提升分片度，对冷点降低分片度，在反向传播期间异步迁移参数分片
- **接入点：** `primus/backends/megatron/core/transformer/moe/` 下新增 `load_planner.py`
- **依赖：** 建议 4 的静态 FSEP 作为前置；与 ZeRO/FSDP checkpoint 的参数重分片需协调

#### 建议 6：拓扑感知 A2A 路由（对标 MegaScale-MoE）

- **做什么：** Expert 放置时优先将频繁被同节点 token 路由的 Expert 放在同节点；运行时区分 intra-node / inter-node 两段 A2A
- **接入点：** 现有 `collective_model.py::hierarchical_alltoall()` 有建模基础；需扩展到运行时实际 EP group 分层；DeepEP 需支持分层 dispatch API
- **收益：** 万卡场景 A2A 延迟降低 45%；在跨 Rack 带宽受限（25 Gbps vs 节点内 896 GB/s）时效果显著

#### 建议 7：优化器状态与 Expert 参数解耦（对标 SwiftMoE）

- **做什么：** 扩展 `LayerWiseDistributedOptimizer`，允许 Adam m/v 状态存放在不同于参数的 GPU 上；基于 expert_load 统计将冷 Expert 的优化器状态 offload 到 CPU
- **接入点：** `primus/backends/megatron/core/optimizer/layer_wise_optimizer.py::shard_params()` + `expt_dp_params_list` 逻辑扩展
- **收益：** Expert 优化器状态占总显存约 30~40%（FP32 Adam），offload 后可显著扩大 batch 或 Expert 数量

---

### 3.3 🟢 长期建议（结构性改造，高收益高复杂度）

#### 建议 8：MoE Parallel Folding（对标 NVIDIA 论文）

- **做什么：** 在 `TransformerLayer.forward()` 的 Attention→MoE 边界插入 All-to-All GPU 分组重映射（Folding），使 Attn 子层用 TP=4/EP=1，MoE 子层自动切换 TP=1/EP=N
- **接入点：** `core/transformer/transformer_layer.py` + 新增 `ParallelFoldingModule`；修改 `process_groups_config.py` 支持 per-submodule 的 `pg_collection`
- **收益：** 大规模集群（1024+ GPU）MFU 提升到 49%+（论文数据）
- **复杂度最高**：需要 Megatron-Core pg_collection 机制完整支持；Folding A2A 本身有通信开销需 benchmark 验证

#### 建议 9：Tile 级 GEMM-RDMA 融合（对标 Comet，ROCm 适配）

- **做什么：** 针对 AMD ROCm 实现 Warp Specialization 的 GEMM-RDMA overlap kernel，Expert GEMM tile 计算完成后立即通过 RCCL/HIP 发出 RDMA
- **接入点：** `primus_turbo` 底层 kernel；替换 `PrimusTurboDeepEPTokenDispatcher._exec_dispatch()` 与 Expert GEMM 的边界
- **收益：** A2A-GEMM 重叠率 15% → 90%，MoE 层吞吐 2.3x
- **挑战：** ROCm Warp Specialization 与 CUDA 有差异；需深度定制 HIP kernel

---

## 第四部分：优化路线总图

```
                    MoE 训练瓶颈 × Primus 优化建议覆盖图

                  ┌──通信墙──┬──负载不均──┬──显存压力──┬──并行冲突──┐
                  │          │            │            │            │
  建议 1 (Chunk)  │          │            │  ████████  │            │
  建议 2 (SmtAC)  │          │            │  ██████    │            │
  建议 3 (ChunkA2A│  █████   │            │  ███       │            │
  建议 4 (FSEP静) │  ██      │  ████████  │            │            │
  建议 5 (FSEP动) │          │  ████████  │            │            │
  建议 6 (拓扑A2A)│  ██████  │            │            │            │
  建议 7 (OptzDec)│          │  ████      │  ████      │            │
  建议 8 (Folding)│  ██      │            │            │  ████████  │
  建议 9 (Tile)   │  ████████│            │            │            │
                  └──────────┴────────────┴────────────┴────────────┘

图例：█ = 优化覆盖强度
```

### 推荐落地顺序（ROI 递减）

```
第一阶段（短期，1~2 个月）：低风险，显存收益立竿见影
  ├─ 建议 1：Expert GEMM Chunked 执行（MemFine 方向）
  └─ 建议 2：Smart AC 扩展（MoEBlaze 方向）
  → 目标：显存降低 30~50%，解锁更大 batch / 更长序列

第二阶段（中期，2~4 个月）：通信 + 负载，核心性能突破
  ├─ 建议 3：A2A Tensor Chunk 切分（FlowMoE 方向）
  └─ 建议 4：FSEP 静态分片（LAER-MoE 阶段一）
  → 目标：通信 overlap 率提升，负载均衡从软约束到硬约束

第三阶段（长期，4~8 个月）：结构性收益，万卡 MFU 突破
  ├─ 建议 5：FSEP 动态 Re-layout（LAER-MoE 完整版）
  ├─ 建议 6：拓扑感知分层 A2A（MegaScale-MoE 方向）
  ├─ 建议 7：优化器状态解耦（SwiftMoE 方向）
  └─ 建议 8：MoE Parallel Folding（大规模 MFU 天花板）
  → 目标：端到端 1.5x+ 加速，万卡 MFU 40%+

可选（依赖 ROCm kernel 投入）：
  └─ 建议 9：Tile 级 GEMM-RDMA 融合（Comet 方向）
  → 目标：MoE 层吞吐 2.3x（需深度 HIP kernel 开发）
```

---

## 第五部分：一句话核心结论

Primus 在**通信异步化**（DeepEP + sync-free + 跨层 1f1b overlap）和 **Router 融合 kernel** 上已有较好基础，但在三个方向与论文 SOTA 有明显差距：

1. **负载均衡**（FSEP 完全缺失）——端到端收益最大（1.69x），是最高优先级突破口
2. **显存峰值精细控制**（无 chunked GEMM / Smart AC）——实现简单，ROI 高
3. **并行策略异构化**（无 Parallel Folding）——大规模集群 MFU 上限的关键

建议以 **"显存→负载→通信→并行"** 为序逐步落地，每阶段产出可验证的端到端性能数据。

---

*文档整理于 2026-03-09 | Primus MoE 优化规划 | AIInfra-Book*
