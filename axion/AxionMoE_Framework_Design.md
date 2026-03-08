# AxionMoE — A Diagnostic-First MoE Training Framework

> **作者:** Xiaoming Peng  
> **版本:** v0.1-draft  
> **定位:** 面向中大规模 MoE 模型训练的**可诊断、可复现、模块化**训练框架  
> **差异化关键词:** *Observability-First · Decoupled Scheduling · Reproducible by Default*

---

## 0. 动机：现有框架缺少什么？

| 框架 | 优势 | 缺失 |
|------|------|------|
| **Megatron-LM** | 极致性能，5D 并行 | 黑盒调度，难以诊断；与 PyTorch 生态割裂 |
| **TorchTitan** | PyTorch 原生，易于扩展 | MoE 支持浅，无 Expert Parallel 一等公民 |
| **DeepSpeed MoE** | EP 完善，ZeRO 集成好 | 调度固化，无跨层 overlap；通信可见性差 |
| **veScale / FSDP2** | ZeRO-3 优雅，SPMD 友好 | MoE 负载均衡原语缺失 |

**AxionMoE 的核心判断：**

> 当前 MoE 训练最大的隐性成本不是"峰值 MFU 不够高"，而是：  
> **"你不知道为什么今天的 MFU 比昨天低了 8%。"**

Megatron 和 TorchTitan 都缺少一套面向 MoE 的**在线诊断 + 可复现 profiling** 能力。  
AxionMoE 把 **Observability（可观测性）** 作为一等公民设计进框架，而非事后贴补。

---

## 1. 框架总览

```
AxionMoE Architecture
═══════════════════════════════════════════════════════════════════
  ┌────────────────────── User Layer ─────────────────────────────┐
  │   axion.train(model, dataloader, config)                       │
  │   axion.eval_mfu()  ·  axion.preflight()  ·  axion.replay()   │
  └────────────────────────────────────────────────────────────────┘
              │                        │
  ┌───────────▼────────────┐  ┌────────▼───────────────────────────┐
  │  Scheduling Layer      │  │  Observability Layer               │
  │  (FlowGraph Engine)    │  │  (AxionScope)                      │
  │  · DAG-based dispatch  │  │  · Per-step MoE telemetry          │
  │  · Decoupled Attn/MoE  │  │  · Expert load heatmap             │
  │  · Chunk-priority sched│  │  · Comm/compute timeline           │
  └─────────┬──────────────┘  └──────────────┬─────────────────────┘
            │                                 │
  ┌─────────▼─────────────────────────────────▼──────────────────┐
  │  Parallelism Layer                                            │
  │  · EP (Expert Parallel)  ·  TP  ·  PP  ·  DP  ·  CP         │
  │  · Attn-MoE decoupled parallel config (Parallel Folding)     │
  │  · Hierarchical EP (intra-node NVLink / inter-node IB)       │
  └──────────────────────────┬────────────────────────────────────┘
                             │
  ┌──────────────────────────▼────────────────────────────────────┐
  │  Expert Management Layer                                      │
  │  · Static optimizer state + dynamic param placement (Swift)  │
  │  · Load-adaptive re-layout (FSEP / LAER)                     │
  │  · Topology-aware expert placement (MegaScale)               │
  └───────────────────────────────────────────────────────────────┘
              │
  ┌───────────▼────────────────────────────────────────────────────┐
  │  Kernel Layer                                                  │
  │  · Tile-level compute-comm overlap (Comet-style)              │
  │  · Expert Group GEMM  ·  Fused Gate+Dispatch                  │
  │  · Chunk-based activation memory (MemFine)                    │
  └────────────────────────────────────────────────────────────────┘
```

---

## 2. 核心 Feature 详解

### Feature 1: AxionScope — MoE 训练可观测系统（最醒目差异化）

**来源灵感：** MegaScale-MoE 的生产诊断需求 + 我的 profiling/diagnostics 工作背景

**问题：** 现有框架（Megatron、TorchTitan）没有针对 MoE 特有事件的结构化追踪：
- 不知道哪个 Expert 在哪个 step 过载
- 不知道 All-to-All 延迟在哪层最高
- 不知道 Parallel Folding 的 Fold/Unfold 实际消耗多少时间
- 复现一个 "MFU 抖动" 现象需要手动加大量 print

**AxionScope 的设计：**

```python
# 用法示例 — 无需改模型代码
import axion

trainer = axion.Trainer(model, config)

# 自动采集：每 step 的 MoE 遥测数据
with axion.scope("run_001") as scope:
    trainer.train(dataloader)

# 查询诊断报告
report = scope.report()
report.expert_load_heatmap()     # Expert 负载热力图（按层、按 step）
report.comm_timeline()           # 每层 All-to-All 延迟时间轴
report.mfu_breakdown()           # MFU 归因：计算 / 通信 / 调度 / 内存
report.anomaly_steps()           # 异常 step 自动定位（MFU 骤降 / 死锁预警）
```

**AxionScope 追踪的核心事件（不同于 PyTorch Profiler 的粒度）：**

| 事件类型 | 追踪内容 | 用途 |
|---------|---------|------|
| `ExpertLoadEvent` | 每层每 Expert 收到的 token 数、计算耗时 | 负载均衡诊断 |
| `A2ALatencyEvent` | All-to-All Dispatch/Gather 延迟、带宽利用率 | 通信瓶颈定位 |
| `FoldLatencyEvent` | Parallel Folding 的 Fold/Unfold 实际耗时 | 并行策略调优 |
| `MemPressureEvent` | 激活内存峰值时刻、Chunk 释放时机 | 内存 OOM 预防 |
| `GradSyncEvent` | 梯度 All-Reduce 与 Backward 的实际 overlap 率 | 识别反向传播瓶颈 |
| `ReplayCheckpoint` | 可复现 profiling 所需的所有随机种子 + 路由状态 | 复现 profiling |

**关键差异：** AxionScope 是**零侵入**的——通过 PyTorch Dispatch Key 和 CUDA Event Hook 注入，不修改模型代码，开销 < 2%。

---

### Feature 2: FlowGraph Engine — 统一 DAG 调度引擎

**来源灵感：** FlowMoE 的 Chunk 优先级调度 + Comet 的 Tile 级 overlap

**问题：** Megatron 的调度是"Micro-batch 级别的流水线"，粒度太粗；FlowMoE 是独立系统，没有整合进主流框架。

**AxionMoE 的设计：** 把 FlowMoE 的 DAG 调度作为框架**内置调度引擎**，而非可选插件。

```
FlowGraph DAG（一个 MoE Transformer Block）：

Forward DAG:
  Input
    │
    ├─[MHA_chunk_1] ─→ [A2A_D_chunk_1] ─→ [Expert_chunk_1] ─→ [A2A_G_chunk_1]
    │      ↕ overlap
    └─[MHA_chunk_2] ─→ [A2A_D_chunk_2] ─→ [Expert_chunk_2] ─→ [A2A_G_chunk_2]
                 ↕ overlap
              [MHA_chunk_3] ...

关键设计：
  1. Chunk 大小自适应：根据 GPU 内存和网络带宽自动决定 Chunk Size
  2. 优先级计算：Critical Path First（通信 task 给额外优先级）
  3. 跨层 overlap：Layer i 的 All-to-All 与 Layer i+1 的 MHA 并发
  4. 反向传播：AllReduce 与 Backward GEMM 流水线化
```

**与 Megatron 的关键区别：**

| 维度 | Megatron | AxionMoE FlowGraph |
|------|----------|--------------------|
| 调度粒度 | Micro-batch | Tensor Chunk |
| MoE 通信 overlap | 有限（PP bubble 优化为主）| 主动 DAG 跨层 overlap |
| 调度可见性 | 无 | AxionScope 全程追踪 |
| 自适应 Chunk | 无（手动配置）| 自动调优 |

---

### Feature 3: 解耦并行配置（Attn-MoE Parallel Decoupling）

**来源灵感：** MoE Parallel Folding（NVIDIA arXiv'25）

**问题：** Megatron 虽然支持 MoE，但并行配置是整个 Transformer Block 共用的，Attention 和 MoE FFN 无法独立设置 TP 和 EP。

**AxionMoE 的设计：** 把解耦并行配置做成**声明式 API**，用户只需一个 config，框架自动管理 Fold/Unfold 通信。

```python
# AxionMoE 并行配置（声明式）
parallel_config = axion.ParallelConfig(
    # Attention 层并行策略
    attn=axion.ParallelDim(tp=4, cp=2, dp=8),

    # MoE FFN 层并行策略（可以与 Attention 完全不同）
    moe=axion.ParallelDim(ep=32, tp=1, dp=8),

    # Pipeline 并行（跨两者）
    pp=4,

    # Fold/Unfold 通信是否与 All-to-All 合并（默认开启）
    fuse_fold_a2a=True,
)

# 框架自动管理 Parallel Fold 通信，用户无感知
trainer = axion.Trainer(model, parallel_config)
```

**与 TorchTitan 的关键区别：**

TorchTitan 采用 `DeviceMesh` + FSDP2 的方式，只有一套 Mesh 定义，无法原生表达"Attention 用 TP=4，MoE 用 EP=32"的异构并行。  
AxionMoE 内置 `ParallelConfig` 的 Attention/MoE 解耦，是一等公民设计。

---

### Feature 4: AxionExpert — 智能 Expert 生命周期管理

**来源灵感：** SwiftMoE（参数-优化器解耦）+ LAER-MoE（FSEP 动态重排）+ MegaScale-MoE（拓扑感知放置）

**问题：** 现有框架的 Expert 要么完全静态（Megatron），要么动态但迁移开销大（FlexMoE）。

**AxionMoE 的三层 Expert 管理：**

```
Expert 生命周期三层架构：

┌─────────────────────────────────────────────────────────┐
│ Layer 3: 拓扑感知初始放置（训练开始时一次性）              │
│   · 分析训练数据前 K 步的路由统计（warm-up profiling）    │
│   · 将高相关性 Expert 放在同 Rack（减少跨 Rack A2A）      │
│   · 参考 MegaScale-MoE 的 METIS 图分区放置算法           │
├─────────────────────────────────────────────────────────┤
│ Layer 2: 优化器状态静态 + 参数动态（训练过程中）           │
│   · Expert 参数根据负载动态迁移（仅参数，1x 开销）         │
│   · 优化器状态（Adam m1/m2）永远固定在"主 GPU"            │
│   · 梯度汇聚回主 GPU 更新，更新后广播参数副本              │
│   · 参考 SwiftMoE 的参数-优化器解耦                       │
├─────────────────────────────────────────────────────────┤
│ Layer 1: FSEP 全分片 + 负载重排（热点 Expert 实时均衡）    │
│   · 每 K steps 检测一次负载不均                           │
│   · 触发 Expert 分片重布局（参考 LAER-MoE FSEP）          │
│   · 重排与训练 overlap，不阻塞前向计算                    │
└─────────────────────────────────────────────────────────┘
```

**AxionExpert API：**

```python
expert_config = axion.ExpertConfig(
    # Expert 放置策略
    placement=axion.ExpertPlacement.TOPOLOGY_AWARE,   # or UNIFORM / LOAD_ADAPTIVE

    # 参数-优化器解耦（SwiftMoE 风格）
    decouple_optimizer_state=True,

    # 负载均衡策略
    load_balance=axion.LoadBalance(
        strategy="fsep",            # or "auxiliary_loss" / "capacity_factor"
        rebalance_interval=50,      # 每 50 steps 检测一次
        rebalance_threshold=1.5,    # 最大/最小 Expert 负载比 > 1.5 时触发
    ),
)
```

---

### Feature 5: Preflight System — 训练前健康检查

**来源灵感：** 我的 profiling/diagnostics 工作背景 + MegaScale-MoE 的生产容错经验

**问题：** 大规模 MoE 训练启动一次成本极高（数百 GPU 小时的准备时间）。  
现有框架没有在**训练开始前**系统性检测潜在问题：

- 并行配置是否会导致内存 OOM？（发现时已跑了 30 分钟）
- 网络拓扑与 Expert Parallel 分组是否匹配？
- 数据集 token 分布是否会导致特定 Expert 严重过载？
- 当前 checkpoint 与配置版本是否兼容？

**AxionMoE Preflight 的检测项：**

```python
# 一行启动预检
axion.preflight(model, config, dataloader_sample)

# 输出示例：
"""
AxionMoE Preflight Report
═══════════════════════════════════════════════════

[✅] Memory check: Peak activation est. 38.2 GB / 40 GB (95.5%) — OK
[✅] Network topology: EP=32 matches rack boundaries — OK
[⚠️] Expert load imbalance: Warm-up routing shows Expert 17, 43 at 2.3x avg load
      Suggestion: Enable FSEP rebalancing or adjust routing temperature
[❌] Checkpoint compat: config.num_experts=128 but ckpt has 64 experts
      Action required: Re-initialize expert parameters or use --expand-experts
[✅] NCCL All-to-All latency: 4.2ms (expected < 10ms) — OK
[⚠️] PP bubble ratio est.: 18% with pp=8, micro_batch=4
      Suggestion: Increase micro_batch to 8 for better efficiency
[✅] Reproducibility: Random seeds captured, routing states logged

Estimated first-step MFU: 41.2% ± 3.1%
Estimated time to 1B tokens: 2h 37m
"""
```

**与现有框架的差异：** Megatron 和 TorchTitan 没有 Preflight 系统。训练失败只能事后 debug。  
AxionMoE 把 "fail fast" 文化嵌入框架，在训练开始前暴露 80% 的潜在问题。

---

### Feature 6: Reproducible Training — 可复现训练

**来源灵感：** 我的 reproducibility 工作背景

**问题：** MoE 训练由于动态路由（每 step 路由结果不同），天然难以复现：
- 相同 checkpoint 在不同 GPU 数量下行为不同
- Expert 过载导致的 Token Drop 路径不确定
- 梯度 All-Reduce 的 floating point 累加顺序随 GPU 拓扑变化

**AxionMoE 的可复现设计：**

```python
# 开启完整可复现模式
config = axion.TrainConfig(
    reproducibility=axion.Reproducibility(
        mode="full",               # or "best_effort" (更快但非完全确定)
        capture_routing_states=True,   # 记录每 step 的路由决策
        deterministic_a2a=True,    # All-to-All 确定性顺序
        seed=42,
    )
)

# 在任意 step 重放（用于 profiling 和 debug）
axion.replay(
    checkpoint="run_001/step_1000.ckpt",
    from_step=1000,
    to_step=1010,
    scope=axion.scope("debug_replay"),  # 配合 AxionScope 诊断
)
```

**核心机制：**
1. **路由状态快照：** 每隔 N steps 保存完整的 Expert 路由统计（不是参数，只是统计量，开销极小）
2. **确定性 All-to-All：** 固定 token-to-GPU 排序，消除顺序不确定性
3. **Replay Mode：** 从任意 step 精确重放 K 步，配合 AxionScope 做细粒度 profiling

---

## 3. 与主流框架的全面对比

| Feature | Megatron-LM | TorchTitan | DeepSpeed MoE | **AxionMoE** |
|---------|------------|------------|--------------|--------------|
| **MoE 支持深度** | ✅ 完善 | ⚠️ 基础 | ✅ 完善 | ✅ 完善 |
| **Attn/MoE 并行解耦** | ⚠️ 有限 | ❌ 无 | ❌ 无 | ✅ **一等公民** |
| **DAG 调度（跨层 overlap）** | ❌ | ❌ | ⚠️ 部分 | ✅ **FlowGraph** |
| **Expert 负载诊断（实时）** | ❌ | ❌ | ❌ | ✅ **AxionScope** |
| **通信延迟可视化** | 靠 Nsight | 靠 PyTorch Profiler | ❌ | ✅ **内置时间线** |
| **Preflight 检查** | ❌ | ❌ | ❌ | ✅ **内置** |
| **可复现训练** | ⚠️ 依赖种子 | ⚠️ 依赖种子 | ⚠️ 依赖种子 | ✅ **路由状态快照** |
| **参数-优化器解耦** | ❌ | ❌ | ❌ | ✅ **SwiftMoE 风格** |
| **拓扑感知 Expert 放置** | ❌ | ❌ | ❌ | ✅ **内置** |
| **Tile 级 compute-comm overlap** | ⚠️ 需手动内核 | ❌ | ❌ | ✅ **Comet 风格** |
| **PyTorch 原生兼容** | ❌（自有生态）| ✅ | ⚠️ 部分 | ✅ **原生** |
| **Benchmark / 诊断工具链** | 外部 | 外部 | 外部 | ✅ **内置一体** |

---

## 4. 技术栈与依赖

```
AxionMoE 技术栈：

用户接口层：  Python API (axion.*)
框架层：      PyTorch 2.x + torch.distributed + FSDP2
调度层：      自研 FlowGraph Engine (C++ + Python binding)
通信层：      NCCL / DeepEP (可替换) + 自研 A2A Tile 发送
内核层：      Triton / CUDA (Expert Group GEMM, Fused Gate)
诊断层：      AxionScope (PyTorch Dispatch Hook + CUDA Event)
存储层：      分布式 Checkpoint (torch.distributed.checkpoint)

外部依赖：
  - PyTorch >= 2.4
  - NCCL >= 2.20（或 DeepEP 替代）
  - Flash Attention 2/3
  - Triton >= 2.3（Expert 内核）
  - wandb / TensorBoard（可选，AxionScope 输出对接）
```

---

## 5. 目标场景与规模定位

| 场景 | GPU 规模 | 模型规模 | 推荐配置 |
|------|---------|---------|---------|
| **研究探索** | 8~64 GPUs | 7B~30B MoE | `AxionMoE-Lite`（只开 AxionScope + FlowGraph）|
| **中规模训练** | 64~512 GPUs | 30B~100B MoE | `AxionMoE-Standard`（全 Feature）|
| **大规模训练** | 512~4096 GPUs | 100B~300B MoE | `AxionMoE-Pro`（+ 分层 EP + 拓扑感知）|
| **生产诊断** | 任意 | 任意 | `AxionMoE-Scope`（仅 Scope，可嵌入现有 Megatron/DeepSpeed）|

> **注：** `AxionMoE-Scope` 是一个独立的 MoE 诊断工具包，可以作为现有框架的插件使用。  
> 这是最低 MVP，也是最容易先落地的差异化产品。

---

## 6. Roadmap

### Phase 0: MVP（AxionScope 独立发布）
- [ ] AxionScope 核心事件追踪（ExpertLoad + A2ALatency）
- [ ] 与 Megatron-LM / DeepSpeed MoE 的插件集成
- [ ] Expert 负载热力图可视化

### Phase 1: 框架核心
- [ ] FlowGraph Engine（基于 FlowMoE 思路）
- [ ] Attn-MoE 解耦并行配置（基于 Parallel Folding 思路）
- [ ] Preflight System（基础检测项）

### Phase 2: Expert 管理
- [ ] SwiftMoE 风格参数-优化器解耦
- [ ] FSEP 负载重排（LAER-MoE 思路）
- [ ] 拓扑感知 Expert 初始放置

### Phase 3: 性能内核
- [ ] Comet 风格 Tile 级 compute-comm overlap
- [ ] Expert Group GEMM 优化内核
- [ ] MemFine 风格 Chunk 激活调度

### Phase 4: 生产化
- [ ] 分层 EP（节点内 NVLink + 节点间 IB）
- [ ] 在线故障检测 + Expert 热重路由
- [ ] 可复现训练全流程

---

## 7. 核心差异化总结

AxionMoE 有三个在现有框架中**完全空白**的 Feature：

### 🔭 Observability-First（最核心差异化）
Megatron/TorchTitan/DeepSpeed 都没有针对 MoE 的**结构化实时诊断**。  
AxionMoE 把 Expert 负载、通信延迟、Fold 开销、内存压力做成**一等公民遥测数据**，  
并支持从任意 step **精确复现**来 debug。

### ⚡ Scheduling Decoupling（性能差异化）
把 FlowMoE 的 DAG 调度 + MoE Parallel Folding 的解耦并行**整合进框架核心**，  
而非像 Megatron 那样通过复杂 config 手动拼装。  
用户用声明式 `ParallelConfig` 就能自动获得 Attn/MoE 解耦并行 + 跨层 overlap。

### 🛡️ Preflight + Reproducibility（工程差异化）
"训练开始前知道会发生什么"是工业级框架和学术框架的分水岭。  
AxionMoE 把 preflight + 可复现机制内置，  
对齐 MegaScale-MoE 的生产工程经验，但以 **PyTorch 原生 + 开源** 的方式提供。

---

*设计文档 v0.1 | AxionMoE | Xiaoming Peng | 2026-03-08*
