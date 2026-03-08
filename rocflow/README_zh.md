# ROCflow：面向 AMD GPU 的通信优先 MoE 训练框架

> **版本:** 0.1（设计草案）  
> **状态:** 预发布设计阶段  
> **目标硬件:** AMD MI300X / MI325X 及后续 Instinct 系列  
> **学术支撑:** ASPLOS'26 · MLSys'25 · NeurIPS'25 · EuroSys'26 · arXiv'26

---

## 为什么需要 ROCflow？

### 一个没有人真正解决的问题

MoE（混合专家）模型——DeepSeek-V3、Mixtral、Grok-2 及其后继者——已成为前沿大语言模型的主流架构。然而用于训练它们的框架（Megatron-LM、torchtitan、DeepSpeed）都是以 **Dense 模型为核心** 设计的，MoE 支持只是事后打补丁的结果。

这个问题是架构性的：这些框架把 MoE 视为一个带有通信开销的计算问题。ROCflow 从相反的前提出发：

> **MoE 本质上是一个通信问题。计算只是简单的那部分。**

一个 MoE Transformer Block 在前向+反向传播中会产生 **9 次独立的通信操作**（2× All-to-All 分发/聚合、TP All-Reduce、CP 环形注意力、2× A2A 反向、2× 梯度 All-Reduce、FSDP 分片同步）——是同等规模 Dense Block 通信密度的 3~4 倍。在万卡规模下，仅 All-to-All 延迟就可以消耗 **50~60% 的总 Step 时间**。

现有开源框架中，没有一个拥有统一的、通信感知的调度器来同时管理这 9 种操作。ROCflow 做到了。

### AMD 生态的空白

AMD MI300X GPU 所具备的硬件能力，与解决 MoE 通信问题高度契合：

| AMD 硬件特性 | 规格 | 对 MoE 训练的价值 |
|-------------|------|-----------------|
| XGMI（Infinity Fabric）| 896 GB/s 节点内带宽 | 节点内 Expert A2A 接近内存带宽速度 |
| HBM3e | 5.3 TB/s 内存带宽 | ExpertSlotTensor 零拷贝分发 |
| 192 GB 统一 HBM 内存池 | 单节点最大内存 | FSEP 分片不 OOM |
| 独立 XGMI + RDMA 双通道 | 两条并发通信路径 | 节点内外通信真正并行 |

没有任何现有框架以对 MoE 训练有益的方式暴露这些能力。NCCL/RCCL 对所有通信一视同仁；Megatron 没有拓扑感知路由。ROCflow 从第一天起就围绕 AMD 硬件拓扑构建。

### 研究基础

ROCflow 不是一个研究项目，而是一次工程综合。每个核心 Feature 都有顶会论文背书：

| Feature | 来源论文 | 发表会议 |
|---------|---------|---------|
| FSEP 动态专家分片 | LAER-MoE | ASPLOS '26 |
| Tile 级 GEMM-RDMA Overlap | Comet | MLSys '25 |
| 跨 Block DAG 调度 | FlowMoE | NeurIPS '25 |
| Attention/MoE 并行解耦 | MoE Parallel Folding | NVIDIA/arXiv '25 |
| 智能激活检查点 | MoEBlaze + MemFine | arXiv '26/'25 |
| 拓扑感知 All-to-All | MegaScale-MoE | EuroSys '26 |
| 通信感知 Token 路由 | ROCflow 原创 | — |
| 通信原生 Tensor 布局 | ROCflow 原创 | — |

---

## 核心设计原则

### 原则一：通信是第一等公民

ROCflow 的每一个设计决策——Tensor 布局、并行配置、Kernel 设计、内存管理——都首先从 **通信代价和 Overlap 潜力** 的视角来评估。

计算效率重要，通信效率对 MoE 更重要。

### 原则二：三层 Overlap 全覆盖

现有框架在一到两个层次上解决通信 Overlap。ROCflow 同时覆盖全部三层：

```
第 3 层 │ 跨 Block 流水线       │ FlowMoE 风格的 DAG 调度器
        │                       │ Block i 通信与 Block i+1 计算重叠
────────┼───────────────────────┼────────────────────────────────────
第 2 层 │ Block 内解耦          │ Attn 和 MoE 使用独立通信路径
        │                       │ XGMI（节点内）≠ RDMA（跨节点）
────────┼───────────────────────┼────────────────────────────────────
第 1 层 │ Kernel 级 Tile        │ HIP Wavefront 专用化
        │ GEMM-RDMA 流水线      │ 计算 Wavefront + 通信 Wavefront
        │                       │ 在同一个 Kernel 内并发执行
```

没有任何现有开源框架同时实现这三层。

### 原则三：通信原生 Tensor 布局

标准的 `[Batch, Sequence, Hidden]` Tensor 布局是为计算优化的，不是为通信优化的。在做 All-to-All 分发之前，现有框架必须先把分散的 Token 收集到连续 Buffer——这是一次完整的内存拷贝，**每个 MoE 层前向传播发生两次**（反向传播中还会再来一遍）。

ROCflow 引入 **ExpertSlotTensor**：从分配之初就按通信目的地组织的 Tensor 布局。All-to-All 分发变成直接的连续内存发送，零中间拷贝。

### 原则四：路由感知通信代价

MoE 模型的 Gate 网络完全基于 Token-Expert 亲和度分数来选择专家，对目标 Expert 在同一 GPU、同一节点还是三跳之外的远端节点一无所知。

ROCflow 引入 **通信感知路由（Comm-Aware Routing）**：在路由分数中加入一个软性拓扑代价惩罚项，引导 Token 路由倾向于低代价专家，在不损害模型质量的情况下减少跨节点 All-to-All 流量。

### 原则五：统一通信 DAG

MoE Block 中的全部 9 种通信操作——A2A 分发、A2A 聚合、TP All-Reduce、梯度同步、FSDP 分片操作——都被建模为单一有向无环图（DAG）中的节点。一个中央调度器负责将每个操作分配到最优硬件路径（XGMI 或 RDMA），解析依赖关系，并最大化所有可用带宽的并发利用率。

---

## 系统架构

```
╔══════════════════════════════════════════════════════════════════════╗
║                        ROCflow 系统架构                              ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  ┌──────────────────────────────────────────────────────────────┐   ║
║  │                       用户 API 层                             │   ║
║  │                                                               │   ║
║  │  rocflow.init(...)          rocflow.MoELayer(...)             │   ║
║  │  rocflow.train(...)         rocflow.ParallelConfig(...)       │   ║
║  │  rocflow.profile(...)       @rocflow.compile                  │   ║
║  └──────────────────────────────────────────────────────────────┘   ║
║                              │                                       ║
║  ┌───────────────────────────▼──────────────────────────────────┐   ║
║  │                     并行策略引擎                               │   ║
║  │                                                               │   ║
║  │  • FSEP（全分片专家并行）                                      │   ║
║  │    - 专家参数跨所有 EP GPU 分片存储                            │   ║
║  │    - 负载自适应规划器：周期性专家重布局                         │   ║
║  │    - 无 Token Drop；通过重分布管理容量                         │   ║
║  │                                                               │   ║
║  │  • Attention / MoE 并行解耦                                   │   ║
║  │    - Attn 层：TP=N，EP=1（张量并行，参数小）                   │   ║
║  │    - MoE 层：TP=1，EP=M（专家并行，Expert 数量多）             │   ║
║  │    - 每种层类型激活不同的 GPU 分组                             │   ║
║  │                                                               │   ║
║  │  • 五维并行：TP × EP × DP × PP × CP                          │   ║
║  └───────────────────────────┬──────────────────────────────────┘   ║
║                              │                                       ║
║  ┌───────────────────────────▼──────────────────────────────────┐   ║
║  │                   统一通信 DAG 调度器                          │   ║
║  │                                                               │   ║
║  │  ┌─────────────────────────────────────────────────────────┐ │   ║
║  │  │  DAG 节点类型：                                          │ │   ║
║  │  │   [A2A分发] [A2A聚合] [TP_AR] [DP_AR] [CP环] [FSDP_AG] │ │   ║
║  │  │   [Expert_GEMM] [Attn_GEMM] [Gate] [LayerNorm]          │ │   ║
║  │  └─────────────────────────────────────────────────────────┘ │   ║
║  │                                                               │   ║
║  │  • 关键路径优先调度                                            │   ║
║  │  • 硬件路径分配：每个操作独立选择 XGMI 或 RDMA                │   ║
║  │  • 跨 Block Overlap：第 i 层通信 ∥ 第 i+1 层计算              │   ║
║  │  • 梯度 AllReduce 与反向传播计算重叠                           │   ║
║  │  • FSDP AllGather 与 MoE A2A 流水线化                         │   ║
║  └───────────────────────────┬──────────────────────────────────┘   ║
║                              │                                       ║
║  ┌───────────────────────────▼──────────────────────────────────┐   ║
║  │                   通信感知 MoE 层                              │   ║
║  │                                                               │   ║
║  │  ┌──────────────────────┐   ┌──────────────────────────────┐ │   ║
║  │  │   通信感知路由器      │   │    ExpertSlotTensor 布局     │ │   ║
║  │  │                      │   │                              │ │   ║
║  │  │  score(t,e) =        │   │  [GPU0槽][GPU1槽]...         │ │   ║
║  │  │   affinity(t,e)      │   │  按通信目的地组织，           │ │   ║
║  │  │   - λ·comm_cost(t,e) │   │  而非按序列位置组织。         │ │   ║
║  │  │                      │   │  零拷贝 A2A 分发。            │ │   ║
║  │  │  通信代价层次：       │   │                              │ │   ║
║  │  │   本地 GPU   = 0     │   │  前向/反向 Buffer             │ │   ║
║  │  │   XGMI 节点  = α     │   │  共享同一块内存区域。         │ │   ║
║  │  │   RDMA 远端  = β     │   │                              │ │   ║
║  │  └──────────────────────┘   └──────────────────────────────┘ │   ║
║  └───────────────────────────┬──────────────────────────────────┘   ║
║                              │                                       ║
║  ┌───────────────────────────▼──────────────────────────────────┐   ║
║  │                  HIP Kernel 层（AMD 原生）                     │   ║
║  │                                                               │   ║
║  │  ┌─────────────────────────────────────────────────────────┐ │   ║
║  │  │  rocflow_expert_kernel（HIP）                            │ │   ║
║  │  │                                                           │ │   ║
║  │  │  Wavefront 组 0,1,2 → Expert GEMM（矩阵计算）             │ │   ║
║  │  │  Wavefront 组 3     → XGMI/RDMA Tile 发送                │ │   ║
║  │  │                                                           │ │   ║
║  │  │  Tile N GEMM 完成 → 写入 LDS（64KB/CU）                  │ │   ║
║  │  │  通信 Wavefront 读 LDS → DMA 到对端 GPU                  │ │   ║
║  │  │  Tile N+1 GEMM 启动 ← 与上述并发执行                     │ │   ║
║  │  │                                                           │ │   ║
║  │  │  目标：在 MI300X 上达到 85%+ GEMM-RDMA Overlap 率        │ │   ║
║  │  └─────────────────────────────────────────────────────────┘ │   ║
║  │                                                               │   ║
║  │  • Gate+Dispatch+GEMM+Gather 融合 Kernel（MoE-compile pass） │   ║
║  │  • 智能激活检查点（按 Chunk，MoE 感知）                        │   ║
║  │  • Expert Group GEMM（hipBLASLt）                             │   ║
║  └───────────────────────────┬──────────────────────────────────┘   ║
║                              │                                       ║
║  ┌───────────────────────────▼──────────────────────────────────┐   ║
║  │                    AMD 硬件抽象层                              │   ║
║  │                                                               │   ║
║  │  XGMI Fabric    896 GB/s  ←→  节点内 Expert A2A              │   ║
║  │  RDMA/RoCE      400 Gbps  ←→  跨节点 A2A + 梯度同步          │   ║
║  │  HBM3e          5.3 TB/s  ←→  ExpertSlotTensor 读写          │   ║
║  │  RCCL                     ←→  兜底集合通信操作                │   ║
║  └──────────────────────────────────────────────────────────────┘   ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## 与竞争框架的核心差异

| Feature | Megatron-LM | torchtitan | DeepSpeed | **ROCflow** |
|---------|:-----------:|:----------:|:---------:|:-----------:|
| MoE 专家并行 | ✅ 基础 EP | ✅ 基础 EP | ✅ 基础 EP | ✅ **FSEP + 动态重布局** |
| Overlap 粒度 | 层级（PP） | 微批次 | 层级 | **三层：Block/Layer/Tile** |
| Tile 级 Kernel Overlap | ❌ | ❌ | ❌ | ✅ **HIP Wavefront 专用化** |
| 通信原生 Tensor 布局 | ❌ | ❌ | ❌ | ✅ **ExpertSlotTensor** |
| Attn/MoE 并行解耦 | ❌ | ❌ | ❌ | ✅ |
| 拓扑感知 A2A | ❌ | ❌ | ❌ | ✅ **XGMI 优先路由** |
| 通信感知 Token 路由 | ❌ | ❌ | ❌ | ✅ **代价惩罚 Gate** |
| 统一通信 DAG 调度器 | ❌ | ❌ | ❌ | ✅ |
| FSDP + MoE A2A 协同调度 | ❌ | ⚠️ 部分 | ❌ | ✅ **流水线化** |
| AMD ROCm 原生 Kernel | ❌ | ❌ | ❌ | ✅ **第一等公民** |
| torch.compile MoE 路径 | ❌ 不稳定 | ⚠️ 部分 | ❌ | ✅ **MoE-compile pass** |
| 无 Token Drop | ❌ | ❌ | ❌ | ✅ **通过 FSEP 重布局实现** |

---

## 用户 API

### 1. 初始化

```python
import rocflow

rocflow.init(
    backend="rccl",       # 通信后端："rccl" | "nccl"（兼容 NVIDIA）
    topology="auto",      # 硬件拓扑检测："auto" | "xgmi" | "rdma_only"
    log_level="info",     # 日志详细程度
)
```

### 2. 并行配置

```python
from rocflow import ParallelConfig

# ROCflow 允许 Attention 层和 MoE 层使用独立的并行策略
config = ParallelConfig(
    # Attention 层：TP 优先（参数小，计算密集）
    attn_tp=4,
    attn_dp=2,
    attn_cp=1,                    # 长序列上下文并行

    # MoE 层：EP 优先（专家多，通信密集）
    moe_ep=8,                     # 专家并行度
    moe_tp=1,                     # Expert 内部张量并行（超大 Expert 时有意义）
    moe_dp=2,

    # 流水线并行（两种层通用）
    pp=2,

    # FSEP：全分片专家并行
    fsep=True,
    fsep_rebalance_interval=50,   # 每 N 个 step 重新布局一次专家

    # 通信感知路由
    comm_aware_routing=True,
    routing_comm_lambda=0.1,      # 通信代价惩罚权重
                                  # 0.0 = 标准路由，增大可减少跨节点流量
)
```

### 3. MoE 层（即插即用替换）

```python
from rocflow.nn import MoELayer

# 可直接替换任意标准 MoE FFN 层
moe = MoELayer(
    hidden_size=4096,
    ffn_hidden_size=14336,
    num_experts=256,
    num_experts_per_token=8,       # Top-K 路由

    # ROCflow 专属选项
    tensor_layout="comm_native",   # 使用 ExpertSlotTensor 布局（默认值）
    overlap_mode="tile",           # Overlap 粒度："tile" | "chunk" | "layer"
    smart_ac=True,                 # 智能激活检查点（MoE 感知）
    ac_chunk_size=64,              # 每个激活检查点 Chunk 的 Token 数

    # 通信感知路由（若设置则覆盖 ParallelConfig）
    comm_aware_routing=None,       # None = 继承自 ParallelConfig
)
```

### 4. 模型定义

```python
import torch
import torch.nn as nn
from rocflow.nn import MoELayer
from rocflow.nn import wrap_attention  # 为标准 Attention 包装 TP/CP 并行逻辑

class MyMoETransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn    = wrap_attention(MyAttention(config))  # 自动处理 TP/CP
        self.norm1   = nn.RMSNorm(config.hidden_size)
        self.norm2   = nn.RMSNorm(config.hidden_size)
        self.moe_ffn = MoELayer(
            hidden_size=config.hidden_size,
            ffn_hidden_size=config.ffn_hidden_size,
            num_experts=config.num_experts,
            num_experts_per_token=config.top_k,
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.moe_ffn(self.norm2(x))
        return x
```

### 5. 训练循环

```python
import rocflow
from rocflow import ParallelConfig, Trainer

parallel_config = ParallelConfig(
    attn_tp=4, attn_dp=2,
    moe_ep=8,  moe_dp=2,
    pp=2,
    fsep=True,
    comm_aware_routing=True,
)

trainer = rocflow.Trainer(
    model=model,
    optimizer=optimizer,
    parallel_config=parallel_config,

    # 精度与梯度
    grad_clip=1.0,
    precision="bf16",
    compile=True,              # 启用 rocflow.compile（MoE 感知的 torch.compile pass）

    # Overlap 配置
    overlap_attn_moe=True,     # 启用 Attn/MoE 通信路径解耦
    overlap_grad_ar=True,      # 梯度 AllReduce 与反向传播重叠

    # 检查点
    checkpoint_dir="./checkpoints",
    checkpoint_interval=1000,
    async_checkpoint=True,     # 非阻塞检查点保存
)

for batch in dataloader:
    loss = trainer.step(batch)
    if trainer.is_rank_zero():
        print(f"step={trainer.step_count}, loss={loss:.4f}, "
              f"mfu={trainer.mfu:.1%}, "
              f"comm_overlap={trainer.comm_overlap_rate:.1%}")
```

### 6. 性能分析与诊断

```python
# 在正式训练前分析通信瓶颈
from rocflow.profiler import CommProfiler

profiler = CommProfiler(model, parallel_config)
report = profiler.analyze(sample_batch)

print(report.summary())
# 输出示例：
# ┌─────────────────────────────────────────────────────────────┐
# │ ROCflow 通信分析报告                                          │
# ├──────────────────────┬─────────────┬────────────────────────┤
# │ 操作                  │ 耗时(ms)    │ Overlap 机会           │
# ├──────────────────────┼─────────────┼────────────────────────┤
# │ A2A Dispatch（前向）  │ 8.3ms       │ ✅ 与 GEMM Tile 重叠   │
# │ A2A Gather（前向）    │ 7.9ms       │ ✅ 与 GEMM Tile 重叠   │
# │ TP All-Reduce         │ 2.1ms       │ ✅ XGMI 路径，并行     │
# │ 梯度 All-Reduce（DP） │ 12.4ms      │ ✅ 与反向传播重叠      │
# │ FSDP All-Gather       │ 5.6ms       │ ✅ 与 A2A 流水线化     │
# ├──────────────────────┼─────────────┼────────────────────────┤
# │ 总暴露通信时间         │ 3.2ms       │ Overlap 率：91.3%     │
# └──────────────────────┴─────────────┴────────────────────────┘
#
# 专家负载分布：max/avg 比 = 1.08（健康，FSEP 生效）
# 跨节点路由流量：23% 的 Token（comm_lambda=0.1）
# 建议：将 comm_lambda 提升至 0.15 以进一步减少跨节点流量

# 逐层详细分析
print(report.per_layer_breakdown())   # 每层通信耗时拆解
print(report.expert_load_heatmap())   # 专家负载热力图
print(report.topology_traffic_map())  # 拓扑流量分布图
```

### 7. torch.compile 集成

```python
# ROCflow 提供专为 MoE 设计的 compile pass，正确处理动态路由
from rocflow import rocflow_compile

# 方案 A：编译整个模型（生产环境推荐）
model = rocflow_compile(
    model,
    mode="max-autotune",      # 使用 hipBLASLt 对 Expert GEMM 自动调优
    dynamic_routing=True,     # 启用符号形状追踪，支持可变 Token 数
    fuse_gate_dispatch=True,  # 将 Gate + Dispatch 融合为单个 HIP Kernel
)

# 方案 B：仅编译 MoE 层（调试非 MoE 部分时使用）
from rocflow.nn import compile_moe_layers
model = compile_moe_layers(model, mode="reduce-overhead")
```

### 8. 容错（生产级大规模训练）

```python
# 启用在线容错，适用于大规模训练任务
trainer = rocflow.Trainer(
    model=model,
    optimizer=optimizer,
    parallel_config=parallel_config,

    # 容错配置
    fault_tolerance=True,
    ft_detection_interval=60,     # 每 60 秒检查一次 GPU 健康状态
    ft_strategy="reroute",        # "reroute"（在线重路由）| "checkpoint_restart"
    ft_spare_gpus=2,              # 保留 N 块备用 GPU 用于热替换
)
```

---

## 性能目标（MI300X，8×MI300X 节点）

| 模型 | 配置 | MFU 目标 | 相比 Megatron 基线 |
|------|------|---------|-----------------|
| Mixtral 8×7B | 64 GPUs，EP=8 | **52%+** | 约 +30% |
| DeepSeek-V3 量级（671B MoE）| 1024 GPUs，EP=64 | **45%+** | 约 +25% |
| 自定义 100B MoE | 256 GPUs，EP=32 | **48%+** | 约 +28% |

相比 Megatron-LM 在同等 AMD 硬件上的预期改善：
- **计算-通信 Overlap 率**：85~92%（Megatron 约 20%）
- **专家负载不均衡比**：< 1.1×（无 FSEP 时为 2~3×）
- **跨节点 A2A 流量**：通过通信感知路由减少 30~40%
- **激活内存峰值**：通过 Smart AC + ExpertSlotTensor 降低 40~50%

---

## 路线图

### Phase 1 — 核心框架（2026 Q2）
- [ ] ExpertSlotTensor 布局 + 零拷贝 A2A 分发
- [ ] 统一通信 DAG 调度器
- [ ] FSEP 实现 + 负载自适应规划器
- [ ] Attention/MoE 并行解耦
- [ ] Expert GEMM 基础 HIP Kernel（hipBLASLt Group GEMM）
- [ ] CommProfiler 工具

### Phase 2 — Overlap 最大化（2026 Q3）
- [ ] HIP Wavefront 专用化 Kernel（Tile 级 GEMM-RDMA Overlap）
- [ ] 跨 Block 流水线 Overlap（FlowMoE 风格 DAG）
- [ ] XGMI 优先的拓扑感知路由
- [ ] 梯度 AllReduce + 反向传播 Overlap
- [ ] FSDP2 + MoE A2A 协同调度

### Phase 3 — AMD 深度优化 + compile（2026 Q4）
- [ ] MoE-compile pass（Gate+Dispatch+GEMM+Gather 融合）
- [ ] 通信感知路由的 λ 自适应调优
- [ ] 智能激活检查点（MoE 感知的 Chunk 策略）
- [ ] MI325X / 下一代 Instinct 支持
- [ ] 万卡级完整容错能力

### Phase 4 — 生态建设（2027 Q1）
- [ ] 内置 DeepSeek-V3 / Mixtral 模型配置
- [ ] 与 Megatron-LM 的公开 Benchmark 套件
- [ ] 与 AMD ROCm 性能工具集成（Omniperf / rocProf）
- [ ] 社区插件 API（支持自定义 Expert 实现）

---

## 设计讨论记录

> *本节记录仍在讨论中的开放设计问题。*

**Q1：通信感知路由的 λ 调度策略**  
λ 应该是静态超参，还是基于观测到的跨节点流量比例进行在线自动调优？LAER-MoE 的 Load Planner 提供了在线自适应的先例。候选方案：训练初期 λ=0 保证收敛，后期跟随路由熵稳定性的 schedule 逐步增大到目标值。

**Q2：ExpertSlotTensor 与 torch.compile 的兼容性**  
动态路由导致的可变长 Slot 会破坏 torch.compile 的静态形状假设。候选解决方案：compile 时将 Slot padding 到 `capacity = max_tokens_per_expert`，实际分发使用 mask。以约 5% 的 padding 浪费换回静态形状。

**Q3：FSDP2 + FSEP + A2A 的死锁规避**  
当 FSDP2 AllGather、FSEP 专家重布局和 MoE A2A 同时激活时，统一 DAG 必须强制执行顺序约束以防止 RCCL 死锁。这是 Phase 2 最高优先级的正确性挑战。

**Q4：反向传播 A2A 的调度自由度**  
前向 A2A 的顺序由层依赖关系固定。反向传播 A2A（作为其逆操作）拥有更多调度自由度——第 N 层 A2A-Gather 的梯度理论上可以与第 N-1 层 Expert 的反向 GEMM 重叠。这个自由度至今没有被任何现有框架系统性地利用。

---

## 相关工作

| 论文 | 发表会议 | ROCflow 的借鉴点 |
|------|---------|----------------|
| LAER-MoE | ASPLOS '26 | FSEP 架构、负载自适应规划器 |
| Comet | MLSys '25 | Tile 级 GEMM-通信 Overlap、Warp 专用化 |
| FlowMoE | NeurIPS '25 | 统一 DAG 调度器、跨 Block Overlap |
| MoE Parallel Folding | arXiv '25 | 五维并行、Attn/MoE 解耦 |
| MegaScale-MoE | EuroSys '26 | 拓扑感知 A2A、生产级容错 |
| MoEBlaze | arXiv '26 | Token 分发数据结构、Smart AC |
| MemFine | arXiv '25 | Chunk 级激活调度 |
| SwiftMoE | arXiv '25 | 优化器-参数解耦 |

---

## 参与贡献

ROCflow 被设计为模块化框架。每个子系统（DAG 调度器、FSEP 规划器、HIP Kernel、通信感知路由器）都有清晰的接口，可以独立贡献。

开发环境搭建和测试框架说明见 `CONTRIBUTING.md`。

---

*ROCflow —— MoE 训练值得一个从一开始就把通信当作首要问题来解决的框架。*
