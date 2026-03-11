# MoEX: Communication-First MoE Training System

> **设计哲学**: MoE 本质上是一个通信问题。MoEX 从最底层的 tensor 存储结构出发，
> 让 tensor 本身就适合 dispatch / combine / overlap / 异构并行，而不是在计算友好的
> tensor 上层叠通信优化。

---

## 目录

- [背景与动机](#背景与动机)
- [核心创新：CommTensor](#核心创新commtensor)
- [系统架构](#系统架构)
- [与现有系统的对比](#与现有系统的对比)
- [论文支撑](#论文支撑)
- [设计文档](#设计文档)

---

## 背景与动机

### MoE 训练的本质是通信问题

以 DeepSeek-V3（671B 参数，256 专家）为例，一次 MoE forward 的时间分布如下：

```
传统 MoE Forward 时间分布（无优化）:
┌─────────────────────────────────────────────────┐
│ Gate   │ A2A-Dispatch │ Expert-GEMM │ A2A-Gather │
│  ~5%   │    ~35%      │    ~25%     │   ~35%     │
└─────────────────────────────────────────────────┘
                 ↑
         通信占 70%，计算只占 30%
```

现有系统的改进思路：在计算友好的 tensor 上**叠加**通信优化：
- MegatronCore：双流调度，overlap A2A 与 GEMM（~93% overlap）
- Comet：tile 级 GEMM-RDMA pipeline（90%+ overlap）
- FlowMoE：跨层 chunk 级 DAG 调度（13-57% 加速）
- LAER-MoE：FSEP 动态 expert 重布局（1.69× 端到端）

### 问题的根源

这些系统都面临一个共同的**根本性障碍**：

```python
# 传统 MoE 的 tensor 存储顺序
hidden_states = torch.tensor([seq_len, d_model])  # 按 sequence 顺序存储

# dispatch 时必须 permute（按 expert 目标排序）→ 内存拷贝
sorted_tokens = hidden_states[routing_indices]      # O(T*H) 拷贝

# 发送时还需构建 send buffer（再次拷贝）
send_buffer = pack_tokens(sorted_tokens, counts)    # O(T*H) 再次拷贝

# All-to-All 之后还需 unpack（再次拷贝）
received = unpack_tokens(recv_buffer, expert_map)  # O(T*H) 第三次拷贝
```

**每次 dispatch/combine 都有 2-3 次 O(T×H) 的内存拷贝**，在 DeepSeek-V3 规模下，
T×H = 4096 × 7168 = 29M float16 = ~58MB per dispatch，严重占用内存带宽。

### MoEX 的核心洞察

> **让 tensor 的物理存储顺序 = 通信目标顺序**

MoEX 设计了 **CommTensor**（Communication-Native Tensor），其物理内存布局本身就是
"按通信目标（rank）优先排列的"。路由完成后，CommTensor 已经是 dispatch 就绪状态：

```
CommTensor 物理布局: [num_ranks, slots_per_rank, tile_size, d_model]
                       ↑           ↑              ↑          ↑
                  EP目标rank   每rank的token槽   GEMM tile粒度  隐维度

dispatch = RDMA send CT[rank_i, :, :, :]  ← 零拷贝，直接从内存地址发送
combine  = RDMA recv → CT[rank_i, :, :, :]  ← 零拷贝，直接写入内存
GEMM     = 对 CT[:, :, tile, :] 做矩阵乘  ← tile 对齐，即完成即发送（Comet风格）
FSEP     = ReduceScatter on CT[rank, :, :, :]  ← NVLink 高带宽路径
```

---

## 核心创新：CommTensor

### 1. 四维通信原生布局

```
CommTensor Shape: [R, S, T, H]

R = num_ep_ranks       # Expert Parallel 的 rank 数（通信目标）
S = max_slots_per_rank # 每个 rank 分配的 token 槽位数（预分配，避免动态内存分配）
T = tile_size          # GEMM tile 粒度（对齐 Comet-style kernel-RDMA pipeline）
H = d_model            # 隐藏层维度

物理内存地址：addr(CT[r, s, t, h]) = base + r*(S*T*H) + s*(T*H) + t*H + h
                                              ↑
                              Rank r 的所有 token 在连续内存中
                              → dispatch CT[r] 只需一次 RDMA send
```

### 2. CommTensor 元数据（零开销路由）

每个 CommTensor 携带紧凑的路由元数据：

```python
class CommTensorMeta:
    token_indices: Tensor  # [R, S] int32 - 原始 token 位置（sequence 中的下标）
    routing_scores: Tensor # [R, S] fp16  - 路由权重（softmax 后的概率）
    slot_counts: Tensor    # [R]    int32 - 每个 rank 实际使用的槽位数（可能 < S）
    layout_version: int    # 布局版本（支持动态 re-layout，LAER-MoE 风格）
    is_fsep_sharded: bool  # 是否已按 FSEP 分片
    tile_alignment: int    # tile 对齐大小（字节）
```

路由权重直接嵌入 meta，expert GEMM 完成后不需要额外的 gather-and-weight 操作。

### 3. 支持的操作（均为零拷贝或最小拷贝）

| 操作 | 传统实现 | CommTensor 实现 | 节省 |
|------|---------|----------------|------|
| Dispatch | permute + pack + A2A | 直接 RDMA send CT[r] | ~2× 内存带宽 |
| Combine | A2A recv + unpack + weighted sum | RDMA recv + index-scatter | ~2× 内存带宽 |
| Expert GEMM | 全量后 route | tile GEMM → tile RDMA (Comet) | 90%+ overlap |
| FSEP RS | gather 后 RS | 直接在 CT 上 RS | 避免 gather |
| 路由权重应用 | scatter_add（后） | meta 中的 scores（前） | 零额外内存 |

---

## 系统架构

```
MoEX 系统架构（通信为第一等公民）

┌─────────────────────────────────────────────────────────────────┐
│                        MoEX Runtime                              │
│                                                                   │
│  ┌─────────────────┐    ┌──────────────────────────────────────┐ │
│  │  CommTensor      │    │         OverlapScheduler             │ │
│  │  Storage Engine  │    │  (DAG-based, comm-priority first)    │ │
│  │                  │    │                                      │ │
│  │  [R,S,T,H] layout│    │  Tile 1 GEMM → RDMA → Tile 2 GEMM  │ │
│  │  + Meta (indices,│    │  (Comet-style, warp specialization)  │ │
│  │    scores,counts)│    │                                      │ │
│  └────────┬─────────┘    └──────────────┬───────────────────────┘ │
│           │                             │                          │
│  ┌────────▼─────────────────────────────▼───────────────────────┐ │
│  │                   MoEX Layer Components                       │ │
│  │                                                               │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐  │ │
│  │  │  Router  │  │ Dispatch │  │  Expert  │  │   Combine   │  │ │
│  │  │          │  │  (RDMA   │  │  (FSEP   │  │  (RDMA recv │  │ │
│  │  │ 路由结果  │  │  零拷贝)  │  │  tile    │  │   零拷贝)   │  │ │
│  │  │ 直写CT   │  │          │  │  GEMM)   │  │             │  │ │
│  │  └──────────┘  └──────────┘  └──────────┘  └─────────────┘  │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │              Heterogeneous Parallelism Manager               │  │
│  │                                                               │  │
│  │  Attention Side      │      MoE Side                         │  │
│  │  [TP+CP+DP groups]   │      [EP+EDP groups]                  │  │
│  │  Sequence-ordered     │      CommTensor (rank-ordered)        │  │
│  │  Standard tensor      │      FSEP-sharded                    │  │
│  │         ↕ Fold/Unfold (merged with dispatch A2A)              │  │
│  └─────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 数据流（正向传播）

```
输入: hidden_states [B, L, H]  （Attention 输出，sequence-ordered）
  │
  ▼
【Router】
  Gate GEMM: [B*L, num_experts] logits
  TopK → routing_indices [B*L, K], routing_scores [B*L, K]
  直接写入 CommTensor 元数据（zero-copy，无需额外 buffer）
  │
  ▼
【CommTensor 构建】（O(B*L*K) 轻量操作，无内存拷贝）
  CT shape: [R, S, T, H]
  CT.meta.token_indices[r, :] = {属于 rank r 的原始 token 下标}
  CT.meta.routing_scores[r, :] = {对应路由权重}
  CT.meta.slot_counts[r] = {rank r 的实际 token 数}
  CT.data[r, :, :, :] = hidden_states 的视图（视图！不拷贝）
  │
  ▼
【Dispatch】（RDMA 直发，零拷贝）
  for rank r in EP_group:
      RDMA_send(CT.data[r, :CT.meta.slot_counts[r], :, :], dst=r)
  异步发送，立即返回（RDMA one-sided put）
  │
  ▼
【Expert GEMM + tile-level RDMA】（Comet 风格）
  接收到 CT_remote（来自其他 rank 的 CommTensor 片段）
  for tile t in range(num_tiles):
      result_tile = GEMM(CT_remote[:, :, t, :], W_expert[:, t*T:(t+1)*T])
      RDMA_send(result_tile, dst=original_rank)  # 即完即发
  │
  ▼
【Combine】（RDMA 直收 + 加权求和）
  RDMA_recv → 写入 CT_result（已按 token_indices 对齐）
  output[CT.meta.token_indices] += CT_result * CT.meta.routing_scores
  │
  ▼
输出: hidden_states [B, L, H]  （还原 sequence-ordered）
```

---

## 与现有系统的对比

### 内存拷贝次数对比

| 操作 | 传统 Megatron | Axion/DeepEP | MoEX CommTensor |
|------|-------------|--------------|-----------------|
| Dispatch 前 permute | 1次 O(T×H) | 优化（token 粒度）| **0次**（视图）|
| Dispatch send buffer | 1次 O(T×H) | 1次 | **0次**（直发）|
| Combine recv buffer | 1次 O(T×H) | 1次 | **0次**（直收）|
| Combine unpermute | 1次 O(T×H) | 优化 | **0次**（index映射）|
| **总拷贝次数** | **4次** | **~2次** | **0次** |

### 通信 overlap 质量对比

| 系统 | Overlap 粒度 | 理论最大 Overlap | 实测 Overlap |
|------|------------|----------------|-------------|
| MegatronCore | Tensor 级（~100MB）| 70% | ~60-70% |
| FlowMoE | Chunk 级（~256MB）| 80% | ~68% |
| Comet | Tile 级（~KB）| 95% | ~90% |
| **MoEX** | **Tile 级 + 零拷贝前置** | **~98%** | **目标 92%+** |

MoEX 的优势：CommTensor 消除了 dispatch 前的准备时间，让 RDMA 能更早开始，
给 overlap 留出更多窗口。

### 异构并行支持对比

| 能力 | Megatron EP | FSEP (LAER-MoE) | MoEX |
|------|------------|-----------------|------|
| Expert 动态 re-layout | ❌ | ✅ | ✅ |
| Attention/MoE 解耦并行 | ✅（Folding）| ❌ | ✅ |
| Tile-level GEMM-RDMA | ❌ | ❌ | ✅ |
| CPU+GPU 异构（Inference）| ❌ | ❌ | ✅（规划中）|
| Tensor layout 版本管理 | ❌ | ❌ | ✅ |

---

## 论文支撑

MoEX 的每一个设计决策都有对应的学术支撑：

| 创新点 | 来源论文 | 关键结论 |
|--------|---------|---------|
| CommTensor [R,S,T,H] 布局 | 原创（综合以下论文） | 消除 4 次内存拷贝 |
| Tile 对齐维度 T | Comet (MLSys'25) | 90%+ GEMM-RDMA overlap |
| Rank 优先维度 R | LAER-MoE (ASPLOS'26) + FSEP | ReduceScatter 零拷贝 |
| 路由权重嵌入 meta | MegatronCore (Megatron-Core) | 省 26.3 GB/GPU 显存 |
| 异构并行 Fold/Unfold | MoE Parallel Folding (arXiv'25) | 49.3% MFU on 1024 GPUs |
| DAG 跨层调度 | FlowMoE (NeurIPS'25) | 57% 训练时间减少 |
| Warp 专化（Comm/Compute）| Comet (MLSys'25) | 90%+ overlap |
| FSEP ReduceScatter | LAER-MoE (ASPLOS'26) | 1.69× 端到端 |
| 动态 re-layout | LAER-MoE (ASPLOS'26) | 消除负载不均衡 |

---

## 设计文档

| 文档 | 内容 |
|------|------|
| [`design/CommTensor_Design.md`](design/CommTensor_Design.md) | CommTensor 详细设计：布局、元数据、操作语义 |
| [`design/MoEX_Architecture.md`](design/MoEX_Architecture.md) | 系统架构：组件、接口、数据流 |
| [`design/Overlap_Strategy.md`](design/Overlap_Strategy.md) | 三层 overlap 策略：Tile 级 / Block 级 / 跨层级 |
| [`design/Heterogeneous_Parallelism.md`](design/Heterogeneous_Parallelism.md) | 异构并行模型：5D 并行 + FSEP + Fold/Unfold |

## 代码原型

| 文件 | 内容 |
|------|------|
| [`code/comm_tensor.py`](code/comm_tensor.py) | CommTensor 核心实现（PyTorch 原型）|
| [`code/moex_layer.py`](code/moex_layer.py) | MoEX MoE Layer 完整实现 |
| [`code/overlap_scheduler.py`](code/overlap_scheduler.py) | Tile 级 Overlap 调度器 |

---

## 快速开始

```python
from moex.code.comm_tensor import CommTensor, CommTensorConfig
from moex.code.moex_layer import MoEXLayer

# 配置 CommTensor 布局
config = CommTensorConfig(
    num_ep_ranks=8,           # Expert Parallel degree
    max_slots_per_rank=512,   # 每 rank 最多 token 数（预分配）
    tile_size=64,             # GEMM tile size（对齐 H100 tensor core）
    d_model=4096,             # 隐藏层维度
    use_fsep=True,            # 启用 FSEP 分片
    dtype=torch.float16,
)

# 创建 MoEX layer
moe_layer = MoEXLayer(
    config=config,
    num_experts=64,
    expert_hidden_size=14336,
    top_k=2,
)

# 正向传播（CommTensor 在内部透明管理）
output = moe_layer(hidden_states)  # [B, L, H] → [B, L, H]
```

---

## 当前状态

| 组件 | 状态 | 说明 |
|------|------|------|
| CommTensor 设计文档 | ✅ 完成 | 核心布局、语义、API 设计 |
| 系统架构文档 | ✅ 完成 | 完整数据流和组件接口 |
| Overlap 策略文档 | ✅ 完成 | 三层 overlap 详细设计 |
| 异构并行文档 | ✅ 完成 | 5D + FSEP + Fold/Unfold |
| CommTensor 原型代码 | ✅ 完成 | PyTorch 原型，可运行 |
| MoEX Layer 原型 | ✅ 完成 | 端到端 MoE Layer 实现 |
| Overlap Scheduler 原型 | ✅ 完成 | 调度器框架实现 |
| CUDA kernel 实现 | 🔲 规划中 | 依赖原型验证 |
| 与 Megatron 集成 | 🔲 规划中 | 依赖 kernel 实现 |
| 性能基准测试 | 🔲 规划中 | 依赖集成完成 |
