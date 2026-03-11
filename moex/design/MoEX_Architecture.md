# MoEX 系统架构设计

> 本文档描述 MoEX 的完整系统架构，包括各组件的设计、接口和数据流。
> MoEX 以 CommTensor 为中心，围绕通信原生数据结构构建完整的 MoE 训练系统。

---

## 目录

1. [架构总览](#1-架构总览)
2. [核心组件设计](#2-核心组件设计)
3. [训练数据流（正向传播）](#3-训练数据流正向传播)
4. [训练数据流（反向传播）](#4-训练数据流反向传播)
5. [进程组与并行配置](#5-进程组与并行配置)
6. [与 Megatron-Core 的集成接口](#6-与-megatron-core-的集成接口)
7. [运行时系统设计](#7-运行时系统设计)
8. [容错与动态 Re-layout](#8-容错与动态-re-layout)
9. [配置系统](#9-配置系统)
10. [性能模型](#10-性能模型)

---

## 1. 架构总览

### 1.1 层次结构

```
MoEX 系统层次（从底层到上层）

Layer 0: 硬件感知层
  ├── 网络拓扑感知（IB vs NVLink vs XGMI 路径区分）
  ├── RDMA 原语封装（GDR, GPUDirect RDMA）
  └── CUDA/HIP stream 管理（Compute Stream + Comm Stream）

Layer 1: CommTensor 存储引擎
  ├── CommTensor [R, S, T, H] 四维布局
  ├── CommTensorMeta（路由元数据，零额外 buffer）
  ├── CommTensorPool（预分配池，消除 malloc 延迟）
  └── CommTensorConfig（配置管理）

Layer 2: MoEX 操作库
  ├── Router（路由，直写 CommTensor meta）
  ├── Dispatcher（零拷贝 RDMA dispatch）
  ├── ExpertEngine（FSEP-compatible tile-level GEMM）
  ├── Combiner（零拷贝 combine + weighted scatter_add）
  └── OverlapScheduler（tile 级 pipeline 调度）

Layer 3: MoEX Layer
  ├── MoEXLayer（完整 MoE layer 实现）
  ├── MoEXBlock（含 Attention，支持并行 Fold/Unfold）
  └── MoEXModel（多层模型，支持跨层 overlap）

Layer 4: 集成接口
  ├── Megatron-Core 适配器（替换 MegatronMoE）
  ├── 分布式训练 Driver
  └── 性能监控与 Profiling
```

### 1.2 核心设计原则

**原则 1：通信是第一等公民**
- CommTensor 的物理布局由通信模式决定，而非计算模式
- 所有 API 设计以"减少通信开销"为第一优先级
- 内存分配策略考虑 RDMA 对齐要求

**原则 2：零拷贝贯穿始终**
- Dispatch：CommTensor[r] 地址连续 → 直接 RDMA（零拷贝）
- Combine：scatter_add 直写输出 → 无中间 buffer
- GEMM：tile 完成即 RDMA → 绕过 HBM（理想情况）

**原则 3：Overlap 内生（不是外加）**
- CommTensor 的 T 维度天然对应 GEMM tile，overlap 是设计的一部分
- Router 直写 meta，消除 overlap 前的准备延迟
- 预分配池消除动态内存分配对 overlap 的打断

**原则 4：异构并行原生支持**
- CommTensor 携带 `layout_version` 支持 LAER-MoE 动态 re-layout
- FSEP 分片是 CommTensor 的内置属性（不是后处理）
- 进程组配置支持 Attention（TP+CP）与 MoE（EP+EDP）独立配置

---

## 2. 核心组件设计

### 2.1 Router（路由器）

**职责**：执行 Gate GEMM，生成路由决策，直写 CommTensor 元数据

```
Router 内部设计：

Gate GEMM：
  hidden_states [B*L, H] × W_gate [H, num_experts] → logits [B*L, num_experts]
  执行位置：Attention TP group 内（TP 并行）

TopK Selection：
  logits → TopK → (expert_ids [B*L, K], scores [B*L, K])
  执行位置：本地（无通信）

Load Balancing（对应 LAER-MoE 的 Load Planner 接口）：
  → 软约束：routing scores 偏置（引导均匀分布）
  → 硬约束：capacity factor（token 溢出时丢弃或随机分配）

CommTensor Meta 填充（路由直写）：
  for each token i, expert k:
    r = expert_ids[i,k] // experts_per_rank  # rank ID
    s = atomic_add(slot_cursors[r], 1)       # slot ID
    ct.meta.token_indices[r, s] = i
    ct.meta.routing_scores[r, s] = scores[i, k]
  ct.meta.slot_counts = slot_cursors
```

**接口**：
```python
class MoEXRouter:
    def forward(
        self,
        hidden_states: Tensor,        # [B*L, H]
        comm_tensor: CommTensor,      # 输出填充到此 CT（预分配）
    ) -> CommTensor:
        """
        执行 Gate GEMM + TopK + 填充 CommTensor meta
        同时设置 hidden_states 的引用（使 CT.data 视图有效）
        """
        ...
```

### 2.2 Dispatcher（分发器）

**职责**：基于 CommTensor 执行零拷贝 RDMA dispatch

```
Dispatcher 内部设计：

发送路径（本 GPU 发出）：
  for r in ep_ranks:
    if r == my_rank: continue
    count = ct.meta.slot_counts[r]
    if count == 0: continue
    # RDMA one-sided PUT
    rdma_put(
      local_addr = ct.data.ptr + r * stride_R,
      remote_rank = r,
      remote_addr = peer_comm_buffer[my_rank].ptr,
      size = count * T_tiles * H * element_size,
      stream = comm_stream,
    )
    # 同时发送 meta（小包，走 control plane 或 RDMA）
    send_meta(ct.meta.slot_counts[r], ct.meta.token_indices[r, :count], dst=r)

接收路径（处理来自其他 rank 的 tokens）：
  收到的数据直接落在 peer_comm_buffer 的相应位置
  → 无需 unpack，直接构建接收侧的 CommTensor 视图

本地路径（自己的 tokens）：
  local_ct = ct.view_rank(my_rank)  # 零拷贝切片
```

**RDMA 对齐要求**：
- CommTensor 分配时按 4KB 对齐（IB RDMA 最小对齐单位）
- 每个 rank 区域（`ct.data[r]`）的起始地址按 4KB 对齐
- 这通过 `CommTensor.allocate()` 保证（使用 `cudaMallocAligned`）

### 2.3 ExpertEngine（专家计算引擎）

**职责**：FSEP 兼容的 tile 级 Expert GEMM，即完即发

```
ExpertEngine 内部设计：

支持两种模式：

Mode 1: 传统 EP（每 GPU 持有完整专家）
  input_ct: CommTensor[R, S, T_tiles, H]（接收到的 tokens）
  W_expert: [H, 4H]（本地专家权重）
  
  for tile t in range(T_tiles):
    tile_input = input_ct[:, :, t, :]           # [R*S, H_per_tile]
    tile_output = GEMM(tile_input, W_expert[t]) # [R*S, 4H]
    → Comm Warp 触发 RDMA 发回原始 rank

Mode 2: FSEP（所有 GPU 持有所有专家的 1/R 份）
  input_ct: CommTensor[R, S, T_tiles, H]（所有 rank 都收到所有 token）
  W_shard:  [H, 4H/R]（本地专家权重分片）
  
  for tile t in range(T_tiles):
    tile_input = input_ct[:, :, t, :]
    partial_output[t] = GEMM(tile_input, W_shard[t])  # [R*S, 4H/R]
  → ReduceScatter（NVLink）→ 每 GPU 获得 S/R 个 token 的完整 output

GroupedGEMM 优化：
  多个专家的 GEMM 合并为一个 GroupedGEMM kernel
  利用 cuBLAS `cublasGemmGroupedBatched` 或 Triton 实现
  → 消除小 GEMM 的 kernel launch overhead
```

**关键决策：Weight 在哪里存储？**

```
传统 EP：
  W_expert_e 完整存在 rank_r（对应 expert e 所在的 GPU）
  → 每 GPU 内存：num_experts_per_rank × 2 × H × 4H × dtype_size

FSEP（LAER-MoE）：
  W_expert_e 的第 r 列存在 rank_r
  W_shape per GPU: num_experts_per_rank × H × (4H/R) × dtype_size
  → 更小的单 GPU 内存需求（适合更大的专家数）

MoEX 两种模式都支持，通过 ExpertEngine.mode 切换
```

### 2.4 Combiner（合并器）

**职责**：接收 Expert 输出，执行加权 scatter_add 到输出 tensor

```
Combiner 内部设计：

输入：
  expert_out_ct: CommTensor（从远端 rank RDMA 接收的 expert 输出）
  original_meta: CommTensorMeta（原始路由信息：token_indices, routing_scores）

操作（per rank）：
  for r in ep_ranks:
    count = original_meta.slot_counts[r]
    data = expert_out_ct.data[r, :count]           # [count, T_tiles, H]
    data_flat = data.view(count, H)                # [count, H]
    scores = original_meta.routing_scores[r, :count].unsqueeze(-1)  # [count, 1]
    indices = original_meta.token_indices[r, :count]  # [count]
    
    # 原地加权累加（无 buffer 拷贝）
    output.scatter_add_(0, indices.unsqueeze(-1).expand(-1, H), data_flat * scores)

输出：
  output: Tensor[B*L, H]（sequence-ordered，可直接输入下一层）
```

### 2.5 OverlapScheduler（重叠调度器）

**职责**：协调 Compute/Comm Warp，管理 tile-level pipeline

详细设计见 [`Overlap_Strategy.md`](Overlap_Strategy.md)。

---

## 3. 训练数据流（正向传播）

### 3.1 单层完整流程

```
输入: hidden_states [B*L, H]  (Attention 输出，TP All-Reduce 已完成)

┌─────────────────────────────────────────────────────────────────────────┐
│                         MoEX Layer Forward                               │
│                                                                           │
│  Step 1: Router（~5% layer time）                                         │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │ Gate GEMM → TopK → 直写 CommTensor Meta（token_indices, scores）│     │
│  │ CommTensor.data[r, s] = hidden_states[token_indices[r,s]]       │     │
│  │（1次 index_select，之后全程零拷贝）                               │     │
│  └─────────────────────────────────────────────────────────────────┘     │
│          │                                                                 │
│          ▼                                                                 │
│  Step 2: Dispatch（异步，~25-30% layer time，但与 GEMM overlap）           │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │ RDMA PUT ct.data[r] → peer_rank_r（comm_stream，非阻塞）         │     │
│  │ 同时：comm_stream 等待 compute_stream 完成 Step 1               │     │
│  └─────────────────────────────────────────────────────────────────┘     │
│          │                                                                 │
│          ▼（与 Dispatch 并行）                                              │
│  Step 3: Expert GEMM + Tile RDMA（~40-50% layer time）                     │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │ 接收 peer tokens（RDMA 完成）                                    │     │
│  │ for tile t:                                                      │     │
│  │   Compute Warp: GEMM(input_tile_t, W_expert) → output_tile_t   │     │
│  │   Comm Warp:    RDMA PUT output_tile_t → original_rank（即完即发）│    │
│  └─────────────────────────────────────────────────────────────────┘     │
│          │                                                                 │
│          ▼                                                                 │
│  Step 4: Combine（~15-20% layer time）                                     │
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │ RDMA RECV expert outputs → expert_out_ct                        │     │
│  │ scatter_add(output, expert_out_ct * scores, token_indices)       │     │
│  │（原地，零中间 buffer）                                            │     │
│  └─────────────────────────────────────────────────────────────────┘     │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘

输出: hidden_states [B*L, H]  （sequence-ordered，Layer Norm 输入）
```

### 3.2 跨层 overlap 流程（FlowMoE 风格）

```
Layer N 时间线：

时刻  t=0    t=1    t=2    t=3    t=4    t=5    t=6
      │      │      │      │      │      │      │
      ▼      ▼      ▼      ▼      ▼      ▼      ▼
 Router│ D-A2A│       Expert GEMM（tile 0,1,...,N）      │Combine
      └──┬───┘       └─────────────────────────┘
         │ Dispatch 异步，已完成，等待 Expert 开始
         └──────────────────────────────────────┘

Layer N+1 时间线（与 Layer N 交错）：

时刻  t=0    t=1    t=2    t=3    t=4    t=5    t=6    t=7
                                  │      │      │      │
                                  ▼      ▼      ▼      ▼
                             Router│ D-A2A│    Expert   │ Combine

关键：Layer N+1 的 Dispatch（D-A2A）在 Layer N 的 Expert GEMM 进行时开始！
→ 实现跨层 communication 隐藏
```

### 3.3 CUDA Stream 分配

```
Stream 架构：

compute_stream_0：
  - Router（Gate GEMM）
  - Expert GEMM（所有 GEMM tile 计算）
  - Combine 的 scatter_add

compute_stream_1（前序 micro-batch）：
  - Attention Forward
  - Layer Norm

comm_stream_dispatch：
  - All-to-All / RDMA PUT（dispatch 方向）
  - 等待 compute_stream_0 完成 index_select

comm_stream_combine：
  - All-to-All / RDMA GET（combine 方向）
  - 触发 scatter_add 到 compute_stream_0

同步点（最小化）：
  compute_stream_0.record_event(E1)  # index_select 完成
  comm_stream_dispatch.wait_event(E1)  # dispatch 等待数据就绪

  comm_stream_dispatch.record_event(E2)  # dispatch 完成（对端确认）
  compute_stream_0.wait_event(E2)  # Expert GEMM 等待数据到达
```

---

## 4. 训练数据流（反向传播）

### 4.1 反向传播中的 CommTensor

```
正向：hidden_states → CommTensor → Expert → hidden_states
反向：grad_output → CommTensor_grad → Expert_bwd → grad_input

CommTensor 的反向传播：
  CommTensor.from_hidden_states 的反向：
    → 相当于反向方向的 dispatch：grad_output 需要发给 Expert 所在的 rank
    → 同样可以直接构建 grad CommTensor（rank-ordered）
    → 零拷贝 RDMA（相同机制）
```

### 4.2 W/D 解耦（来自 MegatronCore）

```
Expert GEMM 的梯度计算：
  dW = input^T × grad_output  （权重梯度，不依赖通信）
  dX = grad_output × W^T      （数据梯度，依赖 combine 通信）

MoEX 反向传播 overlap 策略：
  ┌────────────────────────────────────────────────────────────────┐
  │ Compute Stream: dX = grad_output × W^T（数据梯度）             │
  │ Comm Stream:    All-Reduce(dX)（返回给 dispatch 来源）         │
  │ Compute Stream: dW = input^T × grad_output（权重梯度，并行）  │
  └────────────────────────────────────────────────────────────────┘

关键：dW 计算不需要通信（本地专家参数），可以完全 overlap 通信！
→ 与 MegatronCore 的 W/D splitting 相同策略，但在 CommTensor 语义下实现
```

### 4.3 梯度 CommTensor

```python
class CommTensor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, routing_indices, routing_scores, config):
        ct = CommTensor.from_hidden_states(hidden_states, routing_indices, routing_scores, config)
        ctx.save_for_backward(routing_indices, routing_scores)
        ctx.config = config
        return ct.data, ct.meta

    @staticmethod
    def backward(ctx, grad_ct_data, grad_ct_meta):
        routing_indices, routing_scores = ctx.saved_tensors
        config = ctx.config

        # 梯度 CommTensor：反向方向的 dispatch
        # grad_ct_data: [R, S, T_tiles, H]（与正向相同布局）
        # 需要：对每个 slot (r, s)，grad 加权累加回 token_indices[r, s] 位置

        grad_hidden = torch.zeros(B_L, H, ...)
        # scatter_add（combine 的反向）
        # 注意：routing_scores 的梯度也需要计算（但通常很小，可近似忽略）
        ...
        return grad_hidden, None, None, None
```

---

## 5. 进程组与并行配置

### 5.1 进程组层次

```
MoEX 进程组设计（融合 MoE Parallel Folding 与 FSEP）：

全局进程组（World）：
  ├── Pipeline Parallel (PP) groups：跨节点，层级划分
  └── 单 PP Stage 内：
      ├── Attention 进程组
      │   ├── TP group：同节点内，QKV 分割
      │   ├── CP group：跨节点，长序列环形通信
      │   └── DP group：数据并行
      └── MoE 进程组（可与 Attention 独立！）
          ├── EP group：Expert Parallel（跨节点）
          ├── EDP group：Expert Data Parallel（副本）
          └── FSEP group：节点内 NVLink 组（ReduceScatter）
```

### 5.2 并行度配置示例

```
示例：1024 GPU，DeepSeek-V3 规模

PP = 16（Pipeline Parallel）：16 个 pipeline 阶段
每阶段 64 GPU：

Attention 层：
  TP = 8（Tensor Parallel，节点内）
  CP = 4（Context Parallel，长序列）
  DP = 2（Data Parallel）
  → 8 × 4 × 2 = 64 ✓

MoE 层（解耦！）：
  EP = 32（Expert Parallel，跨节点）
  EDP = 2（Expert Data Parallel）
  → 32 × 2 = 64 ✓

FSEP 子组（节点内，8 GPU/node）：
  intra_node_group：每 8 GPU 一组（用于 ReduceScatter on NVLink）

优势：
  Attention：高 TP（8）适合大矩阵，高 CP（4）支持长序列
  MoE：高 EP（32）均摊通信，EDP（2）提升吞吐
  → 49.3% MFU（MoE Parallel Folding 论文的实测值）
```

### 5.3 进程组管理 API

```python
class MoEXProcessGroups:
    """MoEX 进程组管理器，支持 Attention/MoE 独立并行配置"""

    # Attention 侧进程组
    tp_group: ProcessGroup          # Tensor Parallel
    cp_group: ProcessGroup          # Context Parallel
    dp_group: ProcessGroup          # Data Parallel
    pp_group: ProcessGroup          # Pipeline Parallel

    # MoE 侧进程组（与 Attention 独立！）
    ep_group: ProcessGroup          # Expert Parallel（主通信组）
    edp_group: ProcessGroup         # Expert Data Parallel
    fsep_group: ProcessGroup        # FSEP 节点内组（NVLink）

    @classmethod
    def create(
        cls,
        pp: int, tp: int, cp: int, dp: int,
        ep: int, edp: int,
        use_fsep: bool = True,
    ) -> 'MoEXProcessGroups': ...

    def get_ep_rank(self) -> int:
        """本 GPU 在 EP group 中的 rank（CommTensor 的 R 维对应此）"""
        return dist.get_rank(self.ep_group)

    def get_intra_node_group(self) -> ProcessGroup:
        """节点内 NVLink 通信组（用于 FSEP ReduceScatter）"""
        return self.fsep_group
```

---

## 6. 与 Megatron-Core 的集成接口

### 6.1 集成策略

MoEX 设计为 Megatron-Core MoE 模块的**直接替换**（drop-in replacement）：

```python
# 原始 Megatron-Core
from megatron.core.transformer.moe import MoELayer

# 替换为 MoEX
from moex import MoEXLayer

# 接口兼容
class MoEXLayer(torch.nn.Module):
    def __init__(self, config: TransformerConfig, submodules: ...):
        # 内部使用 CommTensor，对外行为与 MegatronMoELayer 相同
        ...

    def forward(self, hidden_states: Tensor, ...) -> Tensor:
        # 输入/输出接口与 Megatron 相同
        # 内部全程 CommTensor
        ...
```

### 6.2 关键替换点

```
Megatron-Core MoE 数据流（当前）：
  hidden_states → MoETokenDispatcher.dispatch()
              → [All-to-All]
              → GroupedMLP.forward()
              → [All-to-All]
              → MoETokenDispatcher.restore()
              → weighted_output

MoEX 替换后：
  hidden_states → MoEXRouter.forward()     # 填充 CommTensor meta
              → [CommTensor.dispatch_async()]  # 零拷贝 RDMA
              → [ExpertEngine.forward()]    # tile-level GEMM + 即发 RDMA
              → [Combiner.combine()]        # scatter_add
              → output
```

### 6.3 配置映射

```python
# Megatron-Core TransformerConfig → MoEXConfig
def megatron_config_to_moex(cfg: TransformerConfig) -> MoEXConfig:
    return MoEXConfig(
        num_experts=cfg.num_moe_experts,
        top_k=cfg.moe_router_topk,
        d_model=cfg.hidden_size,
        expert_hidden_size=cfg.ffn_hidden_size,
        ep_size=cfg.expert_model_parallel_size,
        use_fsep=True,          # MoEX 默认启用 FSEP
        tile_size=128,           # H100 最优 tile size
        safety_factor=2.0,
        # 并行配置
        tp_size=cfg.tensor_model_parallel_size,
        pp_size=cfg.pipeline_model_parallel_size,
        cp_size=cfg.context_parallel_size,
    )
```

---

## 7. 运行时系统设计

### 7.1 运行时组件

```
MoEX Runtime:

┌──────────────────────────────────────────────────────────────┐
│  CommTensorPool（每个 MoEX Layer 一个）                       │
│  ├── pool_size = 4（双倍缓冲 × 2，for forward + backward）    │
│  └── 自动管理：acquire / release / zero-copy 重置             │
├──────────────────────────────────────────────────────────────┤
│  StreamManager                                                │
│  ├── compute_stream_0：Expert GEMM + scatter_add              │
│  ├── compute_stream_1：Attention（与 MoE 并行）               │
│  ├── comm_stream_dispatch：A2A / RDMA dispatch                │
│  └── comm_stream_combine：A2A / RDMA combine                  │
├──────────────────────────────────────────────────────────────┤
│  OverlapScheduler（跨 tile 调度）                             │
│  ├── Tile DAG：tile 级依赖关系（GEMM done → RDMA send）        │
│  ├── Warp Specialization：Compute/Comm warp 分配              │
│  └── Priority Queue：通信优先（comm latency > compute）       │
├──────────────────────────────────────────────────────────────┤
│  LayoutManager（LAER-MoE re-layout 接口）                     │
│  ├── layout_version 单调递增                                   │
│  ├── 异步迁移：backward 期间执行，forward 前完成              │
│  └── 双缓冲：迁移期间保持旧 layout 可用                       │
└──────────────────────────────────────────────────────────────┘
```

### 7.2 MoEX Runtime 生命周期

```python
class MoEXRuntime:
    def __init__(self, config: MoEXConfig, process_groups: MoEXProcessGroups):
        self.pool = CommTensorPool(config)
        self.streams = StreamManager()
        self.scheduler = OverlapScheduler(config)
        self.layout_mgr = LayoutManager(config)

    def step_begin(self):
        """训练步开始：准备 CommTensor"""
        self.current_ct = self.pool.acquire()
        self.layout_mgr.apply_pending_layout(self.current_ct)

    def step_end(self):
        """训练步结束：释放 CommTensor，触发异步 re-layout"""
        self.pool.release(self.current_ct)
        if self.layout_mgr.should_relayout():
            self.layout_mgr.schedule_relayout_async(stream=self.streams.comm_stream_dispatch)
```

---

## 8. 容错与动态 Re-layout

### 8.1 动态 Expert Re-layout（LAER-MoE 接口）

```
Re-layout 场景：
  训练过程中，某些专家持续过载（token 数 >> 均值），
  LAER-MoE 的 Load Planner 决定迁移专家到负载较轻的 GPU

MoEX 的支持：
  1. CommTensor.meta.layout_version 记录当前 expert 布局版本
  2. 迁移时：
     - 源 GPU：将 W_expert 的参数通过 RDMA 发到目标 GPU
     - 目标 GPU：更新本地 expert 参数表
     - 所有 GPU：layout_version++，CommTensor 中的 rank → expert 映射更新
  3. CommTensor.dispatch_async() 使用最新 layout_version 计算目标 rank

无缝切换：
  Step T 的 backward：异步执行迁移（使用空闲 comm_stream）
  Step T+1 的 forward：使用新 layout_version
  无需暂停训练！
```

### 8.2 CommTensor 的 layout_version 机制

```python
class CommTensorMeta:
    layout_version: int  # 当前布局版本
    expert_placement: ExpertPlacement  # expert_id → rank 映射

class ExpertPlacement:
    """Expert 到 GPU rank 的映射，支持动态更新"""
    mapping: Dict[int, int]   # {expert_id: ep_rank}

    def get_rank(self, expert_id: int) -> int:
        """获取 expert 所在的 EP rank"""
        return self.mapping[expert_id]

    def update(self, new_mapping: Dict[int, int], new_version: int):
        """原子更新（使用 Python GIL 或 spinlock）"""
        self.mapping = new_mapping
        self.layout_version = new_version
```

---

## 9. 配置系统

### 9.1 MoEXConfig 完整参数

```python
@dataclass
class MoEXConfig:
    # 模型架构
    num_experts: int                    # 专家总数
    top_k: int = 2                      # TopK 路由
    d_model: int = 4096                 # 隐藏层维度
    expert_hidden_size: int = 14336     # 专家 FFN 中间层维度
    num_layers: int = 1                 # MoE 层数

    # 并行配置
    ep_size: int = 8                    # Expert Parallel 度
    edp_size: int = 1                   # Expert Data Parallel 度
    tp_size: int = 1                    # Tensor Parallel 度（Attention）
    pp_size: int = 1                    # Pipeline Parallel 度
    cp_size: int = 1                    # Context Parallel 度

    # CommTensor 配置
    tile_size: int = 128                # GEMM tile 大小（元素数）
    max_tokens_per_step: int = 4096     # 每 step 最大 token 数
    safety_factor: float = 2.0          # Slot 预分配安全系数
    dtype: torch.dtype = torch.float16  # 数据类型

    # FSEP 配置
    use_fsep: bool = True               # 是否启用 FSEP
    fsep_intra_node_size: int = 8       # 节点内 GPU 数（用于 FSEP NVLink 组）

    # Overlap 配置
    overlap_dispatch: bool = True       # 是否 overlap dispatch
    overlap_combine: bool = True        # 是否 overlap combine
    tile_rdma_immediate: bool = True    # tile GEMM 后是否立即 RDMA（Comet 风格）
    warp_comm_ratio: float = 0.2        # Comm Warp 占比

    # Re-layout 配置（LAER-MoE）
    use_dynamic_relayout: bool = True   # 是否启用动态 re-layout
    relayout_interval: int = 100        # re-layout 检查间隔（step 数）
    relayout_threshold: float = 1.5     # 触发 re-layout 的负载不均衡阈值

    # 性能监控
    enable_profiling: bool = False      # 是否启用 profiling
    profile_interval: int = 50          # profiling 间隔

    @property
    def max_slots_per_rank(self) -> int:
        """每 EP rank 预分配的 slot 数"""
        base = self.max_tokens_per_step * self.top_k // self.ep_size
        return int(base * self.safety_factor)

    @property
    def num_tiles(self) -> int:
        """每个 token 的 tile 数（按 hidden dim 分割）"""
        return (self.d_model + self.tile_size - 1) // self.tile_size

    @property
    def experts_per_rank(self) -> int:
        """每个 EP rank 持有的专家数"""
        return self.num_experts // self.ep_size
```

---

## 10. 性能模型

### 10.1 理论分析

```
MoEX 性能模型（单 MoE layer，忽略 overlap）：

T_layer = T_route + T_dispatch + T_gemm + T_combine

T_route：
  = T_gate_gemm + T_topk
  = (B*L*H*num_experts) / TFLOPS + O(B*L*K*log(num_experts))
  ≈ 5% of layer time（典型）

T_dispatch（无 overlap）：
  = T_rdma_put = (B*L*K/ep_size) * H * dtype_size / IB_BW
  ≈ (4096*2/32) * 4096 * 2 / (50 GB/s) = 1024 * 8192 / 50e9 ≈ 0.17ms

T_gemm：
  = (B*L*K/ep_size) * H * expert_hidden_size * 2 / TFLOPS
  ≈ (4096*2/32) * 4096 * 14336 * 2 / (312 TFLOPS) ≈ 0.24ms

T_combine（无 overlap）：
  ≈ T_dispatch（对称）≈ 0.17ms

总计（无 overlap）：≈ 0.05 + 0.17 + 0.24 + 0.17 = 0.63ms

MoEX 的 overlap 效果（目标）：
  T_dispatch_visible ≈ 0.17ms × (1 - 0.92) ≈ 0.013ms  （92% overlap）
  T_combine_visible ≈ 0.017ms
  T_layer_with_overlap ≈ 0.05 + 0.013 + 0.24 + 0.017 = 0.32ms

加速比：0.63 / 0.32 ≈ 1.97×（接近 Comet 的 2.3× MoE layer 加速）
```

### 10.2 内存分析

```
CommTensor 内存（R=32, S=512, T_tiles=32, H=4096, fp16）：
  data:  32 * 512 * 32 * 4096 * 2 = 4 GB（较大！）

优化：
  1. 不需要同时存储所有 rank 的数据
     → 流水线发送：发完 rank r 的数据后立即复用该 slot
  2. 实际有效 slot 数 << S（load balance 下）
     → 仅分配 slot_counts[r] 的实际 data

优化后内存：
  实际数据 = sum(slot_counts[r]) * T_tiles * H * 2
           = B*L*K * T_tiles * H * 2
           = 4096 * 2 * 32 * 4096 * 2 = 2 GB

与传统实现对比：
  传统 send_buffer + recv_buffer ≈ 2 * B*L*K * H * 2 = 256 MB
  CommTensor 额外开销：2 GB - 256 MB = ~1.7 GB（T_tiles 维度）

权衡：
  CommTensor 额外内存换取：
    - 零拷贝 dispatch（节省 ~58MB 内存带宽 × 4 次 = 232MB/step）
    - tile-level overlap（提升 92% overlap 率）
  在 H100 80GB 上：1.7 GB 额外内存完全可接受
```

### 10.3 扩展性分析

```
MoEX 在不同规模下的通信开销：

EP=8（8 GPU，1 节点）：
  A2A 走 NVLink：T_dispatch ≈ 0.03ms（900 GB/s）
  → 通信几乎可以忽略

EP=32（32 GPU，4 节点）：
  A2A 走 IB：T_dispatch ≈ 0.17ms
  FSEP ReduceScatter 走 NVLink：T_rs ≈ 0.02ms
  → 总通信 ≈ 0.19ms，92% overlap 后 ≈ 0.015ms visible

EP=256（256 GPU，32 节点，DeepSeek-V3 规模）：
  A2A 走 IB（更大延迟）：T_dispatch ≈ 0.8-1.5ms
  FSEP ReduceScatter on NVLink（不变）：T_rs ≈ 0.02ms
  → 关键：CommTensor 零拷贝使 RDMA 更早开始（额外 ~0.1ms overlap 窗口）
  → 92% overlap：0.8ms × 0.08 = 0.064ms visible
  → 相比传统：0.8ms × 0.4 = 0.32ms visible
  → 节省：0.32 - 0.064 = 0.256ms per MoE layer（~40% 通信时间节省）
```
