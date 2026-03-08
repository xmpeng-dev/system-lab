# Axion Architecture Design: Communication-First Sparse Training Runtime

> **设计讨论文档** | 版本 v0.1 | 2026-03-08  
> **定位:** A communication-first sparse training runtime designed for large-scale MoE systems.  
> **理念来源:** LAER-MoE (FSEP) + veScale-FSDP + 自定义 IR 优先原则

---

## 0. 核心设计理念

```
三个核心支柱：

  Compile-time 感知          通信友好 Tensor 数据结构        FSEP MoE 并行
        ↓                              ↓                           ↓
  静态分析通信模式              消除 pack/unpack 开销          动态负载均衡
  编译期决定调度方案             物理布局 = 通信友好布局         Expert 按负载迁移
```

**最核心的判断：通信不是计算的"副作用"，而是与计算同等地位的一等公民。**

---

## 1. 为什么不用 FX Graph 补丁方案

### 1.1 FX Graph 的设计目标与 Axion 不匹配

```
torch.compile / FX Graph 的设计目标：
  捕获 PyTorch eager 语义 → 做局部优化（算子融合、内存规划）

  FX Graph IR 节点是：
    call_function(torch.mm, args=(...))
    call_function(torch.add, args=(...))
    ...

  通信是"外挂"进去的：
    call_function(dist.all_reduce, ...)   ← 在算子图里是个黑盒
    call_function(dist.all_to_all, ...)   ← 编译器看不穿它
```

### 1.2 根本矛盾

| 维度 | FX Graph 的粒度 | Axion 需要的粒度 |
|------|----------------|----------------|
| **语义单元** | 单个张量算子 | 通信+计算的联合调度单元 |
| **通信建模** | 黑盒函数调用 | 一等公民，类型系统原生支持 |
| **Expert 状态** | 运行时元数据，IR 不可见 | IR 内置类型 `ExpertShard` |
| **Overlap 分析** | 启发式猜测 | 编译器 Pass 精确推导 |
| **扩展性** | 打新补丁 | 添加新 Op 到指令集 |

### 1.3 结论

> **FX Graph 是为单机算子优化设计的，它的类型系统里根本没有"通信"这个概念。**  
> Axion 要做的是通信和计算的联合优化，需要一套从第一天起就把通信建模进类型系统的 IR。  
> 补丁式方案迟早会遇到天花板，没有捷径。

---

## 2. Axion 系统整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     用户代码 (Python DSL / PyTorch-like API)     │
├─────────────────────────────────────────────────────────────────┤
│  Axion Compiler                                                 │
│  ├── Pass 1: Type Inference                                     │
│  ├── Pass 2: Communication Pattern Analysis                     │
│  ├── Pass 3: FSEP Sharding Plan                                 │
│  ├── Pass 4: Overlap Insertion                                  │
│  ├── Pass 5: CommTensor Layout Lowering                         │
│  └── Pass 6: Kernel Code Generation                             │
├──────────────────────┬──────────────────────────────────────────┤
│  CommTensor 运行时    │  FSEP MoE 并行运行时                      │
│  (通信友好数据结构)   │  (动态 Expert 分片 + 重布局)               │
│  ├── index map 管理  │  ├── Slow Planner (每 K step)             │
│  └── 零 copy A2A     │  └── Fast Router  (每 step)               │
├──────────────────────┴──────────────────────────────────────────┤
│  CommFabric（NVLink / IB RDMA 统一抽象）                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Axion IR 设计

### 3.1 IR 类型系统

```
Axion IR 的核心类型（Type System）：

┌─────────────────────────────────────────────────────┐
│  Value Types                                        │
│                                                     │
│  DenseTensor(shape, dtype, shard_spec)              │
│    └── 普通 dense 参数，携带分片信息                  │
│                                                     │
│  CommTensor(shape, dtype, layout, comm_spec)        │
│    └── 正在"途中"的 tensor，携带通信元数据            │
│        layout: BLOCKED_BY_DST                       │
│               BLOCKED_BY_EXPERT                     │
│               SPARSE_CSR                            │
│                                                     │
│  ExpertShard(expert_id, shard_rank, gpu_id)         │
│    └── FSEP 的 Expert 参数分片，是 IR 内置类型        │
│                                                     │
│  RoutingTable(token_to_expert, send_counts)         │
│    └── MoE 路由表，编译器可以静态分析其结构            │
└─────────────────────────────────────────────────────┘
```

### 3.2 IR 指令集（Op Set）

```
── Compute Ops ──────────────────────────────────────────────────
  Dense.MatMul    (a: DenseTensor, b: DenseTensor) → DenseTensor
  Dense.LayerNorm (x: DenseTensor)                 → DenseTensor
  Expert.FFN      (x: CommTensor, e: ExpertShard)  → CommTensor
  Expert.Gate     (x: DenseTensor)                 → RoutingTable

── Comm Ops ─────────────────────────────────────────────────────
  Comm.A2A        (x: CommTensor,                  → CommTensor
                   routing: RoutingTable,
                   recv_layout: Layout)

  Comm.AllGather  (x: ExpertShard,                 → DenseTensor
                   group: ExpertGroup)

  Comm.ReduceScatter(x: DenseTensor,               → DenseTensor
                   group: ProcessGroup,
                   hier: INTRA_NODE | INTER_NODE)

── Shard Ops ────────────────────────────────────────────────────
  Shard.Split     (x: DenseTensor,                 → ExpertShard[]
                   spec: ShardSpec)

  Shard.Migrate   (x: ExpertShard,                 → ExpertShard
                   dst_gpu: int)         ← Expert 迁移是 IR 一等公民!

── Schedule Ops（编译器生成，用户不直接写）──────────────────────
  Sched.Overlap   (compute: Op, comm: CommOp)      → Pipeline
  Sched.Prefetch  (x: ExpertShard, step_offset: int)
```

### 3.3 IR 示例：一个 MoE Layer

```
func @moe_layer(%hidden: DenseTensor<[B,S,H]>) -> DenseTensor<[B,S,H]> {

  # 1. Gate：生成路由表
  %routing = Expert.Gate(%hidden)
  # routing: RoutingTable，编译器可分析其稀疏性

  # 2. 把 hidden 转换为通信友好格式（零 copy）
  %hidden_comm = CommTensor.FromDense(
      %hidden,
      layout = BLOCKED_BY_DST,   # ← 编译期决定布局
      routing = %routing,
  )

  # 3. Expert Dispatch（A2A）
  %dispatched = Comm.A2A(
      %hidden_comm,
      routing  = %routing,
      recv_layout = BLOCKED_BY_EXPERT,
  )

  # 4. Expert FFN（编译器从这里推断 Sched.Overlap 机会）
  %expert_out = Expert.FFN(
      %dispatched,
      shards = @expert_shards,   # 引用 FSEP 分片表
  )

  # 5. Expert Combine（A2A 返回）
  %combined = Comm.A2A(
      %expert_out,
      routing = %routing,
      recv_layout = BLOCKED_BY_DST,
  )

  # 6. 转回 Dense
  %output = CommTensor.ToDense(%combined)

  return %output
}
```

---

## 4. CommTensor：通信友好数据结构

### 4.1 设计动机：消除 pack/unpack

```
传统 All-to-All 的开销来源（4步）：

  tokens[seq, hidden]         ← 按 token 顺序
      │  pack（重排内存）       ← O(seq_len × hidden_dim) copy
  packed[gpu0_tokens, gpu1_tokens, ...]
      │  All-to-All（网络传输）
  received[gpu0_tokens, gpu1_tokens, ...]
      │  unpack（恢复顺序）     ← O(seq_len × hidden_dim) copy
  result[seq, hidden]

CommTensor 路径（2步，消除 pack/unpack）：

  CommTensor（物理已经是 BLOCKED_BY_DST 布局）
      │  All-to-All（直接 DMA，无需重排）
  CommTensor（接收端，物理即正确布局）
      │  逻辑视图变换（index map，零拷贝）
  logical_view[seq, hidden]
```

### 4.2 三种物理布局

| 布局 | 物理内存排列 | 适用场景 | 优势 |
|------|------------|---------|------|
| `BLOCKED_BY_DST` | `[GPU0的tokens \| GPU1的tokens \| ...]` | Expert dispatch A2A | 直接 DMA，零 copy |
| `BLOCKED_BY_EXPERT` | `[Expert0的参数shard \| Expert1的参数shard \| ...]` | FSEP Expert 参数 All-Gather | shard 天然连续，gather 无需重排 |
| `SPARSE_CSR` | CSR 格式（value + col_idx + row_ptr） | 稀疏路由场景 | 跳过空 expert，减少通信量 |

### 4.3 CommTensor 数据结构

```python
@dataclass
class CommTensor:
    # 物理存储：按通信模式友好的方式排列
    _physical_data: torch.Tensor       # 实际 CUDA 内存
    _physical_layout: TensorLayout     # 描述物理排布方式

    # 通信元数据（编译期插入，运行期只读）
    comm_spec: CommSpec
    # CommSpec 包含：
    #   target_gpus:  List[int]     # 数据要去往哪些 GPU
    #   send_counts:  List[int]     # 发给每个 GPU 多少元素
    #   recv_counts:  List[int]     # 从每个 GPU 收多少元素（预估）
    #   element_dtype: torch.dtype
    #   layout_hint:  LayoutHint    # BLOCKED / INTERLEAVED / SPARSE

    # 逻辑视图（供计算 kernel 使用，零拷贝）
    @property
    def logical_view(self) -> torch.Tensor:
        return self._physical_data[self._logical_to_physical_idx]
```

### 4.4 FSEP 专用：FSEPExpertTensor

```python
class FSEPExpertTensor(CommTensor):
    # Expert 分片元数据
    expert_id:      int
    shard_rank:     int            # 这是 expert 参数的第几个 shard
    total_shards:   int            # expert 参数总共切成几份

    # 迁移状态（运行时更新）
    current_gpu:    int
    migration_state: MigrationState  # STABLE / MIGRATING / SHADOW

    # 影子 buffer（迁移期间使用，借鉴 LAER-MoE double buffer）
    shadow_buffer:  Optional[torch.Tensor] = None

    def gather_full_expert(self) -> torch.Tensor:
        """
        BLOCKED_BY_EXPERT 布局保证：shard 天然连续
        All-Gather 直接 DMA，无需重排
        """
        if self.total_shards == 1:
            return self._physical_data  # 无需通信
        return comm_fabric.all_gather(
            self._physical_data,
            group=self.expert_parallel_group,
            output_layout=BLOCKED_BY_EXPERT,
        )
```

---

## 5. 编译器 Pass 流水线

```
Axion IR（用户语义级）
    │
    ▼  Pass 1: Type Inference
    │  推断每个 Value 的精确类型（DenseTensor / CommTensor / ExpertShard）
    │  标注 CommTensor 的 layout
    │
    ▼  Pass 2: Communication Pattern Analysis
    │  识别所有 Comm.A2A / Comm.AllGather 的数据依赖
    │  构建 CommGraph（通信依赖有向图）
    │  → 输出：哪些 Comm Op 可以并发，哪些有依赖
    │
    ▼  Pass 3: FSEP Sharding Plan
    │  基于 CommGraph + 历史 profile 数据
    │  生成最优 ExpertShard 初始分配方案
    │  → 输出：ShardingPlan（expert_id → gpu_id 的映射）
    │  （贪心近似即可，运行时 Planner 做微调）
    │
    ▼  Pass 4: Overlap Insertion
    │  识别 Compute Op 和 Comm Op 的 overlap 机会
    │  插入 Sched.Overlap 和 Sched.Prefetch 节点
    │  → 输出：带调度信息的 IR
    │
    ▼  Pass 5: CommTensor Layout Lowering
    │  把 CommTensor 的逻辑布局转化为物理内存分配指令
    │  为每个 CommTensor 生成 index map
    │  → 输出：带内存分配的 IR
    │
    ▼  Pass 6: Kernel Code Generation
       生成 CUDA Kernel + NCCL/RDMA 通信调用
       绑定 CommTensor 物理地址
       → 输出：可执行二进制
```

### Pass 4 生成的静态调度示例

```
Layer N (MoE)，编译期生成的执行时间线：

  ─────────────────────────────────────────────────────→ time

  Compute:  [Attn FWD]  [Expert FFN chunk 0]  [Expert FFN chunk 1] ...
  Comm:       [AG Dense] [A2A chunk 1]  [A2A chunk 2] ...  [RS Dense]
                         ↑─────────────────────────────────↑
                         Expert A2A 与 Expert 计算完全 overlap

  关键：这个时间线在编译期就确定，运行时按 schedule 执行，无动态决策开销
```

---

## 6. FSEP 运行时：双层规划

借鉴 LAER-MoE Load-Adaptive Planner，扩展为双层协同规划：

```
Slow Planner（每 K=50 step 执行一次）
├── 输入：历史 token → expert 路由分布
├── 输出：Expert 重布局方案（哪个 Expert Shard 迁到哪个 GPU）
└── 目标：minimize max_gpu(compute_time + migration_comm_cost)

Fast Router（每 step 执行）
├── 输入：当前 batch 的 gate logits
├── 输出：动态偏置调整（软性引导 token 路由）
└── 目标：在当前 Expert 布局下，进一步均衡负载

协作方式：
  Slow Planner 改变物理布局（大调，有迁移开销）
  Fast Router 通过 routing bias 微调（小调，无迁移开销）
  → 两者叠加，持续保持负载均衡
```

**Fast Router 路由偏置公式：**

```
gate_logits_adjusted[i] = gate_logits[i] - α · load_penalty[i]

load_penalty[i] = (tokens_on_expert_i / avg_tokens) ^ β
                  × gpu_utilization[gpu_of_expert_i]

# α, β 是超参，控制均衡强度
# 与 LAER-MoE 不同：不仅看 expert 负载，还看 GPU 利用率
# 风险：需要严格验证不影响 MoE 收敛性（这是激进设计）
```

---

## 7. 与源论文的继承关系

| Axion 组件 | 继承自 LAER-MoE | 继承自 veScale-FSDP | Axion 的扩展 |
|-----------|---------------|-------------------|-------------|
| **CommTensor** | FSEP 专家分片思想 | RaggedShard 语义感知 | 统一 Dense+Expert 的通信感知数据结构 |
| **FSEP 运行时** | Load-Adaptive Planner | 结构感知规划算法 | 双层规划（Slow Planner + Fast Router）|
| **FlowEngine** | 细粒度 A2A chunk 调度 | Lazy AG + 分层 RS | 统一 CommFlow 抽象，编译期静态排布 |
| **Axion IR** | — | — | 通信作为一等公民的全新 IR 设计 |

---

## 8. 关键工程挑战

| 挑战 | 描述 | 建议解法 |
|------|------|---------|
| **CommTensor index map 维护** | Expert 迁移后 index map 需更新 | 存 GPU SRAM，迁移后原子更新 |
| **编译期通信分析准确性** | MoE 路由动态，编译期无法精确预测 token 分布 | 编译期分析"通信结构"（拓扑），运行时填"通信参数"（数量），两层分离 |
| **Fast Router 收敛影响** | routing bias 改变 gate 梯度，可能影响收敛 | 严格消融实验；LAER-MoE 是物理迁移不改路由，Fast Router 更激进需验证 |
| **FSEP 联合规划 NP-hard** | Expert+Dense 联合规划比纯 Dense 复杂得多 | 先贪心，Pass 3 用近似算法，不求最优 |
| **IR Parser 冷启动** | 从零写 IR 前端成本高 | Bootstrap：先从 FX Graph 有损转换，用于验证 IR 正确性，不作为终态 |

---

## 9. 实现路径

```
Phase 1（2 周）：IR 类型系统 + 指令集纸面设计
  核心问题：CommTensor 的 layout 枚举是否覆盖所有 MoE 通信模式？
  验证方式：手写几个典型 MoE layer 的 IR 表示，看是否自然表达

Phase 2（1 月）：IR Parser + Pass 1/2（Type Inference + Comm Analysis）
  方案 A：提供 Python DSL，用户直接写 Axion IR 风格代码        ← 终态
  方案 B：从 FX Graph 有损转换（bootstrap 手段）               ← 验证用
  目标：能正确表示一个完整 MoE layer，CommGraph 构建正确

Phase 3（2 月）：Pass 3/4（FSEP Sharding Plan + Overlap Insertion）
  Pass 3 先用贪心算法，跑通 FSEP 初始分配
  Pass 4 是最高价值 Pass，重点投入

Phase 4（2 月）：CommTensor 运行时 + FSEP Slow/Fast Planner
  CommTensor index map 机制
  FSEP double buffer 迁移执行器
  Fast Router routing bias（需配合收敛实验）

Phase 5（1 月）：Pass 5/6 + 端到端训练验证
  CommTensor Layout Lowering
  CUDA Kernel + NCCL 绑定
  端到端跑通一个 MoE 模型

→ 约 6 个月可以有第一个可训练的 prototype
```

---

## 10. 系统定位总结

```
┌────────────────────────────────────────────────────────────┐
│               Axion 在 MoE 训练优化栈中的位置               │
├────────────────────────────────────────────────────────────┤
│ 应用层：模型架构 + 优化器（Shampoo, Muon, Gemini 等）        │
├────────────────────────────────────────────────────────────┤
│ Axion 层：CommTensor + Axion IR + FSEP 运行时  ← 这里       │
│  （统一处理通信感知分片、调度、负载均衡）                     │
├────────────────────────────────────────────────────────────┤
│ 现有框架层：Megatron-LM / veScale / Hetu-Galvatron          │
├────────────────────────────────────────────────────────────┤
│ MoE 优化层：LAER-MoE (Expert Re-layout), MoEBlaze          │
├────────────────────────────────────────────────────────────┤
│ 通信底层：NCCL, DeepEP, RDMA                               │
└────────────────────────────────────────────────────────────┘

核心价值主张：
  把通信从"副作用"提升为"一等公民"
  CommTensor 让数据天生就是通信友好格式
  Axion IR 让编译器完全理解通信语义
  FSEP 运行时保证负载均衡
  三者形成闭环，而非三个独立补丁叠加
```

---

*架构讨论文档 v0.1 | 基于 LAER-MoE (arXiv:2602.11686) 和 veScale-FSDP (arXiv:2602.22437) 的设计推演*
