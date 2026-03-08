# Axion Architecture Design

> **Compile-First · Communication-Native · Deterministic · Scalable**  
> A sparse training runtime for large-scale MoE systems.  
> 版本 v0.3 | 2026-03-08

---

## 0. 设计理念：两个系统的融合

Axion 的设计融合了两套理念：

**来自 Axon 的工程哲学：**
> *"每一个优化决策都应该可以被检查、重现、测量。"*
> — Compile-First · Deterministic · Debuggable · Inspectable

**来自 LAER-MoE + veScale-FSDP 的系统洞察：**
> *"通信不是计算的副作用，而是与计算同等地位的一等公民。"*
> — Communication-Native · Load-Adaptive · Structure-Aware

**融合后的 Axion 设计原则：**

```
原则 1：Compile-First（来自 Axon）
  不在运行时做本可以在编译期做的决策。
  通信模式、Expert 分片方案、Overlap 调度 —— 全部在编译期静态确定。
  运行时只负责执行已经确定的计划，不做动态决策（除负载均衡微调外）。

原则 2：Communication-Native（来自通信优先理念）
  通信不是算子之间的"副作用"，而是类型系统的一等公民。
  CommTensor 携带物理布局语义，消除 pack/unpack 开销。
  IR 的 Op 集同等对待 Compute Op 和 Comm Op。

原则 3：Inspectable（来自 Axon）
  每个图变换有 SHA-256 hash 可追踪。
  每个 Pass 的前后 hash 记录在 Compile Report 中。
  DistributedExecutablePlan 是 immutable 的 —— 给定输入图必然产生相同的 Plan。

原则 4：Deterministic（来自 Axon）
  相同的 ModelSpec + 相同的数据 → 相同的执行顺序。
  Expert 迁移和 Overlap 调度在编译期静态生成，不依赖运行时随机决策。
  Fast Router 的 routing bias 是确定性公式，不是随机采样。

原则 5：Load-Adaptive（来自 LAER-MoE）
  Expert 物理位置跟着负载走，不固定。
  Slow Planner 每 K step 重新规划 Expert 分布（大调）。
  Fast Router 每 step 微调 routing bias（小调）。
  两者协同，持续消除木桶效应。

原则 6：Structure-Aware（来自 veScale-FSDP）
  参数分片粒度由计算语义决定，不强制按行均分。
  量化 block 边界、Kronecker 因子的 co-location 约束 —— 体现在分片计划中。
  RaggedShard：每个 GPU 的分片大小可以不同，但语义完整。
```

---

## 1. 系统全景

### 1.1 整体流水线

Axion 的核心流水线直接采用 Axon 的设计结构，在其中嵌入通信维度：

```
ModelSpec（用户声明）
    │
    ▼
ModelGraph（计算图 IR）
    │  typed, immutable, hashable
    │  OpNode + OpSpec + Wire
    │  Pure Python，不依赖 torch
    ▼
Pass Manager
    │  每个 Pass：immutable 变换，记录 hash before/after
    │
    ├── [Axon 设计] AnalysisPass       FLOPs / params / activation 估算
    ├── [Axon 设计] FusionPass         RMSNorm+QKV+RoPE，SwiGLU 融合
    │
    ├── [Axion 新增] CommInferencePass  标注每个 Op 的通信需求
    ├── [Axion 新增] FSEPShardingPass   生成 Expert 初始分片方案
    ├── [Axion 新增] OverlapInsertionPass 插入 Sched.Overlap 节点
    └── [Axion 新增] CommTensorLayoutPass 决定 CommTensor 物理布局
    │
    ▼
DistributedExecutablePlan
    │  immutable, hashable（继承 Axon ExecutablePlan 设计）
    │  PlannedKernel（compute）+ PlannedCommOp（comm）交织
    │  StaticSchedule：编译期确定的执行时间线
    ▼
DistributedModule（nn.Module）
    │  compute kernels（来自 PlannedKernel）
    │  CommTensor bindings（通信友好内存）
    │  ExpertShard registry（FSEP 分片表）
    ▼
Runtime
    │  执行 StaticSchedule（不做动态决策）
    │  FSEP Slow Planner（每 K step，调整 Expert 分布）
    │  FSEP Fast Router（每 step，routing bias 微调）
    └  CommFabric（NVLink / IB RDMA 统一抽象）
```

### 1.2 与 Axon 设计的对应关系

| Axon 设计概念 | Axion 对应设计 | 扩展点 |
|-------------|--------------|--------|
| `ModelGraph` | `ModelGraph`（同等设计） | 增加 `CommOpSpec`、`ExpertShardSpec` 节点类型 |
| `OpSpec` (frozen dataclass) | `OpSpec` + `CommOpSpec` | `CommOpSpec` 继承 frozen 约定，携带 `CommLayout` |
| `Pass` (immutable 变换) | 4 个 Axion Comm Pass | 同等遵守 hash before/after 记录约定 |
| `ExecutablePlan` (immutable) | `DistributedExecutablePlan` | 增加 `comm_steps`、`static_schedule`、`expert_registry` |
| `Compile Report` | 扩展 Compile Report | 增加通信 overlap 率、FSEP 分片统计 |
| `torch.compile` 集成 | 保持（compute 部分） | 通信 Op 不走 `torch.compile`，走 CommFabric |

---

## 2. ModelGraph：计算图 IR 设计

### 2.1 设计原则（来自 Axon）

```
ModelGraph 的四个不变量（invariants）：

  1. Immutable：图一旦构建，节点和边不可修改
     → 每次变换产生新图，旧图保留
     → 保证 Pass 的输入始终稳定

  2. Hashable：每个图有确定性的 SHA-256 hash
     → hash 基于图结构（节点类型 + 拓扑），不依赖内存地址
     → 相同结构的图 hash 相同，不同结构必然不同

  3. Pure Python：图构建不依赖 torch，不分配 GPU 内存
     → 可以在 CPU 上构建和分析整个图
     → 编译报告可以在没有 GPU 的机器上生成

  4. Typed：每条 Wire（边）有明确的类型
     → 类型信息在 Pass 中传播，编译器可静态检查
```

### 2.2 核心数据结构

```python
# ── Wire（图的边，携带类型）─────────────────────────────────────
@dataclass(frozen=True)
class Wire:
    """
    图中两个节点之间的边，携带数据类型信息。
    frozen=True 保证 hashable（Axon 设计约定）。
    """
    id:       str           # 唯一标识，格式："{node_id}:{output_idx}"
    type:     WireType      # 见下方 WireType 定义
    shape:    tuple[int | Symbol, ...]
    dtype:    ScalarDtype

class WireType(Enum):
    """
    Axion 扩展了 Axon 的 Wire 类型，加入通信和分片类型。
    """
    # Axon 原有类型
    DENSE         = "dense"         # 普通 dense tensor
    SCALAR        = "scalar"        # 标量（loss、logits 等）

    # Axion 新增类型（通信和分片感知）
    COMM          = "comm"          # 正在通信中的 tensor（CommTensor）
    EXPERT_SHARD  = "expert_shard"  # FSEP Expert 参数分片
    ROUTING_TABLE = "routing_table" # MoE 路由表
    PIPELINE      = "pipeline"      # Sched.Overlap 的输出（调度原语）

# ── OpNode（图的节点）────────────────────────────────────────────
@dataclass(frozen=True)
class OpNode:
    """
    图中的一个计算/通信节点。
    frozen=True 保证 hashable（Axon 设计约定）。
    """
    id:       str
    spec:     OpSpec          # 算子描述（frozen，决定 hash）
    inputs:   tuple[Wire, ...]
    outputs:  tuple[Wire, ...]

# ── ModelGraph（完整计算图）──────────────────────────────────────
@dataclass(frozen=True)
class ModelGraph:
    name:     str
    nodes:    tuple[OpNode, ...]
    inputs:   tuple[Wire, ...]   # 图的输入（token ids 等）
    outputs:  tuple[Wire, ...]   # 图的输出（logits 等）

    @property
    def hash(self) -> str:
        """SHA-256 hash，基于图结构，不依赖内存地址（Axon 设计）"""
        return sha256(self._canonical_repr()).hexdigest()

    @property
    def vocab_size(self) -> int:
        ...  # 从 nodes 中提取

    def summary(self) -> GraphSummary:
        """输出 FLOPs、params、通信量估算（对应 Axon 的 compile report）"""
        ...
```

### 2.3 OpSpec 类型层次

```python
# ── 基类（Axon 设计约定）─────────────────────────────────────────
@dataclass(frozen=True)
class OpSpec:
    """所有算子描述的基类。frozen=True 是硬约定，保证 hashable。"""
    op_type: str

# ── Compute Ops（Axon 设计，Axion 直接采用）─────────────────────
@dataclass(frozen=True)
class MatMulSpec(OpSpec):
    op_type:    str = "matmul"
    transpose_b: bool = False

@dataclass(frozen=True)
class AttentionSpec(OpSpec):
    op_type:     str = "attention"
    num_heads:   int = 0
    num_kv_heads: int = 0
    head_dim:    int = 0
    backend:     str = "sdpa"   # sdpa / flash / aiter / reference

@dataclass(frozen=True)
class RMSNormSpec(OpSpec):
    op_type: str = "rmsnorm"
    eps:     float = 1e-5

@dataclass(frozen=True)
class SwiGLUSpec(OpSpec):
    op_type:          str = "swiglu"
    intermediate_size: int = 0

# ── Expert Ops（Axion 新增）──────────────────────────────────────
@dataclass(frozen=True)
class ExpertGateSpec(OpSpec):
    op_type:             str = "expert_gate"
    num_experts:         int = 0
    topk:                int = 2
    router_type:         str = "top_k"
    aux_loss:            bool = True

@dataclass(frozen=True)
class ExpertFFNSpec(OpSpec):
    op_type:          str = "expert_ffn"
    expert_id:        int = 0          # -1 表示"批量 Expert FFN"
    ffn_intermediate: int = 0

# ── Comm Ops（Axion 新增，与 Compute Ops 同等地位）─────────────
@dataclass(frozen=True)
class CommSpec(OpSpec):
    """所有通信 Op 的基类。frozen=True 是硬约定。"""
    comm_type:  str = ""         # "a2a" / "all_gather" / "reduce_scatter" / "p2p"
    group_hint: str = ""         # "ep_group" / "dp_group" / "tp_group"
    async_:     bool = True

@dataclass(frozen=True)
class A2ASpec(CommSpec):
    comm_type:   str = "a2a"
    recv_layout: str = "blocked_by_src"  # 接收端期望的物理布局

@dataclass(frozen=True)
class AllGatherSpec(CommSpec):
    comm_type:     str = "all_gather"
    output_layout: str = "blocked_by_expert"

@dataclass(frozen=True)
class ReduceScatterSpec(CommSpec):
    comm_type: str = "reduce_scatter"
    hier:      str = "intra_first"   # "flat" | "intra_first"

# ── Shard Ops（Axion 新增，FSEP 专用）────────────────────────────
@dataclass(frozen=True)
class ShardMigrateSpec(OpSpec):
    op_type:   str = "shard_migrate"
    expert_id: int = 0
    src_gpu:   int = 0
    dst_gpu:   int = 0

# ── Schedule Ops（编译器生成，用户不直接写）──────────────────────
@dataclass(frozen=True)
class OverlapSpec(OpSpec):
    """Sched.Overlap 节点，由 OverlapInsertionPass 生成。"""
    op_type:         str = "sched_overlap"
    compute_node_id: str = ""
    comm_node_id:    str = ""
    chunk_size:      int | None = None
```

---

## 3. Pass 系统设计

### 3.1 Pass 设计原则（来自 Axon，全部继承）

```
Axon 的 Pass 设计约定，Axion 全部遵守：

  约定 1：Immutable 变换
    Pass.run(graph) → new_graph
    不修改输入 graph，每次返回新图
    → 旧图保留，方便 debug 和回滚

  约定 2：Hash 追踪
    每个 Pass 执行前后记录 graph.hash
    hash 不变 → 图结构未改变（Pass 是 no-op 或只改 metadata）
    hash 改变 → 图结构变化，对应 Compile Report 中的 "Changed: yes"

  约定 3：可组合
    Pass 之间通过 graph 传递信息，不通过全局状态
    → 任意顺序组合 Pass 是安全的（顺序由 PassManager 决定）

  约定 4：可独立运行
    每个 Pass 可以单独 run 来调试
    → 不依赖其他 Pass 的副作用
```

### 3.2 Pass 基类

```python
class Pass:
    """
    所有 Pass 的基类，遵循 Axon 的 immutable 变换约定。
    """
    name: str

    def run(self, graph: ModelGraph) -> ModelGraph:
        """
        核心方法：immutable 变换。
        子类实现具体逻辑，必须返回新 graph（不修改输入）。
        """
        raise NotImplementedError

    def run_with_report(self, graph: ModelGraph) -> tuple[ModelGraph, PassResult]:
        """
        带 hash 追踪的执行（PassManager 调用）。
        对应 Axon Compile Report 中的 Pass Traceability 表格。
        """
        hash_before = graph.hash
        new_graph   = self.run(graph)
        hash_after  = new_graph.hash
        return new_graph, PassResult(
            pass_name    = self.name,
            hash_before  = hash_before,
            hash_after   = hash_after,
            changed      = hash_before != hash_after,
        )


class PassManager:
    """
    顺序执行 Pass 列表，收集每个 Pass 的执行结果。
    输出 CompileReport（对应 Axon 的 Compile Report）。
    """
    passes: list[Pass]

    def run(self, graph: ModelGraph) -> tuple[ModelGraph, CompileReport]:
        results = []
        current = graph
        for p in self.passes:
            current, result = p.run_with_report(current)
            results.append(result)
        return current, CompileReport(pass_results=results, final_graph=current)
```

### 3.3 Pass 执行顺序与职责

```
Pass 1：AnalysisPass（来自 Axon 设计）
  职责：静态估算 FLOPs、参数量、激活内存
  hash 变化：no（只增加 metadata，不改图结构）
  输出：GraphSummary（FLOPs 分布、参数分布、内存峰值估算）

Pass 2：FusionPass（来自 Axon 设计）
  职责：识别并融合 RMSNorm+QKV+RoPE（F1）、SwiGLU MLP（F2）等 kernel
  hash 变化：yes（图节点数减少，融合后节点替换原节点）
  输出：融合后 ModelGraph，Compile Report 中记录融合数量

Pass 3：CommInferencePass（Axion 新增）
  职责：分析每个 OpNode 的通信需求，在图上标注 CommAnnotation
  hash 变化：yes（OpNode 增加 comm_annotation，图结构扩展）
  输入依赖：ParallelismSpec（需要知道 EP/TP/DP 划分）
  核心逻辑：
    ExpertGateSpec → 后续需要 A2A dispatch（标注 A2ASpec）
    ExpertFFNSpec  → 前后需要 A2A（dispatch + combine）
    MatMulSpec（TP 分片）→ 后续需要 AllReduce
    参数节点  → 前向前 AllGather，反向后 ReduceScatter

Pass 4：FSEPShardingPass（Axion 新增）
  职责：基于 CommAnnotation 生成 Expert 初始分片方案（ShardingPlan）
  hash 变化：yes（ExpertShard Wire 类型和 ExpertShardSpec 节点插入）
  输入依赖：ParallelismSpec、ExpertPlacementSpec、历史 profile（可选）
  核心逻辑：贪心算法（Phase 1）或 ILP（Phase 3）
  输出：ShardingPlan（expert_id → {gpu_id, shard_rank, total_shards}）

Pass 5：OverlapInsertionPass（Axion 新增）
  职责：识别 Compute Op 和 Comm Op 的 overlap 机会，插入 OverlapSpec 节点
  hash 变化：yes（新增 OverlapSpec 节点，图拓扑变化）
  核心逻辑：
    对每对 (Compute Op K, Comm Op C)：
      if not data_depends(K→C) and not data_depends(C→K)：
        插入 Sched.Overlap(K, C)
  策略：AGGRESSIVE（默认）/ CONSERVATIVE / DISABLED

Pass 6：CommTensorLayoutPass（Axion 新增）
  职责：将 CommAnnotation 中的 layout_hint 转化为 CommTensor 物理布局决策
  hash 变化：yes（Wire 类型从 DENSE → COMM，增加 layout 属性）
  决策规则：
    Expert dispatch A2A 输入 → BLOCKED_BY_DST
    Expert dispatch A2A 输出 → BLOCKED_BY_SRC
    Expert combine  A2A 输入 → BLOCKED_BY_SRC
    Expert 参数 AllGather   → BLOCKED_BY_EXPERT
    稀疏路由（sparsity>70%）→ SPARSE_CSR
```

---

## 4. CommTensor：通信友好数据结构

### 4.1 设计动机：消除 pack/unpack

```
传统 MoE All-to-All 的四步开销：

  tokens [seq, hidden]
      │  ① pack（按目标 GPU 重排内存）   O(seq × hidden) copy
      ▼
  packed [gpu0_block | gpu1_block | ...]
      │  ② All-to-All（网络传输）
      ▼
  received [src0_block | src1_block | ...]
      │  ③ Expert FFN 计算
      ▼
  results [src0_block | src1_block | ...]
      │  ④ unpack（恢复原始 token 顺序）  O(seq × hidden) copy
      ▼
  output [seq, hidden]

CommTensor 的消除方案：

  物理内存布局 = 通信友好布局（编译期由 CommTensorLayoutPass 决定）
  逻辑视图    = 原始语义视图（通过 index map 零拷贝访问）

  CommTensor（物理已经是 BLOCKED_BY_DST）
      │  直接 DMA → All-to-All（无需 pack）
      ▼
  CommTensor（接收端，物理已经是 BLOCKED_BY_SRC）
      │  Expert FFN 直接访问 logical_view（无需 unpack）
      ▼
  结果 CommTensor
      │  直接 DMA → All-to-All combine
      ▼
  CommTensor → logical_view → output

  节省：2 次 O(seq × hidden) 的内存 copy
```

### 4.2 CommLayout 五种枚举

```
CommLayout 枚举（覆盖 MoE 训练所有主要通信模式）：

┌─────────────────────┬───────────────────────────────┬──────────────────────────┐
│  Layout             │  物理内存排列                   │  适用场景                 │
├─────────────────────┼───────────────────────────────┼──────────────────────────┤
│  BLOCKED_BY_DST     │  [rank0数据 | rank1数据 | ...]  │  Expert dispatch A2A      │
│                     │  按目标 GPU 分组               │  直接 DMA，零 pack copy   │
├─────────────────────┼───────────────────────────────┼──────────────────────────┤
│  BLOCKED_BY_SRC     │  [rank0数据 | rank1数据 | ...]  │  Expert combine 接收端    │
│                     │  按来源 GPU 分组               │  AllGather 输出          │
├─────────────────────┼───────────────────────────────┼──────────────────────────┤
│  BLOCKED_BY_EXPERT  │  [Expert0params|Expert1params] │  FSEP Expert 参数 AG      │
│                     │  按 Expert ID 分组             │  shard 连续，AG 无重排    │
├─────────────────────┼───────────────────────────────┼──────────────────────────┤
│  INTERLEAVED        │  [t0, t1, t2, ..., tN]        │  Attention 计算输入/输出  │
│                     │  原始 token 顺序               │  CommTensor 的退化状态   │
├─────────────────────┼───────────────────────────────┼──────────────────────────┤
│  SPARSE_CSR         │  CSR: value+col_idx+row_ptr   │  路由稀疏度 > 70% 时      │
│                     │  稀疏格式                      │  跳过空 Expert，减少量   │
└─────────────────────┴───────────────────────────────┴──────────────────────────┘

Layout 状态机（编译器追踪）：

  INTERLEAVED
      │  CommTensor.FromDense（routing 分组）
      ▼
  BLOCKED_BY_DST
      │  Comm.A2A（dispatch）
      ▼
  BLOCKED_BY_SRC
      │  Expert.FFN（本地计算，layout 不变）
      ▼
  BLOCKED_BY_SRC
      │  Comm.A2A（combine）
      ▼
  BLOCKED_BY_DST
      │  CommTensor.ToDense（恢复顺序）
      ▼
  INTERLEAVED
```

### 4.3 CommTensor 内存结构

```python
@dataclass
class CommTensor:
    """
    通信友好的 Tensor 数据结构。
    物理布局由编译期 CommTensorLayoutPass 决定，运行时只读。
    """
    # 物理存储（按 CommLayout 排列的 CUDA 内存）
    _physical_data:   torch.Tensor
    _physical_layout: CommLayout

    # 通信元数据（编译期写入，运行时只读）
    comm_spec: CommSpec
    # CommSpec 字段：
    #   comm_type:   CommType          A2A / ALL_GATHER / REDUCE_SCATTER
    #   target_gpus: list[int]         数据要去往哪些 GPU
    #   send_counts: list[int|Symbol]  发给每个 GPU 的元素数（Symbol=运行时填）
    #   recv_counts: list[int|Symbol]  从每个 GPU 收的元素数
    #   group:       ProcessGroup      通信组

    # 逻辑→物理的 index map（零拷贝访问，由 CommTensorLayoutPass 生成）
    _logical_to_physical: torch.Tensor   # int64 index tensor，存于 GPU SRAM

    @property
    def logical_view(self) -> torch.Tensor:
        """逻辑视图：按原始语义顺序访问，零拷贝（index_select）"""
        return self._physical_data[self._logical_to_physical]

    @property
    def physical_view(self) -> torch.Tensor:
        """物理视图：直接访问底层内存，用于 CommFabric DMA"""
        return self._physical_data


@dataclass
class FSEPExpertTensor(CommTensor):
    """
    FSEP 专用的 Expert 参数 Tensor。
    一个 Expert 可以分片存储在多个 GPU 上（热点 Expert）。
    """
    expert_id:       int
    shard_rank:      int              # 这是第几个 shard（0-indexed）
    total_shards:    int              # Expert 参数总共几个 shard
    current_gpu:     int
    migration_state: MigrationState   # STABLE / MIGRATING / SHADOW

    # 迁移期间的影子 buffer（借鉴 LAER-MoE double buffer 设计）
    shadow_buffer:   torch.Tensor | None = None

    def gather_full_expert(self) -> torch.Tensor:
        """
        AllGather 完整 Expert 参数。
        BLOCKED_BY_EXPERT 布局保证 shard 天然连续 → 直接 DMA。
        当 total_shards == 1 时，编译器消除此 AllGather（零通信）。
        """
        if self.total_shards == 1:
            return self._physical_data
        return comm_fabric.all_gather(
            self._physical_data,
            group=self.expert_parallel_group,
            output_layout=CommLayout.BLOCKED_BY_EXPERT,
        )
```

---

## 5. DistributedExecutablePlan：分布式执行计划

### 5.1 设计原则（继承 Axon）

```
Axon 对 ExecutablePlan 的设计约定，Axion 全部遵守：

  Immutable：Plan 一旦生成不可修改
  Hashable：给定相同的 ModelGraph → 产生相同 hash 的 Plan（确定性）
  Self-contained：Plan 携带执行所需的全部信息，不依赖外部状态
  Inspectable：Plan 的每一步骤可以单独检查和打印
```

### 5.2 数据结构

```python
@dataclass(frozen=True)
class PlannedKernel:
    """一个 compute 步骤，对应 Axon 设计中的 PlannedKernel"""
    node_id:    str
    op_spec:    OpSpec
    input_wires: tuple[str, ...]
    output_wires: tuple[str, ...]
    kernel_key:  str      # 选用哪个 kernel 实现（如 "flash_attn_v3"）

@dataclass(frozen=True)
class PlannedCommOp:
    """一个 comm 步骤（Axion 新增，与 PlannedKernel 同等地位）"""
    node_id:      str
    comm_spec:    CommSpec
    input_wires:  tuple[str, ...]
    output_wires: tuple[str, ...]
    overlap_with: str | None    # 与哪个 PlannedKernel overlap

@dataclass(frozen=True)
class StaticSchedule:
    """
    编译期确定的执行时间线（Axion 的关键设计）。
    运行时按 schedule 顺序执行，不做动态调度决策。
    体现 Deterministic 原则。
    """
    steps: tuple[PlannedKernel | PlannedCommOp, ...]
    # 相互 overlap 的 (kernel_id, comm_id) 对
    overlap_pairs: tuple[tuple[str, str], ...]

@dataclass(frozen=True)
class DistributedExecutablePlan:
    """
    Axion 的分布式执行计划。
    在 Axon ExecutablePlan 设计基础上增加通信和分布式字段。
    immutable + hashable（frozen dataclass）。
    """
    # Compute 步骤（来自 Axon ExecutablePlan 设计）
    compute_steps:   tuple[PlannedKernel, ...]

    # Comm 步骤（Axion 新增）
    comm_steps:      tuple[PlannedCommOp, ...]

    # 静态调度时间线（编译期确定，Axion 新增）
    static_schedule: StaticSchedule

    # FSEP Expert 分片注册表（Axion 新增）
    expert_registry: ExpertShardRegistry

    # 编译元数据（扩展 Axon CompileMetadata）
    metadata:        DistCompileMetadata

    @property
    def hash(self) -> str:
        """确定性 hash，相同输入图 → 相同 hash（Axon 设计约定）"""
        return sha256(self._canonical_repr()).hexdigest()
```

---

## 6. Compile Report（扩展 Axon 设计）

Axion 完整继承 Axon 的 Compile Report 设计，增加通信维度：

```markdown
# Axion Compile Report

**Model**: DeepSeek-V3-like (256 experts, 61 layers)  
**Final graph hash**: `d4e5f6a7b8c9...`

---

## Pass Traceability（来自 Axon 设计）
| Pass                    | Hash Before   | Hash After    | Changed |
|-------------------------|--------------|--------------|---------|
| analysis                | `21618b3c...` | `21618b3c...` | no      |
| fusion                  | `21618b3c...` | `f6e5d4c3...` | yes     |
| comm_inference          | `f6e5d4c3...` | `a1b2c3d4...` | yes     |
| fsep_sharding           | `a1b2c3d4...` | `b2c3d4e5...` | yes     |
| overlap_insertion       | `b2c3d4e5...` | `c3d4e5f6...` | yes     |
| comm_tensor_layout      | `c3d4e5f6...` | `d4e5f6a7...` | yes     |

---

## Resource Estimates（来自 Axon 设计）
- Total FLOPs:     **248T** (per step, B=1, S=4096)
- Total params:    **671B**
- Activation mem:  **~38 GB** (per GPU, EP=64)

---

## Fusion Summary（来自 Axon 设计）
- Ops before fusion: **3,721**
- Ops after fusion:  **1,876**
- F1 (RMSNorm+QKV+RoPE): **61 fusions**
- F2 (SwiGLU MLP):        **58 fusions**

---

## Communication Analysis（Axion 新增）
- Total Comm Ops:               **496**
  - A2A Expert dispatch:         124
  - A2A Expert combine:          124
  - AllGather (Dense params):    124
  - ReduceScatter (Dense grads): 124
- Overlapped Comm Ops:  **434 / 496 = 87.5%**
- Exposed (critical path): **62 / 496 = 12.5%**
- Estimated comm overhead: **~8%** of total step time

---

## FSEP Sharding Plan（Axion 新增）
- Total Experts:              **256**
- Predicted hot experts:      **14** (load > 2.0x average)
- FSEP-sharded experts:       **14** (split to 2~4 shards)
- Avg shards per expert:      **1.05**
- Cross-node A2A reduction:   **~32%** (topology-aware routing)

---

## Recommendations
1. 通信 overlap 率 87.5%，剩余 12.5% 在关键路径（A2A→Expert FFN 依赖链）。
2. 14 个热点 Expert 已启用 FSEP，建议监控迁移频率（MigrationConfig.check_interval）。
3. Attention 占 FLOPs 的 34.8%，建议 attention_backend='flash'（aiter for MI300X）。
4. Cross-node A2A 已减少 32%，主要收益来自拓扑感知的 Expert 初始分配。
```

---

## 7. FSEP 运行时：双层规划

### 7.1 Slow Planner（每 K step）

借鉴 LAER-MoE 的 Load-Adaptive Planner：

```
输入：
  过去 K step 的 token → expert 路由统计
  当前 Expert 物理分布（expert_id → gpu_id）
  集群拓扑（节点内/跨节点带宽）

目标函数（最优化问题）：
  minimize:   max_gpu( compute_time(gpu_i) + comm_time(gpu_i) )
  subject to:
    ∑ expert_params_on(gpu_i) ≈ total_params / N_gpus  （内存均衡）
    migration_comm_cost ≤ compute_saving                （ROI 约束）
    cross_node_experts ≤ C_max                          （拓扑约束）

输出：
  ExpertMigrationPlan（哪个 Expert Shard 迁到哪个 GPU）
  执行时机：在当前 step 的反向传播期间异步完成迁移

迁移过程（LAER-MoE double buffer 设计）：
  Step T：Slow Planner 输出迁移计划
    │  异步 P2P：src_gpu → dst_gpu（与反向传播重叠）
    │  迁移期间：migration_state = MIGRATING
    │           src_gpu 上 SHADOW 副本仍有效
    ▼
  Step T+1：新的 Expert 分布生效
    SHADOW 副本释放，migration_state = STABLE
```

### 7.2 Fast Router（每 step）

在当前 Expert 物理布局下，通过 routing bias 微调进一步均衡：

```
gate_logits_adjusted[i] = gate_logits[i] - α · load_penalty[i]

load_penalty[i] = (tokens_on_expert_i / avg_tokens) ^ β
                  × gpu_utilization[gpu_of_expert_i]

α, β：超参，控制均衡强度（默认 α=0.1, β=2.0）

设计约定（Deterministic 原则）：
  load_penalty 基于上一个 step 的统计，不引入随机性
  给定相同的历史统计 → 相同的 routing bias → 确定性

风险（需验证）：
  routing bias 改变 gate 的梯度，可能影响模型收敛
  LAER-MoE 是物理迁移（不改路由语义），Fast Router 更激进
  建议：先用消融实验验证，再在生产中启用
```

### 7.3 两层协同

```
Slow Planner 改变物理布局（大调，有迁移开销，每 50 step）
Fast Router  通过 routing bias 微调（小调，无迁移开销，每 step）

协同效果：
  Fast Router 在 Slow Planner 改变布局后的"过渡期"提供即时均衡
  Slow Planner 从根本上改变 Expert 分布，消除系统性不均衡
  两者叠加 → 持续保持负载均衡，无需依赖 Token Drop
```

---

## 8. CommFabric：通信底层抽象

```
设计目标：统一 NVLink / InfiniBand / RDMA 的抽象，
          让 OverlapInsertionPass 和 FSEP 运行时与底层网络解耦。

┌────────────────────────────────────────────────────────┐
│                  Pass / FSEP Runtime                   │
│  comm_fabric.all_to_all_async(tensor, ...)             │
│  comm_fabric.all_gather_async(shard, ...)              │
│  comm_fabric.p2p_async(src, dst, ...)                  │
├──────────────────┬─────────────────┬───────────────────┤
│  NVLink Driver   │  IB/RDMA Driver │  Simulate Driver  │
│  节点内 A2A      │  节点间 A2A     │  no-op 模拟       │
│  ~600 GB/s       │  ~200 GB/s      │  单机验证用       │
└──────────────────┴─────────────────┴───────────────────┘

拓扑感知路由（ClusterTopologySpec 驱动）：
  Expert A2A 节点内部分：优先走 NVLink（高带宽）
  Expert A2A 跨节点部分：走 IB RDMA，FSEPShardingPass 已最小化跨节点量

Simulate Driver（单机验证用，对应 Axon 的单机调试理念）：
  CommFabric = no-op → 可以在单机上运行完整编译流程
  验证通信图正确性，不需要多 GPU
  对应 Phase 1 的验证策略
```

---

## 9. 实现路径

```
Phase 0（基础，对应 Axon 设计）：单机 Compile-First 基础设施
  ModelGraph · PassManager · DistributedExecutablePlan（无通信）
  AnalysisPass · FusionPass
  Attention backends：sdpa / flash / aiter / reference
  Llama 3.1 8B 单机跑通，建立计算性能基线

Phase 1（2 周）：通信图可视化
  CommInferencePass（标注通信需求，不实际执行通信）
  CommFabric = Simulate Driver（no-op）
  扩展 Compile Report：显示通信拓扑、CommLayout 分配
  单机验证通信图分析正确性

Phase 2（1 月）：分布式基础
  FSEPShardingPass + 基础 CommFabric（NVLink）
  DistributedExecutablePlan（包含 comm_steps）
  2~8 GPU 验证 Expert dispatch/combine 正确性
  Slow Planner（关闭 Expert 迁移，只做初始分配）

Phase 3（2 月）：Overlap + CommTensor
  OverlapInsertionPass + CommTensorLayoutPass
  CommTensor 零 copy 实现（index map 机制）
  StaticSchedule 执行引擎
  IB RDMA 接入

Phase 4（2 月）：FSEP 完整运行时
  Expert 迁移执行器（double buffer 机制）
  Slow Planner 完整实现（ILP 或贪心）
  Fast Router（需配合收敛实验）
  64+ GPU，DSv3-like 模型验证

→ 约 6 个月，端到端可训练 prototype
→ 对标 LAER-MoE 1.69x 加速、veScale-FSDP 5~66% 吞吐提升
```

---

## 10. 系统定位与设计决策总结

### 10.1 在 MoE 训练优化栈中的位置

```
┌────────────────────────────────────────────────────────────┐
│  应用层：模型架构 + 优化器（Shampoo, Muon, AdamW）           │
├────────────────────────────────────────────────────────────┤
│  Axion 层：CommTensor + IR + FSEP 运行时        ← 这里      │
│  ├── Compile-First（来自 Axon）                             │
│  ├── Communication-Native                                  │
│  └── Load-Adaptive（来自 LAER-MoE）                        │
├────────────────────────────────────────────────────────────┤
│  MoE 优化层：LAER-MoE（负载均衡），MoEBlaze（内存）          │
├────────────────────────────────────────────────────────────┤
│  通信底层：NCCL，DeepEP，RDMA                               │
└────────────────────────────────────────────────────────────┘
```

### 10.2 核心设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| **IR 基础** | 自定义 `ModelGraph`（采用 Axon 设计） | FX Graph 无通信语义；Axon 的 immutable+hashable 设计正确 |
| **通信地位** | `CommOpSpec` 与 `OpSpec` 同等地位 | 通信是一等公民，不是黑盒副作用 |
| **Pass 系统** | 采用 Axon 的 immutable Pass 约定 | 可追踪、可调试、可复现 |
| **CommTensor 布局** | 编译期静态决定（CommTensorLayoutPass） | Deterministic 原则，运行时零决策 |
| **Expert 迁移** | `ShardMigrateSpec` 是 IR 一等公民 | 迁移逻辑必须对编译器可见，才能规划 overlap |
| **负载均衡** | Slow Planner + Fast Router 双层 | Slow 改物理布局（根本），Fast 改路由偏置（即时） |
| **单机验证** | Simulate Driver no-op | 继承 Axon 的调试理念，不需要多 GPU 就能验证编译逻辑 |

### 10.3 三个设计目标的实现

```
目标 1：Compile-First · Deterministic（来自 Axon）
  实现：ModelGraph immutable + Pass hash 追踪
       StaticSchedule 编译期确定 → 运行时零动态决策
       Slow Planner 是唯一的运行时决策点（受约束）

目标 2：Communication-Native（Axion 核心）
  实现：CommOpSpec 与 ComputeOpSpec 同等地位
       CommTensor 物理布局消除 pack/unpack
       Comm Pass 与 Compute Pass 在同一 PassManager 中

目标 3：Load-Adaptive · Scalable（来自 LAER-MoE）
  实现：FSEPShardingPass 初始化最优 Expert 分布
       Slow Planner 动态重平衡（Expert 迁移）
       Fast Router 即时微调（routing bias）
```

---

*Axion Architecture Design v0.3 | 2026-03-08*  
*理念融合：Axon (Compile-First · Deterministic · Debuggable) + LAER-MoE (arXiv:2602.11686) + veScale-FSDP (arXiv:2602.22437)*
