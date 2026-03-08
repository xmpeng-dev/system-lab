# ROCflow IR Design: RFGraph

> **文档版本:** 0.1  
> **状态:** 设计讨论稿  
> **关联模块:** `rocflow/ir/`, `rocflow/scheduler/`, `rocflow/compiler/`

---

## 1. 为什么需要自己的 IR

### 1.1 `nn.Module` eager 模式的根本局限

ROCflow 的三层 Overlap 目标（Block 级 / Layer 级 / Tile 级）要求框架具备**全局调度视野**：在一个训练 Step 内，哪两个操作可以并发，哪个通信走 XGMI，哪个走 RDMA，哪块内存可以提前复用。

`nn.Module.forward()` 是一个 Python 函数，其执行模型天然是：

```
Python 调用栈（串行、同步）：

forward(x)
  → norm1(x)                    # 等待完成
  → attn(...)                   # 等待完成
  → A2A_Dispatch(...)           # 通信阻塞，GPU 空转
  → expert_gemm(...)            # 等待完成
  → A2A_Gather(...)             # 通信阻塞，GPU 空转
  → ...

框架在运行时完全不知道：
  ✗ Block_i 的 A2A-Gather 和 Block_{i+1} 的 Attn GEMM 是否可以并发
  ✗ 这个 AllReduce 和当前的 Expert GEMM 哪个先做更好
  ✗ 这个 Tensor 什么时候不再被使用（内存何时可以复用）
  ✗ 这个 A2A 应该走 XGMI 还是 RDMA
```

### 1.2 `torch.compile` 解决了什么，没解决什么

`torch.compile` 通过 `torch.fx` trace 将 Python eager 转成静态图，再由 TorchInductor 生成优化 Kernel。但它有三个对 ROCflow 至关重要的缺口：

| 缺口 | 原因 | 对 ROCflow 的影响 |
|------|------|-----------------|
| **通信是 graph break** | `dist.all_to_all` 等集合通信无法被 trace 进 FX Graph | 计算-通信 Overlap 无法在 compiled graph 内部实现 |
| **跨 Module 边界不可见** | 每个 `nn.Module.forward()` 是独立的 trace 单元 | 跨 Block 的 Overlap 分析没有信息来源 |
| **无硬件路径感知** | torch.compile 输出的 Kernel 不区分 XGMI / RDMA | 拓扑感知调度无法表达 |

### 1.3 设计目标

ROCflow 需要一个 IR，满足以下能力：

1. **通信操作是一等公民节点**，不是 graph break
2. **跨 Module 展平**，一个 Step 的所有操作在同一张图里
3. **硬件路径可标注**，每个通信节点知道自己走哪条物理路径
4. **Tensor lifetime 可追踪**，支持激活内存的精确复用
5. **不替换 `nn.Module`**，用户侧零迁移成本
6. **与 `torch.compile` 协作**，ComputeNode 最终委托 compile 生成 Kernel

---

## 2. 整体架构：两层模型

```
┌─────────────────────────────────────────────────────────────┐
│  用户层（User-Facing）：nn.Module                            │
│                                                              │
│  class MyMoEBlock(nn.Module):                                │
│      def forward(self, x): ...                               │
│                                                              │
│  用户正常写 PyTorch，ROCflow 提供 MoELayer / wrap_attention  │
│  对用户完全透明                                               │
└───────────────────────────┬─────────────────────────────────┘
                             │
                  rocflow.compile() 触发 Trace
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  框架内部层（Framework-Internal）：RFGraph IR                │
│                                                              │
│  • 通过 RFTracer 将 nn.Module 展开为 RFGraph                │
│  • 通信操作升级为 CommNode（一等公民）                        │
│  • 跨 Block 边界展平到同一张图                               │
│  • DAG Scheduler 在图上做调度分析                            │
│  • Lowering：生成 Scheduled Execution Plan                   │
└───────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│  执行层（Execution）：Scheduled Execution Plan               │
│                                                              │
│  • 精确排好顺序的 Stream 调度序列                             │
│  • ComputeNode → torch.compile region（Kernel 生成）         │
│  • CommNode    → RCCL call（绑定到指定 stream + hw_path）     │
│  • SyncNode    → stream.wait_event（跨 stream 依赖点）        │
│  • 运行时几乎无 Python overhead（接近 CUDA/HIP Graph）        │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. RFGraph 数据结构定义

### 3.1 节点类型（RFNode）

```
RFNode 基类属性：
  id          : int             # 全局唯一节点 ID
  op_type     : RFOpType        # 操作类型枚举（见 3.2）
  stream_id   : int             # 分配到的 HIP Stream 编号
  tensor_in   : List[RFTensor]  # 输入 Tensor 列表
  tensor_out  : List[RFTensor]  # 输出 Tensor 列表
  deps        : Set[RFNode]     # 数据依赖的前置节点集合（DataEdge）
  stream_deps : Set[RFNode]     # 同 stream 内的顺序依赖（StreamEdge）
  event_deps  : Set[RFNode]     # 跨 stream 的异步依赖（EventEdge）
  meta        : Dict            # 附加元信息（shape、dtype、block_id 等）
```

四种具体节点类型：

```
ComputeNode（计算节点）：
  继承自 RFNode
  + kernel_fn    : Callable       # 对应的 HIP Kernel 或 compile region
  + tile_size    : Optional[int]  # Tile 粒度（Tile-level overlap 时使用）
  + wavefront_groups : int        # 分配给计算的 Wavefront 组数

CommNode（通信节点）：
  继承自 RFNode
  + comm_op      : CommOpType     # A2A_DISPATCH / A2A_GATHER / ALL_REDUCE /
                                  # ALL_GATHER / REDUCE_SCATTER / RING_ATTN
  + hw_path      : HWPath         # XGMI / RDMA / LOCAL
  + process_group: ProcessGroup   # 通信参与的进程组
  + is_async     : bool           # 是否异步发起（默认 True）

MemNode（内存节点）：
  继承自 RFNode
  + mem_op       : MemOpType      # ALLOC / FREE / COPY / SWAP
  + size_bytes   : int            # 操作的内存大小
  + src_device   : Device         # 源设备（COPY 操作时）
  + dst_device   : Device         # 目标设备（COPY 操作时）

SyncNode（同步节点）：
  继承自 RFNode
  + wait_stream  : int            # 等待哪个 stream 的 event
  + event_source : RFNode         # event 来自哪个节点
```

### 3.2 操作类型枚举（RFOpType）

```python
class RFOpType(Enum):
    # 计算操作
    EXPERT_GEMM         = "expert_gemm"        # Expert FFN 矩阵乘
    ATTN_GEMM           = "attn_gemm"          # Attention QKV / output 矩阵乘
    GATE                = "gate"               # 路由 Gate 网络
    LAYERNORM           = "layernorm"          # LayerNorm / RMSNorm
    ACTIVATION          = "activation"         # SiLU / GELU 等激活函数
    ELEMENTWISE         = "elementwise"        # 逐元素操作（Add、Mul 等）
    GENERIC_COMPUTE     = "generic_compute"    # 其他计算（torch.compile 托管）

    # 通信操作（MoE 专属）
    A2A_DISPATCH        = "a2a_dispatch"       # Expert 分发（前向）
    A2A_GATHER          = "a2a_gather"         # Expert 聚合（前向）
    A2A_DISPATCH_BWD    = "a2a_dispatch_bwd"   # A2A_GATHER 的反向
    A2A_GATHER_BWD      = "a2a_gather_bwd"     # A2A_DISPATCH 的反向

    # 通信操作（并行同步）
    TP_ALL_REDUCE       = "tp_all_reduce"      # 张量并行梯度同步
    DP_ALL_REDUCE       = "dp_all_reduce"      # 数据并行梯度同步
    CP_RING_ATTN        = "cp_ring_attn"       # 上下文并行 KV 传递
    FSDP_ALL_GATHER     = "fsdp_all_gather"    # FSDP 参数聚合
    FSDP_REDUCE_SCATTER = "fsdp_reduce_scatter"# FSDP 梯度分散

    # 内存操作
    MEM_ALLOC           = "mem_alloc"
    MEM_FREE            = "mem_free"
    MEM_COPY            = "mem_copy"
    AC_SAVE             = "ac_save"            # 激活检查点保存
    AC_RECOMPUTE        = "ac_recompute"       # 激活检查点重算

    # 同步操作
    STREAM_SYNC         = "stream_sync"        # stream.synchronize()
    EVENT_RECORD        = "event_record"       # stream.record_event()
    EVENT_WAIT          = "event_wait"         # stream.wait_event()
```

### 3.3 硬件路径枚举（HWPath）

```python
class HWPath(Enum):
    LOCAL       = "local"       # 同 GPU 内，无通信
    XGMI        = "xgmi"        # AMD XGMI / Infinity Fabric（节点内）
    RDMA        = "rdma"        # RDMA over RoCE / InfiniBand（跨节点）
    PCIE        = "pcie"        # PCIe（无 XGMI 的节点内场景）
    AUTO        = "auto"        # 由调度器自动决定
```

### 3.4 RFTensor

```
RFTensor：
  id           : int            # 全局唯一 Tensor ID
  shape        : List[SymInt]   # 支持 symbolic shape（动态路由场景）
  dtype        : torch.dtype
  layout       : TensorLayout   # STANDARD / EXPERT_SLOT / PINNED
  producer     : RFNode         # 生产该 Tensor 的节点
  consumers    : List[RFNode]   # 消费该 Tensor 的所有节点
  lifetime     : Tuple[int,int] # (first_use_step, last_use_step) 用于内存规划
  is_parameter : bool           # 是否是模型参数（不参与激活内存规划）
  is_pinned    : bool           # 是否 pinned memory（RDMA 直接访问）
```

`TensorLayout` 枚举：
- `STANDARD`：标准 `[B, S, H]` 布局
- `EXPERT_SLOT`：`ExpertSlotTensor` 布局，按通信目的地组织
- `PINNED`：Pinned memory，供 RDMA DMA 直接访问

### 3.5 边类型（Edge）

```
三种边编码三种依赖关系：

DataEdge（数据依赖）：
  src → dst：dst 必须等 src 的输出 tensor 可用后才能执行
  在同 stream 或跨 stream 都会强制等待

StreamEdge（stream 内顺序依赖）：
  src → dst：同一个 stream 内，dst 在 src 之后提交
  不产生跨 stream 的同步开销

EventEdge（跨 stream 异步依赖）：
  src → dst：src 完成后 record_event，dst 开始前 wait_event
  允许 src 和 dst 在不同 stream 上并发执行到同步点
  这是实现 Overlap 的核心机制
```

---

## 4. Tracing：从 `nn.Module` 到 RFGraph

### 4.1 两条 Trace 路径

```
路径 A：编译期 Symbolic Trace（基于 torch.fx 扩展）
  适用场景：模型结构静态，routing 可以用 symbolic shape 表达
  优点：编译期完成分析，运行时 overhead 接近零
  缺点：对动态控制流支持有限

路径 B：运行期 Hook Trace（干运行记录操作序列）
  适用场景：模型有动态分支（如条件专家激活），或路径 A 无法 trace 的情况
  优点：完全支持动态控制流
  缺点：需要一次干运行（warmup step）才能生成 RFGraph
```

ROCflow 优先使用路径 A，路径 B 作为兜底。

### 4.2 路径 A：RFTracer（扩展 torch.fx.Tracer）

核心扩展点：让通信操作不再成为 graph break，而是成为可追踪的节点。

```python
class RFTracer(torch.fx.Tracer):
    """
    扩展 torch.fx.Tracer，将 MoE 通信操作识别为 CommNode。
    """

    # ROCflow 识别的通信操作集合
    COMM_OPS = {
        torch.distributed.all_to_all,
        torch.distributed.all_to_all_single,
        torch.distributed.all_reduce,
        torch.distributed.all_gather,
        torch.distributed.reduce_scatter,
        rccl.all_to_all,          # ROCm RCCL 原生 API
        rccl.all_reduce,
    }

    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        # ROCflow 的 MoELayer 需要展开追踪（内部有 A2A，不能作为 leaf）
        if isinstance(m, rocflow.nn.MoELayer):
            return False
        # 标准 nn.Module（Attention、LayerNorm 等）保持 leaf
        return super().is_leaf_module(m, module_qualified_name)

    def create_node(self, kind, target, args, kwargs, name=None, type_expr=None):
        node = super().create_node(kind, target, args, kwargs, name, type_expr)

        # 识别通信操作，标注为 CommNode 的元信息
        if kind == 'call_function' and target in self.COMM_OPS:
            node.meta['rf_node_type'] = 'CommNode'
            node.meta['rf_comm_op']   = self._infer_comm_op_type(target, args)
            node.meta['rf_hw_path']   = HWPath.AUTO  # 后续由 HWPathAssigner 填充
            node.meta['rf_async']     = True

        # 识别 Expert GEMM（hipBLASLt Group GEMM 调用）
        elif kind == 'call_function' and target in EXPERT_GEMM_OPS:
            node.meta['rf_node_type'] = 'ComputeNode'
            node.meta['rf_op_type']   = RFOpType.EXPERT_GEMM
            node.meta['rf_tile_size'] = DEFAULT_TILE_SIZE  # 后续由 TileSizeOptimizer 调整

        return node
```

### 4.3 路径 B：RFHookTracer（干运行 Hook）

```python
class RFHookTracer:
    """
    通过注册 forward/backward hook，在干运行中记录操作序列，
    构建 RFGraph。用于路径 A 无法处理的动态模型。
    """

    def trace(self, model: nn.Module, sample_input: torch.Tensor,
              parallel_config: ParallelConfig) -> 'RFGraph':
        self._trace_log: List[RFNode] = []
        hooks = self._register_hooks(model)

        # 干运行（不计算梯度，不做实际通信）
        with self._mock_comm_context():
            with torch.no_grad():
                output = model(sample_input)

        for h in hooks:
            h.remove()

        graph = self._build_rfgraph(self._trace_log, parallel_config)
        return graph

    def _register_hooks(self, model: nn.Module) -> List:
        hooks = []
        for name, module in model.named_modules():
            # 记录每个 Module 的 forward 输入输出
            h = module.register_forward_hook(
                lambda m, inp, out, n=name: self._on_forward(m, inp, out, n)
            )
            hooks.append(h)
        # 全局拦截通信操作
        hooks.append(self._patch_comm_ops())
        return hooks
```

### 4.4 从 FX Graph 到 RFGraph 的转换

Trace 完成后，FX Graph（或 Hook Log）经过以下 Pass 转换为 RFGraph：

```
Pass 1: NodeClassifier
  遍历所有节点，为每个节点分配 RFNode 类型（Compute/Comm/Mem/Sync）
  来源：node.meta['rf_node_type']（Trace 阶段已标注）

Pass 2: TensorLifetimeAnalyzer
  分析每个 RFTensor 的 producer 和所有 consumers
  计算 lifetime = (first_consumer.step, last_consumer.step)
  标注可以提前释放的 Tensor（激活内存复用的基础）

Pass 3: HWPathAssigner
  为每个 CommNode 分配 hw_path：
    - A2A 操作：检查通信的进程组是否在同一 XGMI 域内
      → 是：hw_path = XGMI
      → 否：hw_path = RDMA
    - AllReduce（TP）：通常节点内 → XGMI
    - AllReduce（DP）：通常跨节点 → RDMA
    - FSDP AllGather：取决于分片策略

Pass 4: DependencyBuilder
  遍历所有节点，构建完整的依赖边集合：
    - 有 tensor 数据依赖 → DataEdge
    - 同 stream 内顺序 → StreamEdge
    - 跨 stream 异步等待 → EventEdge（初始为空，由 Scheduler 填充）

Pass 5: BlockAnnotator
  为每个节点标注所属的 Transformer Block（block_id）
  用于后续跨 Block Overlap 分析
```

---

## 5. DAG Scheduler：在 RFGraph 上做调度优化

DAG Scheduler 是 ROCflow 最核心的组件，接收 RFGraph，输出带有 stream 分配和 event 同步点的 Scheduled RFGraph。

### 5.1 调度目标

```
minimize:  Step_latency = critical_path_length(scheduled_graph)

subject to:
  (1) 数据依赖不被违反（DataEdge 约束）
  (2) 同 stream 内顺序不被违反（StreamEdge 约束）
  (3) 硬件路径约束（XGMI 操作不能分配 RDMA stream，反之亦然）
  (4) 同时运行的操作不超过硬件并发限制
      （MI300X: compute streams × SIMD 使用率 + comm streams × 带宽使用率）
```

### 5.2 Stream 分配策略

```
ROCflow 默认 Stream 池：

Stream 0 (compute_stream_main)  : 主计算流，Expert GEMM、Attn GEMM
Stream 1 (compute_stream_aux)   : 辅助计算流，LayerNorm、Gate、激活函数
Stream 2 (comm_stream_xgmi)     : XGMI 通信流（节点内 A2A、TP AllReduce）
Stream 3 (comm_stream_rdma)     : RDMA 通信流（跨节点 A2A、DP AllReduce）
Stream 4 (comm_stream_fsdp)     : FSDP 专用流（AllGather / ReduceScatter）
Stream 5 (mem_stream)           : 内存操作流（异步激活释放、ExpertSlotTensor 重组）

分配规则：
  CommNode(hw_path=XGMI)    → comm_stream_xgmi
  CommNode(hw_path=RDMA)    → comm_stream_rdma
  CommNode(comm_op=FSDP_*)  → comm_stream_fsdp
  ComputeNode(EXPERT_GEMM)  → compute_stream_main
  ComputeNode(ATTN_GEMM)    → compute_stream_main（或 aux，视并发情况）
  ComputeNode(GATE/NORM/...) → compute_stream_aux
  MemNode                   → mem_stream
```

### 5.3 三层 Overlap 的调度实现

**Layer 3：跨 Block Pipeline Overlap**

```
条件：Block_i 的 CommNode C_i 和 Block_{i+1} 的 ComputeNode P_{i+1}
      之间没有 DataEdge

分析：
  C_i.deps ∩ P_{i+1}.deps == ∅  （互相不依赖对方的输出）
  → 可以并发

调度动作：
  C_i     → comm_stream_xgmi（或 rdma）
  P_{i+1} → compute_stream_main
  在 C_i 和 P_{i+1} 的共同下游节点前插入：
    SyncNode(event_wait, wait_stream=comm_stream_xgmi,
             event_source=C_i)
  → C_i 和 P_{i+1} 并发执行，只在真正需要 C_i 结果时才同步
```

**Layer 2：Attn 和 MoE 通信路径解耦**

```
同一 Block 内：
  TP_AllReduce（Attn 后）  → hw_path = XGMI → comm_stream_xgmi
  A2A_Dispatch（MoE 前）   → hw_path = XGMI → comm_stream_xgmi

问题：两个都在 XGMI，会串行！

解耦策略：
  若 TP_AllReduce 和下一层 Block 的 A2A_Dispatch 没有依赖关系：
    TP_AllReduce   → comm_stream_xgmi
    A2A_Dispatch   → comm_stream_rdma（即使是节点内，借用第二条流）

  AMD MI300X 有两条独立的 XGMI ring：
    Ring 0 → comm_stream_xgmi（主 A2A 流）
    Ring 1 → comm_stream_xgmi_aux（TP AllReduce 流）
  → 两条 XGMI ring 真正并发，带宽利用翻倍
```

**Layer 1：Tile 级 GEMM-RDMA Overlap**

```
这一层不在 DAG Scheduler 层面处理，而是在 HIP Kernel 内部实现。

DAG Scheduler 的职责：
  将 EXPERT_GEMM ComputeNode 标注为 tile_overlap=True
  指定 tile_size（由 TileSizeOptimizer 根据硬件参数计算）
  指定配套的 CommNode（哪个 A2A 与这个 GEMM 做 Tile 级 overlap）

HIP Kernel 层负责：
  Wavefront 专用化（计算组 vs 通信组）
  LDS 作为 Tile 输出的中间缓冲
  DMA 从 LDS 到对端 GPU（每个 Tile 完成后立即触发）
```

### 5.4 调度算法（Critical Path First）

```python
def schedule(self, graph: RFGraph) -> ScheduledRFGraph:
    """
    关键路径优先调度。
    以 as-soon-as-possible (ASAP) 为基础，对通信节点给予额外优先级。
    """
    ready_queue = PriorityQueue()  # (priority, node)
    scheduled   = {}               # node -> (stream_id, start_time_estimate)

    # 初始化：将无依赖的节点加入队列
    for node in graph.nodes:
        if len(node.deps) == 0:
            priority = self._compute_priority(node, graph)
            ready_queue.push((-priority, node))  # 最大堆

    while ready_queue:
        _, node = ready_queue.pop()
        stream_id = self._assign_stream(node)

        # 计算最早可开始时间（所有依赖中最晚结束的时间）
        earliest_start = max(
            (scheduled[dep].end_time + self._sync_cost(dep, node)
             for dep in node.deps),
            default=0
        )

        scheduled[node] = ScheduledNode(
            stream_id   = stream_id,
            start_time  = earliest_start,
            end_time    = earliest_start + self._estimate_duration(node),
        )

        # 为跨 stream 的依赖插入 EventEdge
        for dep in node.deps:
            if scheduled[dep].stream_id != stream_id:
                graph.add_event_edge(src=dep, dst=node)

        # 将新就绪的节点加入队列
        for successor in graph.successors(node):
            if all(dep in scheduled for dep in successor.deps):
                priority = self._compute_priority(successor, graph)
                ready_queue.push((-priority, successor))

    return ScheduledRFGraph(graph, scheduled)

def _compute_priority(self, node: RFNode, graph: RFGraph) -> float:
    """
    优先级 = 从该节点到图末尾的关键路径长度
    通信节点额外加权：通信延迟高，应尽早发起
    """
    cp = graph.critical_path_length_from(node)
    if isinstance(node, CommNode):
        cp *= COMM_PRIORITY_WEIGHT   # 默认 1.5
    return cp
```

### 5.5 内存调度：Tensor Lifetime 优化

```python
def optimize_memory(self, graph: ScheduledRFGraph) -> ScheduledRFGraph:
    """
    基于 Tensor lifetime 信息，在合法的最早时刻插入 MEM_FREE 节点，
    减少激活内存峰值。
    """
    for tensor in graph.tensors:
        if tensor.is_parameter:
            continue  # 参数不释放

        last_consumer = max(tensor.consumers, key=lambda n: n.scheduled_step)

        # 在最后一个消费者完成后，立即插入 MEM_FREE
        free_node = MemNode(
            op_type    = RFOpType.MEM_FREE,
            tensor_in  = [tensor],
            stream_id  = MEM_STREAM,
        )
        free_node.add_dep(last_consumer, edge_type=StreamEdge)
        graph.insert_node_after(last_consumer, free_node)

    return graph
```

---

## 6. Lowering：从 RFGraph 到 Execution Plan

调度完成后，RFGraph 经过 Lowering 生成可直接执行的 Python callable。

### 6.1 Lowering Pass

```
Pass 1: ComputeRegionGrouper
  将相邻的 ComputeNode（同 stream，无通信节点间隔）合并为 compile region
  送入 torch.compile 生成优化 Kernel

Pass 2: CommCallGenerator
  为每个 CommNode 生成对应的 RCCL API 调用
  绑定 process_group 和 stream

Pass 3: SyncPointInserter
  在所有 EventEdge 处插入 event_record + event_wait 调用

Pass 4: TileOverlapInjector
  将标注了 tile_overlap=True 的 EXPERT_GEMM ComputeNode
  替换为 rocflow_expert_tile_kernel 调用（HIP Wavefront 专用化 Kernel）
```

### 6.2 Execution Plan 结构

```python
class ExecutionPlan:
    """
    一个训练 Step 的完整执行计划。
    内部是有序的操作序列，每个操作绑定到特定 stream。
    运行时几乎无 Python overhead。
    """
    def __call__(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # 示例：一个经过调度的 MoE Block 的执行序列
        # （实际由 Lowering 自动生成，非手写）

        # Block_i 前向
        x_norm   = self.norm1_kernel(inputs['x'])               # Stream 1
        attn_out = self.attn_compiled_region(x_norm)             # Stream 0
        tp_ar    = self.tp_allreduce(attn_out,
                                     stream=XGMI_AUX_STREAM)     # Stream 2-aux（非阻塞）

        # 同时：Block_{i+1} 的 Attn 已可以开始（若无数据依赖）

        gate_out = self.gate_kernel(x_norm)                      # Stream 1
        slot_tensor = self.reorder_to_expert_slots(gate_out)     # Stream 1

        a2a_dispatch = self.rccl_a2a(slot_tensor,
                                      stream=XGMI_STREAM)         # Stream 2（非阻塞）

        XGMI_STREAM.wait_event(tp_ar.done_event)                  # 等 TP AR 完成

        dispatched = a2a_dispatch.wait()                           # 等 A2A 完成

        # Tile-level overlap：GEMM + RDMA 在 HIP Kernel 内部并发
        expert_out = self.rocflow_expert_tile_kernel(
            dispatched, self.expert_weights,
            tile_size=TILE_SIZE,
            comm_stream=XGMI_STREAM,
        )

        a2a_gather = self.rccl_a2a(expert_out,
                                    stream=XGMI_STREAM)            # Stream 2（非阻塞）

        gathered = a2a_gather.wait()
        output   = inputs['x'] + gathered
        return output
```

---

## 7. 与 `torch.compile` 的协作关系

RFGraph 和 `torch.compile` 不是竞争关系，而是分工明确的两层：

```
职责分工：

RFGraph / DAG Scheduler：
  ✅ 跨操作的调度顺序（哪个先，哪个并发）
  ✅ 通信操作的硬件路径分配（XGMI vs RDMA）
  ✅ Stream 分配和 event 同步点插入
  ✅ Tensor lifetime 追踪和内存复用
  ✅ 跨 Module / 跨 Block 的全局视野
  ✗ 单个 Kernel 内部的优化（这是 compile 的工作）

torch.compile（TorchInductor / hipBLASLt autotuning）：
  ✅ 单个 ComputeRegion 内的算子融合（Gate + LayerNorm + 激活函数）
  ✅ Expert GEMM 的 hipBLASLt autotuning（最优 tile / split-k 参数）
  ✅ 内存布局优化（NHWC vs NCHW 等）
  ✅ 循环展开、向量化
  ✗ 跨通信操作的优化（无法看到）
  ✗ 多 stream 并发调度（不是 compile 的职责）
```

---

## 8. 关键设计决策 Q&A

**Q1：为什么不直接基于 `torch.fx` FX Graph 做所有事情？**

FX Graph 的 graph break 机制会在 `dist.all_to_all` 处截断图，导致通信操作两侧是两个独立的 compiled region。RFGraph 通过扩展 `RFTracer` 使通信操作可追踪，让整个 Block（甚至多个 Block）的操作共享一张图，这是关键区别。

**Q2：RFGraph 和 JAX 的 jaxpr IR 有什么本质区别？**

JAX 是全函数式 IR，所有操作包括副作用都被 trace。RFGraph 不追求函数式纯洁性，它只关心 **调度相关的语义**（操作类型、依赖关系、硬件路径、Tensor lifetime），不可调度的细节（Kernel 参数、矩阵维度等）委托给 `torch.compile`。这让 RFGraph 更轻量，也更容易与现有 PyTorch 生态集成。

**Q3：动态路由（token 数量逐 step 变化）如何处理？**

RFGraph 的 `RFTensor.shape` 支持 `SymInt`（符号维度），对应 `torch.compile` 的 `dynamic=True` 模式。对于通信操作，token 数量的动态性体现在 `A2A_DISPATCH` 的 send_counts 参数上，RFTracer 会将其记录为 symbolic 变量，Scheduler 在图结构层面仍然是静态的，只是某些节点的执行时间估算需要依赖 runtime profile。

**Q4：Tile-level overlap（Layer 1）为什么不在 RFGraph 层面做，而在 Kernel 层面做？**

Tile 粒度（KB 量级）远小于 RFGraph 节点的调度粒度（操作级）。在图层面为每个 Tile 创建一个节点会让 RFGraph 膨胀到不可操作的规模（一个 Expert GEMM 可能有数千个 Tile）。因此 Layer 1 的 overlap 通过 HIP Wavefront 专用化 Kernel 实现，RFGraph 只负责标注「这个 GEMM 节点需要 Tile-level overlap」和配套的参数（tile_size、comm_stream），具体的 Wavefront 调度在 Kernel 内部完成。

**Q5：FSDP2 + FSEP + A2A 同时激活时如何避免 RCCL 死锁？**

三者同时活跃时，RCCL 可能因为进程组交叉导致死锁。RFGraph 的 DependencyBuilder 会为这类场景插入显式的 SyncNode 作为序列化屏障：FSDP AllGather → SyncNode → FSEP Re-layout A2A → SyncNode → MoE A2A。调度器会识别这种约束并保证顺序，代价是部分 overlap 机会丧失，但安全性优先。这是 Phase 2 需要深入分析的正确性问题。

---

## 9. 模块结构

```
rocflow/
  ir/
    rfgraph.py          # RFGraph、RFNode、RFTensor 数据结构定义
    rfnode_types.py     # RFOpType、HWPath、CommOpType 枚举
    rftracer.py         # RFTracer（torch.fx 扩展）
    rfhooktracer.py     # RFHookTracer（干运行 Hook）
    passes/
      node_classifier.py       # Pass 1：节点类型分类
      lifetime_analyzer.py     # Pass 2：Tensor lifetime 分析
      hwpath_assigner.py       # Pass 3：硬件路径分配
      dependency_builder.py    # Pass 4：依赖边构建
      block_annotator.py       # Pass 5：Block ID 标注

  scheduler/
    dag_scheduler.py    # DAG Scheduler 主逻辑
    stream_pool.py      # Stream 池管理
    tile_size_optimizer.py  # Tile 大小自动计算
    memory_scheduler.py     # Tensor lifetime 驱动的内存调度

  compiler/
    lowering.py         # RFGraph → Execution Plan 的 Lowering Pass 集合
    compute_grouper.py  # ComputeRegionGrouper Pass
    comm_generator.py   # CommCallGenerator Pass
    sync_inserter.py    # SyncPointInserter Pass
    tile_injector.py    # TileOverlapInjector Pass
    execution_plan.py   # ExecutionPlan callable

  kernels/
    expert_tile_kernel.hip  # HIP Wavefront 专用化 Expert Kernel
    gate_dispatch_fused.hip # Gate + Dispatch 融合 Kernel
```

---

## 10. 未解决问题（待后续讨论）

**IR-1：Backward 图的生成策略**  
当前设计描述的主要是 Forward 图。Backward 的 RFGraph 可以通过两种方式生成：
- 方案 A：在 Forward trace 时同步生成 backward DAG（类似 autograd engine 的反向图）
- 方案 B：运行时由 PyTorch autograd 生成，ROCflow 只 intercept 通信操作  
方案 A 调度更优但实现复杂；方案 B 实现简单但 backward overlap 能力受限。

**IR-2：FSEP Re-layout 如何表达为 RFGraph 节点**  
FSEP 的专家参数重布局是一个跨 Step 的操作（每 K step 触发一次）。它不属于单个 Step 的 RFGraph，需要一个独立的 Re-layout Graph，与正常训练 Graph 异步执行。接口设计待定。

**IR-3：PP（流水线并行）与 RFGraph 的集成**  
流水线并行的 micro-batch 调度（GPipe / PipeDream 风格）本身也是一个调度问题，与 DAG Scheduler 存在交互。需要决定 PP 调度是在 RFGraph 层面统一处理，还是在更高层独立处理后再展开进 RFGraph。

---

*ROCflow IR Design Document — 初稿于 2026-03-08*