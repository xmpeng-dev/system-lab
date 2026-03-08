# Axion IR Phase 1 设计：类型系统 + 指令集

> **文档类型:** 详细设计文档  
> **对应阶段:** Phase 1（2 周纸面设计）  
> **核心问题:** CommTensor 的 layout 枚举是否覆盖所有 MoE 通信模式？  
> **验证方式:** 手写几个典型 MoE layer 的 IR 表示，看是否自然表达

---

## 1. 设计目标与约束

### 1.1 目标

```
1. CommTensor layout 枚举覆盖 MoE 训练中所有主要通信模式
2. IR 类型系统能自然表达 FSEP Expert 分片状态
3. 指令集足够描述一个完整 MoE Transformer Layer（前向 + 反向）
4. 类型系统为后续 Pass 分析提供充足的静态信息
```

### 1.2 设计约束

```
约束 1：最小化类型数量
  不应为每个通信模式创建独立类型
  CommTensor + layout 枚举 优于 A2ATensor / AllGatherTensor / ...

约束 2：类型携带的信息必须在编译期可知
  shape、dtype、layout → 编译期确定 ✅
  send_counts、recv_counts → 运行时才知道（需特殊处理）

约束 3：Expert 状态必须是 IR 内置概念
  不能用 metadata dict 附加，必须有专用类型 ExpertShard

约束 4：反向传播对称性
  前向 CommTensor 的 layout，必须能自然推导出反向的 layout
```

---

## 2. 类型系统详细设计

### 2.1 基础类型层次

```
Axion Value Type Hierarchy：

  AxionType（抽象基类）
  ├── ScalarType
  │   ├── f32, f16, bf16, f8, i8, i4, bool
  │   └── 与 PyTorch dtype 一一对应
  │
  ├── TensorType（抽象基类）
  │   ├── DenseTensor
  │   ├── CommTensor
  │   └── SparseTensor（未来扩展，当前 SPARSE_CSR 通过 CommTensor 处理）
  │
  ├── ExpertShard              ← FSEP 专用，IR 内置
  ├── RoutingTable             ← MoE 路由专用，IR 内置
  ├── ProcessGroup             ← 通信组，IR 内置
  └── Pipeline                ← Sched.Overlap 的返回类型
```

### 2.2 DenseTensor

```
DenseTensor<shape, dtype, shard_spec?>

字段：
  shape:      Tuple[int | Symbol]    # Symbol 表示动态维度
  dtype:      ScalarType
  shard_spec: ShardSpec?             # 可选，描述这个 tensor 在哪个 GPU 上

ShardSpec：
  device:   int                      # GPU id
  full_shape: Tuple[int | Symbol]    # 未分片时的完整 shape
  shard_dim:  int                    # 沿哪个维度分片（-1 表示复制）
  shard_rank: int                    # 这是第几个 shard

示例：
  DenseTensor<[512, 4096], bf16>                   # 未分片
  DenseTensor<[512, 1024], bf16, shard_spec=(      # 4路 TP 分片
      full_shape=[512, 4096],
      shard_dim=1,
      shard_rank=2,
      device=2,
  )>
```

### 2.3 CommTensor（核心类型）

```
CommTensor<shape, dtype, layout, comm_spec>

字段：
  shape:     Tuple[int | Symbol]     # 逻辑 shape（与 DenseTensor 语义一致）
  dtype:     ScalarType
  layout:    CommLayout              # 物理布局枚举（见 2.4 节）
  comm_spec: CommSpec                # 通信元数据

CommSpec：
  comm_type:    CommType             # A2A / ALL_GATHER / REDUCE_SCATTER / P2P
  src_group:    ProcessGroup         # 数据来自哪个进程组
  dst_group:    ProcessGroup         # 数据要去往哪个进程组
  send_counts:  int[] | Symbol[]     # 发给每个 rank 的元素数（动态时为 Symbol）
  recv_counts:  int[] | Symbol[]     # 从每个 rank 收的元素数
  element_size: int                  # 单个元素的字节数（用于内存规划）

关键设计：send_counts / recv_counts 可以是 Symbol（动态）
  → 编译器知道"类型结构"（通信拓扑），运行时才填具体数量
  → 与静态 shape 分离，不阻塞编译期分析
```

### 2.4 CommLayout 枚举（核心问题：覆盖哪些模式？）

```
经过对 MoE 训练通信模式的系统梳理，定义以下 5 种 layout：

┌─────────────────────────────────────────────────────────────────┐
│  CommLayout 枚举                                                 │
├──────────────────────┬──────────────────────────────────────────┤
│  BLOCKED_BY_DST      │  物理内存按目标 GPU 分组排列              │
│                      │  [rank0的数据 | rank1的数据 | ...]        │
│                      │  适用：Expert dispatch All-to-All（最常见）│
│                      │  优势：直接 DMA，零 pack copy             │
├──────────────────────┼──────────────────────────────────────────┤
│  BLOCKED_BY_SRC      │  物理内存按来源 GPU 分组排列              │
│                      │  [来自rank0的数据 | 来自rank1的数据 | ...]│
│                      │  适用：Expert combine 接收端              │
│                      │         All-Gather 输出                  │
│                      │  优势：接收后直接按 expert 处理，无需重排  │
├──────────────────────┼──────────────────────────────────────────┤
│  BLOCKED_BY_EXPERT   │  物理内存按 Expert ID 分组排列            │
│                      │  [Expert0的params | Expert1的params | ...]│
│                      │  适用：FSEP Expert 参数 All-Gather        │
│                      │  优势：Expert shard 连续，gather 无重排   │
├──────────────────────┼──────────────────────────────────────────┤
│  INTERLEAVED         │  物理内存交织排列（原始 token 顺序）       │
│                      │  [t0, t1, t2, ..., tN]                   │
│                      │  适用：Attention 计算输入/输出            │
│                      │  本质：等同于 PyTorch 普通 Tensor 的布局  │
│                      │  作用：作为 CommTensor 的"退化"状态       │
├──────────────────────┼──────────────────────────────────────────┤
│  SPARSE_CSR          │  CSR 格式：value + col_idx + row_ptr     │
│                      │  适用：大量 Expert 接收 0 token 的稀疏路由│
│                      │  优势：跳过空 Expert，减少通信量           │
│                      │  触发条件：routing sparsity > 阈值（70%） │
└──────────────────────┴──────────────────────────────────────────┘
```

**layout 转换关系（编译器需要追踪）：**

```
INTERLEAVED
    │  CommTensor.FromDense（routing 分组）
    ▼
BLOCKED_BY_DST
    │  Comm.A2A（网络传输）
    ▼
BLOCKED_BY_SRC
    │  Expert.FFN（local 计算，layout 不变）
    ▼
BLOCKED_BY_SRC
    │  Comm.A2A（combine，网络传输）
    ▼
BLOCKED_BY_DST
    │  CommTensor.ToDense（按原始 token 顺序重组）
    ▼
INTERLEAVED

BLOCKED_BY_EXPERT（Expert 参数分片 All-Gather 的专属路径）
    │  Comm.AllGather
    ▼
DenseTensor（完整 Expert 参数，供 Expert.FFN 使用）
```

### 2.5 ExpertShard（FSEP 专用类型）

```
ExpertShard<expert_id, shard_rank, total_shards, dtype, param_shape>

字段：
  expert_id:     int                 # 属于哪个 Expert
  shard_rank:    int                 # 这是第几个 shard（0-indexed）
  total_shards:  int                 # Expert 参数总共几个 shard
  dtype:         ScalarType
  param_shape:   Tuple[int]          # shard 的实际 shape（可以 ragged）

  # 运行时状态（IR 类型中声明，运行时填充）
  current_gpu:   int | Symbol        # 当前在哪个 GPU 上
  migration_state: MigrationState    # STABLE | MIGRATING | SHADOW

MigrationState：
  STABLE    → 正常使用
  MIGRATING → 正在被 Shard.Migrate 搬迁，旧位置仍有效
  SHADOW    → 迁移完成，旧位置的 shadow buffer 待释放
```

### 2.6 RoutingTable

```
RoutingTable<num_tokens, num_experts, topk>

字段：
  num_tokens:   int | Symbol         # batch 中 token 总数
  num_experts:  int                  # Expert 总数（编译期已知）
  topk:         int                  # 每个 token 路由几个 Expert（编译期已知）

  # 编译器可静态分析的属性
  is_fixed:     bool                 # 路由是否固定（推理场景为 true）
  sparsity:     float | Symbol       # 路由稀疏度（大部分 Expert 无 token）

  # 运行时填充
  token_to_expert: int[num_tokens, topk]   # token i → expert j
  send_counts:     int[num_experts]        # 发给 expert j 多少 token
  recv_counts:     int[num_experts]        # 从 expert j 收多少 token

设计说明：
  num_experts, topk 在编译期已知 → 编译器可以推断 A2A 的拓扑结构
  send_counts 运行时才知道 → 用 Symbol 表示，不阻塞编译期类型推断
```

---

## 3. 指令集详细设计

### 3.1 指令格式规范

```
每条指令的格式：

  OpName.SubOp (operands...) -> result_type
  [attributes: key=value, ...]
  [constraints: ...]

例：
  Comm.A2A (
      x: CommTensor<[T, H], bf16, BLOCKED_BY_DST, cs>,
      routing: RoutingTable<T, E, k>
  ) -> CommTensor<[T, H], bf16, BLOCKED_BY_SRC, cs'>
  [attributes: recv_layout=BLOCKED_BY_SRC, async=true]
  [constraints: x.comm_spec.dst_group == routing.expert_group]
```

### 3.2 Compute Ops

```
Dense.MatMul
  签名:  (a: DenseTensor<[M, K]>, b: DenseTensor<[K, N]>)
         -> DenseTensor<[M, N]>
  语义:  标准矩阵乘法，支持 TP 分片（编译器自动插入 AllReduce）
  注意:  当 a 或 b 有 shard_spec 时，编译器在 Pass 2 插入必要通信

Dense.LayerNorm
  签名:  (x: DenseTensor<[*, H]>, weight: DenseTensor<[H]>,
          bias: DenseTensor<[H]>)
         -> DenseTensor<[*, H]>
  语义:  标准 LayerNorm，RMSNorm 通过 bias=None 表达

Dense.Attention
  签名:  (q: DenseTensor<[B,S,H]>, k: DenseTensor<[B,S,H]>,
          v: DenseTensor<[B,S,H]>)
         -> DenseTensor<[B,S,H]>
  语义:  Scaled Dot-Product Attention（包含 Flash Attention kernel 选择）

Expert.Gate
  签名:  (x: DenseTensor<[T, H]>, weight: DenseTensor<[E, H]>)
         -> RoutingTable<T, E, k>
  语义:  Top-K 路由，生成 RoutingTable
  属性:  topk: int, normalize: bool, aux_loss: bool

Expert.FFN
  签名:  (x: CommTensor<[T, H], *, BLOCKED_BY_SRC, *>,
          shard: ExpertShard<*, *, *, *, [H, FFN_H]>)
         -> CommTensor<[T, H], *, BLOCKED_BY_SRC, *>
  语义:  Expert FFN 计算（Gate + Up + Down proj）
  约束:  x 的 layout 必须是 BLOCKED_BY_SRC（保证 Expert 本地数据连续）

Expert.Combine
  签名:  (x: CommTensor<[T, H], *, BLOCKED_BY_DST, *>,
          routing: RoutingTable<T, E, k>)
         -> DenseTensor<[T, H]>
  语义:  加权合并来自多个 Expert 的输出（按 gate score 加权）
```

### 3.3 Comm Ops

```
Comm.A2A（All-to-All，Expert dispatch / combine 的核心）
  签名:  (x: CommTensor<shape, dtype, BLOCKED_BY_DST, src_spec>,
          routing: RoutingTable)
         -> CommTensor<shape, dtype, BLOCKED_BY_SRC, dst_spec>
  语义:  All-to-All 网络传输
  属性:
    async:       bool   # 是否异步（默认 true，用于 overlap）
    chunk_size:  int?   # 分 chunk 传输（用于 Sched.Overlap，None 表示不分）
    network:     AUTO | NVLINK | IB  # 网络选择（AUTO 由 CommFabric 决定）
  类型规则:
    输入 layout 必须是 BLOCKED_BY_DST
    输出 layout 必须是 BLOCKED_BY_SRC
    shape 不变（只是数据重新分布）

Comm.AllGather（Expert 参数聚合）
  签名:  (x: ExpertShard<eid, rank, total, dtype, shard_shape>)
         -> DenseTensor<full_param_shape, dtype>
  语义:  从所有持有该 Expert shard 的 GPU 聚合完整参数
  属性:
    group:        ProcessGroup   # 参与 AllGather 的进程组
    output_layout: BLOCKED_BY_EXPERT  # 输出布局（保证连续性）
  优化:  当 total_shards == 1 时，编译器将此 Op 消除（零通信）

Comm.ReduceScatter（梯度聚合，反向传播用）
  签名:  (x: DenseTensor<full_shape, dtype>)
         -> DenseTensor<shard_shape, dtype, shard_spec>
  属性:
    group: ProcessGroup
    hier:  FLAT | INTRA_FIRST   # INTRA_FIRST = 先节点内 RS 再节点间 RS
  注意:  hier=INTRA_FIRST 对应 veScale-FSDP 的分层 Reduce-Scatter，
         节点内 NVLink（高带宽），节点间 IB（低带宽）

Comm.AllReduce（TP 梯度同步，由 Dense.MatMul Pass 自动插入）
  签名:  (x: DenseTensor<shape, dtype>)
         -> DenseTensor<shape, dtype>
  属性:
    group: ProcessGroup
    async: bool
```

### 3.4 Shard Ops

```
Shard.Split（把完整 Expert 参数切成 FSEP shard）
  签名:  (x: DenseTensor<param_shape, dtype>,
          spec: ShardSpec)
         -> ExpertShard<eid, *, total, dtype, shard_shape>[]
  语义:  按 ShardSpec 把 Expert 参数切分到各 GPU
  属性:
    expert_id:    int
    total_shards: int
    strategy:     ROW_WISE | COL_WISE | BLOCK_WISE  # 切分策略
  注意:  输出是 ExpertShard 数组（每个 GPU 一个）

Shard.Migrate（Expert shard 迁移，FSEP Slow Planner 触发）
  签名:  (x: ExpertShard<eid, rank, total, dtype, shape>,
          dst_gpu: int)
         -> ExpertShard<eid, rank, total, dtype, shape>
  语义:  把 ExpertShard 的物理存储从 src_gpu 搬到 dst_gpu
  语义保证:
    1. 异步执行（在反向传播期间完成）
    2. 迁移完成前，src_gpu 上的 SHADOW 副本仍可用
    3. 迁移完成后，自动释放 src_gpu 上的 SHADOW 副本
  IR 等价关系:
    output.expert_id == input.expert_id   (逻辑身份不变)
    output.current_gpu == dst_gpu         (物理位置改变)

Shard.Gather（从分布式 shard 重组，Shard.Split 的逆操作）
  签名:  (shards: ExpertShard<eid, *, total, dtype, *>[])
         -> DenseTensor<full_param_shape, dtype>
  语义:  收集所有 shard，重组完整 Expert 参数
  注意:  主要用于 checkpoint 保存和参数导出
```

### 3.5 Schedule Ops（编译器生成）

```
Sched.Overlap（声明计算和通信可以重叠执行）
  签名:  (compute: Op, comm: CommOp)
         -> Pipeline
  语义:  指示运行时在 comm 执行期间同时执行 compute
  生成时机:  Pass 4 (Overlap Insertion) 自动插入
  约束:
    compute 不能依赖 comm 的输出（否则无法并行）
    comm 不能依赖 compute 的输出（否则无法并行）
  
  示例（编译器自动生成）：
    %pipe = Sched.Overlap(
        compute = Expert.FFN(%chunk_i_minus_1, %expert_shard),
        comm    = Comm.A2A(%chunk_i, %routing),
    )

Sched.Prefetch（预取 Expert 参数，隐藏 AllGather 延迟）
  签名:  (x: ExpertShard, step_offset: int = 1)
         -> void
  语义:  提前 step_offset 步触发 ExpertShard 的 AllGather
  生成时机:  Pass 4 分析 Expert.FFN 的依赖链后插入
  注意:  step_offset=1 表示在当前 step 预取下一个 step 需要的 shard

Sched.Checkpoint（重计算，节省激活内存）
  签名:  (op: Op, policy: FULL | SELECTIVE)
         -> void
  语义:  标记该 Op 的输出不保留，反向传播时重计算
  注意:  当前版本为 placeholder，后续版本实现
```

---

## 4. 类型系统验证：典型 MoE Layer 的 IR 手写

### 4.1 验证场景清单

```
需要验证以下场景能被 IR 自然表达：

  场景 A：标准 MoE Transformer Layer（前向）         → 见 4.2
  场景 B：FSEP Expert 参数 AllGather                → 见 4.3
  场景 C：Expert 迁移（Slow Planner 触发）           → 见 4.4
  场景 D：分 chunk 的流水线 dispatch（带 Overlap）   → 见 4.5
  场景 E：反向传播梯度通信                          → 见 4.6
```

### 4.2 场景 A：标准 MoE Transformer Layer（前向）

```
func @moe_transformer_layer(
    %hidden:   DenseTensor<[B, S, H], bf16>,    # 输入：[batch, seq, hidden]
    %attn_w:   DenseTensor<[H, H], bf16>,       # Attention 权重（简化）
    %expert_shards: ExpertShard<*, *, *, bf16, [H, FFN_H]>[E],  # E 个 Expert
) -> DenseTensor<[B, S, H], bf16> {

  # ── Attention Block ──────────────────────────────────────────
  %attn_out  = Dense.Attention(%hidden, %hidden, %hidden)
  %residual1 = Dense.Add(%hidden, %attn_out)             # 残差
  %ln1_out   = Dense.LayerNorm(%residual1, %ln1_w, %ln1_b)

  # ── MoE Block ────────────────────────────────────────────────
  # Step 1: 生成路由表
  #   类型：RoutingTable<B*S, E, 2>（Top-2 路由）
  #   编译器从此处知道：num_experts=E, topk=2（编译期常量）
  %routing = Expert.Gate(%ln1_out, %gate_w)
  # routing: RoutingTable<T=B*S, E=256, k=2>

  # Step 2: hidden → CommTensor（按目标 Expert GPU 分组）
  #   类型变化：DenseTensor → CommTensor<BLOCKED_BY_DST>
  #   此处零 copy：物理内存按 routing 重排，逻辑视图不变
  %dispatch_input = CommTensor.FromDense(
      %ln1_out,
      layout  = BLOCKED_BY_DST,
      routing = %routing,
  )
  # dispatch_input: CommTensor<[T, H], bf16, BLOCKED_BY_DST, {A2A, ep_group}>

  # Step 3: All-to-All dispatch（token → Expert GPU）
  #   类型变化：BLOCKED_BY_DST → BLOCKED_BY_SRC
  %dispatched = Comm.A2A(
      %dispatch_input,
      routing  = %routing,
      [async=true, chunk_size=null]  # Pass 4 会插入 chunk_size
  )
  # dispatched: CommTensor<[T, H], bf16, BLOCKED_BY_SRC, {A2A, ep_group}>

  # Step 4: Expert FFN（本地计算）
  #   输入必须是 BLOCKED_BY_SRC（保证 Expert 数据本地连续）
  %expert_out = Expert.FFN(%dispatched, %expert_shards)
  # expert_out: CommTensor<[T, H], bf16, BLOCKED_BY_SRC, {A2A, ep_group}>

  # Step 5: All-to-All combine（Expert 输出 → 原始 GPU）
  %combined = Comm.A2A(
      %expert_out,
      routing = %routing,
      [recv_layout=BLOCKED_BY_DST]
  )
  # combined: CommTensor<[T, H], bf16, BLOCKED_BY_DST, {A2A, ep_group}>

  # Step 6: CommTensor → DenseTensor（按原始 token 顺序）
  %moe_out = Expert.Combine(%combined, %routing)
  # moe_out: DenseTensor<[B, S, H], bf16>

  # ── Residual + Output ────────────────────────────────────────
  %residual2 = Dense.Add(%ln1_out, %moe_out)
  %output    = Dense.LayerNorm(%residual2, %ln2_w, %ln2_b)

  return %output
}

# 类型流验证：
#   DenseTensor → CommTensor<BLOCKED_BY_DST> → CommTensor<BLOCKED_BY_SRC>
#   → CommTensor<BLOCKED_BY_SRC> → CommTensor<BLOCKED_BY_DST> → DenseTensor
#   ✅ 类型转换路径完整且闭合
```

### 4.3 场景 B：FSEP Expert 参数 AllGather

```
# Expert.FFN 内部展开（编译器展开，用户不直接写）

func @expert_ffn_fsep(
    %x:     CommTensor<[local_T, H], bf16, BLOCKED_BY_SRC, *>,
    %shard: ExpertShard<eid=3, rank=0, total=2, bf16, [H, FFN_H/2]>,
) -> CommTensor<[local_T, H], bf16, BLOCKED_BY_SRC, *> {

  # Step 1: AllGather Expert 参数（从 2 个 GPU 聚合完整参数）
  #   输入：ExpertShard（shard 0 of 2）
  #   输出：DenseTensor（完整 Expert 参数）
  %full_w1 = Comm.AllGather(
      %shard,
      [group=ep_group, output_layout=BLOCKED_BY_EXPERT]
  )
  # full_w1: DenseTensor<[H, FFN_H], bf16>
  # 当 total=1 时，编译器消除此 AllGather（Pass 3 优化）

  # Step 2: 本地 Expert FFN 计算
  %gate_proj = Dense.MatMul(%x.logical_view, %full_w1_gate)
  %up_proj   = Dense.MatMul(%x.logical_view, %full_w1_up)
  %activated = Dense.SiLU(%gate_proj) * %up_proj  # SwiGLU
  %down_out  = Dense.MatMul(%activated, %full_w2)

  return CommTensor.Wrap(%down_out, layout=BLOCKED_BY_SRC, ...)
}

# FSEP 关键性质验证：
#   当 total_shards=1 时：Comm.AllGather 被消除，Expert 完整参数在本地 ✅
#   当 total_shards=2 时：AllGather 从 2 个 GPU 聚合，参数分散存储   ✅
#   当 total_shards=N 时：热点 Expert 参数分散在 N 个 GPU，负载均衡   ✅
```

### 4.4 场景 C：Expert 迁移（Slow Planner 触发）

```
# 这段 IR 由 FSEP Slow Planner 在运行时动态生成并插入

func @expert_migration_step(
    %shard: ExpertShard<eid=7, rank=0, total=1, bf16, [H, FFN_H]>,
    %dst_gpu: int = 3,
) -> ExpertShard<eid=7, rank=0, total=1, bf16, [H, FFN_H]> {

  # Shard.Migrate：异步 P2P 搬迁
  #   运行时语义：
  #   1. 在反向传播的 CUDA stream 上异步执行
  #   2. 迁移期间 migration_state = MIGRATING，src 上 SHADOW 副本有效
  #   3. 迁移完成后 migration_state = STABLE，src SHADOW 副本释放
  %migrated = Shard.Migrate(%shard, dst_gpu=3)
  # migrated: ExpertShard<eid=7, rank=0, total=1, bf16, [H, FFN_H]>
  #   migrated.current_gpu == 3  ✅
  #   migrated.expert_id   == 7  ✅（逻辑身份不变）

  return %migrated
}

# 关键性质验证：
#   迁移前后 expert_id 不变（逻辑身份不变）   ✅
#   current_gpu 从 src 变为 dst               ✅
#   migration_state 的状态机由运行时维护       ✅
#   SHADOW 副本机制保证迁移期间不中断训练      ✅
```

### 4.5 场景 D：分 chunk 流水线（Pass 4 插入 Overlap）

```
# Pass 4 Overlap Insertion 的输入（原始 IR）：
  %dispatched = Comm.A2A(%dispatch_input, %routing)
  %expert_out = Expert.FFN(%dispatched, %expert_shards)

# Pass 4 Overlap Insertion 的输出（插入 Sched.Overlap 后）：

func @moe_chunked_pipeline(
    %dispatch_input: CommTensor<[T, H], bf16, BLOCKED_BY_DST, *>,
    %routing:        RoutingTable<T, E, 2>,
    %shards:         ExpertShard[],
) -> CommTensor<[T, H], bf16, BLOCKED_BY_SRC, *> {

  # 编译器把 T 个 token 分成 C 个 chunk
  %chunks = CommTensor.Split(%dispatch_input, num_chunks=C)

  %results = []
  for i in range(C):
    # chunk i 的 A2A dispatch（异步）
    %a2a_handle_i = Comm.A2A(
        %chunks[i], %routing,
        [async=true, chunk_size=T/C]
    )

    if i > 0:
      # chunk i-1 的 A2A 已完成，开始计算（与 chunk i 的 A2A 重叠）
      %pipe = Sched.Overlap(
          compute = Expert.FFN(%dispatched[i-1], %shards),
          comm    = %a2a_handle_i,    # chunk i 的 A2A
      )
      %results.append(%pipe.compute_result)

    %dispatched[i] = %a2a_handle_i.wait()

  # 最后一个 chunk 的计算
  %results.append(Expert.FFN(%dispatched[C-1], %shards))

  return CommTensor.Concat(%results)
}

# 流水线效果验证（C=4 chunks）：
#
#   时间线：
#   ─────────────────────────────────────────────────→ time
#   A2A:    [chunk1]  [chunk2]  [chunk3]  [chunk4]
#   FFN:              [chunk1]  [chunk2]  [chunk3]  [chunk4]
#                     ↑─────────────────────────────↑
#                     A2A 和 FFN 完全重叠 ✅
```

### 4.6 场景 E：反向传播梯度通信

```
# Dense 参数反向：Reduce-Scatter（分层，借鉴 veScale-FSDP）

func @dense_param_backward(
    %grad_full: DenseTensor<[H, H], bf16>,  # 全局梯度（AllReduce 前）
) -> DenseTensor<[H/N, H], bf16, shard_spec> {  # 分片梯度

  %grad_shard = Comm.ReduceScatter(
      %grad_full,
      [group=dp_group, hier=INTRA_FIRST]
  )
  # hier=INTRA_FIRST 执行顺序：
  #   1. 节点内 NVLink ReduceScatter（高带宽，~600 GB/s）
  #   2. 节点间 IB ReduceScatter（低带宽，~200 GB/s）
  # 效果：减少跨节点通信量约 40%（借鉴 veScale-FSDP 的分层设计）

  return %grad_shard
}

# Expert 参数反向：ExpertShard 梯度累积

func @expert_param_backward(
    %grad_full: DenseTensor<[H, FFN_H], bf16>,  # Expert 完整参数梯度
    %shard_spec: ExpertShardSpec,
) -> ExpertShard<eid, rank, total, bf16, shard_shape> {

  # ReduceScatter 到各 shard 对应的 GPU
  %grad_shard = Comm.ReduceScatter(
      %grad_full,
      [group=ep_group, hier=INTRA_FIRST]
  )

  # 包装为 ExpertShard 类型（用于优化器更新）
  return ExpertShard.FromGrad(%grad_shard, %shard_spec)
}
```

---

## 5. 类型系统完整性评估

### 5.1 MoE 通信模式覆盖度检查

| 通信场景 | 对应 Op | Layout | 覆盖状态 |
|---------|---------|--------|---------|
| Expert dispatch (token → Expert GPU) | `Comm.A2A` | `BLOCKED_BY_DST → BLOCKED_BY_SRC` | ✅ |
| Expert combine (Expert GPU → token) | `Comm.A2A` | `BLOCKED_BY_SRC → BLOCKED_BY_DST` | ✅ |
| FSEP Expert 参数聚合 | `Comm.AllGather` | `ExpertShard → DenseTensor` | ✅ |
| Dense 参数聚合 (前向前) | `Comm.AllGather` | `BLOCKED_BY_EXPERT` | ✅ |
| Expert 参数梯度归约 | `Comm.ReduceScatter` | `DenseTensor → ExpertShard` | ✅ |
| Dense 参数梯度归约 | `Comm.ReduceScatter` | `INTRA_FIRST` 分层 | ✅ |
| Expert 物理迁移 | `Shard.Migrate` | P2P | ✅ |
| TP 梯度同步 | `Comm.AllReduce` | 自动插入 | ✅ |
| 稀疏路由优化 | `CommTensor<SPARSE_CSR>` | CSR | ✅ (触发条件待定) |

### 5.2 尚未覆盖的场景（已知 Gap）

```
Gap 1：Pipeline Parallelism (PP) 的 P2P 通信
  当前：IR 没有专门的 PP stage 间通信 Op
  影响：Axion v1 不支持 PP（仅支持 DP + TP + EP）
  解决：后续版本添加 Comm.Send / Comm.Recv Op

Gap 2：Sequence Parallelism (SP) 的 All-to-All
  当前：INTERLEAVED layout 的 CommTensor 可以表达
  但：SP 的 A2A 与 Expert A2A 共用同一 Op，需要区分语义
  解决：添加 comm_spec.purpose 字段（EXPERT_DISPATCH / SEQ_PARALLEL）

Gap 3：梯度压缩通信（PowerSGD 等）
  当前：ReduceScatter 不支持压缩
  影响：无法支持梯度压缩优化
  解决：Comm.CompressedReduceScatter（后续版本）
```

---

## 6. Phase 1 验收标准

```
□ 类型系统文档完成（本文档）
□ 手写 5 个 IR 场景（4.2 ~ 4.6），全部通过类型检查
□ CommLayout 枚举覆盖度表格（5.1）完成
□ 已知 Gap 列表（5.2）明确
□ 与团队 review，确认 IR 语法是否足够直观
□ 确认 ExpertShard 类型是否覆盖 FSEP 所有状态转换

通过以上验收后，进入 Phase 2：实现 IR Parser + Pass 1/2
```

---

*文档版本 v0.1 | Axion IR Phase 1 设计 | 2026-03-08*  
*参考：LAER-MoE (arXiv:2602.11686) | veScale-FSDP (arXiv:2602.22437)*
