# MoEX Overlap 策略设计

> MoEX 实现三层 overlap：
> - Layer 1（最细粒度）：Tile 级 GEMM-RDMA pipeline（Comet 风格，kernel 内部）
> - Layer 2（中粒度）：Block 级 Dispatch/Combine 与 Expert GEMM 解耦
> - Layer 3（跨层）：跨 MoE Layer 的 DAG 调度（FlowMoE 风格）

---

## 目录

1. [三层 Overlap 架构总览](#1-三层-overlap-架构总览)
2. [Layer 1：Tile 级 GEMM-RDMA Pipeline](#2-layer-1tile-级-gemm-rdma-pipeline)
3. [Layer 2：Block 级 Dispatch/Combine 解耦](#3-layer-2block-级-dispatchcombine-解耦)
4. [Layer 3：跨层 DAG 调度](#4-layer-3跨层-dag-调度)
5. [Overlap 调度器设计](#5-overlap-调度器设计)
6. [三层 Overlap 的协同](#6-三层-overlap-的协同)
7. [CommTensor 对 Overlap 的贡献](#7-commtensor-对-overlap-的贡献)
8. [性能分析与论文对比](#8-性能分析与论文对比)

---

## 1. 三层 Overlap 架构总览

```
MoEX 三层 Overlap 架构

┌────────────────────────────────────────────────────────────────────────┐
│ Layer 3: 跨层 DAG 调度（FlowMoE 风格，micro-ms 粒度）                  │
│                                                                         │
│  Layer N:  ┌──Route──┐ ┌─Dispatch─┐      ┌─Expert─┐ ┌─Combine─┐       │
│            └─────────┘ └──────────┘      └────────┘ └─────────┘       │
│                                    ↕ overlap                            │
│  Layer N+1:           ┌──Route──┐ ┌─Dispatch─┐      ┌─Expert─┐        │
│                        └─────────┘ └──────────┘      └────────┘        │
├────────────────────────────────────────────────────────────────────────┤
│ Layer 2: Block 级 Dispatch/Combine 解耦（µs 粒度）                      │
│                                                                         │
│  Compute Stream: ──[Expert GEMM]──────────────────────────────────      │
│  Comm Stream D:  ──[A2A D]────[wait]────                               │
│  Comm Stream C:                      ────[A2A C]────                   │
│                                                                         │
│  关键：Expert GEMM 掩盖 Dispatch 的等待时间                             │
├────────────────────────────────────────────────────────────────────────┤
│ Layer 1: Tile 级 GEMM-RDMA Pipeline（ns~µs 粒度）                      │
│                                                                         │
│  Compute Warps: [GEMM tile0]─[GEMM tile1]─[GEMM tile2]─...            │
│  Comm Warps:              ─[RDMA tile0]─[RDMA tile1]─...              │
│                                                                         │
│  即完即发：tile GEMM 结束 → 立即 RDMA send（< 1µs 延迟）               │
└────────────────────────────────────────────────────────────────────────┘

综合效果目标：
  通信延迟 visible = 总通信时间 × (1 - 0.92) = 8% 可见
  vs. 传统实现：30-40% 通信时间可见
  → 4-5× 通信开销减少
```

---

## 2. Layer 1：Tile 级 GEMM-RDMA Pipeline

### 2.1 设计基础（来自 Comet MLSys'25）

Comet 论文的核心发现：
- 将 Expert GEMM 切分为 KB 级 tile，每个 tile 完成后立即 RDMA
- 相比 chunk 级（MB 级）overlap，overlap 率从 68% 提升到 90%+
- 关键：RDMA 的"即完即发"消除了等待全量 GEMM 完成的等待时间

MoEX 的扩展：CommTensor 的 T 维度直接对应 tile，**tile 边界天然存在于 tensor 结构中**，
无需动态切分或 index 计算。

### 2.2 Warp 专化机制

```
GPU Warp 分配（以 H100 SM 为例）：
  每个 SM：64 个 Warp slots
  分配方案：
    Compute Warps：51 warps（80%）→ 执行 GEMM tile 计算
    Comm Warps：13 warps（20%）→ 监控完成，触发 RDMA

协调机制（shared memory）：
  // 在 Shared Memory 中
  __shared__ volatile int tile_done[NUM_TILES];  // 0=未完成，1=已完成
  __shared__ void* tile_rdma_addr[NUM_TILES];    // tile 的 RDMA 目标地址

// Compute Warp 伪代码
for tile_id in compute_warp_tiles:
    // 1. 执行 GEMM tile
    result = gemm_tile(input_ct.view_tile(r, s, tile_id), W_expert)
    
    // 2. 写入输出 CommTensor
    output_ct.data[r, s, tile_id, :] = result
    
    // 3. 通知 Comm Warp（内存屏障确保顺序）
    __threadfence()  // 确保 result 写入全局内存
    tile_done[tile_id] = 1  // 原子标记

// Comm Warp 伪代码
for tile_id in comm_warp_tiles:
    // 等待 GEMM 完成（自旋，极短等待）
    while tile_done[tile_id] == 0:
        __threadfence()  // 内存屏障
    
    // 立即 RDMA send（one-sided PUT）
    rdma_put_async(
        src = output_ct.data.ptr + tile_offset(r, s, tile_id),
        size = H * element_size,
        dst_rank = original_rank(r, s),
        dst_offset = ...,
    )
    // 不等待 RDMA 完成！继续处理下一个 tile
```

### 2.3 CommTensor 对 Tile 级 Overlap 的贡献

```
传统实现（无 CommTensor）：

tile GEMM 完成 → 写入全局内存（HBM）→ RDMA 读取 HBM → 发送
  额外延迟：HBM 写 + HBM 读 ≈ 2 × (tile_size * H * 2) / HBM_BW
  = 2 × (128 * 4096 * 2) / (3350 GB/s) ≈ 0.63µs per tile

MoEX CommTensor：

tile GEMM 完成 → 写入 CommTensor.data[r, s, tile_id, :]
→ Comm Warp 读取 commTensor.data 地址（O(1) 算术）
→ RDMA 直接从该地址发送（可能 bypass HBM，走 L2 cache → NIC）
  
优化：如果 tile 结果还在 L2 cache 中（tile_size × H × 2 = 1 MB，恰好 < L2 大小），
RDMA 可以从 L2 直接读取，进一步减少延迟。

H100 L2 cache：50 MB
Tile 大小：128 × 4096 × 2 = 1 MB
→ 每个 tile 结果完全在 L2 中 ✓
→ RDMA 延迟从 HBM 读取变为 L2 读取：节省 ~2-5× 延迟
```

### 2.4 Tile 大小选择策略

```
Tile 大小权衡：

太小的 tile（e.g., T=32）：
  优点：overlap 粒度细，更快触发 RDMA
  缺点：GEMM 效率低（GPU 利用率下降），Warp 调度开销增加
  适用：高带宽网络（NVLink，RDMA 延迟 < 1µs）

适中的 tile（e.g., T=128，H100 推荐）：
  GEMM tile 128×128：充分利用 H100 Tensor Core（128×128 是最优 MMA shape）
  RDMA 粒度：128 × 4096 × 2 = 1 MB（适中）
  Comm Warp 触发频率：4096 / 128 = 32 次/token（合理）

较大的 tile（e.g., T=512）：
  优点：GEMM 效率高
  缺点：overlap 粒度粗（退化为 chunk 级），overlap 率下降
  适用：低带宽网络（IB，RDMA 延迟 > 5µs，粗粒度无损）

MoEX 默认：T=128（H100），可配置
AMD MI300X 最优：T=256（MI300X Tensor Core 最优 shape 更大）
```

---

## 3. Layer 2：Block 级 Dispatch/Combine 解耦

### 3.1 双流调度模型

```
传统串行执行：

时刻:  0    1    2    3    4    5    6    7    8    9   10
       ↓    ↓    ↓    ↓    ↓    ↓    ↓    ↓    ↓    ↓    ↓
       [Route] [────Dispatch────] [────Expert────] [──Combine──]
              ← 30-40% →         ← 40-50% →       ← 20-30% →

Expert GEMM 无法开始直到 Dispatch All-to-All 完全结束
```

```
MoEX 双流执行：

Compute Stream: [Route] ───────────── wait ─[Expert GEMM, tile 0..N]──────── [Combine]
                         ↑                 ↑
Comm Stream D:  ──────── [Dispatch A2A] ──→ （通知 compute stream 可以开始）
                                          ↓
Comm Stream C:                            [Combine A2A]─────────────────────→

关键：
1. Route 在 Compute Stream 执行（Gate GEMM）
2. Dispatch 在 Comm Stream D 执行，**并行** 于下一个 micro-batch 的 Attention
3. Expert GEMM 等待 Dispatch 完成（event wait），但等待窗口被其他计算填充
4. Combine 在 Comm Stream C 执行，Expert GEMM 同时也在执行（tile 级 overlap）
```

### 3.2 CommTensor 对 Layer 2 的贡献

```
传统 Dispatch 前置工作（阻塞 comm_stream 的时间）：
  1. permute（拷贝）：需要在 compute_stream 完成，才能通知 comm_stream
  2. pack（拷贝）：在 comm_stream 执行，但需要 compute_stream 的 sorted_hidden
  总前置时间：permute_time + pack_time ≈ 0.28ms（见 CommTensor_Design.md 分析）

MoEX CommTensor：
  - Route 完成后，CommTensor 已经是"dispatch 就绪"状态
  - comm_stream 可以立即开始 RDMA PUT
  - 节省 0.28ms 的前置准备时间

这 0.28ms 的节省意味着：
  - comm_stream 更早开始 → Expert GEMM 更早开始 → 整体更低延迟
  - overlap 窗口从 0.28ms 之后变为立即开始 → 实际 overlap 率更高
```

### 3.3 Micro-batch 流水线

```
PP + Micro-batch 场景（标准 1F1B，MegatronCore 风格）：

时刻:  0    1    2    3    4    5    6    7    8
       ↓    ↓    ↓    ↓    ↓    ↓    ↓    ↓    ↓
MB 0:  [Att]──[MoE: Route+Dispatch]──[Expert]──[Combine]
MB 1:        [Att]──[MoE: Route+Dispatch]──[Expert]──[Combine]
MB 2:               [Att]──[MoE: Route+Dispatch]──[Expert]──[Combine]

关键观察：
  MB 0 的 [MoE Dispatch] 与 MB 1 的 [Att] 是独立的！
  → Attention 和 Dispatch 可以并行（不同 CUDA stream）

实际时间线：

Compute Stream: [Att0] ─[Route0]─ wait ─[Expert0]──────── [Combine0] ─[Att1] ...
                                   ↑
Comm Stream D:   ─────────[Dispatch0]────→                                ────[Dispatch1]─→
Comm Stream C:                         ────[Combine A2A]──→
```

---

## 4. Layer 3：跨层 DAG 调度

### 4.1 跨 MoE Layer 的 overlap

对应 FlowMoE (NeurIPS'25) 的核心创新，但在 CommTensor 基础上实现：

```
2 个 MoE Layer 的时间线（传统）：

Layer N:   [Route] ─[Dispatch]─ [Expert] ─[Combine]
Layer N+1:                                            [Route] ─[Dispatch]─ [Expert] ─[Combine]

串行执行，无 overlap
```

```
MoEX 跨层 DAG：

Layer N:   [Route] ─[Dispatch]─ [Expert, tiles 0..N/2] ─[Expert, tiles N/2..N] ─[Combine]
                          ↕ 触发 Layer N+1 的 Route（依赖：Layer N 的 Route 结果独立）
Layer N+1:    [Route] ─[Dispatch]─ (等待 Layer N Expert 完成) ─[Expert] ─[Combine]

重叠部分：
  Layer N Expert 后半段 ↔ Layer N+1 Route + Dispatch 前半段

进一步：Layer N+1 的 Dispatch 与 Layer N 的 Combine 也可以 overlap！
```

### 4.2 DAG 节点定义

```python
@dataclass
class MoEXTask:
    """DAG 中的一个任务节点"""
    task_type: TaskType          # ROUTE, DISPATCH, EXPERT_TILE, COMBINE
    layer_id: int                # 所属 MoE Layer
    tile_id: int = -1            # tile 编号（仅 EXPERT_TILE）
    priority: int = 0            # 调度优先级（comm 任务优先级高）
    
    # 依赖关系
    predecessors: List['MoEXTask'] = field(default_factory=list)
    successors: List['MoEXTask'] = field(default_factory=list)
    
    # 执行状态
    stream: CudaStream = None
    cuda_event: CudaEvent = None
    status: TaskStatus = TaskStatus.PENDING

class TaskType(Enum):
    ROUTE = 1
    DISPATCH = 2          # 通信任务，优先级 HIGH
    EXPERT_TILE = 3       # 计算任务，优先级 NORMAL
    COMBINE_RDMA = 4      # 通信任务，优先级 HIGH
    COMBINE_SCATTER = 5   # 计算任务，优先级 NORMAL
```

### 4.3 DAG 构建（2 个 MoE Layer 示例）

```
Task Graph:

ROUTE_N ──→ DISPATCH_N ──→ [等待 peer tokens] ──→ EXPERT_TILE_N[0]
                                                    → EXPERT_TILE_N[1]
                                                    → ...
                                                    → EXPERT_TILE_N[k]
                                                         → COMBINE_RDMA_N[k]  (Comm)
ROUTE_N ──→ (also)──────────────────────────────────────────────────────
                                                         → COMBINE_SCATTER_N

        ↓（DISPATCH_N 完成后，立即开始 Layer N+1 的 Route，不等 Expert）
ROUTE_N+1 ──→ DISPATCH_N+1 ──→ [等待 peer tokens] ──→ EXPERT_TILE_N+1[0] ...

注意：ROUTE_N+1 仅依赖 ROUTE_N 的 hidden_states 输出（通过 Residual），
     不依赖 EXPERT_N 的完成（专家输出还没回来）！
     → ROUTE_N+1 可以在 DISPATCH_N 完成后立即开始。
     
     这是因为：Route 的输入是 Attention 的输出（已通过 Residual 连接），
     而 Attention 在 Expert 执行前就完成了。
```

### 4.4 跨层调度的 CommTensor 优势

```
传统实现的跨层 overlap 困难：

Layer N 的 Combine 完成后，输出为：unpacked_output [B*L, H]
Layer N+1 需要此 output 作为 Attention 的输入
→ 必须等待 Combine 完全结束才能开始 Layer N+1

MoEX CommTensor 的改进：

Layer N Combine = scatter_add（原地写 output）
→ 写入是分 rank 逐步完成的（每接收到一个 rank 的结果就写入）
→ 对于 Layer N+1 的 Attention，只需要所有 output 的位置写入完成
→ 可以在最后一个 rank 的 Combine 完成后立即开始，而无需额外的 pack/unpack

进一步优化（流式 Combine）：
如果 Layer N+1 的 Attention 采用 Ring Attention（CP），
可以按 CP chunk 顺序 Combine（与 Ring Attention 的 KV 传递方向一致），
实现 Combine 与下一层 Attention 的 chunk 级 overlap！
```

---

## 5. Overlap 调度器设计

### 5.1 调度器架构

```python
class OverlapScheduler:
    """
    MoEX 三层 Overlap 调度器

    职责：
    1. 构建跨层 task DAG
    2. 分配任务到 CUDA stream
    3. 管理 Comm/Compute Warp 专化（通过 CUDA kernel 启动配置）
    4. 实时调整优先级（基于 profiling 反馈）
    """

    def __init__(self, config: MoEXConfig):
        self.config = config
        self.streams = StreamManager(config)
        self.dag = MoEXDAG()
        self.profiler = OverlapProfiler(enabled=config.enable_profiling)

    def schedule_moe_layer(
        self,
        layer_id: int,
        hidden_states: Tensor,
        expert_weights: List[Tensor],
    ) -> Tensor:
        """
        调度单个 MoE Layer 的执行，返回 output tensor（async，通过 event 同步）
        """
        # Step 1: 构建本层 task DAG
        tasks = self._build_layer_tasks(layer_id, hidden_states)

        # Step 2: 与前一层的 tasks 融合（跨层 overlap）
        self.dag.merge(tasks, prev_layer_id=layer_id - 1)

        # Step 3: 按优先级分配到 streams（comm 优先）
        self._dispatch_tasks(tasks)

        # Step 4: 返回 output（通过 CUDA event 等待）
        return self._get_output_async(tasks)

    def _build_layer_tasks(self, layer_id: int, hidden_states: Tensor):
        tasks = []

        # Route Task
        route_task = MoEXTask(TaskType.ROUTE, layer_id)
        route_task.stream = self.streams.compute

        # Dispatch Task（高优先级！）
        dispatch_task = MoEXTask(TaskType.DISPATCH, layer_id, priority=HIGH)
        dispatch_task.stream = self.streams.comm_dispatch
        dispatch_task.predecessors = [route_task]

        # Expert Tile Tasks（per tile）
        prev_task = dispatch_task
        for t in range(self.config.num_tiles):
            tile_task = MoEXTask(TaskType.EXPERT_TILE, layer_id, tile_id=t)
            tile_task.stream = self.streams.compute
            tile_task.predecessors = [prev_task]

            # Combine RDMA Task（高优先级，紧跟 tile 完成）
            rdma_task = MoEXTask(TaskType.COMBINE_RDMA, layer_id, tile_id=t, priority=HIGH)
            rdma_task.stream = self.streams.comm_combine
            rdma_task.predecessors = [tile_task]

            tasks.extend([tile_task, rdma_task])
            prev_task = tile_task

        # Combine Scatter Task
        scatter_task = MoEXTask(TaskType.COMBINE_SCATTER, layer_id)
        scatter_task.stream = self.streams.compute
        scatter_task.predecessors = [t for t in tasks if t.task_type == TaskType.COMBINE_RDMA]

        tasks.extend([route_task, dispatch_task, scatter_task])
        return tasks
```

### 5.2 Stream 管理

```python
class StreamManager:
    """CUDA Stream 管理，根据配置创建最优 stream 分配"""

    def __init__(self, config: MoEXConfig):
        # 主计算流
        self.compute = torch.cuda.Stream(priority=-1)  # 高优先级

        # 通信流（dispatch 方向）
        self.comm_dispatch = torch.cuda.Stream(priority=0)

        # 通信流（combine 方向）
        self.comm_combine = torch.cuda.Stream(priority=0)

        # Attention 计算流（与 MoE 并行）
        self.attention = torch.cuda.Stream(priority=-1)

        # Event 池（避免重复创建）
        self.event_pool = CudaEventPool(size=256)

    def sync_streams(self, src_stream: CudaStream, dst_stream: CudaStream):
        """使用 CUDA Event 在两个 stream 间同步（比 cudaStreamSynchronize 开销小）"""
        event = self.event_pool.acquire()
        src_stream.record_event(event)
        dst_stream.wait_event(event)
        return event

    def release_event(self, event: CudaEvent):
        self.event_pool.release(event)
```

### 5.3 优先级调度规则

```
任务优先级（越高越先调度）：

Priority 3 (最高)：
  - DISPATCH 任务（通信延迟最长，必须最早开始）
  - 跨层 DISPATCH（当前层已在 Expert GEMM，下层 Dispatch 应立即开始）

Priority 2：
  - COMBINE_RDMA 任务（tile GEMM 完成后，RDMA 应立即发出）

Priority 1：
  - EXPERT_TILE 任务（计算任务，但与 RDMA 紧密耦合）
  - COMBINE_SCATTER 任务

Priority 0 (最低)：
  - ROUTE 任务（本地计算，不阻塞通信）
  - Layer Norm 等辅助操作

调度规则：
  1. 通信任务始终优先于同层的计算任务
  2. 较早层的任务优先于较晚层的任务（关键路径优先）
  3. 同优先级按 DAG 拓扑顺序（BFS）
```

---

## 6. 三层 Overlap 的协同

### 6.1 时间线综合视图

```
3 个 MoE Layer，完整 overlap 时间线：

Stream:          t=0   t=1   t=2   t=3   t=4   t=5   t=6   t=7   t=8   t=9   t=10
                 ↓     ↓     ↓     ↓     ↓     ↓     ↓     ↓     ↓     ↓     ↓

compute:    [Rte0][─────────Expert0 (tile 0..N)─────────][Cmb0][Rte1][─Expert1─][Cmb1]
                   ↕ L1 Tile RDMA overlap                      ↕ L1 overlap

comm_disp:  [────D0────]                                       [────D1────]
                 ↕ L2 Block overlap (D0 ↔ Expert0)               ↕ L2

comm_comb:              [───C0_RDMA───]                                 [C1_RDMA]
                                       ↕ L3 跨层（C0_RDMA ↔ Expert1 setup）

attention:  [───────────Att0───────────][───────────Att1───────────][───Att2─]
            ↕ L2 Block（Att 与 MoE D/C 并行）

细化时间线（Layer 0，tile 级）：

tile:       [E0_t0][E0_t1][E0_t2][E0_t3][E0_t4]...
rdma:              [R_t0] [R_t1] [R_t2] [R_t3]...
            ← L1 tile 级 offset，每 tile 错位 1 步 →
```

### 6.2 Overlap 率计算

```
理论最大 Overlap 率（三层综合）：

Layer 1（Tile 级）贡献：
  - Expert GEMM 时间：T_gemm
  - RDMA Combine 时间：T_combine_rdma
  - Overlap 窗口：T_gemm - T_tile_latency（≈ T_gemm - 0.001ms）
  - Layer 1 overlap 率：(T_combine_rdma / T_gemm)（约 85-95%）

Layer 2（Block 级）贡献：
  - Dispatch 时间：T_dispatch（与 Expert GEMM overlap）
  - 可 overlap 时间：min(T_dispatch, T_gemm)
  - Layer 2 overlap 率：T_dispatch / T_gemm（约 70-80%，dispatch << gemm 时）

Layer 3（跨层）贡献：
  - 下层 Route + Dispatch 时间：T_route_N1 + T_dispatch_N1
  - 可 overlap 时间：Expert GEMM 的后半段
  - Layer 3 overlap 率：(T_route_N1 + T_dispatch_N1) / T_gemm（约 30-50%）

综合效果（通信完全隐藏的条件）：
  T_dispatch + T_combine_rdma ≤ T_gemm + T_gemm_prev_layer（三层联合）
  即：通信总量 ≤ 2 倍计算时间

典型参数（H100, EP=32, DeepSeek-V3 规模）：
  T_dispatch ≈ 0.17ms
  T_combine_rdma ≈ 0.17ms（对称）
  T_gemm ≈ 0.24ms（单层）
  T_gemm_prev_layer ≈ 0.24ms
  通信总量 = 0.34ms ≤ 计算 0.48ms → 通信完全隐藏！✓

实测目标 Overlap 率：92%+（vs. Comet 的 90%，高 2%，因为 CommTensor 消除了前置拷贝）
```

---

## 7. CommTensor 对 Overlap 的贡献

### 7.1 CommTensor 提前了 RDMA 开始时间

```
传统实现的 Dispatch 时间线：

t=0: Route 完成（Gate GEMM + TopK）
t=1: permute（拷贝，0.28ms）
t=2: pack（拷贝，0.14ms）
t=3: RDMA/A2A 开始（最早可以开始通信）
t=4: RDMA 完成
t=5: Expert GEMM 开始

RDMA 结束 → Expert 开始：约 5 个时间步

MoEX CommTensor 的时间线：

t=0: Route 完成（同时 CommTensor meta 已填写）
t=1: RDMA PUT 开始（CT.data[r] 地址已连续，立即可发）  ← 提前了 2 步！
t=2: Expert GEMM tile 0 开始（无需等待完整 RDMA 完成，等第一批 tile 就绪）
t=3: RDMA PUT 完成（与 tile 1,2 的 GEMM overlap）
```

### 7.2 CommTensor 使 Combine 更早完成

```
传统 Combine 时间线：

t=0: Expert GEMM 最后一个 tile 完成
t=1: A2A Combine（ALL tokens 一起发送，0.17ms）
t=2: unpack（拷贝，0.14ms）
t=3: weighted scatter（拷贝，0.14ms）
t=4: Combine 完成 → 下一层 Route 可以开始

MoEX CommTensor Combine 时间线：

t=0: Expert GEMM tile 0 完成 → RDMA tile 0 发出（Layer 1 overlap）
t=1: tile 1 完成 → RDMA tile 1 发出（逐 tile 发送）
     同时：远端 GPU 已收到 tile 0 → 已写入 output[token_indices[0]]
...
t=N: 最后一个 tile RDMA 完成（大部分 output 已写入）
t=N+1: 下一层 Route 开始 ← 比传统早 ~0.28ms！
```

### 7.3 量化总结

```
CommTensor 对 Overlap 的量化贡献：

前置准备时间节省（早 0.28ms 开始通信）：
  → Dispatch 有更多时间 overlap Expert GEMM
  → 从 Layer 2 overlap 率 60% → 75%（估计）

Tile-level pipeline（Layer 1）：
  → Combine RDMA 逐 tile 完成，无需等全量
  → Layer 1 overlap 率：90%+

综合效果：
  传统（无 CommTensor，有 FlowMoE 风格跨层）：~68% overlap
  MoEX（CommTensor + 三层 overlap）：目标 92%+
  差异来源：
    +8%：CommTensor 零拷贝 → 更早开始 RDMA（2% 直接，6% 间接）
    +16%：Tile 级 RDMA（vs. chunk 级）
```

---

## 8. 性能分析与论文对比

### 8.1 与对标系统的 Overlap 率对比

| 系统 | Overlap 机制 | 粒度 | 理论 Overlap 率 | 实测/目标 |
|------|-------------|------|----------------|----------|
| MegatronCore (2024) | 双流，tensor 级 | ~100MB | 70% | ~60-70% |
| FlowMoE (NeurIPS'25) | DAG，chunk 级 | ~256MB | 80% | ~68% |
| Comet (MLSys'25) | Warp 专化，tile 级 | ~1MB | 95% | ~90% |
| **MoEX** | **三层，tile+零拷贝** | **~1MB + 0拷贝** | **~98%** | **目标 92%+** |

MoEX 相比 Comet 的理论优势：
1. 零拷贝前置（CommTensor）：RDMA 早 0.28ms 开始 → 更多 overlap 窗口
2. 跨层 DAG（FlowMoE 风格）：下一层 Dispatch 与当前层 Expert 并行
3. FSEP ReduceScatter on NVLink：Intra-node 通信更快，不占 IB 带宽

### 8.2 端到端加速预测

```
基于论文数据的预测（DeepSeek-V3 规模，EP=32）：

各优化的独立贡献（叠加）：
  CommTensor 零拷贝（dispatch/combine 前置节省）：+8-12%
  Tile-level GEMM-RDMA（Layer 1）：+20-25%（from Comet: 90% vs 68%）
  跨层 DAG 调度（Layer 3）：+8-15%（from FlowMoE）
  FSEP 负载均衡（from LAER-MoE）：+35-45%（消除负载不均）
  CommTensor 内存节省 → 更大 batch：+5-10%

综合预测（考虑相互依赖，叠加效应不完全独立）：
  vs. 无优化的传统 MegatronCore EP：~2.5-3.5× 端到端加速
  vs. Comet（最佳单一系统）：~1.4-1.8×（因 FSEP + 零拷贝的额外贡献）
  vs. LAER-MoE（FSEP 基础）：~1.2-1.4×（因 tile-level overlap + 零拷贝）

保守估计（实际工程实现通常低于理论）：
  vs. 传统 EP：~2.0×
  vs. Comet：~1.2-1.4×
  vs. LAER-MoE：~1.1-1.2×
```

### 8.3 适用场景分析

```
MoEX 的最优适用场景：

① 大规模 EP（EP ≥ 32，跨节点）：
   IB 通信延迟大 → tile-level overlap 效益高
   负载不均衡明显 → FSEP 效益高
   → CommTensor 零拷贝节省更多（越大通信包，前置拷贝比例越小，但总量越大）

② 长序列训练（L ≥ 4096）：
   Token 数多 → CommTensor slot 利用率高
   Ring Attention → 与 Combine 的 chunk 级 overlap 可能性更高

③ 高专家数（num_experts ≥ 64）：
   load imbalance 更严重 → FSEP + 动态 re-layout 效益更高
   CommTensor 路由直写 meta 优化更明显（更多路由计算量）

MoEX 相对较小优势的场景：
  单节点训练（EP=8，NVLink）：通信延迟本来就小，Comet 的 tile-level overlap 已足够
  小专家数（num_experts ≤ 8）：负载相对均衡，FSEP 额外通信不合算
```
