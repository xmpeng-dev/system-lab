"""
MoEX Overlap Scheduler：三层 Overlap 调度器

实现 MoEX 的三层 Overlap 策略：
  Layer 1：Tile 级 GEMM-RDMA Pipeline（Comet 风格）
  Layer 2：Block 级 Dispatch/Combine 与 Expert GEMM 解耦
  Layer 3：跨 MoE Layer 的 DAG 调度（FlowMoE 风格）

调度器管理：
  - CUDA Stream 分配（compute + comm_dispatch + comm_combine）
  - Task DAG 构建与执行
  - Comm/Compute Warp 专化（通过 kernel launch config）
  - 跨层 overlap（下一层 Route+Dispatch 与当前层 Expert GEMM 并行）
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor

from .comm_tensor import CommTensor, CommTensorConfig


# ---------------------------------------------------------------------------
# Task 类型与优先级
# ---------------------------------------------------------------------------

class TaskType(Enum):
    ROUTE = auto()
    DISPATCH = auto()
    DISPATCH_BARRIER = auto()  # 等待所有 dispatch 完成的同步屏障
    EXPERT_TILE = auto()
    COMBINE_RDMA = auto()
    COMBINE_SCATTER = auto()
    REDUCE_SCATTER = auto()   # FSEP 专用


class TaskStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    DONE = auto()


# 优先级：值越大，优先级越高
TASK_PRIORITY = {
    TaskType.DISPATCH: 4,          # 最高：通信延迟长，最早开始
    TaskType.COMBINE_RDMA: 3,      # 高：tile 完成后立即发出
    TaskType.EXPERT_TILE: 2,       # 中：计算任务
    TaskType.REDUCE_SCATTER: 3,    # 高：FSEP ReduceScatter on NVLink
    TaskType.COMBINE_SCATTER: 1,   # 中低：等所有 RDMA 完成后执行
    TaskType.ROUTE: 0,             # 最低：本地计算，不影响通信
    TaskType.DISPATCH_BARRIER: 1,  # 中低：等待所有 dispatch 完成
}


@dataclass
class MoEXTask:
    """
    MoEX DAG 中的一个任务节点

    每个任务对应一个最小可调度操作：
    - ROUTE: Gate GEMM + TopK
    - DISPATCH: All-to-All / RDMA PUT（通信）
    - EXPERT_TILE: 单个 GEMM tile 的计算（细粒度）
    - COMBINE_RDMA: tile 结果的 RDMA PUT（通信）
    - COMBINE_SCATTER: scatter_add 到输出 tensor
    - REDUCE_SCATTER: FSEP 的节点内 ReduceScatter
    """
    task_type: TaskType
    layer_id: int               # 所属 MoE Layer 编号
    tile_id: int = -1           # tile 编号（仅 EXPERT_TILE 和 COMBINE_RDMA）
    rank_id: int = -1           # 目标 rank（仅 DISPATCH 和 COMBINE_RDMA）

    # 优先级（调度时使用）
    priority: int = field(init=False)

    # 依赖关系（有向边）
    predecessors: List['MoEXTask'] = field(default_factory=list)
    successors: List['MoEXTask'] = field(default_factory=list)

    # 执行上下文
    stream: Optional[torch.cuda.Stream] = None
    cuda_event: Optional[torch.cuda.Event] = None
    status: TaskStatus = TaskStatus.PENDING

    # 关联数据（执行时使用）
    comm_tensor: Optional[CommTensor] = None
    output_tensor: Optional[Tensor] = None

    def __post_init__(self):
        self.priority = TASK_PRIORITY.get(self.task_type, 0)

    def is_ready(self) -> bool:
        """检查所有前驱任务是否完成"""
        return all(p.status == TaskStatus.DONE for p in self.predecessors)

    def add_dependency(self, predecessor: 'MoEXTask'):
        """添加依赖关系"""
        self.predecessors.append(predecessor)
        predecessor.successors.append(self)

    def __repr__(self) -> str:
        return (
            f'Task({self.task_type.name}, '
            f'layer={self.layer_id}, '
            f'tile={self.tile_id}, '
            f'prio={self.priority}, '
            f'status={self.status.name})'
        )


# ---------------------------------------------------------------------------
# Stream Manager
# ---------------------------------------------------------------------------

class StreamManager:
    """
    CUDA Stream 管理器

    维护专用 stream：
    - compute：Expert GEMM + scatter_add
    - comm_dispatch：dispatch 方向通信（All-to-All / RDMA PUT）
    - comm_combine：combine 方向通信（RDMA PUT / GET）
    - attention：Attention 计算（与 MoE 并行）
    """

    def __init__(self):
        if torch.cuda.is_available():
            # 使用不同优先级确保通信 stream 优先
            self.compute = torch.cuda.Stream(priority=-1)      # 高优先级计算
            self.comm_dispatch = torch.cuda.Stream(priority=0) # 通信
            self.comm_combine = torch.cuda.Stream(priority=0)  # 通信
            self.attention = torch.cuda.Stream(priority=-1)    # 高优先级计算
            self._event_pool: List[torch.cuda.Event] = [
                torch.cuda.Event(enable_timing=False) for _ in range(64)
            ]
            self._event_cursor = 0
        else:
            # CPU 模式（调试用）
            self.compute = None
            self.comm_dispatch = None
            self.comm_combine = None
            self.attention = None
            self._event_pool = []
            self._event_cursor = 0

    def acquire_event(self) -> Optional[torch.cuda.Event]:
        """从 event pool 获取一个事件（避免重复创建）"""
        if not self._event_pool:
            return None
        event = self._event_pool[self._event_cursor % len(self._event_pool)]
        self._event_cursor += 1
        return event

    def sync(self, src_stream, dst_stream):
        """在两个 stream 间插入事件同步"""
        if src_stream is None or dst_stream is None:
            return
        event = self.acquire_event()
        if event is not None:
            src_stream.record_event(event)
            dst_stream.wait_event(event)

    def get_stream_for_task(self, task: MoEXTask) -> Optional[torch.cuda.Stream]:
        """根据 task 类型选择 stream"""
        if task.task_type in (TaskType.DISPATCH, TaskType.REDUCE_SCATTER):
            return self.comm_dispatch
        elif task.task_type in (TaskType.COMBINE_RDMA,):
            return self.comm_combine
        else:
            return self.compute


# ---------------------------------------------------------------------------
# Overlap Scheduler（核心调度器）
# ---------------------------------------------------------------------------

class OverlapScheduler:
    """
    MoEX 三层 Overlap 调度器

    调度逻辑：
    1. 接收每个 MoE Layer 的任务列表
    2. 合并到全局 DAG（支持跨层依赖）
    3. 按优先级（通信优先）分配到 CUDA stream
    4. 管理同步点（CUDA event）

    设计目标：
    - 通信任务尽可能早地开始（最大化 overlap 窗口）
    - tile 级 GEMM → 立即 RDMA（Comet 风格）
    - 当前层 Expert 执行时，下一层 Dispatch 也在执行（FlowMoE 风格）
    """

    def __init__(self, config: CommTensorConfig):
        self.config = config
        self.streams = StreamManager()

        # 全局 DAG（跨层）
        self._pending_tasks: List[MoEXTask] = []
        self._running_tasks: List[MoEXTask] = []
        self._done_tasks: List[MoEXTask] = []

        # 层间依赖追踪
        self._last_route_task: Dict[int, MoEXTask] = {}      # {layer_id: route_task}
        self._last_dispatch_task: Dict[int, MoEXTask] = {}   # {layer_id: dispatch_task}
        self._last_expert_task: Dict[int, MoEXTask] = {}     # {layer_id: last_tile_task}
        self._last_combine_task: Dict[int, MoEXTask] = {}    # {layer_id: scatter_task}

        # 性能统计
        self.stats = SchedulerStats()

    def build_layer_dag(
        self,
        layer_id: int,
        comm_tensor: CommTensor,
        output_tensor: Tensor,
    ) -> List[MoEXTask]:
        """
        为单个 MoE Layer 构建 task DAG

        Layer 1 (tile 级) + Layer 2 (block 级) 的 overlap 都在这里体现。
        Layer 3 (跨层) 通过与前一层的 DAG 融合实现。

        Returns:
            该层的所有 task 列表
        """
        config = self.config
        R = config.num_ep_ranks
        T = config.num_tiles
        tasks = []

        # ── Task 1: ROUTE（Gate GEMM + TopK）──────────────────────────────
        route_task = MoEXTask(
            task_type=TaskType.ROUTE,
            layer_id=layer_id,
            comm_tensor=comm_tensor,
        )
        # 依赖上一层的 combine（确保 hidden_states 已写入）
        if layer_id - 1 in self._last_combine_task:
            route_task.add_dependency(self._last_combine_task[layer_id - 1])
        tasks.append(route_task)

        # ── Task 2: DISPATCH（高优先级，一个 task per rank）──────────────
        dispatch_tasks = []
        for r in range(R):
            dispatch_task = MoEXTask(
                task_type=TaskType.DISPATCH,
                layer_id=layer_id,
                rank_id=r,
                comm_tensor=comm_tensor,
            )
            dispatch_task.add_dependency(route_task)
            dispatch_tasks.append(dispatch_task)
            tasks.append(dispatch_task)

        # 聚合 dispatch 完成事件（Expert GEMM 等待所有 dispatch 完成）
        dispatch_barrier = MoEXTask(
            task_type=TaskType.DISPATCH_BARRIER,
            layer_id=layer_id,
            comm_tensor=comm_tensor,
        )
        for dt in dispatch_tasks:
            dispatch_barrier.add_dependency(dt)
        tasks.append(dispatch_barrier)

        # ── Layer 3：跨层 overlap ──────────────────────────────────────
        # 下一层的 Route+Dispatch 可以在当前层 Expert GEMM 执行时开始
        # （这里通过调整依赖关系实现：dispatch_barrier 不阻塞下一层的 route）
        # 注：实际中，下一层的 route 依赖当前层 route 的 hidden_states 输出（通过 residual）
        # 这里记录当前层的 dispatch_barrier，下一层的 route 只依赖当前层的 route（不是 expert）

        # ── Task 3: EXPERT_TILE（tile 级 GEMM）───────────────────────────
        tile_tasks = []
        for t in range(T):
            tile_task = MoEXTask(
                task_type=TaskType.EXPERT_TILE,
                layer_id=layer_id,
                tile_id=t,
                comm_tensor=comm_tensor,
            )
            if t == 0:
                tile_task.add_dependency(dispatch_barrier)
            else:
                # tile 之间串行（实际中可以 pipeline 化）
                tile_task.add_dependency(tile_tasks[-1])
            tile_tasks.append(tile_task)
            tasks.append(tile_task)

            # ── Layer 1：tile 完成即触发 RDMA（COMBINE_RDMA）──────────────
            combine_rdma_task = MoEXTask(
                task_type=TaskType.COMBINE_RDMA,
                layer_id=layer_id,
                tile_id=t,
                comm_tensor=comm_tensor,
            )
            combine_rdma_task.add_dependency(tile_task)
            tasks.append(combine_rdma_task)

        # ── Task 4: COMBINE_SCATTER（所有 tile RDMA 完成后 scatter_add）──
        scatter_task = MoEXTask(
            task_type=TaskType.COMBINE_SCATTER,
            layer_id=layer_id,
            comm_tensor=comm_tensor,
            output_tensor=output_tensor,
        )
        for t in range(T):
            # 找到 tile t 的 combine_rdma task
            rdma_task = tasks[-(T - t) - T + t - 1] if t < T else tasks[-1]
        # 简化：scatter_task 依赖最后一个 tile task
        scatter_task.add_dependency(tile_tasks[-1])
        tasks.append(scatter_task)

        # 记录当前层的关键 tasks（供下一层引用）
        self._last_route_task[layer_id] = route_task
        self._last_dispatch_task[layer_id] = dispatch_tasks[0]  # 第一个 dispatch
        self._last_expert_task[layer_id] = tile_tasks[-1]       # 最后一个 tile
        self._last_combine_task[layer_id] = scatter_task

        return tasks

    def execute_tasks(self, tasks: List[MoEXTask]) -> None:
        """
        执行 task 列表（按优先级 + 拓扑顺序）

        简化实现：同步执行（演示 DAG 调度逻辑）
        实际实现：需要 CUDA callback + event，实现真正的异步 overlap
        """
        # 按优先级和拓扑顺序排序
        ready_queue = [t for t in tasks if not t.predecessors]
        ready_queue.sort(key=lambda t: -t.priority)

        executed = set()
        iteration = 0

        while ready_queue:
            iteration += 1
            task = ready_queue.pop(0)

            if not task.is_ready():
                # 重新放入队列（等待前驱完成）
                ready_queue.append(task)
                ready_queue.sort(key=lambda t: -t.priority)
                continue

            # 执行 task
            self._execute_single_task(task)
            executed.add(id(task))
            task.status = TaskStatus.DONE

            # 将就绪的后继任务加入队列
            for succ in task.successors:
                if id(succ) not in executed and succ.is_ready():
                    if succ not in ready_queue:
                        ready_queue.append(succ)
            ready_queue.sort(key=lambda t: -t.priority)

        self.stats.total_tasks_executed += len(tasks)

    def _execute_single_task(self, task: MoEXTask) -> None:
        """
        执行单个 task（使用对应的 CUDA stream）

        实际实现中，每个 task 对应一个 CUDA kernel 或 RDMA 调用
        """
        stream = self.streams.get_stream_for_task(task)
        task.status = TaskStatus.RUNNING

        if stream is not None:
            ctx = torch.cuda.stream(stream)
        else:
            ctx = _null_context()

        with ctx:
            if task.task_type == TaskType.ROUTE:
                # Gate GEMM + TopK（已在 Router.forward() 中完成）
                pass

            elif task.task_type == TaskType.DISPATCH:
                # RDMA PUT：ct.data[rank_id] → 对端 GPU
                if task.comm_tensor is not None and task.rank_id >= 0:
                    # 模拟：实际中是 RDMA one-sided PUT
                    self.stats.dispatch_count += 1

            elif task.task_type == TaskType.DISPATCH_BARRIER:
                # 等待所有 rank 的 dispatch 完成后通知 Expert GEMM 可以开始
                # 实际：CUDA event sync（等待所有 comm_dispatch stream 的 events）
                pass

            elif task.task_type == TaskType.EXPERT_TILE:
                # GEMM tile 计算
                if task.tile_id >= 0:
                    self.stats.tile_gemm_count += 1

            elif task.task_type == TaskType.COMBINE_RDMA:
                # tile 结果 RDMA PUT（Comet 风格：即完即发）
                if task.tile_id >= 0:
                    self.stats.combine_rdma_count += 1

            elif task.task_type == TaskType.COMBINE_SCATTER:
                # scatter_add 到 output tensor
                self.stats.scatter_count += 1

            elif task.task_type == TaskType.REDUCE_SCATTER:
                # FSEP ReduceScatter（NVLink）
                self.stats.reduce_scatter_count += 1

        task.status = TaskStatus.DONE


# ---------------------------------------------------------------------------
# Tile 级 Pipeline（Layer 1 Overlap 的具体实现）
# ---------------------------------------------------------------------------

class TilePipeline:
    """
    Tile 级 GEMM-RDMA Pipeline（对应 Comet 的 Warp 专化）

    在 Python 层面模拟 Comet 的 Compute Warp / Comm Warp 分工：
    - Compute Thread：执行 tile GEMM
    - Comm Thread：监控完成，触发 RDMA

    实际生产实现：通过 CUDA warp 专化 + shared memory 协调
    """

    def __init__(self, config: CommTensorConfig, comm_stream: Optional[torch.cuda.Stream]):
        self.config = config
        self.comm_stream = comm_stream
        self._tile_done = {}  # {tile_id: bool}

    def run_tile_gemm_rdma(
        self,
        input_ct: CommTensor,
        expert_weights: Tensor,
        output_ct: CommTensor,
    ) -> None:
        """
        执行 tile 级 GEMM-RDMA pipeline：
        tile 0 GEMM 完成 → RDMA tile 0 → tile 1 GEMM（并行）→ RDMA tile 1 → ...
        """
        T = self.config.num_tiles
        H = self.config.d_model
        tile = self.config.tile_size

        for t in range(T):
            # Compute：GEMM tile t
            # input_ct 中所有 token 的 tile t：[total_tokens, tile_size]
            tokens_tile_t = input_ct.data[:, :, t, :].reshape(-1, tile)  # [R*S, tile]

            # 使用 expert weight 的 tile t 列
            if t * tile < expert_weights.shape[-1]:
                w_tile = expert_weights[:, t * tile:(t + 1) * tile]      # [H, tile]
                result_tile = tokens_tile_t @ w_tile.T                   # [R*S, tile]

                # 写入 output_ct
                output_ct.data[:, :, t, :] = result_tile.view(
                    output_ct.data.shape[0],
                    output_ct.data.shape[1],
                    tile,
                )

            self._tile_done[t] = True

            # Comm：立即触发 RDMA（Comet 风格，在 comm_stream 上异步）
            # 实际：RDMA PUT output_ct.data[:, :, t, :] 到原始 rank
            if self.comm_stream is not None:
                # 模拟：插入 event，实际是 RDMA PUT
                pass

    def get_overlap_rate(self) -> float:
        """
        估算实际 overlap 率
        （基于 tile GEMM 时间 / tile RDMA 时间 的比值）
        """
        T = self.config.num_tiles
        tile = self.config.tile_size
        H = self.config.d_model

        # 粗略估算（实际需要 profiling）
        gemm_time_per_tile = tile * H / (312e12)   # H100 TFLOPS
        rdma_time_per_tile = tile * H * 2 / (50e9)  # IB 带宽（bytes）

        if rdma_time_per_tile <= gemm_time_per_tile:
            # RDMA 可以被 GEMM 完全覆盖
            return 1.0
        else:
            # RDMA 比 GEMM 慢，overlap 率 = GEMM/RDMA
            return gemm_time_per_tile / rdma_time_per_tile


# ---------------------------------------------------------------------------
# 调度器性能统计
# ---------------------------------------------------------------------------

@dataclass
class SchedulerStats:
    """调度器执行统计"""
    total_tasks_executed: int = 0
    dispatch_count: int = 0
    tile_gemm_count: int = 0
    combine_rdma_count: int = 0
    scatter_count: int = 0
    reduce_scatter_count: int = 0

    def reset(self):
        self.total_tasks_executed = 0
        self.dispatch_count = 0
        self.tile_gemm_count = 0
        self.combine_rdma_count = 0
        self.scatter_count = 0
        self.reduce_scatter_count = 0

    def __repr__(self) -> str:
        return (
            f'SchedulerStats('
            f'tasks={self.total_tasks_executed}, '
            f'dispatch={self.dispatch_count}, '
            f'tile_gemm={self.tile_gemm_count}, '
            f'combine_rdma={self.combine_rdma_count}, '
            f'scatter={self.scatter_count}'
            f')'
        )


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

class _null_context:
    def __enter__(self): return self
    def __exit__(self, *args): pass
