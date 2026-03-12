"""
MoEPackage Module 4: Dual-Channel Comm Scheduler

AMD 独有优势：XGMI 和 RDMA 是物理独立的两条通信通道。
NVIDIA 的 NVLink 和 InfiniBand 共享 NCCL 通信栈，软件层面无法真正并发。
AMD 可以同时使用 XGMI P2P API + ibverbs API 实现双通道并发。

有效带宽对比：
  NVIDIA: effective_bw ≈ max(nvlink_bw, ib_bw)  # 共享 NCCL，互斥
  AMD:    effective_bw = xgmi_bw + rdma_bw        # 物理独立，可叠加

DAG 驱动调度：
  将所有通信操作按拓扑分类到 XGMI 或 RDMA 通道
  两条通道独立推进，互不阻塞
  计算操作与通信操作三路并发（Compute + XGMI + RDMA）
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# 通道类型与硬件参数
# ---------------------------------------------------------------------------

# MI300X 通信带宽参数
MI300X_XGMI_BW_GBS = 896.0     # XGMI 双向带宽 (GB/s)
MI300X_RDMA_BW_GBS = 50.0      # 单端口 RoCE 400Gbps ≈ 50 GB/s
MI300X_XGMI_LATENCY_US = 3.0   # XGMI 启动延迟 (μs)
MI300X_RDMA_LATENCY_US = 5.0   # RDMA 启动延迟 (μs)


# NVIDIA 对比参数（用于带宽分析）
NVIDIA_NVLINK_BW_GBS = 900.0    # NVLink 5 双向带宽 (GB/s)
NVIDIA_IB_BW_GBS = 50.0         # InfiniBand NDR 400Gbps ≈ 50 GB/s


class ChannelType(Enum):
    """通信通道类型"""
    XGMI = auto()    # 节点内 XGMI P2P（高带宽低延迟）
    RDMA = auto()    # 跨节点 RDMA RoCE（中带宽中延迟）
    LOCAL = auto()   # 本地计算（不占用通信通道）


class TaskStatus(Enum):
    """任务状态"""
    PENDING = auto()
    READY = auto()
    RUNNING = auto()
    DONE = auto()


class TaskType(Enum):
    """通信/计算任务类型"""
    # XGMI 通道任务
    DISPATCH_LOCAL = auto()     # 节点内 Expert Dispatch (XGMI P2P)
    REDUCE_SCATTER = auto()     # FSEP ReduceScatter (XGMI)
    EXPERT_RELAYOUT = auto()    # Expert 参数搬迁 (XGMI)
    TP_ALLREDUCE = auto()       # TP All-Reduce (XGMI, 节点内)
    COMBINE_LOCAL = auto()      # 节点内 Combine 反向 (XGMI P2P)

    # RDMA 通道任务
    DISPATCH_REMOTE = auto()    # 跨节点 Expert Dispatch (RDMA)
    COMBINE_REMOTE = auto()     # 跨节点 Combine (RDMA)
    GRAD_ALLREDUCE = auto()     # 梯度 AllReduce (RDMA, DP 组)
    FSDP_ALLGATHER = auto()     # FSDP AllGather (RDMA)

    # 本地计算任务
    GATE_COMPUTE = auto()       # Gate 路由计算
    EXPERT_GEMM = auto()        # Expert GEMM 计算
    QUANTIZE = auto()           # FP8 量化
    PERMUTE = auto()            # Token Permutation


# 任务 → 通道映射
TASK_CHANNEL_MAP: Dict[TaskType, ChannelType] = {
    TaskType.DISPATCH_LOCAL:  ChannelType.XGMI,
    TaskType.REDUCE_SCATTER:  ChannelType.XGMI,
    TaskType.EXPERT_RELAYOUT: ChannelType.XGMI,
    TaskType.TP_ALLREDUCE:    ChannelType.XGMI,
    TaskType.COMBINE_LOCAL:   ChannelType.XGMI,
    TaskType.DISPATCH_REMOTE: ChannelType.RDMA,
    TaskType.COMBINE_REMOTE:  ChannelType.RDMA,
    TaskType.GRAD_ALLREDUCE:  ChannelType.RDMA,
    TaskType.FSDP_ALLGATHER:  ChannelType.RDMA,
    TaskType.GATE_COMPUTE:    ChannelType.LOCAL,
    TaskType.EXPERT_GEMM:     ChannelType.LOCAL,
    TaskType.QUANTIZE:        ChannelType.LOCAL,
    TaskType.PERMUTE:         ChannelType.LOCAL,
}

# 任务优先级（值越大越优先调度）
TASK_PRIORITY: Dict[TaskType, int] = {
    TaskType.DISPATCH_REMOTE: 5,   # 跨节点延迟最高，最先启动
    TaskType.DISPATCH_LOCAL:  4,   # 节点内 dispatch 紧随其后
    TaskType.GRAD_ALLREDUCE:  3,   # 梯度通信可与计算重叠
    TaskType.EXPERT_GEMM:     3,   # 计算任务优先级适中
    TaskType.COMBINE_REMOTE:  3,
    TaskType.COMBINE_LOCAL:   2,
    TaskType.REDUCE_SCATTER:  2,
    TaskType.FSDP_ALLGATHER:  2,
    TaskType.TP_ALLREDUCE:    1,
    TaskType.GATE_COMPUTE:    1,
    TaskType.EXPERT_RELAYOUT: 1,
    TaskType.QUANTIZE:        0,
    TaskType.PERMUTE:         0,
}


# ---------------------------------------------------------------------------
# 通信通道抽象
# ---------------------------------------------------------------------------

@dataclass
class CommChannel:
    """
    A physical communication channel (XGMI or RDMA).

    每条通道有独立的带宽、延迟和任务队列。
    AMD 上 XGMI 和 RDMA 可以真正并发（物理独立）。

    Attributes:
        channel_type: 通道类型
        bandwidth_gbs: 带宽 (GB/s)
        latency_us: 启动延迟 (μs)
        current_task: 当前执行中的任务
        utilization_bytes: 本 step 已传输字节数
    """
    channel_type: ChannelType
    bandwidth_gbs: float
    latency_us: float
    current_task: Optional[CommTask] = field(default=None, repr=False)
    utilization_bytes: int = 0
    total_tasks_completed: int = 0

    @classmethod
    def create_xgmi(cls) -> CommChannel:
        """Create XGMI channel with MI300X specs."""
        return cls(
            channel_type=ChannelType.XGMI,
            bandwidth_gbs=MI300X_XGMI_BW_GBS,
            latency_us=MI300X_XGMI_LATENCY_US,
        )

    @classmethod
    def create_rdma(cls) -> CommChannel:
        """Create RDMA channel with MI300X specs."""
        return cls(
            channel_type=ChannelType.RDMA,
            bandwidth_gbs=MI300X_RDMA_BW_GBS,
            latency_us=MI300X_RDMA_LATENCY_US,
        )

    def estimate_transfer_time_us(self, data_bytes: int) -> float:
        """Estimate transfer time in microseconds."""
        transfer_us = data_bytes / (self.bandwidth_gbs * 1e3)  # GB/s → bytes/μs
        return self.latency_us + transfer_us

    def is_busy(self) -> bool:
        return self.current_task is not None

    def reset_stats(self) -> None:
        self.utilization_bytes = 0
        self.total_tasks_completed = 0


# ---------------------------------------------------------------------------
# 通信/计算任务
# ---------------------------------------------------------------------------

@dataclass
class CommTask:
    """
    A communication or compute task in the DAG.

    每个 CommTask 是调度器的最小单位，包含：
    - 任务类型和通道归属
    - 数据大小（用于带宽估算）
    - DAG 依赖关系
    - 可选的执行回调

    Attributes:
        task_id: 唯一标识
        task_type: 任务类型
        layer_id: 所属 MoE Layer
        data_bytes: 传输数据量（字节）
        channel: 分配的通信通道
        status: 执行状态
        predecessors: 前驱任务列表
        successors: 后继任务列表
        estimated_time_us: 预估执行时间 (μs)
        callback: 任务完成时的回调函数
    """
    task_id: int
    task_type: TaskType
    layer_id: int = 0
    data_bytes: int = 0

    channel: ChannelType = field(init=False)
    status: TaskStatus = TaskStatus.PENDING
    predecessors: List[CommTask] = field(default_factory=list)
    successors: List[CommTask] = field(default_factory=list)
    estimated_time_us: float = 0.0
    callback: Optional[Callable] = field(default=None, repr=False)

    # 调度器填充
    priority: int = field(init=False)
    start_time_us: float = 0.0
    end_time_us: float = 0.0

    def __post_init__(self):
        self.channel = TASK_CHANNEL_MAP.get(self.task_type, ChannelType.LOCAL)
        self.priority = TASK_PRIORITY.get(self.task_type, 0)

    def add_dependency(self, predecessor: CommTask) -> None:
        """Add a DAG dependency (predecessor must complete before self)."""
        if predecessor not in self.predecessors:
            self.predecessors.append(predecessor)
        if self not in predecessor.successors:
            predecessor.successors.append(self)

    def is_ready(self) -> bool:
        """Check if all predecessors are done."""
        return all(p.status == TaskStatus.DONE for p in self.predecessors)

    def __repr__(self) -> str:
        return (
            f"CommTask(id={self.task_id}, type={self.task_type.name}, "
            f"ch={self.channel.name}, status={self.status.name}, "
            f"data={self.data_bytes}, est={self.estimated_time_us:.1f}μs)"
        )


# ---------------------------------------------------------------------------
# Dual-Channel Scheduler：双通道调度器
# ---------------------------------------------------------------------------

class DualChannelScheduler:
    """
    DAG-driven Dual-Channel Communication Scheduler.

    同时管理 XGMI + RDMA 两条独立通道，实现通信操作的真正并发。
    这是 MoEPackage 相对 Megatron-Core 的最大差异化优势。

    调度策略：
      节点内 Expert A2A → XGMI P2P（消除 RCCL 开销）
      跨节点 Expert A2A → RDMA RoCE（与 XGMI 并发）
      梯度 AllReduce   → RDMA（与 Expert GEMM 反向重叠）
      FSEP RS          → XGMI In-Place（消除 RCCL）

    Timeline（FWD 示例）：
      XGMI: [Dispatch_local] [RS_FSEP]         [Combine_local]
      RDMA:     [Dispatch_remote]   [Expert_GEMM]   [Combine_remote]
      GPU:  [Gate]  ──overlap──→  [Expert GEMM]  ──overlap──→

    Usage:
        scheduler = DualChannelScheduler()
        tasks = scheduler.build_moe_forward_dag(layer_id=0, ...)
        timeline = scheduler.schedule(tasks)
        scheduler.execute(tasks)
    """

    def __init__(self):
        self.xgmi_channel = CommChannel.create_xgmi()
        self.rdma_channel = CommChannel.create_rdma()
        self._task_counter = 0
        self._timeline: List[Dict] = []

    def _next_task_id(self) -> int:
        self._task_counter += 1
        return self._task_counter

    def get_channel(self, channel_type: ChannelType) -> CommChannel:
        """Get the physical channel for a given type."""
        if channel_type == ChannelType.XGMI:
            return self.xgmi_channel
        elif channel_type == ChannelType.RDMA:
            return self.rdma_channel
        return self.xgmi_channel  # LOCAL tasks don't use a channel

    # -----------------------------------------------------------------------
    # DAG Construction
    # -----------------------------------------------------------------------

    def build_moe_forward_dag(
        self,
        layer_id: int,
        num_tokens: int,
        d_model: int,
        top_k: int = 8,
        intra_ratio: float = 0.75,
        bytes_per_element: int = 2,
    ) -> List[CommTask]:
        """
        Build a DAG for one MoE layer forward pass.

        构建单层 MoE 前向的完整任务 DAG，包括：
        Gate → Dispatch (XGMI + RDMA 并发) → Expert GEMM → Combine (并发)

        Args:
            layer_id: MoE Layer 编号
            num_tokens: Token 数
            d_model: 隐藏层维度
            top_k: TopK 路由
            intra_ratio: 节点内 token 占比
            bytes_per_element: 每元素字节数

        Returns:
            所有 task 的列表
        """
        total_bytes = num_tokens * top_k * d_model * bytes_per_element
        intra_bytes = int(total_bytes * intra_ratio)
        inter_bytes = total_bytes - intra_bytes

        tasks: List[CommTask] = []

        # Task 1: Gate 路由计算（本地）
        gate_task = CommTask(
            task_id=self._next_task_id(),
            task_type=TaskType.GATE_COMPUTE,
            layer_id=layer_id,
            data_bytes=0,
        )
        gate_task.estimated_time_us = 20.0  # Gate GEMM ~20μs
        tasks.append(gate_task)

        # Task 2: 节点内 Dispatch（XGMI）
        dispatch_local = CommTask(
            task_id=self._next_task_id(),
            task_type=TaskType.DISPATCH_LOCAL,
            layer_id=layer_id,
            data_bytes=intra_bytes,
        )
        dispatch_local.estimated_time_us = (
            self.xgmi_channel.estimate_transfer_time_us(intra_bytes)
        )
        dispatch_local.add_dependency(gate_task)
        tasks.append(dispatch_local)

        # Task 3: 跨节点 Dispatch（RDMA）— 与 XGMI 并发！
        dispatch_remote = CommTask(
            task_id=self._next_task_id(),
            task_type=TaskType.DISPATCH_REMOTE,
            layer_id=layer_id,
            data_bytes=inter_bytes,
        )
        dispatch_remote.estimated_time_us = (
            self.rdma_channel.estimate_transfer_time_us(inter_bytes)
        )
        dispatch_remote.add_dependency(gate_task)
        tasks.append(dispatch_remote)

        # Task 4: Expert GEMM 计算（本地）— 等待所有 dispatch 完成
        expert_gemm = CommTask(
            task_id=self._next_task_id(),
            task_type=TaskType.EXPERT_GEMM,
            layer_id=layer_id,
            data_bytes=0,
        )
        expert_gemm.estimated_time_us = 500.0  # Expert GEMM ~500μs
        expert_gemm.add_dependency(dispatch_local)
        expert_gemm.add_dependency(dispatch_remote)
        tasks.append(expert_gemm)

        # Task 5: 节点内 Combine（XGMI）
        combine_local = CommTask(
            task_id=self._next_task_id(),
            task_type=TaskType.COMBINE_LOCAL,
            layer_id=layer_id,
            data_bytes=intra_bytes,
        )
        combine_local.estimated_time_us = (
            self.xgmi_channel.estimate_transfer_time_us(intra_bytes)
        )
        combine_local.add_dependency(expert_gemm)
        tasks.append(combine_local)

        # Task 6: 跨节点 Combine（RDMA）— 与 XGMI Combine 并发！
        combine_remote = CommTask(
            task_id=self._next_task_id(),
            task_type=TaskType.COMBINE_REMOTE,
            layer_id=layer_id,
            data_bytes=inter_bytes,
        )
        combine_remote.estimated_time_us = (
            self.rdma_channel.estimate_transfer_time_us(inter_bytes)
        )
        combine_remote.add_dependency(expert_gemm)
        tasks.append(combine_remote)

        return tasks

    # -----------------------------------------------------------------------
    # Scheduling: Timeline-based dual-channel scheduling
    # -----------------------------------------------------------------------

    def schedule(self, tasks: List[CommTask]) -> List[Dict]:
        """
        Schedule tasks onto dual channels with timeline estimation.

        使用贪心策略：优先级高的任务先调度，同一通道内串行，
        不同通道之间可以并发。

        Args:
            tasks: DAG 中所有任务

        Returns:
            Timeline 记录列表
        """
        # 通道时间线：记录每条通道的当前结束时间
        channel_end_time = {
            ChannelType.XGMI: 0.0,
            ChannelType.RDMA: 0.0,
            ChannelType.LOCAL: 0.0,
        }

        # 按拓扑序 + 优先级排列
        scheduled: List[Dict] = []
        remaining = list(tasks)

        while remaining:
            # 找出所有可调度的 task（前驱已完成）
            ready = [t for t in remaining if t.is_ready()]
            if not ready:
                # 不应出现死锁；如果出现，标记所有剩余 task 为 done
                for t in remaining:
                    t.status = TaskStatus.DONE
                break

            # 优先级排序
            ready.sort(key=lambda t: -t.priority)

            task = ready[0]
            remaining.remove(task)

            # 计算最早开始时间 = max(前驱结束, 通道可用)
            pred_end = max(
                (p.end_time_us for p in task.predecessors),
                default=0.0,
            )
            channel_available = channel_end_time[task.channel]
            start_time = max(pred_end, channel_available)

            task.start_time_us = start_time
            task.end_time_us = start_time + task.estimated_time_us
            task.status = TaskStatus.DONE

            # 更新通道时间线
            channel_end_time[task.channel] = task.end_time_us

            scheduled.append({
                'task_id': task.task_id,
                'type': task.task_type.name,
                'channel': task.channel.name,
                'start_us': round(task.start_time_us, 1),
                'end_us': round(task.end_time_us, 1),
                'duration_us': round(task.estimated_time_us, 1),
                'data_bytes': task.data_bytes,
            })

        self._timeline = scheduled
        return scheduled

    def execute(self, tasks: List[CommTask]) -> None:
        """
        Execute scheduled tasks (simulation mode).

        实际生产中：
          XGMI 任务 → HIP P2P API（hipMemcpyPeer / custom kernel）
          RDMA 任务 → ibverbs API（ibv_post_send / GDR）
          LOCAL 任务 → HIP Kernel Launch

        三条执行路径互不阻塞。
        """
        for task in tasks:
            if task.callback is not None:
                task.callback()
            task.status = TaskStatus.DONE

            # 更新通道统计
            channel = self.get_channel(task.channel)
            channel.utilization_bytes += task.data_bytes
            channel.total_tasks_completed += 1

    # -----------------------------------------------------------------------
    # Bandwidth Analysis：带宽分析
    # -----------------------------------------------------------------------

    def estimate_effective_bandwidth(
        self,
        total_data_bytes: int,
        intra_ratio: float = 0.75,
    ) -> Dict[str, float]:
        """
        Estimate effective bandwidth with dual-channel concurrency.

        AMD 独有优势：XGMI + RDMA 物理独立，可叠加。
        effective_bw = xgmi_bw + rdma_bw（而非 max）

        Args:
            total_data_bytes: 总传输数据量
            intra_ratio: 节点内占比

        Returns:
            带宽估算字典
        """
        intra_bytes = total_data_bytes * intra_ratio
        inter_bytes = total_data_bytes * (1 - intra_ratio)

        # AMD 双通道并发
        xgmi_time_us = self.xgmi_channel.estimate_transfer_time_us(
            int(intra_bytes)
        )
        rdma_time_us = self.rdma_channel.estimate_transfer_time_us(
            int(inter_bytes)
        )
        # 双通道并发 → 总时间取 max
        amd_total_us = max(xgmi_time_us, rdma_time_us)
        amd_effective_bw = total_data_bytes / (amd_total_us * 1e-6) / 1e9

        # NVIDIA 对比（NVLink + IB 共享 NCCL，近似取 max 而非简单加法）
        # 注意：NVIDIA 的 NVLink 和 IB 虽然可以部分重叠，但受限于单一 NCCL 通信栈，
        # 实际并发度远低于物理独立通道。此处以 max + 10% 串行化开销为保守估计。
        nv_intra_us = intra_bytes / (NVIDIA_NVLINK_BW_GBS * 1e3)
        nv_inter_us = inter_bytes / (NVIDIA_IB_BW_GBS * 1e3)
        nv_time_us = max(nv_intra_us, nv_inter_us) * 1.1  # 10% 串行化开销
        nv_effective_bw = total_data_bytes / (nv_time_us * 1e-6) / 1e9

        return {
            'amd_xgmi_time_us': round(xgmi_time_us, 1),
            'amd_rdma_time_us': round(rdma_time_us, 1),
            'amd_total_time_us': round(amd_total_us, 1),
            'amd_effective_bw_gbs': round(amd_effective_bw, 1),
            'nvidia_total_time_us': round(nv_time_us, 1),
            'nvidia_effective_bw_gbs': round(nv_effective_bw, 1),
            'amd_speedup': round(nv_time_us / max(amd_total_us, 0.01), 2),
            'bw_advantage_pct': round(
                (amd_effective_bw - nv_effective_bw)
                / max(nv_effective_bw, 0.01) * 100, 1
            ),
        }

    def print_timeline(self) -> str:
        """
        Format the scheduling timeline as a readable string.

        展示 XGMI / RDMA / LOCAL 三条通道的任务执行时间线。
        """
        if not self._timeline:
            return "No timeline available. Call schedule() first."

        lines = [
            "=" * 80,
            "Dual-Channel Scheduling Timeline",
            "=" * 80,
            f"{'Task':<25} {'Channel':<8} {'Start(μs)':<12} "
            f"{'End(μs)':<12} {'Duration(μs)':<14} {'Data(KB)'}",
            "-" * 80,
        ]

        for entry in self._timeline:
            data_kb = entry['data_bytes'] / 1024
            lines.append(
                f"{entry['type']:<25} {entry['channel']:<8} "
                f"{entry['start_us']:<12.1f} {entry['end_us']:<12.1f} "
                f"{entry['duration_us']:<14.1f} {data_kb:.1f}"
            )

        # 总结
        total_time = max(e['end_us'] for e in self._timeline) if self._timeline else 0
        lines.extend([
            "-" * 80,
            f"Total wall-clock time: {total_time:.1f} μs",
            f"XGMI utilization: {self.xgmi_channel.utilization_bytes / 1024:.1f} KB",
            f"RDMA utilization: {self.rdma_channel.utilization_bytes / 1024:.1f} KB",
            "=" * 80,
        ])

        return '\n'.join(lines)

    def reset(self) -> None:
        """Reset scheduler state for new step."""
        self._task_counter = 0
        self._timeline = []
        self.xgmi_channel.reset_stats()
        self.rdma_channel.reset_stats()

    def __repr__(self) -> str:
        return (
            f"DualChannelScheduler(\n"
            f"  xgmi: {self.xgmi_channel.bandwidth_gbs} GB/s, "
            f"rdma: {self.rdma_channel.bandwidth_gbs} GB/s,\n"
            f"  effective_bw: xgmi + rdma = "
            f"{self.xgmi_channel.bandwidth_gbs + self.rdma_channel.bandwidth_gbs} GB/s "
            f"(vs NVIDIA max = ~900 GB/s),\n"
            f"  tasks_scheduled: {self._task_counter}\n"
            f")"
        )
