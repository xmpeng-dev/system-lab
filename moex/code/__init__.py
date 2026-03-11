"""
MoEX: Communication-First MoE Training System

设计原则：让 tensor 的物理存储顺序 = 通信目标顺序
从最底层的 tensor 存储结构出发，使 tensor 本身就适合：
  - dispatch（零拷贝 RDMA）
  - combine（零拷贝 scatter_add）
  - overlap（tile 级 GEMM-RDMA pipeline）
  - 异构并行（FSEP + Parallel Folding）
"""

from .comm_tensor import (
    CommTensor,
    CommTensorConfig,
    CommTensorMeta,
    CommTensorPool,
    load_balance_loss,
    route_to_comm_tensor,
)

from .moex_layer import (
    ExpertFFN,
    ExpertEngine,
    MoEXRouter,
    MoEXLayer,
    MoEXBlock,
    create_moex_layer,
)

from .overlap_scheduler import (
    OverlapScheduler,
    StreamManager,
    TilePipeline,
    TaskType,
    MoEXTask,
    SchedulerStats,
)

__all__ = [
    # CommTensor
    'CommTensor',
    'CommTensorConfig',
    'CommTensorMeta',
    'CommTensorPool',
    'load_balance_loss',
    'route_to_comm_tensor',
    # MoEX Layer
    'ExpertFFN',
    'ExpertEngine',
    'MoEXRouter',
    'MoEXLayer',
    'MoEXBlock',
    'create_moex_layer',
    # Overlap Scheduler
    'OverlapScheduler',
    'StreamManager',
    'TilePipeline',
    'TaskType',
    'MoEXTask',
    'SchedulerStats',
]
