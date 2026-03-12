"""
MoEPackage: AMD GPU MoE Training Optimization Package

面向 AMD MI300X/MI325X 的 MoE 训练全栈优化方案。
不是 Megatron-Core 的 AMD 移植版，而是利用 AMD 硬件差异化特性，
在 Megatron-Core 做不到/做不好的 4 个缺口上提供原生最优解。

四个核心模块 = 四个 Megatron-Core 缺口的 AMD 原生解：
  Module 1: XGMI-Native Expert Dispatch  ← 替代 DeepEP (TMA+IBGDA)
  Module 2: Persistent P2P Buffer Pool   ← 替代 NCCL User Buffer Registration
  Module 3: AMD Dropless GEMM            ← 替代 Device-Initiated Kernels
  Module 4: Dual-Channel Comm Scheduler  ← AMD 独有 XGMI+RDMA 双通道并发

跨模块整合器：
  Fused Permute-Quantize-Dispatch Pipeline
  → 单 HIP Kernel 完成 Permute + FP8 Quant + P2P Write
  → Dispatch 流水线 HBM 流量减少 80%
"""

from .xgmi_dispatch import (
    XGMIDispatcher,
    XGMIDispatchConfig,
    P2PAddressMap,
    DispatchMode,
)

from .p2p_buffer_pool import (
    P2PBufferPool,
    BufferPoolConfig,
    BufferSlot,
    BufferUsage,
)

from .dropless_gemm import (
    DroplessGroupedGEMM,
    DroplessGEMMConfig,
    ExpertCapacityManager,
    OverflowPolicy,
    GEMMBackend,
)

from .dual_channel_scheduler import (
    DualChannelScheduler,
    CommChannel,
    CommTask,
    ChannelType,
    TaskType,
    TaskStatus,
)

from .fused_pipeline import (
    MoEPackageLayer,
    FusedPipelineConfig,
    create_moepackage_layer,
)

__all__ = [
    # Module 1: XGMI-Native Expert Dispatch
    'XGMIDispatcher',
    'XGMIDispatchConfig',
    'P2PAddressMap',
    'DispatchMode',
    # Module 2: Persistent P2P Buffer Pool
    'P2PBufferPool',
    'BufferPoolConfig',
    'BufferSlot',
    'BufferUsage',
    # Module 3: AMD Dropless GEMM
    'DroplessGroupedGEMM',
    'DroplessGEMMConfig',
    'ExpertCapacityManager',
    'OverflowPolicy',
    'GEMMBackend',
    # Module 4: Dual-Channel Comm Scheduler
    'DualChannelScheduler',
    'CommChannel',
    'CommTask',
    'ChannelType',
    'TaskType',
    'TaskStatus',
    # Fused Pipeline Integrator
    'MoEPackageLayer',
    'FusedPipelineConfig',
    'create_moepackage_layer',
]
