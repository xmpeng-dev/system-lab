# MoEPackage：面向 AMD GPU 的 MoE 训练全栈优化包

> **定位：** 基于 Megatron-Core MoE 全栈分析 + 10 篇核心论文综合，为 AMD MI300X 原生设计的 MoE 训练加速包  
> **目标硬件：** AMD MI300X / MI325X（Instinct 系列）  
> **对标基线：** Megatron-Core v0.16 on GB200（1,233 TFLOPS）/ H100（368 TFLOPS）  
> **核心理念：** 不重造轮子——找到 Megatron-Core 做不到的、AMD 硬件能做到的差异化优化

---

## 核心定位

```
MoEPackage 不是 Megatron-Core 的 AMD 移植版。
MoEPackage 是利用 AMD 硬件差异化特性，在 Megatron-Core 做不到/做不好的 4 个缺口上，
提供原生最优解的 MoE 训练加速包。

四个核心模块 = 四个 Megatron-Core 缺口的 AMD 原生解：

  缺口 ⑥ DeepEP 不可移植    → Module 1: XGMI-Native Expert Dispatch
  缺口 ⑨ NCCL UBR 不可用    → Module 2: Persistent P2P Buffer Pool
  缺口 ⑬ Device-Initiated   → Module 3: AMD Dropless GEMM (hipBLASLt)
  全行空白: AMD 硬件利用     → Module 4: Dual-Channel Comm Scheduler

加上跨模块的系统集成：
  整合器: Fused Permute-Quantize-Dispatch Pipeline
  → 将 Module 1~4 串联成端到端优化流水线
```

---

## 文档索引

| 文档 | 说明 | 路径 |
|------|------|------|
| **MoEPackage 完整设计** | Megatron-Core 分析 + 论文矩阵 + 四模块设计 + 性能预估 | [moe/MoEPackage_design.md](../moe/MoEPackage_design.md) |
| **Megatron-Core 详细笔记** | MoE 三面墙 + 14 项优化深度解析 | [moe/MegatronCore_MoE_reading_notes.md](../moe/MegatronCore_MoE_reading_notes.md) |
| **AMD 硬件分析** | MI300X XGMI/HBM3e/RDMA 特性 | [rocflow/README.md](../rocflow/README.md) |
| **FSEP AMD 原生实现** | LAER-MoE FSEP 在 AMD 上的超越方案 | [moe/research_direction_C.md](../moe/research_direction_C.md) |
| **100+ 论文全景图** | MoE 优化论文分类索引 | [moe/README.md](../moe/README.md) |

---

## 代码模块

```
moepackage/code/
├── __init__.py                  # 包入口，导出所有公共接口
├── xgmi_dispatch.py             # Module 1: XGMI-Native Expert Dispatch
├── p2p_buffer_pool.py           # Module 2: Persistent P2P Buffer Pool
├── dropless_gemm.py             # Module 3: AMD Dropless Grouped GEMM
├── dual_channel_scheduler.py    # Module 4: Dual-Channel Comm Scheduler
└── fused_pipeline.py            # 整合器：Fused Pipeline + MoEPackageLayer
```

### Module 1: XGMI-Native Expert Dispatch（`xgmi_dispatch.py`）

**替代目标：** DeepEP 的 TMA + IBGDA 方案（NVIDIA 专属硬件原语）

```python
from moepackage.code import XGMIDispatcher, XGMIDispatchConfig

config = XGMIDispatchConfig(
    num_experts=256,
    hidden_dim=7168,
    ep_size=64,
    gpus_per_node=8,
)
dispatcher = XGMIDispatcher(config)

# 节点内: XGMI P2P 直写，延迟 ~3μs (vs RCCL ~15μs)
# 跨节点: Two-Phase (聚合 → RDMA → 分发)
dispatched, metadata = dispatcher.dispatch(tokens, routing_map)
```

### Module 2: Persistent P2P Buffer Pool（`p2p_buffer_pool.py`）

**替代目标：** NCCL User Buffer Registration（RCCL 无等效 API）

```python
from moepackage.code import P2PBufferPool, BufferPoolConfig

config = BufferPoolConfig(
    num_peers=8,
    capacity_per_peer=8192,
    hidden_dim=7168,
)
pool = P2PBufferPool(config)
pool.initialize()  # 训练初始化一次

# 训练循环中: 零 malloc，零注册/注销
send_buf = pool.acquire(peer_id=3, usage=BufferUsage.DISPATCH)
```

### Module 3: AMD Dropless GEMM（`dropless_gemm.py`）

**替代目标：** CUDA Device-Initiated Kernels（依赖 CUDA 13.1+ Blackwell）

```python
from moepackage.code import DroplessGroupedGEMM, DroplessGEMMConfig

config = DroplessGEMMConfig(
    num_experts=64,
    hidden_dim=7168,
    ffn_dim=18432,
    safety_factor=1.5,
)
gemm = DroplessGroupedGEMM(config)

# 静态形状 → HIP Graph 可完整捕获
# Padding + valid_mask → 无 drop，无 host-device sync
output = gemm.forward(expert_inputs, token_counts)
```

### Module 4: Dual-Channel Comm Scheduler（`dual_channel_scheduler.py`）

**AMD 独有优势：** XGMI 和 RDMA 是物理独立通道，可真正并发

```python
from moepackage.code import DualChannelScheduler, CommTask, TaskType, ChannelType

scheduler = DualChannelScheduler(
    xgmi_bw_gbs=896.0,
    rdma_bw_gbs=50.0,
)

# 自动分配到最优通道
scheduler.submit(CommTask(TaskType.EXPERT_DISPATCH_LOCAL, data_bytes=...))  # → XGMI
scheduler.submit(CommTask(TaskType.EXPERT_DISPATCH_REMOTE, data_bytes=...))  # → RDMA
scheduler.execute_all()  # XGMI + RDMA 真正并发执行

# 有效带宽 = XGMI + RDMA = 896 + 50 = 946 GB/s
# vs NVIDIA: 有效带宽 ≈ max(NVLink, IB)
```

### Fused Pipeline Integrator（`fused_pipeline.py`）

**端到端 MoE 层：** 串联 Module 1~4 为完整前向/反向流水线

```python
from moepackage.code import create_moepackage_layer, FusedPipelineConfig

config = FusedPipelineConfig(
    num_experts=256,
    expert_parallel_size=64,
    hidden_dim=7168,
    ffn_dim=18432,
    top_k=8,
    gpus_per_node=8,
)
moe_layer = create_moepackage_layer(config)

# 前向: Gate → Fused Dispatch → Expert GEMM → Fused Combine
output = moe_layer(hidden_states)

# Dispatch HBM 流量减少 80%:
#   Megatron-Core: 5 × T×K×H×2B = 2.24 GB
#   MoEPackage:    1 × T×K×H×2B = 0.45 GB
```

---

## 性能预估

以 DeepSeek-V3 on 256 MI300X（EP=64）为例：

| 对比项 | TFLOPS | MFU |
|--------|--------|-----|
| Megatron-Core on MI300X（基线移植） | ~486 | ~37% |
| **MoEPackage on MI300X** | **~680-760** | **~52-58%** |
| Megatron-Core on H100 | 368 | ~37% |
| Megatron-Core on GB200 | 1,048 | ~85% |

**MoEPackage MI300X ≈ 1.8~2.0× H100，≈ 0.65~0.72× GB200**

---

## 与其他子项目的关系

```
moepackage/     ← 核心加速引擎（可独立使用）
    ↑
rocflow/        ← 完整 AMD MoE 训练框架（使用 MoEPackage 作为后端）
    ↑
axion/          ← 通信优先的稀疏训练运行时（IR 层调度）
    ↑
moex/           ← CommTensor 系统（MoEPackage 的 Fused Pipeline 是其最小实现）
```

---

*MoEPackage 设计原型 | 2026-03-12 | AIInfra-Book*
