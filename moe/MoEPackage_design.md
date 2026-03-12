# MoEPackage：面向 AMD GPU 的 MoE 训练全栈优化方案

> **定位：** 基于 Megatron-Core MoE 全栈分析 + 10 篇核心论文综合，设计 AMD MI300X 原生的 MoE 训练优化方案  
> **目标硬件：** AMD MI300X / MI325X（Instinct 系列）  
> **对标基线：** Megatron-Core v0.16 on GB200（1,233 TFLOPS）/ H100（368 TFLOPS）  
> **核心理念：** 不重造轮子——找到 Megatron-Core 做不到的、AMD 硬件能做到的差异化优化  
> **更新：** 2026-03-12

---

## 第一部分：Megatron-Core MoE 优化全景解剖

### 1.1 三面墙与对应优化

Megatron-Core v0.16 将 MoE 训练的瓶颈归为三面墙，每面墙都有对应的完整优化栈：

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    Megatron-Core MoE 优化全景（v0.16）                            │
├───────────────┬──────────────────────────────────────┬──────────────────────────┤
│   内存墙       │   通信墙                              │   计算效率墙              │
├───────────────┼──────────────────────────────────────┼──────────────────────────┤
│               │                                      │                          │
│ ① Memory-    │ ⑥ DeepEP / HybridEP                  │ ⑩ Grouped GEMM           │
│   Efficient   │   Token-level dispatch               │   (cuBLASLt / CUTLASS)   │
│   Permutation │   TMA + IBGDA 硬件原语                │   多 Expert 合并单 Kernel  │
│   节省 26 GB   │   HybridEP: NVLink+RDMA 分层        │                          │
│               │                                      │ ⑪ 三级 Kernel 融合       │
│ ② 细粒度重算  │ ⑦ EP 通信重叠（1F1B Merged）          │   Permutation Fusion     │
│   MLA/SwiGLU/ │   FWD micro_batch_i BWD 合并         │   Router Fusion          │
│   LayerNorm   │   双 CUDA Stream: Compute + Comm     │   Aux-Loss Fusion        │
│   节省 42 GB   │   W/D 分割: dW 与 dispatch 并行      │                          │
│               │   93% EP 通信隐藏                     │ ⑫ CUDA Graphs            │
│ ③ 激活 Offload│                                      │   Full(drop-and-pad)     │
│   D2H 异步    │ ⑧ Parallel Folding                   │   Partial(layered)       │
│   Layer-      │   Attention: TP×CP×DP                │   +10% 端到端加速         │
│   Staggered   │   MoE: ETP×EP×EDP                    │                          │
│   Reload      │   PP 一致，其余独立                    │ ⑬ Device-Initiated       │
│   仅 -1.6%    │   EP 不受 DP 限制                     │   GPU 直读形状信息        │
│               │                                      │   Sync-Free Dispatch     │
│ ④ FSDP + EP  │ ⑨ 持久双 Buffer                      │   ECHO 弹性克隆           │
│   双 DeviceMesh│   NCCL User Buffer Registration     │   Paged Stashing         │
│   非均匀分片  │   SM 占用 8-32→1-4 SM                │   Full CUDA Graph        │
│   零拷贝通信  │   RDMA 可用 IBGDA                     │   for Dropless MoE       │
│               │                                      │                          │
│ ⑤ FP8/FP4    │                                      │ ⑭ 长上下文支持           │
│   MXFP8(BW)  │                                      │   CP + TP 层次化          │
│   Blockwise   │                                      │   Dynamic-CP             │
│   (Hopper)    │                                      │   Packed Sequences       │
│   NVFP4(BW)  │                                      │   THD 格式               │
│   选择性精度  │                                      │                          │
└───────────────┴──────────────────────────────────────┴──────────────────────────┘
```

### 1.2 Megatron-Core 优化的详细分解

#### A. 内存墙优化（5 项）

| # | 优化 | 机制 | 效果 | AMD 可移植性 |
|---|------|------|------|-------------|
| ① | Memory-Efficient Permutation | 路由权重在 W2 线性层前应用（数学等价） | -26.3 GB/GPU | ✅ 纯算法，直接移植 |
| ② | 细粒度重算 | 仅重算 MLA/SwiGLU/LayerNorm（内存大/计算小的操作） | -42.4 GB，<5% 计算开销 | ✅ 直接移植 |
| ③ | 激活 Offload | GPU Copy Engine 异步 D2H，Layer-Staggered Reload | -10.7% 内存，-1.6% 吞吐 | ✅ 直接移植（MI300X PCIe Gen5） |
| ④ | FSDP + EP | 双 DeviceMesh 架构，非均匀分片，持久双 Buffer + NCCL UBR | 灵活参数分片 | ⚠️ 需适配 RCCL（无 User Buffer Registration） |
| ⑤ | FP8/FP4 训练 | MXFP8（Blackwell 原生）/ Blockwise FP8 / NVFP4 | +22% 性能（FP8 vs BF16） | ⚠️ MI300X FP8 支持不同（见 1.3） |

#### B. 通信墙优化（4 项）

| # | 优化 | 机制 | 效果 | AMD 可移植性 |
|---|------|------|------|-------------|
| ⑥ | DeepEP / HybridEP | Token-level dispatch，TMA+IBGDA 硬件原语 | 延迟 675μs（vs A2A 930μs） | ❌ **NVIDIA 专属**（TMA/IBGDA） |
| ⑦ | EP 通信重叠 | 1F1B FWD-BWD Merged + 双 CUDA Stream + W/D Split | 93% EP 通信隐藏 | ⚠️ 需重写为 HIP Stream（架构相同） |
| ⑧ | Parallel Folding | Attention/MoE 独立并行配置 | 打破 EP≤DP 限制 | ✅ 纯策略层，直接移植 |
| ⑨ | 持久双 Buffer | NCCL User Buffer Registration + RDMA MR | SM 占用降低 8× | ⚠️ RCCL 无等效 API（需 P2P 替代） |

#### C. 计算效率墙优化（5 项）

| # | 优化 | 机制 | 效果 | AMD 可移植性 |
|---|------|------|------|-------------|
| ⑩ | Grouped GEMM | cuBLASLt / CUTLASS / cuteDSL | 多 Expert 合并 | ⚠️ 需用 hipBLASLt / composable_kernel |
| ⑪ | 三级 Kernel 融合 | Permutation/Router/Aux-Loss 融合 | 减少 Kernel Launch | ⚠️ 需用 HIP 重写 |
| ⑫ | CUDA Graphs | Full（静态）/ Partial（分层）| +10% 端到端 | ⚠️ HIP Graph 支持但成熟度不同 |
| ⑬ | Device-Initiated Kernels | GPU 直读形状 + ECHO + Paged Stashing | Dropless MoE Full Graph | ❌ **依赖 CUDA 13.1+**（Blackwell） |
| ⑭ | 长上下文 | CP+TP 层次化 / Dynamic-CP / Packed Seq | 256K seq 达短序列 88% MFU | ✅ 纯策略层，直接移植 |

### 1.3 Megatron-Core 做不到 / 做不好的点（AMD 差异化机会）

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  Megatron-Core 的 14 项优化在 AMD 上的状态                     │
├──────────────────────────────┬────────────────────────────────────────────┤
│  可直接移植（6 项）            │  ①②③⑤(BF16)⑧⑭                           │
│  需适配但架构相同（4 项）       │  ④⑦⑩⑫                                   │
│  需要 AMD 原生替代方案（2 项） │  ⑨⑪                                      │
│  完全不可移植（2 项）          │  ⑥⑬                                      │
├──────────────────────────────┴────────────────────────────────────────────┤
│                                                                           │
│  ⑥ DeepEP/HybridEP 完全不可移植 ← 这是最大的缺口                          │
│     TMA（Tensor Memory Accelerator）= NVIDIA Hopper 专属                  │
│     IBGDA（InfiniBand GPUDirect Async）= NVIDIA IB 专属                   │
│     → AMD 需要全新的 Expert Dispatch 方案                                  │
│                                                                           │
│  ⑬ Device-Initiated Kernels 不可移植 ← 第二大缺口                         │
│     依赖 CUDA 13.1 + cuBLASLt device-side API（Blackwell+）               │
│     → AMD 需要替代的 Dropless MoE + CUDA Graph 方案                       │
│                                                                           │
│  ⑨ 持久 Buffer + NCCL UBR ← 第三大缺口                                   │
│     RCCL 没有 User Buffer Registration API                                │
│     → AMD 的 XGMI P2P Direct Access 是天然替代（甚至更优）                 │
│                                                                           │
│  ⑤ FP8 训练 ← 精度方案不同                                                │
│     MI300X: MFMA FP8 = 2615 TFLOPS，但无 MXFP8/NVFP4 硬件支持           │
│     → 需要 Blockwise FP8 或 AMD 特定的量化方案                            │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 第二部分：论文优化技术统一分类

### 2.1 十篇论文的优化矩阵

```
维度               MC-MoE    DeepEP   FlowMoE   Comet    LAER    MoEBlaze  MemFine  Swift   MScale  Folding
────────────────────────────────────────────────────────────────────────────────────────────────────────────
Tensor 布局优化     ●         ○        ○         ○        ○       ●         ○        ○       ○       ○
Dispatch 内核       ●●        ●●●      ○         ○        ●       ●●        ○        ○       ○       ○
Expert GEMM        ●●        ○        ○         ●●       ●       ○         ○        ○       ●       ○
Kernel 融合        ●●        ●        ○         ●●●      ○       ●●        ○        ○       ○       ○
通信-计算 Overlap  ●●        ●●       ●●●       ●●●●     ●●      ○         ●        ○       ●●      ●●
跨层调度           ●         ○        ●●●●      ●        ○       ○         ●        ○       ○       ○
负载均衡           ●         ○        ○         ○        ●●●●    ○         ○        ●●●     ●●      ○
内存优化           ●●●       ○        ●●        ●        ○       ●●●●      ●●●●     ○       ○       ○
量化训练           ●●●●      ○        ○         ○        ○       ○         ○        ○       ○       ○
并行策略           ●●●●      ○        ○         ○        ●●      ○         ○        ○       ●●●     ●●●●
拓扑感知           ●         ●        ○         ○        ○       ○         ○        ○       ●●●●    ○
容错               ○         ○        ○         ○        ○       ○         ○        ○       ●●●●    ○
CUDA Graph         ●●●       ○        ○         ○        ○       ○         ○        ○       ○       ○
AMD 硬件利用       ○         ○        ○         ○        ○       ○         ○        ○       ○       ○

● = 涉及程度（越多越深）  ○ = 未涉及
```

### 2.2 从论文矩阵看 MoEPackage 的定位

```
每一列的空白 = 该方向的创新机会
每一行的空白 = 该论文可以借鉴的方向

MoEPackage 应该填充的核心空白：
  ① "AMD 硬件利用" 行全为 ○ → 所有优化都未针对 AMD 设计
  ② "Tensor 布局优化" 仅 MC 和 MoEBlaze 涉及 → CommTensor/ExpertSlotTensor 是创新点
  ③ Dispatch 内核在 AMD 上空白 → DeepEP 不可移植，需要 AMD 原生替代
  ④ 没有论文同时做到 "拓扑感知" + "负载均衡" + "Overlap" → 叠加是系统贡献
```

---

## 第三部分：MoEPackage 总体设计

### 3.1 核心定位

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

### 3.2 系统架构

```
╔══════════════════════════════════════════════════════════════════════════╗
║                        MoEPackage 系统架构                               ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  ┌────────────────────────────────────────────────────────────────────┐ ║
║  │              Module 4: Dual-Channel Comm Scheduler                 │ ║
║  │                                                                    │ ║
║  │  统一 DAG 调度器，同时管理：                                        │ ║
║  │  • XGMI 通道（节点内 ReduceScatter / P2P 直访 / Expert 重布局）    │ ║
║  │  • RDMA 通道（跨节点 All-to-All / 梯度 AllReduce）                 │ ║
║  │  两条通道物理独立，真正并发（非 NVIDIA 的 NVLink+IB 共享 NCCL）     │ ║
║  │                                                                    │ ║
║  │  调度策略：                                                        │ ║
║  │  • 节点内 Expert A2A → XGMI P2P（消除 RCCL 开销）                  │ ║
║  │  • 跨节点 Expert A2A → RDMA RoCE（与 XGMI 并发）                   │ ║
║  │  • 梯度 AllReduce → RDMA（与 Expert GEMM 反向重叠）                │ ║
║  │  • FSEP ReduceScatter → XGMI In-Place Reduction（消除 RCCL）       │ ║
║  └──────────────────────────┬─────────────────────────────────────────┘ ║
║                              │                                           ║
║  ┌──────────────────────────▼─────────────────────────────────────────┐ ║
║  │         Fused Pipeline: Permute → Quantize → Dispatch               │ ║
║  │                                                                     │ ║
║  │  Megatron-Core 当前：4 次独立 HBM 读写（Permute + Pack + Quant）   │ ║
║  │  MoEPackage：单 HIP Kernel 完成 Permute+FP8Quant+写入 P2P Buffer  │ ║
║  │  → 减少 2 次 HBM round-trip（~50% dispatch 侧带宽节省）            │ ║
║  └──────────────────────────┬─────────────────────────────────────────┘ ║
║                              │                                           ║
║  ┌────────────┬─────────────▼──────────┬──────────────────────────────┐ ║
║  │ Module 1   │ Module 2               │ Module 3                     │ ║
║  │            │                        │                              │ ║
║  │ XGMI-     │ Persistent P2P         │ AMD Dropless                 │ ║
║  │ Native    │ Buffer Pool            │ GEMM                         │ ║
║  │ Expert    │                        │                              │ ║
║  │ Dispatch  │ • 训练开始一次性分配    │ • hipBLASLt Grouped GEMM     │ ║
║  │           │ • XGMI P2P 注册        │ • 上界静态启动 +             │ ║
║  │ • 节点内: │ • RDMA MR 注册         │   运行时跳过多余计算          │ ║
║  │   P2P     │ • 零 malloc 延迟       │ • HIP Graph 兼容             │ ║
║  │   直写    │ • 前后向共享            │ • 替代 NVIDIA 的             │ ║
║  │ • 跨节点: │   同一物理内存          │   Device-Initiated Kernel    │ ║
║  │   RDMA    │ • 类 MC 持久双 Buffer   │                              │ ║
║  │   RoCE    │   但用 P2P 替代 UBR    │                              │ ║
║  └────────────┴────────────────────────┴──────────────────────────────┘ ║
║                              │                                           ║
║  ┌──────────────────────────▼─────────────────────────────────────────┐ ║
║  │                    AMD 硬件抽象层（HAL）                             │ ║
║  │                                                                     │ ║
║  │  XGMI P2P: hipMemcpyPeer / hipIpcGetMemHandle / hsa_amd_memory_*  │ ║
║  │  RDMA:     ibv_post_send / GPUDirect RDMA for ROCm                │ ║
║  │  GEMM:     hipBLASLt / composable_kernel / rocBLAS                │ ║
║  │  Graph:    hipGraphLaunch / hipStreamBeginCapture                  │ ║
║  │  Memory:   hipMalloc / hipMemPool / hipMallocAsync                │ ║
║  └─────────────────────────────────────────────────────────────────────┘ ║
╚══════════════════════════════════════════════════════════════════════════╝
```

### 3.3 四个核心模块详细设计

---

#### Module 1: XGMI-Native Expert Dispatch（替代 DeepEP）

**问题：** DeepEP 的核心加速来自 NVIDIA 专属的 TMA（Tensor Memory Accelerator）和 IBGDA（InfiniBand GPUDirect Async），这两个硬件原语在 AMD 上不存在。

**AMD 解法：**

```
DeepEP 在 NVIDIA 上的工作方式：
  Token → TMA 自动搬运到 Shared Memory → FIFO 队列 → 写入目标 GPU
  跨节点: 先 RDMA 交换 → 再节点内转发（利用 NVLink+IB 重叠）

MoEPackage 在 AMD 上的工作方式：

  节点内 Dispatch（替代 NVLink 路径）：
  ┌─────────────────────────────────────────────────────────┐
  │  XGMI P2P Direct Write                                  │
  │                                                         │
  │  GPU_src 的 HIP Kernel 直接写入 GPU_dst 的预注册 Buffer │
  │                                                         │
  │  __global__ void xgmi_dispatch_kernel(                  │
  │      half* __restrict__ src_tokens,    // 本卡 token     │
  │      half** __restrict__ dst_buffers,  // 各卡 P2P 指针  │
  │      int* routing_map,                                  │
  │      int* slot_offsets                                   │
  │  ) {                                                    │
  │      int token_id = blockIdx.x;                         │
  │      int dst_gpu = routing_map[token_id];               │
  │      int slot = atomicAdd(&slot_offsets[dst_gpu], 1);   │
  │      // 直接通过 XGMI 写入远端 GPU 的 HBM              │
  │      memcpy_async(                                      │
  │          dst_buffers[dst_gpu] + slot * H,               │
  │          src_tokens + token_id * H,                     │
  │          H * sizeof(half)                               │
  │      );                                                 │
  │  }                                                      │
  │                                                         │
  │  关键优势：                                              │
  │  • 无 RCCL 协议开销（P2P = 内存操作，不是通信操作）      │
  │  • XGMI 全互联拓扑：任意 GPU 对之间带宽相等             │
  │  • 延迟 ~3μs vs RCCL ~15μs（节点内）                   │
  │  • 带宽利用率 ~90% vs RCCL ~65%                        │
  └─────────────────────────────────────────────────────────┘

  跨节点 Dispatch（替代 IBGDA 路径）：
  ┌─────────────────────────────────────────────────────────┐
  │  Two-Phase Dispatch                                      │
  │                                                         │
  │  Phase 1: 节点内聚合（XGMI P2P）                        │
  │    同节点 8 GPU 的跨节点 token → 聚合到 1 个 Gateway GPU │
  │    聚合 = XGMI P2P Direct Write（节点内零开销）          │
  │                                                         │
  │  Phase 2: 跨节点发送（RDMA RoCE）                       │
  │    Gateway GPU → RDMA 发送到目标节点的 Gateway GPU       │
  │    一次 RDMA 传输（vs 8 次独立小传输）                   │
  │    → 减少 RDMA 启动开销，提高网络带宽利用率              │
  │                                                         │
  │  Phase 3: 节点内分发（XGMI P2P）                        │
  │    目标节点 Gateway GPU → P2P 分发到各 Expert 所在 GPU   │
  │                                                         │
  │  与 DeepEP HybridEP 的本质相同（节点内外分层处理）       │
  │  但利用 XGMI P2P 替代 TMA+NVLink                       │
  └─────────────────────────────────────────────────────────┘
```

**性能预估：**
- 节点内 Dispatch 延迟：~50μs（vs DeepEP ~80μs on NVLink）→ **XGMI 更优**
- 跨节点 Dispatch 延迟：~800μs（vs DeepEP ~675μs on IB）→ **略慢**（RoCE vs IB）
- 综合（EP=64, 8 GPU/node）：大部分 token 节点内解决 → **整体接近持平或略优**

---

#### Module 2: Persistent P2P Buffer Pool（替代 NCCL User Buffer Registration）

**问题：** Megatron-Core 使用 NCCL User Buffer Registration 将通信 Buffer 持久注册，减少 SM 占用。RCCL 没有等效 API。

**AMD 解法：**

```
Megatron-Core on NVIDIA:
  持久 Buffer → NCCL UBR 注册 → NCCL 直接使用注册地址 → SM 占用 1-4 SM

MoEPackage on AMD:
  持久 Buffer → XGMI P2P IPC Handle 注册 → 各 GPU 直接使用远端地址 → 0 SM 占用

  ┌─────────────────────────────────────────────────────────┐
  │  Persistent P2P Buffer Pool                              │
  │                                                         │
  │  训练初始化阶段：                                        │
  │    for each EP rank r:                                  │
  │      send_buf[r] = hipMalloc(capacity * H * sizeof(T))  │
  │      recv_buf[r] = hipMalloc(capacity * H * sizeof(T))  │
  │      ipc_handle[r] = hipIpcGetMemHandle(send_buf[r])    │
  │                                                         │
  │  P2P 地址交换（一次性）：                                │
  │    All-to-All 交换 IPC Handle                           │
  │    各 GPU 获得所有其他 GPU 的 Buffer 远端指针            │
  │    peer_ptr[i][r] = hipIpcOpenMemHandle(handle_from_r)  │
  │                                                         │
  │  训练循环中：                                            │
  │    Dispatch: 直接写入 peer_ptr[dst_gpu][slot]           │
  │    Combine:  直接读取 peer_ptr[src_gpu][slot]           │
  │    无 malloc / free / 注册 / 注销                       │
  │                                                         │
  │  优势：                                                 │
  │  • 比 NCCL UBR 更彻底：零 SM 占用（P2P = 内存操作）    │
  │  • 无 RCCL 启动开销                                     │
  │  • 天然支持 HIP Graph（地址不变 → 图捕获安全）          │
  └─────────────────────────────────────────────────────────┘
```

---

#### Module 3: AMD Dropless GEMM（替代 Device-Initiated Kernels）

**问题：** Megatron-Core 使用 CUDA 13.1+ 的 Device-Initiated API 让 GPU 直接从设备内存读取 GEMM 形状信息，实现 Dropless MoE + Full CUDA Graph。AMD 没有等效 API。

**AMD 解法：**

```
Megatron-Core on Blackwell:
  cuBLASLt Grouped GEMM → device-side shape → 静态启动 + 运行时跳过

MoEPackage on MI300X:
  hipBLASLt Grouped GEMM → 预分配上界 + Mask 机制 + HIP Graph

  ┌─────────────────────────────────────────────────────────┐
  │  Padded Static Grouped GEMM                              │
  │                                                         │
  │  策略：每个 Expert 预分配 capacity 个 token 的空间       │
  │    capacity = ceil(avg_tokens_per_expert × safety_factor)│
  │    safety_factor = 1.5（基于 ECHO 思路动态调整）         │
  │                                                         │
  │  Dispatch 时：                                          │
  │    实际 token 数 ≤ capacity → 直接写入                  │
  │    实际 token 数 > capacity → 溢出 token 路由到克隆      │
  │    （ECHO 风格：热门 Expert 的权重广播到空闲 GPU）       │
  │                                                         │
  │  Expert GEMM：                                          │
  │    hipBLASLt Grouped GEMM 以 capacity 为静态形状启动     │
  │    实际 token < capacity 的 Expert → padding 区域计算    │
  │    padding 结果被 valid_mask 过滤（不影响输出）          │
  │                                                         │
  │  HIP Graph 兼容：                                       │
  │    所有形状静态 → hipGraph 可完整捕获 MoE 前向 + 反向    │
  │    无 host-device 同步                                   │
  │    无动态内存分配                                        │
  │                                                         │
  │  内存开销：                                              │
  │    padding 浪费 ≈ safety_factor - 1 ≈ 50%               │
  │    但通过 Paged Stashing 优化：                          │
  │      跨层共享一个 capacity 大小的 tmp buffer             │
  │      各层仅 stash 实际 token 到 paged buffer            │
  │      内存从 O(layers × capacity) → O(capacity + actual) │
  └─────────────────────────────────────────────────────────┘
```

---

#### Module 4: Dual-Channel Comm Scheduler（AMD 独有优势）

**问题：** NVIDIA 的 NVLink 和 InfiniBand 共享 NCCL 通信栈，软件层面无法真正并发。AMD 的 XGMI 和 RDMA 是物理独立通道。

**AMD 独有解法：**

```
NVIDIA 上的通信：
  NVLink 通信 ──┐
                ├── 都走 NCCL → 串行化 / 带宽竞争
  IB 通信 ──────┘

AMD 上的通信：
  XGMI 通信 ────── HIP P2P API ── 独立通道 ──→ 节点内 896 GB/s
  RDMA 通信 ────── ibverbs API ── 独立通道 ──→ 跨节点 400 Gbps

  ┌─────────────────────────────────────────────────────────┐
  │  Dual-Channel Comm Scheduler                             │
  │                                                         │
  │  通信操作分类：                                          │
  │                                                         │
  │  XGMI 通道（节点内高带宽低延迟）：                       │
  │    • Expert Dispatch（节点内部分）                       │
  │    • FSEP ReduceScatter                                 │
  │    • Expert Re-layout（参数搬迁）                        │
  │    • TP All-Reduce（如果 TP 在节点内）                   │
  │                                                         │
  │  RDMA 通道（跨节点）：                                   │
  │    • Expert Dispatch（跨节点部分）                       │
  │    • Expert Combine（跨节点部分）                        │
  │    • 梯度 AllReduce（DP 组跨节点）                       │
  │    • FSDP AllGather / ReduceScatter                     │
  │                                                         │
  │  调度策略（DAG 驱动）：                                  │
  │    FWD 时间线：                                          │
  │    XGMI: [Dispatch_local][RS_FSEP]       [Combine_local]│
  │    RDMA:        [Dispatch_remote]  [Expert_GEMM]  [Comb]│
  │    GPU:  [Gate] ──overlap──→ [Expert GEMM] ──overlap──→ │
  │                                                         │
  │    两条通道永不阻塞对方 → 有效带宽 = XGMI + RDMA        │
  │    vs NVIDIA: 有效带宽 ≈ max(NVLink, IB)                │
  │                                                         │
  │  这是 MoEPackage 相对 Megatron-Core 的最大差异化优势     │
  └─────────────────────────────────────────────────────────┘
```

---

### 3.4 Fused Permute-Quantize-Dispatch Pipeline

**这是将 4 个模块串联的关键整合器。**

```
Megatron-Core 当前的 Dispatch 流水线（即使有 Permutation Fusion）：

  tokens [T, H] (BF16)
    ↓  Kernel 1: Fused Permute + Pack         (1R BF16, 1W BF16)
  permuted [T*K, H] (BF16)
    ↓  Kernel 2: FP8 Quantize                 (1R BF16, 1W FP8)
  quantized [T*K, H] (FP8)
    ↓  Kernel 3: DeepEP Dispatch              (1R FP8, 网络发送)

  HBM 访问：3 次读 + 2 次写 = 5 次

MoEPackage 的 Fused Pipeline：

  tokens [T, H] (BF16)
    ↓  Single HIP Kernel: Permute + Quantize + P2P Write
  [直接写入目标 GPU 的 P2P Buffer] (FP8)

  HBM 访问：1 次读 + 1 次写（通过 XGMI，写入远端 = 零本地写）

  ┌─────────────────────────────────────────────────────────┐
  │  fused_permute_quant_dispatch_kernel                     │
  │                                                         │
  │  每个 thread block 处理一个 token：                      │
  │  1. 从 HBM 读入 token 到 LDS (64KB/CU)                 │
  │  2. 在 LDS 中查询 routing_map → 目标 GPU + slot         │
  │  3. 在 LDS 中完成 BF16 → FP8 量化                      │
  │  4. 从 LDS 直接 XGMI P2P 写入目标 GPU 的 recv buffer   │
  │     （如果是跨节点目标 → 写入本节点 RDMA staging buf）  │
  │                                                         │
  │  性能分析（DeepSeek-V3 参数）：                          │
  │  T=4096, K=8, H=7168, BF16                              │
  │                                                         │
  │  Megatron-Core: 5 × 4096×8×7168×2B = 2.24 GB HBM 流量  │
  │  MoEPackage:    1 × 4096×8×7168×2B = 0.45 GB HBM 读取  │
  │                 + XGMI P2P 写入（不占本卡 HBM 带宽）     │
  │                                                         │
  │  单层带宽节省：~80% (4.5× 减少)                         │
  │  61 层 × FWD+BWD：~273 GB → ~55 GB（节省 ~218 GB）     │
  │  MI300X 5.3 TB/s：~41ms → ~10ms                        │
  │  → Dispatch 流水线加速 ~4×                              │
  └─────────────────────────────────────────────────────────┘

Combine 方向（反向）同样融合：
  Fused Dequantize + Unpermute + Weighted Reduce
  FP8 recv buffer → 1 次 HIP Kernel → BF16 输出
  消除中间 BF16 buffer 的 2 次 HBM 读写
```

---

## 第四部分：MoEPackage 与其他论文优化的关系

### 4.1 借鉴 vs 原创 vs 替代

```
┌─────────────────────────────────────────────────────────────────────────┐
│  MoEPackage 各组件的技术来源                                             │
├──────────────────────────┬──────────┬──────────────────────────────────┤
│  组件                     │  关系     │  来源/说明                       │
├──────────────────────────┼──────────┼──────────────────────────────────┤
│  XGMI P2P Dispatch       │  原创     │  替代 DeepEP，利用 AMD P2P 语义  │
│  Persistent P2P Pool     │  替代     │  替代 NCCL UBR，P2P IPC Handle  │
│  Padded Static GEMM      │  借鉴     │  MC 的 ECHO + Paged Stashing    │
│  Dual-Channel Scheduler  │  原创     │  AMD XGMI+RDMA 双通道独有        │
│  Fused Permute-Quant     │  原创     │  MoEX 思想的最小实现             │
│  Parallel Folding        │  移植     │  MC Parallel Folding 直接使用    │
│  Memory-Efficient Perm.  │  移植     │  MC 优化直接使用                 │
│  细粒度重算              │  移植     │  MC 优化直接使用                 │
│  FP8 Blockwise 训练      │  适配     │  MC Blockwise FP8 适配 MI300X    │
│  EP 通信重叠             │  适配     │  MC 1F1B Merged 适配 HIP Stream  │
│  FSEP ReduceScatter      │  替代     │  LAER-MoE RS → XGMI In-Place    │
│  通信感知路由            │  借鉴     │  MegaScale-MoE 拓扑感知思想      │
│  Comm-Aware Routing      │  原创     │  路由打分加入通信代价惩罚项      │
└──────────────────────────┴──────────┴──────────────────────────────────┘
```

### 4.2 MoEPackage 相对 Megatron-Core 的收益分析

```
以 DeepSeek-V3 on 256 MI300X（EP=64）为例：

Megatron-Core 在 MI300X 上的预估基线（基于 H100 368 TFLOPS 按硬件比例换算）：
  MI300X BF16 峰值 = 1307 TFLOPS（vs H100 989 TFLOPS）
  带宽：MI300X 5.3 TB/s（vs H100 3.35 TB/s）
  互联：XGMI 896 GB/s（vs NVLink 900 GB/s）≈ 持平
  MI300X 基线预估 ≈ 368 × 1307/989 ≈ 486 TFLOPS（MFU ~37%）
  
  注：实际可能更低，因为 MC 的 CUDA 特定优化（DeepEP/CUDA Graph）在 AMD 上不可用

MoEPackage 各模块的预估收益：

  Module 1: XGMI-Native Dispatch
    替代 RCCL All-to-All → 节点内延迟降低 ~3×
    贡献：+3~5% MFU（主要在通信受限场景）

  Module 2: Persistent P2P Pool
    消除 per-iteration Buffer 分配 + RCCL 启动开销
    贡献：+1~2% MFU

  Module 3: AMD Dropless GEMM + HIP Graph
    Dropless MoE → 更好的 Expert 利用率 + HIP Graph 加速
    贡献：+5~8% MFU（对标 MC 的 +10% CUDA Graph）

  Module 4: Dual-Channel Scheduler
    XGMI + RDMA 真正并发 → 有效通信带宽翻倍
    贡献：+5~8% MFU（这是 AMD 独有优势）

  Fused Pipeline: Permute + Quant + Dispatch
    Dispatch 流水线 HBM 流量减少 ~80%
    贡献：+3~5% MFU

  综合叠加（非简单加法，有交互效应）：
    预估 MFU 从 ~37% → ~52~58%（+15~21 个百分点）
    预估 TFLOPS 从 ~486 → ~680~760

  对比 Megatron-Core on H100:
    MC H100 = 368 TFLOPS → MoEPackage MI300X = 680~760 TFLOPS
    MI300X 以约 80% 的 H100 价格（同等集群规模）达到 ~2× H100 吞吐

  对比 Megatron-Core on GB200:
    MC GB200 = 1,048 TFLOPS → MoEPackage MI300X = 680~760 TFLOPS
    MI300X 无法追上 GB200（NVLink 5 + Blackwell Tensor Core 太强）
    但 MI300X 的性价比仍有优势
```

---

## 第五部分：实现路线图

### 5.1 分阶段实施

```
Phase 0：验证可行性（4 周）
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ① XGMI P2P Direct Access 微基准测试
     验证 P2P 延迟和带宽是否达到理论值
     对比 RCCL All-to-All 的差距
  ② hipBLASLt Grouped GEMM 性能评估
     验证静态 padded 形状的 GEMM 效率
  ③ Dual-Channel 并发测试
     同时跑 XGMI P2P + RDMA 传输，验证带宽叠加
  交付：可行性报告 + 微基准数据

Phase 1：核心模块（8 周）
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ① Module 1: XGMI-Native Dispatch
     HIP Kernel 实现 + 单节点 8 GPU 验证
  ② Module 2: Persistent P2P Pool
     IPC Handle 管理 + 与 Module 1 集成
  ③ Module 3: Padded Static GEMM
     hipBLASLt 封装 + valid_mask 机制
  交付：3 个独立模块 + 单元测试 + 性能数据

Phase 2：集成 + 融合（8 周）
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ① Fused Permute-Quantize-Dispatch HIP Kernel
  ② Module 4: Dual-Channel Scheduler（DAG 驱动）
  ③ EP 通信重叠（1F1B Merged，HIP Stream 版本）
  ④ 端到端 MoE Layer forward/backward 集成
  交付：完整 MoEPackage + 8 GPU 端到端性能

Phase 3：扩展 + 优化（8 周）
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ① 多节点扩展（32/64/256 GPU）
  ② HIP Graph 全覆盖
  ③ FSEP 集成（XGMI In-Place Reduction）
  ④ 通信感知路由 + Parallel Folding 集成
  ⑤ Megatron-Core 集成接口（作为 Expert Dispatcher 插件）
  交付：生产级 MoEPackage + DeepSeek-V3 scale 端到端数据
```

### 5.2 关键风险与缓解

```
风险 R1：XGMI P2P 实际带宽不及理论值
  原因：P2P 原子操作竞争、XGMI 链路拥塞
  缓解：Phase 0 微基准验证；若差距 > 30%，回退到 RCCL 节点内通信

风险 R2：hipBLASLt Grouped GEMM 成熟度不足
  原因：hipBLASLt 可能不支持某些 Grouped GEMM 变体
  缓解：composable_kernel 作为备选；Triton for ROCm 作为最终后备

风险 R3：Dual-Channel 并发引入死锁
  原因：XGMI 和 RDMA 同时操作同一 Buffer 时的内存一致性问题
  缓解：严格的 Buffer 分区（XGMI Zone / RDMA Zone）+ Fence 机制

风险 R4：HIP Graph 功能不完整
  原因：ROCm 的 hipGraph 对某些动态特性支持不如 CUDA Graph
  缓解：仅对 Expert GEMM + 通信部分做 Graph，Gate/Router 用 eager mode

风险 R5：FP8 训练在 MI300X 上的数值稳定性
  原因：MI300X 的 FP8 实现与 NVIDIA E4M3/E5M2 有细微差异
  缓解：先用 BF16 验证功能正确性，再逐步启用 FP8
```

---

## 第六部分：与现有工作的差异定位

```
┌────────────────────────────────────────────────────────────────────────────┐
│                     MoEPackage 差异定位一览                                  │
├─────────────────────┬──────────────────────────────────────────────────────┤
│ vs Megatron-Core    │ 不是移植，是在 MC 做不到的 4 个缺口上提供 AMD 原生解 │
│                     │ DeepEP→P2P，UBR→P2P Pool，Device-Init→Padded GEMM │
│                     │ + AMD 独有的双通道并发调度                           │
├─────────────────────┼──────────────────────────────────────────────────────┤
│ vs DeepEP           │ 完全不同的实现路径（P2P vs TMA+IBGDA），            │
│                     │ 但达成相同目标（高效 Expert Dispatch）               │
├─────────────────────┼──────────────────────────────────────────────────────┤
│ vs LAER-MoE FSEP    │ ReduceScatter 用 XGMI In-Place 替代 NCCL，         │
│                     │ Re-layout 用 XGMI P2P memcpy 替代 NCCL A2A         │
├─────────────────────┼──────────────────────────────────────────────────────┤
│ vs Comet            │ Comet 的 Warp Specialization → Wavefront 专用化     │
│                     │ 但 MoEPackage Phase 1 不含 Tile 级 Overlap（Phase 3）│
├─────────────────────┼──────────────────────────────────────────────────────┤
│ vs FlowMoE          │ FlowMoE 是 Python 层调度，MoEPackage 是 Kernel 层  │
│                     │ 两者互补，可叠加                                    │
├─────────────────────┼──────────────────────────────────────────────────────┤
│ vs MoEX (CommTensor)│ MoEPackage 的 Fused Pipeline 是 MoEX "通信原生张量" │
│                     │ 思想的最小可验证实现，不需要完整 CommTensor 抽象      │
├─────────────────────┼──────────────────────────────────────────────────────┤
│ vs ROCflow          │ MoEPackage 是 ROCflow 的核心加速引擎                │
│                     │ ROCflow = 完整框架，MoEPackage = 可插拔优化包        │
│                     │ MoEPackage 可以独立集成到 Megatron-Core / torchtitan │
└─────────────────────┴──────────────────────────────────────────────────────┘
```

---

## 第七部分：核心结论

```
一句话总结：

  Megatron-Core 的 14 项 MoE 优化中，6 项可直接移植到 AMD，4 项需适配，
  2 项需要 AMD 原生替代方案，2 项完全不可移植。

  MoEPackage 精准针对这 4 个缺口（DeepEP / NCCL UBR / Device-Initiated / 双通道），
  用 AMD MI300X 的 XGMI P2P + Dual-Channel + 192GB HBM3e 硬件优势，
  提供比简单移植 Megatron-Core 更好的性能。

  核心差异化：
    ① Fused Permute+Quantize+P2P Dispatch → Dispatch 流水线 HBM 减少 80%
    ② Dual-Channel XGMI+RDMA 并发 → 有效通信带宽翻倍
    ③ XGMI P2P 替代 NCCL → 节点内零协议开销

  预估 MoEPackage 在 MI300X 上可达 680~760 TFLOPS（DeepSeek-V3 训练），
  约为 Megatron-Core 在 H100 上的 1.8~2.0 倍，
  约为 Megatron-Core 在 GB200 上的 0.65~0.72 倍。
```

---

*MoEPackage 设计文档 | 2026-03-12 | AIInfra-Book*
*基于 Megatron-Core v0.16、DeepEP、FlowMoE、Comet、LAER-MoE、MoEBlaze、MemFine、SwiftMoE、MegaScale-MoE、MoE Parallel Folding 十篇论文的系统分析*
