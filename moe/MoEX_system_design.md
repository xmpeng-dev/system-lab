# MoEX: Communication-First MoE Training System

> **定位：** 全新 MoE 训练系统设计，从底层 Tensor 结构出发为通信而生  
> **核心理念：** Design the tensor for the wire, not for the ALU  
> **创新层次：** Tensor 格式 → 执行模型 → 内存架构 → 调度协议 → 并行映射  
> **目标：** 消除 MoE 训练中 60%+ 的数据搬运开销，实现通信-计算真正融合  
> **关联论文：** MegatronCore MoE · FlowMoE · Comet · LAER-MoE · DeepEP · MegaScale-MoE

---

## 0. 一句话定位

> **所有现有 MoE 系统都把 Tensor 当「计算对象」—— 先算好再想怎么发。MoEX 把 Tensor 当「通信对象」—— 从诞生的那一刻起就为 dispatch/combine/overlap/异构并行而组织。**

---

## 1. 根本问题：MoE 训练的数据搬运税

### 1.1 当前系统的数据生命周期

```
一个 token 在 MoE 层中经历的数据操作（当前最优系统）：

  Attention 输出 x [T, H]（contiguous, per-GPU）
         │
    ①  Gate Linear（GEMM）→ routing scores
    ②  Top-K Selection → routing_map (bool mask)
    ③  Permute：按 Expert 重排 x → permuted_x [T×K, H]    ← 全局内存读写 #1
    ④  Pack：组织成 per-destination 缓冲区                  ← 全局内存读写 #2
    ⑤  All-to-All Dispatch（NCCL/DeepEP/HybridEP）         ← 网络通信
    ⑥  Unpack：从接收缓冲区提取 per-expert tokens           ← 全局内存读写 #3
    ⑦  Padding：对齐到 FP8/Grouped GEMM 要求的倍数          ← 全局内存读写 #4
    ⑧  Expert GEMM（真正的计算）
    ⑨  Unpadding                                           ← 全局内存读写 #5
    ⑩  Pack：组织成 per-source 缓冲区                       ← 全局内存读写 #6
    ⑪  All-to-All Combine                                  ← 网络通信
    ⑫  Unpermute：恢复原始 token 顺序                       ← 全局内存读写 #7
    ⑬  加权合并 → 输出

总结：1 次实际计算（⑧），2 次网络通信（⑤⑪），7 次全局内存读写（①②③④⑥⑦⑨⑩⑫）
```

### 1.2 数据搬运税的量化

```
以 DeepSeek-V3 为例（h=7168, T=4096, K=8, E=256, EP=64）：

每个 MoE 层每 GPU 的数据搬运量：
  Permute (#3):        T × K × h × 2B = 4096 × 8 × 7168 × 2 = 449 MB
  Pack (#4):           同上 ≈ 449 MB
  Unpack (#6):         接收量 ≈ T × K × h × 2B / EP × (EP-1) ≈ 442 MB
  Padding (#7):        ~5-20% 额外
  Pack Combine (#10):  ≈ 442 MB
  Unpermute (#12):     ≈ 449 MB
  ─────────────────────────────────────────────
  单层单方向搬运总量:   ~2.2 GB
  FWD + BWD × 61 层:   ~268 GB / step

  vs 实际计算所需内存带宽（Expert GEMM 输入）：
  单层 Expert 输入：    T × K / E × EP × h × 2B ≈ 7 MB
  
  搬运量 : 计算量 ≈ 300 : 1
  
  → 超过 99% 的 HBM 带宽用在了"搬运"而非"计算"上
  → 这就是 MoE 训练的根本瓶颈：数据搬运税

MegatronCore 的优化减少了多少？
  Memory-Efficient Permutation → 减少 ⑫ 的中间 buffer（-26 GB 激活）
  Permutation Fusion → 融合 ③④ 为一次读写（-1 次全局读写）
  HybridEP → 融合 ⑤⑥ / ⑩⑪（减少通信内核数）
  
  但本质上还是 "算完了再搬" 的范式
  → 搬运次数从 7 次降到 ~4 次，但 4 次仍然是 4 次
```

### 1.3 MoEX 的根本洞察

```
所有现有系统的隐含假设：

  假设 1：Tensor 以「计算最优」的 contiguous 布局存储
         → 对 GEMM 友好，但对通信不友好
  假设 2：路由决策和数据组织是两个独立步骤
         → 先决定谁去哪，再搬数据
  假设 3：通信缓冲区是临时分配的
         → 每次 dispatch 都需要 pack/unpack

MoEX 打破这三个假设：

  MoEX 假设 1：Tensor 以「通信最优」的 destination-partitioned 布局存储
              → 数据从诞生就按目标 GPU 组织
  MoEX 假设 2：路由决策与数据写入是同一个操作（Scatter-on-Write）
              → 计算结果直接写入目标缓冲区，消除 permute
  MoEX 假设 3：通信缓冲区是持久的、预注册的、拓扑感知的
              → 零拷贝 dispatch，缓冲区按硬件拓扑分区
```

---

## 2. MoEX 系统架构总览

```
┌──────────────────────────────────────────────────────────────────────┐
│                     MoEX System Architecture                         │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Layer 5: Heterogeneous Parallel Mapper (HPM)                        │
│    Attention 用 TP×CP×DP，MoE 用 EP×EDP（Parallel Folding）          │
│    但 MoEX 的 CommTensor 使得转换是零拷贝的                           │
│                                                                      │
│  Layer 4: Overlap-Native Communication Protocol (ONCP)               │
│    流式 dispatch/combine（永不阻塞）                                   │
│    Pre-Routed Pipeline：路由决策提前到 Attention 阶段                  │
│    FWD-BWD Merged Overlap：前反向通信-计算全融合                       │
│                                                                      │
│  Layer 3: Topology-Aware Memory Pool (TAMP)                          │
│    GPU 内存按拓扑分区：Local / NVLink / RDMA Zone                     │
│    持久通信缓冲区 + NCCL User Buffer Registration                     │
│    Expert 权重按 Zone 分布，缓存热门 Expert                            │
│                                                                      │
│  Layer 2: Scatter-on-Write Execution Model (SoW)                     │
│    GEMM 输出直接 scatter 到目标缓冲区（消除 permute + pack）           │
│    Gate-Scatter 融合内核：路由+写入一步完成                             │
│    Grouped Scatter-GEMM：Expert 计算结果直接写入 combine 缓冲区        │
│                                                                      │
│  Layer 1: CommTensor — Communication-Native Tensor Format             │
│    Destination-Partitioned Layout (DPL)                               │
│    Dual-View Descriptor：Compute View + Comm View                     │
│    内嵌 routing metadata + precision tag + topology hint               │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. Layer 1: CommTensor — 通信原生张量格式

### 3.1 核心设计：Destination-Partitioned Layout (DPL)

```
传统 Tensor 布局（Compute-Optimal）：
  x = [T, H]   # T 个 token，连续存储
  
  内存布局：
    [token_0 | token_1 | token_2 | ... | token_T-1]
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            所有 token 连续，对 GEMM 友好

CommTensor 布局（Communication-Optimal）：
  x = CommTensor(shape=[T, H], num_partitions=EP)
  
  内存布局：
    ┌─── Partition 0 (→ GPU 0) ─┐┌─── Partition 1 (→ GPU 1) ─┐ ... 
    │ [tok_a | tok_f | tok_m]    ││ [tok_b | tok_g | tok_q]    │ ...
    └───────────────────────────┘└───────────────────────────┘ ...
    
  每个 Partition 是一块连续内存，直接作为 All-to-All 的 send buffer
  → Dispatch = 释放 Partition 供目标 GPU 读取（零数据拷贝）

  关键：Partition 内 token 顺序不影响 Expert GEMM 正确性
       （因为 MoE 是 token-independent 的）

分区策略（按路由目标 GPU 分组）：
  Partition_i = {token_j : expert(token_j) 位于 GPU_i}
  
  每个 Partition 的大小是动态的（路由决策决定）
  但 Partition 的位置和上限是静态的（持久预分配）
```

### 3.2 Dual-View Descriptor

```
CommTensor 不是重新排列数据，而是维护两套「视图描述符」：

Compute View（计算视图）：
  用于 Attention、LayerNorm 等需要全序列操作的模块
  逻辑上看：x[T, H]，token 按序列位置排列
  物理上：通过 index map 将逻辑位置映射到 DPL 中的物理位置

Comm View（通信视图）：
  用于 MoE Dispatch / Combine
  逻辑上看：partitions[EP][T_max, H]，token 按目标 GPU 分组
  物理上：直接指向 DPL 的物理内存（零拷贝）

数据结构定义：

  struct CommTensor {
      // 物理存储（DPL 格式）
      void*     data;                    // 预分配的 DPL 内存池
      int       total_tokens;
      int       hidden_dim;
      DType     dtype;                   // BF16 / FP8 / FP4
      
      // Compute View
      int*      compute_index;           // [T] → 物理偏移映射
      // 等价于：data[compute_index[i]] = 逻辑第 i 个 token
      
      // Comm View（per-partition 描述符）
      struct PartitionDesc {
          int   offset;                  // 在 data 中的起始偏移
          int   count;                   // 本 partition 的 token 数
          int   capacity;                // 本 partition 的最大容量
          int   target_gpu;              // 目标 GPU rank
          Zone  zone;                    // LOCAL / NVLINK / RDMA
      } partitions[MAX_EP];
      int num_partitions;
      
      // Routing Metadata（内嵌路由信息）
      int*      token_to_expert;         // [T, K] → expert assignment
      float*    routing_weights;         // [T, K] → routing probabilities
      bool      routing_ready;           // 路由信息是否已就绪
      
      // Precision Tag
      DType     compute_dtype;           // 计算用精度 (BF16)
      DType     comm_dtype;              // 通信用精度 (FP8)
      void*     comm_data;               // 如果 comm_dtype != compute_dtype，
                                         // 指向已量化的通信缓冲区
  };

两个视图之间的转换：
  Compute → Comm：更新 partition 描述符（O(T) 的索引操作，无数据移动）
  Comm → Compute：更新 compute_index（O(T) 的索引操作，无数据移动）
  
  → 视图转换的代价 = 更新几 KB 的索引表，不是搬运几百 MB 的数据
```

### 3.3 内嵌精度管理

```
通信精度与计算精度分离（Communication-Precision Decoupling）：

传统流程：
  BF16 激活 → 量化为 FP8 → Pack → Send → Receive → 反量化 → BF16 GEMM
  
MoEX 流程：
  GEMM 输出时同时生成两份：
    compute_data (BF16)：供本地后续计算使用
    comm_data (FP8/FP4)：供通信使用，直接在 GEMM 输出时融合量化

  Gate-Scatter 内核输出时：
    token 写入 Partition 时，同时写入 BF16 和 FP8 两个位置
    BF16 位置 → 本地 Expert GEMM 使用
    FP8 位置 → 直接作为 All-to-All send buffer

  → 消除了单独的 "quantize before send" 步骤
  → 量化操作被融合进了 scatter-write 中
  
内存开销：
  每个 CommTensor 额外存储 FP8 通信副本
  = T × H × 1B = T × H × 50% of BF16
  但消除了临时量化缓冲区的分配/释放开销
```

### 3.4 与传统 Tensor 的互操作

```
CommTensor 需要与现有模块无缝协作：

Attention 模块（需要 Compute View）：
  attention_input = comm_tensor.as_compute_view()
  # 返回一个 strided tensor view，逻辑上 [T, H] 连续
  # 物理上通过 compute_index 映射（对 Flash Attention 不影响性能）

MoE Dispatch（需要 Comm View）：
  partitions = comm_tensor.as_comm_view()
  # 返回 per-partition 描述符列表
  # 每个 partition 是一块物理连续内存，可直接作为 NCCL send buffer

LayerNorm / Residual Add（需要 Compute View）：
  可以在 DPL 格式上直接操作（token-independent 操作不关心顺序）
  → LayerNorm 在任意 token 排列上结果相同
  → Residual Add 只需要对应位置 token 相加

关键洞察：
  MoE 层之间的大部分操作（LayerNorm, Residual, Gate）都是 token-independent 的
  → 不需要 token 按序列位置排列
  → CommTensor 的 DPL 格式可以直接使用，无需转换为 contiguous
  
  唯一需要 Compute View 的：Attention（因为 Q-K 点积需要序列位置关系）
  → 只在 Attention 处做一次视图切换（更新索引表，不移动数据）
```

---

## 4. Layer 2: Scatter-on-Write (SoW) 执行模型

### 4.1 核心思想：GEMM 输出即 Dispatch

```
传统 MoE 前向路径：
  Step 1: gate_logits = x @ W_gate          # Gate GEMM
  Step 2: routing = TopK(softmax(gate_logits))  # 路由决策
  Step 3: permuted_x = permute(x, routing)   # 按 Expert 重排
  Step 4: packed = pack(permuted_x)          # 组织为 send buffer
  Step 5: received = all_to_all(packed)      # 网络通信
  Step 6: expert_in = unpack(received)       # 提取 per-expert 输入
  Step 7: expert_out = grouped_gemm(expert_in)  # Expert 计算
  Step 8: combined = all_to_all(expert_out)  # 返回通信
  Step 9: output = unpermute_and_combine(combined, routing)

MoEX Scatter-on-Write 路径：
  Step 1: routing = gate_and_route(x, W_gate)       # 融合 Gate+TopK+Softmax
  Step 2: scatter_write(x, routing, comm_tensor)     # 直接写入 DPL 分区
          → comm_tensor 的 Partition_i 包含了发往 GPU_i 的所有 token
          → 已经在正确的 send buffer 位置，无需 pack
  Step 3: stream_dispatch(comm_tensor)               # 流式发送（见 Layer 4）
  Step 4: expert_out = scatter_grouped_gemm(recv_tensor, expert_weights)
          → Expert GEMM 输出直接 scatter 到 combine 的 send buffer
  Step 5: stream_combine(expert_out_tensor)          # 流式返回
  Step 6: combine_write(expert_out_tensor, output)   # 加权合并直接写入输出

步骤数：9 → 6
全局内存读写：7 次 → 2 次（scatter_write + combine_write）
```

### 4.2 Gate-Scatter 融合内核

```
传统 Gate + Permute 是两个独立操作：
  Gate:    读 x[T,H] → 算 gate_logits[T,E] → 写 routing[T,K]
  Permute: 读 x[T,H] + routing[T,K] → 写 permuted[T×K,H]
  
  x[T,H] 被从 HBM 读取了两次！（Gate 读一次，Permute 读一次）
  
MoEX Gate-Scatter 融合内核（单次读取 x）：

  __global__ void gate_scatter_kernel(
      const half* x,              // [T, H] 输入
      const half* W_gate,         // [H, E] Gate 权重
      CommTensor* output,         // 输出 CommTensor（DPL 格式）
      int T, int H, int E, int K
  ) {
      // 每个 thread block 处理一个 token
      int tid = blockIdx.x;
      if (tid >= T) return;
      
      // Phase 1: 从 HBM 读取 token 到 shared memory（只读一次！）
      __shared__ half token_smem[H];
      cooperative_load(token_smem, &x[tid * H], H);
      
      // Phase 2: 在 shared memory 上计算 gate logits（复用已加载的 token）
      half gate_logits[E];
      gemv_in_smem(gate_logits, token_smem, W_gate, H, E);
      
      // Phase 3: TopK + Softmax（寄存器内完成）
      int top_k_indices[K];
      half top_k_weights[K];
      topk_softmax(gate_logits, E, K, top_k_indices, top_k_weights);
      
      // Phase 4: Scatter-Write（从 shared memory 直接写入目标 Partition）
      for (int k = 0; k < K; k++) {
          int expert_id = top_k_indices[k];
          int target_gpu = expert_to_gpu(expert_id);
          
          // 原子递增目标 Partition 的 token 计数
          int slot = atomicAdd(&output->partitions[target_gpu].count, 1);
          
          // 直接从 shared memory 写入目标 Partition（一次 HBM 写入）
          int dst_offset = output->partitions[target_gpu].offset + slot * H;
          cooperative_store(&output->data[dst_offset], token_smem, H);
          
          // 同时写入 FP8 通信版本（如果需要低精度通信）
          if (output->comm_dtype == FP8) {
              quantize_and_store(&output->comm_data[dst_offset/2], 
                                token_smem, H);
          }
          
          // 记录路由信息
          output->token_to_expert[tid * K + k] = expert_id;
          output->routing_weights[tid * K + k] = top_k_weights[k];
      }
  }

性能分析：
  传统 Gate + Permute：读 x 2 次，写 routing + permuted 各 1 次 = 3 次 HBM 访问
  Gate-Scatter 融合：  读 x 1 次，写 DPL 分区 1 次 = 2 次 HBM 访问（含 FP8）
  
  HBM 带宽节省：~33%
  同时消除了 routing_buffer 的中间内存（~T×K×4B）
```

### 4.3 Scatter-Grouped-GEMM：Expert 计算直出 Combine Buffer

```
传统 Expert GEMM 后还需要 Pack 再 Combine：
  expert_out = grouped_gemm(expert_in)    # 写入 expert_out buffer
  packed = pack(expert_out)               # 重组为 per-source GPU 缓冲区
  combined = all_to_all(packed)           # 发送回源 GPU

MoEX Scatter-Grouped-GEMM：
  Expert GEMM 的输出直接写入 Combine 用的 CommTensor 的 Partition

  核心：Expert GEMM 内核的 epilogue 阶段（从寄存器写回 HBM）
        不写入连续的 expert_out buffer
        而是根据 token 的「来源 GPU」信息，直接 scatter 到对应的 Partition

  这要求 Grouped GEMM 内核支持 scatter epilogue：
    标准 GEMM epilogue：C[m, n] = alpha * A @ B + beta * C
    Scatter epilogue：  partition[src_gpu].data[slot * H + n] = alpha * A @ B

  实现方式：
    方案 A：cuteDSL Grouped GEMM 的 custom epilogue（开发中的特性）
    方案 B：将 scatter 作为一个融合的后处理步骤
            （Expert GEMM → LDS → scatter write，一个 kernel 内完成）

  效果：消除了 Expert GEMM 后的 Pack 步骤（-1 次全局内存读写）
```

---

## 5. Layer 3: Topology-Aware Memory Pool (TAMP)

### 5.1 内存拓扑分区

```
GPU 内存按通信拓扑分为三个 Zone：

  ┌──────────────────────────────────────────────┐
  │              GPU_i HBM Memory                 │
  ├──────────────────────────────────────────────┤
  │                                              │
  │  ┌─── Local Zone ──────────────────────┐     │
  │  │  Expert 权重 / 优化器状态 / 本地激活  │     │
  │  │  不参与通信，纯本地使用              │     │
  │  └─────────────────────────────────────┘     │
  │                                              │
  │  ┌─── NVLink Zone ────────────────────┐      │
  │  │  CommTensor Partition[同节点 GPU]    │      │
  │  │  → NCCL User Buffer Registered     │      │
  │  │  → NVLink P2P 可直接访问           │      │
  │  │  → dispatch 到同节点 GPU = 零拷贝  │      │
  │  └─────────────────────────────────────┘     │
  │                                              │
  │  ┌─── RDMA Zone ─────────────────────┐       │
  │  │  CommTensor Partition[跨节点 GPU]   │       │
  │  │  → RDMA User Buffer Registered    │       │
  │  │  → GPUDirect RDMA 直接发送        │       │
  │  │  → dispatch 到跨节点 = 一次 RDMA  │       │
  │  └─────────────────────────────────────┘     │
  │                                              │
  └──────────────────────────────────────────────┘

分区比例（以 EP=64, 8 GPU/node 为例）：
  同节点 GPU：8/64 = 12.5% → NVLink Zone
  跨节点 GPU：56/64 = 87.5% → RDMA Zone
  本地计算：Expert 权重 + 本地缓冲 → Local Zone

Zone 大小动态可调（基于实际路由统计）
```

### 5.2 持久通信缓冲区

```
传统 MoE 的通信缓冲区管理：
  每次 dispatch：malloc send_buf → pack → send → free send_buf
  每次 combine： malloc recv_buf → receive → unpack → free recv_buf
  
  问题：
    1. malloc/free 有 CUDA 内存分配器开销
    2. NCCL 每次需要重新注册缓冲区（如果地址变了）
    3. 无法预注册为 RDMA User Buffer

MoEX 持久缓冲区：
  训练开始时一次性分配，整个训练过程不变

  CommTensorPool = {
      send_partitions: [EP][capacity][H]    // 持久 send 缓冲区
      recv_partitions: [EP][capacity][H]    // 持久 recv 缓冲区
      
      // NCCL User Buffer Registration（一次注册，永久使用）
      nccl_registered: true
      // RDMA MR (Memory Region) Registration
      rdma_mr: ibv_mr*
  }

  每次 dispatch 只是更新 Partition 描述符中的 count 字段
  → 数据写入的位置始终在同一块预注册内存中
  → NCCL/RCCL 可以直接使用，零注册开销

  效果：
    消除 per-iteration 的缓冲区分配/释放/注册开销
    NCCL SM 占用从 8-32 SM 降至 1-4 SM（同 MegatronCore FSDP 的优化）
    RDMA 可使用 IBGDA（InfiniBand GPUDirect Async）直接从 Zone 读取
```

### 5.3 Expert Weight 拓扑感知放置

```
传统 Expert 权重放置：所有 Expert 权重在 Local Zone
MoEX 策略：热门 Expert 的权重缓存到多个 Zone

  基于路由统计的 Expert 热度监控：
    hot_expert_map = monitor_routing_stats(every_K_steps)
    
  热门 Expert 处理（类 MegatronCore ECHO 思路，但基于 Zone）：
    若 Expert_j 被大量跨节点 token 访问：
      在 RDMA Zone 缓存 Expert_j 的权重副本
      跨节点 token dispatch 后可以在本地直接计算
      → 消除了 "dispatch token 到 Expert 所在 GPU" 的通信
      → 改为 "广播 Expert 权重到 token 所在 GPU"
      
    适用条件：Expert 权重 << 需要 dispatch 的 token 总量
    DeepSeek-V3: Expert 权重 ≈ 2 × h × FFN/EP ≈ 2 × 7168 × 2048 × 2B ≈ 56 MB
                 热门 Expert token 量 ≈ T × K/E × load_factor ≈ 数百 MB
                 → 广播权重比 dispatch token 更经济（当 load_factor > 2 时）
```

---

## 6. Layer 4: Overlap-Native Communication Protocol (ONCP)

### 6.1 Pre-Routed Pipeline：路由决策前置

```
传统执行顺序：
  Attention → LayerNorm → Gate → Route → Dispatch → Expert → Combine
  
  问题：Gate+Route 在 Attention 之后，此时才知道路由决策
        → Dispatch 必须等 Attention 完全完成

MoEX Pre-Routed Pipeline：

  关键洞察：
    Gate 的输入是 LayerNorm(Attention_output + x_residual)
    但路由决策（TopK）只需要 gate_logits 的相对大小关系
    → 可以用一个"近似路由"替代精确路由
    
  两种实现策略：

  策略 A：投机路由（Speculative Routing）
    在 Attention 开始时，用 x_residual（已知）做一次"预测路由"：
      speculative_routing = TopK(softmax(x_residual @ W_gate_cached))
    
    基于预测路由，提前组织 CommTensor 的 DPL 布局
    Attention 完成后，做真正的路由，与预测对比：
      if match_rate > 0.9:   # 90%+ token 路由不变
          直接使用预 dispatch（仅修正 10% 不匹配的 token）
      else:
          fallback 到标准 scatter-write
    
    预测准确率分析：
      相邻层的路由决策高度相关（Expert 偏好短期稳定）
      实际匹配率通常 > 85%（根据 DeepSeek-V3 路由统计）
      → 85% token 免费 dispatch，15% 需要修正

  策略 B：流水线化路由（Pipelined Routing）
    将 Gate 计算分解：
      Gate_phase1: gate_logits = x_norm @ W_gate    # 可在 Attention 并行
      Gate_phase2: routing = TopK(softmax(gate_logits))  # 依赖 phase1
      
    当 Attention 和 Gate_phase1 共享输入（x_norm）时：
      Gate_phase1 可以在 Attention 的第一个 GEMM 之后立刻启动
      （x_norm 在 Attention 开始时就已就绪）
      
    时间线：
      传统：  [──── Attention ────][Gate][Route][── Dispatch ──]
      MoEX：  [──── Attention ────]
                [Gate_1][Gate_2][Route]
                                    [── Dispatch ──]  ← 提前 ~Gate 延迟
                                    
    提前量 ≈ Gate GEMM 时间 ≈ 0.5-2ms（取决于 H 和 E）
```

### 6.2 Streaming Dispatch / Combine（流式通信）

```
传统 All-to-All：所有 token 打包完成后一次性发送（批量阻塞）

MoEX Streaming Dispatch：
  Token 被 scatter 到 Partition 后立即可发送，不等其他 Partition

  Gate-Scatter 内核每完成一个 Partition 的填充：
    → 立即通知通信引擎：Partition_i ready, count = N_i
    → 通信引擎异步发送 Partition_i
    → 不等待其他 Partition

  接收端：
    → 每收到一个 Partition 的部分数据
    → Expert GEMM 立即开始处理已到达的 token
    → 不等待所有 Partition 都到达

  这本质上是把 All-to-All 变成了一组独立的 Point-to-Point 流

  实现：
    基于 HybridEP 的 FIFO 队列机制（MegatronCore 已有）
    但 MoEX 的优势：CommTensor 的 Partition 已经是就绪的 send buffer
    → 消除了 HybridEP 中的 "读数据 → 共享内存 → FIFO" 步骤
    → 直接从 Partition 地址发起 RDMA/NVLink 传输

  时间线对比：
    传统 A2A：[Gate][Route][Permute][Pack][════ A2A ════][Unpack][Expert]
    HybridEP：[Gate][Route][══ HybridEP Dispatch ══][Expert]
    MoEX：    [Gate+Scatter][→流式D→][Expert on partial data...]
              ↑ 最早的 token 在 Gate 完成后 ~微秒 内就开始传输
```

### 6.3 FWD-BWD Merged Communication Overlap

```
构建在 MegatronCore 的 DualPipe 思路之上，但更激进：

MegatronCore 方案（1F1B FWD-BWD Merged）：
  micro_batch_0 BWD 与 micro_batch_1 FWD 合并
  双 CUDA Stream：Compute + Comm
  效果：93% EP 通信被掩盖

MoEX 增强（基于 CommTensor 的零拷贝特性）：

  增强 1：BWD W/D Split 更精细
    传统 W/D Split：整个 MLP 的 dW 和 dX 分离
    MoEX：per-Expert dW 和 dX 分离
      → dX_expert_i 完成后立即 scatter 到 combine buffer
      → 不等其他 Expert 的 dX 完成
      → combine 通信粒度从 "整层" 降到 "单 Expert"

  增强 2：梯度 AllReduce 完全隐藏
    dW 的 AllReduce 使用 RDMA Zone 的持久缓冲区
    dX 的 combine 使用 NVLink Zone 的持久缓冲区
    → 两类通信走不同物理路径，真正并发

  增强 3：Attention BWD 与 MoE Combine 重叠
    CommTensor 使 MoE Combine 的结果直接写入下一层的输入
    → Attention BWD 可以在 MoE Combine 还在进行时就开始处理已到达的 token

  时间线：
    传统：
      Stream 0: [F/attn][F/expert]         [B/expert]  [B/attn]
      Stream 1:        [F/dispatch][F/combine]  [B/dispatch][B/combine]
    
    MoEX：
      Compute:  [F/attn][F/expert]  [B/expert_dX][B/expert_dW][B/attn]
      NVLink:          [→F/disp→][→F/comb→]    [→B/disp→][→B/comb→]
      RDMA:                                                [→dW_AR→]
                       ↑ 全部重叠 ↑               ↑ 全部重叠 ↑

  预期效果：EP 通信暴露时间 < 2% 总迭代时间（vs MegatronCore 的 < 5%）
```

---

## 7. Layer 5: Heterogeneous Parallel Mapper (HPM)

### 7.1 CommTensor 使 Parallel Folding 零开销

```
MegatronCore Parallel Folding 的代价：
  Attention 使用 TP×CP×DP，MoE 使用 ETP×EP×EDP
  两套并行配置的切换需要重组 Tensor 布局
  → 有隐含的内存重排开销

MoEX 的 Parallel Folding：
  CommTensor 的 Dual-View 使切换免费：
    Attention 结束 → 切到 Comm View → MoE Dispatch
    MoE 结束 → 切到 Compute View → 下一层 Attention

  关键：视图切换只更新描述符（几 KB），不搬运数据
  
  进一步优化：
    Attention 输出可以直接写入 CommTensor 的 DPL 格式
    只需要 Attention GEMM 的 epilogue 支持 indexed scatter
    → 消除 Attention → MoE 转换点的所有数据搬运
```

### 7.2 自适应并行配置

```
基于 CommTensor 的路由统计（routing metadata），运行时调整：

  监控指标（每 K steps）：
    per_expert_load[E]       ← CommTensor 路由统计自然积累
    cross_node_traffic       ← 从 RDMA Zone 使用量直接读取
    nvlink_utilization       ← 从 NVLink Zone 使用量直接读取
    expert_compute_time      ← 从 Expert GEMM 耗时读取

  自适应决策：
    IF cross_node_traffic > threshold:
        增大 EP（更多 Expert 在本地）或 启用 ECHO 克隆
    IF nvlink_utilization < 50%:
        考虑增大 TP（更多 Attention 通信走 NVLink）
    IF expert_compute_time variance > threshold:
        触发 Expert Re-layout（类 LAER-MoE，但基于 TAMP Zone）

  MoEX 的优势：所有监控信息天然在 CommTensor 中
  → 不需要额外的 profiling 基础设施
  → 决策延迟 < 1ms（读取已有数据）
```

---

## 8. 性能模型与分析

### 8.1 数据搬运次数对比

```
操作                    传统 EP    MegatronCore    MoEX
─────────────────────────────────────────────────────────
Gate GEMM               1R        1R              1R（融合）
Permute                 1R+1W     —（融合）       —
Pack (dispatch)         1R+1W     1R+1W（融合）    —（DPL 免费）
Quantize (FP8)          1R+1W     1R+1W           —（scatter 时融合）
A2A Dispatch            网络      网络            网络（流式）
Unpack                  1R+1W     1R+1W           —（DPL 免费）
Padding                 1R+1W     —（融合）       —（DPL padding）
Expert GEMM             1R+1W     1R+1W           1R+1W
Unpadding               1R+1W     —（融合）       —
Pack (combine)          1R+1W     1R+1W           —（scatter epilogue）
A2A Combine             网络      网络            网络（流式）
Unpermute+Combine       1R+1W     1R+1W（高效）    1R+1W
─────────────────────────────────────────────────────────
总 HBM 读写次数         ~14       ~8              ~4
（不含网络通信）

理论 HBM 带宽节省：
  vs 传统 EP：    (14-4)/14 ≈ 71%
  vs MegatronCore：(8-4)/8 ≈ 50%
```

### 8.2 端到端性能预估

```
DeepSeek-V3 训练性能预估（256 GB200, SeqLen=4096, MXFP8）：

MegatronCore 基线：1,048 TFLOPS/GPU

MoEX 改进来源：
  ① 数据搬运减少 50%
     Expert GEMM 等待输入的 stall 减少
     MoE 层时间减少 ~15-20%
     
  ② 流式 dispatch（Streaming A2A）
     通信暴露时间从 <5% → <2%
     
  ③ Pre-Routed Pipeline
     提前 0.5-2ms 开始 dispatch
     
  ④ 持久通信缓冲区
     消除 per-iteration 注册开销
     CUDA Graph 更完整覆盖
     
  综合预估：
    MoE 层加速 ~20-25%
    Attention 层不变
    MoE 层占比 ~40% 总时间（FP8 + Overlap 后）
    
    端到端加速 ≈ 1 / (0.6 + 0.4 × 0.75) = 1 / 0.9 ≈ +11%
    
  预估性能：1,048 × 1.11 ≈ 1,163 TFLOPS/GPU
  
  注：这是保守估计，不含 Pre-Routed Pipeline 和自适应并行的收益
```

### 8.3 内存开销分析

```
MoEX 额外内存开销：

  CommTensor 描述符：
    compute_index [T] + partition_desc [EP] + routing_metadata [T×K]
    ≈ T × 4B + EP × 20B + T × K × 8B
    ≈ 4096 × 4 + 64 × 20 + 4096 × 8 × 8 = 16 KB + 1.3 KB + 256 KB ≈ 0.3 MB
    → 可忽略
    
  持久通信缓冲区（TAMP）：
    send_partitions [EP][capacity][H] × BF16 + FP8
    ≈ 64 × 512 × 7168 × (2 + 1) = 64 × 512 × 7168 × 3 = 672 MB
    recv_partitions：同上 ≈ 672 MB
    总计 ≈ 1.3 GB
    
    vs 传统方案（每次 malloc/free）：
      峰值相同（都需要 send+recv buffer）
      但 MoEX 不需要碎片化的临时 buffer → 整体内存碎片更少
    
    vs MegatronCore 的持久双 buffer：
      思路相同，MoEX 增加了 FP8 通信副本 → 额外 ~0.4 GB
      
  NVLink Zone FP8 副本：
    每个 CommTensor 的 FP8 通信版本 ≈ 50% BF16 大小
    只对 MoE 层激活，约 ~0.5 GB
    
  总额外内存 ≈ ~2 GB（vs 199.5 GB 总内存 ≈ 1%）
```

---

## 9. 与现有工作的差异定位

### 9.1 系统级差异对比

```
维度              MegatronCore     FlowMoE      Comet      DeepEP      MoEX
──────────────────────────────────────────────────────────────────────────────
Tensor 格式      标准 contiguous  标准         标准       标准        CommTensor DPL
Permute 步骤     有（可融合）     有           有         有          消除（SoW）
Pack/Unpack      有（可融合）     有           有         优化        消除（DPL 直用）
通信缓冲区       临时/持久双buf    临时         临时       预分配      持久+拓扑分区
量化-通信融合     分离            分离         分离       分离        scatter 时融合
通信模式         批量 A2A        chunk A2A    tile 流    token 流    流式 partition
Overlap 粒度     层级（1F1B）    chunk 级     tile 级    token 级    partition 流式
路由-dispatch    两步            两步         两步       两步        一步（Gate-Scatter）
Parallel Folding 描述符切换      不支持       不支持     不支持      零拷贝视图切换
拓扑感知         进程组级         无           无         节点级      Zone 级内存分区
```

### 9.2 MoEX 的核心创新点

```
创新 1：CommTensor — 通信原生张量格式（全新抽象层）
  首次将通信目标编码进 Tensor 的物理存储布局
  Dual-View Descriptor 实现计算/通信视图零拷贝切换
  → 消除了 permute/pack/unpack 的根本原因

创新 2：Scatter-on-Write — 计算即通信（新执行模型）
  Gate-Scatter 融合内核：路由+数据组织一步完成
  Scatter-Grouped-GEMM：Expert 输出直接进入 combine buffer
  → 将 7 次全局内存读写降至 2 次

创新 3：Topology-Aware Memory Pool — 内存即拓扑（新内存架构）
  GPU 内存按通信拓扑物理分区
  持久预注册缓冲区 + RDMA/NVLink 分流
  → dispatch 从 "数据搬运" 变成 "指针发布"

创新 4：Pre-Routed Pipeline + Streaming — 时间即并行（新通信协议）
  投机路由 or 流水线化路由提前启动通信
  流式 partition dispatch 消除批量等待
  → 通信暴露时间从 <5% 降至 <2%

创新 5：通信驱动的自适应并行（新调度策略）
  CommTensor 天然积累路由统计 → 无开销监控
  Zone 使用率直接指导并行配置调整
  → 在线自适应，决策延迟 < 1ms
```

---

## 10. 实现路线图

### 10.1 阶段规划

```
Phase 0（1 个月）：CommTensor 原型 + 可行性验证
  目标：验证 DPL 格式的计算兼容性
  工作：
    ① 实现 CommTensor 数据结构（Dual-View Descriptor）
    ② 验证 LayerNorm / Residual / Gate 在 DPL 格式上的正确性
    ③ 实现 Compute View 的 indexed access（for Attention）
    ④ 单 GPU 功能测试
  交付：CommTensor 原型 + 正确性验证报告

Phase 1（2-3 个月）：Gate-Scatter 融合内核 + TAMP
  目标：消除 permute+pack，实现拓扑内存分区
  工作：
    ① 实现 Gate-Scatter 融合 CUDA/HIP 内核
    ② 实现 TAMP（NVLink/RDMA Zone 分区 + 持久缓冲区）
    ③ 基于 DeepEP/HybridEP 实现流式 partition dispatch
    ④ 8 GPU 端到端功能验证
  交付：消融实验数据（HBM 带宽节省 + dispatch 延迟）

Phase 2（2-3 个月）：Scatter-Grouped-GEMM + Streaming Overlap
  目标：消除 Expert GEMM 后的 pack，实现流式通信
  工作：
    ① 实现 Scatter epilogue for Grouped GEMM（基于 cuteDSL）
    ② 实现 FWD-BWD Merged Streaming Overlap
    ③ 实现 Pre-Routed Pipeline（策略 B：流水线化路由）
    ④ 64+ GPU 扩展性测试
  交付：端到端吞吐对比数据

Phase 3（2-3 个月）：自适应并行 + 生产化
  目标：自适应配置 + CUDA Graph 全覆盖
  工作：
    ① 基于 CommTensor 路由统计的自适应并行配置
    ② Full CUDA Graph 覆盖（CommTensor 支持静态描述符）
    ③ 与 Megatron-Core 集成（复用 PP/VPP/FSDP 基础设施）
    ④ DeepSeek-V3 / Qwen3-235B 规模验证
  交付：完整系统 + 论文初稿

Phase 4（1-2 个月）：论文写作与实验
  目标会议：OSDI / SOSP / MLSys
```

### 10.2 关键技术风险

```
风险 R1：DPL 格式对 Attention 性能的影响
  问题：Attention 需要 Compute View 的 indexed access，可能影响 FlashAttention 性能
  缓解：
    方案 A：在 Attention 前做一次轻量 gather 到 contiguous（代价 < permute）
    方案 B：修改 FlashAttention 支持 indexed input（需要上游支持）
    方案 C：仅对 MoE 层激活使用 CommTensor，Attention 使用标准 Tensor
    → Phase 0 验证哪个方案最优

风险 R2：Gate-Scatter 的原子操作竞争
  问题：多个 token 同时写入同一 Partition 时需要 atomicAdd
  缓解：
    方案 A：per-warp local count → 最终 reduce（减少竞争）
    方案 B：预分配固定 slot（Padded Static 模式，无 atomic）
    方案 C：两阶段 scatter（先 count，再 write）

风险 R3：Scatter-Grouped-GEMM 的 Kernel 复杂度
  问题：修改 GEMM epilogue 需要深入 CUTLASS/cuteDSL 内部
  缓解：
    短期：用独立的 scatter kernel 作为 GEMM 后的融合步骤
    长期：等 cuteDSL Grouped GEMM 的 custom epilogue 特性成熟

风险 R4：投机路由的准确率
  问题：相邻层路由相关性假设可能在某些模型上不成立
  缓解：
    使用策略 B（流水线化路由）作为 fallback（不依赖预测）
    投机路由只在相关性 > 80% 时启用
```

---

## 11. 论文故事线

```
Title: MoEX: Communication-First Tensor Design for 
       Scalable Mixture-of-Experts Training

Narrative Arc:

  Problem（第 1-2 节）：
    "MoE 训练中，一个 token 经历 7 次全局内存读写但只有 1 次真正计算"
    → 量化「数据搬运税」的概念
    → 证明这是所有 MoE 系统的根本瓶颈

  Insight（第 3 节）：
    "问题的根源不在通信算法，而在 Tensor 格式假设"
    → 所有系统假设 Tensor 是 compute-contiguous → 通信必然需要重排
    → 如果 Tensor 从诞生就是 communication-partitioned → 重排被消除

  Design（第 4-8 节）：
    CommTensor → Scatter-on-Write → TAMP → ONCP → HPM
    从底层格式到上层调度，全栈通信优先设计

  Evaluation（第 9-10 节）：
    ① 消融实验：每层创新的独立贡献
    ② 端到端对比：vs MegatronCore / FlowMoE / DeepEP
    ③ 规模扩展：256 → 1024 GPU
    ④ 不同模型：DeepSeek-V3 / Qwen3-235B / Mixtral

  核心卖点：
    "MoEX 将 MoE 层的全局内存读写从 7 次降至 2 次，
     在 GB200 上实现 DeepSeek-V3 训练 X TFLOPS/GPU，
     比 MegatronCore 提升 Y%。"
```

---

## 12. 开放问题与未来方向

```
Q1：CommTensor 能否扩展到 MoE 推理？
  推理场景：KV Cache 也可以按 destination partition
  → 推理的 prefill 和 decode 阶段使用不同的 CommTensor 视图

Q2：CommTensor 与 FP4 训练的交互？
  NVFP4 的 2D scaling + RHT 需要知道 tensor 的全局统计
  CommTensor 的 per-partition 组织可能需要 per-partition scaling
  → 需要设计 partition-aware 量化方案

Q3：动态 Expert 数量（Adaptive MoE）？
  如果 Expert 数量本身是动态的（训练中增减 Expert）
  CommTensor 的 partition 数量需要动态调整
  → TAMP 的 Zone 重分配策略

Q4：跨模态 MoE（Vision-Language）？
  不同模态的 token 有不同的路由特征
  → CommTensor 可以按 modality 分区（在 destination 分区之上再加一层）

Q5：与稀疏注意力的协同？
  如果 Attention 也是稀疏的（如 Mixture-of-Attention）
  → CommTensor 的 Dual-View 可以扩展为 Triple-View
     （Attention View + MoE View + Dense View）
```

---

*MoEX 系统设计 | 2026-03-11 | AIInfra-Book*
*参考论文：MegatronCore MoE · FlowMoE · Comet · LAER-MoE · DeepEP/HybridEP · MegaScale-MoE · SwiftMoE · MoE Parallel Folding*
