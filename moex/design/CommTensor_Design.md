# CommTensor 设计文档

> MoEX 的核心创新：Communication-Native Tensor（通信原生张量）
>
> 设计原则：**让 tensor 的物理存储顺序 = 通信目标顺序**，从而消除 dispatch/combine
> 中的所有内存拷贝，并天然支持 tile 级 GEMM-RDMA pipeline 和 FSEP ReduceScatter。

---

## 目录

1. [问题分析：传统 MoE Dispatch 的内存拷贝链](#1-问题分析传统-moe-dispatch-的内存拷贝链)
2. [CommTensor 物理布局设计](#2-commtensor-物理布局设计)
3. [CommTensor 元数据设计](#3-commtensor-元数据设计)
4. [关键操作的零拷贝实现](#4-关键操作的零拷贝实现)
5. [布局转换：Sequence-Ordered ↔ Rank-Ordered](#5-布局转换sequence-ordered--rank-ordered)
6. [FSEP 分片支持](#6-fsep-分片支持)
7. [Tile 对齐与 GEMM-RDMA Pipeline](#7-tile-对齐与-gemm-rdma-pipeline)
8. [内存管理策略](#8-内存管理策略)
9. [API 设计](#9-api-设计)
10. [与论文的对应关系](#10-与论文的对应关系)

---

## 1. 问题分析：传统 MoE Dispatch 的内存拷贝链

### 1.1 传统实现的数据流

```
hidden_states: [B*L, H]  （sequence 顺序，即 token 0, token 1, ..., token B*L-1）

Step 1: Gate → routing
  indices:  [B*L, K]  (每个 token 的 top-K expert 下标)
  scores:   [B*L, K]  (对应路由权重)

Step 2: Permute（第 1 次 O(T*H) 拷贝）
  sorted_hidden = hidden_states[perm_indices]
  # perm_indices = argsort by (expert_id, token_order)
  # sorted_hidden: [B*L*K, H]，按 expert 顺序排列

Step 3: Pack into send buffer（第 2 次 O(T*H) 拷贝）
  send_counts = [count_to_rank_0, count_to_rank_1, ..., count_to_rank_R-1]
  send_buffer = pack(sorted_hidden, send_counts)  # contiguous send buffer

Step 4: All-to-All Dispatch（通信）
  recv_buffer = all_to_all(send_buffer, ...)  # 接收其他 rank 的 tokens

Step 5: Expert GEMM（在 recv_buffer 上）
  expert_output = expert_mlp(recv_buffer)     # [T_local, H]

Step 6: All-to-All Combine（通信）
  send_back = all_to_all(expert_output, ...)

Step 7: Unpack（第 3 次 O(T*H) 拷贝）
  unpacked = unpack(send_back, original_positions)

Step 8: 加权求和（第 4 次 O(T*H) 拷贝+写）
  output = scatter_add(unpacked * scores, original_indices, dim=0)
```

**总结：** 4 次 O(T×H) 内存操作（T = B*L*K，H = 隐藏层维度）  
以 DeepSeek-V3 规模：T×H = 4096 × 7168 × 2（TopK=2）≈ 117MB fp16 per dispatch

### 1.2 内存带宽浪费分析

```
HBM3e 带宽（A100 80GB）: 2 TB/s
HBM3e 带宽（H100 SXM）:  3.35 TB/s
HBM3e 带宽（MI300X）:    5.3 TB/s

4 次 O(T×H) 拷贝 @ T×H = 117MB：
  - 总 HBM 流量：117MB × 4 次 × (read+write) = ~936MB 内存流量
  - 在 H100 上耗时：936MB / 3350 GB/s = ~0.28ms

每个 MoE layer 的 A2A 延迟（DeepSeek-V3规模，IB网络）：~1-2ms
内存拷贝开销占 A2A 延迟的 14-28%！
这还不考虑 cache miss、内存碎片等实际效应。
```

---

## 2. CommTensor 物理布局设计

### 2.1 四维布局

```
CommTensor: Tensor[R, S, T, H]

R = num_ep_ranks       # Expert Parallel 组中的 GPU 数量（通信目标数）
S = max_slots_per_rank # 每个 rank 预分配的 token 槽位数
T = tile_size          # GEMM tile 粒度（元素数，非字节数）
H = d_model            # 隐藏层维度
```

### 2.2 维度含义与设计依据

```
维度 R（Rank 维，最外层）：
  - 每个 rank 在物理内存中占连续的 S×T×H 区间
  - CT[r, :, :, :] 是发送给 rank r 的全部数据，地址连续
  - → RDMA one-sided put 只需一个 (src_addr, size, dst_rank) 三元组
  - → 无需 scatter/gather list（InfiniBand SGE），减少 NIC 开销

维度 S（Slot 维）：
  - 预分配：避免 dispatch 时动态内存分配（消除 CUDA malloc latency ~50µs）
  - 填充：CT[r, :slot_counts[r], :, :] 有效，CT[r, slot_counts[r]:, :, :] 为 padding
  - S 的选择：= max_tokens_per_step / R × safety_factor（通常取 2.0）
  - Slot 内按 tile 对齐：slot 大小 = ceil(H / T) * T（tile 对齐）

维度 T（Tile 维）：
  - 对齐 GEMM tile 大小（NVIDIA H100 推荐 128×128 = 128 elements）
  - CT[:, :, t, :] 是一个 tile batch，GEMM 完成后立即可 RDMA 发送
  - → 实现 Comet-style kernel-level GEMM-RDMA pipeline
  - 注意：T 不是 token 数，而是 hidden dim 的分割粒度

维度 H（Hidden 维，最内层）：
  - 与 standard tensor 相同
  - T 必须整除 H：H % T == 0
  - 如果 H 不被 T 整除，对最后一个 tile padding（零）
```

### 2.3 地址计算

```python
# CommTensor 中 token (rank=r, slot=s) 的 tile t 的起始地址：
addr = base + r * (S * T_full * H) + s * (T_full * H) + t * H
# T_full = ceil(H / T)，总 tile 数

# 对比传统 tensor [B*L, H] 中 token i 的地址：
addr_trad = base + i * H
# 问题：token i 的目标 rank = routing[i]，不连续！
```

### 2.4 布局对比可视化

```
传统布局（Sequence-Ordered）：
内存: [tok0|tok1|tok2|tok3|tok4|tok5|tok6|tok7]
       ↑dest=R1  ↑dest=R0  ↑dest=R1  ↑dest=R2  ↑dest=R0  ↑dest=R1  ↑dest=R0  ↑dest=R2

要发送给 R0 的 tokens：{tok1, tok4, tok6} → 地址不连续，需要 gather

CommTensor（Rank-Ordered）：
内存: [R0:tok1,tok4,tok6 | R1:tok0,tok2,tok5 | R2:tok3,tok7]
      ←─── R0 区域 ───→ ←──── R1 区域 ────→ ←── R2 区域 ──→

发送给 R0：CT[0, 0:3, :, :] → 连续地址，一次 RDMA！
```

### 2.5 CommTensor 与标准 PyTorch Tensor 的关系

CommTensor 是对 PyTorch Tensor 的轻量级包装，**共享底层存储**：

```python
class CommTensor:
    data: torch.Tensor     # 实际数据，shape [R, S, T_num_tiles, H]
                           # T_num_tiles = ceil(d_model / tile_size)
    meta: CommTensorMeta   # 路由元数据（见第 3 节）
    config: CommTensorConfig

    # 视图操作（零拷贝）
    def view_rank(self, rank: int) -> torch.Tensor:
        """返回发往指定 rank 的数据视图（连续内存，零拷贝）"""
        count = self.meta.slot_counts[rank]
        return self.data[rank, :count, :, :]  # [count, T_tiles, H]

    def view_tile(self, rank: int, slot: int, tile: int) -> torch.Tensor:
        """返回特定 tile 的视图（用于 GEMM tile-by-tile 处理）"""
        return self.data[rank, slot, tile, :]  # [H]
```

---

## 3. CommTensor 元数据设计

### 3.1 元数据结构

```python
@dataclass
class CommTensorMeta:
    # 路由映射
    token_indices: torch.Tensor    # [R, S] int32
    # token_indices[r, s] = 原始 hidden_states 中的 token 下标
    # token_indices[r, s] == -1 表示该槽位未使用（padding）

    routing_scores: torch.Tensor   # [R, S] fp16/bf16
    # routing_scores[r, s] = 路由权重（softmax 后，用于 combine 加权）
    # padding 槽位的 score = 0.0

    slot_counts: torch.Tensor      # [R] int32
    # slot_counts[r] = rank r 实际使用的槽位数
    # 有效范围：data[r, 0:slot_counts[r], :, :]

    # 布局版本（支持 LAER-MoE 动态 re-layout）
    layout_version: int            # 单调递增，re-layout 时++
    layout_timestamp: int          # 对应的训练 step

    # FSEP 元数据
    is_fsep_sharded: bool          # 是否已进行 FSEP 分片
    fsep_shard_rank: int           # 本 GPU 的 FSEP 分片下标（-1 表示未分片）
    fsep_num_shards: int           # FSEP 分片总数

    # Tile 信息
    tile_size: int                 # tile 的 hidden dim 大小（元素数）
    num_tiles: int                 # = ceil(d_model / tile_size)

    # 通信状态（用于 OverlapScheduler）
    dispatch_done: torch.Tensor    # [R] bool，每个 rank 的 dispatch 是否完成
    combine_done: torch.Tensor     # [R] bool，每个 rank 的 combine 是否完成
```

### 3.2 元数据内存开销

```
token_indices:  R × S × 4 bytes (int32)
routing_scores: R × S × 2 bytes (fp16)
slot_counts:    R × 4 bytes
dispatch/combine_done: R × 1 byte

典型配置（R=8, S=512）：
  token_indices:  8 × 512 × 4 = 16 KB
  routing_scores: 8 × 512 × 2 = 8 KB
  其余：可忽略
  总计：~24 KB

与主数据相比（R=8, S=512, H=4096, fp16）：
  data: 8 × 512 × 4096 × 2 = 32 MB
  meta overhead: 24 KB / 32 MB = 0.075%（可忽略）
```

### 3.3 路由直写元数据（零额外 Buffer）

传统路由输出写入 `expert_indices [B*L, K]`，然后再 permute。
MoEX 路由直接写入 CommTensor meta：

```python
def route_to_comm_tensor(
    hidden_states: Tensor,    # [B*L, H]
    gate_logits: Tensor,      # [B*L, num_experts]
    config: CommTensorConfig,
) -> CommTensor:
    B_L = hidden_states.shape[0]

    # Step 1: TopK（标准操作）
    scores, expert_ids = gate_logits.topk(config.top_k, dim=-1)
    scores = torch.softmax(scores, dim=-1)  # [B*L, K]

    # Step 2: 计算每个 rank 的 token 分配（轻量，O(B*L*K)）
    rank_ids = expert_ids // config.experts_per_rank  # [B*L, K] → rank ID
    # rank_ids[i, k] = token i 的第 k 个 expert 所在的 rank

    # Step 3: 分配 slot（原子计数器，避免 sort）
    ct = CommTensor.allocate(config)  # 从预分配池获取，零 malloc
    slot_cursors = torch.zeros(config.num_ep_ranks, dtype=torch.int32)

    for i in range(B_L):
        for k in range(config.top_k):
            r = rank_ids[i, k].item()
            s = slot_cursors[r].item()
            ct.meta.token_indices[r, s] = i
            ct.meta.routing_scores[r, s] = scores[i, k]
            slot_cursors[r] += 1

    ct.meta.slot_counts = slot_cursors

    # Step 4: 将数据写入 CommTensor（使用 index_select，单次写）
    for r in range(config.num_ep_ranks):
        count = slot_cursors[r].item()
        if count > 0:
            tok_ids = ct.meta.token_indices[r, :count]
            ct.data[r, :count, 0, :] = hidden_states[tok_ids]
            # 注意：tile 维度 T 在此为第 2 维，按 H 分片（在 GEMM 时处理）

    return ct
    # 注：Step 4 的 index_select 仍有 1 次 O(T*H) 拷贝
    # 优化路径：使用 CUDA kernel 将 route+scatter 合并为单次 pass（见 code/）
```

**关键洞察**：数据写入 CommTensor 后，后续所有操作（dispatch、GEMM、combine）
均为零拷贝。1 次写入，替代原来的 4 次拷贝。

---

## 4. 关键操作的零拷贝实现

### 4.1 Dispatch（零拷贝 RDMA）

```python
def dispatch(ct: CommTensor, comm_group: ProcessGroup) -> List[CommTensor]:
    """
    零拷贝 dispatch：直接从 CommTensor 的连续内存区域 RDMA 发送

    传统：
      hidden → permute（拷贝）→ pack（拷贝）→ A2A → unpack（拷贝）

    MoEX：
      ct.data[r] 已连续 → 直接 RDMA put（零拷贝）
    """
    remote_cts = []
    for r in range(ct.config.num_ep_ranks):
        if r == get_rank():
            # 本地数据：直接切片，零拷贝
            local_ct = ct.view_rank(r)
            remote_cts.append(local_ct)
        else:
            count = ct.meta.slot_counts[r].item()
            if count > 0:
                # RDMA one-sided PUT：src = ct.data[r], dst = peer_buffer[my_rank]
                rdma_put_async(
                    src_tensor=ct.data[r, :count, :, :],  # 连续内存！
                    dst_rank=r,
                    dst_offset=get_rank() * ct.config.max_slots_per_rank,
                    stream=comm_stream,
                )

    # 同时发送元数据（slot_counts，用于接收方知道接收了多少）
    # 元数据很小（R × 4 bytes），可以走 control plane

    return remote_cts  # 接收到的 CommTensor 片段（RDMA 完成后有效）
```

### 4.2 Combine（零拷贝 + Index Scatter）

```python
def combine(
    expert_output_ct: CommTensor,  # expert 计算结果（rank-ordered）
    original_meta: CommTensorMeta, # 原始路由元数据
    output: Tensor,                # [B*L, H]，输出（sequence-ordered）
) -> Tensor:
    """
    零拷贝 combine：RDMA 直接写入 output 的对应位置

    核心操作：scatter_add（非拷贝，原地累加）
    """
    for r in range(expert_output_ct.config.num_ep_ranks):
        count = original_meta.slot_counts[r].item()
        if count == 0:
            continue

        # expert_output 已通过 RDMA 接收到本地
        expert_out = expert_output_ct.data[r, :count, :, :]  # [count, T_tiles, H]
        expert_out_flat = expert_out.view(count, -1)          # [count, H]

        # 路由权重（从 meta 读取，无需额外 buffer）
        scores = original_meta.routing_scores[r, :count].unsqueeze(-1)  # [count, 1]

        # Scatter-add 到 output（原地操作，无拷贝）
        token_ids = original_meta.token_indices[r, :count]   # [count]
        output.scatter_add_(
            dim=0,
            index=token_ids.unsqueeze(-1).expand(-1, output.shape[-1]),
            src=expert_out_flat * scores,
        )

    return output  # 加权求和已完成
```

### 4.3 为什么 scatter_add 不算"拷贝"

严格来说，scatter_add 是一次写操作（O(T×H)），但它：
1. **原地写入**：写入最终目标内存，无中间 buffer
2. **与计算融合**：乘以 routing_scores 与写入同时完成
3. **无读取已有数据**：acc_add 模式，不需要先读后写

传统实现：read(recv_buffer) → write(unpacked) → read(unpacked) → write(output)
MoEX：read(expert_out) * score → write(output)

省去了 2 次读操作，**内存带宽节省 ~50%**。

---

## 5. 布局转换：Sequence-Ordered ↔ Rank-Ordered

### 5.1 Fold 操作（Attention 输出 → CommTensor）

对应 MoE Parallel Folding 论文中的 Fold 操作，但 MoEX 将其与路由合并：

```
传统流程：
  Attention 输出 [B*L, H] → All-Reduce（TP 同步）→ Gate → permute → A2A

MoEX 流程：
  Attention 输出 [B*L, H] → Gate → route_to_comm_tensor（直接写 CommTensor）
                                  → CommTensor[R, S, T_tiles, H]（dispatch 就绪）

合并 Fold 与 Dispatch：
  传统 Fold 的 All-to-All（TP→EP 转换）+ Dispatch 的 All-to-All 合并为 1 次通信
  CommTensor 的 rank 维度直接反映 EP rank，无需额外 Fold 通信
```

### 5.2 Unfold 操作（CommTensor → Attention 输入）

Combine 完成后，输出已是 sequence-ordered（通过 scatter_add 写回原始位置），
无需单独的 Unfold 操作：

```python
# combine 的输出天然是 sequence-ordered：
output: Tensor[B*L, H]  # scatter_add 写入了正确的位置
# 等价于 Unfold（无需额外通信）
```

### 5.3 与 Parallel Folding 的差异

| 方面 | MoE Parallel Folding | MoEX |
|------|---------------------|------|
| Fold 实现 | 独立 All-to-All | 合并到 Dispatch |
| Unfold 实现 | 独立 All-to-All | 合并到 Combine (scatter_add) |
| 额外通信 | 2 次 All-to-All（Fold+Unfold） | 0 次额外通信 |
| 对 TP 的支持 | 支持不同 TP | 路由阶段前需 TP All-Reduce |

---

## 6. FSEP 分片支持

### 6.1 FSEP 与 CommTensor 的天然契合

FSEP（Fully Sharded Expert Parallel，来自 LAER-MoE）将每个 expert 的参数
分片存储在所有 GPU 上。CommTensor 的 [R, S, T, H] 布局与 FSEP 高度契合：

```
传统 EP（每 GPU 持有完整 expert）：
  Dispatch: token → rank_r（持有 expert_e）
  GEMM: W_expert_e [H, 4H] × token [H] → output [4H]

FSEP（每 GPU 持有所有 expert 的 1/R 份）：
  Dispatch: token → broadcast 到所有 rank
  GEMM: W_shard [H, 4H/R] × token [H] → partial_output [4H/R]
  ReduceScatter: partial_output → full_output（token 维度分配）

CommTensor 在 FSEP 下的变化：
  ct.data[r, s, :, :] = token s 的数据（发送给所有 rank）
  → FSEP dispatch: 每个 ct.data[r, s] 广播给 R 个 rank（一次 broadcast）
  → partial GEMM: 在每个 rank 上独立执行
  → ReduceScatter on NVLink: 利用 896 GB/s NVLink（vs IB 的 400 GB/s）
```

### 6.2 CommTensor FSEP 扩展布局

```
FSEP CommTensor: [R, S, T_tiles, H_shard]
                                  ↑
                             H_shard = H / R（本 GPU 负责的 hidden 分片）

ct.meta.is_fsep_sharded = True
ct.meta.fsep_shard_rank = my_rank
ct.meta.fsep_num_shards = R

FSEP GEMM（第 r_expert 个专家）：
  W_shard = expert_params[r_expert, :, my_shard*H_shard:(my_shard+1)*H_shard]
  partial_out = GEMM(ct.data[:, :, t, :], W_shard)  # [R, S, H_shard]

ReduceScatter（FSEP 特有，走 NVLink）：
  full_out = all_reduce_scatter(partial_out, group=ep_intra_node_group)
  # 每个 GPU 获得 full output 的 1/R token 子集的完整 H 维度
```

### 6.3 FSEP ReduceScatter 为什么走 NVLink

```
InfiniBand 带宽（跨节点）：~400 Gb/s per port = ~50 GB/s
NVLink 带宽（节点内）：  H100 NVSwitch: 900 GB/s bidirectional
                         MI300X XGMI:  896 GB/s effective

FSEP 策略：
  All-to-All（跨节点 dispatch）：走 InfiniBand，必须跨节点
  ReduceScatter（节点内聚合）：  走 NVLink，18× 更快！

CommTensor 的 FSEP 实现确保：
  ct.meta 中记录哪些 rank 在同一节点内（intra_node_ranks）
  ReduceScatter 使用 intra_node_group（NVLink 组），不走 IB
```

---

## 7. Tile 对齐与 GEMM-RDMA Pipeline

### 7.1 Tile 维度设计依据（来自 Comet 论文）

Comet 的核心发现：GEMM tile 完成后立即 RDMA 发送，而无需等待全部 GEMM 完成，
可将 overlap 率从 68%（chunk 级）提升到 90%+（tile 级）。

CommTensor 的 T 维度直接对应 GEMM tile：

```
GEMM tile 大小（H100 Tensor Core 最优）：128 × 128 = 128 elements
→ CommTensor tile_size T = 128

Expert GEMM（W: [H, 4H]）：
  分解为 num_tiles = 4H / T = 4*4096 / 128 = 128 个 tile

每个 tile 的 GEMM 完成后：
  result_tile: [num_tokens, H]（已在 register file 中）
  → 立即 RDMA send（无需写回 HBM）
  → Compute Warp 开始下一个 tile
  → 实现 zero-copy, HBM-bypass RDMA！
```

### 7.2 Warp 专化（来自 Comet）

```
传统：单一 warp pool，计算和通信串行

MoEX：
  ├── Compute Warps（80%）：执行 GEMM tiles
  └── Comm Warps（20%）：  监控 tile 完成，触发 RDMA

协调机制（shared memory）：
  tile_done[t] = 1      # Compute Warp 写（tile t 完成）
  while tile_done[t] == 0: pass  # Comm Warp 自旋等待
  rdma_put(tile_result[t], ...)  # Comm Warp 触发 RDMA
```

### 7.3 CommTensor tile 视图与 RDMA 地址

```python
def get_tile_rdma_descriptor(ct: CommTensor, rank: int, slot: int, tile: int):
    """
    获取 tile 的 RDMA 描述符（src_addr, size, dst_rank, dst_offset）

    由于 CommTensor 的连续布局，地址计算为 O(1) 算术运算，无需查表。
    """
    src_ptr = ct.data.data_ptr() + (
        rank * ct.config.max_slots_per_rank * ct.config.num_tiles * ct.config.d_model +
        slot * ct.config.num_tiles * ct.config.d_model +
        tile * ct.config.d_model
    ) * ct.data.element_size()

    return RDMADescriptor(
        src_ptr=src_ptr,
        size=ct.config.d_model * ct.data.element_size(),
        dst_rank=rank,
        dst_offset=...,  # 对端 CommTensor 的对应位置
    )
```

---

## 8. 内存管理策略

### 8.1 CommTensor 池（避免动态 malloc）

```python
class CommTensorPool:
    """
    预分配的 CommTensor 池，避免 dispatch 时动态内存分配

    CUDA malloc 延迟：~50-100µs（不可接受）
    池化分配延迟：~1µs（原子操作）
    """
    def __init__(self, config: CommTensorConfig, pool_size: int = 4):
        # 预分配 pool_size 个 CommTensor（用于流水线：当前用 1 个，下一个准备 1 个）
        self.tensors = [CommTensor.allocate(config) for _ in range(pool_size)]
        self.cursor = 0

    def acquire(self) -> CommTensor:
        ct = self.tensors[self.cursor % len(self.tensors)]
        self.cursor += 1
        return ct

    def release(self, ct: CommTensor):
        # 重置 meta（O(R)，极轻量）
        ct.meta.slot_counts.zero_()
        ct.meta.dispatch_done.zero_()
        ct.meta.combine_done.zero_()
        ct.meta.layout_version = 0
```

### 8.2 内存使用分析

```
CommTensor 的峰值内存（相比传统实现）：

传统实现：
  hidden_states: B*L × H × 2 bytes
  sorted_tokens: B*L*K × H × 2 bytes（K = top_k = 2）
  send_buffer:   B*L*K × H × 2 bytes
  recv_buffer:   B*L*K × H × 2 bytes
  expert_output: B*L*K × H × 2 bytes
  weighted_out:  B*L × H × 2 bytes
  峰值：约 6 × B*L × H × 2 bytes（TopK=2）

MoEX CommTensor：
  hidden_states: B*L × H × 2 bytes（必需）
  CommTensor:    R × S × T_tiles × H × 2 bytes
               = R × (B*L*K/R × 2) × 1 × H × 2 bytes（S = safety_factor × tokens_per_rank）
               = 2K × B*L × H × 2 bytes（K=2, safety_factor=2）
               = 4 × B*L × H × 2 bytes
  峰值：约 5 × B*L × H × 2 bytes（节省 ~17%）

注：CommTensor 复用了发送/接收 buffer，进一步节省内存。
实际节省取决于 overlap 程度（overlap 越好，峰值越低）。
```

---

## 9. API 设计

### 9.1 核心 API

```python
class CommTensor:
    """Communication-Native Tensor，MoEX 的基础数据结构"""

    # 构造
    @classmethod
    def from_hidden_states(
        cls,
        hidden_states: Tensor,        # [B*L, H]
        routing_indices: Tensor,       # [B*L, K] expert ID
        routing_scores: Tensor,        # [B*L, K] 路由权重
        config: CommTensorConfig,
    ) -> 'CommTensor': ...

    @classmethod
    def allocate(cls, config: CommTensorConfig) -> 'CommTensor': ...

    # 视图（零拷贝）
    def view_rank(self, rank: int) -> Tensor: ...
    def view_slot(self, rank: int, slot: int) -> Tensor: ...
    def view_tile(self, rank: int, slot: int, tile: int) -> Tensor: ...

    # 通信操作
    def dispatch_async(self, comm_group: ProcessGroup, stream: CudaStream) -> None: ...
    def combine_into(self, output: Tensor) -> None: ...

    # FSEP 操作
    def to_fsep(self, intra_node_group: ProcessGroup) -> 'CommTensor': ...
    def reduce_scatter_async(self, group: ProcessGroup, stream: CudaStream) -> None: ...

    # Layout 版本（用于 LAER-MoE re-layout）
    def update_layout(self, new_placement: ExpertPlacement) -> 'CommTensor': ...

    # 调试
    def __repr__(self) -> str: ...
    def memory_report(self) -> dict: ...


class CommTensorConfig:
    num_ep_ranks: int
    max_slots_per_rank: int          # = ceil(max_tokens / num_ep_ranks * safety_factor)
    tile_size: int                   # GEMM tile 大小（元素数）
    d_model: int
    dtype: torch.dtype
    use_fsep: bool = False
    top_k: int = 2
    experts_per_rank: int = 1        # 每个 EP rank 持有的专家数
    safety_factor: float = 2.0       # slot 预分配的安全系数
```

### 9.2 生命周期

```
CommTensor 生命周期：

[预分配] CommTensorPool.acquire()
    ↓
[路由填充] CommTensor.from_hidden_states()  ← 1次 index_select（写入CT）
    ↓
[Dispatch] CommTensor.dispatch_async()      ← RDMA send（零拷贝）
    ↓
[Expert GEMM] tile-by-tile（Comm Warp 同时触发 RDMA）
    ↓
[Combine] CommTensor.combine_into(output)   ← scatter_add（零拷贝）
    ↓
[归还] CommTensorPool.release()             ← O(R) meta 重置
```

---

## 10. 与论文的对应关系

| CommTensor 特性 | 对应论文 | 引用 |
|----------------|---------|------|
| [R,S,T,H] 四维布局 | 原创综合 | - |
| Rank 维（R）优先 | LAER-MoE FSEP, DeepEP | 连续内存 RDMA |
| Tile 维（T） | Comet MLSys'25 | tile-level GEMM-RDMA |
| 路由权重嵌入 meta | MegatronCore | 省 26.3 GB/GPU |
| FSEP ReduceScatter | LAER-MoE ASPLOS'26 | 1.69× 加速 |
| Dispatch 零拷贝 | DeepEP (DeepSeek) | token-centric dispatch |
| Slot 预分配 | MoEBlaze | 消除 malloc 延迟 |
| 布局版本管理 | LAER-MoE（re-layout）| 动态 expert 迁移 |
| Warp 专化 | Comet MLSys'25 | Comm/Compute Warp |
