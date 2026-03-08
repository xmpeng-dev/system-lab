# Feature 4 详细设计：CommTensor zero-copy

> **版本:** v0.1 | 2026-03-08

---

## 1. 问题定义

标准 MoE dispatch 的内存操作：

```
hidden_states [S, H]   ← 按 token 序号排列（batch-seq 顺序）
     ↓
sort_by_dst_rank()     ← pack 操作：重新排列为"按目标 GPU 分组"的顺序
     ↓                    内存拷贝 #1：S × H × dtype_size bytes
sorted_hidden [S, H]   ← 按目标 GPU 分组排列
     ↓
rccl_alltoall()        ← RCCL 做 DMA，发送到各 GPU
     ↓
dispatched [R, H]      ← 收到的 token（R = 本 GPU 负责的 Expert 收到的 token 数）
     ↓
Expert FFN
     ↓
rccl_alltoall()        ← 反向 A2A（combine）
     ↓
combined_sorted [S, H] ← 收到的结果（仍然是按发送方 GPU 分组的顺序）
     ↓
index_select()         ← unpack 操作：恢复原始 token 顺序
     ↓                    内存拷贝 #2：S × H × dtype_size bytes
combined [S, H]        ← 按原始 token 序号排列
```

**每次 MoE forward 有 2 次额外内存拷贝**，在 MI300X（HBM3 5.3 TB/s）上代价：
- S=4096, H=7168, bf16：4096 × 7168 × 2 = 58 MB
- 一次拷贝耗时：58 MB / 5300 GB/s ≈ **0.011 ms**
- 两次拷贝：**0.022 ms**
- 如果 A2A 总时间 = 10ms，则 pack/unpack 占比 **~0.2%**

> ⚠️ **关键不确定性**：pack/unpack 是否是显著瓶颈，需要实验 A 验证。
> 理论计算显示比例很低，但实际还需考虑内存访问 pattern（随机 index → cache miss）。

---

## 2. zero-copy 方案

### 2.1 核心思路

不在 dispatch 前 pack，而是**在 token 生成（embedding/attention 输出）时，直接写入按目标 GPU 分组的 buffer**：

```python
# 传统分配方式：
hidden = attention_output(...)  # shape: [S, H]，按 token 顺序

# zero-copy 分配方式（概念）：
sorted_buffer = allocate_sorted_buffer(routing_table, S, H)
# sorted_buffer 的物理内存布局已经是按目标 GPU 分组的
# attention 输出直接写入 sorted_buffer 的对应位置

# dispatch 时无需额外 pack：
dispatched = rccl_alltoall(sorted_buffer)
```

### 2.2 实际可行的简化版本

在不改 attention 输出逻辑的前提下，优化 pack 的实现：

```python
def dispatch_zero_copy(hidden_states, routing_table):
    """
    改进版 pack：使用 index_add_ + 预分配 sorted buffer，
    避免额外的内存分配和拷贝。
    """
    # 预分配目标 buffer（按目标 GPU 分组）
    sorted_buffer = torch.empty_like(hidden_states)
    index_map = routing_table.dst_sorted_indices  # [S]，每个 token 在 sorted_buffer 中的位置

    # 用 index_copy 直接写入（比 sort + gather 更高效）
    sorted_buffer.index_copy_(0, index_map, hidden_states)
    #  ↑ 仍然是一次拷贝，但用的是顺序写入而非随机写入
    #    可以配合 CUDA kernel 优化（coalesced write）

    dispatched = rccl_alltoall(sorted_buffer, routing_table.send_counts)
    return dispatched, index_map
```

### 2.3 combine 优化

```python
def combine_zero_copy(expert_output, index_map, routing_table):
    """combine 端的 unpack 优化"""
    # A2A gather（combine）
    combined_sorted = rccl_alltoall(expert_output, routing_table.recv_counts)

    # 用 index_map 恢复顺序
    # 原始：combined = combined_sorted[inverse_index_map]  ← 随机读
    # 优化：combined = torch.empty_like(combined_sorted)
    #        combined.index_copy_(0, inverse_index_map, combined_sorted)  ← 顺序写
    combined = torch.empty_like(combined_sorted)
    combined.index_copy_(0, routing_table.inverse_indices, combined_sorted)
    return combined
```

---

## 3. CommTensor 的类型系统价值（Axion 构建阶段）

在 Axion 构建阶段，CommTensor 的价值不仅是 zero-copy 性能，更是**编译期布局保证**：

```python
# Axion IR 中的类型
class CommTensor(Tensor):
    layout: CommLayout  # ORIGINAL_ORDER | SORTED_BY_DST | DISPATCHED

# 编译期检查：
#   rccl_alltoall 只接受 SORTED_BY_DST 布局的 CommTensor
#   Expert FFN 只接受 DISPATCHED 布局的 CommTensor
#
# 如果布局不对，编译时报错，而不是运行时静默产生错误结果

def alltoall_dispatch(x: CommTensor[SORTED_BY_DST]) -> CommTensor[DISPATCHED]:
    ...

def expert_ffn(x: CommTensor[DISPATCHED]) -> CommTensor[DISPATCHED]:
    ...
```

这可以消除一类隐性 bug：在手工实现中，"忘记 pack"或"pack 顺序错误"会导致数值错误，很难调试。

---

## 4. 实验设计细节

### 实验 A：隔离测量 pack/unpack 开销

```bash
# 使用 hipperf 测量 sort_by_dst_rank 的实际耗时
hipperf --metrics HBM_BW,EXEC_TIME \
    python measure_pack_overhead.py \
    --seq_len 4096 --hidden 7168 --dtype bf16 --num_experts 64
```

期望数据格式：
```
seq_len | hidden | dtype | pack_ms | a2a_ms | pack_fraction
  1024  |  7168  | bf16  |  0.008  |  8.5   |    0.09%
  2048  |  7168  | bf16  |  0.016  |  9.2   |    0.17%
  4096  |  7168  | bf16  |  0.022  | 10.1   |    0.22%
  8192  |  7168  | bf16  |  0.044  | 11.8   |    0.37%
```

如果 `pack_fraction < 2%`：停止 Feature 4，记录结论。

---

## 5. 已知局限

1. **真正的 zero-copy**（不分配 sorted_buffer）需要修改上游 attention 的输出逻辑，侵入性太高
2. 当前方案仍然有一次写拷贝（index_copy），只是写入 pattern 更友好（coalesced write）
3. combine 端的 unpack 无法完全消除，只能优化为顺序写
