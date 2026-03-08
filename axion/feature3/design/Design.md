# Feature 3 详细设计：OverlapScheduler

> **版本:** v0.1 | 2026-03-08  
> **参考:** FlowMoE (arXiv:2510.00207), Comet (MLSys '25)

---

## 1. 问题定义

MoE 的标准执行顺序是串行的：

```
[A2A dispatch] → [Expert FFN] → [A2A combine]
  (10~20ms)        (5~15ms)       (10~20ms)

总时间 = A2A_dispatch + FFN + A2A_combine
```

两次 A2A 期间，GPU 计算单元完全空闲（等待网络）。实际上：
- `Expert FFN` 不依赖 `A2A dispatch` 的**全部结果**，只依赖**已到达的部分**
- 如果将 token 切成 N 个 chunk，chunk i 的 A2A 完成后就可以立即开始 chunk i 的 FFN
- 同时 chunk i+1 的 A2A 在网络上并行传输

**OverlapScheduler** 将串行改为流水线，理论上可消除全部等待开销。

---

## 2. 流水线设计

### 2.1 执行模型（4 chunks）

```
时间轴 →

Stream: COMM  [A2A_D_chunk0]  [A2A_D_chunk1]  [A2A_D_chunk2]  [A2A_D_chunk3]
                     ↓              ↓               ↓               ↓
Stream: COMP               [FFN_chunk0]   [FFN_chunk1]   [FFN_chunk2]   [FFN_chunk3]

理想情况（FFN ≈ A2A 时）：
  总时间 ≈ A2A_chunk + N × FFN_chunk
         ≈ A2A/N + N × FFN/N
         = A2A/N + FFN

对比串行：
  总时间 ≈ A2A + FFN

加速比 ≈ (A2A + FFN) / (A2A/N + FFN)
```

### 2.2 chunk 数量选择

```
太少（N=1）：退化为串行，无 overlap
太多（N=8）：每个 chunk 的 token 数太少，Expert GEMM 效率下降（矩阵太小）

最优 N 的启发式规则：
  N* = round(A2A_ms / FFN_chunk_ms)   ← 使 A2A 和 FFN 时间相近
  通常 N* = 2~4（MI300X 上实测）
```

---

## 3. 实现方案

### 3.1 CUDA Stream 管理

```python
class OverlapScheduler:
    def __init__(self, num_chunks: int = 4):
        self.num_chunks = num_chunks
        self.comm_stream    = torch.cuda.Stream()   # RCCL A2A 在此 stream 上
        self.compute_stream = torch.cuda.Stream()   # Expert FFN 在此 stream 上
        # 用于跨 stream 同步的 Event
        self._events = [torch.cuda.Event() for _ in range(num_chunks + 1)]
```

### 3.2 Dispatch with Overlap

```python
def dispatch_with_overlap(self, tokens, routing_table, expert_fn):
    """
    tokens: [total_tokens, hidden]
    routing_table: 包含每个 chunk 的发送/接收计数
    expert_fn: callable，输入 dispatched_chunk，输出 expert_output_chunk
    """
    token_chunks = tokens.chunk(self.num_chunks, dim=0)
    dispatched_chunks = [None] * self.num_chunks
    results = []
    pending_ffn_idx = None

    for i, chunk in enumerate(token_chunks):
        # 1. 在 comm_stream 上启动 chunk i 的 A2A dispatch（非阻塞）
        with torch.cuda.stream(self.comm_stream):
            dispatched_chunks[i] = rccl_a2a_async(chunk, routing_table.chunk(i))
            self._events[i].record(self.comm_stream)  # 标记 A2A_i 完成

        # 2. 在 compute_stream 上执行 chunk i-1 的 Expert FFN
        if i > 0:
            with torch.cuda.stream(self.compute_stream):
                # 等待 A2A_{i-1} 完成
                self.compute_stream.wait_event(self._events[i - 1])
                results.append(expert_fn(dispatched_chunks[i - 1]))

    # 3. 处理最后一个 chunk 的 FFN
    with torch.cuda.stream(self.compute_stream):
        self.compute_stream.wait_event(self._events[self.num_chunks - 1])
        results.append(expert_fn(dispatched_chunks[-1]))

    # 4. 等待所有 compute 完成，合并结果
    torch.cuda.current_stream().wait_stream(self.compute_stream)
    return torch.cat(results, dim=0)
```

---

## 4. 与 FlowMoE 的对比

| 维度 | FlowMoE | OverlapScheduler（本方案） |
|------|---------|--------------------------|
| 调度粒度 | Tensor chunk（MB 级） | Tensor chunk（MB 级，相同） |
| 调度时机 | 动态（运行时根据实际 A2A 延迟调整） | 静态（编译前固定 num_chunks） |
| 实现复杂度 | 高（需要动态调度器） | 低（固定流水线，容易验证正确性） |
| overhead | 中（动态调度有开销） | 低（固定切分，overhead 可预测） |
| 适用场景 | 网络延迟不稳定（跨 POD） | 网络延迟稳定（节点内或固定拓扑） |

**本方案的选择理由**：MI300X 节点内 A2A 延迟非常稳定，静态调度足够；动态调度的复杂性不值得。

---

## 5. 正确性保证

chunk 切分必须满足：

1. **Expert 独立性**：每个 Expert 只处理路由到它的 token，chunk 切分不能让不同 Expert 的 token 混在一个 chunk 里
   - 实际上 chunk 是按 **token 维度** 切分，不是按 Expert 切分
   - 路由表（routing_table）也对应切分，保证每个 chunk 的 A2A 只发送该 chunk 的 token

2. **顺序恢复**：combine 之后需要将 token 恢复到原始顺序
   - 用 routing_table 中的 index_map 做 `combined[index_map] = combined_sorted`
   - chunk 合并后再统一恢复顺序

3. **数值等价性**：Expert FFN 是按 token 独立计算的，chunk 切分不影响结果
   - 验证：`assert torch.allclose(overlap_output, serial_output, atol=1e-5)`
