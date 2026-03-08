# Feature 0 详细设计：CommProfiler

> **版本:** v0.1 | 2026-03-08  
> **依赖:** megatron-core (NVIDIA/Megatron-LM), PyTorch, plotly

---

## 1. Megatron-LM MoE 代码结构（Hook 接入点分析）

```
megatron-core 的 MoE 调用栈（关键路径）：

megatron/core/transformer/transformer_layer.py
  └─ TransformerLayer.forward()
       └─ self.mlp(hidden_states)           ← MoE 层入口

megatron/core/transformer/moe/moe_layer.py
  └─ MoELayer.forward(hidden_states)
       ├─ [1] self.router(hidden_states)    ← TopKRouter，计算 routing weights + indices
       └─ [2] self.experts(hidden_states, ...) ← dispatch → expert compute → combine

megatron/core/transformer/moe/experts.py
  └─ GroupedMLP.forward() / SequentialMLP.forward()
       ← 本地 Expert 的实际计算

megatron/core/transformer/moe/token_dispatcher.py
  └─ MoEAlltoAllTokenDispatcher
       ├─ [3] dispatch(hidden_states, max_prob, top_indices)
       │       ├─ preprocess(indices)         ← 计算每个 expert 的 token 数（负载统计点）
       │       ├─ alltoall(hidden_states)     ← A2A dispatch（计时点 1）
       │       └─ 返回 dispatched_input
       └─ [4] combine(expert_output, ...)
               ├─ alltoall(expert_output)     ← A2A gather（计时点 2）
               └─ 返回 combined_output

megatron/core/transformer/moe/router.py
  └─ TopKRouter.forward()
       └─ [5] routing_map, probs             ← 路由结果（负载分析的原始数据）
```

### Hook 接入策略汇总

| 目标模块 | 接入方式 | 收集数据 |
|---|---|---|
| `MoELayer` | `register_forward_pre/post_hook` | MoE 层总耗时 (`moe_layer_ms`) |
| `TopKRouter` | `register_forward_hook` | `routing_map` → `tokens_per_expert` |
| `MoEAlltoAllTokenDispatcher` | **monkey-patch** `dispatch/combine` | A2A 耗时 + 通信量（字节） |

---

## 2. 数据流设计

```
训练循环
  │
  ├─ step N（warmup，不采样）
  ├─ step N+1（warmup，不采样）
  │   ...
  ├─ step N+5（开始采样）
  │   │
  │   ├─ MoELayer pre_hook  → 记录 t0
  │   ├─ TopKRouter post_hook → 记录 tokens_per_expert[layer_idx]
  │   ├─ dispatcher.dispatch → 记录 dispatch_a2a_ms, dispatch_bytes
  │   ├─ (Expert FFN 计算)
  │   ├─ dispatcher.combine  → 记录 combine_a2a_ms, combine_bytes
  │   └─ MoELayer post_hook  → 记录 moe_layer_ms = now - t0
  │
  │   → StepStats(layer_idx, step_idx, ...) 追加到 _stats 列表
  │
  └─ step N+25（采样结束，profiler 自动停止）

report() 调用
  │
  ├─ 按 layer_idx 聚合 _stats
  ├─ 计算每层的均值/最大值/不均衡系数/带宽
  └─ 生成 CommReport → print_summary() + save_html() + save_json()
```

---

## 3. 关键设计决策

### 3.1 为什么 dispatcher 用 monkey-patch？

`MoEAlltoAllTokenDispatcher` 不是 `nn.Module` 的标准调用路径：
- 其 `dispatch()` / `combine()` 是被 `MoELayer` 直接调用的普通方法
- 不走 `nn.Module.__call__()` → `register_forward_hook` 无法捕获
- 解决方案：用 `types.MethodType` monkey-patch 替换方法，是最低侵入的方式

```python
import types
module.dispatch = types.MethodType(patched_dispatch, module)
module.combine  = types.MethodType(patched_combine,  module)
```

### 3.2 为什么需要 `torch.cuda.synchronize()`？

MI300X 上的 RCCL A2A 是**异步启动**的：
- 不 synchronize 直接计时 → 测到 "kernel 提交时间"（几乎为 0）
- 不是 "A2A 完成时间"（实际耗时 5~20ms）
- `synchronize` 引入额外延迟 ~0.1ms，但保证计时语义正确

开销估算：`2 次 sync/A2A × 2 A2A/layer × N layers` ≈ 可接受（< 1% step time）

### 3.3 为什么不用 torch.profiler / rocprof？

| 工具 | 粒度 | 问题 |
|------|------|------|
| `torch.profiler` | kernel 级 | 无法关联到"哪个 MoE 层"的语义；开销大（不适合长期采样） |
| `rocprof` | kernel 级 | 同上；需要重启训练进程；无法实时分析 |
| `AxionCommProfiler` | MoE 语义级 | 直接输出 Expert 负载 + A2A 分层时间；在线采样，无需重启 |

两者互补：AxionCommProfiler 给出高层语义指标，rocprof 用于深入 kernel 级分析。

### 3.4 分布式场景下的数据汇总

`routing_map` 在每个 rank 上只有**本地**的 token（EP 并行下每个 rank 管理部分 Expert）：
- **当前版本（v0.1）**：各 rank 独立统计本地负载，CommReport 反映本地视角
- **后续扩展（v0.2）**：在 `report()` 时做 `dist.all_reduce` 汇总全局视图

```python
# v0.2 扩展点（TODO）
if dist.is_initialized():
    dist.all_reduce(tokens_per_expert, op=dist.ReduceOp.SUM)
    # 此时 tokens_per_expert 反映全局所有 token 的路由分布
```

### 3.5 step 计数策略

- 用**第 0 层 MoELayer 的 post_hook** 驱动 `_step_idx` 递增
- 每个 training step 中，layer 0 的 post_hook 恰好触发一次
- 多层 MoE 模型（如 DSv3-like 的 61 个 MoE 层）中，其他层使用同一 `_step_idx`

---

## 4. CommReport 字段说明

### LayerReport（per-layer）

| 字段 | 类型 | 含义 |
|------|------|------|
| `layer_idx` | int | MoE 层编号（0-indexed） |
| `mean_tokens_per_expert` | float | 平均每个 expert 收到的 token 数 |
| `max_tokens_per_expert` | float | 最忙 expert 收到的 token 数 |
| `load_imbalance_ratio` | float | `max / mean`，理想值 = 1.0 |
| `hot_experts` | List[int] | 负载 top-5 的 expert id |
| `dispatch_a2a_ms_mean` | float | dispatch A2A 平均耗时（ms） |
| `combine_a2a_ms_mean` | float | combine A2A 平均耗时（ms） |
| `a2a_total_ms_mean` | float | dispatch + combine 总 A2A 耗时 |
| `moe_layer_ms_mean` | float | MoE 层总耗时（含 A2A + Expert FFN） |
| `a2a_fraction` | float | `a2a_total / moe_layer`，**核心决策指标** |
| `dispatch_gbps` | float | 实测 dispatch 带宽（GB/s） |
| `combine_gbps` | float | 实测 combine 带宽（GB/s） |

### CommReport（global）

| 字段 | 含义 |
|------|------|
| `global_load_imbalance` | 所有层的平均 `load_imbalance_ratio` |
| `global_a2a_fraction` | 所有层的平均 `a2a_fraction` |

---

## 5. HTML 可视化布局

```
┌─────────────────────────────────────────────────────┐
│  AxionCommProfiler: MoE Communication Analysis      │
├──────────────────────┬──────────────────────────────┤
│ Expert Load          │ A2A Time Fraction            │
│ Imbalance (per layer)│ (per layer)                  │
│ [Bar chart]          │ [Bar chart]                  │
├──────────────────────┼──────────────────────────────┤
│ Dispatch vs Combine  │ Effective A2A Bandwidth      │
│ A2A Time (ms)        │ (GB/s)                       │
│ [Grouped bar]        │ [Line chart]                 │
└──────────────────────┴──────────────────────────────┘
```

---

## 6. 已知限制 & 后续改进

| 限制 | 影响 | 计划改进版本 |
|------|------|------------|
| dispatcher monkey-patch 无法通过 `detach()` 恢复 | profiler 只能 attach 一次 | v0.2：保存原始方法引用，`detach()` 时恢复 |
| 各 rank 独立统计，不汇总全局负载 | 多机 EP 场景下负载视角不完整 | v0.2：`report()` 时 `dist.all_reduce` |
| `synchronize()` 引入 ~0.1ms 开销 | 高频采样时有影响 | v0.2：改用 CUDA event 计时，开销 < 0.01ms |
| 仅支持 `MoEAlltoAllTokenDispatcher` | 不支持 `MoEAllGatherTokenDispatcher` | v0.2：添加 AllGather 变体支持 |
