# Feature 1 详细设计：FastRouter

> **版本:** v0.1 | 2026-03-08

---

## 1. 问题定义

MoE 训练中，`TopKRouter` 按 gate logit 最高的 top-K expert 路由 token。在没有负载约束的情况下，高频出现的 token pattern 会持续命中同一批 Expert，导致这些 Expert 所在 GPU 过载，其余 GPU 空闲等待（木桶效应）。

**FastRouter 的目标**：在不改变 Expert 物理位置的情况下，通过轻微调整 gate logits，使得过载 Expert 的被选概率降低，将部分 token 分流到相近质量的 Expert。

---

## 2. 算法设计

### 2.1 核心公式

```
gate_logits_adjusted = gate_logits - α × log((load_ema / load_ema.mean())^β)
```

等价于在 softmax 空间中：

```
p(expert_i) ∝ exp(logit_i) / (load_ema_i / mean_load)^(α×β)
```

直觉：负载越高的 Expert，其被选概率被压低越多。

### 2.2 超参分析

| 超参 | 含义 | 推荐范围 | 影响 |
|------|------|----------|------|
| `alpha` | 惩罚强度 | 0.05 ~ 0.2 | 越大均衡越激进，收敛风险越高 |
| `beta` | 非线性系数 | 1.0 ~ 3.0 | 越大对重载 Expert 惩罚越集中 |
| `ema_decay` | 平滑系数 | 0.8 ~ 0.99 | 越小负载统计响应越快，但噪声更大 |

### 2.3 EMA 更新时机

负载统计 `load_ema` 在每个 step 的 dispatch 完成后更新：
```python
# TokenDispatcher.dispatch() 返回时，可以读取 input_splits（每个 expert 收到的 token 数）
load_ema = ema_decay * load_ema + (1 - ema_decay) * current_expert_counts
```

**不在 router 的 forward 内部更新**（避免影响梯度计算），而是作为外部状态由训练循环驱动更新。

---

## 3. 接入 Megatron-LM 的方式

### 3.1 最小改动方案（推荐）

不修改 megatron-core 代码，只 monkey-patch `TopKRouter.forward`：

```python
# patch_megatron.py 的核心逻辑
orig_forward = TopKRouter.forward

def patched_forward(self, hidden_states):
    # 1. 计算原始 gate logits（调用原始 forward 的前半段）
    logits = self.linear_fc1(hidden_states)  # 或直接读取 self.gate_logits

    # 2. 插入负载惩罚
    if fast_router.load_ema is not None:
        logits = fast_router.adjust(logits)

    # 3. 执行原始的 softmax + topk（调用原始 forward 的后半段）
    ...

TopKRouter.forward = patched_forward
```

### 3.2 更干净的方案（需改 megatron-core）

在 `TopKRouter.forward()` 中，在 softmax 之前预留 hook 点：

```python
# megatron/core/transformer/moe/router.py（改动 1 行）
class TopKRouter(nn.Module):
    def forward(self, hidden_states):
        logits = self._compute_logits(hidden_states)
        logits = self._apply_load_penalty(logits)  # ← 新增，默认 identity
        scores = logits.softmax(-1)
        ...

    def _apply_load_penalty(self, logits):
        """由外部 FastRouter 覆写此方法"""
        return logits  # 默认不做任何修改
```

FastRouter 接入时只需：
```python
router._apply_load_penalty = fast_router.adjust
```

---

## 4. 分布式场景下的负载统计

在 Expert Parallelism（EP）场景下：
- 每个 rank 只管理部分 Expert
- `load_ema` 需要在 EP group 内做 `all_reduce`，才能得到全局负载视图
- FastRouter 的惩罚应基于全局视图，而不是本地 rank 的局部负载

```python
# 在 update() 中，EP group 内 all_reduce
if ep_group is not None:
    dist.all_reduce(current_counts, op=dist.ReduceOp.SUM, group=ep_group)
self.load_ema = ema_decay * self.load_ema + (1 - ema_decay) * current_counts
```

---

## 5. 收敛分析

FastRouter 对模型质量的影响来源：
1. **引入路由偏差**：部分 token 被引导到"次优" Expert，可能降低模型表达能力
2. **动态惩罚噪声**：EMA 统计有延迟，惩罚信号有噪声，可能引入训练不稳定

**缓解措施**：
- `alpha` 保持小值（0.05~0.1），惩罚幅度不超过 gate logit 方差的 20%
- `beta` 不超过 2.0，避免极端情况下某个 Expert 被完全屏蔽
- 实验 A 作为红线：任何超参组合先做收敛验证再上生产

---

## 6. 与 Feature 2（SlowPlanner）的关系

| | FastRouter | SlowPlanner |
|---|---|---|
| 修改对象 | 路由决策（软性偏置） | Expert 物理位置 |
| 响应速度 | 每个 step 实时调整 | 每 50 step 迁移一次 |
| 收敛风险 | 中等（改变路由语义） | 低（不改语义，只改物理位置） |
| 效果上界 | 受限于 Expert 质量差异 | 理论上可消除全部不均衡 |
| 叠加性 | 可叠加 | 可叠加 |

推荐顺序：先做 FastRouter（实现简单），再叠加 SlowPlanner（效果更好）。
