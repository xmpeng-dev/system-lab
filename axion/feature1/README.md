# Feature 1：路由负载均衡（FastRouter）

> **目标:** 通过调整 gate logits，软性引导 token 远离过载 Expert，减轻负载不均衡  
> **状态:** 待开始（依赖 Feature 0 数据）  
> **前置:** Feature 0 CommReport 显示 `global_load_imbalance ≥ 1.5x`

---

## 目录结构

```
feature1/
├── README.md                   ← 本文件
├── design/
│   └── Design.md               ← 详细设计：FastRouter 算法、超参分析
├── code/
│   ├── fast_router.py          ← FastRouter 核心实现
│   └── patch_megatron.py       ← 接入 Megatron TopKRouter 的 patch
└── experiments/
    ├── exp_A_convergence/      ← 实验 A：收敛性验证（红线）
    ├── exp_B_load_balance/     ← 实验 B：负载均衡效果
    ├── exp_C_throughput/       ← 实验 C：吞吐测量
    └── results/                ← 实测数据
```

---

## 一句话描述

在 Megatron-LM 的 `TopKRouter.forward()` 中，在 softmax 之前插入一行负载惩罚偏置，**动态压低过载 Expert 的被选概率**，无需改动模型结构和 Expert 参数。

---

## 核心算法

```python
# 在 TopKRouter 的 forward 中，softmax 之前插入：
gate_logits = gate_logits - alpha * (load_ema / load_ema.mean()).pow(beta).log()
# 然后正常执行：
scores = gate_logits.softmax(-1)
topk_scores, topk_indices = scores.topk(k)
```

超参说明：
- `alpha`（默认 0.1）：惩罚强度，越大越激进均衡，但收敛风险越高
- `beta`（默认 2.0）：非线性系数，控制对重载 Expert 的惩罚曲线
- `ema_decay`（默认 0.9）：负载统计的指数移动平均平滑系数

---

## 实验计划

| 实验 | 设置 | 指标 | 红线 |
|------|------|------|------|
| A：收敛性（**必做红线**） | 2B MoE, 1000 steps | loss curve, perplexity | loss 差异 > 1% → 停止 |
| B：负载均衡效果 | CommProfiler 前后各 100 steps | `load_imbalance_ratio` 变化 | — |
| C：吞吐测量 | 64 MI300X | tok/s 提升 % | < 3% → 评估是否继续 |

---

## 决策门

```
if 实验 A 红线触发（loss 差异 > 1%）：
  → 停止 FastRouter
  → 直接跳 Feature 2（SlowPlanner，物理迁移，无收敛风险）

if 实验 C 吞吐提升 < 3%：
  → FastRouter 收益不显著
  → 查看 CommReport：A2A 时间占比是否更高？
  → 如果是 → 跳到 Feature 3（OverlapScheduler）

if 实验 C 吞吐提升 ≥ 5%：
  → 继续 Feature 2（SlowPlanner，物理迁移，叠加收益）
```

---

## 验收标准

- [ ] 实验 A：收敛性通过（loss 差异 < 1%）
- [ ] 实验 B：`load_imbalance_ratio` 下降（方向正确）
- [ ] 实验 C：吞吐提升 ≥ 3%

---

## 时间估计

```
实现：1 周 | 收敛实验：2 周 | 总计 3~4 周（业余 6~8 周）
```

---

## 相关链接

- 详细设计：[design/Design.md](design/Design.md)
- 核心实现：[code/fast_router.py](code/fast_router.py)
- 上层路线图：[../Roadmap_Unified.md](../Roadmap_Unified.md)
- 前置依赖：[../feature0/README.md](../feature0/README.md)
