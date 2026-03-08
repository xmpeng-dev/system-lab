# Feature 2：Expert 物理迁移（SlowPlanner）

> **目标:** 每隔 K 个 step，根据历史路由统计，将过载 Expert 参数迁移到负载较轻的 GPU  
> **状态:** 待开始（依赖 Feature 0 数据）  
> **前置:** Feature 0 CommReport 显示 `global_load_imbalance ≥ 1.5x`

---

## 目录结构

```
feature2/
├── README.md                   ← 本文件
├── design/
│   └── Design.md               ← 详细设计：迁移算法、P2P 通信、收敛分析
├── code/
│   ├── slow_planner.py         ← SlowPlanner 核心实现
│   └── patch_megatron.py       ← 接入 Megatron 训练循环的 patch
└── experiments/
    ├── exp_A_convergence/      ← 实验 A：收敛性（loss spike 验证）
    ├── exp_B_migration_roi/    ← 实验 B：迁移开销 vs 收益
    ├── exp_C_combined/         ← 实验 C：与 Feature 1 叠加效果
    ├── exp_D_scaling/          ← 实验 D：规模扩展效果（8/16/32/64 GPU）
    └── results/
```

---

## 一句话描述

每隔 `check_interval` 步，用贪心算法将**热点 Expert 参数**通过 P2P 异步传输迁移到**冷点 GPU**，从根本上消除 Expert 负载不均衡，无需修改路由语义。

---

## 核心设计

```python
# 训练循环中只加一行：
for step in range(num_steps):
    loss = forward_step(...)
    loss.backward()
    optimizer.step()
    planner.maybe_migrate(step, model)  # ← 只加这一行
```

迁移策略：
- **触发条件**：`load_imbalance_ratio ≥ imbalance_threshold`（默认 1.3）
- **规划算法**：贪心 — 将负载最高 GPU 的 Expert 迁移到负载最低 GPU
- **执行方式**：`dist.isend/irecv` 异步 P2P，与下一个 step 重叠
- **MI300X 优化**：优先节点内迁移（走 XGMI/Infinity Fabric，高带宽低延迟）

---

## 实验计划

| 实验 | 设置 | 指标 | 红线 |
|------|------|------|------|
| A：收敛性（**必做**） | 7B MoE, 2000 steps | loss spike 幅度 | spike > 5% moving avg → 收紧触发条件 |
| B：迁移 ROI | 单次迁移测量 | 迁移耗时 vs 节省计算时间 | ROI < 2x → 降低迁移频率 |
| C：与 Feature 1 叠加 | 4 组对比 | 吞吐、imbalance | — |
| D：规模扩展 | 8/16/32/64 MI300X | 提升比例随规模变化 | — |

---

## 决策门

```
if 实验 A 出现持续 loss spike：
  → 提高 imbalance_threshold 或减少单次迁移数量
  → 仍无法解决 → 停止 SlowPlanner，转向 Feature 3

if 实验 B ROI < 2x：
  → 减少迁移频率（check_interval 50 → 100）
  → 或限制到节点内迁移

if 实验 C/D 吞吐提升 ≥ 10%：
  → Feature 1+2 组合有明确收益，继续 Feature 3
```

---

## 验收标准

- [ ] 实验 A：无持续 loss spike（单次 spike < 5% moving avg）
- [ ] 实验 B：迁移 ROI ≥ 2x（节省时间 / 迁移通信时间）
- [ ] 实验 D：提升比例随 GPU 数量扩大（说明问题在大规模下更严重，方案有价值）

---

## 时间估计

```
实现：2~3 周 | 实验：3 周 | 总计 5~6 週（业余 10~12 周）
```

---

## 参考

- LAER-MoE 论文（Hetu-Galvatron 实现）
- MI300X XGMI/Infinity Fabric P2P 带宽文档

---

## 相关链接

- 详细设计：[design/Design.md](design/Design.md)
- 核心实现：[code/slow_planner.py](code/slow_planner.py)
- 上层路线图：[../Roadmap_Unified.md](../Roadmap_Unified.md)
- 前置依赖：[../feature0/README.md](../feature0/README.md)
