# Feature 0：通信可观测性（CommProfiler）

> **目标:** 在 MI300X 上跑 Megatron-LM MoE，量化通信瓶颈，为 Feature 1~4 提供优先级依据  
> **状态:** 设计阶段  
> **前置:** 无（这是所有 Feature 的起点）

---

## 目录结构

```
feature0/
├── README.md                   ← 本文件：概述、决策门、进度
├── design/
│   └── Design.md               ← 详细设计文档（Hook 接入点分析、数据结构、关键决策）
├── code/
│   ├── comm_profiler.py        ← AxionCommProfiler 核心实现
│   ├── comm_report.py          ← CommReport 数据结构 + HTML 可视化
│   └── example_megatron.py    ← 接入 Megatron pretrain 的使用示例
└── experiments/
    └── results/                ← 实测数据（CommReport JSON + HTML）放这里
```

---

## 一句话描述

**零侵入挂载在 Megatron-LM MoE 层上的通信 Profiler**，采集 Expert 负载分布和 A2A 时间，生成可视化报告，作为所有后续优化 Feature 的数据依据。

---

## 收集的核心指标

| 指标 | 含义 | 用途 |
|------|------|------|
| `global_load_imbalance` | Expert 负载 max/mean，逐层 | 决定 Feature 1/2 优先级 |
| `global_a2a_fraction` | A2A 时间 / MoE 层总时间 | 决定 Feature 3/4 优先级 |
| `dispatch_gbps` / `combine_gbps` | 实测 A2A 带宽 | 判断跨节点流量占比 |
| `hot_experts` | 负载 top-5 的 expert id（逐层）| Feature 2 迁移的目标 |

---

## 决策门

```
Feature 0 完成后，根据 CommReport 数据：

  if global_load_imbalance < 1.3x：
    → Feature 1/2 优先级降低，优先看 A2A 时间占比

  if global_load_imbalance ≥ 2.0x：
    → Feature 1（FastRouter）高优，立即开始

  if global_a2a_fraction < 10%：
    → A2A 不是瓶颈，重新评估路线

  if global_a2a_fraction ≥ 20%：
    → Feature 3（OverlapScheduler）和 Feature 4（CommTensor）高优

  if dispatch_gbps << MI300X 节点内理论带宽（~1.6 TB/s）：
    → 大量跨节点流量，Feature 2（Expert 物理迁移）有价值
```

---

## 验收标准

- [ ] CommReport 数据与 rocprof 手动 profiling 误差 < 10%
- [ ] 接入开销 < 1%（profiling 期间不超过 step time 的 1%）
- [ ] 支持多卡分布式场景（EP 并行）
- [ ] 生成可读的 HTML 可视化报告

---

## 时间估计

```
Week 1：熟悉 megatron-core MoE 代码 + 实现 hook + 单卡验证
Week 2：CommReport 聚合 + 多卡验证 + 对比 rocprof
Week 3：HTML 可视化 + 文档 + 内部报告初稿
```

---

## 相关链接

- 详细设计：[design/Design.md](design/Design.md)
- 核心实现：[code/comm_profiler.py](code/comm_profiler.py)
- 报告生成：[code/comm_report.py](code/comm_report.py)
- 使用示例：[code/example_megatron.py](code/example_megatron.py)
- 上层路线图：[../Roadmap_Unified.md](../Roadmap_Unified.md)
