# Feature 3：静态 Overlap 调度（OverlapScheduler）

> **目标:** 将 A2A dispatch/combine 与 Expert FFN 计算流水线化，消除 GPU 等待通信的空泡  
> **状态:** 待开始（依赖 Feature 0 数据）  
> **前置:** Feature 0 CommReport 显示 `global_a2a_fraction ≥ 20%` 且实际 overlap 率 < 理论上界 × 80%

---

## 目录结构

```
feature3/
├── README.md                   ← 本文件
├── design/
│   └── Design.md               ← 详细设计：chunk 流水线、依赖分析、与 FlowMoE 对比
├── code/
│   ├── overlap_scheduler.py    ← OverlapScheduler 核心实现
│   └── patch_megatron.py       ← 替换 Megatron MoE dispatch/combine 的 patch
└── experiments/
    ├── exp_A_overlap_rate/     ← 实验 A：overlap 率对比（启用前后）
    ├── exp_B_chunk_sweep/      ← 实验 B：num_chunks 超参扫描
    ├── exp_C_combined/         ← 实验 C：与 Feature 1+2 叠加
    ├── exp_D_correctness/      ← 实验 D：数值正确性验证（必做）
    └── results/
```

---

## 一句话描述

将 MoE 的 token 序列切分为 N 个 chunk，以流水线方式交错执行：**chunk i 的 A2A** 与 **chunk i-1 的 Expert FFN** 并行，最大化 RCCL 通信与 GPU 计算的重叠。

---

## 核心设计

```
传统串行执行：
  [A2A dispatch] ──── 等待 ────→ [Expert FFN] ──── 等待 ────→ [A2A combine]

OverlapScheduler（4 chunk 流水线）：
  Chunk0: [A2A_D0] ────────────────────────────────────────→
  Chunk1:           [A2A_D1]  [FFN_0] ──────────────────→
  Chunk2:                     [A2A_D2]  [FFN_1] ────────→
  Chunk3:                               [A2A_D3]  [FFN_2]→ [FFN_3]
                                                              [A2A_C3]...
  理论加速比：接近 (A2A + FFN) / max(A2A, FFN)
```

关键约束：
- chunk 切分不能破坏 Expert FFN 的计算语义（每个 Expert 独立处理自己的 token）
- A2A 在 `comm_stream` 上执行，FFN 在 `compute_stream` 上执行
- 两个 stream 之间用 `torch.cuda.Event` 同步

---

## 实验计划

| 实验 | 设置 | 指标 | 红线 |
|------|------|------|------|
| A：overlap 率 | CommProfiler 前后各 100 steps | 实际 overlap 率（%） | — |
| B：chunk 数超参 | num_chunks = 1/2/4/8 | step time vs overlap 率 | — |
| C：叠加 Feature 1+2 | 4 组对比 | 累计吞吐提升 % | — |
| D：正确性（**必做红线**） | 相同输入对比输出 | max abs diff < 1e-5 | 任何数值不一致 → 立即停止 |

---

## 决策门

```
if 实验 A overlap 率提升 < 5%（绝对值）：
  → 当前系统已有较好 overlap
  → Feature 3 收益有限，评估 Feature 4

if 实验 C 三个 Feature 叠加总提升 ≥ 20%：
  → 技术路线验证充分
  → 考虑开始 Axion 构建阶段

if 实验 D 数值不一致：
  → 立即停止，debug chunk 边界条件
```

---

## 验收标准

- [ ] 实验 D：数值正确性通过（max diff < 1e-5）
- [ ] 实验 A：overlap 率提升 ≥ 10%（绝对值）
- [ ] 实验 C：吞吐提升 ≥ 5%

---

## 时间估计

```
实现：2 周 | 实验：2~3 周 | 总计 4~5 周（业余 8~10 周）
```

---

## 参考

- FlowMoE（arXiv:2510.00207）：动态 chunk 调度（本方案是静态版本）
- Comet（MLSys '25）：更激进的 Tile 级 overlap（本方案是 chunk 级，实现更简单）

---

## 相关链接

- 详细设计：[design/Design.md](design/Design.md)
- 核心实现：[code/overlap_scheduler.py](code/overlap_scheduler.py)
- 上层路线图：[../Roadmap_Unified.md](../Roadmap_Unified.md)
- 前置依赖：[../feature0/README.md](../feature0/README.md)
