# Feature 4：CommTensor zero-copy

> **目标:** 消除 Expert dispatch/combine 中的 pack/unpack 内存拷贝，降低 A2A 端到端时间  
> **状态:** 待开始（依赖 Feature 0 数据）  
> **前置:** Feature 0 CommReport 显示 dispatch pack/unpack 开销 ≥ 5% of A2A 时间

---

## 目录结构

```
feature4/
├── README.md                   ← 本文件
├── design/
│   └── Design.md               ← 详细设计：zero-copy 原理、index_map 方案、MI300X 特性
├── code/
│   ├── zero_copy_dispatch.py   ← zero-copy dispatch/combine 实现
│   └── patch_megatron.py       ← 替换 Megatron TokenDispatcher 的 patch
└── experiments/
    ├── exp_A_pack_overhead/    ← 实验 A：pack/unpack 单独开销（hipperf 测量）
    ├── exp_B_e2e_a2a/          ← 实验 B：zero-copy vs 原始 A2A 端到端时间
    ├── exp_C_throughput/       ← 实验 C：端到端 step time 提升
    ├── exp_D_correctness/      ← 实验 D：正确性验证（必做）
    └── results/
```

---

## 一句话描述

**直接分配按目标 rank 分组的内存 buffer**（物理上已是 A2A 所需布局），让 RCCL 做 direct DMA，消除 dispatch 前的 `sort_by_dst_rank` 拷贝和 combine 后的 `unpack` 拷贝。

---

## 核心思路

```
传统方式：
  hidden [seq, hidden]                       ← 按 token 顺序排列
    → pack（sort_by_dst_rank）               ← 内存拷贝 1
    → sorted_hidden [seq, hidden]            ← 按目标 rank 分组
    → rccl_a2a(sorted_hidden)
    → dispatched [recv_tokens, hidden]
    → Expert FFN
    → rccl_a2a(expert_output)
    → combined_sorted [seq, hidden]
    → unpack（index_select 恢复顺序）         ← 内存拷贝 2
    → combined [seq, hidden]

zero-copy 方式：
  在分配 hidden_states 时，直接分配按目标 rank 分组的 buffer
    → 写入时已经是 sorted 布局（需要改上游写入逻辑）
    → rccl_a2a 直接 DMA，无需 pack 拷贝
    → combine 后用 index_map 恢复（仍需一次 index_select，但高效）

MI300X 优势：HBM3 带宽 5.3 TB/s，pack 内存拷贝代价比 H100（3.35 TB/s）高
```

---

## 实验计划

| 实验 | 设置 | 指标 | 红线 |
|------|------|------|------|
| A：pack 开销（**决策实验**） | hipperf 隔离测量 | pack 耗时 / A2A 总时间（%） | < 5% → Feature 4 价值有限，可能停止 |
| B：zero-copy A2A 时间 | 相同 routing，两版本对比 | A2A 端到端时间（ms） | — |
| C：端到端吞吐 | 64 MI300X | tok/s 提升 % | — |
| D：正确性（**必做**） | 相同输入对比输出 | max abs diff | 不一致 → 停止 |

---

## 决策门

```
if 实验 A pack 开销 < 5% of A2A 时间：
  → pack/unpack 不是显著瓶颈（A2A 带宽限制更主要）
  → Feature 4 的绝对收益 < 2% 端到端
  → 停止，记录结论：
    "CommTensor zero-copy 在 MI300X 上的价值主要是设计严谨性，
     而非当前规模下的性能收益"

if 实验 C 端到端提升 ≥ 3%：
  → 有增量价值，继续完善
  → 注意：实现复杂度高，需评估 ROI
```

---

## 关于 CommTensor 的更大价值

```
zero-copy 的性能价值在 MI300X 上可能有限（等实验 A 数据验证）。
但 CommTensor 设计最大的价值是：

  编译期类型系统保证 layout 正确性：
    CommTensor<SORTED_BY_DST>  ← 编译期确保 buffer 已是 dispatch 所需布局
    CommTensor<ORIGINAL_ORDER> ← 编译期确保 buffer 是原始顺序

  这可以消除一类隐性 bug：
    "dispatch 之前忘记 pack" 或 "combine 之后忘记 unpack"

  这个设计价值体现在 Axion 构建阶段，而不是当前的 Primus/Megatron patch 阶段。
```

---

## 验收标准

- [ ] 实验 A：量化 pack/unpack 开销占比（提供数据，不论结论）
- [ ] 实验 D：正确性通过
- [ ] 基于实验 A 数据，给出明确的"继续 / 停止"决策理由

---

## 时间估计

```
实现：2~3 周 | 实验：2 周 | 总计 4~5 周（业余 8~10 周）
```

---

## 相关链接

- 详细设计：[design/Design.md](design/Design.md)
- 核心实现：[code/zero_copy_dispatch.py](code/zero_copy_dispatch.py)
- 上层路线图：[../Roadmap_Unified.md](../Roadmap_Unified.md)
- 前置依赖：[../feature0/README.md](../feature0/README.md)
