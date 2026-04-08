# PP / EP / VPP Layer View

本文给出一个统一视图，帮助理解在多节点训练中 `PP`、`EP`、`VPP` 如何共同决定 layer 与 rank 的映射关系。

---

## 1. 概念速览

- `PP (Pipeline Parallel)`：按层切分模型，不同 stage 放在不同 rank 组上。
- `EP (Expert Parallel)`：仅在 MoE 层中切分专家，token 在 EP group 内 dispatch/combine。
- `VPP (Virtual Pipeline Parallel)`：把一个 PP stage 再切成多个 virtual chunk，用于降低 bubble、提高流水线利用率。

---

## 2. 例子设定（与你当前讨论一致）

- 集群：`8 nodes`
- 每个节点：`8 GPUs`
- 总 rank 数：`64`
- 并行配置：`PP=8, EP=8, VPP=2`
- 假设：无 TP，且 `EP group` 尽量 node-local（单节点内 8 卡）
- 模型层数：`64 layers`（示意）

---

## 3. PP 视图：layer 到 node 的分布

在 `PP=8` 且各 stage 均匀时，每个 stage 负责 `64 / 8 = 8` 层：

```text
Node0  (PP0): Layer  0 -  7
Node1  (PP1): Layer  8 - 15
Node2  (PP2): Layer 16 - 23
Node3  (PP3): Layer 24 - 31
Node4  (PP4): Layer 32 - 39
Node5  (PP5): Layer 40 - 47
Node6  (PP6): Layer 48 - 55
Node7  (PP7): Layer 56 - 63
```

全局流水方向（前向）：

```text
PP0 -> PP1 -> PP2 -> PP3 -> PP4 -> PP5 -> PP6 -> PP7
```

---

## 4. EP 视图：每个 PP stage 内的专家分布

由于 `EP=8` 且每节点 8 卡，最自然映射是每个 node 内组成一个 EP group：

```text
Node3 / PP3 (示例)
GPU0 GPU1 GPU2 GPU3 GPU4 GPU5 GPU6 GPU7
r24  r25  r26  r27  r28  r29  r30  r31

EP group = {r24..r31}
Experts e0..e7 分散在这 8 个 rank 上
MoE dispatch/combine 在该 EP group 内完成
```

这意味着：

- MoE 主通信（All-to-All）尽量留在 node 内高带宽域。
- 跨 node 主要是 PP 边界激活传递，而不是 MoE 的大规模 dispatch。

---

## 5. VPP 视图：一个 PP stage 内的虚拟层块

`VPP=2` 表示每个物理 PP stage 被拆成两个 virtual chunks。  
如果该 stage 有 8 层，常见切法如下：

```text
PPk stage:
  V0: 4 layers
  V1: 4 layers
```

以 `Node3 / PP3 (Layer 24-31)` 为例：

```text
Node3 / PP3
  V0: Layer 24,25,26,27
  V1: Layer 28,29,30,31
```

流水线调度时，micro-batch 会在 `V0/V1` 间交错执行，从而降低 pipeline bubble。

---

## 6. 组合视图（PP + EP + VPP）

### 6.1 Node-rank 网格

```text
            GPU0 GPU1 GPU2 GPU3 GPU4 GPU5 GPU6 GPU7
Node0 PP0    r0   r1   r2   r3   r4   r5   r6   r7
Node1 PP1    r8   r9   r10  r11  r12  r13  r14  r15
Node2 PP2    r16  r17  r18  r19  r20  r21  r22  r23
Node3 PP3    r24  r25  r26  r27  r28  r29  r30  r31
Node4 PP4    r32  r33  r34  r35  r36  r37  r38  r39
Node5 PP5    r40  r41  r42  r43  r44  r45  r46  r47
Node6 PP6    r48  r49  r50  r51  r52  r53  r54  r55
Node7 PP7    r56  r57  r58  r59  r60  r61  r62  r63
```

### 6.2 三层映射关系

```text
Layer -> PP stage (跨节点)
MoE experts in that layer -> EP group (节点内 8 卡)
PP stage 内层块 -> VPP virtual chunks (V0/V1)
```

---

## 7. 一个 token 在 MoE 层的路径（抽象）

```text
Token enters PPk
  -> Router (in PPk)
  -> Dispatch to EP8 group (node-local)
  -> Expert compute
  -> Combine back
  -> Continue to next PP stage (可能跨 node)
```

核心边界：

- `EP 通信边界`：尽量 node-local
- `PP 通信边界`：stage 间激活传递（可能跨 node）
- `VPP 作用`：改变 stage 内执行时序，不改变全局拓扑边界

---

## 8. 实践备注

- 当某 stage 显存压力更高时，可采用非对称 VPP（不同 stage 不同层数切分）。
- 当单层 MoE 过大时，仅增大 PP 可能无效，需提高 EP/ETP/FSDP 分片能力。
- 对 FP8/FP4 场景，建议优先保证 EP group 内布局稳定，避免频繁跨 node dispatch。

