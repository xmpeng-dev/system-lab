# 对话记录

> 议题：Cairn 的 `MathGraph → compiler → Program → Runtime` 是否适合 coding-agent
> 时代？  
> 开始时间：2026-10-03  
> 状态：历史 transcript，只读；新发言写入 [`turns/`](./turns/README.md)

---

## Round 0：建立共同事实

### GPT-5.6 Sol（主持人 / 综合评审）

我是 GPT-5.6 Sol，当前 Cursor 主会话里的 coding agent。我的职责是检查 Cairn
当前代码，主持不同立场的 agent 讨论，并把未解决问题交还给 Cairn Owner。

当前代码实际执行路径是：

```text
model.build_graph()
  → Graph
  → fusion / MoE lowering / autodiff / recompute
  → micro-batch task expansion / schedule / residency / memory
  → Program<Task>
  → Runtime setup closures
  → step 顺序执行
```

因此：

- `passes` 是 compiler 内部变换，不是独立表示；
- 当前没有独立的 `ExecGraph` class；
- `Program` 仍引用 Graph `Node` 和 live module，kernel binding 在 Runtime setup 时解析；
- 更准确的描述是
  `Graph → compiler/planner → executable program → executor`。

当前 `ir/ + compiler/ + runtime/` 约 6,300 行。一份四层 DeepSeek-V3 artifact 记录：

```text
118 forward nodes
131 dgrad + 34 wgrad + 21 fused backward nodes
3,106 tasks
3,440 slots
```

同一 profile 中：

```text
model_ms_per_step           = 2137.886 ms
stage_remainder_ms_per_step = 6.357 ms
```

这个数字不能单独测出 Python closure dispatch，但不支持“Runtime 解释器已经是主要
瓶颈”的判断。当前争议主要是语义所有权和 agent edit distance。

PithTrain checkout 当前不在 Cairn 仓库中，因此本轮只能依据 Cairn 文档讨论 PithTrain，
不能声称完成了最新源码级对照。

---

## Round 1：独立立场

### Compiler Advocate（general-purpose coding subagent）

我是 Compiler Advocate，一个评审 Cairn 的 general-purpose coding subagent。我的职责
是为当前 compiler spine 寻找最强论据，同时指出证据不支持的宣称。

`Graph → compiler/planner → executable plan → executor` 对 Cairn 是合理的。原因不是
“agent 喜欢 IR”，而是 Cairn 同时改变 MoE decomposition、communication placement、
pipeline order、recompute、FSDP residency 和 kernel binding。

最强实例是 `RoutedExperts`。模型只表达一次 routed expert 数学，compiler 可以选择
dynamic all-to-all、static capacity 或 fused MoE。显式 Program 还能统一验证 slot
lifetime、collective wait、residency、channel order 和跨 rank P2P progress。

但现状不是完整的 searchable implementation system：

- `Implementation` 只包含六类 compile policy；
- precision、topology、batch 和 optimizer 仍来自别处；
- Runtime setup 仍解析 binding 和 distributed state；
- Program 不是完全自包含的 executable artifact；
- search、attribution 和 evidence store 尚未实现。

我建议保留 Graph、planner、Program 和 executor，但把架构宣称收窄为：

```text
Graph
  → semantic lowering
  → execution planning
  → ExecutablePlan
  → Executor
```

### Complexity Skeptic（general-purpose coding subagent）

我是 Complexity Skeptic，一个评审 Cairn 的 general-purpose coding subagent。我的职责
是检查当前设计是否为了未来可能性承担了过早复杂度。

我同意 coarse schedule/communication planning 有价值，但怀疑 primitive MathGraph。
Cairn 同时维护 operator shape/effects、runtime forward/VJP 和独立 PyTorch reference。
这对通信、`RoutedExperts` 和 deferred wgrad 可能必要，但没有证明 `Reshape`、
`Split`、`Concat`、`Expand` 和普通 dense `Linear` 都应成为 Cairn 自有 IR。

我建议保留 schedule、MoE lowering、P2P verifier、residency、Program 和 run record，
但把前端收敛为：

```text
Attention
Router
Dispatch
Experts
Combine
DenseMLP
Loss
```

region body 使用直接 PyTorch、自定义 op 或 `torch.compile(fullgraph=True)`。保留
primitive MathGraph 的证据应是无法由 region boundary 表达的真实端到端收益，而不是
“未来可能搜索”。

### Agent Maintainability Reviewer（general-purpose coding subagent）

我是 Agent Maintainability Reviewer，一个从 coding-agent 可维护性评审 Cairn 的
general-purpose coding subagent。

我推荐混合方案：

```text
RegionGraph → Program → Executor
```

PithTrain 式直接控制流的局部 edit distance 最短，但 schedule 和 decomposition 不是
可寻址数据。当前 MathGraph 的可寻址性最强，但普通操作可能同时修改 builder、op
inference、binding、implementation、VJP、verifier 和 tests。

不能只测代码行数。应测：

- semantic edit sites；
- 完成任务所需文件和上下文；
- first-pass contract success；
- agent retry 次数；
- GPU 分配前捕获的问题；
- 每个决策是否进入 run record 和 digest。

### GPT-5.6 Sol（主持人）

Round 1 共识：

1. 不退回纯手写 `step()`；
2. schedule、communication、residency 和 Program 继续成为数据；
3. 当前没有新增独立 `ExecGraph` 的证据；
4. Runtime dispatch 目前不是已证明的主瓶颈；
5. 真正分歧是 Graph 应到 primitive 还是 semantic region 粒度。

未解决问题：

1. explicit VJP 和 fine-grained effects 能否安全封装在 region contract 中？
2. 哪个近期 transform 必须看到 attention/MLP 内部 tensor algebra？
3. RegionGraph 内部是否会重新长出一套 fine graph？
4. 如何公平 A/B 两种前端而不同时重写 Runtime？

---

## Round 2：相互回应

### Compiler Advocate（general-purpose coding subagent）

Complexity Skeptic，我同意当前证据没有证明 Cairn 必须拥有细粒度 operator IR。已有
MoE lowering 和 scheduling 收益可能由 RegionGraph 表达。

我的最强反对意见是：fine IR 还承担 typed layout propagation、explicit VJP、fusion
rewrite、effect declaration，以及 unsafe recompute/reorder 的编译期拒绝。RegionGraph
必须保持这些保证，同时不能在每个 region 内重建第二套 fine representation。

会改变我判断的实验，是让三种真实变化分别在 fine IR 和 RegionGraph 分支实现：

1. 新 MoE strategy；
2. fusion/VJP 变化；
3. schedule/residency 变化。

比较 E2E correctness/performance、语义修改点、contract 首次通过率、agent 重试次数，
以及 GPU 分配前捕获的问题。如果 RegionGraph 保持保证并稳定缩短 edit distance，我
支持收敛 fine IR。

我问 Agent Maintainability Reviewer：RegionGraph 如何验证 operator effects 和
explicit backward dependencies，同时避免 opaque Python callback 或嵌入 fine graph？

### Complexity Skeptic（general-purpose coding subagent）

Compiler Advocate，我同意 `RoutedExperts → dispatch/experts/combine` lowering 和跨 rank
verification 足以证明 semantic planning layer。

我的剩余异议仍是粒度。这些收益需要显式 effects 和 communication boundary，但没有
证明 Cairn 必须拥有普通 tensor math、gradient accumulation、saved tensor 和全部
VJP。四层模型展开为 3,106 tasks 和 3,440 slots，是明显的语义税。

会改变我判断的实验：fine MathGraph 产生至少两个 RegionGraph 无法表达的 transform，
每个都有可复现的超过 3% E2E 收益，并保持 gradient 和 memory 正确。我不以 runtime
dispatch 已经很慢为论据，当前 profile 不支持它。

我问 Compiler Advocate：接下来哪个明确规划的 transform 需要看到 dense
attention/MLP 的 tensor algebra，而不是 region boundary 和 custom-op metadata？

### Agent Maintainability Reviewer（general-purpose coding subagent）

我的仲裁是：保留 planner、Program、executor 和 verifier，只替换 semantic front
end，避免把多个变量混进实验。

**A — Primitive Graph**

- 使用现有 MathGraph；
- 编译一个代表性 MoE layer；
- 保留 dynamic/static/fused lowering；
- 保留现有 autodiff、schedule、residency 和 Program。

**B — RegionGraph**

- regions 为 `Attention`、`Router`、`RoutedExperts`、`DenseMLP`、`Loss`；
- contract 包含 typed ports、effects、parameters、explicit VJP 和 provenance；
- body 使用直接 PyTorch/owned operations；
- lower 到同一个 Program，使用同一个 executor 和 verifier。

共同 acceptance gates：

1. 两个模型、五种 scheduler、三种 MoE lowering 的 normalized Program 等价；
2. loss 和每个 gradient 满足现有 tolerance；
3. 两边都拒绝 deadlock、channel reorder、effectful recompute、residency misuse 和
   missing gradient reduction；
4. 每个性能决策都改变 `Implementation.describe()` 和 program digest；
5. 盲测 new model composition、new MoE lowering、new fusion/kernel binding；
6. B 若只使用 A 不超过 60% 的 semantic edit sites 且不削弱前四项，则 B 胜；
7. A 若展示 B 无法局部表达的真实 primitive transform，则 A 胜。

我问 Cairn Owner：未来两个 milestone 中，哪个 primitive-level transformation 是
Region implementation 或 Region lowering 无法表达、但产品必须支持的？

### GPT-5.6 Sol（主持人）

对 Compiler Advocate 问题的临时回答：Region contract 只声明跨 region 可观察的
effect 和 backward dependency；region 内通过 `fullgraph=True`、custom op schema、
独立 gradient reference 和 region-local tests 负责。如果一个 effect 会改变跨 region
scheduling，它必须提升为 region boundary；如果这种提升频繁发生，RegionGraph 假设
被证伪。

目前检查到的明确搜索维度集中于 MoE lowering、collective 位置、schedule、
recompute、residency 和 kernel binding。尚未找到一个已排期且必须任意观察 dense
attention/MLP primitive algebra 的 transform。这个缺失使 RegionGraph 值得做原型，
但不证明它已经获胜。

---

## Round 3：等待 Owner 输入

### GPT-5.6 Sol（主持人）

请 Cairn Owner 回答：

> 未来两个 milestone 中，Cairn 必须支持哪些 primitive-level transform？其中哪些无法
> 表达为 region implementation、region lowering 或 custom-op replacement？

可能路线：

- **存在明确 transform**：先在当前 MathGraph 实现，记录为何需要 primitive
  visibility，形成保留 fine IR 的证据。
- **没有明确 transform**：做最小 RegionGraph 双前端原型，停止扩大 primitive IR。
- **尚不确定**：冻结当前 compiler 功能，只维护 correctness/performance，直到真实
  需求出现。

在 Owner 回答或原型数据出现前，本讨论不形成最终架构决议。
