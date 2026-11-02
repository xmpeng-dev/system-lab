# Agent-native 训练系统工程

> 作者：GPT-5.6 Sol（讨论角色 A）  
> 日期：2026-10-03  
> 状态：设计原则总结；不代表 Cairn 已经通过工程验证

## 核心判断

Agent 时代最重要的变化不是代码生成速度提高，而是“提出修改—执行验证—定位原因—
继续修改”可以成为机器自主运行的闭环。

因此，训练系统工程的主要优化目标应从降低人工编码成本，转向降低：

```text
Intent → Candidate → Evidence → Verdict
```

这段路径的端到端时间，即 **Intent-to-Verdict latency**。

一个系统是否 agent-native，不取决于它是否接入了聊天界面、生成了更多代码或拥有更多
IR，而取决于 agent 能否在有限上下文内提出局部、可逆的改变，并快速获得确定、可归因、
可复现的结论。

## Agent 时代的七个特征

### 1. 生成变便宜，验证成为主要瓶颈

Agent 可以快速生成 schedule、kernel binding、graph transform、配置、测试和 benchmark。
稀缺资源转变为：

- 如何证明修改正确；
- 如何判断性能收益来自哪里；
- 如何识别 workload 特化和测量噪声；
- 如何确认没有破坏其他受保护场景；
- 如何决定接受、拒绝或继续实验。

因此，可验证性比可生成性重要。五分钟完成修改、两天才能确认结果的系统，仍然不是
agent-native。

### 2. Agent 的语义上下文有限

Agent 难以稳定维护分散在大量文件、隐式配置和动态控制流中的约束。文档数量增加也不会
自动解决这个问题。

系统应缩小一次工程动作需要理解的完整语义工作集。例如修改 schedule 时，理想上下文是：

```text
RegionGraph
+ CompilePolicy
+ ScheduleTransform
+ verifier diagnostics
+ protected regimes
```

而不是重新理解完整训练栈。应度量的不是总代码行数，而是一次修改涉及的文件数、语义点
数量和输入上下文大小。

### 3. Agent 是概率性的，系统边界必须是确定性的

Agent 负责提出候选，不能同时充当最终裁判：

```text
Agent      proposes a Transform
Compiler   produces an executable artifact
Verifier   checks structural contracts
Runtime    executes the declared artifact
Evaluator  produces evidence
System     issues a mechanical Verdict
```

Runtime 不应静默改变 kernel、layout、precision、schedule 或 residency。否则即使结果变好，
也无法判断真正执行了什么，后续 agent 不能可靠复现和继续优化。

### 4. 封闭、类型化的动作空间优于任意源码修改

“编辑任意 Python 文件”是开放动作空间，难以验证、归因和回滚。更适合 agent 的主要接口是：

```text
Candidate = BasePlan + Transform
```

Transform 应具备：

- 类型化和可序列化；
- 静态检查；
- immutable parent；
- 可逆、可比较和可重放；
- 明确的影响范围。

源码编辑不会消失，但应逐渐从主要控制接口下降为实现新 primitive、compiler pass 或
runtime capability 的扩展接口。

### 5. 结构化 observation 优于更多文本日志

Agent 需要的不是更长的日志，而是可机器消费、相互关联的报告：

```text
CompileReport
VerifyReport
CorrectnessReport
ExecutionManifest
MeasurementReport
AttributionReport
Verdict
```

每份报告都应回答：

- 执行的是哪个 source、plan 和 binding；
- 与哪个 baseline 比较；
- 哪个决策导致当前结果；
- 证据覆盖哪些 workload；
- 哪些结论仍然未知或已经 stale。

### 6. 并行 agent 要求隔离和事务语义

多个 agent 可以并行探索 kernel、schedule、correctness 和性能，但共享可变状态会导致：

- baseline 漂移；
- 候选互相污染；
- 重复实验；
- turn 或 artifact 冲突；
- 结果覆盖和错误归因。

Agent-native 系统需要原生提供：

```text
immutable baseline
isolated candidate
append-only evidence
content-addressed artifact
explicit ownership
atomic promotion
```

这些不是外围 CI 细节，而是自动工程闭环的基础设施。

### 7. 工程记忆必须存在于系统外部

Agent 本身不是可靠的长期记忆载体。系统必须持久化：

- 哪些 Transform 已经尝试；
- 每次失败的 fault class；
- 结果对应的硬件、软件版本和 workload；
- 证据的适用边界和失效条件；
- 哪些历史结论已被新结果推翻。

完整闭环最终应形成：

```text
Candidate → Execution → Evidence → Verdict → Reusable Knowledge
```

否则更换 agent 或压缩上下文后，系统仍会重复探索相同空间。

## 对训练框架架构的要求

Agent-native 不规定必须采用 `MathGraph` 或 `RegionGraph`，也不规定 IR 层数。它要求每一层
都能说明自己如何降低 Intent-to-Verdict，而不是仅增加表达能力。

一个合理的职责划分是：

```text
Agent Control Plane
  → Candidate = BasePlan + Transform
  → semantic graph
  → explicit compile policies
  → verified executable program
  → BindingManifest
  → policy-free executor
  → structured evidence
  → mechanical verdict
```

其中：

- semantic graph 提供紧凑、稳定的修改对象；
- compiler 将隐式决策物化为可检查的 Program；
- verifier 在占用 GPU 前拒绝结构错误；
- BindingManifest 固定实际执行对象；
- executor 忠实执行，不重新做策略决策；
- evidence system 保存 correctness、性能、显存和归因；
- verdict 依据预先声明的 gates 机械生成。

任何新增抽象都应回答三个问题：

1. 它是否缩小了 agent 的语义工作集？
2. 它是否增加了可静态验证的约束？
3. 它是否缩短了从修改到可信结论的时间？

如果三个答案都是否，抽象大概率只增加了 agent 负担。

## 应度量的核心指标

判断 agent-native 不能只看最终吞吐。至少需要持续记录：

- **Intent-to-Verdict latency**：从工程意图到机械结论的总时间；
- **Edit-to-Verdict latency**：开始修改到结论的时间；
- **semantic context size**：完成任务需要加载的语义上下文；
- **edit sites**：一次变更触及的文件和语义位置；
- **agent actions**：工具调用、修改、重试和恢复次数；
- **first-pass contract success**：第一次候选通过结构与正确性门禁的比例；
- **debug iterations**：得到可接受候选前的失败轮数；
- **GPU minutes per accepted change**：每个被接受变更消耗的 GPU 时间；
- **evidence reproducibility**：相同 artifact 和环境能否重放同一结论；
- **protected performance and memory**：受保护场景是否保持吞吐和显存边界。

## Cairn 的当前判断

Cairn 的架构讨论已经形成 agent-ready 假设：

- 采用单一 hierarchical MathGraph，semantic region 是其中的一等 composite operator；
- 将可移植 `OptimizationIntent` 与目标相关 `Implementation` 分离；
- 将 schedule、lowering、residency 和 binding 变为显式决策；
- 使用可逆 Transform 表达候选；
- 将 Compiler、Verifier、Executor 和 Verdict 分离；
- 使用有限 protected regimes 控制证据成本；
- Verdict 只在本 regime 有效，跨 regime evidence 只能标为 comparable 或 prior；
- 采用 kernel portfolio：允许 AITER、Primus-Turbo、hipBLASLt、FlyDSL、Triton 和
  reference implementation，Cairn 拥有选择、契约和证据，不强制拥有全部源码。

但它尚不能被称为已验证的 agent-native 系统。当前仍缺少：

- 现有 MathGraph 的 agent 工程效率基线；
- 已实现的 OptimizationIntent、Transform、Report、Manifest 和 Verdict schema；
- Attention 与 RoutedExperts 的纵向 slice；
- policy-free Runtime 的完整边界；
- first-class provider/ownership schema 与 provider capability catalog；
- 覆盖并行调度和动态 MoE 的 correctness/performance 证据；
- 成功与失败实验形成的 append-only evidence。

Owner 已决定不建立独立 `RegionGraph → MathGraph` 双栈。Region 作为 MathGraph 内部的
typed composite operator，通过统一 compiler lowering 成细粒度 operator，并保留
provenance。只有代表性任务证明单一分层 MathGraph 无法维持 bounded semantic working
set 时，才重新打开第二 IR 的讨论。

## 最小证明方式

固定同一 agent、代码 revision、硬件环境和验证预算，在 hierarchical MathGraph 路径完成：

1. 替换 `AttentionRegion` implementation；
2. 修改 `RoutedExperts` lowering/overlap；
3. 修改 schedule/residency policy。

这些任务用于验证 region contract 和 policy 分层是否真正降低 Edit-to-Verdict，同时保持
gradients、verifier coverage、protected throughput 和 memory。若 region 内部重新生长出
通用 infer、autodiff 和 rewrite 系统，应撤销该 region 抽象，而不是新增第二套 graph。

## 总结

Agent-native 训练系统工程可以归纳为三个关键词：

1. **闭环**：工程修改能够自动到达可信 Verdict；
2. **约束**：概率性的 agent 在确定、可验证的边界内行动；
3. **证据**：系统依据可重放事实，而不是 agent 的自我解释作出决策。

最终目标不是让 agent 写出更多训练框架代码，而是让每一个被接受的修改都更快、更可靠、
更容易归因，并成为下一轮工程工作的可复用证据。
