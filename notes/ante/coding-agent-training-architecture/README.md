# Cairn coding-agent 训练架构讨论

这个目录以持续对话的方式讨论 Cairn（原 ante）的执行架构，当前核心问题是：

> 面向 coding-agent 时代，Cairn 应保留细粒度
> `MathGraph → compiler → Program → Runtime`，还是收敛为
> `RegionGraph → Program → Executor`？

讨论在 slab 中维护；Cairn 仓库只保留执行代码和形成决议后的正式架构文档。

## 参与者

### 常驻三人（A → B → C）

- **A — GPT-5.6 Sol**：主会话 agent，主持与提案。负责读取上一轮 C、核实代码事实、
  综合前一 cycle，并向 B 提出可反驳命题。
- **B — Compiler Advocate**：general-purpose coding subagent，立场是为 Graph、compiler、
  Program 和静态验证寻找最强的事实依据。会话 ID：
  `96c1aae4-6864-4d6b-a523-4c8aed8b7381`。
- **C — Complexity Skeptic**：general-purpose coding subagent，检查重复 PyTorch 语义、
  agent 上下文负担和不必要的 IR 粒度。会话 ID：
  `a6ab1c16-ef83-4287-928c-6eb92992b852`。

### 特邀历史评审

- **Agent Maintainability Reviewer**：general-purpose coding subagent，从语义编辑距离、
  首次 contract 成功率和长期维护成本进行仲裁。参与 Round 1–2，不计入后续常驻三人。
  会话 ID：
  `d9f64185-f64d-4450-b9d5-fac055ab328e`。
- **Cairn Owner**：人类决策者。决定产品边界、下一阶段必须支持的变换，以及何时进入
  原型和迁移。

三个 subagent 均使用主会话继承模型；角色名表示评审立场，不表示不同厂商或独立组织。

## 文件

- [`agent-native-summary.md`](./agent-native-summary.md)：agent-native 训练系统工程的核心
  定义、架构要求、度量指标与 Cairn 当前判断。
- [`protocol.md`](./protocol.md)：A → B → C → A 状态机、发言格式、恢复和停止规则。
- `state-<run-id>.json`：每个自动讨论 run 独立记录下一位角色、turn、cycle 和状态；
  不共享可写 state。
- [`turns/`](./turns/README.md)：一份 Markdown 一次发言，按时间戳和序号排序。
- [`dialogue.md`](./dialogue.md)：启用环形协议前的历史 transcript，只读、不再追加。

## 如何启动

在当前 Cursor 会话中发送：

```text
运行 Cairn 环形讨论，最多 <N> 次发言。
```

例如：

```text
运行 Cairn 环形讨论，最多 6 次发言。
```

orchestrator 启动时先生成唯一 `run-id`，之后只从对应 state 的 `next_role` 恢复，不需要
用户指定从谁开始：

```text
A → B → C → A
```

每位角色一次只发言一次，并写入：

```text
turns/YYYY-MM-DD-HHMMSS-<run-slug>-NNNN-<role>.md
```

写完 turn 后才更新本 run state，因此中断后可以按 frontmatter 的 `run` 和最大序号
恢复。完整事务规则见
[`protocol.md`](./protocol.md)。

无需定时器。一次用户触发自动完成有限个顺序 turn，避免没有新证据时按时间空转。
subagent 不能彼此直接通信；Markdown turn 是 mailbox，GPT-5.6 Sol 是 orchestrator。

## 自动讨论的停止条件

满足任一项即停止并等待 Owner：

- 双方结论和理由连续一轮没有新增；
- 分歧已经收敛为需要 GPU 实验或产品选择；
- 某项事实无法从当前 checkout 核实；
- 达到用户指定的轮数；
- 下一步会修改 Cairn 代码、启动昂贵 GPU 作业或作出不可逆迁移。

## 维护协议

1. 新内容写入 `turns/` 的新文件，不回写旧发言来制造事后共识。
2. 每条发言以 `角色名（身份）` 开头，并区分事实、判断和待验证假设。
3. 引用性能数字时注明 artifact；无法从当前 checkout 核实的内容明确标注。
4. 主持人可以总结共识，但不能把未解决分歧写成最终架构。
5. 架构选择必须落到可证伪实验：correctness、Program/verifier 等价、性能、显存和
   agent edit-distance。
6. 后续继续讨论时优先恢复以上会话 ID；失效时按相同角色 prompt 新建 agent，并在
   turn 中记录新 ID。
7. 最终决议若形成，应写回 Cairn 正式架构文档；本目录保留讨论过程，不替代
   [`docs/engine.md`](../../../../Cairn/docs/engine.md) 或
   [`docs/architecture.md`](../../../../Cairn/docs/architecture.md)。

## 当前状态

### Owner 决议（2026-10-03）

讨论后的未决项已经关闭，并写入 Cairn
[`docs/architecture.md`](../../../../Cairn/docs/architecture.md)：

1. 只保留一个 hierarchical `MathGraph`；Region 是其中的 typed composite operator，
   不建立 `RegionGraph → MathGraph` 双栈。
2. 可移植目标和约束进入 `OptimizationIntent`；精确 schedule、communication、
   precision、layout、residency 和 kernel 决策进入目标相关 `Implementation`。只有后者
   可执行、可测量。
3. Verdict 只在本 regime 有效；跨 regime evidence 必须显式标记为
   `comparable` 或 `prior`，不能自动继承 accepted verdict。
4. `BindingManifest` 是一个 immutable、content-addressed 的逻辑对象，canonical
   serialization 只在 `run.json` 中保存一次。
5. Cairn 采用 kernel portfolio，不采用 source monoculture。AITER、Primus-Turbo、
   hipBLASLt、FlyDSL、Triton 和 reference implementation 都可成为候选；Cairn 必须拥有
   semantic contract、typed adapter、选择、provenance、correctness 与 E2E evidence，
   但不要求拥有每个获胜 kernel 的源码。

### Owner 决议（2026-10-04，采纳 B 的架构 review）

B 的六条 review 经代码核对全部属实，按 A 的修订意见写入 Cairn 架构文档 §4.6–§4.11、
§7.1、§10、§11.7–§11.13 和 §13：

1. Lowering 只枚举和拒绝；Resolver 是唯一把测量结果变成选择的地方。
   `Implementation` 是基于 typed key 的声明式规则，Manifest 是规则在每个任务上的展开。
2. 每个字段只有一个主人（§4.11）。可接受误差归 Workload，格式的预期误差归
   Implementation；改 Workload 字段不算 transform，只开启新的 baseline 和 regime。
3. Program 覆盖整个 step：step 尾部先作为粗粒度任务进入；FSDP2 短期作为 pinned
   provider。
4. 动态分派是 Program 的一等节点，分为三类：形状动态、选择动态、语义动态；分派表
   在 CPU 上做完整性检查，每步只记聚合分布和丢弃计数。
5. Cairn 之下的决策分四类：fixed、chosen-and-recorded、environment、opaque；
   opaque 下产生的证据最多是 comparable。
6. 先建评估与归因层，再建 proposer；Transform 的主要形式是新增候选；T1 上界由
   attribution 推出。
7. 每个 rank 一条计算流是显式决定；`OptimizationIntent` 等到第二个架构出现再实现。

### Owner 决议（2026-10-04，第二轮 review）

第二轮七条 review 经代码核对属实，写入 Cairn 架构文档 §4.4–§4.11、§5.4–§5.5、
§7.1–§7.4、§10、§11.7–§11.15 和 §13：

1. Implementation digest = 规则 + 规则所引用机制的内容摘要 + provider 版本 + tuning
   摘要；Cairn revision 进入 regime。受保护 regime 在 CPU 上比较 program/manifest
   digest：不变即证明未受影响，变了就重跑该 regime 的 gate。§11.13 前加第 0 步
   provenance。
2. Resolution 分阶段：先结构性规则，再 lowering，再在 lowering 之后的 key 域上解析
   binding 规则，最后展开。候选枚举是 provider adapter + capabilities 上的独立函数。
3. 粗任务必须声明内部集合通信（group 与顺序）；参数带版本 effect，verifier 检查跨步
   依赖。
4. 归因上界只能拒绝作用范围被 profile 完整界定的 transform；作用于共享资源的只排序、
   不拒绝。
5. 分派表完整性是类型性质：key 值域声明为有限且封闭，由 verifier 证明包含于表；表中
   所有 artifact 在第 1 步之前编译、加载完毕。
6. Cairn 设置的运行时配置（GC、workspace、它设置的环境变量）一律归 Implementation，
   属于 fixed；environment 只指 Cairn 不设置的东西。
7. Trace 每个 regime 抓一次、toolchain 变化时重抓，前提是 provider 的选择由已记录的
   输入确定；现有两个受保护 floor 在分类完成前只算 comparable。

A 的修订（第二轮）：#1 的 gate 不要求 digest 必须不变，而是“不变即免跑、变了就重跑”，否则合法的
共享代码改进会被一概禁止；#5 中预编译代价记入证据，值域过大时粗化而不是懒加载；#6 中
用 `setdefault` 设置却被 launcher 覆盖的值按实际值记录并标注；#7 中选择不能由已记录
输入确定的仍归 opaque。

### Owner 决议（2026-10-04，模型前端）

采用“TorchTitan 风格前端 + Cairn 执行后端”，写入 Cairn 架构文档 §11.16、§12.2 和
不变式 22：

1. 模型是普通 `nn.Module` 加 `forward`，写法同 TorchTitan 和 HF。Cairn 把这个
   forward capture 成分层 MathGraph；`GraphBuilder` 只是 compiler 内部的构造 API，
   模型不调用它。
2. 路由、load-balance、routed experts、attention、RoPE 是 `cairn/frontend.py` 里的
   双实现函数：capture 时发出 region operator，eager 时是 plain-torch reference。
   同一个 forward eager 运行就是 reference，不再另写一份数学。
3. Cairn 没有对应 operator 的 torch 调用变成 `TorchFunction` 节点，走
   `AutogradVjp`，在 `run.json` 里逐个列出，优化阶段再替换。原地更新、随机数生成等
   编译后无法重放的操作在 capture 时直接拒绝。
4. 采用 TorchTitan 的模型写法、模型代码、HF 权重转换和 meta 初始化；不采用它的执行
   机制（module hook、包装、autograd 顺序和线程同步）。Primus 的 TorchTitan 后端作为
   外部性能锚点。

证据（Cairn 工作区会话完成，见 `docs/engine.md` §2.1，均为 scratch）：TorchTitan
llama3 原样 capture，loss 和 45 个梯度一致；TorchTitan Qwen3 MoE 只替换 `MoE` 一个
模块后 capture，`ep=1` 和 `ep=8` 下 loss 和 41 个梯度一致。原计划的 builder 与
capture 对照实验因此不再需要。HF 模型文件尚未尝试。

剩余缺口：两个模型文件仍读 `distributed.ep_size`/`ep_rank`，并调用
`compiler.partition.layer_partition` 自己切层。目标是由 placement 显式传入本 rank 的
层和专家范围。

### Run `2026-10-03-103157-overall-sol56`（整体架构）— **stopped**

18/18 turns 完成；A=GPT-5.6 Sol，B=Compiler Advocate，C=Complexity Skeptic。最终发言：
[`0018-c`](./turns/2026-10-03-104903-overall-sol56-0018-c.md)。

该 run 当时收敛为 coarse RegionGraph → fine Program，并把 RegionGraph 迁移交给
matched trials 决定。上面的 Owner 决议已经取代该项设计；旧 turn 保持只读，作为推理
过程而不是当前架构。

### Run `2026-10-03-103157-overall`（整体架构）— **stopped**

18/18 turns 完成（`turns/*-overall-*.md`）。B=主会话论证，C=质疑并选题。

地图共识摘要：Train 承诺面 / Search 愿景；`Runtime.step` 零策略分支；并行差=Program；冻结扩展 primitive；loss+全梯度门禁≠性能；引用 floor+`run.json`；瘦 agent 接口；Region/Search 待 Owner。

该 run 的 Region/Search Owner 问题已经由上面的正式决议关闭；未实现能力仍按 Cairn
架构文档的 current-state mapping 管理。

### 先前 RegionGraph 专题 run

历史细节见早期 `turns/`（无 `overall` slug）与 `dialogue.md`（只读）；当前决议以上方
Owner 决议和 Cairn 架构文档为准。
