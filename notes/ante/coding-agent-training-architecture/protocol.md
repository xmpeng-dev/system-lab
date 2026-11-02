# A → B → C → A 自动讨论协议

## 1. 环形角色

```text
┌──────────────────────────────────────────────────────────┐
│                                                          │
▼                                                          │
A — GPT-5.6 Sol → B — Compiler Advocate → C — Complexity Skeptic
```

- **A / 主持与提案者**：读取上一轮 C，核实代码事实，综合前一 cycle，并提出当前最强
  架构命题和一个交给 B 的可反驳问题。
- **B / 编译器支持方**：只回应最新 A，为 Graph/compiler/Program 找出最强事实依据，
  同时标明证据边界，再向 C 提出需要攻击的命题。
- **C / 复杂度质疑方**：读取本 cycle 的 A 和 B，检查隐含假设、重复语义和反例，决定
  `continue` 或 `stop`，并在继续时把下一题交回 A。

角色不并行发言。严格执行 `A → B → C → A`，这样每个 turn 只有一个明确前驱，恢复时
不需要合并并发分支。

## 2. 持久化布局

```text
coding-agent-training-architecture/
├── README.md
├── protocol.md
├── state-<run-id>.json          # 每个并发 run 独立
├── dialogue.md                 # 启用环形协议前的历史 transcript，只读
└── turns/
    ├── README.md
    └── YYYY-MM-DD-HHMMSS-<run-slug>-NNNN-<role>.md
```

文件名使用本地时间、秒和单调递增序号：

```text
2026-10-03-101600-sol-a-0000-c.md
2026-10-03-101701-sol-a-0001-a.md
2026-10-03-101745-sol-a-0002-b.md
2026-10-03-101830-sol-a-0003-c.md
```

按文件名排序就是完整发言顺序。序号是最终顺序依据；时间戳用于审计，不用来处理
并发。`run-slug` 防止两个 Cursor 会话在同一秒写同名文件；序号只在一个 run 内单调
递增。

## 3. 单次发言格式

每次发言只写一个 Markdown：

```markdown
---
discussion: cairn-coding-agent-training-architecture
run: 2026-10-03-101600-sol-a
turn: 1
cycle: 1
timestamp: 2026-10-03T10:17:01+08:00
role: A
agent: compiler-advocate
responds_to: 2026-10-03-101600-0000-c.md
status: continue
---

# A：本轮标题

## 我是谁

我是 Compiler Advocate，...

## 我回应的命题

...

## 发言

...

## 证据

- 路径、artifact 或明确标注的假设。

## 交给下一位

向 B 提出一个具体、可反驳的问题。
```

约束：

- 一份文件只允许一个角色发言；
- 不修改历史 turn；
- 事实、判断、假设分开；
- 没有新证据时必须明确写“本轮无新增证据”；
- A/B 不能宣布最终决议；只有 C 能设置 `stop`，Owner 才能批准迁移。

## 4. 状态机

每个 run 拥有独立的 `state-<run-id>.json`，最少包含：

```json
{
  "run": "2026-10-03-101600-sol-a",
  "status": "ready",
  "next_role": "A",
  "next_turn": 1,
  "cycle": 1,
  "last_turn": "turns/2026-10-03-101600-0000-c.md",
  "question": "本 cycle 交给 A 的具体问题",
  "max_turns": 9,
  "stop_reason": null
}
```

每个 turn 的事务顺序：

1. 读取本 run 的 `state-<run-id>.json`；
2. 验证 `next_role` 与即将发言的角色一致；
3. 读取 `last_turn`；C 还要读取本 cycle 的 A、B；
4. 生成角色发言；
5. 写入新的 timestamped Markdown；
6. 验证 frontmatter、前驱和序号；
7. 最后更新本 run 的 state。

若第 5 步成功、第 7 步失败，恢复者只扫描相同 `run` frontmatter 的最大序号，核对后
修复 state。禁止先更新 state 再写发言。禁止多个 run 共享一个可写 state 文件。

## 5. 角色上下文

每人只加载完成本轮所需的最小上下文：

- **A**：`README.md`、`protocol.md`、最新 C turn、需要核实的源文件；
- **B**：`README.md`、`protocol.md`、最新 A turn、A 引用的证据；
- **C**：本 cycle 的 A/B turns、B 引用的证据、本 run state。

默认不把完整历史 transcript 送入每个 agent。需要历史时按 turn 链向前读取，避免讨论
越长，上下文成本线性增长。

## 6. 自动运行器

在 Cursor 中发送：

```text
运行 Cairn 环形讨论，最多 <N> 次发言。
```

GPT-5.6 Sol 作为 orchestrator：

```text
while state.status == "ready" and emitted < N:
    if next_role == A:
        GPT-5.6 Sol performs A
    elif next_role == B:
        resume/spawn Compiler Advocate
    else:
        resume/spawn Complexity Skeptic

    write one turn markdown
    advance A→B→C→A
    update state last
```

这里的“自动”表示：用户触发一次后，orchestrator 连续完成有限个 turn；不是三个 agent
互相建立网络连接。subagent 不能直接给另一个 subagent 发消息，Markdown turn 是它们的
mailbox，GPT-5.6 Sol 是调度器。

## 7. C 的停止规则

C 满足任一条件时把 `state.status` 设为 `stopped`：

- A 和 B 连续一个 cycle 没有新增证据或论点；
- 分歧只剩 Owner 产品选择；
- 下一步需要修改 Cairn、运行 GPU 实验或产生显著成本；
- 关键事实无法从当前 checkout 核实；
- 达到 `max_turns`；
- 发现前驱、序号或状态不一致。

C 的 stop turn 必须写清：

- 已达成共识；
- 未解决分歧；
- stop reason；
- 唯一下一步；
- 谁拥有下一步。

Owner 或主持人提供新证据后，可以更新 `question`、增加 `max_turns` 并把状态恢复为
`ready`；历史 turn 保持不变。

## 8. 失败与恢复

- **subagent 失败**：不写 turn，不推进 state；可以同角色重试一次。
- **发言格式错误**：orchestrator 要求同角色重写，不把坏响应落盘。
- **重复运行**：先检查 `state.status` 和最大 turn 序号，不创建重复 turn。
- **会话 ID 失效**：按 README 中角色定义启动新 subagent；turn 中记录新的 agent ID。
- **源代码变化**：C 在下一 turn 记录 git state 或相关文件 digest，不能把旧结论无条件
  延伸到新实现。
