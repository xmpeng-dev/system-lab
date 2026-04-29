# Primus Pilot — Agent Skeleton (Optional Python Runtime)

**Version**: v2.0 skeleton (aligned with `pilot/README.v2.md`)

> ⚠️ **这是一个可选的参考实现，不是 Pilot 的主要集成路径。**
>
> 按 `pilot/README.v2.md` 的 **Scope & Positioning**：Pilot 本身是调优领域的
> **知识 + 工具包**（skills / prompts / tools / schemas / state），runtime
> 应该交给具体的 agent 框架（Claude Code / Cursor / Codex / ...）。
>
> **生产路径推荐**：让你的 agent 框架直接消费 `pilot/skills` + `pilot/prompts` +
> `pilot/tools`（未来会在 `pilot/integrations/<framework>/` 下放薄适配层）。
>
> **本目录适用场景**：
> - 你想要一个**无人值守、headless 常驻进程**跑长时调优（cron / CI）
> - 你想要**严格的 context budget 强制**，不信任某个框架的 context 管理
> - 你在**没有原生 subagent 机制**的 LLM 上跑（比如只有原始 API）
> - 你想看"如果自己搭 harness 该怎么写"的参考实现
>
> 如果你用的是 Claude Code / Cursor，**不需要这个目录**——它们的 Task /
> subagent 机制已经兑现了 v2.0 §13 策略 B 与 C。

---

## 这是什么

这是 Primus Pilot 的 Agent 层 **Python 参考实现**，手动落地 v2.0 的双层 Agent 架构：

- **Orchestrator Agent**（长生命周期、瘦 context）：`orchestrator.py`
- **Stage Worker**（一次性、上下文隔离）：`subagent.py`

只搭骨架：业务工具（`submit.run` / `observe.snapshot` / `constraint.check` 等）
以 Anthropic tool schema + stub handler 占位；需要接到真实 Primus / Slurm
时在 `worker_tools.py` 里实现对应 handler。

## 关键设计对应

| 骨架文件 | 对应 README.v2.md |
|----------|-------------------|
| `orchestrator.py` | §2.2 Orchestrator、§13 Context Management |
| `subagent.py` | §2.2 Stage Worker、§13.2 策略 B |
| `state.py::trim()` | §5 `state.trim()`、§12.1 Context hygiene、§13.2 策略 A |
| `state.py::handoff()` | §5 `state.handoff()`、§13.2 策略 C |
| `orchestrator_tools.py` | §5 Orchestrator 专属工具集 |
| `worker_tools.py` | §5 业务工具（Orchestrator 不可见） |
| `schemas.py::SubagentResult` | §8.11 SubagentResult |
| `schemas.py::OrchestratorState` | §8.7 TuningState（trimmed 版） |

## 运行

```bash
# 从 slab/ 根目录
pip install -r pilot/agent/requirements.txt
export ANTHROPIC_API_KEY=sk-...

# 真实模式
python -m pilot.agent --session demo_001 \
    --skills-dir pilot/skills \
    --state-dir pilot/state \
    --gpu-h 10 --max-rounds 5

# Dry-run（无 API key 也能跑，subagent 返回 mock SubagentResult）
python -m pilot.agent --session demo_001 --dry-run
```

## 已落地 vs TODO

**已落地（协议骨架）**：
- Orchestrator 主循环 `resume → decide → spawn → apply → checkpoint → trim`
- `state.trim()` 强制只保留指针类字段
- `subagent.spawn()` 通过独立 Claude 会话实现 context 隔离
- `SubagentResult` schema 校验（summary token 上限）
- `state.handoff()` 在 context 压力大时的接力
- Orchestrator 工具集与 Worker 工具集物理隔离（Orchestrator 见不到业务工具）

**TODO（业务接入）**：
- `worker_tools.py`：各业务工具的真实 handler（当前只有 schema）
- `subagent.py::_handle_tool_use`：业务工具调用路由
- `state.py::apply_result`：完善各 Stage 的产物字段映射
- `skills/workflow/orchestration.md` 与 `state_machine.md`：实际知识内容（这里只有占位）
- Token 计数：目前用 `resp.usage` 估算；真实场景要接入 tiktoken / anthropic token counter

## 推荐阅读顺序

1. `orchestrator.py::Orchestrator.run()` — 看主循环骨架
2. `orchestrator.py::_decide()` — 看 Orchestrator 每步如何保持 thin context
3. `orchestrator_tools.py` — 看 Orchestrator 只有 5 个工具
4. `subagent.py::StageWorker.run()` — 看 Worker 如何独立跑
5. `state.py::StateStore.trim()` — 看 context hygiene 规则
