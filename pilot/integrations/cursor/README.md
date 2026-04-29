# Pilot × Cursor 集成

在 Cursor Desktop / CLI 里把 Pilot 跑起来的薄适配层。不改 Pilot 主干，只提供 Cursor 原生能消费的入口文件。

## 目录内容

| 文件 | 作用 | Cursor 如何消费 |
|------|------|----------------|
| `AGENTS.md` | 项目级 agent 上下文：说明 Pilot 是什么、Orchestrator 怎么跑、哪里找知识 | Cursor 自动读取仓库根或 `pilot/` 下的 `AGENTS.md` 作为主 agent 的常驻上下文 |
| `rules/*.mdc` | 分片的 Cursor Rules：角色 prompt / 工具调用规范 / state hygiene | 复制到用户项目的 `.cursor/rules/` 下即可自动生效 |
| `mcp.json.example` | MCP server 配置模板（把 `pilot/tools/*` 暴露成 MCP tool） | 用户参考后放到 `.cursor/mcp.json` 或 `~/.cursor/mcp.json` |

## 安装步骤

1. **放入 AGENTS.md**（二选一）：
   - **仓库级**：把 `pilot/integrations/cursor/AGENTS.md` 复制到仓库根 `AGENTS.md`（Cursor 会全局常驻）
   - **目录级**：直接保留在 `pilot/AGENTS.md`，Cursor 在该目录下工作时自动加载
2. **安装 Rules**：
   ```bash
   mkdir -p .cursor/rules
   cp pilot/integrations/cursor/rules/*.mdc .cursor/rules/
   ```
3. **（可选）配置 MCP**：
   ```bash
   cp pilot/integrations/cursor/mcp.json.example .cursor/mcp.json
   ```
   或直接用 shell 调用 `python -m pilot.tools.<name>`，不走 MCP 也行。
4. **验证**：在 Cursor Agent 面板输入 `Start a tuning session for <model> on <cluster>`，应能自动读到 Orchestrator 角色并开始 PREFLIGHT。

## Cursor 侧如何兑现 v2 角色

| v2 角色 | Cursor 侧承载 |
|---------|---------------|
| **Orchestrator**（主会话） | Cursor 的 main Agent 会话；读 `AGENTS.md` + `rules/10-orchestrator-role.mdc` 获得身份 |
| **Stage Worker**（一次性 subagent） | Cursor 的 Task tool 派生子 agent；prompt 来自 `rules/30-worker-*.mdc`（Agent Requested 模式） |
| **State Layer** | 直接读写 `pilot/state/*.yaml` 文件 |
| **Tool 调用** | Shell 调 `python -m pilot.tools.*` 或 MCP（如果启用） |
| **Context hygiene** | `rules/00-pilot-core.mdc` + `rules/20-state-hygiene.mdc` 规定每 stage 出口 checkpoint + trim |
| **Handoff** | Cursor 自身无常驻进程；靠 `state/checkpoints/handoff/` + `state.resume()`，下一次 Cursor 会话从落点续起 |

## 降级说明

- Cursor 的 Task tool 兑现策略 B（subagent isolation）；若某版本 Cursor 没有 Task，Worker 退化为同会话内的"子对话段"，依然走 `rules/30-worker-*.mdc` 的 scope 约束，但 context 隔离弱化。
- Cursor 无常驻进程这一点决定了策略 C（session handoff）以"下一次打开 Cursor"的形式落地，不是自动重启。

## 和其他 integrations 的关系

`pilot/integrations/claude-code/` 与 `pilot/integrations/codex/` 用相同思路，各自兑现对应框架的原生能力。Skills / Tools / Schemas 三份主干完全共享，不会因为接入新框架而分叉。
