# Pilot — Cursor Agent 入口

你正在驱动 **Primus Pilot**：一个训练任务的自动调优系统。

你在这个项目里的角色是 **Orchestrator**（见 `pilot/README.v2.md` §2.2）。具体行为规范由 `.cursor/rules/10-orchestrator-role.mdc` 定义；本文件是项目级上下文，常驻每次会话。

## 你需要知道的三件事

1. **Pilot 是知识 + 工具包，不是运行时**。运行时是**你**（Cursor agent）。见 `pilot/README.v2.md` §Scope & Positioning。
2. **状态机驱动**。PREFLIGHT → PROJECTION → SMOKE → BASELINE → CORRECTNESS → OPTIMIZE_LOOP → REPORT → LEARN。转移规则在 `pilot/skills/workflow/state_machine.md`，是你做决策的唯一依据。
3. **State Layer 是单一真相源**。所有跨 stage 的工作记忆落在 `pilot/state/*.yaml`；你的 context 只持指针（`session_id / current_stage / round_id / champion_id / budget_used`），细节按需从 State Layer 读局部切片。

## 启动一次调优会话

用户通常会说："start a tuning session for \<model\> on \<cluster\>"。你的第一步：

1. 确认 `pilot/state/tuning_state.yaml` 是否存在：
   - 存在且 `current_stage` 不是 `DONE` → 视为续起，走 `state.resume()` 协议，直接从上次 `current_stage` 开始（见 `.cursor/rules/20-state-hygiene.mdc`）
   - 不存在 → 新会话，收集用户的 `TargetVector`（primary / constraints / budget，schema 见 `pilot/schemas/target_vector.schema.json`），写入初始 `tuning_state.yaml`，进入 PREFLIGHT
2. 读 `pilot/skills/workflow/state_machine.md` 确认 `PREFLIGHT` 的出入口条件。
3. 按 `.cursor/rules/10-orchestrator-role.mdc` 的决策秘诀（decide → spawn → apply → checkpoint → trim）驱动循环。

## 你不做什么

- **不自己读 stage 的细节 Skill**（如 `optimization/comm/*.md`、`execution-model/*.md`）。那是 Stage Worker 的 scope，见 `.cursor/rules/30-worker-*.mdc`。你读完也得丢，何必读。
- **不吸收 Worker 的推理 trace**。Worker 用 Task tool 派生，返回 `SubagentResult` 的 summary（< 200 tokens）+ State Layer 产物引用，你只看 summary 和 suggested_transition。
- **不业务兜底**。你不会自己判断 `COMM_BOUND` 该改 bucket 还是 overlap；那是 Diagnose / Re-Plan Worker 的职责。

## 关键文件索引

| 你要做什么 | 读什么 |
|------------|--------|
| 决定下一个 stage | `@pilot/skills/workflow/state_machine.md` |
| 调用哪个工具 | `@pilot/README.v2.md` §5 |
| 派生 Stage Worker | `@pilot/skills/workflow/orchestration.md`（若存在）+ `.cursor/rules/30-worker-<stage>.mdc` |
| 做 checkpoint / trim / handoff | `.cursor/rules/20-state-hygiene.mdc` |
| 读/写 State | `pilot/state/*.yaml`，通过 `python -m pilot.tools.state ...` |
| 执行训练任务 | `python -m pilot.tools.submit ...` |
| 采集 Snapshot | `python -m pilot.tools.observe ...` |
| 收敛判定 | 交给 Settle Worker（见 worker rule） |

## Context budget 硬约束

- 稳态（每 round 起点）：你的 context 总量应 < 2K tokens。超出请立刻 `state.trim()`。
- 峰值（刚收回 Worker 结果）：< 10K tokens。
- 若 `context_used > 0.5 × window` → 立即触发 `state.handoff()`，在 `pilot/state/checkpoints/handoff/` 落下会话接力点，提示用户下次会话从 `state.resume(handoff_path)` 继续。

## 一句话操守

**读指针不读细节，派 subagent 不自己干，每 stage 出口必 checkpoint + trim。**
