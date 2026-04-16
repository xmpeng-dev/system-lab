# Primus Pilot — 训练调优 Agent

> 面向多节点训练集群的窄域调优系统。不替代工程师，把分散在 Primus / Megatron / TorchTitan / ROCm / RCCL / WandB / Slurm 里的调优经验变成可自动诊断、自动试验、自动收敛的闭环。

---

## 1. 问题与边界

多节点训练调优的核心难题：

| 挑战 | 具体问题 |
|------|---------|
| **参数空间爆炸** | DP / TP / PP / EP / VPP / CP × MBS / GBS × recompute × 通信参数，组合 10⁴+，靠专家经验手动选 |
| **瓶颈定位困难** | compute / comm / memory / bubble / load imbalance / data IO 混合交织，每次从零排查 |
| **试错成本高** | 一次多节点实验数百 GPU·h，坏实验跑完才知道 |
| **经验碎片化** | best-known config 散落在 Slack / wiki / 脑中，不可检索不可复用 |
| **环境漂移** | 新 ROCm / RCCL / 框架版本 / 拓扑变更导致性能回归，无自动检测 |

**做什么**：Dense / MoE bring-up、scaling 退化诊断、OOM / 吞吐抖动分析、并行 + 通信联合调参、性能回归检测。

**不做什么**：自动改 kernel / 模型结构 / 通信库实现、无约束大规模黑盒搜索。

---

## 2. 设计原则

| 原则 | 含义 |
|------|------|
| 窄域深做 | 只做训练系统配置调优 |
| 全自动 + 全记录 | 自动运行不需要人工审批，但每步决策和观测数据完整保存、可追溯 |
| 小步试验 | 每次只改一个维度，先小规模验证 |
| 可观测优先 | 没有 metrics 支撑的诊断不输出 |
| 经验可沉淀 | 每次实验结果结构化写回知识库 |
| 安全回滚 | 任何试验都能一键回退 |

---

## 3. 系统架构

### 3.1 总览

```
┌──────────────────────────────────────────────────────────────────────┐
│                          Pilot Agent (唯一的 LLM 大脑)               │
│  skills/knowledge → 推理上下文   function calling → tools            │
│  structured output → schema 约束  guardrails → tool 内嵌硬约束       │
└──────────┬───────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  ┌────────────┐    ClusterBaseline (硬件真值锚点)                    │
│  │ Preflight  │───────────────────┬──────────────┬──────────┐        │
│  │ Skill      │                   │              │          │        │
│  └────────────┘                   ▼              ▼          ▼        │
│       (首轮)               ┌───────────┐  ┌──────────┐  ┌──────┐    │
│                            │  Observe   │  │ Diagnose │  │ Plan │    │
│                            │           │  │          │  │      │    │
│  ClusterBaseline 的使用:    │ gpu_util=  │  │ 实测bw vs│  │ 理论 │    │
│                            │ 实测TFLOPS/│  │ baseline │  │ 增益 │    │
│  Observe: 算利用率          │ baseline   │  │ bw → 配置│  │ 上限 │    │
│  Diagnose: 判定配置问题     │ 1280T      │  │ or 硬件  │  │ 估算 │    │
│    vs 硬件问题              │            │  │ 问题？   │  │      │    │
│  Plan: 估算优化上限         └─────┬─────┘  └────┬─────┘  └──┬───┘    │
│                                  │              │           │        │
│                                  ▼              ▼           ▼        │
│                            RunSnapshot → DiagnosisReport → Plans     │
│                                                                      │
│                            ┌──────────┐         ┌──────────┐         │
│                            │ Execute  │────────►│  Settle  │         │
│                            │ 全量执行  │ Results │ 贪心选优  │         │
│                            └──────────┘         └────┬─────┘         │
│                                                      │               │
│                                    ┌─────────────────┤               │
│                                    ▼                 ▼               │
│                                 CONTINUE        TARGET_REACHED       │
│                              (best-plan →       / CONVERGED          │
│                               下一轮 Observe)   / EXHAUSTED          │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  基础设施层                                                          │
│  Primus Preflight │ WandB/Prometheus │ rocprof │ Slurm │ 知识库     │
└──────────────────────────────────────────────────────────────────────┘
```

**ClusterBaseline 是 Pilot 区别于其他调优系统的关键**——它不是一次性的健康检查，而是贯穿整个 Tuning Loop 的**硬件真值锚点**：

| 消费方 | 怎么用 ClusterBaseline | 示例 |
|--------|----------------------|------|
| **Observe** | 算利用率：实测值 / 基线值 = 实际利用率 | `gpu_util = 实测 342T / baseline 1280T = 26.7%` |
| **Diagnose** | 判"配置问题"还是"硬件问题"：实测带宽 vs 基线带宽 | `allreduce bw = 61GB/s vs baseline 85GB/s → 72%，是配置问题不是硬件坏了` |
| **Plan** | 估算优化上限：基线值 × 理论效率 = 当前配置下的 throughput 天花板 | `baseline IB 85GB/s → comm 优化最多能省 ~24% comm time → throughput +X%` |

没有 Preflight 基线，诊断只能靠经验阈值（"comm_ratio > 0.3 就是瓶颈"）；有了基线，诊断变成**定量对比**（"allreduce 实测 61 vs 理论 85，差 28%，值得优化"）。

### 3.2 六个 Skill 的功能

| Skill | 功能 | 输入 | 输出 |
|-------|------|------|------|
| **Preflight** | 跑 Primus Preflight 获取集群硬件基线（GPU 算力、HBM 带宽、IB/XGMI 带宽、RCCL 集合通信基线、健康检查），作为全程锚点 | 集群信息 | `ClusterBaseline` |
| **Observe** | 采集训练 run 的完整状态，**基于 ClusterBaseline 计算利用率**。支持两种入口：分析已有 run 或以 best-plan 启动新 run | ClusterBaseline + run_id / best-plan | `RunSnapshot` |
| **Diagnose** | 定位性能瓶颈并给出 root cause，**基于 ClusterBaseline 区分配置问题 vs 硬件问题** | RunSnapshot + ClusterBaseline | `DiagnosisReport` |
| **Plan** | 生成 K 个候选调优方案（K=2~3），**基于 ClusterBaseline 估算优化上限**，经约束验证和知识库匹配 | DiagnosisReport + ClusterBaseline | `CandidatePlans {P1..Pk}` |
| **Execute** | 串行执行所有 K 个候选，每个独立 early stop + 完整日志记录，收集完整结果集 | CandidatePlans | `Results {R1..Rk}` |
| **Settle** | 贪心选出 best-plan，全部结果写入知识库，判断收敛/达标/继续迭代 | Results | `SettleReport` |

---

## 4. Tuning Loop：收敛设计（核心）

这是 Pilot 最关键的设计——如何让一个 LLM 驱动的调优循环能可靠地收敛到目标。

### 4.1 循环流程

```
                    ┌────────────────┐
                    │   Preflight    │  获取集群硬件基线 (首轮)
                    │   Skill        │
                    └───────┬────────┘
                            │ ClusterBaseline
                            ▼
                    ┌────────────────┐
          ┌────────►│   Observe      │  分析已有 run / 启动新 run
          │         │   Skill        │
          │         └───────┬────────┘
          │                 │ RunSnapshot
          │                 ▼
          │         ┌────────────────┐
          │         │   Diagnose     │  瓶颈分析 + root cause
          │         │   Skill        │
          │         └───────┬────────┘
          │                 │ DiagnosisReport
          │                 ▼
          │         ┌────────────────┐
          │         │   Plan         │  生成 K 个候选 (K=2~3)
          │         │   Skill        │
          │         └───────┬────────┘
          │                 │ CandidatePlans {P1..Pk}
          │                 ▼
          │         ┌────────────────┐
          │         │   Execute      │  全量执行 + per-candidate early stop
          │         │   Skill        │
          │         └───────┬────────┘
          │                 │ Results {R1..Rk}
          │                 ▼
          │         ┌────────────────┐
          │         │   Settle       │  贪心选 best-plan + 写知识库
          │         │   Skill        │
          │         └───────┬────────┘
          │                 │
          │     ┌───────────┴───────────┐
          │     │                       │
          │  gain > δ               收敛 / 达标 / 耗尽
          │  且未达标                    │
          │     │                       ▼
          └─────┘                  输出最终报告
       (下一轮: Observe
        以 best-plan 启动新 run)
```

### 4.2 收敛性保证

调优的本质是**带约束的噪声搜索**。LLM 可以做推理和决策，但"收敛"不能靠 LLM 自觉——需要**结构化机制**来保证。

#### 4.2.1 贪心递增 baseline

每轮 Settle 从所有候选中选出 best-plan，**仅在 best-plan 严格优于当前 baseline 时**才更新 baseline：

```
baseline[r+1] = best-plan[r]   if throughput(best-plan[r]) > throughput(baseline[r])
              = baseline[r]     otherwise (回退，不退化)
```

这保证了 baseline 序列是**单调非递减**的——系统永远不会比上一轮差。

#### 4.2.2 三重终止条件

| 终止条件 | 含义 | 公式 |
|---------|------|------|
| **目标达成** | 达到用户指定的目标 | `throughput(best-plan) ≥ target` |
| **增益收敛** | 连续提升幅度低于阈值 | `gain[r] < δ` (默认 δ=2%)，连续 2 轮 |
| **搜索耗尽** | 连续多轮所有候选都失败 | 连续 N 轮（默认 3）无任何候选 DONE |

加上一个硬约束：**最大迭代轮数**（默认 max_rounds=5）。即使没触发上述条件，也会强制终止。

#### 4.2.3 为什么这样设计能收敛

```
关键假设：搜索空间是有限的（参数组合被约束裁剪到 10² 量级）
         且 Agent 有知识库记忆（不会重复探索已尝试的方案）

论证：
1. baseline 单调非递减 → 不会后退
2. 知识库写入全部结果（含失败） → Plan 不会重复提出已失败方案
3. 每轮 K=2~3 × 每个候选 early stop → 单轮成本有上界
4. gain < δ 连续 2 轮 → 说明当前搜索方向已无显著收益
5. max_rounds 硬约束 → 总成本有上界
6. 综合 2+4 → 搜索空间在逐轮缩小，最终触发收敛或耗尽
```

#### 4.2.4 每轮成本控制

全量执行 K 个候选意味着成本是 O(K)，因此需要在两个地方控制：

| 控制点 | 机制 | 效果 |
|--------|------|------|
| **Plan 严控 K** | K 默认 2~3，硬约束 ≤5 | 控制每轮实验次数上界 |
| **Execute early stop** | OOM / 吞吐退化 / 方差过大 → 立即终止 | 坏候选不浪费时间 |
| **知识库去重** | Settle 写入全部结果，Plan 查重 | 不重复尝试已知失败方案 |
| **约束裁剪** | constraint_check 前置过滤 | 不合法配置不进入候选 |

总成本上界：`max_rounds × K_max × timeout_per_candidate`，以默认值估算：5 × 3 × 30min = 7.5h。实际因 early stop 通常远低于此。

### 4.3 Settle 的决策逻辑（伪代码）

```python
def settle(results: list[Result], baseline: Config, target: float,
           history: list[RoundResult]) -> SettleReport:
    
    # 1. 过滤 + 排序
    valid = [r for r in results if r.status == DONE and r.step_time_cv < 0.05]
    valid.sort(key=lambda r: r.throughput, reverse=True)
    
    # 2. 选 best-plan
    if not valid:
        # 所有候选都失败
        return SettleReport(action=RETRY_NEW_DIRECTION, 
                           note="所有候选失败，需要 Plan 更换方向")
    
    best = valid[0]
    gain = (best.throughput - baseline.throughput) / baseline.throughput
    
    # 3. 写知识库（全部结果，含失败的）
    for r in results:
        knowledge_write(r)  # 含成功经验和失败经验
    
    # 4. 终止判断
    if best.throughput >= target:
        return SettleReport(action=TARGET_REACHED, best_plan=best)
    
    if gain < delta and last_round_gain < delta:  # 连续 2 轮低增益
        return SettleReport(action=CONVERGED, best_plan=best)
    
    if count_consecutive_all_fail(history) >= 3:
        return SettleReport(action=EXHAUSTED, best_plan=best)
    
    # 5. 继续迭代
    return SettleReport(action=CONTINUE, best_plan=best, new_baseline=best.config)
```

### 4.4 知识库的闭环作用

知识库不只是"存历史记录"——它是收敛的关键加速器：

```
第 1 轮：Plan 没有先验 → K=3 个探索性候选
   → Settle 写入 3 个结果（1 成功, 2 失败）

第 2 轮：Plan knowledge_search → 找到第 1 轮的成功/失败经验
   → 排除已知失败方向 → K=2，更聚焦
   → Settle 再写入 2 个结果

第 3 轮：Plan 有 5 条先验 → 高置信推荐 K=1
   → 收敛
```

知识库让搜索从"盲目探索"变成"经验引导"，每轮迭代都在缩小有效搜索空间。

### 4.5 完整示例

```
用户: "16 节点 MoE 吞吐只有 340T，帮我优化到 400T+"

── 第 1 轮 ──────────────────────────────────────────────────
baseline = 342.5T, target = 400T

[Preflight]  获取基线: mfma_bf16=1280T, ib_allreduce=85GB/s
[Observe]    入口 A: 分析已有 run → RunSnapshot(342.5T, comm_ratio=0.31)
[Diagnose]   COMM_BOUND(0.87) + PIPELINE_BUBBLE(0.72)
[Plan]       K=2: P1=Ring→Tree (low risk), P2=去PP (medium risk)
[Execute]    P1=389.2T/DONE, P2=376.8T/DONE
[Settle]     best=P1(389.2T), gain=13.6% > 2% → CONTINUE
             knowledge_write(P1 成功, P2 次优)
             new baseline = 389.2T

── 第 2 轮 ──────────────────────────────────────────────────
baseline = 389.2T, target = 400T

[Observe]    入口 B: 以 P1 配置启动新 run → RunSnapshot(389.2T, bubble=0.08)
[Diagnose]   通信已优化，主瓶颈 PIPELINE_BUBBLE(0.85)
[Plan]       K=2: Q1=VPP=4, Q2=去PP+重分配DP
             (knowledge_search: P2 上轮已试过去PP方向但效果不如P1,
              Q2 换了不同的 DP 分配策略)
[Execute]    Q1=412.8T/DONE, Q2=OOM/FAILED
[Settle]     best=Q1(412.8T) ≥ 400T → TARGET_REACHED
             knowledge_write(Q1 成功, Q2 OOM 失败经验)

总结: 342.5T → 389.2T (+13.6%) → 412.8T (+20.5%), 2 轮迭代
```

---

## 5. 评估体系设计

Pilot 的评估不能只看"最终吞吐提升了多少"——需要多维度评估 Agent 在调优循环中的每个环节是否可靠。

### 5.1 评估维度

| 维度 | 衡量什么 | 为什么重要 |
|------|---------|-----------|
| **诊断准确性** | Agent 的瓶颈判断是否与专家一致 | 诊断错了，后面全错 |
| **方案有效性** | 生成的候选方案中是否有正收益的 | Plan 质量直接决定收敛速度 |
| **收敛效率** | 达到目标需要多少轮、多少 GPU·h | 成本控制 |
| **安全性** | 是否触发过不安全操作、OOM 未被 early stop | 底线 |
| **知识利用率** | Agent 是否有效使用知识库避免重复探索 | 长期效率 |

### 5.2 离线评估（Benchmark Suite）

不需要真集群就能跑的评估，用于开发迭代：

```
评估集 = 一组历史 tuning case，每个 case 包含：
  - 集群配置 + 模型配置 + 初始 RunSnapshot
  - 专家标注的瓶颈标签 (ground truth)
  - 专家标注的最优配置 (参考答案，允许多个)
  - 实际实验结果 (throughput before → after)
```

**离线评估指标**：

| 指标 | 定义 | 目标 |
|------|------|------|
| **诊断一致率** | Agent 诊断的 top-1 瓶颈类型 vs 专家标注 | ≥ 80% |
| **方案命中率** | Agent 的 Top-K 方案中是否包含专家认可的方向 | Top-3 命中率 ≥ 70% |
| **方案合法率** | 生成的候选方案通过 constraint_check 的比例 | 100%（不合法方案不应出现） |
| **幻觉率** | 诊断结论中没有 metrics 证据支撑的比例 | ≤ 5% |

**怎么构建评估集**：

1. 从团队历史调优记录中提取（WandB run + Slack 讨论 + 最终配置）
2. 每个 case 让 2 个工程师独立标注瓶颈和最优方向，取交集
3. MVP 阶段目标：10~20 个 case 覆盖主要瓶颈类型

### 5.3 在线评估（真实集群）

在真实集群上端到端运行 Tuning Loop 的评估：

| 指标 | 定义 | 目标 |
|------|------|------|
| **吞吐提升比** | `(final - initial) / initial` | 场景相关，带 baseline 对比 |
| **收敛轮数** | 达到目标或收敛所需迭代轮数 | ≤ 3 轮 (典型场景) |
| **总 GPU·h** | 整个调优过程消耗的 GPU 时间 | < 人工调优的 50% |
| **early stop 有效率** | 被 early stop 的候选中实际是差方案的比例 | ≥ 90% |
| **安全事件数** | 未被拦截的危险操作次数 | 0 |

### 5.4 对比评估（Agent vs 人工）

最有说服力的评估方式——同一场景，Agent 和人工分别调优：

```
实验设计：
  选取 N 个典型场景（Dense bring-up / MoE scaling / 性能回归）
  每个场景双盲：
    - 组 A：工程师手动调优，记录时间和操作
    - 组 B：Agent 调优（仅在 approve 步骤有人参与）
  
  对比指标：
    1. 最终吞吐 (Agent ≥ 人工 × 0.95 即可接受)
    2. 耗时 (Agent 目标 < 人工 × 0.5)
    3. 尝试次数 (Agent 因 early stop 应更少无效尝试)
    4. 知识产出 (Agent 自动写知识库 vs 人工可能忘记记录)
```

### 5.5 评估建议

**MVP 阶段优先做**：
1. 离线诊断准确性评估（投入低、迭代快）——先确保 Diagnose 准确再往后走
2. 方案合法率（应该是 100%，否则 constraint_check 有 bug）
3. 单场景端到端演示（1 个场景跑通全流程，人工观察验证）

**不要过早做**：
1. 大规模 Agent vs 人工对比（成本高，MVP 阶段场景不够多）
2. 收敛轮数 / GPU·h 的统计分析（样本量太小没有统计意义）

**持续积累**：
- 每次真实调优都录入评估集，评估集随使用自然增长
- 每次 Agent 犯错都分析原因并补充对应的 case + 知识库规则

---

## 6. 核心技术模块

### 6.1 Tuning IR（统一配置表达）

屏蔽 Megatron / TorchTitan 差异，所有 Skill 操作统一配置格式：

```yaml
tuning_ir:
  parallel: {dp, tp, pp, ep, vpp, cp}
  memory:   {recompute, offload, grad_accum_steps, zero_stage}
  batch:    {mbs, gbs, seq_len}
  comm:     {rccl_algo, nchannels, overlap_grad_reduce, bucket_size_mb}
  runtime:  {dtype, compile, num_workers}
  
  constraints:
    dp * tp * pp * ep <= total_gpus
    gbs % (dp * mbs) == 0
    hidden_size % tp == 0
    num_experts % ep == 0
    mem_estimate(config) <= gpu_mem * 0.95
```

Backend 翻译：Tuning IR → Megatron Args / TorchTitan TOML / RCCL Env / Slurm Script。

### 6.2 Safe Search（三层漏斗）

```
Layer 1: Constraint Pruning (零成本)
  硬约束裁剪 + 知识库排除已知失败 → 10⁴ 降到 10²

Layer 2: Heuristic Ranking (低成本)
  知识库匹配 + 规则打分 + 排序 → 10² 降到 K=2~3

Layer 3: Online Execution (有保护的高成本)
  全量执行 K 个 + per-candidate early stop → 选出 best-plan
```

### 6.3 Guardrails（安全边界）

嵌入在 tool 实现中的硬约束，LLM 无法绕过：

- `slurm_submit`：OOM 预估超 95% 自动拒绝提交，每次提交记录完整审计日志
- `config_patch`：每次变更自动 git commit，可 diff / rollback
- `early_stop`：OOM / 吞吐 < 0.8×baseline / CV > 0.15 / 超时 → 自动终止
- 单轮最多 5 次实验，Agent 最多 20 轮 tool call

---

## 7. Guardrails 与可观测性

系统全自动运行，安全性靠代码级硬约束保证，可追溯性靠完整的观测日志保证。

| 机制 | 位置 | 作用 |
|------|------|------|
| OOM 预估 | `constraint_check` + `slurm_submit` | 自动阻止高风险配置 |
| early stop | Execute 监控循环 | 自动终止坏实验（OOM / 吞吐退化 / 方差过大） |
| `safe_rollback` | Execute 每个候选执行后 | 自动恢复环境状态 |
| max_rounds | Settle 终止判断 | 总成本硬约束 |
| 结构化输出 | Pydantic schema | 防止 LLM 幻觉输出 |
| evidence 约束 | DiagnosisReport validator | 每条诊断必须有 ≥2 条数据证据 |
| **审计日志** | 每个 tool call | 每次 LLM 决策、tool 调用、参数、结果完整记录（OpenTelemetry） |
| **实验快照** | Execute 每个候选 | 配置 diff + metrics + profiler trace + 日志 全量保存 |
| **知识库写入** | Settle 每轮 | 所有候选结果（含失败的）结构化写入，可事后审查 |

---

## 8. MVP 路线图

### Phase 0：工具层（Week 1-2）

先把工具做通，确保数据能采集、配置能读写。

| 交付物 | 说明 |
|--------|------|
| `pilot snapshot <run_id>` | 生成 RunSnapshot |
| `pilot config-diff <a> <b>` | 结构化配置 diff |
| Tuning IR v0.1 | Megatron 后端翻译 |

**验证**：对在跑的训练生成完整 snapshot。

### Phase 1：诊断 Agent（Week 3-5）

| 交付物 | 说明 |
|--------|------|
| Diagnose Skill | LLM 诊断 + 规则库 |
| 离线评估集 v0.1 | 10+ 历史 case |
| `pilot diagnose <run_id>` | CLI 入口 |

**验证**：诊断一致率 ≥ 80%（对离线评估集）。

### Phase 2：调优提案（Week 6-8）

| 交付物 | 说明 |
|--------|------|
| Plan Skill | 知识库匹配 + 约束裁剪 + 排序 |
| Safe Search Layer 1+2 | 三层漏斗前两层 |
| `pilot suggest <run_id>` | 输出 Top-K 方案 |

**验证**：Top-3 方案命中率 ≥ 70%。

### Phase 3：闭环执行（Week 9-12）

| 交付物 | 说明 |
|--------|------|
| 完整 Tuning Loop | Preflight → Observe → ... → Settle |
| Execute + early stop | Slurm 提交 + 监控 + 回滚 |
| Settle + 知识库 | 贪心选优 + 收敛判断 + 知识沉淀 |
| `pilot tune <run_id>` | 一键触发完整闭环 |

**验证**：在真实集群完成一次完整闭环（2+ 轮迭代收敛）。

---

## 9. 技术栈

| 组件 | 选型 | 备注 |
|------|------|------|
| Agent 框架 | OpenAI Agents SDK / 轻量自研 | function calling + structured output |
| LLM | GPT-4o (主) / Claude Sonnet (备) | 强 function calling + 长上下文 |
| 输出约束 | Pydantic models | RunSnapshot / DiagnosisReport / ExperimentProposal |
| 配置存储 | Git repo (YAML) | 每次变更 = commit，可 diff / rollback |
| 知识库 | SQLite (MVP) → PostgreSQL | 历史 case + 规则 + 向量索引 |
| 向量检索 | FAISS | knowledge_search 后端 |
| CLI | Typer | `pilot` 命令行入口 |
| Trace | OpenTelemetry | 每次 LLM 推理 / tool call 都记录 |

---

## 10. 风险与缓解

| 风险 | 缓解 |
|------|------|
| LLM 幻觉（诊断无依据） | Pydantic 强制 evidence ≥ 2 + guardrail |
| LLM 执行危险操作 | OOM 预估 + constraint_check 自动拦截（代码级硬约束） |
| 搜索不收敛 | 三重终止条件 + max_rounds 硬约束 |
| 实验导致训练中断 | early stop + safe_rollback + 完整审计日志可追溯 |
| 知识库冷启动 | 手动录入 10+ 历史 case，持续积累 |
| 工具层集成复杂 | Phase 0 先做通工具再写 Agent |

---

## 11. 一句话总结

Primus Pilot = 一个 LLM 大脑 + 六个 Skill（preflight / observe / diagnose / plan / execute / settle）+ 贪心递增的迭代循环 + 知识库驱动的搜索加速——把团队在 Primus / ROCm / RCCL / MoE 上的调优经验变成能自动收敛的闭环系统。
