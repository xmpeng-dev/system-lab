# Primus Pilot — 训练调优系统

> 面向多节点训练集群的自动调优系统。  
> Agent（Cursor / Claude）读取 Skills 中的知识，调用 Tools 执行操作，完成建模→搜索→收敛的闭环。

---

## 1. 问题与边界

| 挑战 | 具体问题 |
|------|---------|
| **参数空间爆炸** | DP / TP / PP / EP / VPP / CP × MBS × recompute × 通信参数，组合 10⁴+ |
| **瓶颈定位困难** | compute / comm / memory / bubble 混合交织，每次从零排查 |
| **试错成本高** | 一次多节点实验数百 GPU·h，坏实验跑完才知道 |
| **经验碎片化** | best-known config 散落在 Slack / wiki / 脑中，不可复用 |

**做什么**：Dense / MoE bring-up、scaling 退化诊断、并行 + 通信联合调参。

**不做什么**：自动改 kernel / 模型结构 / 通信库实现。

---

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Agent Layer                                  │
│                   (Cursor / Claude 承载)                            │
│                                                                     │
│   - 读取 Skills 获取知识（Execution Model、优化策略、诊断规则）        │
│   - 调用 Tools 执行操作（profiling、submit、observe）                │
│   - 驱动 Tuning Loop 完成推理和决策                                  │
└─────────────────────────────────────────────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
┌───────────────────────────┐   ┌───────────────────────────────────┐
│       Skill Layer         │   │          Tool Layer               │
│      (知识 / 非执行)       │   │         (执行 / 代码)              │
│                           │   │                                   │
│  skills/                  │   │  tools/                           │
│  ├── execution-model/     │   │  ├── preflight.py    # 硬件基线   │
│  ├── optimization/        │   │  ├── profiler.py     # 单机采集   │
│  ├── workflow/            │   │  ├── submit.py       # 任务提交   │
│  ├── profiling/           │   │  ├── observe.py      # 运行时采集 │
│  └── constraints/         │   │  └── constraint.py   # 约束检查   │
│                           │   │                                   │
│  Agent 读取这些 .md 文件   │   │  Agent 通过 function call 调用    │
│  获取领域知识和规则        │   │  这些 Python 函数执行实际操作     │
└───────────────────────────┘   └───────────────────────────────────┘
```

**三层职责**：

| 层 | 载体 | 职责 | 示例 |
|----|------|------|------|
| **Agent** | Cursor / Claude | 推理、决策、驱动流程 | "comm_ratio=0.35，根据 skills/optimization/comm/ 应该尝试 overlap" |
| **Skill** | Markdown 文件 | 知识、规则、模型公式 | `T_bubble = (pp-1)/(pp-1+M) × T_comp` |
| **Tool** | Python 函数 | 执行、采集、提交 | `preflight.run()` → ClusterProfile |

---

## 3. 系统流程（泳道图）

```
┌─────────────────────┬─────────────────────────┬─────────────────────────┐
│      Agent          │     workflow Skill      │    optimization Skill   │
│  (Cursor/Claude)    │                         │                         │
├─────────────────────┼─────────────────────────┼─────────────────────────┤
│                     │                         │                         │
│  1. 收集优化需求     │                         │                         │
│       │             │                         │                         │
│       ▼             │                         │                         │
│  2. 读 workflow/    │                         │                         │
│     SKILL.md ─────► │ 告知「流程步骤」         │                         │
│       │             │                         │                         │
│       ▼             │                         │                         │
│  3. 按需求清单       │                         │                         │
│     收集项目信息     │                         │                         │
│       │             │                         │                         │
│       ▼             │                         │                         │
│  4. 交给            │                         │                         │
│     workflow ──────►│ PREFLIGHT               │                         │
│       │             │ 整理并行优化参数         │                         │
│       │             │       │                 │                         │
│       │             │       ▼                 │                         │
│       │             │ PROJECTION              │                         │
│       │             │ 建立 Execution Model    │                         │
│       │             │       │                 │                         │
│       │             │       ▼                 │                         │
│       │             │ 进 optimize-loop ──────►│ BASELINE                │
│       │             │                         │ 跑 test + benchmark     │
│       │             │                         │ 记录起点                │
│       │             │                         │       │                 │
│       │             │                         │       ▼                 │
│       │             │                         │ 自主循环迭代优化         │
│       │             │                         │ (Observe→Diagnose→      │
│       │             │                         │  Re-Plan→Execute→       │
│       │             │                         │  Settle)                │
│       │             │                         │       │                 │
│       │             │                         │       ▼                 │
│       │             │                         │ 收敛 / 达标              │
│       │             │       │                 │       │                 │
│       │             │◄──────┘                 │◄──────┘                 │
│       │             │                         │                         │
│       │             │ REPORT                  │                         │
│       ▼             │ 输出最终报告             │                         │
│  5. 验收            │                         │                         │
│     跑完整测试+审计  │                         │                         │
│     确认 commit     │                         │                         │
│                     │                         │                         │
└─────────────────────┴─────────────────────────┴─────────────────────────┘
```

**流程说明**：

1. **Agent 收集需求**：用户输入 Model Spec / Cluster Size / Target
2. **读 workflow Skill**：Agent 读取 `skills/workflow/SKILL.md` 了解整体流程
3. **收集项目信息**：Agent 调用 Tool 采集集群信息、模型配置
4. **进入 workflow**：
   - **PREFLIGHT**：采集硬件基线（ClusterProfile）
   - **PROJECTION**：单机 profiling + 建立 Execution Model + 生成初始 Plans
   - **进入 optimize-loop**：
     - BASELINE：跑基准测试，记录起点
     - 迭代循环：Observe → Diagnose → Re-Plan → Execute → Settle
     - 直到收敛或达标
   - **REPORT**：输出最终配置和报告
5. **验收**：跑完整测试，审计日志，确认结果

### 3.1 Tuning Loop 内部泳道（展开）

```
┌─────────────────────┬─────────────────────────┬─────────────────────────┐
│      Agent          │        Skill            │        Tool             │
│  (reasoning)        │   (knowledge)           │   (execution)           │
├─────────────────────┼─────────────────────────┼─────────────────────────┤
│                     │                         │                         │
│ [Execute]           │                         │                         │
│     │               │                         │                         │
│     ├── 读 execute.md ► 执行规则              │                         │
│     │               │   - early_stop 阈值     │                         │
│     │               │   - 串行/并行策略       │                         │
│     │               │                         │                         │
│     └── 调 Tool ───────────────────────────────► submit.run(plan)       │
│                     │                         │   └► 返回 run_id        │
│                     │                         │                         │
│ [Observe]           │                         │                         │
│     │               │                         │                         │
│     ├── 读 observe.md ► snapshot schema       │                         │
│     │               │   - tps, bubble_ratio   │                         │
│     │               │   - comm_ratio, mem     │                         │
│     │               │                         │                         │
│     └── 调 Tool ───────────────────────────────► observe.snapshot(run_id)
│                     │                         │   └► 返回 Snapshot      │
│                     │                         │                         │
│ [Diagnose]          │                         │                         │
│     │               │                         │                         │
│     ├── 读 diagnose.md ► 瓶颈分类规则         │                         │
│     │               │   if comm_ratio > X     │                         │
│     │               │     → COMM_BOUND        │                         │
│     │               │   if bubble > X         │                         │
│     │               │     → PIPELINE_BOUND    │                         │
│     │               │                         │                         │
│     └── Agent 推理得出 bottleneck             │                         │
│                     │                         │                         │
│ [Re-Plan]           │                         │                         │
│     │               │                         │                         │
│     ├── 读 optimization/{bottleneck}/ ────────┤                         │
│     │               │   返回优化策略:         │                         │
│     │               │   - 增大 TP             │                         │
│     │               │   - 调 bucket size      │                         │
│     │               │   - 开 overlap          │                         │
│     │               │                         │                         │
│     └── 调 Tool ───────────────────────────────► constraint.check(plan) │
│                     │                         │   └► 返回 valid/invalid │
│                     │                         │                         │
│ [Settle]            │                         │                         │
│     │               │                         │                         │
│     ├── 读 settle.md ► 收敛规则               │                         │
│     │               │   - greedy pick best    │                         │
│     │               │   - stop conditions     │                         │
│     │               │                         │                         │
│     └── Agent 决定: continue / stop           │                         │
│           │                                   │                         │
│           ├── continue → 回到 [Execute]       │                         │
│           └── stop → 输出 Final Config        │                         │
│                     │                         │                         │
└─────────────────────┴─────────────────────────┴─────────────────────────┘
```

---

## 4. Skill 目录结构

```
skills/                                 # 唯一知识源（Agent 读取）
│
├── workflow/                           # 调优主流程
│   ├── SKILL.md                        # tuning loop 总体说明
│   ├── projection.md                   # 建模阶段
│   ├── observe.md                      # 观测数据定义（snapshot schema）
│   ├── diagnose.md                     # 瓶颈分类逻辑
│   ├── plan.md                         # plan 结构定义
│   ├── execute.md                      # 执行与 early stop
│   └── settle.md                       # 收敛逻辑
│
├── execution-model/                    # 训练建模（核心知识）
│   ├── SKILL.md                        # 总体说明
│   ├── compute.md                      # T_comp(layers, mbs)
│   ├── memory.md                       # Mem(layers, mbs)
│   ├── communication.md                # T_comm / allreduce / alltoall
│   ├── pipeline.md                     # Bubble(pp, M)
│   ├── partition.md                    # layer partition / stage balance
│   └── examples.md                     # 建模示例（Dense / MoE）
│
├── optimization/                       # 按瓶颈类型组织的优化策略
│   ├── SKILL.md                        # 总体策略
│   │
│   ├── comm/                           # 通信瓶颈
│   │   ├── SKILL.md                    # reduce_comm_pressure
│   │   ├── bucket.md                   # bucket tuning
│   │   ├── overlap.md                  # overlap 优化
│   │   └── topology.md                 # 跨节点 vs 单节点
│   │
│   ├── pipeline/                       # pipeline 瓶颈
│   │   ├── SKILL.md                    # pipeline 优化策略
│   │   ├── vpp.md                      # VPP tuning
│   │   ├── microbatch.md               # MBS / GAS
│   │   └── balance.md                  # stage balance
│   │
│   ├── memory/                         # 显存瓶颈
│   │   ├── SKILL.md                    # memory 优化策略
│   │   ├── recompute.md                # activation recompute
│   │   ├── offload.md                  # CPU / NVMe offload
│   │   └── fragmentation.md            # 内存碎片
│   │
│   ├── compute/                        # 计算瓶颈
│   │   ├── SKILL.md                    # compute 利用率优化
│   │   ├── mbs.md                      # mbs scaling
│   │   ├── parallel.md                 # dp/tp 调整
│   │   └── kernel.md                   # kernel-level hint
│   │
│   └── moe/                            # MoE 专项
│       ├── SKILL.md
│       ├── routing.md
│       ├── dispatch.md
│       └── load_balance.md
│
├── profiling/                          # 数据采集方法
│   ├── SKILL.md
│   ├── preflight.md                    # cluster baseline
│   ├── gpu.md                          # GPU metrics
│   ├── network.md                      # IB / RCCL
│   └── trace.md                        # timeline 分析
│
├── constraints/                        # 安全约束
│   ├── SKILL.md
│   ├── oom.md                          # OOM 预估规则
│   ├── config.md                       # 配置合法性
│   └── validation.md                   # 验证方法
│
└── knowledge/                          # 经验沉淀
    ├── SKILL.md
    ├── patterns.md                     # 通用规律
    ├── cases.md                        # 历史案例
    └── anti-patterns.md                # 常见错误
```

**Skill 的作用**：Agent（Cursor / Claude）读取这些 Markdown 文件获取领域知识，而不是把知识硬编码在代码里。这样：
- 知识可以独立迭代，不需要改代码
- 不同的 Agent（Cursor / Claude / 其他）共用同一套知识
- 新人可以直接阅读 Skills 了解系统的调优逻辑

---

## 5. Tool 接口

Tools 是 Agent 通过 function calling 调用的 Python 函数：

| Tool | 功能 | 输入 | 输出 |
|------|------|------|------|
| `preflight.run()` | 采集集群硬件基线 | cluster_id | ClusterProfile |
| `profiler.run()` | 单机 profiling | model_spec, configs | ProfilingResult |
| `submit.run()` | 提交训练任务 | plan, cluster | job_id |
| `submit.cancel()` | 取消任务 | job_id | status |
| `observe.snapshot()` | 采集运行时数据 | job_id | Snapshot |
| `constraint.check()` | 验证配置合法性 | plan, cluster | valid, violations |
| `constraint.estimate_mem()` | 估算显存 | plan | mem_gb |

Agent 根据 Skill 中的知识决定"做什么"，然后调用 Tool 执行。

---

## 6. Execution Model（核心知识）

定义在 `skills/execution-model/*.md` 中，Agent 读取这些公式进行预估。

**Step Time 分解**：

```
T_step = T_comp + T_comm + T_bubble - T_overlap

T_comp = model_flops / (num_gpus × peak_tflops × efficiency)
  - efficiency 从 ClusterProfile + profiling 数据得到

T_comm = AllReduce(grad_size/dp) + AllToAll(moe_msg) + AllGather(zero_shard)
  - bandwidth 从 ClusterProfile.rccl_baseline 查表

T_bubble = (pp-1) / (pp-1 + M) × T_comp   # M = num_microbatch

T_overlap = min(T_comm_overlappable, T_comp_spare)
```

**显存估算**：

```
Mem = M_param + M_grad + M_optim + M_act + M_buffer

M_param = params / (tp × pp) × bytes_per_param
M_act   = f(seq, hidden, mbs, layers/pp, recompute)
```

---

## 7. 收敛性保证

| 机制 | 作用 |
|------|------|
| **Execution Model 筛选** | Plan 进入 Execute 前已经过模型预估，质量高 |
| **baseline 单调递增** | `baseline[r+1] ≥ baseline[r]`，不会退化 |
| **history 去重** | Re-Plan 排除已尝试的配置，搜索空间逐轮缩小 |
| **三重终止条件** | 目标达成 / gain < 2% 连续两轮 / max_rounds |

**成本控制**：

| 阶段 | 成本 |
|------|------|
| Preflight | ~30 min（一次性） |
| Projection | ~1 GPU·h（单机） |
| Tuning Loop | ~3-5 GPU·h |
| **总计** | ~5-7 GPU·h |

---

## 8. Guardrails

| 机制 | 位置 | 作用 |
|------|------|------|
| OOM 预估 | `constraint.estimate_mem()` | 自动过滤高风险配置 |
| early stop | `observe.snapshot()` | OOM / 吞吐退化 → 立即终止 |
| history 去重 | `knowledge.search()` | 不重复尝试已失败配置 |
| max_rounds | Settle 逻辑 | 总成本硬约束 |
| 审计日志 | 全流程 | 每步决策完整记录 |

---

## 9. 一句话总结

Primus Pilot = **Agent**（Cursor / Claude 承载推理决策）+ **Skills**（Execution Model / 优化策略 / 诊断规则）+ **Tools**（执行层 Python 函数）—— Agent 读取 Skills 获取知识，调用 Tools 完成 Preflight → Projection → Tuning Loop 的闭环。
