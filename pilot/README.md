# Primus Pilot — 训练调优系统

> 面向多节点训练集群的自动调优系统。  
> Agent（Cursor / Claude）读取 Skills 中的知识，调用 Tools 执行操作，完成建模→搜索→收敛的闭环。

---

## 目录

1. [问题与边界](#1-问题与边界)
2. [系统架构](#2-系统架构)
3. [系统流程（泳道图）](#3-系统流程泳道图)
4. [Skill 目录结构](#4-skill-目录结构)
5. [Tool 接口](#5-tool-接口)
6. [Execution Model（核心知识）](#6-execution-model核心知识)
7. [收敛性保证](#7-收敛性保证)
8. [数据结构（Schema）](#8-数据结构schema)
9. [完整迭代示例](#9-完整迭代示例)
10. [评估指标](#10-评估指标)
11. [与现有系统的集成](#11-与现有系统的集成)
12. [Guardrails](#12-guardrails)
13. [一句话总结](#13-一句话总结)

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
│       │             │ 硬件+env baseline       │                         │
│       │             │ (按 version/age 复用)   │                         │
│       │             │       │                 │                         │
│       │             │       ▼                 │                         │
│       │             │ PROJECTION              │                         │
│       │             │ 建立 Execution Model    │                         │
│       │             │       │                 │                         │
│       │             │       ▼                 │                         │
│       │             │ SMOKE                   │                         │
│       │             │ tiny scale × 100 step   │                         │
│       │             │ 验证可起 / 无 hang      │                         │
│       │             │       │                 │                         │
│       │             │       ▼                 │                         │
│       │             │ 进 optimize-loop ──────►│ BASELINE                │
│       │             │                         │ 跑 test + benchmark     │
│       │             │                         │ 记录起点                │
│       │             │                         │       │                 │
│       │             │                         │       ▼                 │
│       │             │                         │ CORRECTNESS             │
│       │             │                         │ loss/grad vs reference  │
│       │             │                         │ (失败 → ABORT)          │
│       │             │                         │       │                 │
│       │             │                         │       ▼                 │
│       │             │                         │ 外层结构循环             │
│       │             │                         │ (Observe→Diagnose→      │
│       │             │                         │  Re-Plan→Execute→       │
│       │             │                         │  CORRECTNESS-LITE→      │
│       │             │                         │  Settle)                │
│       │             │                         │       │                 │
│       │             │                         │       ▼                 │
│       │             │                         │ 内层 EnvSweep（可选）   │
│       │             │                         │ (锁结构、扫 env diff)   │
│       │             │                         │       │                 │
│       │             │                         │       ▼                 │
│       │             │                         │ 收敛 / 达标              │
│       │             │       │                 │       │                 │
│       │             │◄──────┘                 │◄──────┘                 │
│       │             │                         │                         │
│       │             │ 反向跳转（reentry）：    │                         │
│       │             │  ClusterProfile 过期     │                         │
│       │             │   → 回 PREFLIGHT        │                         │
│       │             │  Plan 结构性失效         │                         │
│       │             │   → 回 PROJECTION       │                         │
│       │             │                         │                         │
│       │             │ REPORT                  │                         │
│       │             │ 输出最终配置 + trace    │                         │
│       │             │       │                 │                         │
│       │             │       ▼                 │                         │
│       │             │ LEARN                   │                         │
│       ▼             │ best/失败 case 回写     │                         │
│  5. 验收            │ skills/knowledge/       │                         │
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
   - **PREFLIGHT**：采集硬件基线 + 探测集群级 env baseline（ClusterProfile）
   - **PROJECTION**：单机 profiling + 建立 Execution Model + 生成初始 Plans（含 scale-aware env diff）
   - **进入 optimize-loop（双层）**：
     - BASELINE：跑基准测试，记录起点
     - **外层（结构）**：Observe → Diagnose → Re-Plan → Execute → Settle
     - **内层（EnvSweep，可选）**：在 Settle 后、进入下一外层迭代前触发；锁定结构，仅扫 env diff
     - 直到收敛或达标
   - **REPORT**：输出最终配置和报告
5. **验收**：跑完整测试，审计日志，确认结果

> **为什么外层/内层分开**：env 调参不改变形状（无 OOM 风险）、单次成本低（30-50 step 即可分辨），适合在每个外层 baseline 稳定后做一次"安全 sweep"；env 的最优值依赖结构（mbs / world_size），所以**不能一次跑完一劳永逸**，必须嵌在外层每轮里。

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
│ [EnvSweep]          │                         │                         │
│  (条件触发：         │                         │                         │
│   bottleneck 命中    │                         │                         │
│   且结构稳定)        │                         │                         │
│     │               │                         │                         │
│     ├── 读 env/SKILL.md & env_probe.md ──────┤                          │
│     │               │   - 候选 flag 列表      │                         │
│     │               │   - 安全 sweep 协议     │                         │
│     │               │                         │                         │
│     ├── 读 optimization/{bottleneck}/env.md   │                         │
│     │               │   - 该瓶颈下相关 flag   │                         │
│     │               │                         │                         │
│     ├── 调 Tool ───────────────────────────────► constraint.check_env() │
│     │               │                         │   └► 拒绝危险组合       │
│     │               │                         │                         │
│     └── 调 Tool ───────────────────────────────► submit.run(plans, short)
│                     │                         │   └► 30-50 step 并行    │
│                     │                         │      返回 best env diff │
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
│   ├── SKILL.md                        # tuning loop 总体说明（含外层/内层切换条件）
│   ├── projection.md                   # 建模阶段
│   ├── observe.md                      # 观测数据定义（snapshot schema）
│   ├── diagnose.md                     # 瓶颈分类逻辑（含 env_suspect 输出）
│   ├── plan.md                         # plan 结构定义（含 env.diff）
│   ├── execute.md                      # 执行与 early stop
│   ├── envsweep.md                     # 内层 EnvSweep 协议（触发条件 / 候选 / 收敛）
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
│   │   ├── topology.md                 # 跨节点 vs 单节点
│   │   └── env.md                      # COMM_BOUND 候选 env（→ env/rccl.md）
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
│   │   ├── fragmentation.md            # 内存碎片
│   │   └── env.md                      # MEMORY_BOUND 候选 env（→ env/alloc.md）
│   │
│   ├── compute/                        # 计算瓶颈
│   │   ├── SKILL.md                    # compute 利用率优化
│   │   ├── mbs.md                      # mbs scaling
│   │   ├── parallel.md                 # dp/tp 调整
│   │   ├── kernel.md                   # kernel-level hint
│   │   └── env.md                      # COMPUTE_BOUND 候选 env（→ env/threading.md, hsa.md）
│   │
│   └── moe/                            # MoE 专项
│       ├── SKILL.md
│       ├── routing.md
│       ├── dispatch.md
│       └── load_balance.md
│
├── env/                                # env 调参 catalog（事实源 / 单点维护）
│   ├── SKILL.md                        # env 调参总论 / 双层循环触发条件
│   ├── rccl.md                         # NCCL_*/RCCL_* 全集（默认 / 范围 / 已知坑）
│   ├── hsa.md                          # HSA_*/HIP_*/GPU_MAX_HW_QUEUES
│   ├── alloc.md                        # PYTORCH_HIP_ALLOC_CONF / MALLOC_*
│   ├── threading.md                    # OMP_*/MKL_*/numactl
│   └── presets.md                      # per-cluster-type 已验证预设组合
│
├── profiling/                          # 数据采集方法
│   ├── SKILL.md
│   ├── preflight.md                    # cluster baseline
│   ├── gpu.md                          # GPU metrics
│   ├── network.md                      # IB / RCCL
│   ├── trace.md                        # timeline 分析
│   └── env_probe.md                    # env 安全探测协议（连通性 → micro-bench → 多节点）
│
└── constraints/                        # 安全约束
    ├── SKILL.md
    ├── oom.md                          # OOM 预估规则
    ├── config.md                       # 配置合法性
    ├── validation.md                   # 验证方法
    └── env.md                          # env 不兼容矩阵（互斥 / 危险组合）
```

**env 知识的组织原则**：
- `skills/env/*.md` 是**唯一 catalog**（每个 flag 只在这里完整定义一次）
- `skills/optimization/{bottleneck}/env.md` 只列**「该瓶颈下应优先看哪些 flag」**，引用 catalog
- 这样新增 flag 只改 catalog 一处，避免知识散落

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
| `env_probe.run()` | 探测/校验集群级 env baseline（连通性 + RCCL micro-bench） | cluster_id, candidate_envs | EnvBaseline (写入 ClusterProfile) |
| `env_probe.sweep()` | 内层 EnvSweep：固定结构，扫描 env diff | base_plan, candidate_envs, max_steps | best_env_diff, results |
| `profiler.run()` | 单机 profiling | model_spec, configs | ProfilingResult |
| `submit.run()` | 提交训练任务 | plan, cluster | job_id |
| `submit.cancel()` | 取消任务 | job_id | status |
| `observe.snapshot()` | 采集运行时数据 | job_id | Snapshot |
| `constraint.check()` | 验证配置合法性 | plan, cluster | valid, violations |
| `constraint.check_env()` | 验证 env 组合（互斥 / 危险） | env_diff, baseline | valid, violations |
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

## 8. 数据结构（Schema）

Agent 与 Tool 之间通过结构化数据交换。下面是几个核心 schema。

### 8.1 ClusterProfile（Preflight 输出）

```yaml
cluster_id: mi300x-16node
collected_at: 2026-04-15T10:00:00Z
nodes: 16
gpus_per_node: 8

compute:
  peak_tflops_bf16: 1300         # 实测 GEMM peak
  peak_tflops_fp8: 2600
  hbm_bandwidth_gbs: 5300
  hbm_capacity_gb: 192

interconnect:
  intra_node:
    type: xgmi
    bandwidth_gbs: 800
  inter_node:
    type: ib
    bandwidth_gbs: 400            # 单卡有效带宽

rccl_baseline:
  allreduce:
    - {size_mb: 1,    bw_gbs: 12}
    - {size_mb: 256,  bw_gbs: 180}
  alltoall:
    - {size_mb: 1,    bw_gbs: 8}
    - {size_mb: 256,  bw_gbs: 95}

env_baseline:                      # 集群级 env 黄金默认（一次探测 / 多任务复用）
  version: mi300x-16node-v3
  status: validated                # validated / tentative
  rccl:
    NCCL_IB_HCA: "mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1"
    NCCL_NET_GDR_LEVEL: 4
    NCCL_IB_GID_INDEX: 3
    NCCL_SOCKET_IFNAME: bond0
  hsa:
    HSA_FORCE_FINE_GRAIN_PCIE: 1
    GPU_MAX_HW_QUEUES: 2
  alloc:
    PYTORCH_HIP_ALLOC_CONF: "expandable_segments:True"
  threading:
    OMP_NUM_THREADS: 8
```

### 8.2 Plan（待执行的配置）

```yaml
plan_id: r3_p2
parent_baseline: r2_p1            # 上一轮的最优 plan
parallelism:
  tp: 4
  pp: 2
  dp: 16
  ep: 8
  vpp: 2
runtime:
  mbs: 2
  gbs: 1024
  recompute: selective
comm:
  bucket_size_mb: 64
  overlap: true
env:                               # 仅记录相对 env_baseline 的差异
  baseline_ref: mi300x-16node-v3
  diff:
    NCCL_MIN_NCHANNELS: 16
    NCCL_BUFFSIZE: 16777216
    PYTORCH_HIP_ALLOC_CONF: "expandable_segments:True,max_split_size_mb:512"
    RCCL_MSCCL_ENABLE: 1
predicted:                         # Execution Model 预估值
  tps: 18500
  mem_peak_gb: 165
  comm_ratio: 0.22
generated_by:
  bottleneck: PIPELINE_BOUND
  strategy: skills/optimization/pipeline/vpp.md
```

### 8.3 Snapshot（Observe 输出）

```yaml
run_id: job_8842
plan_id: r3_p2
collected_at: 2026-04-15T11:23:00Z
metrics:
  tps: 17800
  step_time_ms: 412
  comm_ratio: 0.18
  bubble_ratio: 0.09
  overlap_ratio: 0.61
  mem_peak_gb: 158
  gpu_util_avg: 0.74
status: completed                  # / early_stopped / oom / failed
warnings: []
```

### 8.4 DiagnosisReport（Diagnose 输出）

```yaml
snapshot_id: job_8842
bottleneck: COMPUTE_BOUND          # COMM / PIPELINE / MEMORY / COMPUTE
confidence: 0.85
evidence:
  - "comm_ratio=0.18 < threshold 0.25"
  - "bubble_ratio=0.09 < threshold 0.15"
  - "gpu_util=0.74 vs baseline peak 0.92 → headroom"
recommended_skills:
  - skills/optimization/compute/mbs.md
  - skills/optimization/compute/parallel.md
env_suspect:                       # 若有 → 触发 EnvSweep（先于结构调整）
  - flag: NCCL_BUFFSIZE
    reason: "comm_ratio=0.32 但 msg_size_p95=4MB，buffer 偏小"
    hint: skills/env/rccl.md#buffsize
  - flag: PYTORCH_HIP_ALLOC_CONF
    reason: "mem_reserved/mem_alloc=1.45 → 碎片偏高"
    hint: skills/env/alloc.md#expandable-segments
```

### 8.5 EnvSweepResult（内层循环输出）

```yaml
sweep_id: r3_envsweep_1
parent_plan: r3_p2                 # 锁定的结构 baseline
trigger: COMM_BOUND
candidates:                        # 本轮试过的 env 组合
  - env_diff: {NCCL_BUFFSIZE: 8388608}
    tps: 17900
    delta_pct: +0.6
  - env_diff: {NCCL_BUFFSIZE: 16777216, NCCL_MIN_NCHANNELS: 16}
    tps: 18650
    delta_pct: +4.8                # ← best
  - env_diff: {NCCL_BUFFSIZE: 33554432}
    tps: 17600
    delta_pct: -1.1
selected_diff:                     # 合并进 baseline.env.diff
  NCCL_BUFFSIZE: 16777216
  NCCL_MIN_NCHANNELS: 16
cost_gpu_h: 0.3
```

---

## 9. 完整迭代示例

一次 Tuning Loop 的真实数据流（以 MoE 16 节点为例）：

```
Round 0 (Baseline)
  Plan:     {tp:2, pp:4, ep:8, mbs:1, recompute:full}
  Snapshot: tps=12000, comm_ratio=0.38, bubble=0.12, mem=140GB
  Diagnose: COMM_BOUND (alltoall 占 28%)
  Re-Plan:  读 skills/optimization/comm/ → 生成 3 候选
            P1: bucket_size 16→64 MB
            P2: 启用 alltoall overlap
            P3: ep 8→4（减少跨节点通信）

Round 1 (外层)
  Execute:  3 个 plan 并行跑（每个 50 step early-stop）
  Results:
    P1: tps=14200 (+18%)
    P2: tps=15800 (+32%)  ← best
    P3: tps=13100 (+9%, ep 减小导致 expert 容量下降)
  Settle:   选 P2 作为新 baseline

Round 1' (内层 EnvSweep，因 Diagnose 输出 env_suspect=NCCL_BUFFSIZE)
  Sweep:    锁定 P2 结构，扫 3 个候选（30 step）
            E1: NCCL_BUFFSIZE=8M             → tps=15900 (+0.6%)
            E2: NCCL_BUFFSIZE=16M+MIN_NCH=16 → tps=16550 (+4.7%)  ← best
            E3: NCCL_BUFFSIZE=32M            → tps=15600 (-1.3%)
  Merge:    P2.env.diff += {NCCL_BUFFSIZE:16M, NCCL_MIN_NCHANNELS:16}
            new baseline tps=16550
  Cost:     0.3 GPU·h
  Diagnose: PIPELINE_BOUND (bubble 升至 0.18)
  Re-Plan:  读 skills/optimization/pipeline/ → 生成 2 候选
            P4: vpp 1→2
            P5: mbs 1→2

Round 2
  Execute:  P4, P5
  Results:
    P4: tps=17600 (+11%)  ← best
    P5: OOM → early stop
  Settle:   选 P4，gain=11% > 2%，继续
  Diagnose: COMPUTE_BOUND
  Re-Plan:  ...

Round 3
  gain < 2% (连续两轮) → STOP
  Final: tps=18200 (1.52× over baseline)
  耗时:  ~4 GPU·h (Tuning Loop 部分)
```

---

## 10. 评估指标

如何判断 Pilot 系统本身好不好用：

| 维度 | 指标 | 目标 |
|------|------|------|
| **效果** | 最终 TPS / baseline TPS | ≥ 1.3× |
| **效果** | 与人工 best-known 配置差距 | ≥ 90% |
| **效率** | 总 GPU·h 成本 | ≤ 10 GPU·h |
| **效率** | 收敛轮数 | ≤ 5 轮 |
| **可靠性** | 成功率（无 OOM / 任务挂掉） | ≥ 95% |
| **可靠性** | Cost Model 预估误差 | ≤ 20% |
| **可解释性** | 每步决策可追溯到 Skill | 100% |

**评估方法**：
1. **回归测试集**：维护 5-10 个典型场景（Dense 7B/70B、MoE 8x7B 等），每次系统升级后自动跑一遍
2. **对照实验**：同一模型 + 集群，分别由 Pilot 和资深工程师调优，对比结果
3. **消融实验**：关掉 Execution Model / 关掉 Cost Model 筛选，看 Tuning Loop 退化多少

---

## 11. 与现有系统的集成

Pilot 不重新发明轮子，而是组合现有能力：

| 现有系统 | 集成方式 |
|---------|---------|
| **Primus** | `submit.run()` 调 Primus CLI 提交任务；Preflight 复用 Primus 的硬件检测 |
| **Megatron / TorchTitan** | 通过 Tuning IR 生成各 backend 的 config，Agent 不需要感知具体框架 |
| **WandB / Prometheus** | `observe.snapshot()` 从 WandB API 拉取 metrics |
| **rocprof / RCCL profiler** | `profiler.run()` 包装 rocprof 命令，输出结构化 trace |
| **Slurm** | `submit.run()` 内部生成 sbatch 脚本，复用现有调度 |

这种集成方式的好处：**Pilot 死掉后，所有底层工具仍然可用**；现有工程师的 muscle memory 不被破坏。

---

## 12. Guardrails

| 机制 | 位置 | 作用 |
|------|------|------|
| OOM 预估 | `constraint.estimate_mem()` | 自动过滤高风险配置 |
| early stop | `observe.snapshot()` | OOM / 吞吐退化 → 立即终止 |
| history 去重 | Settle 逻辑 | 不重复尝试已失败配置 |
| max_rounds | Settle 逻辑 | 总成本硬约束 |
| env 连通性 fail-fast | `env_probe.run()` | 错误 NCCL_IB_HCA 等 30s 内炸掉，不进 baseline |
| env 不兼容矩阵 | `constraint.check_env()` | 拒绝危险/互斥组合（如 MSCCL × 某些 GDR 模式） |
| EnvSweep 单次 cap | inner loop | 每次 ≤ 5 flag、≤ 8 组合、≤ 50 step |
| env baseline 版本化 | `ClusterProfile.env_baseline.version` | 集群升级 / 驱动变更触发重新探测 |
| env diff-only 记录 | `Plan.env.diff` | 审计、复现、回滚都基于 diff |
| 审计日志 | 全流程 | 每步决策完整记录，可回放 |

---

## 13. 一句话总结

Primus Pilot = **Agent**（Cursor / Claude 承载推理决策）+ **Skills**（Execution Model / 优化策略 / 诊断规则）+ **Tools**（执行层 Python 函数）—— Agent 读取 Skills 获取知识，调用 Tools 完成 Preflight → Projection → Tuning Loop 的闭环。
