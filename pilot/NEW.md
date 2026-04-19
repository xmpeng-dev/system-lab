


```

                         ┌──────────────────────────────┐
                         │        User Input             │
                         │  - Model Spec                │
                         │  - Cluster Size             │
                         │  - Target (TPS / Cost)      │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
┌──────────────────────────────────────────────────────────────────────┐
│                           1. Preflight++                             │
│                                                                      │
│  Collect:                                                            │
│    - GEMM / MFMA peak                                                │
│    - IB / XGMI bandwidth                                             │
│    - AllReduce / All2All baseline                                    │
│                                                                      │
│  Output:                                                             │
│    ClusterProfile = {                                                │
│        compute_peak, comm_bw, latency, overlap_capability            │
│    }                                                                 │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      2. Projection / Modeling                        │
│                                                                      │
│  Input: Model Spec + ClusterProfile                                  │
│                                                                      │
│  Step 1: Single-node profiling                                       │
│    (layers, mbs, recompute) →                                        │
│        T_comp, Mem_peak                                              │
│                                                                      │
│  Step 2: Build Execution Model                                       │
│    T_comp(l, mbs)                                                    │
│    Mem(l, mbs)                                                       │
│    T_pp_comm(mbs)                                                    │
│    Bubble(P, M)                                                      │
│                                                                      │
│  Step 3: Generate Initial Plans                                      │
│                                                                      │
│    Plan = {                                                          │
│      parallel: (dp,tp,pp,ep)                                         │
│      partition: [l1, l2, ...]                                        │
│      mbs                                                             │
│      recompute                                                       │
│      expected: {tps, bottleneck}                                     │
│    }                                                                 │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
                ┌────────────────────────────────────┐
                │         Tuning Loop (核心)          │
                └────────────────────────────────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         ▼                     ▼                     ▼
┌─────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│   3. Execute    │  │    4. Observe    │  │   5. Diagnose     │
└─────────────────┘  └──────────────────┘  └──────────────────┘

Execute:
  - 提交训练任务
  - 多 plan 并行 / 串行执行
  - early stop

Observe:
  - TPS
  - stage time
  - bubble ratio
  - comm ratio
  - overlap ratio
  - peak memory

  输出：
    Snapshot

Diagnose:
  基于 execution model 判断：

    if comm_ratio > X → COMM_BOUND
    if bubble > X     → PIPELINE_BOUND
    if mem > X        → MEMORY_BOUND
    else              → COMPUTE_BOUND

                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         6. Re-Plan (Skill Engine)                    │
│                                                                      │
│  Input: Diagnosis + History + ExecutionModel                         │
│                                                                      │
│  Skill Mapping:                                                      │
│    COMM      → reduce_comm / overlap / bucket                        │
│    PIPELINE  → rebalance / VPP / microbatch                          │
│    MEMORY    → recompute / reduce layers / offload                   │
│    COMPUTE   → increase mbs / better parallel                        │
│                                                                      │
│  Generate New Plans:                                                 │
│    apply(skill, old_plan)                                            │
│                                                                      │
│  + Constraint Check (OOM / invalid config)                           │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        7. Settle / Convergence                       │
│                                                                      │
│  - 选 best plan                                                      │
│  - baseline 单调递增                                                 │
│  - 去重（避免重复搜索）                                              │
│                                                                      │
│  Stop Conditions:                                                    │
│    - 达到 target                                                     │
│    - gain < 2% (连续两轮)                                            │
│    - max rounds                                                      │
│                                                                      │
│  输出：Final Best Config                                             │
└──────────────────────────────────────────────────────────────────────┘

```

```

Primus/
│
├── agent/                                  # Agent 相关
│   └── skills/                             # 唯一知识源（Claude 读取）
│       │
│       ├── workflow/                       # 调优主流程
│       │   ├── SKILL.md                    # tuning loop 总体说明
│       │   ├── projection.md               # 建模阶段（Execution Model）
│       │   ├── observe.md                  # 观测数据定义（snapshot schema）
│       │   ├── diagnose.md                 # 瓶颈分类逻辑
│       │   ├── plan.md                     # plan 结构定义
│       │   ├── execute.md                  # 执行与 early stop
│       │   └── settle.md                   # 收敛逻辑
│       │
│       ├── execution-model/                # 训练建模
│       │   ├── SKILL.md                    # 总体说明（不是 tool，是知识）
│       │   │
│       │   ├── compute.md                  # T_comp(l, mbs)
│       │   ├── memory.md                   # Mem(l, mbs)
│       │   ├── communication.md            # T_comm / allreduce / alltoall
│       │   ├── pipeline.md                 # Bubble(P, M)
│       │   ├── partition.md                # layer partition / stage balance
│       │   │
│       │   └── examples.md                 # 建模示例（Dense / MoE）
│       │
│       ├── optimization/                   # 机制级优化
│       │   ├── SKILL.md                    # 总体策略（机制驱动）
│       │   │
│       │   ├── comm/                       # 通信瓶颈
│       │   │   ├── SKILL.md                # reduce_comm_pressure
│       │   │   ├── bucket.md               # bucket tuning
│       │   │   ├── overlap.md              # overlap 优化
│       │   │   └── topology.md             # 跨节点 vs 单节点
│       │   │
│       │   ├── pipeline/                   # pipeline 瓶颈
│       │   │   ├── SKILL.md                # pipeline 优化策略
│       │   │   ├── vpp.md                  # VPP tuning
│       │   │   ├── microbatch.md           # MBS / GAS
│       │   │   └── balance.md              # stage balance
│       │   │
│       │   ├── memory/                     # 显存瓶颈
│       │   │   ├── SKILL.md                # memory 优化策略
│       │   │   ├── recompute.md            # activation recompute
│       │   │   ├── offload.md              # CPU / NVMe offload
│       │   │   └── fragmentation.md        # 内存碎片
│       │   │
│       │   ├── compute/                    # 计算瓶颈
│       │   │   ├── SKILL.md                # compute 利用率优化
│       │   │   ├── mbs.md                  # mbs scaling
│       │   │   ├── parallel.md             # dp/tp 调整
│       │   │   └── kernel.md               # kernel-level hint（可选）
│       │   │
│       │   └── moe/                        # MoE 专项
│       │       ├── SKILL.md
│       │       ├── routing.md
│       │       ├── dispatch.md
│       │       └── load_balance.md
│       │
│       ├── profiling/                      # [数据来源]
│       │   ├── SKILL.md
│       │   ├── preflight.md                # cluster baseline
│       │   ├── gpu.md                      # GPU metrics
│       │   ├── network.md                  # IB / RCCL
│       │   └── trace.md                    # timeline 分析
│       │
│       ├── constraints/                    # [安全]
│       │   ├── SKILL.md
│       │   ├── oom.md
│       │   ├── config.md
│       │   └── validation.md
│       │
│       └── knowledge/                     # [经验沉淀]
│           ├── SKILL.md
│           ├── patterns.md                # 通用规律
│           ├── cases.md                   # 历史案例
│           └── anti-patterns.md           # 常见错误
│
└── (Primus 项目源码...)
```