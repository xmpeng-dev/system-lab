# Axion 研究路线计划

> **目标定位:** 顶会论文（MLSys / OSDI / SOSP / EuroSys / ATC）  
> **核心策略:** 分阶段发表，每个 Phase 独立成篇，不等全部完成  
> **总体时间:** 18~24 个月，产出 3~4 篇论文  
> **团队假设:** 1~2 名研究生 + 1 名有系统背景的导师/合作者

---

## 0. 研究路线的核心判断

### 0.1 为什么这个方向值得发论文

```
技术新颖性（三点组合，目前市场空白）：

  ① CommTensor 类型系统
     把"通信正确性"从运行时检查提升到编译期保证
     现有所有 MoE 训练框架均无此设计
     → 系统方向的类型安全创新

  ② Compile-First MoE 通信优化
     OverlapInsertionPass 静态分析通信/计算依赖
     生成 StaticSchedule，运行时零决策开销
     → 对比 LAER-MoE 的运行时调度，有明确的技术差异

  ③ Inspectable 分布式训练
     SHA-256 hash 追踪 + Compile Report
     目前没有框架认真解决分布式系统的可调试性
     → 填补工程工具领域空白
```

### 0.2 分阶段发表策略

```
不要等 18 个月做完再发一篇大论文。

正确策略：每个 Phase 都有独立的论文贡献点

  Paper 1 → Phase 1 完成后投（约第 3 个月）
  Paper 2 → Phase 2/3 完成后投（约第 8 个月）
  Paper 3 → Phase 4 完成后投（约第 15 个月）
  Paper 4 → 完整系统评估（约第 20 个月）

每篇论文之间有依赖关系，但每篇都能独立审稿。
```

---

## 1. Phase 0：研究基础设施（第 1~4 周）

### 目标
建立 ModelGraph + PassManager + AnalysisPass + FusionPass，  
在单机跑通 Llama 3.1 8B，建立计算性能基线。

### 具体任务

```
Week 1-2：ModelGraph 核心数据结构
  □ Wire、OpNode、ModelGraph（frozen dataclass）
  □ SHA-256 hash 机制
  □ OpSpec 类型层次（MatMulSpec、AttentionSpec、RMSNormSpec、SwiGLUSpec）
  □ 单元测试：构造图 → hash → 修改图 → hash 变化

Week 3-4：Pass 系统 + 单机验证
  □ Pass 基类、PassManager、CompileReport
  □ AnalysisPass（FLOPs / params 静态估算）
  □ FusionPass（RMSNorm+QKV+RoPE、SwiGLU 融合）
  □ Llama 3.1 8B 模型图构建
  □ Attention backends：sdpa / reference（flash 后续）
```

### 验收标准
```
□ Llama 3.1 8B 单机前向跑通
□ Compile Report 正确输出 FLOPs、params、fusion 数量
□ Pass hash 追踪正确（fusion 前后 hash 不同）
□ 对齐参考实现的计算结果（数值误差 < 1e-3）
```

---

## 2. Phase 1：Paper 1 ——通信感知编译器（第 1~3 个月）

### 论文定位

```
标题方向：
  "Towards Inspectable Distributed Training: 
   A Communication-Aware Compiler for MoE Systems"

投稿目标：MLSys 2027 或 EuroSys 2027（Deadline 约第 4 个月）

核心贡献：
  1. CommTensor 类型系统设计（CommLayout 枚举 + 状态机）
  2. CommInferencePass：静态通信需求分析
  3. CommTensorLayoutPass：编译期 layout 决策消除 pack/unpack
  4. Simulate Driver：无 GPU 验证分布式编译逻辑
  5. Compile Report 通信维度扩展

与现有工作的差异点：
  vs FX Graph：类型系统原生支持通信，不是黑盒
  vs Megatron：通信决策编译期确定，可追踪
  vs LAER-MoE：增加可检查性（hash + report），不只是优化
```

### 具体任务

```
Month 1：CommTensor + CommInferencePass
  □ CommLayout 枚举（5种：BLOCKED_BY_DST/SRC/EXPERT, INTERLEAVED, SPARSE_CSR）
  □ CommTensor 数据结构（physical_data + index map + comm_spec）
  □ WireType 扩展（COMM, EXPERT_SHARD, ROUTING_TABLE）
  □ CommOpSpec 类型层次（A2ASpec, AllGatherSpec, ReduceScatterSpec）
  □ CommInferencePass：ExpertGateSpec/FFNSpec → 自动标注 A2ASpec
  □ 单元测试：手写 MoE layer IR → 验证 CommAnnotation 正确

Month 2：CommTensorLayoutPass + Simulate Driver
  □ CommTensorLayoutPass：layout_hint → physical_layout 决策
  □ Layout 状态机：验证转换路径合法性（编译期类型检查）
  □ CommFabric 接口定义
  □ Simulate Driver（no-op 实现）
  □ 扩展 Compile Report：通信拓扑、layout 分配、CommLayout 分布

Month 3：实验 + 写作
  □ 实验 1：CommTensor vs 传统 tensor 的 pack/unpack 开销对比
    （在单机上模拟，用 torch.index_select vs 内存 copy 对比）
  □ 实验 2：Compile Report 的可用性评估（用户研究或 case study）
  □ 实验 3：类型系统捕获错误的案例（人工注入 layout 错误 → 编译期报错）
  □ 论文写作（System Design + Evaluation）
```

### 关键实验设计

```
Experiment 1：pack/unpack 消除的实际收益
  设置：模拟 MoE dispatch/combine，对比
    Baseline：标准 pack + all_to_all + unpack
    Axion：CommTensor 直接 DMA
  指标：内存 copy 时间、端到端 A2A latency
  预期：在大 seq_len 下有显著收益（seq=8192 时约 10~15% A2A 时间）

Experiment 2：编译期错误检测
  设置：人工注入 7 类常见分布式 bug（layout 错误、group 错误等）
  对比：
    Axion：编译期报错，精确定位到节点
    传统框架：运行时崩溃或静默错误
  指标：错误检测率、错误定位精确性

Experiment 3：Compile Report 的可用性
  设置：给 5 名研究生用 Axion 和 Megatron 各 debug 一个通信问题
  指标：debug 时间、成功率
```

### 验收标准
```
□ CommInferencePass 正确标注 DSv3-like MoE 模型的所有通信点
□ Layout 状态机编译期捕获所有注入的 layout 错误
□ Compile Report 正确显示通信拓扑图
□ Simulate Driver 在单机跑通完整编译流程（无 GPU 集群）
□ Paper 1 草稿完成，提交给导师 review
```

---

## 3. Phase 2：Paper 2 ——静态 Overlap 调度（第 4~8 个月）

### 论文定位

```
标题方向：
  "Static Communication-Computation Overlap Scheduling 
   for Large-Scale MoE Training"

投稿目标：OSDI 2027 或 MLSys 2027

核心贡献：
  1. FSEPShardingPass：基于通信图的 Expert 初始分片优化
  2. OverlapInsertionPass：静态依赖分析 + Sched.Overlap 插入
  3. StaticSchedule：编译期确定执行时间线
  4. 与 LAER-MoE（运行时调度）的对比实验

与 LAER-MoE 的关键差异：
  LAER-MoE：运行时 Slow Planner 求解优化问题 → 调度开销
  Axion：编译期 FSEPShardingPass → 初始分布最优，运行时零开销
  
  LAER-MoE：启发式 overlap（不保证正确性）
  Axion：静态依赖分析 → 数学证明正确的 overlap 集合
```

### 具体任务

```
Month 4：FSEPShardingPass
  □ CommGraph 构建（从 CommAnnotation 构建通信依赖有向图）
  □ 贪心 Expert 初始分配算法
    目标：minimize max_gpu(compute + comm_time)
    约束：内存均衡、跨节点通信最小化
  □ ExpertShardRegistry（Expert 分片注册表）
  □ 集成 ClusterTopologySpec（节点内/跨节点带宽感知）
  □ 2~4 GPU 验证 Expert dispatch/combine 正确性

Month 5：OverlapInsertionPass + StaticSchedule
  □ 数据依赖分析算法（Compute Op ↔ Comm Op 依赖图）
  □ OverlapInsertionPass：插入 OverlapSpec 节点
  □ StaticSchedule 数据结构（步骤序列 + overlap_pairs）
  □ StaticSchedule 执行引擎（CUDA Stream 绑定）
  □ 正确性验证：overlap 执行结果 == 非 overlap 执行结果

Month 6：CommFabric NVLink 实现 + 集成测试
  □ CommFabric NVLink Driver（基于 NCCL All-to-All）
  □ DistributedExecutablePlan（含 comm_steps + static_schedule）
  □ 8 GPU 端到端训练（MoE 模型，1~2 层）
  □ Slow Planner 基础版（只做初始分配，不做迁移）

Month 7-8：实验 + 写作
  □ 实验 1：StaticSchedule vs 动态调度的调度开销对比
  □ 实验 2：OverlapInsertionPass 的 overlap 率 vs LAER-MoE
  □ 实验 3：FSEPShardingPass 初始分配质量 vs Round-Robin
  □ 消融实验：关闭各个 Pass 逐步测量收益
  □ 论文写作
```

### 关键实验设计

```
Experiment 1：静态 vs 动态调度开销
  设置：相同的 MoE 模型，8/16/32 GPU
  对比：
    Baseline（动态）：运行时判断 overlap 时机
    Axion（静态）：StaticSchedule 直接按序执行
  指标：调度决策延迟、step time 方差（稳定性）
  预期：Axion step time 方差 < 基线 50%

Experiment 2：Overlap 率对比
  设置：DSv3-like 模型，64 GPU
  对比：
    LAER-MoE overlap 率
    Axion OverlapInsertionPass overlap 率
    理论最大 overlap 率（人工分析）
  预期：Axion 接近理论最大值，LAER-MoE 有 gap

Experiment 3：端到端吞吐
  设置：MoE 模型（64 experts），16~64 GPU
  对比：Megatron-LM、LAER-MoE、Axion
  指标：tok/s、MFU（Model FLOP Utilization）
  预期：Axion 吞吐 ≥ LAER-MoE（同等负载下）
```

---

## 4. Phase 3：Paper 3 ——FSEP 完整运行时（第 9~15 个月）

### 论文定位

```
标题方向：
  "FSEP: Fully Sharded Expert Parallelism with 
   Compile-Time Load Analysis for MoE Training"

投稿目标：SOSP 2027 或 EuroSys 2028

核心贡献：
  1. CommTensor 零 copy 实现（index map 机制完整评估）
  2. Expert 迁移执行器（double buffer + 异步 P2P）
  3. Slow Planner 完整实现（ILP 近似求解）
  4. Fast Router（routing bias，含收敛实验）
  5. 与 LAER-MoE 1.69x 加速的直接对比
```

### 具体任务

```
Month 9-10：CommTensor 零 copy 完整实现
  □ index map 生成算法（CommTensorLayoutPass → 物理 index 计算）
  □ 迁移后 index map 原子更新机制
  □ GPU SRAM index map 内存管理
  □ 微基准：index map 访问 vs 内存 copy 开销

Month 11-12：Expert 迁移执行器
  □ double buffer 机制（MIGRATING / SHADOW 状态机）
  □ 异步 P2P 迁移（与反向传播重叠）
  □ 迁移期间的梯度正确性保证
  □ Slow Planner ILP 近似（基于 METIS 图分区）

Month 13：Fast Router + 收敛实验
  □ routing bias 公式实现（load_penalty 计算）
  □ 收敛实验：有/无 Fast Router，loss curve 对比
  □ 超参搜索：α, β 的最优值
  □ 确保 Fast Router 不影响模型质量（perplexity 对比）

Month 14-15：大规模实验 + 写作
  □ 64~128 GPU，DSv3-like 模型（256 experts）
  □ 与 LAER-MoE 直接对比（相同硬件，相同模型）
  □ IB RDMA 接入（跨节点通信）
  □ 消融：FSEP vs 传统 EP，Slow Planner vs 无规划
  □ 论文写作
```

---

## 5. Phase 4：Paper 4 ——完整系统评估（第 16~20 个月）

### 论文定位

```
标题方向：
  "Axion: A Compile-First Communication-Native Runtime 
   for Large-Scale MoE Training"

投稿目标：OSDI 2028 或 SOSP 2028（系统完整论文）

核心贡献：
  完整系统设计与评估
  三个子系统的协同收益（CommTensor + StaticSchedule + FSEP）
  与 veScale-FSDP 的对比（Dense 参数分片场景）
  RaggedShard 支持 Shampoo/Muon 优化器的实验
```

---

## 6. 论文发表时间线

```
第 3 个月   → Paper 1 投稿（通信感知编译器）
第 8 个月   → Paper 2 投稿（静态 Overlap 调度）
第 15 个月  → Paper 3 投稿（FSEP 完整运行时）
第 20 个月  → Paper 4 投稿（完整系统）

Deadline 对应（参考近年时间）：
  MLSys：通常 11月截稿  → Paper 1 目标 MLSys '27
  OSDI：通常 12月截稿   → Paper 2 目标 OSDI '27
  SOSP：通常 4月截稿    → Paper 3 目标 SOSP '28
  OSDI：                → Paper 4 目标 OSDI '28
```

---

## 7. 资源需求

```
GPU 资源：
  Phase 0-1（单机验证）：1× A100/H100，持续 3 个月
  Phase 2（8~32 GPU）：  8~32× GPU，偶发性使用
  Phase 3（64~128 GPU）：64~128× GPU，集中实验期（约 2 个月）
  Phase 4（完整评估）：  128~256× GPU，约 1 个月

人力：
  1 名主力研究生（全职）：负责实现和实验
  1 名合作者（兼职）：负责 CUDA kernel 优化
  1 名导师（指导）：论文写作和方向把控

最小可行版本（只做 Paper 1+2）：
  1 名研究生 + 8 GPU × 2 个月 = 可以发 2 篇会议论文
```

---

## 8. 风险与应对

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| Fast Router 影响收敛 | 40% | Paper 3 贡献点减弱 | 准备替代方案：只用 Slow Planner，去掉 Fast Router |
| ILP 规划太慢 | 30% | Phase 3 延期 | 贪心算法作为备选，实验中标注为"近似最优" |
| 与 LAER-MoE 效果持平 | 25% | Paper 3 差异不足 | 改 framing：强调可调试性和工程价值，而非纯性能 |
| GPU 资源不足 | 20% | 大规模实验受阻 | Phase 3 改用 16~32 GPU 小规模实验，模型也缩小 |
| OverlapInsertionPass 正确性难证明 | 35% | Paper 2 严谨性受质疑 | 形式化定义依赖关系，给出充分条件证明 |

---

*研究路线 v0.1 | 2026-03-08*
