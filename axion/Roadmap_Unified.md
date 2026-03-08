# Axion 整体规划：长期系统路线图

> **定位:** AMD AI Infra 内部长期训练系统，工业收益驱动，技术影响力为辅  
> **版本:** v0.2 | 2026-03-08  
> **背景:** AMD 员工，ROCm + MI300X 硬件，有生产训练任务，无学术约束  
> **时间跨度:** 36 个月（3 年），分 4 个 Stage

---

## 0. 背景与核心思路

### 0.1 AMD AI Infra 视角下的优先级

```
与学术视角的关键差异：

  学术：论文是一等公民，工业收益是配菜
  AMD AI Infra：工业收益是一等公民，技术影响力（论文/开源）是加分项

具体意味着：
  ✅ 每个 Stage 必须有可量化的内部训练收益
  ✅ ROCm/MI300X 原生支持是一等需求，不是"后续适配"
  ✅ 与 AMD 内部框架（ROCm Megatron、内部 MoE 实现）的兼容是前提
  ✅ 技术影响力（博客/论文/开源）作为 Stage 2+ 的额外目标
  ❌ 不为了发论文而延迟工业交付
  ❌ 不做只有学术价值、无法在 AMD 生产环境运行的功能
```

### 0.2 AMD 特有的硬件优势与约束

```
MI300X 的独特性（相比 H100）：

  优势：
    192 GB HBM3 统一内存（远超 H100 的 80 GB）
    → FSEP Expert 分片的内存压力更小
    → 可以支持更大的 Expert 参数量在单卡上

    HBM3 内存带宽 5.3 TB/s（vs H100 的 3.35 TB/s）
    → CommTensor zero-copy 的收益在 MI300X 上更显著
    → pack/unpack 消除的绝对时间节省更大

    CPU-GPU 统一内存寻址（XGMI/Infinity Fabric）
    → Expert 迁移的 P2P 带宽更高（节点内 896 GB/s）

  约束：
    RCCL（ROCm 版 NCCL）与 NCCL 有细微差异
    → CommFabric 的 RCCL Driver 需要专门适配
    → aiter（AMD 的 flash attention 实现）替代 flash_attn
    → hipcc 替代 nvcc，CUDA Stream → HIP Stream

  机会：
    MI300X 的高内存带宽让 CommTensor zero-copy 收益更大
    这是 AMD 硬件上做这个系统的独特优势，值得在技术报告中突出
```

### 0.3 长期系统的核心思路

```
工业收益和技术积累螺旋驱动：

  生产训练任务 → CommReport 暴露真实瓶颈
        ↓
  真实瓶颈 → 驱动针对性的技术改进
        ↓
  技术改进 → 提升 MI300X 训练效率
        ↓
  效率数据 → 支撑 AMD 对外的技术影响力（博客/论文/开源）
        ↓
  技术影响力 → 吸引外部贡献者，加速系统演进
        ↓
  回到顶部（下一轮）
```

### 0.4 统一决策原则

```
原则 1：MI300X 原生，不是事后适配
  CommFabric 的 RCCL/HIP 实现与 CUDA/NCCL 实现同步开发
  任何 CUDA-only 的设计决策，都要先确认 HIP 等价实现存在

原则 2：生产收益优先
  任何功能，先问："这能让内部 MoE 训练加速多少？"
  无法量化收益的功能，推迟到 Stage 3+

原则 3：单机可验证（来自 Axon 设计哲学）
  每个新功能必须能用 Simulate Driver 在单机验证正确性
  不允许"必须上 64 GPU 集群才能发现的 bug"

原则 4：数据驱动决策
  每个 Stage 结束时，用 CommReport 的真实数据决定下一步重点
  不靠直觉，靠 profiling

原则 5：收敛性红线
  任何路由干预（routing bias、Expert 迁移）必须先过收敛实验
  loss 差异 > 1% → 立即停止，不上生产
```

---

## 1. Stage 1：立足（月 1~6）

### 核心目标

```
工业：CommReport 接入内部训练任务，AxionFastRouter 吞吐提升 ≥ 10%
技术：ModelGraph + 6 个 Pass 设计完成，CommTensor 类型系统验证
硬件：MI300X 单机跑通，ROCm/aiter 适配完成
```

### 时间线

```
Month 1：ModelGraph 基础设施（MI300X 单机）
─────────────────────────────────────────────────────────
  工程任务：
    □ Wire、OpNode、ModelGraph（frozen dataclass + SHA-256 hash）
    □ OpSpec 类型层次（MatMulSpec、AttentionSpec、RMSNormSpec、SwiGLUSpec）
    □ Pass 基类 + PassManager + CompileReport
    □ AnalysisPass（FLOPs/params 估算）
    □ FusionPass（RMSNorm+QKV+RoPE，SwiGLU 融合）
    □ Attention backends：sdpa / aiter（MI300X fmha_v3）/ reference

  ROCm 适配：
    □ HIP Stream 替代 CUDA Stream（执行引擎底层）
    □ aiter fmha_v3 接入 AttentionSpec backend
    □ MI300X 单机跑通 Llama 3.1 8B（数值误差 < 1e-3）

  技术任务：
    □ CommLayout 枚举设计（5种）
    □ CommTensor 数据结构草案（物理布局 + index map 概念）

  验收：
    □ MI300X 单机 Llama 3.1 8B 对齐参考实现
    □ aiter vs sdpa 性能基线对比（tok/s）
    □ Compile Report 正确输出 FLOPs、fusion 数量

─────────────────────────────────────────────────────────
Month 2：通信感知编译（CommInferencePass + Simulate Driver）
─────────────────────────────────────────────────────────
  工程任务：
    □ CommOpSpec 类型层次（A2ASpec、AllGatherSpec、ReduceScatterSpec）
    □ WireType 扩展（COMM、EXPERT_SHARD、ROUTING_TABLE）
    □ CommInferencePass：ExpertGateSpec/FFNSpec → 自动标注 A2ASpec
    □ CommTensorLayoutPass：layout_hint → physical_layout 决策
    □ CommFabric 接口定义
    □ Simulate Driver（no-op，单机验证用）
    □ 扩展 Compile Report：通信拓扑、CommLayout 分布

  技术任务：
    □ Layout 状态机正确性验证（手写 MoE layer IR 验证 5 个场景）
    □ 编译期错误检测：人工注入 7 类 layout 错误 → 验证全部捕获
    □ pack/unpack 消除的理论分析（MI300X HBM3 带宽节省估算）

  验收：
    □ CommInferencePass 正确标注 MoE 模型所有通信点
    □ Simulate Driver 单机跑通完整编译流程（无 GPU 集群）
    □ 7 类 layout 错误全部在编译期捕获（不需要运行）

─────────────────────────────────────────────────────────
Month 3：AxionCommProfiler 接入内部训练任务
─────────────────────────────────────────────────────────
  工程任务：
    □ AxionCommProfiler（hook 方式接入内部 MoE 训练框架）
    □ CommReport HTML 可视化
      - Expert 负载热力图（256 experts × N steps）
      - A2A 时序图（overlap 率、暴露时间）
      - MI300X HBM 带宽利用率
    □ 接入内部 MI300X 集群训练任务（目标：DSv3-like 或内部 MoE 模型）
    □ 接入开销 < 0.5% 验证（hipperf 测量）
    □ RCCL 通信统计接入（rccl-trace 或自定义 hook）

  技术产出：
    □ 内部技术报告：MI300X 上 MoE 训练通信瓶颈分析
      （这是后续所有优化工作的数据基础）
    □ 关键发现记录：
      - 实际负载不均衡系数（预期：1.5~4x）
      - A2A 时间占总 step time 比例
      - 当前 overlap 率 vs 理论最大值
      - 跨节点通信占比

  验收：
    □ CommReport 数据与手动 RCCL profiling 结果误差 < 5%
    □ 至少接入 1 个内部生产训练任务

─────────────────────────────────────────────────────────
Month 4~5：AxionFastRouter + 收敛实验
─────────────────────────────────────────────────────────
  工程任务：
    □ AxionFastRouter 实现
      gate_logits_adjusted[i] = gate_logits[i] - α · load_penalty[i]
      load_penalty[i] = (tokens_on_expert_i / avg) ^ β × gpu_util[i]
    □ 一行接入：在 MoE Gate TopK 之前插入 adjust_gate_logits()
    □ 收敛实验设计：
      - 模型：内部 2B MoE（快速验证）
      - 对比：Baseline vs FastRouter（α=0.1, β=2.0）
      - 步数：1000 steps（loss curve + perplexity）
    □ 红线验证：loss 差异 < 0.5%，梯度 norm 无异常
    □ 通过后：接入内部训练任务，测量 MI300X 吞吐提升

  ROCm 适配：
    □ 验证 hipblaslt 在 load_penalty 计算中的正确性
    □ 确认 HIP atomic 操作在 ExpertLoadStats 更新中无竞争

  技术产出：
    □ 超参分析：α, β 对吞吐和收敛的影响（MI300X 实测）
    □ Expert 激活分布变化分析（是否影响 Expert 专业化）

─────────────────────────────────────────────────────────
Month 6：FSEPShardingPass + Stage 1 总结
─────────────────────────────────────────────────────────
  工程任务：
    □ CommGraph 构建（从 CommAnnotation 构建通信依赖有向图）
    □ FSEPShardingPass 贪心算法
      - 目标：minimize max_gpu(compute + comm_time)
      - 约束：内存均衡（MI300X 192GB 更宽裕）、跨节点通信最小化
    □ ExpertShardRegistry（Expert 分片注册表）
    □ 2~8 MI300X 验证 Expert dispatch/combine 正确性（RCCL）
    □ Slow Planner 基础版（只做初始分配，不做迁移）

  Stage 1 总结（数据驱动决策）：
    □ CommReport 数据：负载不均衡系数 → 决定 FSEP 优先级
    □ FastRouter 吞吐提升：决定是否继续激进均衡策略
    □ MI300X HBM 带宽利用率：决定 CommTensor zero-copy 的投入
    □ 内部团队反馈：CommReport 是否真正有助于 debug
```

### Stage 1 交付物

| 交付物 | 时间 | 价值 |
|--------|------|------|
| AxionCommProfiler（内部 pip 包） | Month 3 | 第一次看清 MI300X 上的通信瓶颈 |
| CommReport HTML 可视化 | Month 3 | 内部工程师 debug 工具 |
| 内部技术报告：MI300X MoE 通信分析 | Month 3 | 数据基础，指导后续优化方向 |
| AxionFastRouter（内部 pip 包） | Month 5 | 预期 ≥ 10% 吞吐提升 |
| 收敛实验报告 | Month 5 | 证明路由干预安全性 |
| ModelGraph + 6 Pass 完整实现 | Month 6 | 系统基础设施 |

---

## 2. Stage 2：扩展（月 7~14）

### 核心目标

```
工业：Expert 物理迁移生产部署，MI300X 64 GPU 吞吐提升 ≥ 20%
技术：StaticSchedule + OverlapInsertionPass，对比 LAER-MoE
硬件：RCCL All-to-All 完整接入，Infinity Fabric P2P 迁移
```

### 时间线

```
Month 7~8：OverlapInsertionPass + StaticSchedule
─────────────────────────────────────────────────────────
  工程任务：
    □ 数据依赖分析算法（Compute ↔ Comm 依赖图构建）
    □ OverlapInsertionPass（插入 OverlapSpec 节点）
      策略：AGGRESSIVE（默认）/ CONSERVATIVE / DISABLED
    □ StaticSchedule 数据结构 + 执行引擎
      HIP Stream 绑定：Compute Stream / Comm Stream 分离
    □ 正确性验证：overlap 结果 == 非 overlap 结果（8 MI300X）

  ROCm 适配：
    □ RCCL all_to_all_async 接口封装
    □ HIP Event 用于 Stream 同步（替代 CUDA Event）
    □ 验证 hipStreamWaitEvent 的语义与 cudaStreamWaitEvent 一致

  技术任务：
    □ 依赖分析算法正确性：形式化定义充分条件
    □ overlap 率理论上界 vs 实际值的 gap 分析

─────────────────────────────────────────────────────────
Month 9~10：CommFabric RCCL Driver + DistributedExecutablePlan
─────────────────────────────────────────────────────────
  工程任务：
    □ CommFabric RCCL Driver（完整实现）
      - all_to_all_async（分 chunk 支持）
      - all_gather_async（BLOCKED_BY_EXPERT 输出）
      - reduce_scatter（节点内 → 节点间分层）
      - p2p_async（Infinity Fabric，Expert 迁移用）
    □ DistributedExecutablePlan
      （compute_steps + comm_steps + static_schedule + expert_registry）
    □ 16 MI300X 端到端训练（MoE 模型，完整前向+反向+优化器）
    □ 对比实验：StaticSchedule vs 无 Overlap（step time 方差分析）

  ROCm 特有优化：
    □ MI300X Infinity Fabric P2P 直接传输（比 RDMA 更快的节点内路径）
    □ 验证 XGMI 带宽（896 GB/s 节点内）是否被充分利用
    □ RCCL 分层 All-to-All（节点内走 XGMI，节点间走 ROCEv2）

─────────────────────────────────────────────────────────
Month 11~12：AxionSlowPlanner（Expert 物理迁移）
─────────────────────────────────────────────────────────
  工程任务：
    □ double buffer 机制（MIGRATING / SHADOW 状态机）
    □ 异步 P2P 迁移（借鉴 LAER-MoE，与反向传播重叠）
      MI300X：Infinity Fabric P2P（节点内），ROCEv2 RDMA（跨节点）
    □ 迁移期间梯度正确性保证（shadow buffer 生命周期管理）
    □ 迁移触发条件校准（基于 Stage 1 CommReport 的实测数据）
    □ 64 MI300X 生产部署验证

  收敛实验（必做）：
    □ 内部 7B MoE 模型，2000 steps
    □ 对比：Baseline vs Slow Planner（imbalance_threshold 扫描）
    □ 红线：loss 差异 < 0.5%，无 loss spike

  技术产出：
    □ MI300X 上 Expert 迁移的通信开销实测
      （Infinity Fabric vs ROCEv2 的 P2P 延迟对比）
    □ 迁移 ROI 分析（计算节省 / 迁移通信成本）

─────────────────────────────────────────────────────────
Month 13~14：对外技术影响力 + Stage 2 总结
─────────────────────────────────────────────────────────
  技术影响力任务（可选但推荐）：
    □ AMD 技术博客：
      "Load-Balanced MoE Training on MI300X with FSEP"
      （数据：Stage 1+2 的真实吞吐数据，对比 H100 基线）
    □ 考虑投稿 MLSys 2027 或 SC 2026
      核心贡献：MI300X 上的 FSEP + 静态调度，ROCm 原生实现
    □ 开源 AxionCommProfiler（低风险，高影响力）

  工程任务：
    □ 64 MI300X 稳定性验证（连续训练 72 小时无中断）
    □ CommReport 自动化（接入内部 MLOps 平台，每次训练自动生成）
    □ 内部吞吐提升报告（对比 Stage 1 基线，目标 ≥ 20%）

  Stage 2 总结（数据驱动决策）：
    □ A2A 时间占比（决定 Stage 3 CommTensor 投入是否值得）
    □ Expert 迁移稳定性数据（决定 Stage 3 迁移频率策略）
    □ MI300X HBM 带宽利用率（决定 CommTensor 的预期收益）
    □ 内部用户反馈（谁在用 CommReport？有没有改变工作方式？）
```

### Stage 2 交付物

| 交付物 | 时间 | 价值 |
|--------|------|------|
| CommFabric RCCL Driver | Month 10 | ROCm 原生通信底层 |
| DistributedExecutablePlan + StaticSchedule | Month 10 | 编译期确定执行计划 |
| AxionSlowPlanner 生产部署（64 MI300X） | Month 12 | 预期总计 ≥ 20% 吞吐提升 |
| CommReport 自动化集成（MLOps） | Month 14 | 通信可观测性常态化 |
| 技术博客 / MLSys 投稿（可选） | Month 13 | AMD 对外技术影响力 |

---

## 3. Stage 3：深化（月 15~24）

### 核心目标

```
工业：CommTensor zero-copy 替换 Expert 通信路径
     256 MI300X 支持，系统自治（自动调参），稳定性 7 天
技术：完整 FSEP 系统 vs LAER-MoE 直接对比（MI300X vs A100）
硬件：ROCEv2 RDMA 跨节点，分层 Reduce-Scatter
```

### 时间线

```
Month 15~17：CommTensor zero-copy 生产实现
─────────────────────────────────────────────────────────
  工程任务：
    □ index map 生成算法（CommTensorLayoutPass → 物理 index 计算）
    □ 迁移后 index map 原子更新（HIP atomic，MI300X SRAM 存储）
    □ Step 3a：消除 dispatch pack copy
    □ Step 3b：消除 combine unpack copy（完整版）
    □ 微基准：MI300X HBM3 5.3 TB/s 下 index map vs memcpy 开销

  MI300X 特有优化：
    □ 利用 MI300X 大 HBM（192 GB）允许更大的 CommTensor buffer
    □ HBM3 高带宽（5.3 TB/s）使 zero-copy 绝对收益更大
    □ LDS（Local Data Share）用于高频访问的 index map 缓存

  技术产出：
    □ CommTensor zero-copy 在 MI300X vs H100 的对比分析
      （HBM3 带宽优势是否转化为更大的相对收益？）

─────────────────────────────────────────────────────────
Month 18~19：ROCEv2 RDMA + 跨节点通信优化
─────────────────────────────────────────────────────────
  工程任务：
    □ CommFabric ROCEv2 RDMA Driver（跨节点 All-to-All）
    □ 分层 Reduce-Scatter（节点内 XGMI → 节点间 ROCEv2）
    □ 128~256 MI300X 端到端验证
    □ FSEPShardingPass 增加跨节点感知（优先节点内 Expert 分配）

  ROCm 适配：
    □ rccl_comm_t 的跨节点组管理
    □ MI300X 节点间：400G ROCEv2（8×50G）或 AMD Instinct HGX 配置
    □ 验证 GPUDirect RDMA 在 ROCm 下的 Expert P2P 迁移路径

─────────────────────────────────────────────────────────
Month 20~21：系统自治（自动调参）
─────────────────────────────────────────────────────────
  工程任务：
    □ 自适应 chunk_size
      根据 MI300X HBM 带宽和 RCCL 延迟特性自动选择最优 chunk 大小
    □ 自适应迁移阈值
      imbalance_threshold 根据历史 CommReport 数据自动校准
    □ 自适应 α, β
      Fast Router 超参根据实测负载分布自动调整
    □ 目标：新 MI300X 集群部署 Axion → 零手动调参，开箱即用

  技术产出：
    □ 自适应调参算法 vs 手动调参的 A/B 测试报告

─────────────────────────────────────────────────────────
Month 22~24：对外技术影响力 + Stage 3 总结
─────────────────────────────────────────────────────────
  技术影响力任务：
    □ 投稿 MLSys 2027 / OSDI 2027（重点）
      标题方向：
        "Axion: Communication-Native MoE Training on AMD MI300X"
      核心数据：
        - MI300X 256 GPU vs LAER-MoE（A100）的吞吐对比
        - CommTensor zero-copy 在 HBM3 下的绝对收益
        - StaticSchedule 的 overlap 率数据
    □ AMD 开发者博客系列（3 篇）：
        1. "MI300X 上的 MoE 通信瓶颈分析"（Stage 1 数据）
        2. "FSEP + 静态调度在 MI300X 上的实现"（Stage 2 数据）
        3. "CommTensor：消除 MoE 训练中的 pack/unpack 开销"（Stage 3 数据）
    □ 考虑开源 AxionCommProfiler + CommReport（低风险，高影响力）

  工程任务：
    □ 256 MI300X 稳定性验证（连续训练 7 天）
    □ RaggedShard 支持（Shampoo/Muon 优化器，对接 veScale-FSDP 理念）
    □ 全面性能报告（内部分发）

  Stage 3 总结（数据驱动决策）：
    □ 系统稳定性是否支持 1000+ GPU 扩展？
    □ 开源 ROI：技术影响力 vs 维护成本，值不值得？
    □ Stage 4 重点：新架构（Mamba/多模态）vs 超大规模扩展？
```

### Stage 3 交付物

| 交付物 | 时间 | 价值 |
|--------|------|------|
| CommTensor zero-copy（256 MI300X 生产级） | Month 19 | 额外 5~15% 吞吐提升 |
| CommFabric ROCEv2 RDMA Driver | Month 19 | 跨节点通信 ROCm 原生 |
| 系统自治（零调参） | Month 21 | 降低 MI300X 部署门槛 |
| 256 MI300X 7 天稳定性报告 | Month 23 | 生产可用证明 |
| 论文投稿（MLSys/OSDI）+ 技术博客系列 | Month 22 | AMD 外部技术影响力 |

---

## 4. Stage 4：系统化（月 25~36）

### 核心目标

```
工业：支持新架构（Mamba、多模态 MoE），成为 AMD 内部标准训练框架
     1000+ MI300X 超大规模支持
技术：完整系统论文，考虑开源（评估 AMD IP 约束）
硬件：AMD 下一代硬件（MI350X？）的前瞻性支持
```

### 时间线

```
Month 25~28：通用化 + 新架构
─────────────────────────────────────────────────────────
  工程任务：
    □ ModelArch 扩展：Mamba/SSM 支持（状态空间模型）
    □ 多模态 MoE 支持（视觉 + 语言 Expert 混合路由）
    □ Pipeline Parallelism 支持（Comm.Send/Recv Op）
    □ 1000+ MI300X 扩展性测试（通信拓扑感知的 Expert 分配）

  AMD 前瞻：
    □ 与 AMD 硬件团队对齐：MI350X 的内存/带宽规格变化
    □ CommFabric 抽象是否需要适配新的 Infinity Fabric 拓扑

─────────────────────────────────────────────────────────
Month 29~32：开源评估 + 社区建设
─────────────────────────────────────────────────────────
  决策：评估哪些部分可以开源（AMD IP 和竞争因素）
    □ 可开源（高影响力，低 IP 风险）：
        AxionCommProfiler + CommReport
        Simulate Driver
        ModelGraph + PassManager（通用框架）
    □ 内部保留（核心竞争力）：
        MI300X 特化的 CommFabric RCCL 优化细节
        内部训练任务的具体 profiling 数据

  若决定开源：
    □ 公开 API 设计（隐藏 AMD 内部特化细节）
    □ 文档系统（用户文档 + ROCm 快速入门）
    □ CI/CD（基于 Simulate Driver，不需要真实 GPU）
    □ HuggingFace 权重加载/导出

─────────────────────────────────────────────────────────
Month 33~36：完整系统论文 + 对外影响力峰值
─────────────────────────────────────────────────────────
  技术影响力：
    □ Paper 4（完整系统论文）
      目标：OSDI 2028 或 SC 2027
      内容：1000+ MI300X，多架构，完整系统演进故事
    □ 开源发布（如果 AMD 批准）
    □ 与 ROCm 生态的深度集成（作为官方推荐的 MoE 训练框架）
```

---

## 5. 关键决策门

```
Stage 1 结束（Month 6）：
  数据：负载不均衡系数（CommReport）
  数据：FastRouter 吞吐提升（MI300X 实测）
  数据：A2A 时间占总 step time 比例
  
  if 吞吐提升 < 5%：
    → 检查根因：是计算瓶颈？还是 RCCL 配置问题？
    → 可能需要先解决 ROCm/aiter kernel 性能问题
    → 调整 Stage 2：减少 FSEP，增加 MI300X kernel 优化
    
  if 吞吐提升 5~15%：
    → 继续 Stage 2，但 FastRouter 需要更多收敛验证
    
  if 吞吐提升 ≥ 15%：
    → 全速推进 Stage 2，Expert 迁移是最高优先级

─────────────────────────────────────────────────────────

Stage 2 结束（Month 14）：
  数据：64 MI300X 吞吐提升 vs 内部 baseline
  数据：Expert 迁移 loss spike 频率
  数据：A2A 时间占比（决定 CommTensor 投入）
  
  if A2A 占比 < 15%：
    → CommTensor zero-copy 投入产出比低
    → Stage 3 重点转向：更大规模 + 系统稳定性
    
  if A2A 占比 ≥ 15%：
    → CommTensor 是 Stage 3 最高优先级
    
  if 迁移 loss spike > 2 次/1000 steps：
    → Slow Planner 触发条件需要收紧
    → Stage 3 增加迁移平滑机制

─────────────────────────────────────────────────────────

Stage 3 结束（Month 24）：
  数据：256 MI300X 稳定性（7 天连续训练）
  数据：vs LAER-MoE/Megatron 的端到端吞吐对比
  决策：是否开源？（AMD IP 审查 + 技术影响力评估）
  
  if 开源 ROI 高（技术差异化 > 维护成本）：
    → Stage 4 重点：开源 + 社区 + 新架构
    
  if 开源 ROI 低（内部价值 > 外部影响力）：
    → Stage 4 重点：新架构支持 + 1000+ GPU 扩展
    → 对外以技术博客 + 论文为主
```

---

## 6. 资源规划（AMD 内部视角）

```
Stage 1（月 1~6）：1 名工程师（你自己），业余时间 or 20% 项目
  GPU：1×MI300X（单机开发），内部集群访问权限（64 MI300X 测试）
  工具：ROCm stack，hipcc，rccl-trace，rocprof

Stage 2（月 7~14）：1 名主力 + 1 名临时合作者（RCCL/kernel 背景）
  GPU：64 MI300X 集群（持续使用）
  协作：可能需要 AMD ROCm 团队的 RCCL 支持

Stage 3（月 15~24）：2 名全职（需要争取内部 headcount / 20% 项目合并）
  GPU：256 MI300X 集群（实验期集中使用）
  协作：AMD 硬件团队（MI300X 拓扑信息）

Stage 4（月 25~36）：视开源决策和 AMD 内部推广情况
  GPU：1000+ MI300X（AMD 内部资源）

最小可行路径（个人项目）：
  Stage 1 完整 + Stage 2 的 Month 7~10：
    1 名工程师，业余 / 20% 时间，约 10 个月
    MI300X 单机 + 小规模集群访问
    产出：CommProfiler + 技术报告 + FastRouter
    这已经足够作为内部 proposal 争取更多资源
```

---

## 7. 风险与应对

| 风险 | 阶段 | 概率 | 影响 | 应对 |
|------|------|------|------|------|
| ROCm RCCL 与设计不兼容 | Stage 2 | 20% | CommFabric 延期 | 提前做 RCCL API 兼容性调研（Stage 1 Month 2）|
| FastRouter 影响收敛 | Stage 1 | 40% | 吞吐提升不达预期 | 直接跳 FastRouter，只用 Slow Planner |
| Slow Planner 触发 loss spike | Stage 2 | 25% | 生产部署受阻 | 收紧触发条件，imbalance_threshold 从 2.0 开始（保守）|
| MI300X kernel 性能不达预期 | Stage 1 | 30% | 基线吞吐偏低 | 提前对齐 aiter vs flash_attn，确认 fmha_v3 优化到位 |
| AMD 内部 IP 限制开源 | Stage 4 | 35% | 社区影响力受限 | 以论文 + 博客代替开源，开源工具层（CommProfiler）|
| 个人 20% 项目时间不足 | Stage 1+ | 35% | 进度滞后 | Stage 1 Month 3 产出 CommReport 作为内部立项材料，争取 headcount |

---

## 8. 最小可行路径（MVP，适合从个人项目启动）

```
如果只有个人业余 / 20% 时间，优先做这些：

  Week 1~4（Month 1）：
    □ ModelGraph + PassManager（单机基础设施）
    □ MI300X 单机 Llama 3.1 8B 跑通（aiter backend）
    → 产出：可工作的 Compile-First 基础设施

  Week 5~8（Month 2）：
    □ CommInferencePass + Simulate Driver
    □ Layout 状态机验证
    → 产出：单机可验证的通信图分析

  Week 9~12（Month 3）：
    □ AxionCommProfiler（接入内部框架）
    □ CommReport 可视化
    → 产出：真实的 MI300X 通信瓶颈数据报告

此时停下来，用这份报告向内部展示：
  "我们的 MI300X MoE 训练有 X 倍负载不均衡，A2A 占 Y% 的 step time"
  "我有一个方案可以解决这个问题，需要 N 个 GPU 和 M 个月"

如果内部认可，争取资源继续 Stage 1 Month 4~6 和 Stage 2。
如果不认可，Stage 1 已经产出了有价值的工具和分析报告。
```

---

## 9. 成功标准（3 年后）

```
工业维度（AMD 内部）：
  □ 256+ MI300X 训练任务使用 Axion（至少 2 个内部产品团队）
  □ 端到端吞吐 vs 内部 baseline 提升 ≥ 20%
  □ 通信相关 debug 时间减少 ≥ 50%（基于内部反馈）
  □ 稳定性：7 天连续训练无通信相关中断

技术影响力（AMD 外部）：
  □ 1~2 篇顶会论文（MLSys / OSDI / SC）
  □ AMD 技术博客系列（3 篇以上，有外部引用）
  □ GitHub 开源项目（如果 AMD 批准）≥ 300 Stars

技术维度：
  □ CommTensor 类型系统在 AMD 内部被其他项目引用
  □ 支持 Transformer / MoE / Mamba 三类主流架构
  □ ROCm 原生实现，与 AMD 硬件演进保持同步
```

---

*Axion 整体规划 v0.2 | 2026-03-08*  
*背景：AMD AI Infra 员工 | ROCm + MI300X | 工业收益驱动，技术影响力为辅*
