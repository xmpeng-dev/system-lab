# Axion 整体规划：长期系统路线图

> **定位:** AMD AI Infra 员工的个人长期技术投入，以现有系统（Primus/Megatron）为验证平台  
> **版本:** v0.3 | 2026-03-08  
> **核心策略:** Feature-by-Feature 在现有系统验证 → 有收益才继续 → 最终重写 Axion  
> **时间跨度:** 不设硬性终点，由每个 Feature 的收益数据驱动决策

---

## 0. 核心思路：验证优先，Axion 是终点不是起点

### 0.1 为什么不直接开发 Axion

```
错误路径（之前的计划）：
  直接开发 Axion 全套系统
  → 18 个月后才能看到收益
  → 如果某个技术点不 work，整个投入打水漂
  → 个人业余项目，无法承受这个风险

正确路径（本文档）：
  在 Primus/Megatron 上逐个验证技术点（Feature）
  每个 Feature 独立可交付，有明确的收益指标
  
  Feature 通过验证（有收益）→ 继续下一个 Feature
  Feature 未通过验证（无收益）→ 停止，不浪费更多时间
  
  所有核心 Feature 验证完成 → 用这些已验证的设计重写 Axion
  Axion 不是起点，是终点：一个已经被数据证明的设计的干净实现
```

### 0.2 整体结构

```
验证阶段（Feature 0 ~ Feature N）：
  在 Primus/Megatron 上以 patch/hook/plugin 形式实现
  每个 Feature 有独立的：
    - 实现方案（最小改动接入）
    - 收益假设（预期提升 X%）
    - 验证实验（实测数据）
    - 决策门（通过 → 继续，不通过 → 停止或调整）
    - 技术产出（内部报告 / 博客 / 论文）

  ┌──────────────────────────────────────────────────────┐
  │  Feature 0：通信可观测性（CommProfiler）              │
  │  Feature 1：路由负载均衡（FastRouter）               │
  │  Feature 2：Expert 物理迁移（SlowPlanner）           │
  │  Feature 3：静态 Overlap 调度（OverlapScheduler）    │
  │  Feature 4：CommTensor zero-copy                    │
  │  Feature 5：...（由前序数据驱动）                    │
  └──────────────────────────────────────────────────────┘
                          │
              每个 Feature 通过验证
                          │
                          ▼
  构建阶段（Axion 重写）：
    用已验证的设计，在干净的 Axion IR + Pass 框架中重实现
    此时 Axion 的每个设计决策都有数据背书
    不是"希望它 work"，而是"已经证明它 work"

```

### 0.3 验证 vs 构建的判断标准

```
什么时候从"验证阶段"进入"构建阶段"（开始 Axion 重写）？

触发条件（需要同时满足）：
  □ ≥ 3 个 Feature 已通过验证，且累计收益 ≥ 20%
  □ 这些 Feature 在 Primus/Megatron 上的 patch 开始出现相互耦合
    （说明继续在现有系统上叠加已经比重写更麻烦了）
  □ 有足够时间投入（不再是业余项目，或获得了内部支持）

如果上述条件始终不满足：
  → Axion 的"验证阶段"成果本身就足够有价值
  → 这些 Feature 作为 Primus/Megatron 的改进 patch 独立存在
  → 不强求进入构建阶段
```

---

## 1. Feature 0：通信可观测性（CommProfiler）

### 背景与假设

```
假设：MoE 训练中存在显著的 Expert 负载不均衡和通信瓶颈，
      但工程师目前无法直接看到这些问题，导致优化方向不清晰。

验证目标：在 MI300X 上跑 Primus/Megatron，
          用 CommProfiler 量化以下数据：
            - Expert 负载不均衡系数（max_load / avg_load）
            - A2A 时间占总 step time 的比例
            - 当前 Overlap 率 vs 理论上界
            - 跨节点 A2A 占比

这些数据决定后续所有 Feature 的优先级。
```

### 实现方案（最小侵入）

```python
# 接入方式：在现有 MoE Gate / dispatch / combine 前后挂 hook
# 不改任何计算逻辑，纯观测

class CommProfiler:
    def attach(self, model):
        """挂在现有 Primus/Megatron MoE 层上，无需改模型代码"""
        for layer in model.moe_layers():
            layer.register_forward_pre_hook(self._before_dispatch)
            layer.register_forward_hook(self._after_combine)

    def report(self) -> CommReport:
        return CommReport(
            expert_load_imbalance = ...,  # max/avg token 数
            a2a_time_fraction     = ...,  # A2A 时间 / step time
            actual_overlap_ratio  = ...,  # 实测 compute/comm 重叠率
            cross_node_fraction   = ...,  # 跨节点 A2A 占比
            hot_experts           = ...,  # top-K 热点 expert
        )
```

### 收益假设与验证

```
预期：能生成 CommReport，可视化 MI300X 上的通信热点
量化指标：
  □ CommReport 数据与手动 RCCL profiling 误差 < 5%
  □ 接入开销 < 0.5%（profiling 本身不拖慢训练）

这个 Feature 没有"吞吐提升"，收益是"信息价值"——
  通过数据确认：负载不均衡是否真实？A2A 是否是瓶颈？
  这直接决定 Feature 1~4 是否值得做。
```

### 决策门

```
Feature 0 完成后，根据 CommReport 数据做出判断：

  if Expert 负载不均衡 < 1.3x：
    → 负载均衡收益有限，Feature 1/2 优先级降低
    → 优先看 A2A 时间占比，决定是否做 Feature 3/4

  if Expert 负载不均衡 ≥ 2x：
    → Feature 1（FastRouter）和 Feature 2（SlowPlanner）是高优
    → 立即开始 Feature 1

  if A2A 时间占比 < 10%：
    → Feature 3/4 的 overlap 和 zero-copy 收益有限
    → 重新评估整个路线

  if A2A 时间占比 ≥ 20%：
    → Feature 3/4 高优先级
```

### 技术产出

```
□ AxionCommProfiler（内部 Python 包，pip install）
□ CommReport HTML 可视化（Expert 热力图 + A2A 时序图）
□ 内部技术报告：MI300X MoE 训练通信瓶颈分析
  （这是后续所有 Feature 的立项依据）
□ 可选：AMD 技术博客 "Profiling MoE Communication on MI300X"
```

### 时间估计
**2~3 周**（业余时间 4~6 周）

---

## 2. Feature 1：路由负载均衡（FastRouter）

### 背景与假设

```
假设：通过调整 gate logits，可以软性引导 token 远离过载 Expert，
      在不改变 Expert 物理位置的情况下减轻负载不均衡。

前提：Feature 0 CommReport 显示负载不均衡系数 ≥ 1.5x

实现方式：在 Primus/Megatron 的 MoE Gate 模块，
          在 softmax/topk 之前插入一行偏置调整：
          gate_logits -= α * load_penalty
```

### 实现方案

```python
# 接入方式：在现有 MoE Gate forward 中插入一行
# Primus/Megatron 的 MoE Gate 通常是：
#   scores = gate_logits.softmax(-1)
#   topk_scores, topk_indices = scores.topk(k)
#
# 改为：
#   gate_logits = self.fast_router.adjust(gate_logits)  # ← 只加这一行
#   scores = gate_logits.softmax(-1)
#   topk_scores, topk_indices = scores.topk(k)

class FastRouter:
    def __init__(self, alpha=0.1, beta=2.0, ema_decay=0.9):
        self.alpha = alpha
        self.beta  = beta
        self.load_ema = None  # 指数移动平均，平滑负载统计

    def adjust(self, gate_logits: Tensor) -> Tensor:
        if self.load_ema is None:
            return gate_logits  # 第一个 step 无统计，跳过
        penalty = (self.load_ema / self.load_ema.mean()) ** self.beta
        return gate_logits - self.alpha * penalty.log()

    @torch.no_grad()
    def update(self, expert_counts: Tensor):
        """每个 step dispatch 后更新负载统计"""
        if self.load_ema is None:
            self.load_ema = expert_counts.float()
        else:
            self.load_ema = self.ema_decay * self.load_ema \
                          + (1 - self.ema_decay) * expert_counts.float()
```

### 收益假设与验证

```
预期吞吐提升：5~15%（通过减少过载 GPU 的等待时间）

必做实验（按顺序）：

  实验 A：收敛性验证（红线）
    设置：内部 2B MoE，1000 steps
    对比：Baseline vs FastRouter (α=0.1, β=2.0)
    指标：loss curve、final perplexity
    红线：loss 差异 > 1% → 停止，不做任何生产部署

  实验 B：负载均衡效果
    设置：CommProfiler 在启用 FastRouter 前后各跑 100 steps
    指标：Expert 负载不均衡系数变化（预期从 Xb 降到 Xa）

  实验 C：吞吐测量
    设置：相同模型相同数据，64 MI300X
    指标：tok/s 提升百分比
```

### 决策门

```
if 实验 A 红线触发（loss 差异 > 1%）：
  → 停止 FastRouter，直接跳 Feature 2（SlowPlanner，物理迁移）
  → 物理迁移不改路由语义，收敛风险更低

if 实验 C 吞吐提升 < 3%：
  → FastRouter 收益不显著（负载均衡不是主要瓶颈）
  → 查看 CommReport：A2A 时间占比是否更高？
  → 如果 A2A 占比高 → 跳到 Feature 3（Overlap 调度）

if 实验 C 吞吐提升 ≥ 5%：
  → 继续 Feature 2（叠加 SlowPlanner 物理迁移）
```

### 技术产出

```
□ FastRouter patch（Primus/Megatron PR 或内部 patch）
□ 收敛实验报告（内部文档）
□ 超参分析：α, β 对均衡效果和收敛的 trade-off
□ 可选：结合 Feature 0 数据，整合进一篇 paper（通信优化 + 可观测性）
```

### 时间估计
**实现：1 周 | 收敛实验：2 周 | 总计 3~4 周**（业余 6~8 周）

---

## 3. Feature 2：Expert 物理迁移（SlowPlanner）

### 背景与假设

```
假设：每隔 K 个 step，根据历史路由统计，将过载 Expert 的参数
      迁移到负载较轻的 GPU，从根本上消除负载不均衡。

前提：Feature 0 CommReport 显示负载不均衡系数 ≥ 1.5x
      （Feature 1 可选，SlowPlanner 可以单独做）

参考：LAER-MoE 论文的核心方案，已有公开实现可参考
      (https://github.com/PKUDAIR/Hetu-Galvatron/tree/laer-moe)
```

### 实现方案

```python
# 接入方式：在 Primus/Megatron 的训练循环中增加 hook
#
# 原始训练循环：
#   for step in range(max_steps):
#       loss = model(batch)
#       loss.backward()
#       optimizer.step()
#
# 增加 SlowPlanner：
#   for step in range(max_steps):
#       loss = model(batch)
#       loss.backward()
#       optimizer.step()
#       planner.maybe_migrate(step, model)  # ← 只加这一行

class SlowPlanner:
    def __init__(self, check_interval=50, imbalance_threshold=1.3):
        self.check_interval      = check_interval
        self.imbalance_threshold = imbalance_threshold
        self.load_history        = []

    def maybe_migrate(self, step, model):
        self.load_history.append(collect_expert_loads(model))

        if step % self.check_interval != 0:
            return

        imbalance = compute_imbalance(self.load_history[-self.check_interval:])
        if imbalance < self.imbalance_threshold:
            return  # 不需要迁移

        plan = self._greedy_plan(self.load_history)
        self._execute_migration(model, plan)  # 异步 P2P（与下一个 step 重叠）

    def _greedy_plan(self, history):
        """贪心：把热点 Expert 迁移到冷点 GPU"""
        # 简单贪心，不需要 ILP
        ...

    def _execute_migration(self, model, plan):
        """
        Primus/Megatron 中：直接用 dist.isend/irecv 做异步 P2P
        MI300X：走 Infinity Fabric（节点内高带宽）
        """
        for src_rank, dst_rank, expert_param in plan:
            dist.isend(expert_param, dst=dst_rank)  # 异步，不阻塞训练
```

### 收益假设与验证

```
预期吞吐提升：10~25%（在 FastRouter 基础上额外，或单独使用）

必做实验：

  实验 A：收敛性验证（红线）
    设置：内部 7B MoE，2000 steps
    对比：Baseline vs SlowPlanner
    指标：loss curve，关注迁移时刻是否有 loss spike
    红线：loss spike 幅度 > 5% of moving average → 收紧触发条件

  实验 B：迁移开销 vs 收益
    指标：单次迁移耗时（MI300X P2P 带宽实测）
          迁移后 step time 降低持续时间
          ROI = 节省计算时间 / 迁移通信时间（预期 > 5x）

  实验 C：与 Feature 1 叠加效果
    对比：Baseline / FastRouter only / SlowPlanner only / 两者叠加
    指标：吞吐、负载均衡系数

  实验 D：规模效果
    设置：8 / 16 / 32 / 64 MI300X
    指标：提升比例是否随规模增大（不均衡问题在大规模下更严重）
```

### 决策门

```
if 实验 A 出现持续 loss spike（不可接受）：
  → 提高 imbalance_threshold（更保守触发）
  → 降低 max_experts_per_migration（每次迁更少）
  → 如果问题依然存在，停止 SlowPlanner
  → 转向 Feature 3（Overlap 调度，不改物理分布，无收敛风险）

if 实验 B ROI < 2x（迁移成本太高）：
  → 减少迁移频率（check_interval 从 50 增加到 100）
  → 或限制到节点内迁移（走 XGMI，不走 ROCEv2）

if 实验 C/D 吞吐提升 ≥ 10%：
  → Feature 1+2 组合有明确收益
  → 继续 Feature 3（Overlap 调度叠加）
```

### 技术产出

```
□ SlowPlanner patch（Primus/Megatron，最小改动）
□ 收敛实验报告 + 迁移 ROI 分析
□ MI300X P2P 带宽实测数据（Infinity Fabric vs ROCEv2）
□ 论文方向：Feature 0+1+2 数据足够支撑一篇 paper
  "Load-Adaptive Expert Parallelism on MI300X"
  对比 LAER-MoE（A100），突出 MI300X 的独特优势
```

### 时间估计
**实现：2~3 周 | 实验：3 周 | 总计 5~6 周**（业余 10~12 周）

---

## 4. Feature 3：静态 Overlap 调度（OverlapScheduler）

### 背景与假设

```
假设：目前 Primus/Megatron 中 A2A 和 Expert FFN 的 overlap 不充分，
      存在通信等待计算或计算等待通信的空泡。
      通过静态分析通信/计算的依赖关系，可以精确找到所有安全的 overlap 点，
      显著提升 GPU 利用率。

前提：Feature 0 CommReport 显示实际 overlap 率 < 理论上界 × 80%
      （说明当前调度有优化空间）

参考：FlowMoE（arXiv:2510.00207）的流水线调度思路，
      但 Axion 的设计是静态生成，FlowMoE 是动态的
```

### 实现方案

```python
# 接入方式：替换 Primus/Megatron 的 MoE dispatch/combine 调用方式
#
# 原始（串行）：
#   dispatched = all_to_all(tokens, routing)      # 阻塞等待
#   output = expert_ffn(dispatched, experts)
#   combined = all_to_all(output, routing)        # 阻塞等待
#
# OverlapScheduler（流水线，分 chunk）：
#   with overlap_scheduler as sched:
#       for i, chunk in enumerate(split_chunks(tokens, N_CHUNKS)):
#           dispatched_i = sched.async_a2a(chunk, routing)     # 非阻塞
#           if i > 0:
#               output_i_minus_1 = expert_ffn(dispatched[i-1]) # 与 A2A 重叠
#           dispatched.append(dispatched_i.wait())

class OverlapScheduler:
    """
    静态 chunk 流水线调度器。
    直接在 Primus/Megatron 的 MoE 前向中使用，
    不需要改 IR 或 Pass 系统。
    """
    def __init__(self, num_chunks=4, stream_mode='rccl'):
        self.num_chunks = num_chunks
        self.compute_stream = torch.cuda.Stream()
        self.comm_stream    = torch.cuda.Stream()

    def dispatch_with_overlap(self, tokens, routing, expert_fn):
        chunks = tokens.chunk(self.num_chunks, dim=0)
        results = []
        pending_comm = None

        for i, chunk in enumerate(chunks):
            # 发出本 chunk 的 A2A（在 comm stream 上，非阻塞）
            with torch.cuda.stream(self.comm_stream):
                next_comm = rccl_a2a_async(chunk, routing)

            # 计算上一个 chunk 的 Expert FFN（与本 chunk A2A 重叠）
            if pending_comm is not None:
                with torch.cuda.stream(self.compute_stream):
                    dispatched = pending_comm.wait()
                    results.append(expert_fn(dispatched))

            pending_comm = next_comm

        # 处理最后一个 chunk
        if pending_comm is not None:
            results.append(expert_fn(pending_comm.wait()))

        return torch.cat(results, dim=0)
```

### 收益假设与验证

```
预期吞吐提升：5~15%（取决于当前 A2A 时间占比和现有 overlap 率）

必做实验：

  实验 A：overlap 率对比
    设置：CommProfiler 在启用前后各跑 100 steps
    指标：实际 overlap 率（A2A 时间中有多少被 Expert FFN 覆盖）
    预期：overlap 率从当前值提升到 ≥ 80%

  实验 B：chunk 数量的影响
    设置：num_chunks = 1（关闭）/ 2 / 4 / 8
    指标：step time vs overlap 率
    预期：存在最优 chunk 数（太多 chunk 增加调度开销）

  实验 C：与 Feature 1+2 叠加
    对比：累计收益 vs 单独各自的收益（是否有叠加效果）

  实验 D：正确性验证
    对比：overlap 版本 vs 非 overlap 版本的输出（数值一致性）
    这是最重要的验证：chunk 拆分不能改变计算结果
```

### 决策门

```
if 实验 A overlap 率提升 < 5%（绝对值）：
  → 当前系统的 A2A 和 FFN 已经有较好 overlap
  → Feature 3 收益边际效应低
  → 评估是否值得继续 Feature 4（CommTensor zero-copy）

if 实验 C 三个 Feature 叠加后总提升 ≥ 20%：
  → 技术路线验证充分，可以开始考虑 Axion 构建阶段
  → 这是"进入 Axion 重写"的关键判断点之一

if 实验 D 发现数值不一致：
  → 立即停止，找 chunk 拆分的边界条件 bug
  → 不上生产，不做更多实验
```

### 技术产出

```
□ OverlapScheduler（Primus/Megatron 的 MoE 层替换方案）
□ chunk 大小自动选择（基于 MI300X 带宽参数的启发式规则）
□ 重叠调度的正确性证明（依赖关系分析文档）
□ 论文方向：Feature 3 是 Paper 2 的核心贡献
  "Static Communication-Computation Overlap for MoE on MI300X"
  关键差异：与 FlowMoE 对比（静态 vs 动态调度）
```

### 时间估计
**实现：2 周 | 实验：2~3 周 | 总计 4~5 周**（业余 8~10 周）

---

## 5. Feature 4：CommTensor zero-copy

### 背景与假设

```
假设：Expert dispatch/combine 的 pack/unpack 操作消耗了显著的内存带宽，
      通过在 A2A 之前直接使用通信友好的物理布局，可以消除这两次拷贝。

前提：Feature 0 CommReport 显示 A2A 时间中有明显的 pack/unpack 开销
      （在 MI300X 上：seq_len × hidden × 2 bytes × 2 = 约 58 MB per A2A for S=4096,H=7168）

注意：这个 Feature 不需要完整的 CommTensor 类型系统，
      在 Primus/Megatron 中只需要改内存分配策略即可验证核心假设。
```

### 实现方案

```python
# 核心思路：
# 传统：hidden [S, H]（按 token 顺序）→ pack（重排）→ A2A → unpack
# zero-copy：直接分配按目标 GPU 分组的内存 → A2A（直接 DMA）→ index map 访问
#
# 在 Primus/Megatron 中不需要完整 CommTensor 系统，
# 只需要在 dispatch 前改内存分配方式：

def dispatch_zero_copy(hidden, routing_table):
    # 直接分配按目标 rank 分组的 buffer
    # 物理内存：[rank0_tokens | rank1_tokens | ... | rankN_tokens]
    sorted_hidden, index_map = sort_by_dst_rank(hidden, routing_table)
    # sort_by_dst_rank 就是原来的 pack 操作，但直接输出到目标 buffer

    # A2A：direct DMA，无需额外 copy
    dispatched = rccl_a2a(sorted_hidden, routing_table.send_counts)

    # 返回 dispatched + index_map（供后续 combine 使用）
    return dispatched, index_map

def combine_zero_copy(expert_output, index_map, routing_table):
    # A2A combine：direct DMA
    combined_sorted = rccl_a2a(expert_output, routing_table.recv_counts)

    # 用 index_map 恢复原始顺序（index_select，仍然是一次 copy）
    # 这一步暂时无法完全消除，但比 unpack 高效
    return combined_sorted[index_map]
```

### 收益假设与验证

```
预期：在高 seq_len（S ≥ 4096）场景下，A2A 端到端时间降低 10~20%
      MI300X HBM3（5.3 TB/s）比 H100（3.35 TB/s）收益更大

必做实验：

  实验 A：pack/unpack 单独开销
    设置：隔离 pack 操作，用 hipperf 测量
    指标：pack 耗时（按 seq_len 扫描：1024/2048/4096/8192）
    预期：seq_len=4096，pack 约占 A2A 总时间 15~25%

  实验 B：zero-copy vs 原始的 A2A 端到端时间
    设置：相同 routing，对比两种实现的 A2A 总时间
    排除变量：使用相同 RCCL 配置

  实验 C：端到端 step time 提升
    设置：64 MI300X，接入 Primus/Megatron
    指标：tok/s 提升百分比

  实验 D：正确性验证
    对比：zero-copy 版本 vs 原始版本的 dispatch/combine 输出
```

### 决策门

```
if 实验 A pack 开销 < 5% of A2A 时间：
  → pack/unpack 不是显著瓶颈
  → Feature 4 的绝对收益有限（< 2% 端到端）
  → 不在 Primus/Megatron 上继续推进
  → 记录这个结论（CommTensor 的价值在 MI300X 上有限制条件）

if 实验 C 端到端提升 ≥ 3%：
  → 有增量价值，继续
  → 但注意：这个 Feature 实现复杂度高，需要 ROI 合理

重要认知：
  CommTensor 最大的价值不一定是 zero-copy 的绝对性能，
  而是它的类型系统设计（编译期保证 layout 正确性）。
  如果 zero-copy 性能收益不显著，这个 Feature 的价值
  主要体现在 Axion 构建阶段的设计严谨性，而非现在的性能。
```

### 技术产出

```
□ zero-copy dispatch/combine patch（Primus/Megatron）
□ MI300X pack/unpack 开销的精确量化（hipperf 数据）
□ CommTensor 设计的性能验证报告
  （证明或证伪：zero-copy 在 MI300X 上是否有显著价值）
□ 论文方向：作为 Paper 3 或 Feature 1~3 的一个 section
```

### 时间估计
**实现：2~3 周 | 实验：2 周 | 总计 4~5 周**（业余 8~10 周）

---

## 6. 后续 Feature（由前序数据驱动）

```
目前还没有足够数据来设计 Feature 5+。
根据 Feature 0~4 的结果，可能的方向：

  方向 A：RaggedShard（Dense 参数分片灵活化）
    适用条件：内部开始使用 Shampoo/Muon 优化器
    参考：veScale-FSDP 的核心贡献

  方向 B：FSEP 热点 Expert 分裂（一个 Expert 分片到多 GPU）
    适用条件：256+ GPU 场景，单个 Expert 参数量大
    参考：LAER-MoE FSEP 的完整版

  方向 C：跨节点通信拓扑优化
    适用条件：Feature 0 显示跨节点 A2A 占比 > 60%
    方案：Expert 初始分配拓扑感知，优先节点内路由

  方向 D：Sequence Parallelism 与 EP 的联合优化
    适用条件：长序列训练（S > 8192）成为主要场景

具体做哪个，等 Feature 0~4 的数据说话。
```

---

## 7. Axion 构建阶段（条件触发）

### 触发条件

```
同时满足以下条件，才开始 Axion 重写：

  □ Feature 0~3 中至少 3 个通过验证（有 ≥ 5% 各自收益）
  □ 累计端到端吞吐提升 ≥ 20%（对比 Primus/Megatron baseline）
  □ Feature 之间的 patch 开始出现耦合（维护成本上升）
  □ 有时间投入（不再是纯业余项目，或获得内部支持）

如果条件始终不满足：
  → Axion 构建阶段无限期推迟
  → 验证阶段的成果（各个 Feature patch）本身已有足够价值
```

### 构建阶段的意义

```
进入 Axion 构建阶段时，每个设计决策都有数据背书：

  ModelGraph + CommInferencePass：
    因为 Feature 0 证明了"通信可见性"有价值
    → 我们知道这个 Pass 应该分析什么

  CommTensor + CommTensorLayoutPass：
    因为 Feature 4 量化了 zero-copy 的收益
    → 我们知道 CommLayout 枚举是否需要 SPARSE_CSR

  OverlapInsertionPass：
    因为 Feature 3 证明了静态 chunk 调度有效
    → 我们知道 num_chunks 的合理范围

  FSEPShardingPass：
    因为 Feature 2 验证了 Expert 迁移的 ROI
    → 我们知道 imbalance_threshold 的合理值

  这些不是猜测，而是从真实 MI300X 训练数据中得到的参数。
  Axion 的设计是"基于证据的设计"，而非"基于直觉的设计"。
```

### 构建阶段概要（仅供参考，届时重新规划）

```
Stage A（约 3 个月）：
  ModelGraph + PassManager + AnalysisPass + FusionPass
  单机 MI300X 跑通 Llama 3.1 8B + DSv3-like MoE

Stage B（约 3 个月）：
  CommInferencePass + FSEPShardingPass
  OverlapInsertionPass + CommTensorLayoutPass
  DistributedExecutablePlan

Stage C（约 3 个月）：
  CommFabric RCCL Driver（完整版）
  CommTensor 运行时（zero-copy，index map）
  FSEP Slow/Fast Planner（用验证阶段的参数）

Stage D（按需）：
  新架构支持、更大规模、开源评估
```

---

## 8. 总时间线与里程碑

```
时间线（业余时间，假设每周 5~10 小时）：

Week 1~4    Feature 0：CommProfiler + CommReport
               ↓ 数据：确认瓶颈类型和优先级

Week 5~8    Feature 1：FastRouter（+收敛实验）
               ↓ 数据：路由均衡是否安全有效？

Week 9~16   Feature 2：SlowPlanner（+收敛实验）
               ↓ 数据：物理迁移 ROI？

Week 17~22  Feature 3：OverlapScheduler
               ↓ 数据：静态调度提升多少？

Week 23~28  Feature 4：CommTensor zero-copy
               ↓ 数据：pack/unpack 开销是否显著？

Week 28 后  决策点：
               □ 累计收益是否 ≥ 20%？
               □ Patch 是否开始耦合？
               □ 是否有内部支持？
               → Yes × 3：进入 Axion 构建阶段
               → 否则：停留在验证阶段，持续迭代
```

---

## 9. 技术产出规划

```
Feature 0~2 完成后（约 Week 16）：
  → 内部技术报告：MI300X MoE 训练的通信优化实践
  → 可选：AMD 技术博客（Feature 0+1+2 的数据）
  → 论文方向：MLSys 2027 投稿窗口（约 Week 20）
    "Load-Balanced MoE Training on MI300X: Observations and Optimizations"

Feature 3~4 完成后（约 Week 28）：
  → 内部报告：完整 Feature 0~4 收益拆分
  → 论文方向：OSDI/EuroSys 2027 投稿窗口
    "Towards Communication-Efficient MoE Training on AMD GPUs"

Axion 构建完成后（如果触发）：
  → 完整系统论文：OSDI/SOSP 2028
  → 开源评估（AMD IP 审查）
```

---

## 10. 每个 Feature 的独立可交付性

```
每个 Feature 是独立的——即使后续 Feature 不做，已完成的 Feature 也有价值：

  仅 Feature 0：
    → 内部工程师第一次能看清 MI300X MoE 通信瓶颈
    → 这本身就是有价值的工具

  Feature 0 + 1：
    → 5~15% 吞吐提升（如果收敛实验通过）
    → 一份有数据的内部技术报告

  Feature 0 + 1 + 2：
    → 预期 15~30% 累计吞吐提升
    → 有充分数据支撑一篇 paper

  Feature 0 + 1 + 2 + 3：
    → 预期 20~40% 累计吞吐提升
    → 足够支撑顶会投稿

  所有 Feature 完成 → 进入 Axion 构建阶段
  或者：所有 Feature 完成但不构建 Axion → 已经足够有价值
```

---

*Axion 整体规划 v0.3 | 2026-03-08*  
*核心思路：Feature-by-Feature 验证 → 数据驱动决策 → 有收益才继续 → Axion 是终点不是起点*
