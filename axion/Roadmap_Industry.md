# Axion 工业落地计划

> **目标定位:** 在现有训练框架中产生实际训练收益，3 个月内可见 ROI  
> **核心策略:** 插件优先，不重写框架；验证收益后再决定深度投入  
> **总体时间:** 3 个月获得初步收益，6 个月有可量化的端到端提升  
> **团队假设:** 1~2 名工程师，可以访问内部 GPU 集群（64~256 GPU）

---

## 0. 工业落地的核心原则

### 0.1 与研究路线的根本差异

```
研究路线：
  目标 = 发论文，证明技术上"可以更好"
  接受：18 个月后才有完整结果
  接受：重新设计基础设施（ModelGraph、IR、Pass 系统）

工业落地路线：
  目标 = 生产训练任务提速，3 个月内有可量化收益
  不接受：完全重写现有训练框架
  约束：必须与 Megatron-LM / veScale / 内部框架兼容
  约束：不能影响训练稳定性（收敛性红线）
```

### 0.2 三级投入策略

```
Level 1（低成本，1~2 周）：可观测性插件
  → 不改任何训练逻辑，只加监控
  → 立即获得"通信热点可见"的价值

Level 2（中等成本，1~2 月）：负载均衡插件
  → 在现有框架上挂载 FSEP Slow Planner
  → 预期 10~25% 吞吐提升

Level 3（高成本，3~6 月）：CommTensor 集成
  → 在现有框架中替换 Expert dispatch/combine 的内存管理
  → 预期额外 5~15% 提升

原则：每个 Level 独立交付，Level N+1 的决策取决于 Level N 的收益。
```

---

## 1. Level 1：通信可观测性插件（第 1~2 周）

### 目标

不改任何训练逻辑，在现有框架（Megatron/veScale）上增加通信监控，  
让工程师第一次能**看清**通信瓶颈在哪。

### 为什么先做这个

```
现有痛点（每个做过大规模 MoE 训练的团队都有）：

  "为什么这个训练任务比上次慢了 15%？"
  → 不知道是计算变慢了还是通信变慢了
  → 不知道哪个 Expert 是热点
  → 不知道 A2A 的 overlap 率实际是多少

Level 1 直接解决这个问题，不需要改任何训练代码。
```

### 具体实现

```python
# 以 hook 方式接入现有框架，不改任何模型代码

class AxionCommProfiler:
    """
    通信可观测性插件。
    接入方式：在 MoE 层的 dispatch/combine 前后挂 hook。
    """

    def __init__(self, model, config: ProfilerConfig):
        self.stats = CommStats()
        self._attach_hooks(model)

    def _attach_hooks(self, model):
        for layer in model.moe_layers():
            # dispatch 前后计时
            layer.register_forward_pre_hook(self._on_dispatch_start)
            layer.register_forward_hook(self._on_dispatch_end)

    def report(self) -> CommReport:
        return CommReport(
            # 每个 Expert 收到的 token 数（负载分布）
            expert_load_distribution = self.stats.expert_loads,

            # A2A 实际耗时 vs 计算耗时（overlap 效果）
            a2a_latency_ms    = self.stats.a2a_times,
            compute_latency_ms = self.stats.compute_times,
            overlap_ratio     = self.stats.compute_overlap_ratio(),

            # 跨节点通信量
            cross_node_bytes  = self.stats.cross_node_bytes,
            intra_node_bytes  = self.stats.intra_node_bytes,

            # 热点 Expert 列表
            hot_experts = self.stats.top_k_experts(k=20),
        )
```

### 输出样例

```
=== Axion Comm Report（Step 100~200 平均）===

Expert 负载分布：
  avg tokens/expert:   256
  max tokens/expert:   891 (expert_id=47, 3.48x 平均)  ← 热点
  min tokens/expert:   12  (expert_id=203, 0.05x 平均) ← 冷点
  负载不均衡系数:       3.48

通信/计算重叠：
  A2A 平均耗时:         8.2ms
  Expert FFN 平均耗时:  6.1ms
  实际 overlap 率:      41.3%  ← 远低于理论最大值 74%
  ⚠️  overlap 率偏低，存在优化空间

跨节点通信：
  跨节点 A2A 占比:      68.2%  ← 偏高，Expert 分布不均
  节点内 A2A 占比:      31.8%

推荐：
  1. Expert 47 是严重热点，建议 FSEP 分片或路由偏置
  2. overlap 率 41% 远低于 74% 理论值，A2A chunk 调度待优化
  3. 跨节点通信占 68%，初始 Expert 分布待优化
```

### 交付物与验收

```
交付：
  □ AxionCommProfiler（Python package，pip install 即用）
  □ 支持 Megatron-LM / veScale 的 hook 接入
  □ CommReport HTML 可视化（Expert 负载热力图 + 时序图）
  □ 与现有 NCCL profiler 数据对齐验证

验收：
  □ 在内部 64+ GPU 训练任务上成功接入
  □ CommReport 中 Expert 负载分布与人工 profiling 结果一致
  □ 接入开销 < 0.5%（profiling 本身不拖慢训练）
```

---

## 2. Level 2：负载均衡插件（第 3~8 周）

### 目标

在 Level 1 的监控数据基础上，增加 FSEP Slow Planner，  
自动调整 Expert 物理分布，消除负载不均衡。

### 两个子方案（根据框架支持程度选择）

#### 方案 A：路由偏置（最轻量，1 周）

```python
class AxionFastRouter:
    """
    方案 A：仅调整 routing logits，不移动 Expert 参数。
    侵入性最低，风险最小。
    需要收敛实验验证不影响模型质量。
    """
    def __init__(self, alpha: float = 0.1, beta: float = 2.0):
        self.alpha = alpha
        self.beta  = beta
        self.load_stats = ExpertLoadStats()

    def adjust_gate_logits(
        self,
        gate_logits: torch.Tensor,   # [num_tokens, num_experts]
    ) -> torch.Tensor:
        """在 MoE Gate 之后、TopK 之前调用"""
        load_penalty = self._compute_penalty()
        return gate_logits - self.alpha * load_penalty

    def _compute_penalty(self) -> torch.Tensor:
        avg = self.load_stats.avg_load
        ratio = (self.load_stats.expert_loads / avg) ** self.beta
        return ratio * self.load_stats.gpu_utilization

    def update(self, routing_table: RoutingTable):
        """每 step 更新负载统计（在 Expert dispatch 之后调用）"""
        self.load_stats.update(routing_table)
```

```
方案 A 接入方式（Megatron 示例）：

  # 原始代码
  scores, indices = torch.topk(gate_logits, k=self.topk)

  # 接入 Axion（一行改动）
  gate_logits = axion_router.adjust_gate_logits(gate_logits)
  scores, indices = torch.topk(gate_logits, k=self.topk)

预期收益：5~15% 吞吐提升（间接均衡，软性约束）
风险：可能影响收敛，需要消融实验
```

#### 方案 B：Expert 物理迁移（较重，3~4 周）

```python
class AxionSlowPlanner:
    """
    方案 B：定期重新规划 Expert 物理分布。
    需要框架支持 Expert 参数的 P2P 搬迁。
    借鉴 LAER-MoE 的迁移执行器。
    """
    def __init__(self, check_interval: int = 50):
        self.check_interval  = check_interval
        self.load_history    = LoadHistory(window=check_interval)
        self.current_plan    = ExpertPlacementPlan.round_robin()

    def maybe_replan(
        self,
        step: int,
        routing_stats: RoutingStats,
        topology: ClusterTopology,
    ) -> ExpertMigrationPlan | None:
        """每 check_interval step 调用一次"""
        self.load_history.update(routing_stats)

        if step % self.check_interval != 0:
            return None

        imbalance = self.load_history.imbalance_ratio()
        if imbalance < 1.3:   # 不均衡度 < 30%，不触发迁移
            return None

        new_plan = self._solve_greedy(
            load_profile = self.load_history.avg_profile(),
            topology     = topology,
        )
        return ExpertMigrationPlan.diff(self.current_plan, new_plan)

    def _solve_greedy(self, load_profile, topology) -> ExpertPlacementPlan:
        """
        贪心分配：将热点 Expert 迁移到负载较低的 GPU。
        约束：
          - 每次迁移不超过 8 个 Expert（避免大规模迁移冲击稳定性）
          - 迁移收益/通信开销 > 1.1（ROI 约束）
          - 优先节点内迁移（NVLink，代价低）
        """
        ...
```

```
方案 B 接入方式：

框架需要支持两个 hook：
  1. after_backward_hook：触发 P2P Expert 参数搬迁
  2. before_forward_hook：更新 Expert 位置路由表

Megatron 支持度：需要约 2 周改造
veScale：更容易，已有类似机制

预期收益：15~30% 吞吐提升（直接均衡，物理迁移）
风险：迁移期间内存峰值增加 5~10%
```

### 收敛性验证（必做）

```
在做任何 routing 干预之前，必须做收敛实验：

实验设置：
  模型：2B 参数 MoE（64 experts，快速验证）
  数据：内部预训练数据，1000 steps
  对比：
    Baseline：原始 TopK routing
    方案 A：+ routing bias（α=0.1, β=2.0）
    方案 B：+ Slow Planner Expert 迁移

指标：
  □ Loss curve 对齐（最终 loss 差异 < 0.5%）
  □ Expert 激活分布（方案 A 不改变 Expert 的功能特化）
  □ 梯度 norm（不出现梯度爆炸/消失）

红线：任何方案如果 loss 比 baseline 高 > 1%，立即停止。
```

### 交付物与验收

```
交付（方案 A + B 均实现，根据收敛实验选择生产部署哪个）：
  □ AxionFastRouter（routing bias，pip install 即用）
  □ AxionSlowPlanner（Expert 迁移，需框架适配）
  □ 收敛实验报告（2B 模型，1000 steps）
  □ 在内部框架的接入文档

验收：
  □ 64 GPU，DSv3-like 模型，吞吐提升 ≥ 10%
  □ 收敛实验：loss 差异 < 0.5%
  □ 迁移期间无 OOM（内存峰值增加 < 10%）
```

---

## 3. Level 3：CommTensor 集成（第 9~24 周）

### 目标

在内部框架中替换 Expert dispatch/combine 的内存管理，  
引入 CommTensor 零 copy，消除 pack/unpack 开销。

### 前置决策门

```
只有 Level 2 满足以下条件，才进入 Level 3：

  □ Level 2 吞吐提升 ≥ 10%（证明负载均衡方向正确）
  □ Level 1 CommReport 显示 A2A 时间占比 ≥ 20%
    （如果 A2A 不是瓶颈，CommTensor 优化 ROI 低）
  □ 团队愿意投入 3~6 个月工程量
```

### 实现策略：渐进替换

```
不要一次性替换整个 All-to-All 路径。
分三步，每步独立交付，每步都有可量化收益。

Step 3a（3~4 周）：消除 dispatch pack copy
  只替换 Expert dispatch 前的 pack 操作。
  dispatch 后的 unpack 暂时保留（不改变接收端）。
  → 预期节省约一半的 pack/unpack 开销

Step 3b（3~4 周）：消除 combine unpack copy
  在 Step 3a 基础上，同时消除 combine 后的 unpack。
  → 预期节省剩余的 pack/unpack 开销

Step 3c（4~6 周）：与 Slow Planner 联动
  CommTensor 的 index map 在 Expert 迁移后需要更新。
  实现 index map 原子更新机制，与 AxionSlowPlanner 集成。
  → 实现完整的 CommTensor + FSEP 协同路径
```

### 接入方式（以 Megatron 为例）

```python
# 原始 Megatron MoE dispatch（简化）
def expert_dispatch(hidden, routing_table):
    # pack：按目标 Expert GPU 重排内存
    packed = pack_tokens(hidden, routing_table)          # ← 替换这里
    # All-to-All
    dispatched = all_to_all(packed, routing_table)
    return dispatched

# 接入 CommTensor 后
def expert_dispatch_axion(hidden, routing_table):
    # 直接构造 CommTensor（物理内存已经按 BLOCKED_BY_DST 分配）
    comm_tensor = CommTensor.from_dense(
        hidden,
        layout  = CommLayout.BLOCKED_BY_DST,
        routing = routing_table,
    )
    # All-to-All：直接 DMA，无需 pack
    dispatched = comm_fabric.all_to_all(
        comm_tensor,
        recv_layout = CommLayout.BLOCKED_BY_SRC,
    )
    return dispatched
    # Expert FFN 通过 dispatched.logical_view 访问，零 copy
```

### 交付物与验收

```
交付：
  □ CommTensor Python 实现（含 index map 生成）
  □ CommFabric NVLink 实现（基于 NCCL All-to-All + GPUDirect）
  □ 与 AxionSlowPlanner 的联动接口
  □ Megatron / veScale 接入 patch

验收：
  □ Step 3a：pack copy 时间减少 ≥ 40%
  □ Step 3b：总 pack/unpack 时间减少 ≥ 70%
  □ Step 3c：Expert 迁移后训练可以正确继续（无 index map 错误）
  □ 端到端吞吐（Level 2 + Level 3）较 baseline 提升 ≥ 20%
```

---

## 4. 完整路线时间线

```
Week 1-2   [Level 1]  AxionCommProfiler 接入
               ↓      CommReport 验证通信热点
Week 3-4   [Level 2a] AxionFastRouter（routing bias）
               ↓      收敛实验（2B 模型，1000 steps）
Week 5-8   [Level 2b] AxionSlowPlanner（Expert 迁移）
               ↓      64 GPU 端到端验证
               ↓
            ─────── 决策门：Level 2 收益 ≥ 10%？ ────────
                                    │
               ┌────── Yes ─────────┴──────── No ──────────┐
               ▼                                           ▼
Week 9-12  [Level 3a] CommTensor dispatch         停止，Level 1+2 已足够
Week 13-16 [Level 3b] CommTensor combine
Week 17-24 [Level 3c] Slow Planner + CommTensor 联动
               ↓
Week 24    总结：基于 CommReport 数据，评估是否继续深入
```

---

## 5. 与现有框架的兼容性矩阵

| 框架 | Level 1 接入 | Level 2a 接入 | Level 2b 接入 | Level 3 接入 | 主要挑战 |
|------|-------------|--------------|--------------|-------------|---------|
| **Megatron-LM** | ✅ hook | ✅ gate 修改 | ⚠️ 需改 EP 层 | ⚠️ 需改内存分配 | EP 层耦合度高 |
| **veScale** | ✅ hook | ✅ gate 修改 | ✅ 已有类似机制 | ✅ RaggedShard 可复用 | 较小 |
| **DeepSpeed** | ✅ hook | ✅ gate 修改 | ⚠️ ZeRO 状态管理 | ❌ 内存模型不兼容 | ZeRO 深度耦合 |
| **内部框架** | 视情况 | 视情况 | 视情况 | 视情况 | 取决于 EP 实现 |

---

## 6. ROI 估算

```
假设：DSv3-like 模型（256 experts），64 GPU，bf16 训练

Level 1（1~2 周工程量）：
  直接收益：0（只是监控）
  间接收益：发现并量化通信瓶颈，指导后续优化方向
  ROI：极高（低成本，高信息价值）

Level 2a - routing bias（1 周工程量）：
  预期吞吐提升：5~15%
  风险：收敛影响（需 1 周收敛实验）
  ROI：高（前提是收敛实验通过）

Level 2b - Expert 迁移（3~4 周工程量）：
  预期吞吐提升：15~25%（在 Level 2a 基础上额外）
  风险：内存峰值、框架改造复杂度
  ROI：较高（收益明确，成本中等）

Level 3 - CommTensor（12~16 周工程量）：
  预期吞吐提升：5~15%（在 Level 2 基础上额外）
  风险：框架深度改造，稳定性风险
  ROI：中等（收益较小，成本较高）
  建议：只有在 Level 1+2 已经运行稳定后才做

综合（Level 1+2 完成后）：
  工程投入：约 2 个月（1 名工程师）
  预期端到端吞吐提升：20~35%
  这个收益在大规模训练中直接等于 GPU 成本节省
```

---

## 7. 风险与红线

### 红线（任何一条触发，立即停止相应 Level）

```
红线 1：收敛性
  任何 routing 干预导致 loss 比 baseline 高 > 1% → 停止 Level 2a
  任何 Expert 迁移导致梯度异常 → 停止 Level 2b

红线 2：稳定性
  引入任何 Level 后，训练异常中断率 > baseline × 1.1 → 回滚

红线 3：内存
  任何 Level 导致 OOM（GPU 内存不足）→ 调整实现，不强推

红线 4：性能
  Level 2 完成后吞吐提升 < 5% → 不进入 Level 3
```

### 已知风险

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| routing bias 影响收敛 | 40% | Level 2a 放弃 | 直接上 Level 2b（不用 routing bias） |
| Megatron EP 层改造困难 | 30% | Level 2b 延期 | 改用 veScale 作为主要支撑框架 |
| CommTensor index map 性能不达标 | 20% | Level 3 收益缩水 | 降低预期，标注为"内存友好"而非"零 copy" |
| 内部框架与接入方式不兼容 | 25% | 需要专项适配 | Level 1 阶段就评估框架兼容性，提前识别 |

---

## 8. 与研究路线的协作关系

```
工业路线和研究路线不是互斥的，可以并行推进：

  研究路线（学术论文）          工业路线（内部落地）
       │                              │
       │  Phase 0-1（基础设施）        │  Level 1（监控）
       │  → 验证 CommTensor 类型系统   │  → 发现真实瓶颈
       │                              │
       │  Phase 2（静态 Overlap）      │  Level 2（负载均衡）
       │  → 精确的依赖分析算法         │  → 用贪心算法快速交付
       │                              │
       └──── 相互验证 ────────────────┘
  
  工业路线的真实数据 → 支撑研究论文的实验
  研究路线的严格算法 → 提升工业路线的效果

  具体协作点：
  ① Level 1 的 CommReport 数据是 Paper 1 实验的真实依据
  ② Level 2b 的 Slow Planner 是 Paper 2 算法的工程实现
  ③ Level 3 的 CommTensor 实现是 Paper 3 的实验平台
```

---

*工业落地计划 v0.1 | 2026-03-08*
