# Feature 2 详细设计：SlowPlanner

> **版本:** v0.1 | 2026-03-08  
> **参考:** LAER-MoE (arXiv:2402.xxxxx)

---

## 1. 问题定义

FastRouter（Feature 1）通过偏置调整来**软性**引导路由，但无法从根本上消除负载不均衡：
- 某些 token pattern 对特定 Expert 有强偏好，偏置调整会影响模型质量
- FastRouter 的均衡上界受限于 Expert 质量差异

SlowPlanner 的思路是：**直接移动 Expert 参数**，将过载 Expert 迁移到负载较轻的 GPU，让后续的路由自然地在新布局下均衡。

---

## 2. 算法设计

### 2.1 触发与规划流程

```
每 check_interval 步：
  1. 收集过去 K 步的 Expert 负载统计（来自 Feature 0 CommProfiler 或轻量版收集）
  2. 计算当前负载不均衡系数
  3. if 不均衡系数 < threshold：跳过，不迁移
  4. else：贪心规划迁移方案
  5. 异步执行迁移（P2P，与下一 step 重叠）
```

### 2.2 贪心规划算法

```python
def greedy_plan(load_per_gpu: List[float], experts_per_gpu: List[List[int]]):
    """
    简单贪心：将负载最高 GPU 的 Expert 逐个移到负载最低 GPU，
    直到两者负载差距小于 threshold。

    时间复杂度：O(N_experts × log N_gpus)，不需要 ILP。
    """
    plan = []  # [(src_gpu, dst_gpu, expert_id), ...]

    heap_high = max_heap(load_per_gpu)  # 负载最高的 GPU
    heap_low  = min_heap(load_per_gpu)  # 负载最低的 GPU

    while True:
        src_gpu, src_load = heap_high.pop()
        dst_gpu, dst_load = heap_low.pop()

        if src_load / dst_load < imbalance_threshold:
            break  # 足够均衡了

        # 迁移 src_gpu 上负载最高的 Expert
        expert_id = find_hottest_expert(src_gpu, experts_per_gpu)
        plan.append((src_gpu, dst_gpu, expert_id))

        # 更新负载估算（迁移后的预期负载）
        expert_load = get_expert_load(expert_id)
        heap_high.push(src_gpu, src_load - expert_load)
        heap_low.push(dst_gpu, dst_load + expert_load)

    return plan
```

### 2.3 迁移执行（P2P 异步）

```python
def execute_migration(model, plan, ep_group):
    """
    用 dist.isend/irecv 异步 P2P 迁移 Expert 参数。
    发送方和接收方同时启动，等待完成后更新 Expert 的 GPU 归属映射。
    """
    handles = []
    for src_rank, dst_rank, expert_id in plan:
        expert_params = get_expert_params(model, expert_id)  # List[Tensor]

        local_rank = dist.get_rank(ep_group)

        if local_rank == src_rank:
            for param in expert_params:
                h = dist.isend(param.data, dst=dst_rank, group=ep_group)
                handles.append(h)

        elif local_rank == dst_rank:
            new_params = [torch.empty_like(p) for p in expert_params]
            for p in new_params:
                h = dist.irecv(p, src=src_rank, group=ep_group)
                handles.append(h)

    # 等待所有传输完成（在下一个 step 开始前）
    for h in handles:
        h.wait()

    # 更新 Expert 参数和 GPU 归属映射
    update_expert_assignment(model, plan)
```

---

## 3. MI300X 特性利用

### 3.1 节点内迁移优先（Infinity Fabric / XGMI）

MI300X 8 GPU 节点内，GPU 之间通过 Infinity Fabric 连接：
- 节点内带宽：~700 GB/s（GPU 间 XGMI）
- 跨节点带宽：~400 Gbps（ROCEv2 / InfiniBand）

**策略**：优先执行节点内迁移，跨节点迁移作为最后手段。

```python
def plan_with_topology(load_per_gpu, topology):
    """
    分两轮规划：
    Round 1：只考虑节点内迁移（带宽高，延迟低）
    Round 2：如果节点内无法满足均衡目标，才考虑跨节点迁移
    """
    plan = []
    plan += intra_node_greedy_plan(load_per_gpu, topology)
    if compute_imbalance(plan) > threshold:
        plan += inter_node_greedy_plan(load_per_gpu, topology)
    return plan
```

### 3.2 迁移开销估算

典型 Expert 参数量（以 DSv3-like MoE 为例）：
- FFN hidden = 7168，intermediate = 2048，top_k = 8
- 每个 Expert 参数量 ≈ 2 × 7168 × 2048 × 2 bytes ≈ 58 MB（bf16）
- 节点内迁移耗时：58 MB / 700 GB/s ≈ **0.08 ms**
- 跨节点迁移耗时：58 MB / 50 GB/s ≈ **1.2 ms**

典型 step time ≈ 100~300 ms，节点内迁移开销 < 0.1%。

---

## 4. 收敛风险分析

### 4.1 风险来源

Expert 迁移改变了模型的物理分布，但不改变**路由语义**（token → expert 的映射不变）。
理论上不影响收敛，但实践中有以下风险：

1. **迁移时刻的梯度不一致**：迁移在 optimizer.step() 之后、下一次 forward 之前
   - 迁移时 Expert 参数和其对应的 optimizer state（momentum, variance）都需要同步迁移
   - 如果只迁移参数而不迁移 optimizer state → 迁移后的第一个 step 梯度更新异常

2. **并发问题**：异步 P2P 迁移与下一 step 的 forward 并发
   - 必须确保迁移完成（`.wait()`）后，下一个 forward 才能访问迁移的 Expert

### 4.2 缓解措施

```python
# 迁移时，Expert 参数 + 对应的 optimizer state 一起迁移
def migrate_expert_with_optimizer(model, optimizer, expert_id, src, dst):
    # 迁移参数
    expert_params = get_expert_params(model, expert_id)
    migrate_tensors(expert_params, src, dst)

    # 迁移 optimizer state（Adam: exp_avg, exp_avg_sq）
    for param in expert_params:
        if param in optimizer.state:
            state = optimizer.state[param]
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    migrate_tensors([v], src, dst)
```

---

## 5. 与 LAER-MoE 的区别

| 维度 | LAER-MoE | SlowPlanner（本方案） |
|------|----------|----------------------|
| 硬件 | A100（NVLink） | MI300X（XGMI/ROCEv2） |
| 迁移粒度 | Expert 级 | Expert 级（相同） |
| 规划算法 | ILP（精确） | 贪心（近似，实现更简单） |
| 迁移时机 | 训练暂停后迁移 | 异步 P2P，与 step 重叠 |
| optimizer state 处理 | 文中未提及 | 明确迁移（本方案关键改进） |
