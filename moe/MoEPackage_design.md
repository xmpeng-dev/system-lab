# MoEPackage: MoE-Native Infrastructure Whitepaper

> **定位**: 面向 AMD GPU 的 MoE 原生基础设施，不再以 Dense 框架为中心做补丁式适配  
> **目标**: 用控制平面 + 数据平面 + 执行平面的系统方法，重构 MoE 训练运行时  
> **硬件目标**: AMD MI300X / MI325X (XGMI + RDMA 双通道)  
> **版本**: v2.0 (MoE-native rewrite)  
> **更新**: 2026-03-12

---

## 0. 一句话定义

MoEPackage 不是“给 Dense 训练框架加 MoE 功能”，而是“把 MoE 视为分布式数据流系统”。

---

## 1. 为什么 Dense-first 架构不适合 MoE

### 1.1 根因不是算力，而是系统抽象错误

传统 Dense 训练框架的隐含前提:

- tensor 连续布局优先于通信布局
- layer 完成后才发起通信
- all-to-all 视为一个黑盒集合通信算子
- 参数位置 = 计算位置 = 优化器状态位置

这些前提对 Dense 模型成立，但对 MoE 会造成系统性问题:

- 路由后 token 的目的地高度离散，必须反复 permute/pack/unpack
- 关键路径由 tail expert 决定，平均吞吐掩盖不了 P99 延迟
- 通信不是“某一步操作”，而是贯穿 forward/backward 的持续流
- 负载变化快于静态并行配置，固定 expert 映射会持续失衡

### 1.2 结论

MoE 的核心对象不应是“层内 tensor”，而应是:

- token packet
- route state
- expert lease
- transport credit

---

## 2. MoE 原生系统模型

### 2.1 四个一等公民

1. **TokenPacket**
   - `payload`: hidden chunk
   - `route_epoch`: 路由版本
   - `dst_expert / dst_replica`
   - `priority`: latency-sensitive / throughput

2. **ExpertService**
   - expert 不再是静态参数块，而是可调度服务实例
   - 有 `capacity`, `queue`, `replica_count`, `health`

3. **RouteTable(epoch)**
   - 控制平面发布路由版本
   - 数据平面只消费稳定 epoch，避免每步全局同步

4. **CreditWindow**
   - 每个 expert replica 周期性发布可接收信用额度
   - router 只能向有 credit 的目的地发包

### 2.2 两层一致性语义

- **强一致控制平面**: RouteTable 切换按 epoch 原子生效
- **有界陈旧数据平面**: 允许 `epoch in [E, E-1]` 的短暂并存，确保不断流

---

## 3. 六平面架构

MoEPackage 用六个平面拆解 MoE 系统复杂性，避免“单 runtime 承担所有职责”。

### 3.1 Route Plane (控制平面)

职责:

- 根据 `affinity + topology_cost + credit` 计算路由
- 维护 `RouteTable(epoch)`
- 执行热度感知专家副本策略

核心接口:

- `publish_route_table(epoch, table)`
- `get_route(token_meta) -> (expert, replica, epoch)`
- `rebalance_trigger(metrics)`

### 3.2 Transport Plane (传输平面)

职责:

- token packet 分发与回收
- credit-based 流控
- 背压传播

核心机制:

- per-replica ingress queue
- per-link credit window
- congestion mark (ECN-like)

### 3.3 Execution Plane (执行平面)

职责:

- expert 计算调度
- grouped GEMM 批组拼接
- tail mitigation

核心机制:

- queue-aware launch
- short-job-first for micro-chunks
- replica stealing (仅在同节点)

### 3.4 Memory Plane (内存平面)

职责:

- packet buffer 生命周期管理
- forward/backward 统一缓冲池
- paged stashing 与重算窗口

核心机制:

- region-based allocator (route epoch aware)
- persistent communication slabs
- chunk-level release (不是 layer-level release)

### 3.5 State Plane (参数/优化器平面)

职责:

- 参数、梯度、优化器状态解耦管理
- expert 主副本与计算副本分离

核心机制:

- static optimizer ownership
- dynamic compute replica
- async gradient return + owner update

### 3.6 Observability & Fault Plane (可观测与容错平面)

职责:

- 实时暴露 tail、拥塞、负载、失衡指标
- 故障旁路和热恢复

核心机制:

- route epoch rollback
- replica quarantine
- degraded-mode routing

---

## 4. 两个核心协议

## 4.1 Protocol A: RouteTable Epoch Protocol

目标: 在不暂停训练的条件下更新路由策略。

阶段:

1. Planner 生成 `RouteTable(E+1)`
2. 控制平面广播 `prepare(E+1)`
3. 数据平面继续处理 `E`，并接受 `E+1` 预热缓存
4. 切换点到达后发 `commit(E+1)`
5. 允许短窗口消费 `E` 的在途包，然后 `retire(E)`

性质:

- 无全局 stop-the-world
- 路由切换可审计、可回滚
- 与训练 step 解耦

## 4.2 Protocol B: Credit-Based Expert Transport

目标: 让拥塞控制从“事后丢包/降速”变成“事前准入”。

流程:

1. 每个 expert replica 周期发布 credit (`tokens`)
2. router 发包前必须扣减 credit
3. expert 完成计算后返还 credit
4. credit 枯竭触发背压:
   - 同节点优先改路由
   - 再触发副本弹性租约
   - 最后才启用降级路径

收益:

- 明显降低 P99 tail
- 减少突发流量导致的 OOM
- 提高调度可解释性

---

## 5. AMD-first 设计点

这里不是“把 CUDA 设计翻译成 HIP”，而是利用 AMD 拓扑特性改变系统策略。

### 5.1 双通道通信模型

- XGMI 通道: 节点内高带宽低延迟，优先承载高频短包
- RDMA 通道: 跨节点大流量，承载远端包与梯度同步

调度原则:

- local-first route (在质量约束内优先同节点)
- path-separated queues (XGMI queue 与 RDMA queue 独立)
- no shared lock between channels

### 5.2 P2P 内存语义优先

将节点内 dispatch/combine 视为内存读写，不视为集合通信:

- 远端 buffer 映射到本地地址空间
- packet write 直接落远端 slab
- 取消节点内 all-to-all 黑盒调用

### 5.3 Expert 租约机制

当热点 expert 出现时，不做全局 re-layout，先做短周期租约:

- `lease(expert_id, gpu_id, ttl_steps)`
- TTL 到期自动回收
- 配合 credit window 可避免长期碎片化

---

## 6. MoEPackage Runtime API (草案)

```python
from moepackage import (
    init_runtime,
    RoutePolicy,
    TransportPolicy,
    ExpertPolicy,
    MoENativeLayer,
)

rt = init_runtime(
    topology="auto",          # detect xgmi/rdma graph
    route_epoch_len=64,       # steps per route epoch
    stale_epoch_budget=1,     # bounded staleness
)

route_policy = RoutePolicy(
    objective="quality_then_cost",
    topology_penalty={"local": 0.0, "xgmi": 0.1, "rdma": 0.4},
    credit_aware=True,
)

transport_policy = TransportPolicy(
    mode="credit_based",
    xgmi_queue_depth=4,
    rdma_queue_depth=8,
    backpressure="propagate",
)

expert_policy = ExpertPolicy(
    lease_enabled=True,
    lease_ttl=128,
    max_replicas_per_expert=4,
    optimizer_owner_static=True,
)

layer = MoENativeLayer(
    hidden_size=7168,
    num_experts=256,
    top_k=8,
    route_policy=route_policy,
    transport_policy=transport_policy,
    expert_policy=expert_policy,
)
```

---

## 7. 关键执行路径（非黑盒 all-to-all）

Forward:

1. Router 读取 RouteTable(E) + CreditWindow
2. 生成 TokenPacket 并直接写入目标 ingress slab
3. ExpertService 批组消费 queue，执行 grouped GEMM
4. 输出 packet 按源 token id 回传 combine slab
5. aggregator 按 packet metadata 做 weighted reduce

Backward:

1. dX 路径优先排队（保证上游尽快解锁）
2. dW 异步回 owner（State Plane）
3. 梯度同步在 RDMA 通道与 dX 计算重叠

---

## 8. MVP：最小可证明价值版本

为了尽快拿到不可辩驳数据，MVP 只做两件事:

1. RouteTable Epoch Protocol
2. Credit-Based Transport

范围:

- 只替换 MoE dispatch/combine，不动 attention
- 保持 expert 算子不变（先用现有 grouped GEMM）

对照:

- Baseline A: Megatron EP
- Baseline B: Megatron + DeepEP/HybridEP（若环境可用）

必须产出的三项指标:

- `P99 expert queue latency`
- `exposed comm time per step`
- `end-to-end step time`

目标:

- P99 latency 降低 >= 25%
- exposed comm 降低 >= 20%
- step time 降低 >= 8%

---

## 9. 实验与评测方法

### 9.1 指标层次

- **系统指标**: p50/p99 queue latency, credit starvation rate, backpressure depth
- **训练指标**: step time, tokens/s/GPU, MFU
- **模型指标**: loss curve, perplexity, route entropy

### 9.2 必须做的消融

1. 只开 RouteTable epoch
2. 只开 credit transport
3. 两者同时开启
4. 加不加 expert lease

### 9.3 规模建议

- Stage 1: 单节点 8 GPU
- Stage 2: 4 节点 32 GPU
- Stage 3: 16 节点 128 GPU

---

## 10. 相对 Megatron/DeepSpeed 的边界定义

MoEPackage 不替代训练框架，而是提供 MoE 原生控制/数据平面:

- 上层框架负责: 模型定义、优化器、checkpoint、pipeline
- MoEPackage 负责: route/transport/expert runtime

可插拔点:

- `dispatcher backend`
- `moe layer runtime`
- `route planner service`

---

## 11. 风险与工程现实

风险 1: 控制平面复杂度上升  
缓解: 先做单机控制器，再拆成分布式 planner

风险 2: credit 算法过于保守导致吞吐下降  
缓解: 提供 aggressive mode，自适应调 window

风险 3: route epoch 切换错误影响正确性  
缓解: 双读窗口 + commit/rollback + replay log

风险 4: 与现有框架集成成本过高  
缓解: 后端插件形态先行，不动训练主循环

---

## 12. 路线图

Phase 0 (2-4 周):

- transport slab + credit counter
- route epoch metadata pipeline
- 单节点可运行闭环

Phase 1 (4-8 周):

- 32 GPU 分布式运行
- p99 tail 数据闭环
- 第一版对比实验

Phase 2 (8-12 周):

- expert lease + state plane 解耦
- backward-first scheduling
- 128 GPU 端到端评测

Phase 3 (12+ 周):

- 与 FSEP / topology-aware placement 深度融合
- fault plane 在线恢复
- 论文与生产化准备

---

## 13. 核心亮点（新版）

这个版本的亮点不再是“某个 kernel 更快”，而是:

1. 把 MoE 从算子优化问题提升为分布式基础设施问题
2. 给出可实现的控制平面协议（RouteTable epoch）
3. 给出可实现的数据平面协议（credit transport）
4. 利用 AMD 双通道拓扑形成结构化优势
5. 提供可打穿的 MVP 和可量化验证路径

---

*MoEPackage Whitepaper v2.0 | 2026-03-12 | AIInfra-Book*  
*Keywords: MoE-native runtime, control plane, credit transport, route epoch, AMD-first topology*

