# MoEX 异构并行设计

> MoEX 支持完整的五维并行（TP + CP + DP + PP + EP），并在此基础上融合
> FSEP（Fully Sharded Expert Parallel）和动态 Expert Re-layout（LAER-MoE），
> 实现 Attention 与 MoE 层的独立并行配置（MoE Parallel Folding 风格）。

---

## 目录

1. [并行模型总览](#1-并行模型总览)
2. [Attention/MoE 解耦并行（Parallel Folding）](#2-attentionmoe-解耦并行parallel-folding)
3. [FSEP：完全分片专家并行](#3-fsep完全分片专家并行)
4. [动态 Expert Re-layout（LAER-MoE 接口）](#4-动态-expert-re-layoutlaer-moe-接口)
5. [CPU+GPU 异构（推理扩展）](#5-cpugpu-异构推理扩展)
6. [网络拓扑感知](#6-网络拓扑感知)
7. [CommTensor 与并行模式的对应关系](#7-commtensor-与并行模式的对应关系)
8. [并行配置指南](#8-并行配置指南)

---

## 1. 并行模型总览

### 1.1 MoEX 支持的并行维度

```
MoEX 并行维度（7 维）：

Attention 侧：
  TP (Tensor Parallel)：分割 Q/K/V 矩阵的 head 维度
  CP (Context Parallel)：分割序列长度（Ring Attention）
  DP (Data Parallel)：复制模型，处理不同 batch

MoE 侧（独立于 Attention！）：
  EP  (Expert Parallel)：专家分布在不同 GPU
  EDP (Expert Data Parallel)：专家模型的数据并行副本
  FSEP (Fully Sharded EP)：专家参数跨所有 GPU 分片

共享：
  PP (Pipeline Parallel)：跨层切分，两侧必须一致

关系约束：
  PP 两侧相同（必须）
  TP × CP × DP = EP × EDP（同一 PP stage 内 GPU 数相同）
  FSEP ⊆ EP（FSEP group 是 EP group 的子集，通常是节点内 GPU）
```

### 1.2 与传统 MegatronCore 的对比

```
传统 MegatronCore EP 约束：
  - EP ⊆ DP（Expert Parallel 必须在 Data Parallel 范围内）
  - Attention 和 MoE 共享相同的 TP, CP, DP 配置
  - 限制：高 TP 对 Attention 好，但对 MoE 代价高（TP 内需要 All-Reduce）

MoEX 解耦（MoE Parallel Folding 风格）：
  - EP 不再受 DP 约束（可以比 DP 大！）
  - Attention 和 MoE 独立配置 TP/EP
  - 只有 PP 必须相同

示例对比（64 GPU，单 PP stage）：
  传统：TP=8, DP=8, EP=8（EP ⊆ DP，EP ≤ DP）
    Attention：TP=8（合理）
    MoE：EP=8（每 GPU 8 个专家，通信量适中）
    
  MoEX：Attention TP=8, CP=4, DP=2；MoE EP=32, EDP=2
    Attention：TP=8, CP=4（长序列支持！），DP=2
    MoE：EP=32（更均匀的专家分布，更少的负载不均衡）
    → 同样 64 GPU，更高利用率！
```

---

## 2. Attention/MoE 解耦并行（Parallel Folding）

### 2.1 Fold 操作（Attention 输出 → MoE 输入）

```
Attention 输出的 tensor 布局（TP 并行视角）：

假设：TP=4, Attention 组 {GPU 0, 1, 2, 3}

每个 GPU 持有：hidden[rank, :, H/TP...(rank+1)*H/TP]
                ↑         ↑    ↑
             GPU ID   tokens  TP 分片的 H 维度

全量 Attention 输出：hidden [B*L, H]（需要 TP All-Reduce 才能合并）

MoE 所需的布局（EP 并行视角）：

EP=16, MoE 组 {GPU 0..15}（包含之前 Attention 的 4 个 GPU）

每个 GPU 需要：token 的子集（按 expert 路由分配）
CommTensor: [R=16, S, T_tiles, H]
             ↑
            EP rank（MoE 视角）
```

### 2.2 MoEX 的 Fold 实现（合并 TP All-Reduce 与 CommTensor 构建）

```
传统 Fold（来自 MoE Parallel Folding 论文）：
  Step 1: TP All-Reduce → 获得完整 hidden [B*L, H]
  Step 2: Gate GEMM（全量 hidden）
  Step 3: Fold All-to-All → EP layout
  Step 4: Dispatch All-to-All（另一次通信！）
  总计：2 次 All-to-All + 1 次 All-Reduce

MoEX Fold（融合优化）：
  Step 1: Gate GEMM（在 TP 分片上，无需先 All-Reduce！）
          → 每个 TP GPU 算出 partial gate logits [B*L, num_experts/TP]
  Step 2: TP All-Reduce on gate logits（比 hidden All-Reduce 小 TP 倍！）
          → 完整 gate logits [B*L, num_experts]
  Step 3: TopK + CommTensor meta 填写（本地）
  Step 4: CommTensor Dispatch（1 次 All-to-All/RDMA，零拷贝）
  总计：1 次 All-Reduce（小）+ 1 次 All-to-All

节省：1 次 All-to-All（Fold）+ All-Reduce 从 H 维降低到 num_experts 维
```

### 2.3 CommTensor 如何天然支持 Fold

```python
def fold_and_route(
    hidden_partial: Tensor,           # [B*L, H/TP]（TP 分片）
    W_gate_partial: Tensor,           # [H/TP, num_experts]（TP 分片）
    config: MoEXConfig,
    process_groups: MoEXProcessGroups,
) -> CommTensor:
    """
    融合的 Fold + Route，直接输出 CommTensor
    避免了 Fold All-to-All 和额外的 hidden All-Reduce
    """
    # Step 1: partial Gate GEMM（在 TP 分片上）
    partial_logits = hidden_partial @ W_gate_partial  # [B*L, num_experts]

    # Step 2: All-Reduce on gate logits（比 hidden small TP 倍）
    dist.all_reduce(partial_logits, group=process_groups.tp_group)

    # Step 3: TopK
    scores, expert_ids = partial_logits.topk(config.top_k, dim=-1)
    scores = torch.softmax(scores, dim=-1)

    # Step 4: 构建 CommTensor（需要完整 hidden，此处进行 All-Reduce）
    # 注：hidden 的 All-Reduce 可以 overlap 后续操作
    # 或者：在 TP group 内用 All-Gather 替代 All-Reduce（获得 [B*L, H]）
    hidden_full = all_gather_along_last_dim(hidden_partial, process_groups.tp_group)

    # Step 5: 填充 CommTensor（直接按 EP rank 写入）
    ct = CommTensor.from_hidden_states(hidden_full, expert_ids, scores, config)

    return ct
    # 此 ct 可直接 dispatch，无需额外 Fold All-to-All
```

### 2.4 Unfold 操作（MoE 输出 → Attention 输入）

```
Combine 的输出已是 sequence-ordered：
  output: Tensor[B*L, H]（通过 scatter_add 写入原始位置）

但如果下一层 Attention 需要 TP-sharded 输入：
  需要将 output 切分为 TP 份

MoEX Unfold 实现：
  # 方案 1：直接切片（零通信，适用于 Attention TP=1 或 DP 模式）
  hidden_partial = output[:, rank * H//TP : (rank+1) * H//TP]

  # 方案 2：Scatter（如果 Attention TP 需要特定 head 分配）
  hidden_partial = output.chunk(TP, dim=-1)[tp_rank]

  # 方案 3：Reduce-Scatter（如果 Combine 使用了多副本，合并并切分）
  hidden_partial = dist.reduce_scatter(output, group=tp_group)
```

---

## 3. FSEP：完全分片专家并行

### 3.1 FSEP 原理回顾（来自 LAER-MoE）

```
传统 EP：
  Expert E0 完整地存在 GPU 0
  Expert E1 完整地存在 GPU 1
  ...
  
  当 E0 负载高（token 多）时：
    GPU 0 过载，其他 GPU 空闲 → 利用率 1/r（r=负载不均衡比）

FSEP（Fully Sharded Expert Parallel）：
  Expert E0 的参数分片在所有 EP GPU 上：
    GPU 0: W_E0[:, 0:H/R]（前 1/R 列）
    GPU 1: W_E0[:, H/R:2H/R]（第 2 列）
    ...
    GPU R-1: W_E0[:, (R-1)H/R:H]（最后 1/R 列）

  每个 GPU 处理所有 tokens（均匀负载）：
    All tokens → 每 GPU 各自计算 partial_expert_out（使用本地 W 分片）
    ReduceScatter → 每 GPU 获得 1/R token 的完整输出

  利用率：接近 100%（负载均衡！）
```

### 3.2 FSEP 的通信开销

```
通信量对比（per forward pass，T = total tokens, H = d_model）：

传统 EP：
  Dispatch All-to-All：T × H × 2 bytes
  Combine All-to-All：T × H × 2 bytes
  总：2TH（但利用率 1/r，等效 2rTH 的计算量）

FSEP：
  Dispatch (Broadcast)：T × H × 2 bytes（每 token 发给所有 GPU）
  ReduceScatter (NVLink)：T × H × 2 bytes（NVLink！）
  Combine All-to-All：T × H / R × 2 bytes × R = T × H × 2 bytes（但量均匀）
  总通信量：3TH（多了 ReduceScatter）
  但：ReduceScatter 走 NVLink（18× 更快）→ 实际可忽略

FSEP vs EP 的收益分析（负载不均衡比 r=3）：
  EP 有效计算时间：T_gemm / r（GPU 利用率 1/r）
  FSEP 有效计算时间：T_gemm（GPU 利用率 ~100%）
  FSEP ReduceScatter 额外开销：T_rs ≈ TH / NVLink_BW ≈ 0.02ms（NVLink）
  
  净收益：T_gemm × (1 - 1/r) - T_rs ≈ T_gemm × 0.67 - 0.02ms
  当 T_gemm > 0.03ms 时（几乎总是），FSEP 总是更快！
```

### 3.3 CommTensor 对 FSEP 的支持

```
FSEP 模式下的 CommTensor 变化：

正向 Dispatch（所有 GPU 都需要所有 tokens）：
  传统 FSEP dispatch：broadcast hidden_states 给所有 EP GPU
  
  MoEX CommTensor FSEP：
    ct.data[r, s, :, :] = hidden_states[token]（同样写入 CT）
    但每个 rank r 的 CT 数据相同！（因为 FSEP 需要所有 GPU 看到所有 token）
    
    优化：不重复存储，使用共享指针
    ct.data 实际为 [R, S, T_tiles, H]，但 R 个 rank 的数据相同 → 只存 1 份
    ct.meta.is_fsep_broadcast = True → dispatch 时用 AllGather 而非 AlltoAll

FSEP ReduceScatter（节点内）：
  Expert GEMM 输出：partial_out [B*L, H_shard]（每 GPU 持有 H_shard = H/R）
  ReduceScatter → full_out，每 GPU 持有 B*L/R 个 token 的完整输出 [B*L/R, H]
  
  CommTensor 对应：
    output_ct.data[r, :slot_counts[r], :, :] = 本 GPU 负责的 token 的完整输出
    → combine 时使用 scatter_add（与非 FSEP 相同接口！）

  关键：ReduceScatter 的结果写入 CommTensor 的对应 rank slot，
        接口与传统 EP 的 Combine RDMA 相同 → 代码复用
```

### 3.4 FSEP 实现细节

```python
class FSEPExpertEngine:
    """FSEP 专家计算引擎"""

    def __init__(self, config: MoEXConfig, expert_weights_shard: Tensor):
        """
        expert_weights_shard: [num_experts, H, H_out/R]
            本 GPU 持有所有专家的 H_out 维分片
        """
        self.W_shard = expert_weights_shard  # [num_experts, H, H_out/R]
        self.config = config

    def forward(self, input_ct: CommTensor) -> CommTensor:
        """
        FSEP 正向计算：
          1. 每 GPU 用本地 W_shard 计算所有 tokens 的 partial output
          2. ReduceScatter：汇聚 partial outputs，每 GPU 得到 1/R token 的完整 output
          3. 结果写入 output CommTensor
        """
        B_L = input_ct.meta.slot_counts.sum().item()
        H_out = self.W_shard.shape[-1] * self.config.ep_size  # 完整输出维度

        # Step 1: GroupedGEMM（所有专家合并为一个 kernel）
        # input: [B_L, H]，W: [num_experts, H, H_out/R]
        partial_output = grouped_gemm_fsep(
            input=input_ct.data.view(B_L, -1),  # [B_L, H]
            weights=self.W_shard,
            expert_ids=input_ct.meta.get_expert_ids(),  # [B_L]
        )  # → [B_L, H_out/R]

        # Step 2: ReduceScatter（节点内 NVLink）
        # 每 GPU 持有 partial_output[B_L, H_out/R]
        # 汇聚为 full_output[B_L/R, H_out]（每 GPU 负责 1/R token）
        full_output = dist.reduce_scatter_tensor(
            partial_output,
            group=self.config.process_groups.fsep_group,
            op=dist.ReduceOp.SUM,
        )  # → [B_L/R, H_out]

        # Step 3: 写入 output CommTensor
        output_ct = CommTensor.allocate(self.config)
        # 将 full_output 按 token_indices 填入 output_ct
        # （与非 FSEP 模式的接口相同）
        output_ct.meta = input_ct.meta  # 复用路由元数据
        output_ct.data = full_output.view(...)  # 重塑为 CommTensor 布局

        return output_ct
```

---

## 4. 动态 Expert Re-layout（LAER-MoE 接口）

### 4.1 Re-layout 的必要性

```
训练过程中，由于数据分布变化，expert 负载持续变化：

Step 100:  Expert 0: 2000 tokens, Expert 7: 100 tokens（20:1 不均衡）
Step 200:  Expert 3: 1800 tokens, Expert 1: 150 tokens（12:1 不均衡）

FSEP 能部分缓解（均匀化计算），但仍有通信不均衡问题：
  不均衡的 token 分布 → 不同 rank 接收的 token 数差异大
  → 某些 rank 的 CommTensor slot 大量填充，某些 rank 几乎空置

LAER-MoE 的解法：Load-Adaptive Expert Re-layout
  动态迁移 expert 参数 → 均衡每个 rank 的 token 负载
  MoEX 提供与 LAER-MoE Load Planner 的标准接口
```

### 4.2 CommTensor layout_version 机制

```python
class LayoutManager:
    """
    Expert 动态 Re-layout 管理器
    与 LAER-MoE Load Planner 对接
    """

    def __init__(self, config: MoEXConfig):
        self.current_placement = ExpertPlacement.default(config)
        self.pending_placement = None
        self.layout_version = 0
        self.migration_stream = torch.cuda.Stream()

    def register_load_planner(self, planner: LoadPlanner):
        """注册外部 Load Planner（LAER-MoE 接口）"""
        self.load_planner = planner

    def check_and_schedule_relayout(
        self,
        token_counts: Tensor,  # [num_experts] 当前 step 的每专家 token 数
        step: int,
    ):
        """
        检查是否需要 re-layout，如需则安排异步迁移
        调用时机：backward pass 开始时（有大量通信时间可利用）
        """
        if step % self.config.relayout_interval != 0:
            return

        imbalance = self._compute_imbalance(token_counts)
        if imbalance < self.config.relayout_threshold:
            return

        # 调用 Load Planner 计算新布局
        new_placement = self.load_planner.compute_new_layout(
            current=self.current_placement,
            token_counts=token_counts,
        )

        if new_placement == self.current_placement:
            return

        # 安排异步迁移（在 backward 期间执行）
        self.pending_placement = new_placement
        self._schedule_migration_async(new_placement)

    def _schedule_migration_async(self, new_placement: ExpertPlacement):
        """
        异步专家参数迁移
        使用专用 migration_stream，不阻塞 backward 计算
        """
        with torch.cuda.stream(self.migration_stream):
            for expert_id, (old_rank, new_rank) in new_placement.changes():
                if old_rank == dist.get_rank():
                    # 本 GPU 的专家迁移到其他 GPU
                    rdma_put_async(
                        src=self.expert_params[expert_id],
                        dst_rank=new_rank,
                        dst_offset=expert_id * param_size,
                        stream=self.migration_stream,
                    )
                elif new_rank == dist.get_rank():
                    # 接收其他 GPU 迁移过来的专家
                    rdma_get_async(
                        src_rank=old_rank,
                        src_offset=expert_id * param_size,
                        dst=self.expert_params[expert_id],
                        stream=self.migration_stream,
                    )

    def apply_pending_layout(self, ct: CommTensor):
        """
        在下一个 forward 开始前应用新布局
        如果迁移尚未完成，等待（极短，因为迁移在 backward 期间已进行）
        """
        if self.pending_placement is None:
            return

        # 等待迁移完成（应该已经完成）
        self.migration_stream.synchronize()

        # 更新 CommTensor 的路由元数据（expert_id → rank 映射）
        self.current_placement = self.pending_placement
        self.pending_placement = None
        self.layout_version += 1

        # CommTensor 的 meta 更新
        ct.meta.layout_version = self.layout_version
        ct.meta.expert_placement = self.current_placement
```

### 4.3 Re-layout 的 CommTensor 视角

```
Re-layout 前后，CommTensor 的变化：

Re-layout 前（Expert E5 在 GPU 2，E7 在 GPU 5）：
  CommTensor[r=2, :] = tokens routing to Expert E5
  CommTensor[r=5, :] = tokens routing to Expert E7

Load Planner 决策：将 E5 迁移到 GPU 5（与 E7 共存），E7 的一半分给 GPU 2

Re-layout 后（E5 在 GPU 5，E7[0:1] 在 GPU 2，E7[2:4] 在 GPU 5）：
  CommTensor[r=2, :] = tokens routing to Expert E7[0:1]（更少！）
  CommTensor[r=5, :] = tokens routing to Expert E5 + E7[2:4]（均衡！）

CommTensor 的 from_hidden_states() 使用 expert_placement 计算 rank：
  rank = expert_placement.get_rank(expert_id)
  → re-layout 后，相同的 expert_id 映射到不同的 rank
  → CommTensor 的填充自动适应新 layout（无需修改路由逻辑）
```

---

## 5. CPU+GPU 异构（推理扩展）

### 5.1 KTransformers 启示（CPU offloading）

```
来自 KTransformers 论文：在 671B DeepSeek-V3 上，用消费级硬件（24 GB GPU + 大量 CPU 内存）
  Expert 参数：大量存在 CPU 内存（DDR5，51.2 GB/s）
  活跃 Expert：动态 offload 到 GPU（PCIe 4.0，64 GB/s）

传统实现的问题：
  PCIe 带宽（64 GB/s）>> Expert GEMM 计算带宽需求
  → Expert 访问是瓶颈，需要 CPU 预取

MoEX 的 CPU Offloading 接口（规划中，主要用于推理）：
  CommTensor 的 rank 维度可以映射到 "CPU 节点"：
    rank=0: GPU 0（快速，高优先级 expert）
    rank=1: CPU 0（慢速，低优先级 expert，预取队列）

  预取调度：
    Forward step N：路由决策 → 知道 step N+1 需要哪些 expert
    → 提前发起 CPU→GPU PCIe 传输（异步）
    → step N+1 的 Expert GEMM 时，参数已在 GPU 上
```

### 5.2 CommTensor 的 CPU 扩展

```python
class CPUCommTensor:
    """扩展的 CommTensor，支持 CPU 内存 expert 参数"""

    # CPU 侧：Expert 参数（pinned memory，支持 DMA）
    cpu_expert_params: Dict[int, Tensor]  # {expert_id: W on CPU pinned}

    # GPU 侧：热专家参数（有限 GPU 显存）
    gpu_expert_cache: LRUCache  # LRU cache，容量 = GPU 显存 / expert_param_size

    # 预取队列
    prefetch_queue: asyncio.Queue

    def prefetch_experts(self, next_routing: Tensor):
        """
        根据路由预测，预取下一步需要的 expert 参数
        在当前 step 的 backward 期间执行（异步）
        """
        next_expert_ids = get_top_experts(next_routing)
        for expert_id in next_expert_ids:
            if expert_id not in self.gpu_expert_cache:
                # 异步 CPU→GPU 传输
                self.prefetch_queue.put_nowait(expert_id)

    def prefetch_worker(self):
        """后台线程：执行 CPU→GPU 传输"""
        with torch.cuda.stream(self.prefetch_stream):
            while True:
                expert_id = self.prefetch_queue.get()
                src = self.cpu_expert_params[expert_id]
                dst = self.gpu_expert_cache.allocate(expert_id)
                dst.copy_(src, non_blocking=True)  # 异步 PCIe DMA
```

---

## 6. 网络拓扑感知

### 6.1 拓扑感知对 CommTensor 的影响

```
典型 GPU 集群拓扑：

节点内（8 GPU，NVLink）：
  带宽：H100 NVSwitch 900 GB/s（双向），MI300X XGMI 896 GB/s
  延迟：~1µs

节点间（IB 网络）：
  带宽：IB HDR 200 Gb/s = 25 GB/s per port（通常每节点 8 个 port = 200 GB/s）
  延迟：~5-20µs

拓扑对 CommTensor dispatch 的影响：
  同节点内的 rank 对：走 NVLink → RDMA 延迟 ~1µs，带宽充足
  跨节点的 rank 对：走 IB → RDMA 延迟 ~10µs，带宽有限

CommTensor 拓扑感知优化：
  1. 优先发送节点内数据（NVLink 更快，先完成）
  2. 节点内 dispatch + 节点间 dispatch 并行
  3. FSEP ReduceScatter 仅在节点内（避免占用 IB）
```

### 6.2 拓扑感知调度

```python
class TopologyAwareDispatcher:
    """拓扑感知的 CommTensor Dispatch"""

    def __init__(self, process_groups: MoEXProcessGroups):
        self.intra_node_ranks = process_groups.get_intra_node_ranks()
        self.inter_node_ranks = process_groups.get_inter_node_ranks()

    def dispatch(self, ct: CommTensor):
        """
        拓扑感知 dispatch：
        1. 节点内 (NVLink)：高优先级，先发
        2. 节点间 (IB)：低优先级，后发（但与节点内并行）
        """
        # Step 1: 节点内 dispatch（NVLink，高优先级）
        for r in self.intra_node_ranks:
            if r == get_rank():
                continue
            count = ct.meta.slot_counts[r]
            nvlink_put_async(ct.data[r, :count], dst=r, stream=comm_stream_intra)

        # Step 2: 节点间 dispatch（IB，与节点内并行）
        for r in self.inter_node_ranks:
            count = ct.meta.slot_counts[r]
            ib_rdma_put_async(ct.data[r, :count], dst=r, stream=comm_stream_inter)

        # 等待所有 dispatch 完成（两个 stream 都完成）
        comm_event = torch.cuda.Event()
        comm_stream_intra.record_event(comm_event)
        comm_stream_inter.record_event(comm_event)

        return comm_event  # 供 Expert GEMM 等待
```

### 6.3 节点内 FSEP 路径

```
FSEP ReduceScatter 专用 NVLink 路径：

节点内 8 GPU（NVLink 满连接，900 GB/s）：
  Expert GEMM → partial_output [B_L, H/8]（每 GPU）
  ReduceScatter（8 GPU，NVLink）：
    实现：Ring ReduceScatter（7 步，每步 1/8 数据）
    带宽：900 GB/s → 7 步共 ~7 × H/8 × B_L × 2 bytes
    时间：B_L × H × 2 bytes / 900 GB/s ≈ 0.009ms（B_L=4096, H=4096）

节点间 dispatch（IB）：
  带宽：200 GB/s（8 口 IB）
  时间：B_L × H × 2 bytes / 200 GB/s ≈ 0.04ms

关键：FSEP ReduceScatter 走 NVLink，速度是 IB 的 4.5× ！
→ FSEP 的额外通信几乎可以忽略（完全 overlap 在 IB dispatch 期间）
```

---

## 7. CommTensor 与并行模式的对应关系

### 7.1 各并行模式下 CommTensor 的形态

| 并行模式 | CommTensor 变化 | 备注 |
|---------|----------------|------|
| 纯 EP | CT[R, S, T, H]，R = EP | 基础模式 |
| EP + FSEP | CT[R, S, T, H/R]（可选）或 CT 广播 | ReduceScatter 路径 |
| EP + EDP | 2 个 CT，每个覆盖半数 expert | EDP 副本各自独立 |
| PP + EP | 每 PP stage 独立 CT | PP 界面为 point-to-point |
| TP + EP | CT 构建前进行 TP Fold（见第 2 节）| Gate GEMM 在 TP 分片 |
| CP + EP | CT 的 S 维包含所有 CP chunk | Ring Attention 不影响 EP dispatch |
| 动态 re-layout | CT.meta.layout_version++ | expert_placement 更新 |

### 7.2 并行模式组合示意

```
完整并行配置下的数据流（TP=4, CP=2, EP=16, PP=4）：

PP Stage 2（8 层 Transformer）：

GPU layout in PP Stage 2（64 GPU）：
  TP groups: {0-3}, {4-7}, ..., {60-63}（16 个 TP group，每组 4 GPU）
  CP groups: {0-31} (odd layers), {32-63} (even layers)  (示例)
  EP groups: {0-63}（所有 64 GPU，EP=64，实际 EP=16 subset）

  每个 Transformer Block：
    Attention：TP=4, CP=2 → 8 GPU 协作
    MoE FFN：EP=16 → 64/16 = 4 EDP 副本

  CommTensor[R=16, S=..., T=128, H=4096]：
    R=16 对应 16 个 EP ranks
    每个 EDP 副本（4 个 CommTensor）处理不同的 micro-batch
```

---

## 8. 并行配置指南

### 8.1 选择 EP 度的原则

```
EP 度选择指南：

核心公式：
  EP_optimal = argmin_{ep} (T_dispatch(ep) + T_expert_imbalance(ep))
  T_dispatch(ep) = tokens * H * 2 / ep / IB_BW
  T_expert_imbalance(ep) ≈ T_gemm / ep × (1 - 1/imbalance_ratio)

经验规则：
  小模型（< 10B）：EP = 8（节点内，NVLink）
  中等模型（10-100B）：EP = 32-64（跨 4-8 节点，IB）
  大模型（> 100B，如 DeepSeek-V3）：EP = 128-256 + FSEP（节点内 ReduceScatter）

FSEP 启用条件：
  imbalance_ratio > 2（任意专家的 token 数是平均值的 2 倍以上）
  节点内 GPU 数 ≥ 4（FSEP ReduceScatter 至少需要 4 GPU 才有效）

动态 re-layout 启用条件：
  imbalance_ratio > 1.5（FSEP 无法完全消除的不均衡）
  训练 step > 1000（需要足够的统计数据）
```

### 8.2 推荐配置（不同规模）

```
场景 1：7B MoE 模型（8 experts，2 active）
  Hardware: 8 × H100（1 节点）
  推荐：EP=8, FSEP=False（节点内，NVLink 带宽充足）
  CommTensor: R=8, S=256, T=128, H=4096

场景 2：70B MoE 模型（64 experts，2 active）
  Hardware: 64 × H100（8 节点）
  推荐：EP=32, EDP=2, FSEP=True, dynamic_relayout=True
  CommTensor: R=32, S=512, T=128, H=8192

场景 3：671B MoE 模型（256 experts，8 active，DeepSeek-V3 规模）
  Hardware: 2048 × H100（256 节点）
  推荐：
    PP=16, TP=8, CP=4, DP=4（Attention）
    EP=128, EDP=4, FSEP=True（MoE，节点内 8 GPU 一组 FSEP）
  CommTensor: R=128, S=1024, T=128, H=7168

场景 4：推理（1 GPU + CPU offloading，消费级）
  Hardware: 1 × RTX 4090（24 GB）+ 512 GB DDR5 CPU
  推荐：EP=1（单 GPU），CPU_offload=True
  CommTensor: R=1, S=1024, T=64, H=7168（降低 tile 大小减少显存）
```
