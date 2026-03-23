# SimpleMoE：基于 veScale-FSDP 思想 + Megatron 实现逻辑的 MoE 训练框架

> **设计思路：** veScale-FSDP 提供核心分片与通信思想，Megatron-Core MoE 提供 MoE 层实现逻辑。  
> **并行策略：** 仅 FSDP + EP + PP（不使用 TP / CP）。  
> **来源标注：** 每个设计决策标注 `[veScale]` 或 `[Megatron]`，表明思想来源。

---

## 第一部分：veScale-FSDP 论文的深度拆解

### 1. 论文的核心矛盾

veScale-FSDP 抓住了一个正在发生的范式转换：

```
旧范式（2020-2024）：
  优化器 = Adam（每参数独立更新）
  量化 = per-tensor 或不量化
  FSDP = 按元素/按行均匀切分
  三者兼容，一切正常

新范式（2025-）：
  优化器 = Shampoo/Muon（需要完整子矩阵做矩阵运算）
  量化 = block-wise（128×128 block 共享一个 scale）
  FSDP = ？？？
  
  传统 FSDP 在新范式下「语义错误」：
    W 的 block 被行切分切碎 → 量化 scale 失效
    W 的 Kronecker 因子被分到多个 GPU → 矩阵幂运算需要额外通信
    
  这不是「性能差」，而是「功能上不可用」
```

论文的核心贡献是提出了一个新的分片原语 RaggedShard，使得 FSDP 的分片单元从「元素/行」升级为「语义 block」，从而在新范式下恢复正确性的同时还提升了性能。

### 2. RaggedShard 的三层理解

**第一层：数据结构层面**

```
RaggedShard 的物理表示：
  不再是一个连续的 1D buffer（传统 flat shard），
  而是一组「block descriptor + data」的列表

  传统 shard:
    GPU 0: [w_0, w_1, ..., w_{N/P}]   ← 连续 1D，N/P 个元素

  RaggedShard:
    GPU 0: [Block(offset=(0,0), shape=(128,128), data=...),
            Block(offset=(0,128), shape=(128,128), data=...),
            Block(offset=(128,0), shape=(128,128), data=...)]
    GPU 1: [Block(offset=(128,128), shape=(128,128), data=...),
            Block(offset=(0,256), shape=(128,128), data=...)]
    
  每个 GPU 持有的 block 数量可以不同 → "ragged"（参差不齐）
```

**第二层：语义层面**

```
RaggedShard 的本质是一种「co-location contract」（共置契约）：

  "这些参数元素必须在同一个 GPU 上"

  这个契约的来源有三种：
  
  (a) 量化约束：
      block (i,j) 的所有参数共享一个 scale factor
      → 这些参数必须在一起才能正确量化/反量化
  
  (b) 优化器约束：
      Shampoo 需要 W_block 来计算 L_block = W_block · W_block^T
      → W_block 的所有元素必须在同一 GPU
      → L_block 和 R_block 也必须跟 W_block 在一起

  (c) 通信约束：
      All-Gather 时，每个 GPU 贡献的内容可以不等长
      但需要预先知道每个 GPU 的贡献大小（即 shard_info）
      → RaggedShard 的 metadata 就是这个信息
```

**第三层：系统设计层面**

```
RaggedShard 改变了 FSDP 的整个数据通路：

  传统 FSDP:
    shard = param[rank * chunk : (rank+1) * chunk]    ← 固定偏移
    full = all_gather(shard)                           ← 等大 gather
    output = compute(full)
    grad_shard = reduce_scatter(grad)                  ← 等大 scatter
  
  veScale-FSDP:
    shard = extract_blocks(param, plan[rank])          ← 按 block 提取
    full = ragged_all_gather(shard, shard_info)        ← 不等大 gather
    output = compute(full)
    grad_shard = ragged_reduce_scatter(grad, shard_info) ← 不等大 scatter

  "ragged" 版本的集合操作需要额外的 metadata（shard_info），
  但换来了分片的灵活性。在 NCCL 层面，这通过 ncclSend/ncclRecv
  的组合实现，而非直接调用 ncclAllGather。
```

### 3. Structure-Aware Planning 的算法本质

```
这个规划算法解决的问题可以形式化为：

  输入：
    - 参数集合 P = {p_1, ..., p_n}
    - 每个参数可以划分为 blocks: p_i = {b_{i,1}, ..., b_{i,k_i}}
    - co-location 约束集合 C = {(b_α, b_β) | b_α 和 b_β 必须在同一 GPU}
    - GPU 数量 G，每个 GPU 内存上限 M

  输出：
    - 分配 f: block → GPU，使得：
      ∀(b_α, b_β) ∈ C: f(b_α) = f(b_β)        ← 满足约束
      max_g Σ_{f(b)=g} size(b) 最小化            ← 内存均衡

  算法：
    Step 1: 约束合并 —— 将 co-location 约束用 Union-Find 合并
            得到 super-blocks S = {s_1, ..., s_m}，每个 s_j 是不可分的
    Step 2: 贪心分配 —— 将 super-blocks 按大小降序排列
            依次分配到当前内存最小的 GPU
    Step 3: 局部优化 —— 如果某个 GPU 超过 M，
            尝试将其最小的 super-block 移到内存最小的 GPU
            
  时间复杂度: O(n·α(n)) for Union-Find + O(m·log(m)) for sorting
  实际中 m << n（大多数 block 被合并），所以很快
```

### 4. 通信优化：为什么对 MoE 有价值

veScale-FSDP 的三个通信优化，在 MoE 场景下有不同的适用性：

```
(a) Lazy All-Gather
    原理：不在前一层结束时立即 AG 下一层，而是延迟到真正需要时
    
    对 MoE 的价值：
      MoE Transformer 中，Attention → MoE 层交替
      Attention 结束后，MoE 层需要的不是 AG（Expert 已在本地）
      而是 All-to-All dispatch
      
      但下一个 Attention 层的参数 AG 可以提前开始：
        MoE Expert 计算 ← overlap → 下一层 Attention 参数 AG
      
      这个 overlap 窗口在传统 FSDP 中不存在（因为 AG 在前一层结束就做了）
      veScale-FSDP 的 Lazy AG 正好打开了这个窗口

(b) Hierarchical Reduce-Scatter
    原理：节点内 NVLink RS + 节点间 IB RS，两级分层
    
    对 MoE 的价值：
      Dense 参数（Attention）的梯度 RS 可以走分层路径
      Expert 参数的梯度只需在 Expert Data Parallel (EDP) 组内 RS
      两者通信组不同，天然不冲突

(c) Block-level Overlap
    原理：Block A 的 RS 与 Block B 的反向计算重叠
    
    对 MoE 的价值：
      Attention 反向的 RS 与 MoE 反向的 Expert 计算重叠
      MoE Expert 计算是计算密集型 → 是最好的 overlap 窗口
```

### 5. 论文没覆盖的部分（需要 Megatron 填补）

```
veScale-FSDP 是一个「FSDP 层的系统」，不直接涉及：

  ❌ MoE 路由（Gate Network + Top-K 选择）
  ❌ Token Dispatch / Combine（All-to-All 通信）
  ❌ Expert 计算（Grouped GEMM）
  ❌ Expert Parallel 进程组管理
  ❌ Pipeline Parallel 调度
  ❌ MoE 特有的内存优化（Memory-Efficient Permutation）
  ❌ MoE 特有的激活重算策略
  ❌ 负载均衡（Aux Loss / Expert Choice）
  ❌ FP8 低精度训练
  ❌ Checkpoint 系统

  → 这些全部需要从 Megatron-Core 获取实现逻辑
```

---

## 第二部分：SimpleMoE 框架详细设计

### 1. 总体架构

```
SimpleMoE = veScale-FSDP 的分片思想 + Megatron-Core 的 MoE 实现逻辑

┌──────────────────────────────────────────────────────────────────────────┐
│                          SimpleMoE Framework                             │
│                                                                          │
│  ┌─ User API ──────────────────────────────────────────────────────────┐ │
│  │  model = SimpleMoE(model, parallel_config, optimizer_config)        │ │
│  │  trainer.train_step(batch) → loss                                   │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌─ 并行策略层 ─────────────────────────────────────────────────────────┐ │
│  │                                                                      │ │
│  │  Dense 层 (Attention / Norm / Embedding):                            │ │
│  │    RaggedShard FSDP [veScale]                                        │ │
│  │    ├─ block-wise 分片，对齐量化 block / 优化器 block                 │ │
│  │    ├─ Lazy All-Gather + Hierarchical RS                              │ │
│  │    └─ Buffer Pool 复用                                               │ │
│  │                                                                      │ │
│  │  MoE 层 (Experts + Router + Dispatcher):                             │ │
│  │    Expert Parallel [Megatron]                                        │ │
│  │    ├─ 四阶段通路：Route → Dispatch → Compute → Combine              │ │
│  │    ├─ Grouped GEMM expert 计算                                       │ │
│  │    ├─ Memory-Efficient Permutation                                   │ │
│  │    └─ EP 通信与计算 overlap（双 CUDA Stream）                        │ │
│  │                                                                      │ │
│  │  全局:                                                               │ │
│  │    Pipeline Parallel (1F1B / VPP) [Megatron]                         │ │
│  │    Dual DeviceMesh (Dense mesh + Expert mesh) [Megatron]             │ │
│  │                                                                      │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌─ 统一规划器 ─────────────────────────────────────────────────────────┐ │
│  │  Structure-Aware Planner [veScale]                                   │ │
│  │  ├─ 分析模型结构，识别 block 约束（量化 + 优化器）                   │ │
│  │  ├─ RaggedShard 分配（Dense 参数）                                   │ │
│  │  ├─ Expert 放置（EP 参数）                                           │ │
│  │  └─ PP stage 划分（memory-balanced，考虑 MoE 层更大）[Megatron]      │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌─ 内存优化 ──────────────────────────────────────────────────────────┐ │
│  │  Buffer Pool (AG + A2A buffers) [veScale]                            │ │
│  │  Memory-Efficient Permutation (路由权重前置) [Megatron]              │ │
│  │  Fine-grained Recomputation (选择性重算) [Megatron]                  │ │
│  │  Fine-grained Activation Offloading [Megatron]                       │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌─ 计算优化 ──────────────────────────────────────────────────────────┐ │
│  │  Grouped GEMM (所有 local Expert 单次调用) [Megatron]                │ │
│  │  Router + Permutation 内核融合 [Megatron]                            │ │
│  │  FP8 Block-wise 量化训练 [Megatron] + RaggedShard 对齐 [veScale]    │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌─ 生产特性 ──────────────────────────────────────────────────────────┐ │
│  │  负载均衡 (Aux Loss / Aux-Loss-Free) [Megatron]                      │ │
│  │  Shared Expert [Megatron]                                            │ │
│  │  分布式 Checkpoint (并行无关保存) [Megatron]                          │ │
│  │  Muon 优化器 [Megatron] + RaggedShard 兼容 [veScale]                │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────┘
```

### 2. 进程组设计 [Megatron 逻辑，去掉 TP/CP]

Megatron-Core 的进程组管理是 `ProcessGroupCollection`，支持 TP、CP、DP、EP、PP 五个维度。SimpleMoE 去掉 TP 和 CP 后，精简为三个维度：

```
Megatron 原始进程组（五维）：
  World = TP × CP × PP × DP，其中 EP ⊆ DP
  
  Attention: tp, cp, dp, pp
  Expert:    ep, expt_tp, expt_dp, pp

SimpleMoE 精简进程组（三维）：
  World = PP × EP × FSDP
  
  Attention / Dense:  fsdp_group, pp_group
  Expert / MoE:       ep_group, edp_group, pp_group

  ← 借鉴 Megatron 的 Parallel Folding 思想（Dense 和 MoE 用不同并行配置）
  ← 但大幅简化：不需要 TP/CP，Attention 用 FSDP 替代 TP 的参数分片功能
```

具体 GPU 拓扑映射：

```python
# [Megatron 逻辑] Process Group 创建
# 假设 128 GPUs = 16 nodes × 8 GPUs/node

class SimpleMoEProcessGroups:
    """
    三维并行的进程组管理。
    
    来源 [Megatron]：Parallel Folding 的 Dual DeviceMesh 思想
    简化：去掉 TP/CP 维度，Dense 层用 FSDP 替代 TP
    """
    
    def __init__(self, pp_size, ep_size, fsdp_size):
        # 约束：pp_size × ep_size × fsdp_size = world_size
        # 例：4 × 8 × 4 = 128
        self.pp_size = pp_size
        self.ep_size = ep_size
        self.fsdp_size = fsdp_size
        
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        assert pp_size * ep_size * fsdp_size == world_size
        
        # GPU 编号布局：[PP_stage][EP_rank][FSDP_rank]
        # stage 0: GPU 0-31,  stage 1: GPU 32-63, ...
        # 每个 stage 内: ep_rank 0-7 在同节点(NVLink)
        #                fsdp_rank 0-3 跨节点(IB)
        
        stage_size = ep_size * fsdp_size  # 32
        self.pp_rank = rank // stage_size
        local_rank = rank % stage_size
        self.ep_rank = local_rank // fsdp_size
        self.fsdp_rank = local_rank % fsdp_size
        
        # --- Dense 层的 FSDP 组 ---
        # 同一 PP stage、同一 EP rank 的 GPU 组成 FSDP 组
        # 这些 GPU 在不同节点上，走 IB
        self.fsdp_group = self._create_fsdp_groups()
        
        # --- MoE 层的 EP 组 ---
        # 同一 PP stage、同一 FSDP rank 的 GPU 组成 EP 组
        # 设计目标：EP 组在同节点内(NVLink)
        self.ep_group = self._create_ep_groups()
        
        # --- Expert Data Parallel (EDP) 组 ---
        # [Megatron 逻辑] Expert 参数在 EDP 组内做梯度 Reduce
        # EDP = FSDP（Expert 不走 RaggedShard，走标准 AllReduce）
        self.edp_group = self.fsdp_group
        
        # --- PP 组 ---
        # 同一 EP rank、同一 FSDP rank 但不同 PP stage 的 GPU
        self.pp_group = self._create_pp_groups()
    
    def _create_ep_groups(self):
        """EP 组 = 节点内 8 GPUs，走 NVLink"""
        groups = []
        for pp in range(self.pp_size):
            for f in range(self.fsdp_size):
                ranks = [
                    pp * self.ep_size * self.fsdp_size + e * self.fsdp_size + f
                    for e in range(self.ep_size)
                ]
                groups.append(dist.new_group(ranks))
        return groups[self._get_group_index('ep')]
    
    def _create_fsdp_groups(self):
        """FSDP 组 = 跨节点 4 GPUs，走 IB"""
        groups = []
        for pp in range(self.pp_size):
            for e in range(self.ep_size):
                ranks = [
                    pp * self.ep_size * self.fsdp_size + e * self.fsdp_size + f
                    for f in range(self.fsdp_size)
                ]
                groups.append(dist.new_group(ranks))
        return groups[self._get_group_index('fsdp')]

#
# GPU 映射示意（128 GPUs, PP=4, EP=8, FSDP=4）：
#
# Stage 0 (32 GPUs):
#   Node 0 (8 GPUs): [ep0,f0] [ep0,f1] [ep0,f2] [ep0,f3]
#                     [ep1,f0] [ep1,f1] [ep1,f2] [ep1,f3]
#   ... (实际 EP=8 需跨到 Node 1)
#
# 更合理的映射：EP=8 放在同节点，FSDP=4 跨节点
#   Node 0 (8 GPUs, all in Stage 0):
#     GPU 0-7 = ep_rank 0-7, fsdp_rank=0
#   Node 1 (8 GPUs, all in Stage 0):
#     GPU 8-15 = ep_rank 0-7, fsdp_rank=1
#   Node 2, 3: 同理，fsdp_rank=2,3
#   → EP All-to-All 全在 NVLink 域内 ✅
#   → FSDP All-Gather/RS 跨节点(IB) ✅
```

### 3. Dense 层：RaggedShard FSDP [veScale]

Dense 层（Attention, LayerNorm, Embedding）的分布式处理完全继承 veScale-FSDP 的设计：

```python
class RaggedFSDP:
    """
    [veScale] Dense 参数的灵活分片。
    
    核心区别于传统 FSDP：
      1. 分片对齐优化器/量化 block 边界
      2. 不等大 shard（ragged）
      3. All-Gather/Reduce-Scatter 使用 ragged 版本
    """
    
    def __init__(self, module, fsdp_group, shard_plan, optimizer_type='adam'):
        self.module = module
        self.group = fsdp_group
        self.rank = dist.get_rank(fsdp_group)
        self.world_size = dist.get_world_size(fsdp_group)
        self.shard_plan = shard_plan
        
        # [veScale] 按 RaggedShard 计划切分参数
        self.sharded_params = {}
        for name, param in module.named_parameters():
            spec = shard_plan.get(name)
            if spec is None:
                continue
            
            if spec.shard_type == 'block_wise':
                my_blocks = spec.block_assignments[self.rank]
                shard_data = self._extract_blocks(param.data, my_blocks)
                shard_meta = RaggedShardMeta(
                    block_shapes=[b.shape for b in my_blocks],
                    block_offsets=[b.offset for b in my_blocks],
                    full_shape=param.shape,
                    total_numel=param.numel(),
                )
            else:
                # 标准行切分（小参数的 fallback）
                chunk_size = (param.numel() + self.world_size - 1) // self.world_size
                start = self.rank * chunk_size
                end = min(start + chunk_size, param.numel())
                shard_data = param.data.flatten()[start:end].clone()
                shard_meta = FlatShardMeta(
                    offset=start, length=end - start, total_numel=param.numel()
                )
            
            param.data = shard_data
            param._shard_meta = shard_meta
            self.sharded_params[name] = param
        
        # [veScale] Buffer Pool: 预分配 AG buffer, LRU 复用
        max_full_size = max(p._shard_meta.total_numel for p in self.sharded_params.values())
        self.ag_buffer_pool = BufferPool(
            max_elem_count=max_full_size,
            num_buffers=2,  # double buffering for overlap
            dtype=next(module.parameters()).dtype,
        )
    
    def all_gather_params(self, param_name):
        """
        [veScale] Lazy All-Gather: 只在真正需要时触发
        """
        param = self.sharded_params[param_name]
        meta = param._shard_meta
        buf = self.ag_buffer_pool.acquire(meta.total_numel)
        
        if isinstance(meta, RaggedShardMeta):
            # [veScale] Ragged All-Gather: 
            # 每个 rank 贡献不等大的数据，按 block offset 放回原始位置
            # 底层用 ncclGroupStart + ncclSend/ncclRecv 实现
            ragged_all_gather(
                output=buf.view(meta.full_shape),
                local_shard=param.data,
                shard_meta_all_ranks=self._all_ranks_meta(meta),
                group=self.group,
            )
        else:
            # 标准 All-Gather
            dist.all_gather_into_tensor(buf, param.data, group=self.group)
        
        return buf.view(meta.full_shape)
    
    def reduce_scatter_grad(self, param_name, full_grad):
        """
        [veScale] Hierarchical Reduce-Scatter:
        如果 FSDP 组跨节点，先做节点内 partial reduce，再做节点间 scatter
        """
        param = self.sharded_params[param_name]
        meta = param._shard_meta
        
        if isinstance(meta, RaggedShardMeta):
            grad_shard = ragged_reduce_scatter(
                input=full_grad,
                shard_meta=meta,
                group=self.group,
                hierarchical=True,  # [veScale] 分层 RS
            )
        else:
            grad_shard = torch.empty_like(param.data)
            dist.reduce_scatter_tensor(grad_shard, full_grad.flatten(), group=self.group)
        
        return grad_shard
    
    def release_full_params(self, buf):
        """[veScale] 提前释放 AG buffer，不等到反向传播结束"""
        self.ag_buffer_pool.release(buf)
```

### 4. MoE 层：四阶段通路 [Megatron 逻辑]

MoE 层的核心实现完全采用 Megatron-Core 的四阶段设计：Route → Dispatch → Compute → Combine。

```python
class MoELayer(nn.Module):
    """
    [Megatron] MoE 层的四阶段前向通路。
    
    直接复用 Megatron-Core 的设计：
      Stage 1: TopKRouter
      Stage 2: Token Dispatcher (All-to-All)
      Stage 3: Expert Computation (Grouped GEMM)
      Stage 4: Combine (All-to-All reverse + weighted merge)
    """
    
    def __init__(self, config, ep_group, edp_group):
        super().__init__()
        self.config = config
        self.ep_group = ep_group
        self.edp_group = edp_group
        self.ep_size = dist.get_world_size(ep_group)
        self.ep_rank = dist.get_rank(ep_group)
        
        # [Megatron] Stage 1: Router
        self.router = TopKRouter(
            hidden_dim=config.hidden_dim,
            num_experts=config.num_experts,
            top_k=config.top_k,
            score_func=config.score_func,  # 'softmax' or 'sigmoid'
        )
        
        # [Megatron] Stage 2 & 4: Token Dispatcher
        self.dispatcher = AllToAllDispatcher(
            ep_group=ep_group,
            num_local_experts=config.num_experts // self.ep_size,
        )
        
        # [Megatron] Stage 3: Local Experts
        num_local_experts = config.num_experts // self.ep_size
        self.experts = GroupedMLP(
            num_experts=num_local_experts,
            hidden_dim=config.hidden_dim,
            ffn_dim=config.expert_ffn_dim,
            activation='swiglu',
        )
        
        # [Megatron] Shared Expert (可选, DeepSeek-V3 style)
        if config.num_shared_experts > 0:
            self.shared_expert = MLP(
                hidden_dim=config.hidden_dim,
                ffn_dim=config.shared_expert_ffn_dim,
                activation='swiglu',
            )
        else:
            self.shared_expert = None
        
        # [Megatron] 负载均衡
        self.load_balancer = self._build_load_balancer(config)
    
    def forward(self, hidden_states):
        """
        四阶段前向 [Megatron]:
        
        Input: [B*S, H]
          │
          ├── Stage 1: Route ──────────────────────────────────────┐
          │   router(hidden_states) → probs, routing_map           │
          │   routing_map: [T, E] 布尔掩码 (哪些 token 去哪些 expert)│
          │                                                         │
          ├── Stage 2: Dispatch ───────────────────────────────────┐│
          │   permute(tokens, routing_map) → 按 expert 排序        ││
          │   All-to-All → 发送到目标 GPU                          ││
          │                                                         │
          ├── Stage 3: Expert Compute ─────────────────────────────┐│
          │   GroupedGEMM(received_tokens) → expert_output          │
          │   [同时] shared_expert(hidden_states) → shared_output   │
          │                                                         │
          ├── Stage 4: Combine ────────────────────────────────────┐│
          │   All-to-All reverse → 返回原始 GPU                     │
          │   unpermute → 恢复原始顺序                               │
          │   加权合并 + shared_expert 输出                          │
          │                                                         │
          └─→ Output: [B*S, H]
        """
        T, H = hidden_states.shape
        
        # ---- Stage 1: Route [Megatron] ----
        probs, routing_map = self.router(hidden_states)
        # probs: [T, K] 路由权重
        # routing_map: [T, E] 布尔掩码
        
        # [Megatron] 负载均衡 loss
        aux_loss = self.load_balancer.compute_loss(probs, routing_map)
        
        # ---- Stage 2: Dispatch [Megatron] ----
        # [Megatron] Memory-Efficient Permutation:
        # 路由权重在 Expert 内部应用（W2 之前），而非 Combine 时应用
        # → 反向传播不需要保存 expert 输出，节省大量激活内存
        dispatched_input, dispatch_meta = self.dispatcher.dispatch(
            hidden_states, probs, routing_map
        )
        
        # ---- Stage 3: Expert Compute [Megatron] ----
        # Grouped GEMM: 所有 local Expert 单次内核调用
        expert_output = self.experts(dispatched_input, dispatch_meta.tokens_per_expert)
        
        # Shared Expert 并行计算（如果有）
        if self.shared_expert is not None:
            shared_output = self.shared_expert(hidden_states)
        
        # ---- Stage 4: Combine [Megatron] ----
        combined_output = self.dispatcher.combine(expert_output, dispatch_meta)
        
        if self.shared_expert is not None:
            combined_output = combined_output + shared_output
        
        return combined_output, aux_loss


class TopKRouter(nn.Module):
    """
    [Megatron] Top-K 路由器
    
    支持三种负载均衡策略：
      1. Auxiliary Loss: 可微惩罚项
      2. Expert Choice: 最优传输
      3. Aux-Loss-Free: 可学习偏置项 (DeepSeek-V3 style)
    """
    
    def __init__(self, hidden_dim, num_experts, top_k, score_func='softmax'):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.top_k = top_k
        self.score_func = score_func
        # [Megatron] Aux-Loss-Free: 可学习 expert bias
        self.expert_bias = nn.Parameter(torch.zeros(num_experts))
    
    def forward(self, hidden_states):
        # [T, H] → [T, E]
        logits = self.gate(hidden_states)
        
        if self.score_func == 'softmax':
            scores = F.softmax(logits, dim=-1)
        elif self.score_func == 'sigmoid':
            scores = F.sigmoid(logits)
        
        # Top-K selection (加上 bias 做选择，但 bias 不参与权重计算)
        biased_scores = scores + self.expert_bias.unsqueeze(0)
        topk_values, topk_indices = torch.topk(biased_scores, self.top_k, dim=-1)
        
        # 实际路由权重用原始 scores（不含 bias）
        probs = torch.gather(scores, dim=-1, index=topk_indices)
        if self.score_func == 'softmax':
            probs = probs / probs.sum(dim=-1, keepdim=True)
        
        # 构建 routing_map: [T, E] 布尔掩码
        routing_map = torch.zeros_like(scores, dtype=torch.bool)
        routing_map.scatter_(1, topk_indices, True)
        
        return probs, routing_map


class AllToAllDispatcher:
    """
    [Megatron] Token Dispatcher
    
    实现 Token Permute → All-to-All → Expert 计算 → All-to-All → Unpermute
    
    关键优化 [Megatron]:
      1. Memory-Efficient Permutation: 路由权重在 W2 前应用而非 Combine 时
         → 反向时不需保存 expert output → 节省 ~26 GB (DSv3 规模)
      2. EP 通信重叠: dispatch 与 shared expert 计算 overlap
    """
    
    def __init__(self, ep_group, num_local_experts):
        self.ep_group = ep_group
        self.ep_size = dist.get_world_size(ep_group)
        self.num_local_experts = num_local_experts
        
        # [veScale] Buffer Pool: All-to-All buffer 复用
        self.dispatch_buffer_pool = BufferPool(
            max_elem_count=0,  # 动态调整
            num_buffers=2,
            dtype=torch.bfloat16,
        )
    
    def dispatch(self, hidden_states, probs, routing_map):
        """
        [Megatron] Dispatch 流程：
          1. Permute: 按 expert index 对 token 重排序
          2. All-to-All: 发送到目标 GPU
        
        [Megatron] Memory-Efficient Permutation:
          标准公式:  y = Σ p_i · Expert_i(x)
          优化公式:  y = Σ Expert_i(p_i · x)    ← 错误! 这改变了语义
          
          正确做法（Megatron 论文 §4.1）:
            y = Σ W2_i · (p_i · φ(W1_i · x))
            即路由权重 p_i 乘在 SwiGLU 激活之后、W2 线性层之前
            这在无 bias 时数学等价，但反向时只需 φ(z_i)（已保存）而非整个 Expert_i(x)
        """
        T, H = hidden_states.shape
        E = routing_map.shape[1]
        
        # Permute: 将 token 按目标 expert 排列
        # token_i 被路由到 expert_j → 放到 expert_j 对应的位置
        permuted_tokens, permute_indices, tokens_per_expert = permute_tokens(
            hidden_states, routing_map
        )
        
        # [Megatron] Memory-Efficient: 在 permute 后就应用路由权重
        # 这样 expert output 不需要再乘权重 → 反向不需保存 expert output
        expanded_probs = gather_probs_for_permuted(probs, routing_map, permute_indices)
        permuted_tokens = permuted_tokens * expanded_probs.unsqueeze(-1)
        
        # All-to-All: 按 EP rank 分配
        # 每个 rank 发送去往该 rank 所持 expert 的 token
        send_counts = compute_send_counts(tokens_per_expert, self.ep_size)
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=self.ep_group)
        
        recv_buf = torch.empty(recv_counts.sum(), H, dtype=hidden_states.dtype, 
                               device=hidden_states.device)
        dist.all_to_all(
            list(recv_buf.split(recv_counts.tolist())),
            list(permuted_tokens.split(send_counts.tolist())),
            group=self.ep_group,
        )
        
        meta = DispatchMeta(
            permute_indices=permute_indices,
            send_counts=send_counts,
            recv_counts=recv_counts,
            tokens_per_expert=tokens_per_expert,
            original_shape=(T, H),
        )
        
        return recv_buf, meta
    
    def combine(self, expert_output, meta):
        """
        [Megatron] Combine 流程：
          1. All-to-All reverse: 返回原始 GPU
          2. Unpermute: 恢复原始 token 顺序
          3. (权重已在 dispatch 时应用，此处直接加和)
        """
        # All-to-All reverse
        send_buf = torch.empty(
            meta.send_counts.sum(), expert_output.shape[-1],
            dtype=expert_output.dtype, device=expert_output.device
        )
        dist.all_to_all(
            list(send_buf.split(meta.send_counts.tolist())),
            list(expert_output.split(meta.recv_counts.tolist())),
            group=self.ep_group,
        )
        
        # Unpermute + 加和（同一 token 的多个 expert 输出相加）
        output = unpermute_and_reduce(
            send_buf, meta.permute_indices, meta.original_shape
        )
        
        return output


class GroupedMLP(nn.Module):
    """
    [Megatron] Grouped GEMM Expert 计算
    
    不逐 expert 循环，而是将所有 local expert 的 token 打包为一次 Grouped GEMM。
    
    SwiGLU: out = W_down · (SiLU(W_gate · x) * (W_up · x))
    """
    
    def __init__(self, num_experts, hidden_dim, ffn_dim, activation='swiglu'):
        super().__init__()
        self.num_experts = num_experts
        
        # 每个 expert 的参数：W_gate, W_up ∈ R^{ffn_dim × hidden_dim}, W_down ∈ R^{hidden_dim × ffn_dim}
        self.w_gate = nn.Parameter(torch.randn(num_experts, ffn_dim, hidden_dim))
        self.w_up = nn.Parameter(torch.randn(num_experts, ffn_dim, hidden_dim))
        self.w_down = nn.Parameter(torch.randn(num_experts, hidden_dim, ffn_dim))
    
    def forward(self, x, tokens_per_expert):
        """
        x: [total_recv_tokens, H] — 已按 expert 排序
        tokens_per_expert: [num_local_experts] — 每个 expert 收到的 token 数
        
        [Megatron] 使用 Grouped GEMM 而非循环:
          所有 expert 的 GEMM 打包为一个内核调用
          → 最大化 GPU 利用率，减少内核启动开销
        """
        # 分组信息
        expert_offsets = torch.cumsum(tokens_per_expert, dim=0)
        
        # Grouped GEMM: gate
        gate_out = grouped_gemm(x, self.w_gate, tokens_per_expert)  # [total, ffn_dim]
        
        # Grouped GEMM: up
        up_out = grouped_gemm(x, self.w_up, tokens_per_expert)  # [total, ffn_dim]
        
        # SwiGLU activation
        hidden = F.silu(gate_out) * up_out
        
        # Grouped GEMM: down
        output = grouped_gemm(hidden, self.w_down, tokens_per_expert)  # [total, H]
        
        return output
```

### 5. 内存优化 [Megatron 逻辑]

```python
class SimpleMoEMemoryOptimizer:
    """
    内存优化策略：
    
    [veScale] Buffer Pool — AG 和 A2A buffer 跨层复用
    [Megatron] Memory-Efficient Permutation — 不保存 expert output (已内嵌在 Dispatcher 中)
    [Megatron] Fine-grained Recomputation — 选择性重算
    [Megatron] Fine-grained Offloading — 激活 offload 到 CPU
    """
    
    @staticmethod
    def configure_recomputation(model, strategy='selective'):
        """
        [Megatron] 细粒度重算策略
        
        不做全层重算（那样 MoE 层会重新触发 All-to-All，代价极大），
        而是选择性重算「内存大 / 计算小」的模块：
        
        推荐重算列表（Megatron 论文 §4.2）：
          ┌──────────────────────┬────────────────┬──────────────┐
          │ 模块                  │ 节省内存/GPU    │ 计算开销     │
          ├──────────────────────┼────────────────┼──────────────┤
          │ Attention Up-Proj    │ ~30 GB         │ < 5%         │
          │ SwiGLU Activation    │ ~4 GB          │ < 5%         │
          │ LayerNorm            │ ~8 GB          │ < 5%         │
          ├──────────────────────┼────────────────┼──────────────┤
          │ 总计                  │ ~42 GB         │ < 5%         │
          └──────────────────────┴────────────────┴──────────────┘
        
        绝不重算的：
          ❌ Expert GEMM — 会重新触发 All-to-All 通信
          ❌ Attention SDPA — 计算量太大(O(s²))
        """
        for layer in model.layers:
            # Attention 部分
            if hasattr(layer, 'attn'):
                layer.attn = RecomputeWrapper(
                    layer.attn,
                    save=['input'],
                    recompute=['up_proj', 'layernorm', 'activation'],
                    # 不重算: qkv_proj(已保存), sdpa(太贵)
                )
            
            # MoE 部分: 只重算 SwiGLU 激活（Expert 内部的 activation）
            if hasattr(layer, 'moe'):
                layer.moe.experts = RecomputeWrapper(
                    layer.moe.experts,
                    save=['input', 'gate_output_pre_act'],
                    recompute=['swiglu_activation'],
                    # 绝不重算: expert GEMM 本身, All-to-All dispatch
                )
    
    @staticmethod
    def configure_offloading(model, offload_modules=None):
        """
        [Megatron] 细粒度激活 Offloading
        
        原理: GPU Copy Engine 和 Compute Engine 独立运行
          → D2H copy 与后续计算并行 → 接近零成本
        
        前向: 模块计算后立即 offload 输入激活到 CPU (专用 D2H stream)
        反向: Layer-Staggered Reload — 当前层反向时 reload 下一层激活
        
        适合 offload 的模块:
          ✅ Attention (输入激活大，但下一步是 MoE 计算可以 overlap)
          ✅ Expert FFN (输入激活也大)
        不适合 offload 的:
          ❌ LayerNorm (太小，offload 开销 > 收益 → 直接重算)
        """
        if offload_modules is None:
            offload_modules = ['attention', 'expert_ffn']
        
        d2h_stream = torch.cuda.Stream()
        h2d_stream = torch.cuda.Stream()
        
        for layer in model.layers:
            for module_name in offload_modules:
                module = getattr(layer, module_name, None)
                if module is not None:
                    wrap_with_offloading(module, d2h_stream, h2d_stream)
```

### 6. Structure-Aware Planner：统一规划 [veScale + Megatron]

```python
class StructureAwarePlanner:
    """
    统一规划器，生成完整的三维并行分配计划。
    
    [veScale] 核心算法：Structure-Aware Planning
      → 分析模型结构，识别 co-location 约束，生成 RaggedShard 计划
    
    [Megatron] PP stage 划分逻辑
      → 考虑 MoE 层远大于 Dense 层，按实际负载均分
    
    [Megatron] Expert 放置逻辑
      → 均匀分到 EP 组内的 GPU 上
    """
    
    def __init__(self, model, cluster_config, optimizer_type='adam'):
        self.model = model
        self.cluster = cluster_config
        self.optimizer_type = optimizer_type
    
    def plan(self):
        pp_partition = self._plan_pp_stages()
        fsdp_plan = self._plan_fsdp_sharding()
        ep_placement = self._plan_expert_placement()
        return DistributedPlan(pp_partition, fsdp_plan, ep_placement)
    
    def _plan_fsdp_sharding(self):
        """
        [veScale] Structure-Aware Planning 算法
        
        Step 1: 根据优化器类型确定 block 边界
        Step 2: 构建 co-location 约束图
        Step 3: Union-Find 合并为 super-blocks
        Step 4: 贪心分配到 FSDP ranks
        """
        plan = {}
        
        for name, param in self.model.named_parameters():
            if 'expert' in name:
                continue  # Expert 参数由 EP 管理
            
            if param.dim() < 2:
                # 1D 参数 (bias, layernorm) → 标准切分
                plan[name] = RaggedShardSpec(shard_type='flat', num_shards=self.cluster.fsdp_size)
                continue
            
            # [veScale] Step 1: 确定 block 边界
            if self.optimizer_type == 'shampoo':
                block_size = min(128, *param.shape)
                constraint = 'kronecker'
            elif self.optimizer_type == 'muon':
                block_size = min(256, param.shape[0])
                constraint = 'newton_schulz'
            elif self.cluster.quant_block_size is not None:
                block_size = self.cluster.quant_block_size
                constraint = 'quantization'
            else:
                # Adam + 无量化 → 标准行切分即可
                plan[name] = RaggedShardSpec(shard_type='row_wise', num_shards=self.cluster.fsdp_size)
                continue
            
            # [veScale] Step 2-3: 构建 blocks 并确定 co-location 约束
            blocks = partition_into_blocks(param.shape, block_size)
            
            if constraint == 'kronecker':
                # Shampoo: W_block 和其 Kronecker 因子 L_block, R_block 必须同 GPU
                co_location_groups = group_blocks_for_kronecker(blocks, param.shape)
            elif constraint == 'newton_schulz':
                # Muon: W_block 必须完整在一个 GPU（Newton-Schulz 迭代需要）
                co_location_groups = group_blocks_for_muon(blocks, param.shape)
            else:
                # 量化: 每个 block 不可拆分
                co_location_groups = [[b] for b in blocks]
            
            # [veScale] Step 4: 贪心分配（按 super-block 大小降序，分配到最空的 GPU）
            assignments = greedy_balanced_assign(
                co_location_groups, 
                num_shards=self.cluster.fsdp_size
            )
            
            plan[name] = RaggedShardSpec(
                shard_type='block_wise',
                block_assignments=assignments,
                full_shape=param.shape,
                num_shards=self.cluster.fsdp_size,
            )
        
        return plan
    
    def _plan_pp_stages(self):
        """
        [Megatron] PP stage 划分
        
        关键 [Megatron §11.1]: 最小化模型并行，最大化数据并行
        MoE 层的 cost 远大于 Dense 层 → 按实际负载均分 stage
        
        支持 VPP (Virtual Pipeline Parallel) [Megatron §9.3]:
          不要求均匀层分配，不同 virtual stage 可有不同层数/类型
        """
        layers = list(self.model.layers)
        num_stages = self.cluster.pp_size
        vpp_size = self.cluster.get('vpp_size', 1)
        num_virtual_stages = num_stages * vpp_size
        
        # 计算每层的归一化 cost
        layer_costs = []
        for layer in layers:
            if hasattr(layer, 'moe'):
                # MoE 层: attention params + local_experts params
                local_experts = layer.moe.num_experts // self.cluster.ep_size
                cost = (layer.attn_params + local_experts * layer.expert_params)
            else:
                cost = layer.total_params
            layer_costs.append(cost)
        
        # [Megatron] 贪心均分
        target = sum(layer_costs) / num_virtual_stages
        stages = []
        current, current_cost = [], 0
        
        for i, cost in enumerate(layer_costs):
            current.append(i)
            current_cost += cost
            if current_cost >= target and len(stages) < num_virtual_stages - 1:
                stages.append(current)
                current, current_cost = [], 0
        stages.append(current)
        
        return stages
    
    def _plan_expert_placement(self):
        """
        [Megatron] Expert 均匀放置到 EP 组内
        E 个 expert / EP_size 个 GPU = 每 GPU 若干个 expert
        """
        E = self.model.num_experts
        placement = {}
        experts_per_gpu = E // self.cluster.ep_size
        for rank in range(self.cluster.ep_size):
            for j in range(experts_per_gpu):
                eid = rank * experts_per_gpu + j
                placement[eid] = rank
        return placement
```

### 7. EP 通信 Overlap [Megatron 逻辑]

```python
class EPOverlapScheduler:
    """
    [Megatron] EP 通信与计算的重叠策略
    
    Megatron 论文 §5.2 的核心思路：
      双 CUDA Stream 分离通信和计算
      MoE dispatch 与 shared expert 计算 overlap
      BWD 中 W-grad 与 dispatch 通信 overlap
    
    时间线（单个 MoE 层的前向）：
    
    Compute Stream: [Router] ─────────── [Expert GEMM] ─── [SharedExp]
    Comm Stream:              [Dispatch]                  [Combine]
                               └── overlap ──┘
    
    如果没有 shared expert:
    Compute Stream: [Router] ─────────── [Expert GEMM] ──────────────
    Comm Stream:              [Dispatch]                 [Combine]
                               └ 只在 dispatch 期间 overlap 不了太多
    
    反向时更关键 [Megatron §5.2]:
      BWD MLP 拆分为 W-grad (权重梯度) 和 D-grad (数据梯度)
      W-grad 不依赖 dispatch → 可与 dispatch 通信 overlap
      
    Compute Stream: [D-grad attn] ── [W-grad MoE] ── [D-grad MoE]
    Comm Stream:                     [B/dispatch]    [B/combine]
                                      └── overlap ──┘
    """
    
    def __init__(self):
        self.compute_stream = torch.cuda.current_stream()
        self.comm_stream = torch.cuda.Stream()
    
    def forward_with_overlap(self, moe_layer, hidden_states):
        """MoE 前向：dispatch 与 shared expert overlap"""
        # Router (compute stream)
        probs, routing_map = moe_layer.router(hidden_states)
        
        # Dispatch (comm stream)
        with torch.cuda.stream(self.comm_stream):
            dispatched, meta = moe_layer.dispatcher.dispatch(
                hidden_states, probs, routing_map
            )
        
        # Shared Expert (compute stream, overlap with dispatch)
        if moe_layer.shared_expert is not None:
            shared_out = moe_layer.shared_expert(hidden_states)
        
        # 等待 dispatch 完成
        self.compute_stream.wait_stream(self.comm_stream)
        
        # Expert Compute (compute stream)
        expert_out = moe_layer.experts(dispatched, meta.tokens_per_expert)
        
        # Combine (comm stream)
        with torch.cuda.stream(self.comm_stream):
            combined = moe_layer.dispatcher.combine(expert_out, meta)
        
        self.compute_stream.wait_stream(self.comm_stream)
        
        if moe_layer.shared_expert is not None:
            combined = combined + shared_out
        
        return combined
```

### 8. FP8 Block-wise 量化训练 [Megatron + veScale 协同]

```python
class FP8BlockwiseTraining:
    """
    FP8 Block-wise 量化训练。
    
    [Megatron] FP8 训练的整体框架:
      - 选择性精度：Router 保持 FP32（确保稳定的专家选择）
      - Expert GEMM 使用 FP8（计算量占比最大）
      - Embedding/Output Layer 保持 BF16
    
    [veScale] RaggedShard 的配合:
      - FSDP 分片必须对齐 FP8 block 边界（通常 128×128）
      - RaggedShard 天然满足这个约束
      - FP8 scale factor 是 per-block 的 → block 不能被切碎
      
    这是 veScale-FSDP 和 Megatron FP8 的交汇点：
      没有 RaggedShard → FSDP 切碎 block → FP8 量化语义错误
      有 RaggedShard → block 完整保留 → FP8 正确工作
    """
    
    @staticmethod
    def configure(model, planner, recipe='blockwise_fp8'):
        """
        [Megatron] FP8 配置
        
        Blockwise FP8（Hopper 推荐 [Megatron §7.3]）：
          格式：E4M3 for all
          量化粒度：激活 1×128 tiles, 权重 128×128 blocks
          已在 DeepSeek-V3、Minimax-M2 等生产验证
        
        [veScale] RaggedShard 要求：
          FSDP 分片的 block_size 必须 = 量化 block_size
          → planner 自动对齐
        """
        quant_block_size = 128  # 标准 FP8 block size
        
        # [veScale] 更新 planner 的 block 约束
        planner.cluster.quant_block_size = quant_block_size
        
        # [Megatron] 精度配置
        precision_config = {
            'router': 'fp32',         # 保护路由稳定性
            'expert_gemm': 'fp8_e4m3', 
            'attention': 'fp8_e4m3',
            'embedding': 'bf16',       # 保护
            'output_head': 'bf16',     # 保护
            'optimizer_states': 'bf16', 
        }
        
        # [Megatron §7.5] MoE 特有处理:
        # FP8 GEMM 要求维度对齐到 16 的倍数
        # → routing map padding 而非 token padding
        # → 填充路由表（几个额外 token）而非复制 token（大量内存）
        model.moe_config.fp8_routing_pad_to = 16
        
        return precision_config
```

### 9. 训练循环 [Megatron PP 调度]

```python
class SimpleMoETrainer:
    """
    整合训练循环。
    
    [Megatron] Pipeline Schedule: 1F1B 或 VPP
    [veScale] FSDP 通信调度: Lazy AG + Hierarchical RS
    [Megatron] EP 通信调度: 双 Stream overlap
    """
    
    def __init__(self, model, config):
        # --- 规划 ---
        self.planner = StructureAwarePlanner(model, config.cluster, config.optimizer_type)
        plan = self.planner.plan()
        
        # --- PP 切分 [Megatron] ---
        self.pp_rank = config.cluster.pp_rank
        my_layer_indices = plan.pp_partition[self.pp_rank]
        self.layers = nn.ModuleList([model.layers[i] for i in my_layer_indices])
        
        # --- Dense 层包装 FSDP [veScale] ---
        for layer in self.layers:
            layer.attn = RaggedFSDP(
                layer.attn, config.pg.fsdp_group, plan.fsdp_plan, config.optimizer_type
            )
            layer.norm1 = RaggedFSDP(layer.norm1, config.pg.fsdp_group, plan.fsdp_plan)
            layer.norm2 = RaggedFSDP(layer.norm2, config.pg.fsdp_group, plan.fsdp_plan)
        
        # --- MoE 层包装 EP [Megatron] ---
        for layer in self.layers:
            if hasattr(layer, 'moe'):
                layer.moe = MoELayer(config.moe, config.pg.ep_group, config.pg.edp_group)
        
        # --- 内存优化 [Megatron] ---
        SimpleMoEMemoryOptimizer.configure_recomputation(self, strategy='selective')
        if config.enable_offloading:
            SimpleMoEMemoryOptimizer.configure_offloading(self, ['attention', 'expert_ffn'])
        
        # --- FP8 [Megatron + veScale] ---
        if config.fp8_enabled:
            self.precision_config = FP8BlockwiseTraining.configure(
                model, self.planner, recipe='blockwise_fp8'
            )
        
        # --- EP 通信 Overlap [Megatron] ---
        self.ep_overlap = EPOverlapScheduler()
        
        # --- 优化器 ---
        self.optimizer = self._build_optimizer(config)
        
        # --- PP Schedule [Megatron] ---
        self.schedule = OneFOneBSchedule(
            num_stages=config.cluster.pp_size,
            num_micro_batches=config.num_micro_batches,
            pp_group=config.pg.pp_group,
        )
    
    def train_step(self, batch):
        """
        [Megatron] 1F1B Pipeline 训练
        
        每个 micro-batch 按 stage 执行:
          Forward:
            for layer in my_layers:
              x = layer.norm1(x)
              x = x + layer.attn(x)      ← [veScale] FSDP AG → compute → release
              x = layer.norm2(x)
              if moe_layer:
                x = x + layer.moe(x)     ← [Megatron] Route → Dispatch → Compute → Combine
              else:
                x = x + layer.ffn(x)
          
          Backward:
            反向传播各层
            [veScale] FSDP Reduce-Scatter 梯度
            [Megatron] Expert 梯度在 EDP 组内 AllReduce
        """
        micro_batches = split_into_micro_batches(batch, self.schedule.num_micro_batches)
        total_loss = torch.tensor(0.0, device='cuda')
        
        for action in self.schedule:
            if action.is_forward:
                mb = micro_batches[action.mb_id]
                loss = self._forward(mb)
                total_loss += loss.detach()
                
            elif action.is_backward:
                self._backward(action.mb_id)
                
            elif action.is_send:
                # [Megatron] PP 激活传递
                pp_send(action.activations, action.dest_rank, self.schedule.pp_group)
                
            elif action.is_recv:
                action.activations = pp_recv(action.src_rank, self.schedule.pp_group)
        
        # [veScale] FSDP 梯度 RS 已在 backward 中完成
        # [Megatron] Expert 梯度 EDP AllReduce 也在 backward 中完成
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return total_loss / len(micro_batches)
    
    def _forward(self, micro_batch):
        x = micro_batch
        for layer in self.layers:
            residual = x
            
            # LayerNorm + Attention [veScale FSDP]
            x = layer.norm1(x)
            # [veScale] Lazy AG: 这里才触发 Attention 参数的 All-Gather
            full_attn_params = layer.attn.all_gather_params_all()
            x = layer.attn.compute(x, full_attn_params)
            layer.attn.release_full_params(full_attn_params)
            x = residual + x
            
            residual = x
            x = layer.norm2(x)
            
            if hasattr(layer, 'moe'):
                # MoE [Megatron 四阶段]
                # [Megatron] EP overlap: dispatch 与 shared expert 计算重叠
                moe_out, aux_loss = self.ep_overlap.forward_with_overlap(layer.moe, x)
                x = residual + moe_out
            else:
                x = residual + layer.ffn(x)
        
        return x
    
    def _build_optimizer(self, config):
        """
        [Megatron + veScale] 优化器构建
        
        Muon 优化器 [Megatron §9.3]:
          矩阵感知优化，正交化整个权重矩阵
          MuonClip 处理 query-key 点积无界增长
        
        + [veScale] RaggedShard 兼容:
          Muon 需要完整子矩阵做 Newton-Schulz → RaggedShard 对齐 Muon block
        """
        dense_params = []
        expert_params = []
        
        for layer in self.layers:
            for name, p in layer.named_parameters():
                if 'expert' in name or 'w_gate' in name or 'w_up' in name or 'w_down' in name:
                    expert_params.append(p)
                else:
                    dense_params.append(p)
        
        if config.optimizer_type == 'muon':
            # [Megatron] Muon for dense, Adam for experts (Kimi K2 做法)
            return HybridOptimizer([
                {'params': dense_params, 'optimizer': 'muon', 'lr': config.lr},
                {'params': expert_params, 'optimizer': 'adam', 'lr': config.expert_lr},
            ])
        elif config.optimizer_type == 'shampoo':
            return DistributedShampoo([
                {'params': dense_params, 'lr': config.lr},
                {'params': expert_params, 'lr': config.expert_lr},
            ])
        else:
            return torch.optim.AdamW([
                {'params': dense_params, 'lr': config.lr},
                {'params': expert_params, 'lr': config.expert_lr},
            ], weight_decay=config.weight_decay)
```

### 10. 用户 API

```python
# ==========================================
# SimpleMoE 使用示例
# ==========================================

from simple_moe import SimpleMoETrainer, ClusterConfig, MoEModelConfig

# 1. 模型定义（标准 PyTorch）
model = MoETransformerLM(
    vocab_size=128256,
    hidden_dim=4096,
    num_layers=32,
    num_moe_layers=30,          # 前 2 层 dense, 后 30 层 MoE
    num_experts=64,
    top_k=4,
    expert_ffn_dim=2048,
    num_shared_experts=1,        # [Megatron] DeepSeek-V3 style shared expert
    score_func='sigmoid',        # [Megatron] sigmoid + aux-loss-free
)

# 2. 集群配置
cluster = ClusterConfig(
    total_gpus=128,
    gpus_per_node=8,
    pp_size=4,                   # [Megatron] Pipeline stages
    ep_size=8,                   # [Megatron] Expert Parallel (节点内)
    fsdp_size=4,                 # [veScale] FSDP (跨节点)
    quant_block_size=128,        # [veScale] FP8 block size → RaggedShard 对齐
)

# 3. 训练配置
config = TrainConfig(
    cluster=cluster,
    optimizer_type='muon',       # [Megatron + veScale] Muon + RaggedShard
    fp8_enabled=True,            # [Megatron + veScale] FP8 + block-aligned FSDP
    enable_offloading=True,      # [Megatron] 激活 offload
    lr=3e-4,
    expert_lr=1e-4,
    batch_size=1024,
    seq_len=4096,
    num_micro_batches=8,         # [Megatron] Pipeline micro-batches
)

# 4. 初始化（自动规划 + 包装）
trainer = SimpleMoETrainer(model, config)

# 5. 标准训练循环
for epoch in range(num_epochs):
    for batch in dataloader:
        loss = trainer.train_step(batch)
```

---

## 第三部分：两篇论文在框架中的分工总结

```
┌──────────────────────────────────────────────────────────────────┐
│                  SimpleMoE 技术来源总览                            │
├──────────────────────┬───────────┬───────────┬──────────────────┤
│ 功能模块              │ veScale   │ Megatron  │ 职责             │
├──────────────────────┼───────────┼───────────┼──────────────────┤
│ Dense 参数分片         │ ★ 主导    │           │ RaggedShard      │
│ 分片规划算法           │ ★ 主导    │           │ Structure-Aware  │
│ FSDP 通信（AG/RS）    │ ★ 主导    │           │ Lazy + 分层      │
│ Buffer Pool          │ ★ 主导    │           │ AG/A2A buffer    │
│ Shampoo/Muon 兼容     │ ★ 主导    │ 部分      │ block 对齐       │
│ FP8 block 对齐        │ ★ 主导    │ 部分      │ shard 对齐 block │
├──────────────────────┼───────────┼───────────┼──────────────────┤
│ MoE 四阶段通路        │           │ ★ 主导    │ R→D→C→C          │
│ Top-K Router         │           │ ★ 主导    │ Gate + 负载均衡   │
│ Token Dispatcher     │           │ ★ 主导    │ All-to-All       │
│ Grouped GEMM         │           │ ★ 主导    │ Expert 计算      │
│ Memory-Eff Permut    │           │ ★ 主导    │ 激活内存省 26GB   │
│ EP 通信 Overlap      │           │ ★ 主导    │ 双 Stream        │
│ PP Schedule (1F1B)   │           │ ★ 主导    │ Pipeline 调度    │
│ VPP                  │           │ ★ 主导    │ Virtual Pipeline  │
│ Fine-grained Recomp  │           │ ★ 主导    │ 选择性重算        │
│ Activation Offload   │           │ ★ 主导    │ 细粒度 offload   │
│ FP8 训练方案          │           │ ★ 主导    │ 精度策略          │
│ 负载均衡              │           │ ★ 主导    │ Aux-Loss-Free    │
│ Shared Expert        │           │ ★ 主导    │ 并行 overlap     │
│ 进程组管理            │           │ ★ 主导    │ Dual DeviceMesh  │
│ Checkpoint           │           │ ★ 主导    │ 并行无关保存      │
│ Muon 优化器实现       │           │ ★ 主导    │ MuonClip         │
├──────────────────────┼───────────┼───────────┼──────────────────┤
│ FP8 + RaggedShard    │ 协同      │ 协同      │ 两者交汇点        │
│ Muon + RaggedShard   │ 协同      │ 协同      │ 两者交汇点        │
└──────────────────────┴───────────┴───────────┴──────────────────┘

比例：
  veScale-FSDP 贡献：~30%（核心分片思想 + 通信优化）
  Megatron-Core 贡献：~70%（MoE 全链路 + 内存/计算/生产特性）
  
两者的交汇点（也是本框架的独特价值）：
  1. FP8 block-wise 量化 + FSDP 
     → Megatron 提供 FP8 方案，veScale 提供 block-aligned 分片
     → 传统 FSDP 无法做到（block 被切碎）
  
  2. Muon/Shampoo 优化器 + FSDP + MoE
     → Megatron 提供 Muon 实现，veScale 提供 Kronecker-aware 分片
     → Dense 层用 RaggedShard 保证优化器正确性
     → Expert 层用 EP 天然满足（每个 Expert 完整在一个 GPU）
```

---

## 第四部分：为什么不需要 TP / CP

### 不需要 TP 的理由

```
[Megatron §11.1] 原则："最小化模型并行，最大化数据并行"

TP 的核心作用是切分单层参数（当单层装不下一个 GPU 时）。
在 SimpleMoE 中，用 FSDP 替代 TP：

  TP 做法：
    W_q, W_k, W_v 按 head 维度切分到多个 GPU
    每个 GPU 做部分 Attention → AllReduce 汇总
    通信在每一层、每一次 forward/backward 都发生

  FSDP 做法（本框架）：
    W_q, W_k, W_v 全量 shard 存储（每 GPU 存 1/FSDP_size）
    forward 前 All-Gather 恢复完整参数 → 单 GPU 做完整 Attention
    forward 后释放完整参数 → 内存回到 1/FSDP_size
    
  对比：
    TP 通信量 = 2 × hidden_dim × seq_len × batch_per_gpu (每层)
    FSDP 通信量 = param_size × (FSDP_size - 1) / FSDP_size (每层)
    
    当 batch 足够大时，FSDP 通信量更可预测且可 overlap
    
  [Megatron §11.2] 实际案例：
    GB200 (192GB/GPU) 上 DeepSeek-V3 最优配置是 TP=1
    → 内存足够时，TP 不如 FSDP
    
  Expert 层更不需要 TP：
    [Megatron §11.1] "Expert 层优先 EP 而非 TP（更好 GEMM 效率 + 更低通信）"
    EP 把 Expert 分到不同 GPU → 每个 GPU 做完整 Expert GEMM
    如果 TP 切分 Expert → 小 GEMM 被进一步碎片化 → 利用率更差
```

### 不需要 CP 的理由

```
CP 的核心作用：分布式处理超长序列的 Attention（O(s²) 复杂度）

[Megatron §8.1] 长上下文分析：
  4K tokens:  MoE 占 ~59.4%（MoE 是瓶颈）
  64K tokens: SDPA 占 ~69.7%（Attention 变成瓶颈）

SimpleMoE 的目标场景：
  seq_len ≤ 8K 的标准 MoE 训练
  → MoE 是瓶颈，不是 Attention
  → CP 的收益有限
  
[Megatron §11.1] 指导：
  "长序列 ≥8K 启用 CP；<4K 通常不值得"
  
SimpleMoE 选择 seq_len ≤ 8K → CP 不值得引入
如果未来确实需要长序列：
  可作为正交扩展叠加，不影响框架核心设计
```

---

*设计文档 | veScale-FSDP (arXiv:2602.22437) + Megatron-Core MoE (arXiv:2603.07685)*  
*2026-03-23*
