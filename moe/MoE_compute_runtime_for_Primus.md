# MoE Compute Runtime 设计：面向 Primus-dev 集成

> **作者背景:** Training Systems & AI Infrastructure Engineer — PyTorch / Megatron-style 分布式训练、GPU 性能优化、多后端统一入口  
> **目标:** 设计一套 MoE 计算 Runtime，集成进 `3rd/Primus-dev`，在 AMD ROCm (MI300X) 上实现大规模 MoE 训练  
> **论文基础:** 仅取大厂主导路线 — ByteDance (MegaScale-MoE, Comet), Microsoft (Tutel), Meta (MoEBlaze), NVIDIA (Megatron-Core MoE, Parallel Folding)  
> **集成目标:** Primus-dev 的 Megatron 后端 (flex dispatcher API) + TorchTitan 后端 (grouped GEMM hook)

---

## 0. 设计定位

Primus-dev 当前 MoE 栈：

```text
Megatron MoE Layer (upstream submodule)
  ├── PrimusTopKRouter          ← primus/.../moe/router.py (fused routing + aux loss)
  ├── PrimusTurboDeepEPTokenDispatcher ← primus/.../extensions/primus_turbo.py (DeepEP flex API)
  └── GroupedMLP / TE GroupedMLP      ← Turbo grouped GEMM / FP8
```

**本设计不替换 Megatron MoE Layer 的外壳**，而是在其内部三个阶段（Routing → Dispatch/Compute → Combine）插入一个统一的 **MoE Compute Runtime**，作为 Primus 自有的可演进层。

---

## 1. 总体架构

```text
┌─────────────────────────────────────────────────────────────────────┐
│                    Megatron MoELayer / TorchTitan MoE              │
│  forward(hidden_states) ──────────────────────────────────────────→ │
│         │                                                           │
│         ▼                                                           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              MoE Compute Runtime (本设计)                    │   │
│  │                                                             │   │
│  │  ┌──────────┐   ┌──────────────┐   ┌──────────────────┐   │   │
│  │  │ Runtime  │──→│  Dispatch    │──→│  Expert Compute  │   │   │
│  │  │ Planner  │   │  Engine      │   │  Engine          │   │   │
│  │  │          │   │              │   │                  │   │   │
│  │  │ •策略选择 │   │ •Token排布   │   │ •Grouped GEMM   │   │   │
│  │  │ •内存预算 │   │ •A2A 执行    │   │ •Fused SwiGLU   │   │   │
│  │  │ •内核决策 │   │ •Chunk流水   │   │ •Block-sparse   │   │   │
│  │  └──────────┘   └──────────────┘   └──────────────────┘   │   │
│  │         │               │                    │             │   │
│  │         ▼               ▼                    ▼             │   │
│  │  ┌──────────────────────────────────────────────────┐     │   │
│  │  │           Combine Engine                          │     │   │
│  │  │  •Weighted merge  •Output scatter  •Grad routing  │     │   │
│  │  └──────────────────────────────────────────────────┘     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│         │                                                           │
│         ▼                                                           │
│  output hidden_states                                               │
└─────────────────────────────────────────────────────────────────────┘
```

四个核心组件：

| 组件 | 职责 | 大厂论文来源 |
|------|------|-------------|
| **Runtime Planner** | 控制面：策略选择、内存预算、内核路由 | Tutel (自适应并行), MegaScale (拓扑感知) |
| **Dispatch Engine** | Token 排布 + A2A 通信 + Chunk 流水 | Comet (tile overlap), DeepEP (flex API) |
| **Expert Compute Engine** | 专家计算内核选择与执行 | MoEBlaze (fused), MegaBlocks (block-sparse), Megatron (grouped GEMM) |
| **Combine Engine** | 加权合并 + 输出分发 | MoEBlaze (index-driven gather) |

---

## 2. Runtime Planner（控制面）

### 2.1 设计原则

从 Tutel 和 MegaScale-MoE 提取的核心认知：
- Runtime 必须拥抱动态性（路由负载每 step 变化）
- Layout 优先于算子优化（数据布局决定通信和算子效率上限）
- 内存与通信同等优先级（MoE 里二者常比算力更早成为瓶颈）

### 2.2 策略状态

```python
@dataclass
class MoERuntimeStats:
    """每 step 采集，驱动 Planner 决策"""
    load_skew: float          # max(tokens_per_expert) / avg(tokens_per_expert)
    tokens_per_expert: Tensor  # [num_experts] 每个 expert 的 token 数
    hbm_used_ratio: float     # 当前 HBM 占用比
    a2a_latency_ms: float     # 上一步 A2A 实测延迟
    seq_len: int              # 当前序列长度
    ep_size: int              # Expert Parallel 组大小
    tp_size: int              # Tensor Parallel 组大小
    topology: str             # "xgmi_intra" | "rdma_inter" | "mixed"
```

### 2.3 策略选择器

```python
class MoERuntimePlanner:
    """
    控制面核心：根据运行时统计选择最优配置。
    借鉴 Tutel 的 capacity_bucket -> best_config 字典 +
    MegaScale 的拓扑感知放置。
    """

    def __init__(self, cluster_topo: ClusterTopology, config: MoERuntimeConfig):
        self.config = config
        self.topo = cluster_topo
        self.policy_cache: Dict[int, RuntimePolicy] = {}
        self.eval_interval = config.planner_eval_interval  # 每 N step 重评估

    def select_policy(self, stats: MoERuntimeStats) -> RuntimePolicy:
        bucket_key = self._compute_bucket(stats)

        if bucket_key in self.policy_cache:
            return self.policy_cache[bucket_key]

        policy = RuntimePolicy(
            dispatch_cfg=self._select_dispatch(stats),
            compute_cfg=self._select_compute(stats),
            memory_cfg=self._select_memory(stats),
            overlap_cfg=self._select_overlap(stats),
        )
        self.policy_cache[bucket_key] = policy
        return policy

    def _select_dispatch(self, stats) -> DispatchConfig:
        # Tutel: 小规模 Linear A2A, 大规模 2DH A2A
        if stats.ep_size <= 8 and stats.topology == "xgmi_intra":
            a2a_algo = "linear"
        else:
            a2a_algo = "hierarchical_2d"

        # Comet: chunk 数量由 A2A 延迟和计算量联合决定
        chunk_degree = self._search_chunk_degree(stats)

        return DispatchConfig(
            a2a_algorithm=a2a_algo,
            chunk_degree=chunk_degree,
            use_deepep=stats.ep_size > 1,
            permute_fusion=True,
        )

    def _select_compute(self, stats) -> ComputeConfig:
        # 双内核策略 (MoEBlaze vs MegaBlocks)
        if stats.hbm_used_ratio > 0.85:
            # 内存压力大：MoEBlaze 风格 fused routing+MLP (省 buffer)
            kernel = "fused_index_driven"
        elif stats.load_skew > 3.0:
            # 负载极不均衡：block-sparse (MegaBlocks) 避免 padding 浪费
            kernel = "block_sparse"
        else:
            # 默认：grouped GEMM (Megatron-Core 路线, 最成熟)
            kernel = "grouped_gemm"

        return ComputeConfig(
            kernel_type=kernel,
            use_fp8=self.config.enable_fp8,
            fuse_swiglu=True,
            expert_merge_threshold=self.config.small_expert_threshold,
        )

    def _select_memory(self, stats) -> MemoryConfig:
        if stats.hbm_used_ratio > 0.90:
            return MemoryConfig(
                recompute_level="aggressive",  # SwiGLU + up_proj + LN
                routing_dematerialize=True,
                reduce_chunk_degree=True,
            )
        elif stats.hbm_used_ratio > 0.75:
            return MemoryConfig(
                recompute_level="selective",  # SwiGLU only
                routing_dematerialize=True,
                reduce_chunk_degree=False,
            )
        else:
            return MemoryConfig(
                recompute_level="none",
                routing_dematerialize=False,
                reduce_chunk_degree=False,
            )

    def _select_overlap(self, stats) -> OverlapConfig:
        # 避免伪 overlap：chunk 过小 launch 开销吞掉收益
        min_tokens_per_chunk = 256
        effective_chunk = max(
            stats.tokens_per_expert.float().mean().item() // stats.ep_size,
            min_tokens_per_chunk,
        )
        return OverlapConfig(
            enable_3stage_pipeline=stats.a2a_latency_ms > 0.5,
            min_chunk_tokens=effective_chunk,
            detect_sm_contention=True,
        )
```

---

## 3. Dispatch Engine（Token 排布 + A2A）

### 3.1 设计来源

| 论文 | 关键技术 | 本设计采纳 |
|------|---------|-----------|
| **Comet** (ByteDance) | Tile-level GEMM-comm overlap, warp specialization, RDMA per tile | Chunk 流水 + 多 stream overlap |
| **DeepEP** (Megatron) | Flex dispatcher API (pre/exec/post 三阶段) | 直接复用 Primus 已有的 flex API |
| **MoEBlaze** (Meta) | Index-driven routing, 无大 buffer | 轻量索引结构 |
| **Tutel** (Microsoft) | Adaptive A2A (linear vs 2DH) | A2A 算法自适应 |

### 3.2 与 Primus 的集成接口

Runtime 的 Dispatch Engine 实现 Megatron flex dispatcher 的三阶段协议，作为 `PrimusTurboDeepEPTokenDispatcher` 的可替换后端：

```python
class MoEComputeRuntimeDispatcher(MoETokenDispatcher):
    """
    实现 Megatron flex API，内部由 Runtime Planner 驱动。
    注册方式：通过 Primus patch 机制替换 token_dispatcher。
    """

    def __init__(self, config, ep_group, tp_group, tp_ep_group):
        super().__init__()
        self.planner = MoERuntimePlanner(
            cluster_topo=detect_cluster_topology(ep_group),
            config=config,
        )
        self.ep_group = ep_group
        self.tp_group = tp_group
        self.dispatch_impl = self._build_dispatch_impl(config)

    # ---- Flex Dispatcher Protocol (与 Megatron MoELayer 对齐) ----

    def dispatch_preprocess(self, routing_map, probs, **kwargs):
        """Phase 1: 构建路由元数据 (轻量索引, 不分配大 buffer)"""
        stats = self._collect_stats(routing_map, probs)
        self.current_policy = self.planner.select_policy(stats)

        # MoEBlaze 风格：只构建索引，不 materialize routed_tokens
        route_meta = build_compact_route_indices(
            routing_map=routing_map,
            probs=probs,
            ep_size=dist.get_world_size(self.ep_group),
            dematerialize=self.current_policy.memory_cfg.routing_dematerialize,
        )
        return route_meta

    def token_dispatch(self, hidden_states, route_meta, **kwargs):
        """Phase 2: 执行 permute + A2A dispatch"""
        cfg = self.current_policy.dispatch_cfg

        if cfg.chunk_degree > 1:
            # Comet 启发：chunk 流水 dispatch
            return self._chunked_dispatch(hidden_states, route_meta, cfg)
        else:
            return self._single_dispatch(hidden_states, route_meta, cfg)

    def dispatch_postprocess(self, dispatched, route_meta, **kwargs):
        """Phase 3: 整理 per-expert layout, 返回 tokens_per_expert"""
        return finalize_expert_layout(
            dispatched, route_meta,
            pad_to_capacity=False,  # dropless 优先
        )

    def combine_preprocess(self, expert_output, route_meta, **kwargs):
        return expert_output, route_meta

    def token_combine(self, expert_output, route_meta, **kwargs):
        """反向 A2A gather + unpermute"""
        cfg = self.current_policy.dispatch_cfg
        if cfg.chunk_degree > 1:
            return self._chunked_combine(expert_output, route_meta, cfg)
        else:
            return self._single_combine(expert_output, route_meta, cfg)

    def combine_postprocess(self, combined, probs, route_meta, **kwargs):
        """加权合并 + output scatter"""
        return weighted_merge(combined, probs, route_meta)

    # ---- Chunk 流水实现 (Comet + Tutel 启发) ----

    def _chunked_dispatch(self, hidden, route_meta, cfg):
        """
        三段重叠流水：
          Stream 0: A2A dispatch(chunk i)
          Stream 1: expert compute(chunk i-1)  ← 由 Compute Engine 回调
          Stream 2: A2A gather(chunk i-2)

        借鉴 Comet 的 tile-level overlap 思路，
        但在 ROCm/RCCL 上用 stream + event 实现（非 warp specialization）。
        """
        chunks = split_by_route(hidden, route_meta, cfg.chunk_degree)
        dispatch_stream = torch.cuda.Stream()
        results = []

        for i, chunk in enumerate(chunks):
            with torch.cuda.stream(dispatch_stream):
                dispatched = self._exec_a2a_dispatch(chunk, route_meta, cfg)
            dispatch_stream.synchronize()
            results.append(dispatched)

        return concat_dispatched(results, route_meta)
```

### 3.3 AMD 拓扑感知 A2A

MI300X 8-GPU 节点内 XGMI mesh 带宽远高于跨节点 RDMA。Dispatch Engine 根据 Planner 的拓扑信息选择 A2A 路径：

```text
EP group 在同一节点 (XGMI):
  → Linear A2A via RCCL (低延迟, 高带宽)
  → 可选: P2P memcpy dispatch (research_direction_C 的 FSEP 路线)

EP group 跨节点 (RDMA):
  → Hierarchical 2D A2A (Tutel 风格)
  → Step 1: 节点内 XGMI 聚合
  → Step 2: 跨节点 RDMA 交换
  → Step 3: 节点内 XGMI 分发
```

---

## 4. Expert Compute Engine（专家计算）

### 4.1 三条内核路线

从大厂论文中提取三条互补的计算路线，Runtime 按场景切换：

| 路线 | 来源 | 适用场景 | 优势 | 劣势 |
|------|------|---------|------|------|
| **Grouped GEMM** | Megatron-Core (NVIDIA) | 默认路线，负载均衡时 | 成熟、FP8 支持好、hipBLASLt 可用 | 负载不均时 padding 浪费 |
| **Fused Index-Driven** | MoEBlaze (Meta) | 内存压力大时 | 省 50%+ 激活内存、无大 buffer | 实现复杂度高 |
| **Block-Sparse** | MegaBlocks (Databricks) | 极端负载不均时 | Dropless、无 padding | 稀疏格式开销、ROCm 支持需验证 |

### 4.2 统一 Expert Compute 接口

```python
class ExpertComputeEngine:
    """
    统一专家计算接口，内部按 ComputeConfig 选择内核。
    与 Primus-Turbo grouped GEMM 和 TorchTitan _run_experts_grouped_mm 对齐。
    """

    def __init__(self, expert_weights: ExpertWeights, config: MoERuntimeConfig):
        self.weights = expert_weights
        self.config = config
        self.kernels = {
            "grouped_gemm": GroupedGEMMKernel(expert_weights, config),
            "fused_index_driven": FusedIndexDrivenKernel(expert_weights, config),
            "block_sparse": BlockSparseKernel(expert_weights, config),
        }

    def forward(
        self,
        dispatched_tokens: Tensor,
        tokens_per_expert: Tensor,
        compute_cfg: ComputeConfig,
    ) -> Tensor:
        kernel = self.kernels[compute_cfg.kernel_type]
        return kernel.execute(dispatched_tokens, tokens_per_expert)


class GroupedGEMMKernel:
    """
    Megatron-Core / Primus-Turbo 路线。
    hipBLASLt grouped GEMM + 可选 FP8 (E4M3)。
    """

    def execute(self, tokens, tokens_per_expert):
        # SwiGLU: gate_proj + up_proj → silu(gate) * up → down_proj
        gate_up = primus_turbo.grouped_gemm(
            tokens, self.w_gate_up,
            tokens_per_expert,
            fp8=self.use_fp8,
        )
        gate, up = gate_up.chunk(2, dim=-1)
        intermediate = F.silu(gate) * up
        output = primus_turbo.grouped_gemm(
            intermediate, self.w_down,
            tokens_per_expert,
            fp8=self.use_fp8,
        )
        return output


class FusedIndexDrivenKernel:
    """
    MoEBlaze 路线：index-driven gather → fused SwiGLU → scatter。
    不分配 routed_tokens 大 buffer，按索引 on-the-fly 读取。
    """

    def execute(self, tokens, tokens_per_expert):
        # tokens 仍在原始布局，通过 expert_token_indices 索引访问
        return fused_moe_swiglu(
            hidden_states=tokens,
            expert_indices=self.route_meta.expert_token_indices,
            expert_offsets=self.route_meta.expert_token_offsets,
            w_gate=self.w_gate, w_up=self.w_up, w_down=self.w_down,
            use_checkpoint=True,  # smart checkpoint for SwiGLU intermediate
        )


class BlockSparseKernel:
    """
    MegaBlocks 路线：BCSR 格式 block-sparse GEMM。
    Dropless — 所有 token 都被处理，无 capacity 限制。
    """

    def execute(self, tokens, tokens_per_expert):
        sparse_layout = build_bcsr_layout(tokens_per_expert, self.block_size)
        return block_sparse_moe_forward(
            tokens, sparse_layout,
            self.w_gate, self.w_up, self.w_down,
        )
```

### 4.3 小 Expert 合并执行

当某些 expert 分到的 token 数极少时（< `small_expert_threshold`），单独 launch kernel 的开销占比过大。借鉴 Megatron-Core 的做法：

```python
def merge_small_experts(tokens_per_expert, threshold=32):
    """将 token 数 < threshold 的 expert 合并为一个 grouped launch"""
    small_mask = tokens_per_expert < threshold
    if small_mask.sum() > 1:
        # 合并小 expert 的 token 到一个 batch，单次 GEMM
        merged_indices = torch.where(small_mask)[0]
        return merged_indices
    return None
```

---

## 5. Combine Engine（合并输出）

### 5.1 加权合并

```python
def weighted_merge(expert_outputs, routing_probs, route_meta):
    """
    Top-K 加权合并：output = Σ(prob_k * expert_output_k)
    MoEBlaze 风格：通过 token_index_map 直接 scatter-add，
    不构建完整 [batch, seq, hidden] 中间张量。
    """
    output = torch.zeros_like(route_meta.original_hidden)
    for k in range(route_meta.top_k):
        indices = route_meta.token_expert_indices[:, k]
        weights = routing_probs[:, k].unsqueeze(-1)
        expert_out = gather_expert_output(expert_outputs, indices, route_meta)
        output.scatter_add_(0, route_meta.token_positions, weights * expert_out)
    return output
```

### 5.2 反向传播路由

Combine Engine 同时负责反向传播时的梯度路由：
- 对 expert output 的梯度按相同 routing map 反向 dispatch
- 对 routing probs 的梯度用于更新 router（aux loss 已在 PrimusTopKRouter 中处理）

---

## 6. 内存管理

### 6.1 路由去物化（MoEBlaze 路线）

传统 MoE 实现在 dispatch 阶段分配 `[total_tokens * top_k, hidden_dim]` 的大 buffer。本设计采用 MoEBlaze 的 index-driven 方案：

```text
传统方案:
  routed_tokens = hidden[routing_map]  # 巨大中间张量

本设计:
  只保留 4 个轻量索引:
  ├── expert_token_indices: [num_experts, max_tokens_per_expert]
  ├── expert_token_offsets: [num_experts + 1]
  ├── token_expert_indices: [total_tokens, top_k]
  └── token_index_map: [total_tokens * top_k]

  执行时按索引 on-the-fly gather → compute → scatter
```

内存节省估算（DeepSeek-V3 规模，256 experts, top-8, hidden=7168）：
- 传统 buffer: `T × 8 × 7168 × 2B ≈ 112KB/token`
- 索引方案: `T × 8 × 4B + 257 × 4B ≈ 32B/token` (索引) + 按需 gather

### 6.2 选择性重算（Smart Checkpoint）

```text
张量类型                     策略          原因
─────────────────────────────────────────────────────────
SwiGLU 中间态 (gate * up)   重算          计算便宜，内存大
Attention up_proj 输出       重算          同上
LayerNorm 输出              重算          极便宜
Expert GEMM 输出            保留          计算昂贵
A2A dispatch 结果           保留          通信昂贵，不可重算
Router logits               保留          反向需要精确值
```

### 6.3 内存预算器

```python
class MemoryBudgeter:
    """每 step 检查 HBM 水位，触发降级策略"""

    def check_and_react(self, stats: MoERuntimeStats, policy: RuntimePolicy):
        if stats.hbm_used_ratio > 0.95:
            # 紧急：缩小 micro-batch
            return MemoryAction.REDUCE_MICROBATCH
        elif stats.hbm_used_ratio > 0.90:
            # 高压：提高重算 + 降低 chunk 并发
            policy.memory_cfg.recompute_level = "aggressive"
            policy.overlap_cfg.enable_3stage_pipeline = False
            return MemoryAction.DEGRADE_OVERLAP
        elif stats.hbm_used_ratio > 0.80:
            # 中压：开启路由去物化
            policy.memory_cfg.routing_dematerialize = True
            return MemoryAction.DEMATERIALIZE
        return MemoryAction.NONE
```

---

## 7. Overlap 设计（通信-计算重叠）

### 7.1 三段流水（Comet 启发）

```text
时间 ──────────────────────────────────────────────────────→

Stream 0 (Dispatch):  |==A2A(c0)==|==A2A(c1)==|==A2A(c2)==|
Stream 1 (Compute):              |==Expert(c0)==|==Expert(c1)==|==Expert(c2)==|
Stream 2 (Combine):                            |==A2A_G(c0)==|==A2A_G(c1)==|==A2A_G(c2)==|

c0, c1, c2 = token chunks
```

### 7.2 AMD ROCm 特化

MI300X 上的 overlap 实现与 NVIDIA 有关键差异：

| 维度 | NVIDIA (Comet 原始) | AMD ROCm (本设计) |
|------|--------------------|--------------------|
| 节点内互联 | NVLink/NVSwitch | XGMI mesh (896 GB/s) |
| Warp specialization | CUDA warp | HIP wavefront (64 wide) — 需验证收益 |
| 通信库 | NCCL | RCCL |
| Kernel overlap | CUDA stream + event | HIP stream + event (相同模型) |
| 硬件 SM 分区 | MPS / stream priority | CU masking (turbo_deepep_num_cu) |

实现策略：
- Phase 1: 用 HIP stream + event 实现 chunk 流水（稳定可控）
- Phase 2: 探索 wavefront specialization（research_direction_C 路线）
- 利用 Primus-Turbo 已有的 `turbo_deepep_use_comm_stream` 和 `turbo_deepep_num_cu` 配置

### 7.3 反伪 Overlap 检测

```python
class OverlapMonitor:
    """检测 overlap 是否真正有效，避免 RCCL 和 compute 抢 CU"""

    def evaluate(self, dispatch_time, compute_time, combined_time):
        ideal_time = max(dispatch_time, compute_time)
        actual_overhead = combined_time - ideal_time

        overlap_efficiency = 1.0 - (actual_overhead / ideal_time)

        if overlap_efficiency < 0.05:
            # overlap 收益 < 5%，建议关闭
            return OverlapVerdict.DISABLE
        elif overlap_efficiency < 0.30:
            # 收益不理想，尝试调整 chunk 大小或 CU 分配
            return OverlapVerdict.TUNE
        else:
            return OverlapVerdict.KEEP
```

---

## 8. 与 Primus-dev 的集成方案

### 8.1 集成点

```text
Primus-dev 代码位置                              集成方式
──────────────────────────────────────────────────────────────────────
primus/backends/megatron/patches/                 register_patch 注入 Runtime Dispatcher
  moe_dispatcher_patches.py                       替换 PrimusTurboDeepEPTokenDispatcher
  topk_router_patches.py                          保持 PrimusTopKRouter，增加 stats 采集

primus/backends/megatron/core/extensions/         新增 moe_compute_runtime.py
  primus_turbo.py                                 复用 grouped GEMM / FP8 接口

primus/backends/torchtitan/models/moe/            修改 _run_experts_grouped_mm
  moe.py                                          接入 ExpertComputeEngine

primus/backends/megatron/core/transformer/moe/    新增 runtime_planner.py, memory_budgeter.py
```

### 8.2 Patch 注册

```python
# primus/backends/megatron/patches/moe_runtime_patches.py

from primus.core.patch_registry import register_patch

def apply_moe_compute_runtime():
    """当 enable_moe_compute_runtime=True 时，替换 dispatcher"""
    args = get_args()
    if not getattr(args, 'enable_moe_compute_runtime', False):
        return

    from primus.backends.megatron.core.extensions.moe_compute_runtime import (
        MoEComputeRuntimeDispatcher,
    )

    # 替换 Megatron flex dispatcher
    import megatron.core.transformer.moe.token_dispatcher as td_module
    td_module.MoEFlexTokenDispatcher = MoEComputeRuntimeDispatcher

register_patch("moe_compute_runtime", apply_moe_compute_runtime)
```

### 8.3 配置接口

```python
# 新增训练配置项 (与 Primus args 体系对齐)

@dataclass
class MoERuntimeConfig:
    enable_moe_compute_runtime: bool = False

    # Planner
    planner_eval_interval: int = 100       # 每 N step 重评估策略
    enable_topology_aware_a2a: bool = True
    default_a2a_algorithm: str = "auto"    # "linear" | "hierarchical_2d" | "auto"

    # Compute
    default_kernel: str = "grouped_gemm"   # "grouped_gemm" | "fused_index_driven" | "block_sparse"
    enable_fp8: bool = False
    small_expert_threshold: int = 32       # token 数低于此值的 expert 合并执行

    # Memory
    enable_routing_dematerialize: bool = False
    recompute_level: str = "selective"      # "none" | "selective" | "aggressive"
    hbm_high_watermark: float = 0.90

    # Overlap
    enable_chunk_pipeline: bool = True
    chunk_degree: int = 0                   # 0 = auto
    min_chunk_tokens: int = 256
    detect_sm_contention: bool = True
```

---

## 9. 分阶段落地

### Phase 1（2-4 周）：基础 Runtime 框架 + Grouped GEMM 路线

**目标：** 跑通 Runtime 骨架，性能不低于现有 DeepEP dispatcher

| 任务 | 产出 | 验收 |
|------|------|------|
| 实现 `MoEComputeRuntimeDispatcher` (flex API) | 新文件 `moe_compute_runtime.py` | Megatron MoE Layer forward/backward 正确 |
| 实现 `RuntimePlanner` (静态策略，无自适应) | 新文件 `runtime_planner.py` | 配置驱动，不引入运行时开销 |
| 接入 Primus-Turbo grouped GEMM 作为默认 kernel | 复用 `primus_turbo.grouped_gemm` | FP16/BF16 + FP8 均通过 |
| Patch 注册 + 单元测试 | 扩展 `test_token_dispatcher.py` | EP=1,2,4,8 + TP=1,2 矩阵 |
| 基线性能对比 | benchmark 脚本 | tokens/s 不低于 DeepEP baseline |

### Phase 2（4-8 周）：内存优化 + 自适应调度

| 任务 | 产出 | 验收 |
|------|------|------|
| 路由去物化 (index-driven dispatch) | `FusedIndexDrivenKernel` | 激活内存下降 30%+ |
| 选择性重算 (SwiGLU smart checkpoint) | 集成到 `ExpertComputeEngine` | 峰值内存下降，throughput 回归 < 3% |
| 自适应策略切换 (Tutel 风格 bucket dict) | `RuntimePlanner` 动态路径 | 负载变化时自动切换内核 |
| 内存预算器 | `MemoryBudgeter` | HBM > 90% 时自动降级 |
| Chunk 流水 overlap | 多 stream dispatch-compute-combine | A2A 占比下降 30%+ |

### Phase 3（8-12 周）：AMD 深度优化 + 生产加固

| 任务 | 产出 | 验收 |
|------|------|------|
| XGMI-aware A2A (节点内 P2P vs 跨节点 RDMA) | 拓扑检测 + 双路径 A2A | 节点内 A2A 延迟下降 |
| Block-sparse kernel (MegaBlocks 路线 ROCm 移植) | `BlockSparseKernel` | 极端负载不均场景有效 |
| TorchTitan 后端集成 | 修改 `_run_experts_grouped_mm` | DeepSeek-V3 模型跑通 |
| Overlap monitor + 自动 tune | `OverlapMonitor` | 自动检测并关闭无效 overlap |
| 生产容错 (expert reroute on failure) | MegaScale 风格 | 单 GPU 故障不中断训练 |

### Phase 4（12 周+）：研究方向探索

| 方向 | 对应论文 | 风险 |
|------|---------|------|
| Wavefront specialization (Comet on AMD) | research_direction_C | ROCm wavefront 调度稳定性 |
| MoE-native IR + torch.compile | research_direction_B (RFGraph) | torch.compile + RCCL 集合通信兼容 |
| Parallel Folding (Attn/MoE 异构并行) | NVIDIA MoE Parallel Folding | 需要修改 Megatron PP 调度 |

---

## 10. KPI 指标

| 类别 | 指标 | Phase 1 目标 | Phase 2 目标 | Phase 3 目标 |
|------|------|-------------|-------------|-------------|
| **正确性** | forward/backward 数值一致 | bit-exact vs baseline | bit-exact | bit-exact |
| **吞吐** | tokens/s (E2E training) | ≥ baseline | +10-15% | +20-30% |
| **内存** | 峰值 HBM 占用 | ≤ baseline | -30% 激活 | -40% 激活 |
| **通信** | A2A 时间占比 | ≤ baseline | -30% | -50% |
| **Overlap** | 通信-计算重叠率 | N/A | >50% | >70% |
| **负载** | load_skew (max/avg) | 监控 | 自适应响应 | 自动优化 |
| **稳定性** | 训练 loss 曲线 | 与 baseline 一致 | 一致 | 一致 |

---

## 11. 文件结构（建议）

```text
primus/backends/megatron/core/extensions/
├── moe_compute_runtime.py          # MoEComputeRuntimeDispatcher (flex API)
├── runtime_planner.py              # MoERuntimePlanner + configs
├── expert_compute_engine.py        # ExpertComputeEngine + 三条内核
├── memory_budgeter.py              # MemoryBudgeter
├── overlap_monitor.py              # OverlapMonitor
└── moe_runtime_utils.py            # 轻量索引构建、拓扑检测等

primus/backends/megatron/patches/
└── moe_runtime_patches.py          # Patch 注册

tests/unit_tests/megatron/transformer/moe/
├── test_token_dispatcher.py        # 扩展已有测试
├── test_moe_compute_runtime.py     # Runtime 端到端测试
└── test_runtime_planner.py         # Planner 单元测试
```

---

## 12. 论文技术到代码的映射总结

| 大厂论文 | 核心技术 | Runtime 组件 | 代码位置 |
|---------|---------|-------------|---------|
| **Tutel** (Microsoft) | 自适应并行 + bucket dict | `RuntimePlanner._select_dispatch` | `runtime_planner.py` |
| **Tutel** | 2DH A2A + pipeline | `DispatchConfig.a2a_algorithm` | `moe_compute_runtime.py` |
| **MegaScale-MoE** (ByteDance) | 拓扑感知放置 + 分层 EP | `ClusterTopology` + A2A 路径选择 | `moe_runtime_utils.py` |
| **Comet** (ByteDance) | Tile-level overlap | `_chunked_dispatch` 三段流水 | `moe_compute_runtime.py` |
| **MoEBlaze** (Meta) | Index-driven routing | `build_compact_route_indices` | `moe_runtime_utils.py` |
| **MoEBlaze** | Fused SwiGLU + smart ckpt | `FusedIndexDrivenKernel` | `expert_compute_engine.py` |
| **Megatron-Core** (NVIDIA) | Grouped GEMM + FP8 | `GroupedGEMMKernel` | `expert_compute_engine.py` |
| **Megatron-Core** | DeepEP flex API | Dispatcher 三阶段协议 | `moe_compute_runtime.py` |
| **Parallel Folding** (NVIDIA) | Attn/MoE 异构并行 | Phase 4 探索 | — |
| **MegaBlocks** (Databricks) | Block-sparse dropless | `BlockSparseKernel` | `expert_compute_engine.py` |

---

## 13. 风险与缓解

| 风险 | 影响 | 缓解 |
|------|------|------|
| ROCm grouped GEMM 性能不及 cuBLAS | Phase 1 吞吐不达标 | 先用 hipBLASLt，同时跟踪 Primus-Turbo 内核进展 |
| RCCL A2A 与 compute stream 抢 CU | Overlap 失效 | `OverlapMonitor` 自动检测 + CU masking (`turbo_deepep_num_cu`) |
| Block-sparse 在 ROCm 上无成熟实现 | Phase 3 延期 | 先用 grouped GEMM + padding 兜底 |
| Megatron upstream MoE API 变更 | Patch 失效 | 锁定 submodule 版本 + CI 回归 |
| 自适应策略切换引入训练不稳定 | Loss spike | 保守切换（hysteresis）+ 回退机制 |
