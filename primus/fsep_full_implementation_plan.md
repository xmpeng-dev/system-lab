# 完整 FSEP 实现规划（对标 LAER-MoE 论文）

> **目标：** 实现论文 LAER-MoE（ASPLOS '26）描述的完整 FSEP（Fully Sharded Expert Parallel），  
> 包括 Expert 参数全分片、动态 Re-layout、Load Planner 三大组件  
> **参考代码：** `third_party/laer_moe/`  
> **论文：** `third_party/aiinfra/paper/LAER-MoE_*.pdf`  
> **基础：** 已完成的静态 FSEP（`feat/moe-fsep-static-sharding`）  
> **更新：** 2026-03-09

---

## 现有实现 vs 论文 FSEP 的差距

```
已实现（静态 FSEP）：
  ✅ Expert 参数沿 F 维分片到 S 块 GPU（Mode B: S==EP）
  ✅ ReduceScatter 替换 All-Reduce 聚合 Expert 输出
  ✅ Token 仍通过传统 EP A2A Dispatch 路由到 owner GPU
  ✅ 固定分片度 S（所有 Expert 统一）

论文完整 FSEP 额外要求：
  ❌ Expert 无固定 owner：每块 GPU 持有所有 Expert 的 1/S 参数
  ❌ A2A Dispatch 语义变化：token 广播给参与计算该 Expert 的所有 S 块 GPU
  ❌ 动态 Re-layout：热点 Expert 增加 S，冷点 Expert 减少 S
  ❌ Load Planner：周期性检测负载，决策最优 Expert 分片方案
  ❌ 异步参数搬迁：在反向传播期间完成 Re-layout，零停顿
```

---

## 论文代码关键模块分析

### `galvatron/core/runtime/moe/smart_routing.py`
- **`MoEAlltoAllSmartTokenDispatcher`**：核心 dispatcher，支持 Expert 的动态"virtual location"
- **`get_smart_routing()`**：统计全局 token-per-expert 分布，调用 `smart_routing_map_gpu` 重映射路由
- **`global_expert_indices`**：每个虚拟 Expert 实际在哪块 GPU → 支持 Expert Re-layout

### `galvatron/core/runtime/moe/prefetch/solver.py`
- **`MoEOptimizer`**：LP + Greedy 启发式求解最优 Expert 放置
- 目标：`minimize max_gpu(T_comp + T_comm)`
- 支持 Expert Replication（一个 Expert 放在多块 GPU）

### `galvatron/core/runtime/moe/mlp.py`
- **`GroupedMLP`**：参数按 ETP 分片（`tp_size` 来自 `tp_of_ep_group`）
- 这是论文 FSEP 的参数分片：`weight1[H, F/tp_size]`，tp_size 就是 S

### `csrc/greedy_balancer.cpp` + `moe_all_to_all_kernels.cu`
- Load Planner 的 C++ 贪心算法
- CUDA kernel：高效计算新路由映射（`smart_routing_map_gpu`）

---

## 完整 FSEP 的数据流对比

```
静态 FSEP（当前实现）：

  token_for_E37 → A2A → owner GPU (GPU1) only
                       ↓
                  GPU1 做 partial GEMM（weight [H, F/S]）
                  GPU0,GPU2,...GPU7 同时也做 partial GEMM
                       ↓
                  ReduceScatter(EP group) → 各 GPU [T/S, H]

  问题：A2A 仍然把 token 发送给"owner"GPU，
       其他 GPU 只是被动参与分片计算，
       owner 概念仍然存在（EP 路由决定了 token 的目的地）

论文完整 FSEP：

  每块 GPU 持有所有 N_E 个 Expert 的 1/S 参数
  （没有 owner 概念）
       ↓
  token_for_E37 → A2A → 所有 S 块 GPU（广播式）
                       ↓
                  各 GPU 做自己持有的 Expert 37 分片计算
                       ↓
                  ReduceScatter → 各 GPU [T_total/S, H]
  
  关键差异：A2A 语义从"点对点"变成了"广播给S块GPU"
```

---

## 完整实现方案（分四个阶段）

---

### Phase 1：FSEP A2A Dispatch 语义重构（核心）

**当前问题：** A2A 还是把 token 路由到 "Expert owner" GPU，不是真正的 FSEP  
**目标：** token 被广播给参与计算该 Expert 的所有 S 块 GPU

#### 1.1 新的 Process Group 体系

```
当前 process groups（传统 EP）：
  ep_group   = {GPU0, GPU1, GPU2, GPU3, GPU4, GPU5, GPU6, GPU7}  (EP=8)
  
论文 FSEP 需要的新 groups：
  fsep_group = ep_group（S=EP=8 时，整个 EP group 都参与每个 Expert）
  
  对于 S < EP（部分分片）：
  fsep_sub_group_0 = {GPU0, GPU1}   # 参与 Expert 0,2,4,... 的计算
  fsep_sub_group_1 = {GPU2, GPU3}   # 参与 Expert 1,3,5,... 的计算
  ...
```

#### 1.2 真正的 FSEP Token Dispatch

**参考：** `laer_moe/galvatron/core/runtime/moe/smart_routing.py::MoEAlltoAllSmartTokenDispatcher`

```python
# 论文做法：Expert 的 "virtual location" 和 "global_expert_indices"
# 每个 Expert 在哪些 GPU 上有分片 → global_expert_locations

# 修改 A2A 路由逻辑：
# 原来：token_t 要去 Expert e → 发往 ep_rank = e // num_local_experts
# 现在：token_t 要去 Expert e → 发往 fsep_group 中所有持有 E_e 分片的 GPU
```

**实现策略（与论文对齐）：**

```python
class FSEPAlltoAllTokenDispatcher:
    """
    替换传统 EP A2A，实现 FSEP 的"广播式" dispatch：
    
    token → gate → routing_map [T, N_E]
                ↓
    new_routing_map = smart_routing_map(routing_map, global_expert_locations)
    → 每个 token 被重映射到持有目标 Expert 分片的 GPU 集合
                ↓
    A2A(new_routing_map)  ← 比原来发送更多 token（广播）
                ↓
    每块 GPU 收到属于自己分片的 tokens
                ↓
    partial GEMM → ReduceScatter → [T/S, H]
    """
    
    def __init__(self, ..., global_expert_indices, global_expert_locations):
        # global_expert_indices: [N_E, S] → 每个 Expert 由哪 S 块 GPU 计算
        # global_expert_locations: [N_E × S] → 扁平化的 GPU 列表
        self.global_expert_indices = global_expert_indices
        self.global_expert_locations = global_expert_locations
```

#### 1.3 接入点（Primus 代码）

```
需要修改的文件：
  primus/backends/megatron/core/transformer/moe/fsep_experts.py
  → 新增 FSEPTokenDispatcher（替换 PrimusTurboDeepEPTokenDispatcher 的 dispatch 逻辑）
  
  primus/backends/megatron/core/parallel_state.py（或 Megatron 的 parallel_state）
  → 新增 fsep_sub_group 初始化（当 S < EP 时）
```

---

### Phase 2：Load Planner（负载检测 + 分片决策）

**参考：** `laer_moe/galvatron/core/runtime/moe/prefetch/solver.py::MoEOptimizer`

#### 2.1 负载检测（已有基础）

```python
# 已在 router.py 中实现（moe_log_expert_load）
load = routing_map.float().sum(dim=0)   # [N_E]
dist.all_reduce(load, group=ep_group)   # 全局 token-per-expert

# 需要扩展：
# - 记录历史窗口（论文用 K 步滑动平均）
# - 检测 max/avg 比值是否超过阈值（触发 Re-layout）
```

#### 2.2 最优分片方案求解

**参考论文优化目标：**

```
minimize:  max_gpu_i ( T_comp[i] + T_comm[i] )

T_comp[i] = Σ_{e in GPU_i} T_e × (tokens_e / S_e)
T_comm[i] = data_volume_i × latency_coefficient

约束：
  Σ_i placement[e,i] == S_e  ∀e  (每个 Expert 恰好在 S_e 块 GPU 上)
  0 ≤ S_e ≤ EP              (分片度约束)
  Σ_e params[e]/S_e ≤ memory_per_GPU  (显存约束)
```

**实现策略：**

```python
class FSEPLoadPlanner:
    """
    周期性（每 K 步）检测 Expert 负载，
    调用贪心算法决策新的 Expert 分片方案。
    
    参考：laer_moe/csrc/greedy_balancer.cpp
    """
    
    def __init__(self, num_experts, ep_size, K=50, threshold=1.5):
        self.history = []          # 历史负载（EMA 窗口）
        self.K = K                 # 检测间隔（steps）
        self.threshold = threshold # max/avg 触发阈值
        self.step_count = 0
    
    def update(self, load: torch.Tensor):
        """每步更新负载历史"""
        self.history.append(load.detach().cpu())
        if len(self.history) > self.K:
            self.history.pop(0)
        self.step_count += 1
    
    def should_relayout(self) -> bool:
        """是否需要触发 Re-layout"""
        if self.step_count % self.K != 0:
            return False
        avg_load = torch.stack(self.history).mean(0)
        return (avg_load.max() / avg_load.mean()) > self.threshold
    
    def compute_new_placement(self, avg_load) -> dict:
        """
        贪心求解新的 Expert 分片方案。
        返回 {expert_id: new_sharding_degree}
        
        参考 greedy_balancer.cpp 的算法：
        1. 按负载从高到低排序 Expert
        2. 贪心分配 GPU：每次选负载最轻的 GPU 接收下一个 Expert 分片
        3. 输出 global_expert_indices[N_E, S_new]
        """
        # TODO: port from laer_moe/csrc/greedy_balancer.cpp
        ...
```

---

### Phase 3：Expert Re-layout Executor（异步参数搬迁）

这是论文里最核心但最复杂的组件。

#### 3.1 Re-layout 触发时机

```
Step T（正常训练）：
  Load Planner 决策新布局：E2 需要从 S=2 扩到 S=4
          ↓
  在 Step T 的反向传播期间（利用空闲时间）：
    GPU0 → GPU2, GPU3 发送 E2 的新参数分片（All-to-All 异步搬迁）
          ↓
Step T+1：
  使用新布局，global_expert_indices 更新，
  A2A dispatch 路由到新的 GPU 集合
```

#### 3.2 参数搬迁机制

**参考：** `laer_moe/galvatron/core/runtime/hybrid_parallel_model.py` 中的 FSDP 参数搬迁

```python
class FSEPRelayoutExecutor:
    """
    在反向传播期间异步执行 Expert 参数的物理搬迁。
    
    关键：使用 double buffer + 异步 All-to-All
    """
    
    def schedule_relayout(self, old_placement, new_placement):
        """
        在当前 step 的反向传播期间，异步搬迁参数。
        
        实现步骤：
        1. 计算哪些参数分片需要从哪块 GPU 搬到哪块 GPU
        2. 在 backward stream 上发起异步 send/recv（NCCL point-to-point）
        3. 注册 hook：在 backward 结束后，同步并切换到新布局
        """
        with torch.cuda.stream(self.relayout_stream):
            for src_rank, dst_rank, param_shard in self._compute_transfers(old_placement, new_placement):
                if dist.get_rank() == src_rank:
                    dist.isend(param_shard, dst=dst_rank, tag=expert_id)
                elif dist.get_rank() == dst_rank:
                    dist.irecv(self.new_param_buffer, src=src_rank, tag=expert_id)
    
    def finalize_relayout(self):
        """在 step 结束时，同步并更新 global_expert_indices"""
        self.relayout_stream.synchronize()
        # 原子地切换 global_expert_indices
        self.global_expert_indices = self.new_global_expert_indices
```

#### 3.3 与 Primus 反向传播的集成

```
Primus 的反向传播时间线（PrimusPipelineSchedule）：

  comm_stream: ── combine_bwd ── dispatch_fwd ──
  comp_stream: ── attn_fwd ── mlp_bwd ── attn_bwd ──
                                      ↑
                             这里插入异步 Re-layout

接入点：
  primus/backends/megatron/core/models/common/model_chunk_schedule_plan.py
  → execute_overlapped_1f1b() 中的 b_layer.mlp.backward(b_grad) 之后
  → 注册 post_backward_hook → 触发 FSEPRelayoutExecutor.schedule_relayout()
```

---

### Phase 4：与 DeepEP 的融合（最优通信路径）

论文 FSEP 的 A2A 通信量比传统 EP 更大（广播式），需要与 DeepEP 结合降低开销。

#### 4.1 DeepEP 的 FSEP 扩展

```
现有 DeepEP dispatch（传统 EP）：
  hidden_states → _pre_dispatch → _exec_dispatch（A2A 点对点）

FSEP 的 dispatch 需要：
  hidden_states → new_routing_map（基于 global_expert_locations）
               → _exec_dispatch（A2A，每 token 发给 S 块 GPU）

关键：DeepEP 的 _exec_dispatch 已经支持 token 被发送给多个 GPU
     （这正是 `moe_router_topk > 1` 时的行为）
     FSEP 只需要修改 routing_map 的语义，使每个 token 路由到
     "S 个持有该 Expert 分片的 GPU" 而非 "1 个 owner GPU"
```

---

## 代码架构规划

```
Primus 中完整 FSEP 的新增文件：

primus/backends/megatron/core/transformer/moe/
├── fsep_experts.py          ← 已有（静态 FSEP GroupedMLP）
├── fsep_token_dispatcher.py ← 【新增】Phase 1：FSEP A2A dispatch
├── load_planner.py          ← 【新增】Phase 2：Load Planner
├── relayout_executor.py     ← 【新增】Phase 3：异步 Re-layout
└── fsep_parallel_state.py   ← 【新增】FSEP process group 初始化

primus/backends/megatron/patches/moe_patches/
├── fsep_patches.py          ← 已有（静态 FSEP patch）
└── fsep_full_patches.py     ← 【新增】完整 FSEP 的 patch 注册
```

---

## 与论文代码的对应关系

| 论文代码 | Primus 对应 |
|---------|-----------|
| `smart_routing.py::MoEAlltoAllSmartTokenDispatcher` | `fsep_token_dispatcher.py::FSEPAlltoAllTokenDispatcher` |
| `smart_routing.py::get_smart_routing()` | `load_planner.py::FSEPLoadPlanner.update()` |
| `prefetch/solver.py::MoEOptimizer::greedy_load_balancing_heuristic()` | `load_planner.py::FSEPLoadPlanner.compute_new_placement()` |
| `csrc/greedy_balancer.cpp` | Port 到 Python 或直接编译 C 扩展 |
| `csrc/moe_all_to_all_kernels.cu::smart_routing_map_gpu()` | `fsep_token_dispatcher.py` 中调用（或直接使用论文 CUDA kernel）|
| FSDP parameter migration in `hybrid_parallel_model.py` | `relayout_executor.py::FSEPRelayoutExecutor` |
| `global_expert_indices` | `FSEPParallelState.global_expert_indices` |
| `global_expert_locations` | `FSEPParallelState.global_expert_locations` |

---

## 实施路线图

```
Phase 1（4~6 周）：FSEP A2A Dispatch 语义重构
  Week 1-2：FSEPParallelState（FSEP sub-group 初始化）
  Week 2-3：FSEPAlltoAllTokenDispatcher（新的 dispatch 逻辑）
  Week 3-4：Port smart_routing_map_gpu CUDA kernel
  Week 4-6：端到端测试（验证数值等价性 + 负载均衡效果）

Phase 2（2~3 周）：Load Planner
  Week 1：FSEPLoadPlanner（历史统计 + 触发判断）
  Week 2：Port greedy_balancer（C++ → Python 或保留 C++）
  Week 3：集成测试（检测 → 决策 → 验证新 placement）

Phase 3（3~4 周）：Expert Re-layout Executor
  Week 1-2：FSEPRelayoutExecutor（异步参数搬迁框架）
  Week 2-3：与 Primus pipeline 反向传播的集成（hook 接入）
  Week 3-4：内存管理（double buffer）+ 正确性验证

Phase 4（2~3 周）：与 DeepEP 融合 + 端到端性能
  Week 1-2：DeepEP dispatch 扩展（FSEP routing map 支持）
  Week 2-3：端到端 benchmark（对比论文 1.69x 目标）
```

**总工期：约 11~16 周（同时进行 Phase 1 + 2 可缩短）**

---

## 关键风险与缓解

| 风险 | 等级 | 缓解 |
|------|------|------|
| **A2A 通信量增大**（广播式 dispatch）| ⚠️ 高 | FSEP 论文中的 Greedy Balancer 会优化 Expert 放置，限制总通信量；与 DeepEP Phase 4 配合 |
| **论文代码基于 Hetu-Galvatron，框架适配工作量大** | ⚠️ 高 | 只移植核心算法（solver.py + greedy_balancer.cpp），不依赖 Galvatron 的训练框架 |
| **Re-layout 期间参数版本不一致** | ⚠️ 中 | 使用 double buffer + 原子切换 global_expert_indices |
| **greedy_balancer.cpp 的 C++ 依赖** | ⚠️ 低 | 先用 Python 实现简化版，后续优化时编译 C 扩展 |
| **与 ZeroBubble / Primus Pipeline 的兼容性** | ⚠️ 中 | Phase 3 接入反向传播 hook 需要与 ZB schedule 协调 |

---

## 与静态 FSEP 的演进关系

```
静态 FSEP（已实现，feat/moe-fsep-static-sharding）：
  ✅ 参数分片方向（F 维切分）完全正确
  ✅ ReduceScatter 通信原语正确
  ✅ Token dispatch 仍然点对点（EP 语义保留）
  ✅ 固定分片度（S 不随负载变化）

完整 FSEP（本规划目标）：
  → Phase 1：A2A 语义改为"广播给 S 块 GPU"（去除 owner 概念）
  → Phase 2：Load Planner 动态决定每个 Expert 的 S
  → Phase 3：异步 Re-layout 实际搬迁参数
  → Phase 4：性能优化（与 DeepEP 融合）

静态 FSEP 是完整 FSEP 的"固定 S + owner 保留"的简化版本，
两者共享同一套参数分片和 ReduceScatter 基础设施。
完整 FSEP 是在此之上的演进，不需要推翻重写。
```

---

*规划整理于 2026-03-09 | 基于 LAER-MoE 论文代码分析（third_party/laer_moe）+ Primus FSEP 现有实现*
