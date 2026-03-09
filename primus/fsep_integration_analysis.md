# FSEP 在 Primus 的集成可行性深度分析

> **定位：** 以 LAER-MoE 的 FSEP（Fully Sharded Expert Parallel）方案为参照，  
> 逐模块分析在 Primus Megatron backend 的集成路径、代码改动量、收益与风险  
> **更新：** 2026-03-09

---

## 第一部分：FSEP 核心机制回顾

### 1.1 FSEP 与传统 EP 的本质区别

```
传统 EP（Primus 现状）                   FSEP（目标）
────────────────────────────────────────────────────────────────
参数布局：                                参数布局：
  GPU0: [E0完整: H×F] [E1完整: H×F]        GPU0: [E0_s0: H×F/S] [E1_s0: H×F/S] ...
  GPU1: [E2完整: H×F] [E3完整: H×F]        GPU1: [E0_s1: H×F/S] [E1_s1: H×F/S] ...
  GPU2: [E4完整: H×F] [E5完整: H×F]        GPU2: [E0_s2: H×F/S] [E1_s2: H×F/S] ...
  GPU3: [E6完整: H×F] [E7完整: H×F]        GPU3: [E0_s3: H×F/S] [E1_s3: H×F/S] ...

E0 收到 400 token（过载 4x）：            E0 收到 400 token：
  GPU0: GEMM(400 token, H×F) ← 瓶颈       GPU0: GEMM(400 token, H×F/S)  ← 均衡
  GPU1/2/3: 空等 GPU0                      GPU1: GEMM(400 token, H×F/S)
                                           GPU2: GEMM(400 token, H×F/S)
                                           GPU3: GEMM(400 token, H×F/S)
                                           → ReduceScatter → 完整输出

前向通信量：                              前向通信量：
  2 × T × H                               3 × T × H（+50%，ReduceScatter 节点内）
计算效率（r=4 时）：
  GPU 利用率 = 1/r = 25%                  GPU 利用率 ≈ 95%+
```

### 1.2 FSEP 前向/反向完整通信开销

```
                前向            反向             总计
────────────────────────────────────────────────────
传统 EP：       2T×H            2T×H             4T×H

FSEP（S=4）：
  A2A Dispatch  T×H
  ReduceScatter T×H（节点内，高带宽）
  A2A Gather    T×H
  ────────────────
  前向小计       3T×H

  AllGather(RS的反向)  T×H（节点内）
  AllReduce(d_tokens)  T×H（节点内）
  A2A 反向 ×2         2T×H
  ────────────────────
  反向小计       5T×H（含2T×H节点内）
  ────────────────────────────
  总计           5T×H  vs  4T×H（+25%）
  其中节点内通信  2T×H（MI300X XGMI: ~896 GB/s，极快）
  跨节点通信     3T×H  vs  4T×H（实际跨节点通信反而更少！）
```

**关键认知：FSEP 的额外通信（ReduceScatter/AllGather）全部发生在节点内（XGMI/NVLink 高带宽），不走跨节点低带宽链路。当不均衡比 r ≥ 2 时，FSEP 净收益必然为正。**

---

## 第二部分：Primus 现有代码结构分析

### 2.1 Expert 计算的现有路径

Primus Megatron backend 中，Expert GEMM 有三条路径：

```
路径 A（主力路径，turbo 启用时）：
  PrimusTurboGroupedMLP（primus_turbo.py）
  └─ pt.ops.grouped_gemm（primus_turbo fused GroupedGEMM）
  └─ pt.ops.grouped_gemm_fp8（FP8 GroupedGEMM）

路径 B（deprecated 路径，use_deprecated_20241209_moe_layer=True）：
  DeprecatedGroupedMLP（deprecated_20251209/experts.py）
  └─ gg.ops.gmm（megatron grouped_gemm_util）

路径 C（TE 路径）：
  DeprecatedTEGroupedMLP（deprecated_20251209/experts.py）
  └─ TE GroupedLinear
```

**FSEP 目标路径：路径 A（PrimusTurboGroupedMLP），因为 pt.ops.grouped_gemm 支持 tokens_per_expert 变长输入，天然适配分片后的不规则 token 分布。**

### 2.2 现有参数分片维度

`DeprecatedGroupedMLP` 中的参数分片（Expert TP 已支持）：

```python
# 现有代码（deprecated_20251209/experts.py:122-134）
tp_size = parallel_state.get_expert_tensor_parallel_world_size()
tp_rank = parallel_state.get_expert_tensor_parallel_rank()

fc1_output_size_per_partition = divide(fc1_output_size, tp_size)   # 沿 F 维分片
fc2_input_size_per_partition  = divide(fc2_input_size,  tp_size)   # 沿 F 维分片

self.weight1 = Parameter(torch.empty(H, fc1_output_size_per_partition))  # [H, F/tp]
self.weight2 = Parameter(torch.empty(fc2_input_size_per_partition, H))   # [F/tp, H]
```

**关键发现：Primus 已有沿 FFN 中间维度（F 维）的 Expert TP 分片基础设施！**

FSEP 要做的参数分片方向与 Expert TP **完全一致**：
- `weight1`: `[H, F]` → 沿列切分 → 每卡 `[H, F/S]`（等同于 `expert_tensor_parallel_size=S`）  
- `weight2`: `[F, H]` → 沿行切分 → 每卡 `[F/S, H]`

**FSEP ≈ 在负载均衡语义下使用 Expert TP，关键区别是通信模式不同（TP 用 All-Reduce，FSEP 用 ReduceScatter）。**

### 2.3 现有 Expert TP 的通信模式 vs FSEP

```
现有 Expert TP（megatron 原生）：
  weight1: [H, F/tp]  →  fc1 output: [T, F/tp]
  weight2: [F/tp, H]  →  partial output: [T, H]（部分和）
  → All-Reduce(partial output) → 完整 output: [T, H]

FSEP（目标通信模式）：
  weight1: [H, F/S]   →  fc1 output: [T, F/S]
  weight2: [F/S, H]   →  partial output: [T, H]（部分和）
  → ReduceScatter(partial output) → 各卡持有 output: [T/S, H]
  → 后续 A2A Gather 将 [T/S, H] 片段送回原始 GPU
```

**现有 Expert TP 用 All-Reduce，每卡最终得到完整的 [T, H]；FSEP 用 ReduceScatter，每卡只持有 [T/S, H] 的片段——这是关键的通信模式差异，也是 FSEP 省显存的原因。**

---

## 第三部分：集成方案设计

### 3.1 方案分级

```
方案 A：Load Monitoring + 路由偏置调整（1~2 周）
方案 B：静态 FSEP（固定分片度，2~6 周）
方案 C：动态 FSEP + Re-layout（完整 LAER-MoE，2~3 月）
```

本文重点分析方案 B（静态 FSEP）的完整集成路径，因为：
- 验证正确性后是方案 C 的前置
- 单独使用已能消除静态负载不均衡场景的热点

---

### 3.2 方案 B：静态 FSEP 详细集成设计

#### 3.2.1 新增 Process Group：fsep_group

FSEP 需要一个专属的进程组，作用域是参与同一个 Expert 分片计算的 GPU：

```
当前 EP 组（EP=8, 每卡 2 个 Expert）：
  EP_group: {GPU0, GPU1, GPU2, GPU3, GPU4, GPU5, GPU6, GPU7}

FSEP 分片度 S=4，每个 Expert 由 4 块 GPU 共同计算：
  fsep_group_E0: {GPU0, GPU1, GPU2, GPU3}  ← E0 的 ReduceScatter 组
  fsep_group_E1: {GPU4, GPU5, GPU6, GPU7}  ← E1 的 ReduceScatter 组

关系：fsep_group ⊆ EP_group，且 |fsep_group| = S（分片度）
```

**接入点：** `primus/backends/megatron/core/parallel_state.py`，参考已有 `get_expert_tensor_parallel_group()` 增加 `get_fsep_group()` / `get_fsep_rank()` / `get_fsep_world_size()`。

实际上，当 `S == expert_tensor_parallel_size` 时，`fsep_group` 就是 `expert_tensor_parallel_group`——**可以复用已有的 expert TP process group！**

#### 3.2.2 新增 FSEPGroupedMLP

```
文件：primus/backends/megatron/core/transformer/moe/fsep_experts.py

核心改动（对比 PrimusTurboGroupedMLP）：

① 参数分片（同 Expert TP，可直接复用）
   weight1: [H, F/S]   （沿 ffn_intermediate_dim 列切）
   weight2: [F/S, H]   （沿 ffn_intermediate_dim 行切）

② forward 改动：
   # 现有 GroupedMLP（All-Reduce 模式）：
   fc1_out   = grouped_gemm(tokens, w1, tpe)     # [T, F/S]
   act_out   = activation(fc1_out)               # [T, F/S]
   partial   = grouped_gemm(act_out, w2, tpe)    # [T, H]（部分和）
   output    = all_reduce(partial)               # [T, H]  ← Expert TP 现有

   # FSEP（ReduceScatter 模式）：
   fc1_out   = grouped_gemm(tokens, w1, tpe)     # [T, F/S]
   act_out   = activation(fc1_out)               # [T, F/S]
   partial   = grouped_gemm(act_out, w2, tpe)    # [T, H]（部分和）
   output    = reduce_scatter(partial, fsep_grp) # [T/S, H]  ← 改为 ReduceScatter
   # 注：output [T/S, H] 片段由后续 A2A Gather 收回

③ 反向改动（自动微分，但需确认通信原语正确）：
   ReduceScatter 的反向 = AllGather（PyTorch autograd 可自动处理）
   AllReduce(d_tokens) 的反向 = 不变（原来 Expert TP 路径已有）
```

#### 3.2.3 Token Dispatcher 改动

FSEP 的 token 分发结果形状变了：dispatch 后每卡持有全量 token（与现在相同），但 ReduceScatter 后 expert output 变成 `[T/S, H]` 片段，gather 阶段需要对齐：

```
现有 A2A Dispatch/Gather（DeprecatedMoEAlltoAllTokenDispatcher）：

  Dispatch：  tokens [T, H]  →  A2A  →  每卡收到发往本地 Expert 的 tokens [T_local, H]
  Expert计算：[T_local, H]   →  GEMM  →  expert_output [T_local, H]
  Gather：    expert_output [T_local, H]  →  A2A  →  送回 token 原始 GPU

引入 FSEP 后（需修改 token_unpermutation）：

  Dispatch：  tokens [T, H]  →  A2A  →  每卡收到 tokens [T_local, H]   （无变化）
  Expert计算：[T_local, H]   →  GEMM  →  partial [T_local, H]
  ReduceScatter：partial  →  RS  →  output_shard [T_local/S, H]   （新增）
  Gather：    output_shard [T_local/S, H]  →  A2A  →  送回原始 GPU    （A2A 数据量缩小 S 倍！）
```

**重要：FSEP 的 A2A Gather 数据量是 T/S × H，比传统 EP 的 T × H 小 S 倍！**  
这抵消了 ReduceScatter 的额外通信量，且 ReduceScatter 走节点内高带宽，净效果是跨节点通信量不变甚至减少。

**接入点：**
- `PrimusTurboDeepEPTokenDispatcher.combine_preprocess()` / `combine_postprocess()`
- `DeprecatedMoEAlltoAllTokenDispatcher.token_unpermutation()`

需增加 `fsep_sharding_degree` 参数，在 unpermutation 前插入 ReduceScatter，并调整 output_splits。

#### 3.2.4 配置参数扩展

```yaml
# 新增配置参数（primus/configs/models/megatron/language_model.yaml）

# FSEP 分片度（0 表示不启用，等于 EP 大小则完全均摊）
moe_fsep_sharding_degree: 0

# 负载检测窗口（动态 FSEP 用，静态 FSEP 设 0 表示禁用）
moe_fsep_relayout_interval: 0
```

---

### 3.3 方案 C：动态 FSEP + Re-layout（完整方案）

在方案 B 基础上新增：

#### 3.3.1 LoadPlanner 模块

```python
# primus/backends/megatron/core/transformer/moe/load_planner.py

class FSEPLoadPlanner:
    """
    监控每个 Expert 的 token 负载，决策何时以及如何调整分片度。
    触发条件：max_load / avg_load > threshold
    调整策略：热点 Expert 提升分片度 S；冷点降低 S（节省 ReduceScatter 通信）
    """
    def __init__(self, num_experts, relayout_interval=50, threshold=1.5):
        self.expert_load_ema = torch.zeros(num_experts)  # 指数移动平均
        self.relayout_interval = relayout_interval
        self.threshold = threshold
        self.step_count = 0

    def update(self, routing_map: torch.Tensor):
        # routing_map: [T, N_experts]，来自 PrimusTopKRouter 的 local_tokens_per_expert
        load = routing_map.sum(dim=0).float()
        self.expert_load_ema = 0.9 * self.expert_load_ema + 0.1 * load
        self.step_count += 1

    def should_relayout(self) -> bool:
        return (self.step_count % self.relayout_interval == 0 and
                self.expert_load_ema.max() / self.expert_load_ema.mean() > self.threshold)

    def compute_new_sharding(self, current_sharding: dict) -> dict:
        # 返回 {expert_id: new_sharding_degree}
        avg = self.expert_load_ema.mean()
        new_sharding = {}
        for eid, load in enumerate(self.expert_load_ema):
            ratio = load / avg
            if ratio > 2.0:
                new_sharding[eid] = min(current_sharding[eid] * 2, MAX_SHARD)
            elif ratio < 0.5:
                new_sharding[eid] = max(current_sharding[eid] // 2, 1)
            else:
                new_sharding[eid] = current_sharding[eid]
        return new_sharding
```

#### 3.3.2 Re-layout Executor（异步参数搬迁）

```
触发时机：每 K 个 step，在反向传播期间异步执行

执行流程：
  Step T（正常训练）：
    LoadPlanner 检测到 E2 过载，决定 E2 分片度 S: 2→4
           ↓
    在 Step T 的反向传播期间，异步 All-to-All 搬迁 E2 的参数分片
    （利用反向传播时间隐藏参数搬迁开销）
           ↓
  Step T+1：
    E2 的参数已按新分片度分布，直接使用新布局
    fsep_group 配置自动更新

内存管理：
  搬迁期间：double buffer（旧分片 + 新分片共存）
  搬迁完成：立即释放旧分片
  峰值显存增加：max_expert_params × 5~10%（短暂）
```

---

## 第四部分：与 Primus 现有机制的兼容性分析

### 4.1 与 DeepEP（PrimusTurboDeepEPTokenDispatcher）

```
兼容性：✅ 互补，可叠加

DeepEP 负责：A2A Dispatch 和 Gather 的底层通信优化（fused kernel）
FSEP 负责：Expert 计算后的 ReduceScatter（节点内），改变 Gather 的数据量

接入位置：
  - dispatch_postprocess() 之后：token 已经在目标 GPU 上，无影响
  - token_combine() 之前：FSEP ReduceScatter 在这里插入
  - combine_preprocess() 需调整 hidden_states 的形状（从 [T, H] 变为 [T/S, H]）

注意：DeepEP 的 permute_max_token_num 预分配 buffer 需根据 S 缩小，节省显存
```

### 4.2 与 PrimusTurboGroupedMLP

```
兼容性：✅ 可直接扩展

现有 GroupedMLP 已有 Expert TP 路径（沿 F 维参数分片），
但通信用 All-Reduce，需改为 ReduceScatter。

改动位置：
  forward() 末尾的 output 聚合部分：
    当前：output = all_reduce(partial, expert_tp_group)      # Expert TP 路径
    目标：output = reduce_scatter(partial, fsep_group)       # FSEP 路径

  backward() 无需显式修改（PyTorch autograd 自动对 ReduceScatter 求 AllGather 的梯度）

  与 FP8 grouped_gemm_fp8 兼容：grouped_gemm 本身输出 [T, H] partial，
  ReduceScatter 可在 partial 上独立施加，无需修改 GEMM kernel 本身
```

### 4.3 与 activation_recompute（CheckpointWithoutOutput）

```
兼容性：✅ 无冲突

现有 activation_recompute 在 fc1→act 处做 checkpoint，
FSEP 的 ReduceScatter 在 fc2 输出之后，二者串行无重叠。

注意：开启 FSEP 后，反向传播中 AllGather（RS 的反向）产生的中间激活
     是新增的显存开销，需评估是否也做 checkpoint。
     （AllGather 的数据量 = T×H，与 fc2 输出相同，不额外增大显存峰值）
```

### 4.4 与 ZeRO / FSDP（torch_fsdp2）

```
兼容性：⚠️ 需验证

FSDP 按 data parallel 维度分片参数，FSEP 按 fsep_group 维度分片参数。
当同时启用时，参数的实际物理分布是 FSDP×FSEP 的组合，
分布式 checkpoint（sharded_state_dict）的坐标系需要新增 fsep 轴。

当前 DeprecatedGroupedMLP.sharded_state_dict() 已有 (EP, TP, DP) 三轴，
FSEP 需新增 FSEP 轴，或复用 TP 轴（当 fsep_group == expert_tp_group 时）。

建议：初期禁止 FSEP 与 FSDP 同时使用，或将 FSEP 实现为 Expert TP 的语义扩展
（这样 sharded_state_dict 无需改动）。
```

### 4.5 与分布式 checkpoint

```
兼容性：✅ 可以实现，有工作量

关键：FSEP 下 Expert 参数的 replica_id 和 shard 坐标与传统 EP 不同。

静态 FSEP（固定分片度）：
  可在 sharded_state_dict 中将 fsep_sharding_degree 编码为新的分片轴，
  与现有 expert_model_parallel_rank 并列，复用 ShardedTensor.from_rank_offsets 接口。

动态 FSEP（Re-layout 后）：
  布局变化后需重新生成 sharded_state_dict 的映射关系，
  checkpoint save/load 需要感知当前 layout。
  建议：在 checkpoint 中保存 current_sharding 映射表，load 时恢复。
```

### 4.6 与 Zero Bubble / Primus Pipeline

```
兼容性：✅ 无直接冲突

execute_overlapped_1f1b 中的 moe_dispatch/moe_combine 抽象已经将通信和计算分离，
FSEP 的 ReduceScatter 属于 Expert 计算的一部分（在 moe_combine 之前），
不影响 dispatch/combine 的调度逻辑。

但要注意：use_split_wgrad_op 路径下（ZB/V-schedule），
grouped_gemm_with_weight_gradient_store 的 W 梯度延迟计算，
FSEP 的 ReduceScatter 需在 dW 计算前完成（ReduceScatter 不依赖 dW，顺序正确）。
```

---

## 第五部分：收益预测（基于 Primus DSv3 场景）

### 5.1 关键参数设置

```
DSv3 规格：
  N_experts = 256, Top-K = 4, H = 7168, F = 18944（SwiGLU after expand）
  EP = 8（典型配置），每卡 32 个 Expert
  节点规格：MI300X，XGMI 带宽 ~896 GB/s，跨节点 ~100~400 Gbps

负载不均衡（实测/估计）：
  max_load / avg_load ≈ 2~5（论文实测 3~5 为常见值）
  取 r = 3（保守估计）
```

### 5.2 计算收益模型

```
设：
  avg_tokens_per_expert_per_gpu = T_avg
  r = 不均衡比（max/avg）= 3
  S = FSEP 分片度 = 4

传统 EP 每步耗时（木桶效应）：
  t_EP = r × t_compute(T_avg) + 2 × t_A2A
       = 3 × t_c + 2 × t_A2A

FSEP 每步耗时（S=4，均衡计算）：
  t_FSEP = 1 × t_compute(T_avg)             ← 热点 Expert 被 4 GPU 分担
         + t_RS(T_avg × H, XGMI)             ← 节点内 ReduceScatter（极快）
         + 2 × t_A2A(T_avg/S)               ← A2A 数据量缩小 S 倍！
         ≈ t_c + t_RS_fast + 0.5 × t_A2A

  其中 t_RS_fast ≈ T_avg×H×2bytes / (896GB/s) 约为 A2A 的 1/10

理论加速比：
  t_EP / t_FSEP ≈ (3×t_c + 2×t_A2A) / (t_c + 0.2×t_A2A)

典型场景（A2A=10ms，t_c=5ms）：
  t_EP   = 3×5 + 2×10 = 35ms
  t_FSEP = 5 + 0.2×10 + 0.5×10 = 5 + 2 + 5 = 12ms
  加速比 ≈ 35/12 ≈ 2.9x

典型场景（A2A=5ms，t_c=8ms，r=2 较好情况）：
  t_EP   = 2×8 + 2×5 = 26ms
  t_FSEP = 8 + 1 + 2.5 = 11.5ms
  加速比 ≈ 26/11.5 ≈ 2.3x

论文报告：端到端 1.69x（含 overhead，较保守）
Primus 预期（已有 DeepEP 优化通信）：端到端 1.3~1.8x
```

### 5.3 显存变化分析

```
FSEP 对显存的影响：

① Expert 参数显存（无变化）
   传统 EP：每卡 N_E/EP × H×F 参数 = 32 × H×F
   FSEP：   每卡 N_E × H×F/S = 256 × H×F/4 = 64 × H×F
   → 参数总量相同，但分布更均匀，不存在 OOM 的卡

② 激活显存（ReduceScatter 减少峰值）
   传统 EP：Expert GEMM 输出 [T_max, H]，T_max = r × T_avg（热点 GPU）
   FSEP：  Expert GEMM 输出 [T_avg, H]，均衡后每卡 T_avg
   → 激活峰值降低 r 倍（r=3 时降 3x）

③ A2A buffer（减少）
   传统 EP：gather buffer = T × H
   FSEP：  gather buffer = T/S × H（A2A 数据量缩小 S 倍）

④ ReduceScatter buffer（新增，但节点内）
   新增 partial output buffer = T × H（与现有 Expert 输出等大，无额外峰值）

综合：FSEP 激活显存峰值降低约 2~3x（主要来自热点 Expert 均摊），
      同时 A2A buffer 减少，整体显存压力明显改善。
```

### 5.4 收益场景矩阵

```
                   负载不均衡程度
                   r=1.5      r=2      r=3      r=5
FSEP分片度 S=2     低收益     中       高       极高
FSEP分片度 S=4     低收益     中       高       极高
FSEP分片度 S=8     ↓         中       高       极高

DSv3 实测负载（256 Expert, Top-4）：r ≈ 3~5  ← FSEP 精准命中高收益区间

结论：DSv3 规模下，FSEP 是当前端到端收益最大的单项优化。
```

---

## 第六部分：实现工作量评估

### 6.1 静态 FSEP（方案 B）

| 模块 | 改动类型 | 工作量 | 说明 |
|------|----------|--------|------|
| `parallel_state.py` | 复用 expert_tp_group，新增 `get_fsep_group()` 接口 | 0.5 天 | 可直接映射到 expert TP group |
| `fsep_experts.py`（新增） | `FSEPGroupedMLP`，forward 改 ReduceScatter | 3~5 天 | 参考 PrimusTurboGroupedMLP，改通信原语 |
| `token_dispatcher.py` | `token_unpermutation` 适配 [T/S, H] 输入 | 2~3 天 | 改 output_splits 计算 + 形状对齐 |
| `primus_turbo.py` | `PrimusTurboDeepEPTokenDispatcher` combine 路径 | 2~3 天 | combine_preprocess 适配 T/S |
| 配置参数 | 新增 `moe_fsep_sharding_degree` | 0.5 天 | argument_builder.py |
| Patch 注册 | `moe_patches/__init__.py` + patch 文件 | 1 天 | 参考 permute_fusion_patches |
| 单测 | 正确性验证（EP=4，S=2，固定负载） | 3~5 天 | 关键路径覆盖 |
| **小计** | | **12~18 天** | |

### 6.2 动态 FSEP（方案 C，在方案 B 基础上）

| 模块 | 改动类型 | 工作量 | 说明 |
|------|----------|--------|------|
| `load_planner.py`（新增） | 负载监控 + 分片度决策 | 3~5 天 | EMA 负载统计 + 启发式搜索 |
| Re-layout Executor（新增） | 异步参数搬迁 + double buffer | 5~10 天 | 最复杂部分，需 overlap 反向传播 |
| `sharded_state_dict` 适配 | 动态 layout 下 checkpoint 兼容 | 3~5 天 | ShardedTensor 坐标系更新 |
| 联调测试 | 动态 relayout 正确性 + 性能验证 | 5~10 天 | 含 ZB/pipeline 联测 |
| **小计** | | **16~30 天** | |

**总计：方案 B 约 3~4 周，方案 C 约 7~10 周。**

---

## 第七部分：实施风险与缓解策略

### 7.1 风险矩阵

| 风险 | 等级 | 缓解策略 |
|------|------|----------|
| **ReduceScatter 与 DeepEP A2A 的 stream 冲突** | ⚠️ 中 | ReduceScatter 用独立 nccl stream，避免与 DeepEP comm_stream 阻塞 |
| **token 数量 T 不能被 S 整除** | ⚠️ 低 | padding 到 S 的倍数，或 variable-size ReduceScatter |
| **动态分片度变化时 checkpoint 不一致** | ⚠️ 高 | 强制 relayout 前保存一次 checkpoint；checkpoint 保存 layout 元数据 |
| **Re-layout 期间内存峰值超 OOM** | ⚠️ 中 | relayout 触发前检查 free memory ≥ double buffer 需求 |
| **Expert TP 与 FSEP 语义混淆** | ⚠️ 低 | 明确 FSEP = Expert TP + ReduceScatter 替换 All-Reduce，统一命名 |
| **torch.compile / graph 捕获兼容** | ⚠️ 低 | 初期不与 torch.compile 同时使用；动态 relayout 强制 recompile |

### 7.2 关键正确性验证点

```
验证 1：数学等价性
  FSEPGroupedMLP 输出必须与 FullGroupedMLP（全量权重）输出一致
  测试：固定权重，单 Expert 前向，比较误差 < 1e-5

验证 2：反向传播梯度正确性
  dW_shard 梯度之和 == 全量权重的 dW
  d_tokens 梯度 == 传统 EP 的 d_tokens

验证 3：Token Dispatcher 端到端
  dispatch → FSEP_GEMM → ReduceScatter → A2A_gather 全链路输出与传统 EP 等价

验证 4：负载均衡效果
  构造不均衡场景（E0 收 4x 平均 token），
  测量 FSEP 下各 GPU 实际计算时间标准差 < 5%（传统 EP 可达 100%+）
```

---

## 第八部分：分阶段推进建议

```
Week 1~2（方案 A + 基础设施）：
  ✅ 在 PrimusTopKRouter 中暴露 expert_load 统计（已有 local_tokens_per_expert）
  ✅ 增加 TensorBoard/WandB 的 per-expert token 分布 logging
  ✅ 验证当前 DSv3 训练的真实不均衡比 r（关键！决定 FSEP 的实际收益）
  ✅ 确认 expert_tensor_parallel_size 在目标配置中是否已启用

Week 3~6（方案 B 静态 FSEP）：
  → fsep_experts.py：FSEPGroupedMLP（ReduceScatter 替换 All-Reduce）
  → token_dispatcher 适配 [T/S, H]
  → 单测验证数学等价性
  → profile 对比：传统 EP vs 静态 FSEP（S=2,4）的 step time

Week 7~8（性能调优）：
  → ReduceScatter stream 与 DeepEP A2A 的并发优化
  → 与 activation_recompute 联测
  → 与 Zero Bubble pipeline 联测

Week 9~14（可选：方案 C 动态 FSEP）：
  → LoadPlanner 实现
  → Re-layout Executor（异步参数搬迁）
  → 动态 checkpoint 适配
  → 端到端 benchmark（目标：1.3x~1.7x vs 传统 EP+DeepEP 基线）
```

---

## 第九部分：一句话结论

**FSEP 的核心机制（Expert 参数沿 F 维分片 + ReduceScatter）在 Primus 中有天然的代码基础——`expert_tensor_parallel` 路径已完成参数分片，只需将通信原语从 All-Reduce 改为 ReduceScatter，并在 Token Dispatcher 中适配输出形状。静态 FSEP（方案 B）工程量约 3~4 周，风险可控，是当前 Primus MoE 优化中 ROI 最高的单项改造。**

---

*分析整理于 2026-03-09 | 基于 Primus Megatron backend 代码分析 + LAER-MoE 论文 (arXiv:2602.11686)*
