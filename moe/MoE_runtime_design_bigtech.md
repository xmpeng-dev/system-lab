# MoE Runtime 设计蓝图（优选大厂论文路线）

> **目标:** 基于 `moe/` 目录中大厂主导论文，给出一套可落地的 MoE 运行时（runtime）设计  
> **关注维度:** 调度（Scheduling）/ 显存（Memory）/ 计算（Compute）/ Overlap（Comm-Compute Overlap）  
> **适用场景:** 大规模训练优先，兼容在线推理服务

---

## 1. 设计原则（从论文到工程）

优先吸收的大厂路线：
- **Microsoft - Tutel**：动态并行切换 + 自适应流水 + 2DH/Flexible A2A  
- **ByteDance - MegaScale-MoE / MegaScale-Infer**：拓扑感知调度、分层 EP、分离式推理  
- **Meta - MoEBlaze**：路由中间态去物化、Kernel 融合、智能检查点  
- **NVIDIA - MoE Parallel Folding / Megatron-Core**：Attn 与 MoE 并行解耦、5D 并行组合  
- **Databricks 系（MegaBlocks）**：dropless + block-sparse 表达（工业可用基线）

统一设计信条：
1. **runtime 必须拥抱动态性**（不能假设路由负载稳定）  
2. **layout 优先于算子优化**（数据布局决定后续通信和算子效率上限）  
3. **内存与通信同等优先级**（MoE 里二者常比算力更早成为瓶颈）  
4. **策略层和执行层解耦**（控制面决策，数据面执行）

---

## 2. 总体架构（Control Plane + Data Plane）

```text
                     ┌───────────────────────────────────────┐
                     │           Control Plane               │
                     │  Runtime Planner                      │
                     │  - Parallelism Selector (r, TP/EP/CP) │
                     │  - Pipelining Selector (degree, A2A)  │
                     │  - Memory Policy (checkpoint/offload) │
                     │  - Topology-aware Expert Placement    │
                     └───────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                               Data Plane                                 │
│  Stage A: Routing + Dispatch  ->  Stage B: Expert Compute -> Stage C: Merge │
│  - Lightweight metadata             - Fused kernels          - on-the-fly reduce │
│  - Flexible/2DH A2A                 - Block-sparse/fused MLP - output scatter     │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 调度设计（Scheduling）

### 3.1 并行策略调度器（Adaptive Parallelism）

借鉴 Tutel + Megatron-Core：
- 定义统一并行状态：`{DP, TP, EP, CP, PP, r}`
- 不同 step 按负载特征切换 `r` 和 pipeline degree
- 对同一 block 内 Attn/MoE 允许不同并行配置（Parallel Folding 思路）

建议实现：
- 维护 `capacity_bucket -> best_config` 字典（Tutel 风格）
- key 建议包含：
  - `expert_load_skew`（max/avg）
  - `tokens_per_expert_p95`
  - `a2a_bw_util`
  - `seq_len_bucket`
- 每 N step 在线重评估；平稳期使用缓存配置，避免抖动

### 3.2 拓扑感知路由与放置（Topology-aware）

借鉴 MegaScale-MoE：
- Expert 初始放置按网络层级（同 GPU > 同节点 > 同 rack > 跨 rack）
- Gate 端引入微弱 locality bias（不改变主路由语义）
- 分层 EP：热点 expert 尽量节点内消化，冷 expert 跨节点服务

### 3.3 推理场景的调度（Disaggregated EP）

借鉴 MegaScale-Infer：
- Prefill/Decode/Expert 角色解耦
- Expert Pool 独立扩缩容
- 请求路由使用 WRR/队列时延估计，目标是 TTFT/TBT 双约束

---

## 4. 显存设计（Memory）

### 4.1 路由内存：去物化优先

借鉴 MoEBlaze：
- 禁止大体积 `routed_tokens` 中间缓冲常驻
- 只保留轻量索引结构：
  - `expert_token_indices`
  - `expert_token_offsets`
  - `token_expert_indices`
  - `token_index_map`
- 执行期按索引 on-the-fly gather/reduce

### 4.2 激活内存：选择性检查点

策略：
- `cheap_compute + huge_memory` 张量：不存，反向重算
- `expensive_compute + moderate_memory` 张量：保留
- 对 SwiGLU/SiLU 等路径做细粒度 AC，而不是整层 AC

### 4.3 内存预算器（Memory Budgeter）

每 step 生成预算：
- `routing_budget`
- `activation_budget`
- `workspace_budget`
- `optimizer_state_budget`

当触发水位线（如 >90% HBM）：
1) 降低 pipeline degree（减并发 chunk）  
2) 提高重算比例  
3) 缩小 micro-batch  
4) 必要时触发临时 offload（推理优先）

---

## 5. 计算设计（Compute）

### 5.1 计算内核路线

组合策略：
- **路线 A（动态负载优先）**：MegaBlocks 风格 block-sparse expert compute
- **路线 B（内存墙优先）**：MoEBlaze 风格 fused routing+MLP epilogue
- 二者在 runtime 中可按形状/负载条件切换（kernel policy）

### 5.2 Expert 计算组织

- expert 粒度做 grouped GEMM（同形状批量化）
- 避免小 batch expert 过碎：
  - 按 token 数分桶（small/medium/large experts）
  - 小 expert 采用合并执行（merge launch）

### 5.3 并行冲突处理

借鉴 MoE Parallel Folding：
- Attn 子层用 TP/CP 优先
- MoE 子层用 EP/TP_EP 优先
- 子层间做 layout remap（一次可控 A2A）

---

## 6. Overlap 设计（Communication-Compute）

### 6.1 三段重叠流水

目标流水：
1) `A2A dispatch(chunk i)`  
2) `expert compute(chunk i-1)`  
3) `A2A gather(chunk i-2)`  

通过多 stream + 事件依赖构成稳定重叠。

### 6.2 通信算法自适应

借鉴 Tutel：
- 小规模优先 Linear A2A
- 大规模/跨节点优先 2DH A2A
- 与 pipeline degree 联合搜索，不独立拍脑袋配置

### 6.3 避免“伪 overlap”

runtime 需检测：
- NCCL 与 compute stream 抢 SM 导致双输
- chunk 过小导致 launch 开销吞掉重叠收益

建议设置自动阈值：
- 若 overlap gain < 5%，降 degree 或切换 A2A 算法

---

## 7. 关键运行时策略（可直接实现）

```python
class MoERuntimePolicy:
    def select(self, stats):
        # stats: load_skew, tokens_per_expert, hbm_used, a2a_latency, seq_len ...
        parallel_cfg = self.parallel_dict.lookup(stats.capacity_bucket)
        pipeline_cfg = self.pipe_dict.lookup(stats.capacity_bucket)

        if stats.hbm_used > 0.90:
            self.memory_policy.enable_recompute_ratio("high")
            pipeline_cfg.degree = max(1, pipeline_cfg.degree // 2)

        if stats.load_skew > 2.5:
            parallel_cfg = self.bias_to_ep(parallel_cfg)

        if stats.a2a_latency_ratio > 0.40:
            pipeline_cfg.a2a_algo = "2dh"
            pipeline_cfg.degree = self.search_joint_optimal(parallel_cfg, pipeline_cfg)

        kernel_cfg = self.kernel_policy.choose(
            dynamic_load=stats.load_skew,
            memory_pressure=stats.hbm_used
        )
        return parallel_cfg, pipeline_cfg, kernel_cfg
```

---

## 8. 分阶段落地路线（推荐）

### Phase 1（2-4 周）：先把收益最大的 runtime 基础打稳
- 引入动态配置字典（parallel/pipeline）
- 接入 topology-aware expert placement（静态放置）
- 上线基础 chunked overlap（dispatch-compute-gather）

### Phase 2（4-8 周）：显存与内核深优化
- 路由去物化 + 轻量索引
- SwiGLU 路径融合 + smart checkpoint
- expert compute 双内核策略（block-sparse / fused）

### Phase 3（8 周+）：训练-推理一体化
- 训练侧分层 EP + 容错
- 推理侧 disaggregated EP（prefill/decode/expert 分离）
- 统一 runtime planner（同一控制面）

---

## 9. KPI 指标（验收标准）

调度：
- `load_skew`（max/avg）下降  
- 并行策略切换次数与收益比（避免抖动）

显存：
- 峰值激活内存下降比例  
- 可支持 micro-batch / seq length 提升倍数

计算：
- expert kernel 有效吞吐（TFLOPS）  
- GPU SM 利用率

overlap：
- A2A 时间占比下降  
- 通信-计算重叠率（目标 >70%，视场景）

端到端：
- 训练 tokens/s  
- 推理 TTFT / TBT / cost per token

---

## 10. 最终建议（给当前仓库）

结合你 `moe/` 现有阅读，建议把 runtime 主线定为：
1. **Tutel 的自适应调度框架**（控制面骨架）  
2. **MegaScale 的拓扑与分层 EP**（大规模稳定性）  
3. **MoEBlaze 的显存与路由内核**（数据面效率）  
4. **Parallel Folding 的异构并行解耦**（Attn/MoE 双最优）  
5. **MegaBlocks 作为 dropless 稀疏计算基线**（工程兜底）

这条路线最像“可演进的工业 runtime”，而不是单点优化拼接。

