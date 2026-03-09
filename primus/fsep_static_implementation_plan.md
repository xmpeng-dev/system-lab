# FSEP 静态方案（方案 B）详细实施计划

> **Branch：** `feat/moe-fsep-static-sharding`  
> **目标：** 在 Primus Megatron backend 实现固定分片度的静态 FSEP，消除 Expert 负载不均衡，端到端加速 1.3~1.8x  
> **总工期：** 3~4 周（12~18 工作日）  
> **更新：** 2026-03-09

---

## 总览：任务依赖图

```
Task 1: 负载监控 & Logging（2 天）
    ↓
Task 2: 配置参数注册（1 天）
    ↓
Task 3: FSEPGroupedMLP（5 天）─────────────────┐
    ↓                                           │
Task 4: Dispatcher 适配（3 天）                 │
    ↓                                           │
Task 5: Patch 注册（1 天）                      │
    ↓                                           ↓
Task 6: 单元测试 - 数学等价性（3 天）←─── Task 3 完成后即可并行
    ↓
Task 7: 集成测试 & 性能 Benchmark（3 天）
    ↓
Task 8: 文档 & Code Review（1 天）
```

---

## Task 1：负载监控 & Expert Load Logging

**工期：2 天**  
**优先级：🔴 最高（是后续所有任务的基础数据）**

### 目标
在开始任何代码改动之前，先量化 DSv3 训练中真实的负载不均衡比 `r = max_load / avg_load`。这是决定 FSEP 收益和最优分片度 `S` 的关键输入。

### 改动文件
`primus/backends/megatron/core/transformer/moe/router.py`

### 具体改动

**改动 1-A：在 `PrimusTopKRouter.routing()` 末尾增加 per-expert token 统计上报**

```python
# primus/backends/megatron/core/transformer/moe/router.py
# 在 routing() 返回前插入，条件：仅 training 且 rank 0

def routing(self, logits: torch.Tensor):
    args = get_args()
    # ... 现有逻辑 ...

    # [新增] Expert 负载统计上报
    if args.moe_log_expert_load and torch.is_grad_enabled():
        with torch.no_grad():
            load = routing_map.float().sum(dim=0)  # [N_experts]
            # 全局聚合（EP 维度），得到全局 expert 负载
            if self.config.expert_model_parallel_size > 1:
                dist.all_reduce(load,
                    group=parallel_state.get_expert_model_parallel_group())
            # 写入 tracker，复用已有 MoE metrics 上报机制
            from megatron.core.transformer.moe.moe_utils import (
                get_moe_layer_wise_logging_tracker,
            )
            tracker = get_moe_layer_wise_logging_tracker()
            if "expert_load_max_avg_ratio" not in tracker:
                tracker["expert_load_max_avg_ratio"] = {
                    "values": torch.zeros(
                        self.config.num_layers, device="cuda"
                    ),
                    "reduce_group": None,
                    "avg_group": None,
                }
            layer_idx = getattr(self, "_layer_number", 0)
            avg_load = load.mean()
            if avg_load > 0:
                ratio = load.max() / avg_load
                tracker["expert_load_max_avg_ratio"]["values"][layer_idx] = ratio

    return scores, routing_map
```

**改动 1-B：在 `language_model.yaml` 增加开关**

```yaml
# primus/configs/models/megatron/language_model.yaml
# 在 moe 配置块末尾追加：
moe_log_expert_load: false  # 开启 per-step expert 负载比统计（轻微性能影响）
```

**改动 1-C：在 `moe_utils.py` 的 `track_moe_metrics` 中增加 `expert_load_max_avg_ratio` 的处理**

修改 `primus/backends/megatron/core/transformer/moe/moe_utils.py`，在 `track_moe_metrics` 中把 `expert_load_max_avg_ratio` 加入写 TensorBoard 的逻辑（复用现有 `aux_losses` 循环，无需额外改动，只要 tracker 里有这个 key 即可自动上报）。

### 验收标准
- [ ] 开启 `moe_log_expert_load: true` 运行 100 步，TensorBoard 可见 `expert_load_max_avg_ratio` 曲线
- [ ] 记录 DSv3 配置（256 Expert, EP=8, Top-8）下实测 `r` 值（预期 2~5）

---

## Task 2：配置参数注册

**工期：1 天**  
**依赖：Task 1 完成（确认 r 值后确定 S 的合理范围）**

### 目标
注册 `moe_fsep_sharding_degree` 参数，让后续各模块可以通过 `get_args()` 读取。

### 改动文件

**文件 1：`primus/configs/models/megatron/language_model.yaml`**

在 `expert_tensor_parallel_size` 附近追加：

```yaml
# FSEP (Fully Sharded Expert Parallel) - static sharding
# 0 = disabled; S = N means each Expert is sharded across S GPUs within EP group
# Valid values: 0, 2, 4, 8 (must divide expert_model_parallel_size)
moe_fsep_sharding_degree: 0
```

**文件 2：`primus/backends/megatron/patches/args_patches.py`（或新文件）**

由于 `moe_fsep_sharding_degree` 不是 Megatron 原生参数（Megatron 的 `argparse` 不认识它），需要在 `args_compat_patches.py` 或 Primus 自己的参数注入处理中添加默认值，避免 `get_args()` AttributeError：

```python
# primus/backends/megatron/patches/args_patches.py
# 在 patch_args_defaults() 或等效位置追加：
if not hasattr(args, "moe_fsep_sharding_degree"):
    args.moe_fsep_sharding_degree = 0
if not hasattr(args, "moe_log_expert_load"):
    args.moe_log_expert_load = False
```

**参数约束校验（在 MegatronBaseTrainer 或 megatron_pretrain_trainer 中）：**

```python
# 在训练启动前的参数校验逻辑中添加：
if args.moe_fsep_sharding_degree > 0:
    assert args.moe_fsep_sharding_degree in [2, 4, 8, 16], \
        "moe_fsep_sharding_degree must be power of 2"
    assert args.expert_model_parallel_size % args.moe_fsep_sharding_degree == 0, \
        f"EP size ({args.expert_model_parallel_size}) must be divisible by " \
        f"moe_fsep_sharding_degree ({args.moe_fsep_sharding_degree})"
    assert args.moe_fsep_sharding_degree <= args.expert_model_parallel_size, \
        "moe_fsep_sharding_degree cannot exceed expert_model_parallel_size"
    assert not args.moe_pad_expert_input_to_capacity, \
        "FSEP does not support moe_pad_expert_input_to_capacity"
    log_rank_0(
        f"[FSEP] Static FSEP enabled: sharding_degree={args.moe_fsep_sharding_degree}, "
        f"EP={args.expert_model_parallel_size}"
    )
```

### 验收标准
- [ ] `moe_fsep_sharding_degree: 4` 配置可正常加载，`get_args().moe_fsep_sharding_degree == 4`
- [ ] 参数约束校验正确拦截非法配置（EP=8, S=3 时报错）

---

## Task 3：FSEPGroupedMLP 实现

**工期：5 天**  
**依赖：Task 2 完成**  
**这是整个方案的核心模块**

### 目标
实现 `FSEPGroupedMLP`，将 Expert 计算的聚合通信从 `All-Reduce [T, H]` 改为 `ReduceScatter → [T/S, H]`。

### 新建文件
`primus/backends/megatron/core/transformer/moe/fsep_experts.py`

### 关键设计决策

**与现有 Expert TP 的关系：**

```
现有 Expert TP（megatron GroupedMLP）：
  weight1 已按 expert_tensor_parallel_size 沿 F 维切分 → [H, F/tp]
  weight2 已按 expert_tensor_parallel_size 沿 F 维切分 → [F/tp, H]
  fc2 输出：partial [T, H] → All-Reduce → 完整 [T, H]

FSEP（目标）：
  weight1 按 moe_fsep_sharding_degree 沿 F 维切分 → [H, F/S]
  weight2 按 moe_fsep_sharding_degree 沿 F 维切分 → [F/S, H]
  fc2 输出：partial [T, H] → ReduceScatter → 片段 [T/S, H]

实现策略：
  FSEPGroupedMLP 继承 PrimusTurboGroupedMLP，
  - 覆写 __init__ 以使用 fsep process group（而非 expert_tp_group）初始化权重
  - 覆写 forward 末尾的聚合步骤（ReduceScatter 替换 All-Reduce）
```

### 完整实现

```python
# primus/backends/megatron/core/transformer/moe/fsep_experts.py

###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################

"""
FSEP (Fully Sharded Expert Parallel) GroupedMLP.

Implements Expert parameter sharding along the FFN intermediate dimension,
with ReduceScatter instead of All-Reduce for output aggregation.
This enables load-balanced computation when Expert token assignment is skewed.
"""

import functools
from typing import Optional

import torch
import torch.distributed as dist
from megatron.core import parallel_state, tensor_parallel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.global_vars import get_args

from primus.backends.megatron.core.extensions.primus_turbo import (
    PrimusTurboGroupedMLP,
)
from primus.modules.module_utils import log_rank_0


def get_fsep_group():
    """
    Return the process group for FSEP ReduceScatter/AllGather.

    FSEP uses the expert_tensor_parallel_group as its sharding group.
    When moe_fsep_sharding_degree == expert_tensor_parallel_size,
    these two concepts are identical.

    When expert_tensor_parallel_size is 1 (default), FSEP creates a
    sub-group within the EP group of size moe_fsep_sharding_degree.
    """
    # If expert TP is already set to the desired sharding degree,
    # reuse the existing expert TP group directly.
    args = get_args()
    S = args.moe_fsep_sharding_degree
    etp_size = parallel_state.get_expert_tensor_parallel_world_size()

    if etp_size == S:
        return parallel_state.get_expert_tensor_parallel_group()

    # Otherwise, return the EP group (S == EP means all GPUs in EP group share)
    # This handles the case where expert_tensor_parallel_size=1 and S>1
    # We rely on the EP group for ReduceScatter across all EP ranks
    ep_size = parallel_state.get_expert_model_parallel_world_size()
    assert S == ep_size, (
        f"When expert_tensor_parallel_size ({etp_size}) != S ({S}), "
        f"S must equal EP size ({ep_size}). "
        f"Mixed FSEP+ETP configurations are not yet supported."
    )
    return parallel_state.get_expert_model_parallel_group()


def get_fsep_rank():
    """Return current rank within the FSEP group."""
    return dist.get_rank(group=get_fsep_group())


def get_fsep_world_size():
    """Return world size of the FSEP group (= sharding degree S)."""
    return dist.get_world_size(group=get_fsep_group())


class FSEPGroupedMLP(PrimusTurboGroupedMLP):
    """
    Fully Sharded Expert Parallel GroupedMLP.

    Differs from PrimusTurboGroupedMLP in one key aspect:
    - The output aggregation uses ReduceScatter instead of All-Reduce.
    - Output shape: [T/S, H] (each GPU holds a shard of the token dimension)
      instead of [T, H] (each GPU holds the full result).

    This enables true load balancing: a hot Expert with T_hot >> T_avg tokens
    is computed in parallel across S GPUs, each doing T_hot/S work.

    The downstream Token Dispatcher (combine path) must be aware of the
    [T/S, H] output and adjust A2A Gather splits accordingly.
    """

    def __init__(
        self,
        num_local_experts: int,
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        # Validate FSEP configuration
        args = get_args()
        self.fsep_sharding_degree = args.moe_fsep_sharding_degree
        assert self.fsep_sharding_degree > 1, \
            "FSEPGroupedMLP requires moe_fsep_sharding_degree > 1"

        super().__init__(num_local_experts, config, pg_collection)

        self.fsep_group = get_fsep_group()
        self.fsep_rank = get_fsep_rank()
        self.fsep_world_size = get_fsep_world_size()

        log_rank_0(
            f"[FSEP] FSEPGroupedMLP initialized: "
            f"num_local_experts={num_local_experts}, "
            f"sharding_degree={self.fsep_sharding_degree}, "
            f"weight1={tuple(self.weight1.shape)}, "
            f"weight2={tuple(self.weight2.shape)}"
        )

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ):
        """
        Forward pass with ReduceScatter output aggregation.

        Input:  permuted_local_hidden_states [T_local, H]
                tokens_per_expert [num_local_experts]
        Output: [T_local / fsep_world_size, H]
                (each GPU holds a contiguous shard of T_local tokens)

        Note: T_local must be divisible by fsep_world_size.
              Padding is handled in the dispatcher before this call.
        """
        # Run the base GroupedMLP forward (fc1 → act → fc2)
        # This produces partial output [T_local, H] since weights are sharded
        # along F dim. The base class uses Expert TP All-Reduce internally
        # only when expert_tensor_parallel_size > 1; in FSEP mode we
        # intercept BEFORE that All-Reduce.
        #
        # Implementation note: we call grandparent's forward logic directly
        # to get partial [T_local, H] before any collective, then apply
        # ReduceScatter ourselves.
        partial_output = self._forward_no_reduce(
            permuted_local_hidden_states, tokens_per_expert, permuted_probs
        )

        # ReduceScatter: reduce partial sums, scatter along token dimension
        # Input:  [T_local, H]  (each GPU has a different partial sum)
        # Output: [T_local/S, H] (each GPU gets sum for its token shard)
        output = self._fsep_reduce_scatter(partial_output)

        return output, None

    def _forward_no_reduce(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run GEMM pipeline (fc1 → act → fc2) without final collective.
        Returns partial output [T_local, H].

        Re-implements the GEMM logic from PrimusTurboGroupedMLP.forward()
        but skips the final All-Reduce that Expert TP would normally do.
        """
        import primus_turbo.pytorch as pt
        from primus.backends.megatron.core.extensions.primus_turbo import (
            PrimusTurboLowPrecisionGlobalStateManager,
            use_split_wgrad_op,
        )

        args = get_args()

        if self.config.moe_apply_probs_on_input:
            assert self.config.moe_router_topk == 1
            original_dtype = permuted_local_hidden_states.dtype
            permuted_local_hidden_states = (
                permuted_probs.unsqueeze(-1) * permuted_local_hidden_states
            )
            permuted_local_hidden_states = permuted_local_hidden_states.to(original_dtype)
            permuted_probs = torch.ones_like(permuted_probs)

        if permuted_local_hidden_states.nelement() == 0:
            # No tokens allocated: return zero tensor with correct shape
            T = permuted_local_hidden_states.shape[0]
            return torch.zeros(
                T, self.config.hidden_size,
                dtype=permuted_local_hidden_states.dtype,
                device=permuted_local_hidden_states.device,
            )

        gemm_kargs = [dict(), dict()]
        if use_split_wgrad_op():
            w1 = self.weight1
            w2 = self.weight2
            gemm_kargs[0]["weight_reshape_size"] = (
                self.num_local_experts, self.config.hidden_size, -1
            )
            gemm_kargs[1]["weight_reshape_size"] = (
                self.num_local_experts, -1, self.config.hidden_size
            )
        else:
            w1 = self.weight1.view(self.num_local_experts, self.config.hidden_size, -1)
            w2 = self.weight2.view(self.num_local_experts, -1, self.config.hidden_size)

        tokens_per_expert = tokens_per_expert.to(w1.device)

        # fc1
        if PrimusTurboLowPrecisionGlobalStateManager.is_turbo_fp8_enabled():
            quant_config = PrimusTurboLowPrecisionGlobalStateManager.get_turbo_quant_config()
            fc1_output = pt.ops.grouped_gemm_fp8(
                permuted_local_hidden_states, w1, tokens_per_expert,
                trans_b=False, config=quant_config.data()
            )
        else:
            fc1_output = self.grouped_gemm(
                permuted_local_hidden_states, w1, tokens_per_expert,
                trans_b=False, **(gemm_kargs[0])
            )

        # activation
        if self.activation_recompute:
            if not hasattr(self, "activation_checkpoint"):
                self.activation_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            if args.use_turbo_fused_act_with_probs:
                intermediate = self.activation_checkpoint.checkpoint(
                    self.activation_func_with_probs,
                    fc1_output, permuted_probs, tokens_per_expert,
                )
            else:
                intermediate = self.activation_checkpoint.checkpoint(
                    self.activation_func_with_probs, fc1_output,
                    permuted_probs.unsqueeze(-1)
                )
        else:
            if args.use_turbo_fused_act_with_probs:
                intermediate = self.activation_func_with_probs(
                    fc1_output, permuted_probs, tokens_per_expert
                )
            else:
                intermediate = self.activation_func_with_probs(
                    fc1_output, permuted_probs.unsqueeze(-1)
                )

        # fc2 → partial output [T_local, H]
        if PrimusTurboLowPrecisionGlobalStateManager.is_turbo_fp8_enabled():
            quant_config = PrimusTurboLowPrecisionGlobalStateManager.get_turbo_quant_config()
            partial_output = pt.ops.grouped_gemm_fp8(
                intermediate, w2, tokens_per_expert,
                trans_b=False, config=quant_config.data()
            )
        else:
            partial_output = self.grouped_gemm(
                intermediate, w2, tokens_per_expert,
                trans_b=False, **(gemm_kargs[1])
            )

        if self.activation_recompute:
            self.activation_checkpoint.discard_output_and_register_recompute(partial_output)

        return partial_output  # [T_local, H], partial sum across F-shard

    def _fsep_reduce_scatter(self, partial: torch.Tensor) -> torch.Tensor:
        """
        ReduceScatter along token dimension within fsep_group.

        Input:  partial [T, H]  — each GPU has partial sum from its F-shard
        Output: [T/S, H]       — each GPU gets the reduced sum for T/S tokens

        Uses reduce_scatter_to_sequence_parallel_region semantics:
        splits T into S chunks, GPU_i gets chunk_i after sum-reduction.

        Padding requirement: T must be divisible by S.
        This is enforced by the dispatcher via padding before this call.
        """
        S = self.fsep_world_size
        T = partial.shape[0]

        if S == 1:
            # No sharding, pass-through
            return partial

        assert T % S == 0, (
            f"FSEP ReduceScatter: T={T} must be divisible by S={S}. "
            f"Ensure the dispatcher pads tokens to multiples of {S}."
        )

        # dist.reduce_scatter_tensor:
        # output = chunk[rank] of sum(inputs across group)
        output = torch.empty(
            T // S, partial.shape[1],
            dtype=partial.dtype,
            device=partial.device,
        )
        dist.reduce_scatter_tensor(output, partial, group=self.fsep_group)

        return output  # [T/S, H]
```

### Day-by-Day 开发顺序

| 日期 | 工作内容 |
|------|---------|
| Day 1 | 实现 `get_fsep_group()` 及 `_fsep_reduce_scatter()`；单 GPU 正确性验证（S=1 pass-through）|
| Day 2 | 实现 `_forward_no_reduce()`；对齐 PrimusTurboGroupedMLP.forward 的所有 branch（fp8/非fp8，activation_recompute/非）|
| Day 3 | 实现完整 `FSEPGroupedMLP.forward()`；测试 BF16 + FP8 两路 |
| Day 4 | 验证反向传播梯度（`ReduceScatter` 的反向 = `AllGather`，PyTorch autograd 自动处理）|
| Day 5 | 处理 `use_split_wgrad_op` (ZB/V-schedule) 路径兼容；`sharded_state_dict` 适配 |

### sharded_state_dict 适配

```python
# FSEPGroupedMLP 的 sharded_state_dict 策略：
# 当 fsep_group == expert_tp_group 时，直接复用父类 sharded_state_dict。
# 当 fsep_group == ep_group 时，需调整分片轴（FSEP 沿 EP 维度分片参数）。

def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
    args = get_args()
    if args.moe_fsep_sharding_degree == parallel_state.get_expert_tensor_parallel_world_size():
        # FSEP 与 Expert TP 对齐：直接复用父类逻辑
        return super().sharded_state_dict(prefix, sharded_offsets, metadata)
    else:
        # FSEP 沿 EP 维度分片：weight1/weight2 的分片轴从 EP 维变为 FSEP 维
        # 暂时 raise NotImplementedError，初期不支持与 dist-ckpt 混用
        raise NotImplementedError(
            "FSEP with sharding_degree != expert_tensor_parallel_size "
            "is not yet supported for distributed checkpoint. "
            "Set expert_tensor_parallel_size == moe_fsep_sharding_degree."
        )
```

**初期约束（简化实现）：要求 `moe_fsep_sharding_degree == expert_tensor_parallel_size`。** 这样参数分片轴完全复用 Expert TP，`sharded_state_dict` 零改动。

### 验收标准
- [ ] `FSEPGroupedMLP` 输出与 `FullGroupedMLP`（无分片）在 FP32 下误差 < 1e-5
- [ ] 反向传播 `dW` 之和与全量权重 `dW` 匹配
- [ ] `activation_recompute=True` + FP8 路径均通过
- [ ] S=1 时行为与 `PrimusTurboGroupedMLP` 完全相同

---

## Task 4：Token Dispatcher 适配

**工期：3 天**  
**依赖：Task 3 完成（需要 FSEPGroupedMLP 输出 [T/S, H] 已确认）**

### 目标
Token Dispatcher 的 `combine` 路径（A2A Gather）需要知道 Expert 输出已是 `[T/S, H]` 的片段，而非 `[T, H]`，以正确计算 `output_splits` 并在 combine 后恢复原始 token 形状。

### 涉及的两条路径

**路径 A（主力）：`PrimusTurboDeepEPTokenDispatcher`**

```
文件：primus/backends/megatron/core/extensions/primus_turbo.py
类：PrimusTurboDeepEPTokenDispatcher
```

**路径 B（deprecated）：`DeprecatedMoEAlltoAllTokenDispatcher`**

```
文件：primus/backends/megatron/core/transformer/moe/deprecated_20251209/token_dispatcher.py
类：DeprecatedMoEAlltoAllTokenDispatcher
```

初期**只适配路径 A**（DeepEP dispatcher），路径 B 标注 TODO。

### 路径 A 改动设计

FSEP 后，`combine_preprocess` 接收到的 `hidden_states` 形状是 `[T/S, H]` 而不是 `[T, H]`。DeepEP 的 `_pre_combine` 期望全量 `[T, H]`，因此需要在调用 `_pre_combine` 之前先做 **AllGather** 恢复 `[T, H]`，再交给 DeepEP。

```python
# 改动位置：PrimusTurboDeepEPTokenDispatcher

def __init__(self, ...):
    super().__init__(...)
    # [新增] FSEP 配置
    args = get_args()
    self.fsep_sharding_degree = getattr(args, "moe_fsep_sharding_degree", 0)
    if self.fsep_sharding_degree > 1:
        from primus.backends.megatron.core.transformer.moe.fsep_experts import (
            get_fsep_group,
        )
        self.fsep_group = get_fsep_group()
        self.fsep_world_size = dist.get_world_size(group=self.fsep_group)
        log_rank_0(f"[FSEP] DeepEP dispatcher FSEP mode: S={self.fsep_sharding_degree}")

def combine_preprocess(self, hidden_states: torch.Tensor):
    """Pre-processes hidden states before A2A Gather.

    In FSEP mode, expert output is [T/S, H]. We need to AllGather
    back to [T, H] before feeding into DeepEP's _pre_combine.
    """
    if self.fsep_sharding_degree > 1:
        # AllGather: [T/S, H] × S GPUs → [T, H]
        hidden_states = self._fsep_all_gather(hidden_states)

    # 原有逻辑
    hidden_states = self.deepep_dispatcher._pre_combine(hidden_states)
    return hidden_states

def _fsep_all_gather(self, shard: torch.Tensor) -> torch.Tensor:
    """
    AllGather along token dimension within fsep_group.

    Input:  [T/S, H]  — this GPU's token shard
    Output: [T, H]    — full token sequence (ordered by rank)

    This is the inverse of ReduceScatter done in FSEPGroupedMLP.
    PyTorch autograd treats AllGather as the backward of ReduceScatter,
    so gradients flow correctly without manual intervention.
    """
    S = self.fsep_world_size
    T_shard = shard.shape[0]
    H = shard.shape[1]

    # Allocate output buffer [T, H]
    output = torch.empty(
        T_shard * S, H,
        dtype=shard.dtype,
        device=shard.device,
    )
    dist.all_gather_into_tensor(output, shard, group=self.fsep_group)
    return output  # [T, H]
```

**重要：为什么这样是正确的？**

```
Forward：
  FSEPGroupedMLP:  partial [T, H] → ReduceScatter → shard [T/S, H]
  DeepEP combine:  shard [T/S, H] → AllGather → [T, H] → A2A Gather → token 原始 GPU

Backward（PyTorch autograd 自动处理）：
  d[T, H] → grad of AllGather = ReduceScatter(d[T, H]) → d_shard [T/S, H]
  d_shard [T/S, H] → grad of ReduceScatter = AllGather(d_shard) → partial_grad [T, H]
  partial_grad [T, H] → fc2 backward → dW_shard, d_tokens [T, H]

通信量对比（forward）：
  传统 EP：A2A Gather = T × H
  FSEP：   ReduceScatter(节点内) + AllGather(节点内) + A2A Gather
           = T×H(RS) + T×H(AG) + T×H(A2A)
  等等，AllGather 让 A2A 数据量没有减少？

实际上：AllGather 只发生在 fsep_group（节点内 XGMI），不走跨节点路径。
A2A Gather 依然发送全量 [T, H]（combine 之后已恢复全量）。

修正：FSEP 的跨节点 A2A 数据量与传统 EP 相同，额外开销仅在节点内
（ReduceScatter + AllGather = 2 × T × H，但 XGMI 带宽 896 GB/s，可忽略）。

核心收益来自计算均衡，不来自 A2A 数据量减少。
```

> **注意：** 分析文档中 "A2A Gather 数据量缩小 S 倍" 的说法需要修正。严格来说，如果在 combine_preprocess 用 AllGather 恢复 [T, H] 再进 DeepEP，A2A 数据量不变。真正减少 A2A 数据量需要 DeepEP 原生支持 FSEP 感知的 combine（即接受 [T/S, H] 输入，在 A2A 前做 AllGather）。**初期实现以正确性为先，AllGather 放在 combine_preprocess，不修改 DeepEP 内部。**

### Day-by-Day 开发顺序

| 日期 | 工作内容 |
|------|---------|
| Day 1 | `PrimusTurboDeepEPTokenDispatcher.__init__` 增加 fsep 配置读取；`combine_preprocess` 增加 AllGather 分支 |
| Day 2 | 端到端前向验证（dispatch → FSEP GEMM → AllGather → combine）；与 Task 3 的 FSEPGroupedMLP 联调 |
| Day 3 | 反向传播梯度验证；`sync_free_moe_stage > 1` 路径兼容测试 |

### 验收标准
- [ ] `dispatch → FSEPGroupedMLP → combine_preprocess(AllGather) → token_combine → combine_postprocess` 全链路输出与传统 EP 数值一致
- [ ] 反向梯度 `d_hidden` 与传统 EP 一致（误差 < 1e-4 for BF16）
- [ ] `turbo_sync_free_moe_stage = 0, 1, 2, 3` 四档均测试通过

---

## Task 5：Patch 注册

**工期：1 天**  
**依赖：Task 3、Task 4 完成**

### 目标
通过 Primus patch 机制将 `FSEPGroupedMLP` 和 FSEP-aware dispatcher 自动注入 Megatron，使现有配置只需加 `moe_fsep_sharding_degree: 4` 即可启用。

### 新建文件
`primus/backends/megatron/patches/moe_patches/fsep_patches.py`

```python
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################

"""
Megatron MoE FSEP (Fully Sharded Expert Parallel) Patches.

When moe_fsep_sharding_degree > 1:
  - Replace PrimusTurboGroupedMLP with FSEPGroupedMLP
  - Patch PrimusTurboDeepEPTokenDispatcher to enable FSEP AllGather
    in combine_preprocess
"""

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0


def _is_fsep_enabled(ctx: PatchContext) -> bool:
    args = get_args(ctx)
    return getattr(args, "moe_fsep_sharding_degree", 0) > 1


@register_patch(
    "megatron.moe.fsep_grouped_mlp",
    backend="megatron",
    phase="before_train",
    description="Replace PrimusTurboGroupedMLP with FSEPGroupedMLP for load-balanced expert sharding",
    condition=_is_fsep_enabled,
)
def patch_fsep_grouped_mlp(ctx: PatchContext):
    """
    Patch GroupedMLP to use FSEP (ReduceScatter output aggregation).

    Behavior:
        - Replace GroupedMLP in megatron.core.transformer.moe.experts
          with FSEPGroupedMLP
        - Replace in moe_module_specs (for spec-based model building)
    """
    import megatron.core.transformer.moe.experts as meg_experts
    from megatron.core.models.gpt import moe_module_specs

    from primus.backends.megatron.core.transformer.moe.fsep_experts import (
        FSEPGroupedMLP,
    )

    meg_experts.GroupedMLP = FSEPGroupedMLP
    log_rank_0(
        f"[Patch:megatron.moe.fsep_grouped_mlp]   Patched "
        f"megatron.core.transformer.moe.experts.GroupedMLP -> {FSEPGroupedMLP.__name__}"
    )

    moe_module_specs.GroupedMLP = FSEPGroupedMLP
    log_rank_0(
        f"[Patch:megatron.moe.fsep_grouped_mlp]   Patched "
        f"megatron.core.models.gpt.moe_module_specs.GroupedMLP -> {FSEPGroupedMLP.__name__}"
    )

    args = get_args(ctx)
    log_rank_0(
        f"[Patch:megatron.moe.fsep_grouped_mlp] FSEP enabled: "
        f"sharding_degree={args.moe_fsep_sharding_degree}"
    )
```

### 修改已有文件
`primus/backends/megatron/patches/moe_patches/__init__.py`

```python
# 在 __init__.py docstring 中补充 FSEP patch 说明：
"""
Megatron MoE Patches

This package groups patches for Megatron's Mixture-of-Experts (MoE) components:
    - Deprecated MoE layer implementations
    - Primus TopKRouter
    - MoE permutation fusion with Transformer Engine
    - FSEP (Fully Sharded Expert Parallel) for load-balanced expert computation  # 新增
...
"""
```

### 验收标准
- [ ] `moe_fsep_sharding_degree: 0`（默认）时，无任何行为变化
- [ ] `moe_fsep_sharding_degree: 4`（启用）时，日志显示 patch 生效
- [ ] Patch 与 `megatron.moe.primus_topk_router` 和 `megatron.turbo.moe_dispatcher` 同时启用无冲突

---

## Task 6：单元测试

**工期：3 天**  
**可以在 Task 3 完成后立即并行开始**

### 测试文件
`tests/unit_tests/megatron/transformer/moe/test_fsep_grouped_mlp.py`

### 测试用例设计

**Test 1：数学等价性验证（单进程，无 distributed）**

```python
def test_fsep_math_equivalence():
    """
    Verify FSEPGroupedMLP produces same result as full-weight GroupedMLP
    when S=1 (degenerate case, pass-through).
    """
    # Setup: 1 GPU, 2 experts, H=64, F=256, S=1
    # FSEPGroupedMLP(S=1).forward() == PrimusTurboGroupedMLP.forward()
```

**Test 2：ReduceScatter 正确性（4 GPU / EP=4 / S=4）**

```python
@skip_if_lt_x_gpu(4)
def test_fsep_reduce_scatter_correctness():
    """
    With 4 GPUs, EP=4, S=4:
    - Split weight along F dim across 4 GPUs
    - Run partial GEMM on each GPU
    - ReduceScatter → each GPU gets [T/4, H]
    - AllGather back to [T, H]
    - Compare with single-GPU full-weight GEMM
    """
    # 复用 MoEModelTestContainer 的 setup 模式
    # 参考 test_token_dispatcher.py 的 MultiProcessTestCase 结构
```

**Test 3：反向传播梯度正确性（4 GPU）**

```python
@skip_if_lt_x_gpu(4)
def test_fsep_backward_correctness():
    """
    Verify dW_shard on each GPU, when summed, equals dW from full GEMM.
    Verify d_tokens from FSEP equals d_tokens from full EP.
    """
```

**Test 4：端到端 Dispatcher + FSEP 链路（8 GPU / EP=8 / S=4）**

```python
@skip_if_lt_x_gpu(8)
def test_fsep_full_moe_forward_backward():
    """
    Full MoE layer forward/backward with FSEP enabled.
    dispatch → FSEPGroupedMLP → AllGather → combine
    Numerically matches standard EP (with force_load_balancing for repeatability).
    """
    # 继承并扩展 test_token_dispatcher.py 的 MoEModelTestContainer
```

**Test 5：负载不均衡场景验证**

```python
@skip_if_lt_x_gpu(8)
def test_fsep_load_balance_improvement():
    """
    Construct deliberately imbalanced routing (Expert 0 gets 80% tokens).
    Measure per-GPU compute time with and without FSEP.
    Verify std(compute_time) < 0.1 * mean(compute_time) with FSEP.
    """
```

### 验收标准
- [ ] Test 1~3 在 4 GPU 上 pass（CI 环境）
- [ ] Test 4~5 在 8 GPU 上 pass
- [ ] FP32 误差 < 1e-5，BF16 误差 < 1e-2

---

## Task 7：集成测试 & 性能 Benchmark

**工期：3 天**  
**依赖：Task 1~6 全部完成**

### 7.1 功能集成测试

**测试矩阵：**

| 配置 | EP | S | Top-K | moe_ffn_hidden | 预期 |
|------|----|----|-------|----------------|------|
| DSv3 小规模 | 4 | 2 | 4 | 2048 | 通过 |
| DSv3 小规模 | 8 | 4 | 8 | 2048 | 通过 |
| DSv3 小规模 | 8 | 8 | 8 | 2048 | 通过 |
| + activation_recompute | 8 | 4 | 8 | 2048 | 通过 |
| + FP8 grouped_gemm | 8 | 4 | 8 | 2048 | 通过 |
| + sync_free_moe_stage=2 | 8 | 4 | 8 | 2048 | 通过 |
| + ZB pipeline | 4 | 2 | 4 | 2048 | 通过 |

**测试命令（参考现有 test_megatron_trainer.py 模式）：**

```bash
# 单节点 8 GPU，DSv3 小规模，FSEP S=4
python -m pytest tests/trainer/test_megatron_trainer.py \
  -k test_fsep_dsv3_training \
  --config primus/configs/models/megatron/deepseek_v3.yaml \
  --override moe_fsep_sharding_degree=4 \
  --num-gpus 8
```

### 7.2 性能 Benchmark

**测量目标：**
1. MoE 层单步耗时（不含 PP 通信）
2. 各 GPU 的 Expert 计算时间（标准差 / 均值 < 0.1 为目标）
3. 端到端 step time（目标：比 baseline 降低 20%+）

**Benchmark 脚本位置：** `benchmark/moe/fsep_benchmark.py`

```python
# 核心测量逻辑
def measure_moe_layer_time(config, fsep_degree, num_steps=100):
    """
    Profile MoE layer with nsys / torch.profiler.
    Report:
      - mean/std of per-GPU Expert compute time
      - A2A communication time
      - ReduceScatter/AllGather time (FSEP overhead)
      - total MoE layer time
    """
```

**预期结果记录格式：**

| 指标 | Baseline (EP=8, S=1) | FSEP S=4 | 对比 |
|------|---------------------|----------|------|
| Expert 计算时间 std/mean | ~80% | <10% | 8x 改善 |
| MoE 层总耗时 (ms) | TBD | TBD | 目标 -30% |
| 端到端 step time (ms) | TBD | TBD | 目标 -20% |
| 峰值激活显存 (GB) | TBD | TBD | 目标 -40% |

### 验收标准
- [ ] 所有功能集成测试通过
- [ ] 不均衡场景（r=3）下 MoE 层加速 ≥ 1.5x
- [ ] 均衡场景（force_load_balancing）下 FSEP overhead ≤ 15%（节点内 RS+AG）
- [ ] 无显存泄漏（100 步后显存占用平稳）

---

## Task 8：文档 & Code Review 准备

**工期：1 天**

### 文档更新

**文件 1：** `primus/configs/models/megatron/language_model.yaml` 注释补充

```yaml
# FSEP (Fully Sharded Expert Parallel) - Static Sharding
# Reference: LAER-MoE (ASPLOS '26, arXiv:2602.11686)
#
# When moe_fsep_sharding_degree > 1:
#   - Each Expert's weights are sharded across S=moe_fsep_sharding_degree GPUs
#   - Expert computation is parallelized: each GPU handles T_tokens/S work
#   - Output aggregation: ReduceScatter (intra-node, high bandwidth)
#
# Prerequisites:
#   - enable_primus_turbo: true
#   - use_turbo_deepep: true
#   - expert_tensor_parallel_size must equal moe_fsep_sharding_degree
#   - moe_fsep_sharding_degree must divide expert_model_parallel_size
#
# Recommended value: 4 (for EP=8, r>=2 imbalance scenarios)
moe_fsep_sharding_degree: 0
```

**文件 2：** `primus/backends/megatron/patches/moe_patches/__init__.py` 更新包 docstring

### PR 检查清单

```
Code Quality:
  [ ] 所有新文件有 AMD copyright header
  [ ] 所有公共方法有 docstring
  [ ] 无 print 语句（用 log_rank_0）
  [ ] 无硬编码的 magic number

Test Coverage:
  [ ] Task 6 所有测试通过
  [ ] Task 7 集成测试通过
  [ ] 默认配置（S=0）下无 regression

Performance:
  [ ] Task 7.2 benchmark 数据记录在 PR description
  [ ] 均衡场景 overhead 在可接受范围

Backward Compatibility:
  [ ] moe_fsep_sharding_degree 默认为 0（等价于原有行为）
  [ ] 现有所有 CI 测试通过
```

---

## 里程碑总结

```
Week 1（Day 1~5）：
  Day 1~2  → Task 1: 负载监控上报 + 量化真实 r 值
  Day 3    → Task 2: 配置参数注册
  Day 4~5  → Task 3 Day 1~2: FSEPGroupedMLP 核心 GEMM 逻辑

Week 2（Day 6~10）：
  Day 6~8  → Task 3 Day 3~5: FSEPGroupedMLP 完整实现 + 反向验证
  Day 6~8  → Task 6 Day 1~2: 单测并行开发（可并行）
  Day 9~10 → Task 4 Day 1~2: Dispatcher AllGather 适配

Week 3（Day 11~15）：
  Day 11   → Task 4 Day 3: Dispatcher 完整验证
  Day 12   → Task 5: Patch 注册
  Day 11   → Task 6 Day 3: 单测完成（可并行）
  Day 13~15 → Task 7: 集成测试 + Benchmark

Week 4（Day 16~18，Buffer）：
  Day 16~17 → Bug fix + 性能调优
  Day 18    → Task 8: 文档 + PR 准备
```

---

## 风险与缓解（具体到代码）

| 风险 | 位置 | 缓解 |
|------|------|------|
| T 不能被 S 整除 | `FSEPGroupedMLP._fsep_reduce_scatter()` | 在 assert 前加 padding：`T_padded = ((T + S - 1) // S) * S`，在 dispatcher dispatch_postprocess 中对 tokens 补零 |
| AllGather 引入显存峰值 | `PrimusTurboDeepEPTokenDispatcher.combine_preprocess()` | 用 `dist.all_gather_into_tensor`（in-place 输出 buffer，可预分配）；在 combine_postprocess 后立即释放 |
| FP8 GroupedGEMM 输出类型与 ReduceScatter 不兼容 | `_forward_no_reduce()` | FP8 GEMM 输出转 BF16 再做 ReduceScatter：`partial_output = partial_output.to(torch.bfloat16)` |
| `use_split_wgrad_op` 路径下梯度存储时序 | `_forward_no_reduce()` 末尾 | ReduceScatter 必须在 `activation_checkpoint.discard_output_and_register_recompute` 之后执行，确保 dW 延迟计算队列已注册 |
| Expert TP ≠ FSEP sharding degree | `get_fsep_group()` | 初期强制 `moe_fsep_sharding_degree == expert_tensor_parallel_size`，通过 Task 2 的参数校验拦截 |

---

*计划制定于 2026-03-09 | Branch: feat/moe-fsep-static-sharding | 基于 Primus Megatron backend 代码精读*
