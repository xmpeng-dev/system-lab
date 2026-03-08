# Axion 整体规划：长期系统路线图

> **定位:** AMD AI Infra 员工的个人长期技术投入，以现有系统（Primus/Megatron）为验证平台  
> **版本:** v0.3 | 2026-03-08  
> **核心策略:** Feature-by-Feature 在现有系统验证 → 有收益才继续 → 最终重写 Axion  
> **时间跨度:** 不设硬性终点，由每个 Feature 的收益数据驱动决策

---

## 0. 核心思路：验证优先，Axion 是终点不是起点

### 0.1 为什么不直接开发 Axion

```
错误路径（之前的计划）：
  直接开发 Axion 全套系统
  → 18 个月后才能看到收益
  → 如果某个技术点不 work，整个投入打水漂
  → 个人业余项目，无法承受这个风险

正确路径（本文档）：
  在 Primus/Megatron 上逐个验证技术点（Feature）
  每个 Feature 独立可交付，有明确的收益指标
  
  Feature 通过验证（有收益）→ 继续下一个 Feature
  Feature 未通过验证（无收益）→ 停止，不浪费更多时间
  
  所有核心 Feature 验证完成 → 用这些已验证的设计重写 Axion
  Axion 不是起点，是终点：一个已经被数据证明的设计的干净实现
```

### 0.2 整体结构

```
验证阶段（Feature 0 ~ Feature N）：
  在 Primus/Megatron 上以 patch/hook/plugin 形式实现
  每个 Feature 有独立的：
    - 实现方案（最小改动接入）
    - 收益假设（预期提升 X%）
    - 验证实验（实测数据）
    - 决策门（通过 → 继续，不通过 → 停止或调整）
    - 技术产出（内部报告 / 博客 / 论文）

  ┌──────────────────────────────────────────────────────┐
  │  Feature 0：通信可观测性（CommProfiler）              │
  │  Feature 1：路由负载均衡（FastRouter）               │
  │  Feature 2：Expert 物理迁移（SlowPlanner）           │
  │  Feature 3：静态 Overlap 调度（OverlapScheduler）    │
  │  Feature 4：CommTensor zero-copy                    │
  │  Feature 5：...（由前序数据驱动）                    │
  └──────────────────────────────────────────────────────┘
                          │
              每个 Feature 通过验证
                          │
                          ▼
  构建阶段（Axion 重写）：
    用已验证的设计，在干净的 Axion IR + Pass 框架中重实现
    此时 Axion 的每个设计决策都有数据背书
    不是"希望它 work"，而是"已经证明它 work"

```

### 0.3 验证 vs 构建的判断标准

```
什么时候从"验证阶段"进入"构建阶段"（开始 Axion 重写）？

触发条件（需要同时满足）：
  □ ≥ 3 个 Feature 已通过验证，且累计收益 ≥ 20%
  □ 这些 Feature 在 Primus/Megatron 上的 patch 开始出现相互耦合
    （说明继续在现有系统上叠加已经比重写更麻烦了）
  □ 有足够时间投入（不再是业余项目，或获得了内部支持）

如果上述条件始终不满足：
  → Axion 的"验证阶段"成果本身就足够有价值
  → 这些 Feature 作为 Primus/Megatron 的改进 patch 独立存在
  → 不强求进入构建阶段
```

---

## 1. Feature 0：通信可观测性（CommProfiler）

### 背景与假设

```
假设：MoE 训练中存在显著的 Expert 负载不均衡和通信瓶颈，
      但工程师目前无法直接看到这些问题，导致优化方向不清晰。

验证目标：在 MI300X 上跑 Megatron-LM（megatron-core MoE），
          用 CommProfiler 量化以下数据：
            - Expert 负载不均衡系数（max_load / avg_load，逐层）
            - A2A 时间占总 step time 的比例
            - 当前 Overlap 率 vs 理论上界
            - 跨节点 A2A 占比（intra-node NVLink vs inter-node RCCL）

这些数据决定后续所有 Feature 的优先级。
```

---

### Megatron-LM MoE 代码结构（Hook 接入点分析）

```
megatron-core 的 MoE 调用栈（关键路径）：

megatron/core/transformer/transformer_layer.py
  └─ TransformerLayer.forward()
       └─ self.mlp(hidden_states)           ← MoE 层入口（对 TransformerLayer 来说就是 MLP）

megatron/core/transformer/moe/moe_layer.py
  └─ MoELayer.forward(hidden_states)
       ├─ [1] self.router(hidden_states)    ← TopKRouter，计算 routing weights + indices
       └─ [2] self.experts(hidden_states, ...) ← dispatch → expert compute → combine

megatron/core/transformer/moe/experts.py
  └─ GroupedMLP.forward() / SequentialMLP.forward()
       ← 本地 Expert 的实际计算

megatron/core/transformer/moe/token_dispatcher.py
  └─ MoEAlltoAllTokenDispatcher
       ├─ [3] dispatch(hidden_states, max_prob, top_indices)
       │       ├─ preprocess(indices)         ← 计算每个 expert 的 token 数（负载统计点）
       │       ├─ alltoall(hidden_states)     ← A2A dispatch（计时点 1）
       │       └─ 返回 dispatched_input
       └─ [4] combine(expert_output, ...)
               ├─ alltoall(expert_output)     ← A2A gather（计时点 2）
               └─ 返回 combined_output

megatron/core/transformer/moe/router.py
  └─ TopKRouter.forward()
       └─ [5] routing_map, probs             ← 路由结果（负载分析的原始数据）

Hook 接入策略：
  - 在 MoELayer.forward() 前后挂 hook → 测量 MoE 层总耗时
  - 在 MoEAlltoAllTokenDispatcher.dispatch/combine 前后挂 hook → 测量 A2A 时间
  - 在 TopKRouter.forward() 后拦截 routing_map → 统计 Expert 负载分布
  - 不修改任何 forward 计算逻辑，纯观测
```

---

### 实现方案（最小侵入，基于 Megatron-LM）

#### 1.1 Hook 注册器

```python
# axion/profiler/comm_profiler.py

import time
import torch
import torch.distributed as dist
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from contextlib import contextmanager

# ── 依赖 megatron-core 的导入（运行时动态导入，避免 import 失败）
def _import_megatron_moe():
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
    from megatron.core.transformer.moe.router import TopKRouter
    return MoELayer, MoEAlltoAllTokenDispatcher, TopKRouter


@dataclass
class StepStats:
    """单个 training step 的原始统计数据"""
    layer_idx: int
    step_idx: int
    # Expert 负载
    tokens_per_expert: torch.Tensor        # shape: [num_experts]，每个 expert 收到的 token 数
    # A2A 时间（wall-clock，单位 ms）
    dispatch_a2a_ms: float = 0.0           # dispatch 阶段 A2A 耗时
    combine_a2a_ms: float = 0.0            # combine 阶段 A2A 耗时
    # MoE 层总时间
    moe_layer_ms: float = 0.0             # 整个 MoE 层耗时（含 A2A + expert compute）
    # 通信量（字节）
    dispatch_bytes: int = 0               # dispatch A2A 的实际通信量
    combine_bytes: int = 0                # combine A2A 的实际通信量


class AxionCommProfiler:
    """
    零侵入挂载在 Megatron-LM MoE 层上的通信 Profiler。

    使用方式：
        profiler = AxionCommProfiler(num_warmup_steps=5)
        profiler.attach(model)
        # 正常跑训练...
        report = profiler.report()
        report.save_html("comm_report.html")
    """

    def __init__(
        self,
        num_warmup_steps: int = 5,       # 前 N 步 warmup，不计入统计
        profile_steps: int = 20,          # 采样 N 步后停止（避免长期开销）
        enabled: bool = True,
    ):
        self.num_warmup_steps = num_warmup_steps
        self.profile_steps = profile_steps
        self.enabled = enabled
        self._step_idx = 0
        self._stats: List[StepStats] = []
        self._hooks = []                  # 保存所有 hook handle，方便 detach
        self._layer_map: Dict[int, int] = {}   # module id → layer_idx

    # ──────────────────────────────────────────────
    # 1. 挂载
    # ──────────────────────────────────────────────

    def attach(self, model: torch.nn.Module) -> "AxionCommProfiler":
        """
        扫描 model 中所有 MoELayer，注册 hook。
        不修改模型参数和 forward 逻辑。
        """
        if not self.enabled:
            return self

        MoELayer, MoEAlltoAllTokenDispatcher, TopKRouter = _import_megatron_moe()

        layer_idx = 0
        for name, module in model.named_modules():
            if isinstance(module, MoELayer):
                self._register_moe_layer_hooks(module, layer_idx)
                layer_idx += 1

            elif isinstance(module, MoEAlltoAllTokenDispatcher):
                self._register_dispatcher_hooks(module, layer_idx - 1)

            elif isinstance(module, TopKRouter):
                self._register_router_hooks(module, layer_idx - 1)

        print(f"[AxionCommProfiler] Attached to {layer_idx} MoE layers.")
        return self

    def detach(self):
        """移除所有 hook，恢复原始模型"""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ──────────────────────────────────────────────
    # 2. Hook 实现
    # ──────────────────────────────────────────────

    def _register_moe_layer_hooks(self, module, layer_idx: int):
        """MoELayer 前后：测量整个 MoE 层的 wall-clock 时间"""
        _state = {}

        def pre_hook(mod, inputs):
            if not self._is_profiling():
                return
            # ROCm/CUDA 都支持 synchronize 后计时
            torch.cuda.synchronize()
            _state['t0'] = time.perf_counter()

        def post_hook(mod, inputs, outputs):
            if not self._is_profiling():
                return
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - _state.get('t0', 0)) * 1000
            self._get_current_step_stats(layer_idx).moe_layer_ms = elapsed_ms
            # 每个 MoE 层 post_hook 触发时，认为一个 layer-step 完成
            # 用第一层来驱动 step 计数
            if layer_idx == 0:
                self._step_idx += 1

        h1 = module.register_forward_pre_hook(pre_hook)
        h2 = module.register_forward_hook(post_hook)
        self._hooks.extend([h1, h2])

    def _register_dispatcher_hooks(self, module, layer_idx: int):
        """
        MoEAlltoAllTokenDispatcher：
          - dispatch() 前后：计量 A2A dispatch 时间和通信量
          - combine() 前后：计量 A2A combine 时间和通信量

        注意：megatron-core 的 dispatcher 没有单独的 forward()，
        而是分 dispatch() / combine() 两个方法调用。
        需要用 __call__ wrapper 方式，或直接 monkey-patch 方法。
        """
        _orig_dispatch = module.dispatch.__func__ if hasattr(module.dispatch, '__func__') else None
        _orig_combine  = module.combine.__func__  if hasattr(module.combine,  '__func__') else None

        profiler_ref = self  # 避免闭包问题

        def patched_dispatch(self_mod, hidden_states, max_prob, top_indices):
            if not profiler_ref._is_profiling():
                return _orig_dispatch(self_mod, hidden_states, max_prob, top_indices)

            # 计算通信量估算：每个 token 的 hidden_states 大小 × 发送 token 数
            dispatch_bytes = hidden_states.numel() * hidden_states.element_size()

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = _orig_dispatch(self_mod, hidden_states, max_prob, top_indices)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000

            stats = profiler_ref._get_current_step_stats(layer_idx)
            stats.dispatch_a2a_ms = elapsed_ms
            stats.dispatch_bytes  = dispatch_bytes
            return result

        def patched_combine(self_mod, expert_output, *args, **kwargs):
            if not profiler_ref._is_profiling():
                return _orig_combine(self_mod, expert_output, *args, **kwargs)

            combine_bytes = expert_output.numel() * expert_output.element_size()

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = _orig_combine(self_mod, expert_output, *args, **kwargs)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000

            stats = profiler_ref._get_current_step_stats(layer_idx)
            stats.combine_a2a_ms = elapsed_ms
            stats.combine_bytes  = combine_bytes
            return result

        import types
        module.dispatch = types.MethodType(patched_dispatch, module)
        module.combine  = types.MethodType(patched_combine,  module)
        # monkey-patch 不走 hook handle，用 sentinel 记录以便 detach 时恢复
        # 简化版：不恢复（profiling 结束后调用 detach 即可）

    def _register_router_hooks(self, module, layer_idx: int):
        """
        TopKRouter：在 forward 后拦截 routing_map。
        megatron-core TopKRouter.forward() 返回 (scores, indices)，
        indices shape = [seq_len * batch, top_k]，值为 expert_id。
        """
        profiler_ref = self

        def post_hook(mod, inputs, outputs):
            if not profiler_ref._is_profiling():
                return
            # outputs = (probs, routing_map)
            # routing_map: [tokens, top_k] → expert indices
            if isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
                routing_map = outputs[1]   # [num_tokens, top_k]
                if routing_map is not None and routing_map.dim() == 2:
                    num_experts = mod.config.num_moe_experts
                    # 统计每个 expert 收到的 token 数
                    tokens_per_expert = torch.bincount(
                        routing_map.flatten().long(),
                        minlength=num_experts
                    ).float().cpu()
                    stats = profiler_ref._get_current_step_stats(layer_idx)
                    stats.tokens_per_expert = tokens_per_expert

        h = module.register_forward_hook(post_hook)
        self._hooks.append(h)

    # ──────────────────────────────────────────────
    # 3. 辅助方法
    # ──────────────────────────────────────────────

    def _is_profiling(self) -> bool:
        """只在 warmup 结束后、profile_steps 步内采样"""
        return (self.enabled and
                self._step_idx >= self.num_warmup_steps and
                self._step_idx < self.num_warmup_steps + self.profile_steps)

    def _get_current_step_stats(self, layer_idx: int) -> StepStats:
        """懒创建当前 step 的 StepStats"""
        key = (self._step_idx, layer_idx)
        for s in self._stats:
            if s.step_idx == self._step_idx and s.layer_idx == layer_idx:
                return s
        s = StepStats(
            layer_idx=layer_idx,
            step_idx=self._step_idx,
            tokens_per_expert=torch.zeros(1),
        )
        self._stats.append(s)
        return s

    # ──────────────────────────────────────────────
    # 4. 生成报告
    # ──────────────────────────────────────────────

    def report(self) -> "CommReport":
        return CommReport.from_stats(self._stats)
```

#### 1.2 CommReport 数据结构

```python
# axion/profiler/comm_report.py

import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class LayerReport:
    layer_idx: int
    # Expert 负载
    mean_tokens_per_expert: float
    max_tokens_per_expert: float
    load_imbalance_ratio: float        # max / mean，理想值 = 1.0
    hot_experts: List[int]             # 负载 top-5 的 expert id
    # A2A 时间
    dispatch_a2a_ms_mean: float
    combine_a2a_ms_mean: float
    a2a_total_ms_mean: float           # dispatch + combine
    moe_layer_ms_mean: float
    a2a_fraction: float                # a2a_total / moe_layer，核心指标
    # 通信量
    dispatch_gbps: float               # 实测 dispatch 带宽（GB/s）
    combine_gbps: float


@dataclass
class CommReport:
    num_layers: int
    num_profiled_steps: int
    layers: List[LayerReport] = field(default_factory=list)

    # 全局汇总
    @property
    def global_load_imbalance(self) -> float:
        """所有层的平均负载不均衡系数"""
        if not self.layers:
            return 0.0
        return np.mean([l.load_imbalance_ratio for l in self.layers])

    @property
    def global_a2a_fraction(self) -> float:
        """所有层平均 A2A 时间占比"""
        if not self.layers:
            return 0.0
        return np.mean([l.a2a_fraction for l in self.layers])

    @classmethod
    def from_stats(cls, stats: List) -> "CommReport":
        from collections import defaultdict
        import numpy as np

        # 按 layer_idx 聚合
        layer_data = defaultdict(list)
        for s in stats:
            layer_data[s.layer_idx].append(s)

        num_layers = len(layer_data)
        num_steps  = max((len(v) for v in layer_data.values()), default=0)

        layers = []
        for layer_idx in sorted(layer_data.keys()):
            step_list = layer_data[layer_idx]

            # Expert 负载统计（多步平均）
            all_tpe = np.stack([s.tokens_per_expert.numpy() for s in step_list])  # [steps, experts]
            mean_tpe = all_tpe.mean(axis=0)   # 每个 expert 的平均 token 数
            max_tpe  = mean_tpe.max()
            mean_val = mean_tpe.mean()
            imbalance = float(max_tpe / mean_val) if mean_val > 0 else 1.0
            hot_experts = list(np.argsort(mean_tpe)[::-1][:5])

            # A2A 时间统计
            d_ms   = np.mean([s.dispatch_a2a_ms for s in step_list])
            c_ms   = np.mean([s.combine_a2a_ms  for s in step_list])
            moe_ms = np.mean([s.moe_layer_ms    for s in step_list])
            a2a_ms = d_ms + c_ms
            a2a_frac = float(a2a_ms / moe_ms) if moe_ms > 0 else 0.0

            # 带宽估算
            d_bytes = np.mean([s.dispatch_bytes for s in step_list])
            c_bytes = np.mean([s.combine_bytes  for s in step_list])
            d_gbps  = float(d_bytes / (d_ms * 1e-3) / 1e9) if d_ms > 0 else 0.0
            c_gbps  = float(c_bytes / (c_ms * 1e-3) / 1e9) if c_ms > 0 else 0.0

            layers.append(LayerReport(
                layer_idx              = layer_idx,
                mean_tokens_per_expert = float(mean_val),
                max_tokens_per_expert  = float(max_tpe),
                load_imbalance_ratio   = imbalance,
                hot_experts            = hot_experts,
                dispatch_a2a_ms_mean   = float(d_ms),
                combine_a2a_ms_mean    = float(c_ms),
                a2a_total_ms_mean      = float(a2a_ms),
                moe_layer_ms_mean      = float(moe_ms),
                a2a_fraction           = a2a_frac,
                dispatch_gbps          = d_gbps,
                combine_gbps           = c_gbps,
            ))

        report = cls(num_layers=num_layers, num_profiled_steps=num_steps, layers=layers)
        report._print_summary()
        return report

    def _print_summary(self):
        """rank 0 打印摘要"""
        import torch.distributed as dist
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        print("\n" + "="*60)
        print("  AxionCommProfiler Report")
        print("="*60)
        print(f"  Layers profiled     : {self.num_layers}")
        print(f"  Steps profiled      : {self.num_profiled_steps}")
        print(f"  Global load imbalance (max/mean): {self.global_load_imbalance:.2f}x")
        print(f"  Global A2A fraction : {self.global_a2a_fraction*100:.1f}%")
        print("-"*60)
        for l in self.layers:
            print(f"  Layer {l.layer_idx:2d} | "
                  f"imbalance={l.load_imbalance_ratio:.2f}x | "
                  f"A2A={l.a2a_total_ms_mean:.1f}ms ({l.a2a_fraction*100:.0f}%) | "
                  f"hot_experts={l.hot_experts[:3]}")
        print("="*60 + "\n")

    def save_json(self, path: str):
        import dataclasses
        with open(path, "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=2)

    def save_html(self, path: str):
        """生成 Expert 热力图 + A2A 时序图（基于 Plotly）"""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            print("[CommReport] plotly not installed, skipping HTML report")
            return

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Expert Load Imbalance (per layer)",
                "A2A Time Fraction (per layer)",
                "Dispatch vs Combine A2A (ms)",
                "Effective A2A Bandwidth (GB/s)",
            ]
        )

        layer_ids = [l.layer_idx for l in self.layers]

        # 1. 负载不均衡系数
        fig.add_trace(go.Bar(
            x=layer_ids,
            y=[l.load_imbalance_ratio for l in self.layers],
            name="Load Imbalance (max/mean)",
            marker_color="crimson",
        ), row=1, col=1)

        # 2. A2A 占比
        fig.add_trace(go.Bar(
            x=layer_ids,
            y=[l.a2a_fraction * 100 for l in self.layers],
            name="A2A Fraction (%)",
            marker_color="royalblue",
        ), row=1, col=2)

        # 3. Dispatch vs Combine
        fig.add_trace(go.Bar(
            x=layer_ids,
            y=[l.dispatch_a2a_ms_mean for l in self.layers],
            name="Dispatch A2A (ms)", marker_color="orange",
        ), row=2, col=1)
        fig.add_trace(go.Bar(
            x=layer_ids,
            y=[l.combine_a2a_ms_mean for l in self.layers],
            name="Combine A2A (ms)", marker_color="green",
        ), row=2, col=1)

        # 4. 带宽
        fig.add_trace(go.Scatter(
            x=layer_ids,
            y=[l.dispatch_gbps for l in self.layers],
            name="Dispatch BW (GB/s)", mode="lines+markers",
        ), row=2, col=2)
        fig.add_trace(go.Scatter(
            x=layer_ids,
            y=[l.combine_gbps for l in self.layers],
            name="Combine BW (GB/s)", mode="lines+markers",
        ), row=2, col=2)

        fig.update_layout(title="AxionCommProfiler: MoE Communication Analysis", height=800)
        fig.write_html(path)
        print(f"[CommReport] HTML report saved to {path}")
```

#### 1.3 使用方式（接入 Megatron-LM 训练脚本）

```python
# 在 pretrain_gpt.py 或任意 Megatron 训练脚本中，只需加 3 行

from axion.profiler import AxionCommProfiler

# Step 1：创建 profiler（在 model 构建后）
profiler = AxionCommProfiler(num_warmup_steps=5, profile_steps=20)
profiler.attach(model)   # ← 挂 hook，不改模型

# Step 2：正常跑训练（profiler 自动在 warmup 后开始采样）
for step in range(num_steps):
    train_step(model, optimizer, ...)

# Step 3：在训练结束或 profile_steps 采集完后生成报告
report = profiler.report()
if torch.distributed.get_rank() == 0:
    report.save_html("mi300x_moe_comm_report.html")
    report.save_json("mi300x_moe_comm_report.json")

profiler.detach()  # 移除所有 hook，恢复模型原始状态
```

---

### 关键设计决策

```
1. 为什么用 monkey-patch 而不是 register_forward_hook 接入 dispatcher？

   megatron-core 的 MoEAlltoAllTokenDispatcher 不是 nn.Module 的标准调用路径，
   其 dispatch() / combine() 是被 MoELayer 直接调用的方法，
   不走 nn.Module.forward()，所以 register_forward_hook 无法捕获。
   → 用 types.MethodType monkey-patch 替换方法，是最低侵入的方式。

2. 为什么不用 torch.profiler？

   torch.profiler / rocprof 能给出 kernel 级别的时间，但：
   - 无法直接关联到"哪个 MoE 层"、"哪次 A2A"
   - 无法同步 Expert 负载（routing_map 不在 profiler 里）
   - 开销大，不适合长期采样
   → AxionCommProfiler 专注于 MoE 语义级别的指标，与 rocprof 互补。

3. 为什么要 torch.cuda.synchronize()？

   MI300X 上的 RCCL A2A 是异步启动的，不 synchronize 直接计时会测到
   "kernel 提交时间"而不是"A2A 完成时间"。
   synchronize 会引入额外延迟（~0.1ms），但保证计时准确。
   → profiling 开销估算：2 次 sync/A2A × 2 A2A/layer × N layers ≈ 可接受。

4. 分布式场景下的数据汇总

   routing_map 在每个 rank 上只有本地的 token，
   Expert 负载统计需要 all_reduce 后才能得到全局视图。
   → 当前版本：各 rank 独立统计本地负载，CommReport 只反映本地视角。
   → 后续扩展：在 report() 时做一次 dist.all_reduce 汇总全局数据。
```

---

### 收益假设与验证

```
预期：能生成 CommReport，可视化 MI300X 上的通信热点
量化指标：
  □ CommReport 数据与 rocprof 手动 profiling 误差 < 10%
    （A2A 时间的误差主要来源：synchronize 开销 + timing 分辨率）
  □ 接入开销 < 1%（profiling 期间的额外 synchronize 不超过 step time 的 1%）
  □ 仅采样 20 步，不影响长期训练

这个 Feature 没有"吞吐提升"，收益是"信息价值"——
  通过数据确认：负载不均衡是否真实？A2A 是否是瓶颈？
  这直接决定 Feature 1~4 是否值得做。
```

---

### 决策门

```
Feature 0 完成后，根据 CommReport 数据做出判断：

  if global_load_imbalance < 1.3x：
    → 负载均衡收益有限，Feature 1/2 优先级降低
    → 优先看 A2A 时间占比，决定是否做 Feature 3/4

  if global_load_imbalance ≥ 2.0x：
    → Feature 1（FastRouter）和 Feature 2（SlowPlanner）高优
    → 立即开始 Feature 1

  if global_a2a_fraction < 10%：
    → A2A 不是瓶颈，Feature 3/4 的 overlap 和 zero-copy 收益有限
    → 重新评估整个路线，考虑 Expert compute 优化（Feature X）

  if global_a2a_fraction ≥ 20%：
    → Feature 3（OverlapScheduler）和 Feature 4（CommTensor）高优

  if dispatch_gbps << NVLink 理论带宽（MI300X 节点内 ~1.6 TB/s）：
    → A2A 有大量跨节点流量，Feature 2（Expert 物理迁移）有价值
```

---

### 技术产出

```
□ axion/profiler/comm_profiler.py   （核心实现，~300 行）
□ axion/profiler/comm_report.py     （报告生成 + HTML 可视化，~150 行）
□ examples/megatron_profile_example.py  （接入 Megatron pretrain 的示例）
□ CommReport HTML 可视化：
    - Expert 负载热力图（各层 × 各 Expert 的 token 数热图）
    - A2A 时间条形图（dispatch vs combine，逐层）
    - 带宽利用率折线图
□ 内部技术报告：MI300X MoE 训练通信瓶颈分析
  （Feature 1~4 的立项依据，包含实测数据图表）
□ 可选：AMD 技术博客 "Profiling MoE Communication on MI300X with Zero-Overhead Hooks"
```

---

### 时间估计

```
总计：2~3 周（全职）/ 4~6 周（业余）

Week 1：
  □ 熟悉 megatron-core MoELayer / TokenDispatcher / TopKRouter 代码
  □ 实现 hook 注册（MoELayer + TopKRouter）
  □ 实现 monkey-patch dispatcher
  □ 本地单卡验证 StepStats 收集

Week 2：
  □ 实现 CommReport.from_stats()
  □ 多卡分布式验证（8 GPU，MI300X 节点内）
  □ 对比 rocprof 验证误差
  □ 实现 HTML 可视化

Week 3：
  □ 跨节点测试（多节点 EP 场景）
  □ 文档和 example 脚本
  □ 内部技术报告初稿
```

---

## 2. Feature 1：路由负载均衡（FastRouter）

### 背景与假设

```
假设：通过调整 gate logits，可以软性引导 token 远离过载 Expert，
      在不改变 Expert 物理位置的情况下减轻负载不均衡。

前提：Feature 0 CommReport 显示负载不均衡系数 ≥ 1.5x

实现方式：在 Primus/Megatron 的 MoE Gate 模块，
          在 softmax/topk 之前插入一行偏置调整：
          gate_logits -= α * load_penalty
```

### 实现方案

```python
# 接入方式：在现有 MoE Gate forward 中插入一行
# Primus/Megatron 的 MoE Gate 通常是：
#   scores = gate_logits.softmax(-1)
#   topk_scores, topk_indices = scores.topk(k)
#
# 改为：
#   gate_logits = self.fast_router.adjust(gate_logits)  # ← 只加这一行
#   scores = gate_logits.softmax(-1)
#   topk_scores, topk_indices = scores.topk(k)

class FastRouter:
    def __init__(self, alpha=0.1, beta=2.0, ema_decay=0.9):
        self.alpha = alpha
        self.beta  = beta
        self.load_ema = None  # 指数移动平均，平滑负载统计

    def adjust(self, gate_logits: Tensor) -> Tensor:
        if self.load_ema is None:
            return gate_logits  # 第一个 step 无统计，跳过
        penalty = (self.load_ema / self.load_ema.mean()) ** self.beta
        return gate_logits - self.alpha * penalty.log()

    @torch.no_grad()
    def update(self, expert_counts: Tensor):
        """每个 step dispatch 后更新负载统计"""
        if self.load_ema is None:
            self.load_ema = expert_counts.float()
        else:
            self.load_ema = self.ema_decay * self.load_ema \
                          + (1 - self.ema_decay) * expert_counts.float()
```

### 收益假设与验证

```
预期吞吐提升：5~15%（通过减少过载 GPU 的等待时间）

必做实验（按顺序）：

  实验 A：收敛性验证（红线）
    设置：内部 2B MoE，1000 steps
    对比：Baseline vs FastRouter (α=0.1, β=2.0)
    指标：loss curve、final perplexity
    红线：loss 差异 > 1% → 停止，不做任何生产部署

  实验 B：负载均衡效果
    设置：CommProfiler 在启用 FastRouter 前后各跑 100 steps
    指标：Expert 负载不均衡系数变化（预期从 Xb 降到 Xa）

  实验 C：吞吐测量
    设置：相同模型相同数据，64 MI300X
    指标：tok/s 提升百分比
```

### 决策门

```
if 实验 A 红线触发（loss 差异 > 1%）：
  → 停止 FastRouter，直接跳 Feature 2（SlowPlanner，物理迁移）
  → 物理迁移不改路由语义，收敛风险更低

if 实验 C 吞吐提升 < 3%：
  → FastRouter 收益不显著（负载均衡不是主要瓶颈）
  → 查看 CommReport：A2A 时间占比是否更高？
  → 如果 A2A 占比高 → 跳到 Feature 3（Overlap 调度）

if 实验 C 吞吐提升 ≥ 5%：
  → 继续 Feature 2（叠加 SlowPlanner 物理迁移）
```

### 技术产出

```
□ FastRouter patch（Primus/Megatron PR 或内部 patch）
□ 收敛实验报告（内部文档）
□ 超参分析：α, β 对均衡效果和收敛的 trade-off
□ 可选：结合 Feature 0 数据，整合进一篇 paper（通信优化 + 可观测性）
```

### 时间估计
**实现：1 周 | 收敛实验：2 周 | 总计 3~4 周**（业余 6~8 周）

---

## 3. Feature 2：Expert 物理迁移（SlowPlanner）

### 背景与假设

```
假设：每隔 K 个 step，根据历史路由统计，将过载 Expert 的参数
      迁移到负载较轻的 GPU，从根本上消除负载不均衡。

前提：Feature 0 CommReport 显示负载不均衡系数 ≥ 1.5x
      （Feature 1 可选，SlowPlanner 可以单独做）

参考：LAER-MoE 论文的核心方案，已有公开实现可参考
      (https://github.com/PKUDAIR/Hetu-Galvatron/tree/laer-moe)
```

### 实现方案

```python
# 接入方式：在 Primus/Megatron 的训练循环中增加 hook
#
# 原始训练循环：
#   for step in range(max_steps):
#       loss = model(batch)
#       loss.backward()
#       optimizer.step()
#
# 增加 SlowPlanner：
#   for step in range(max_steps):
#       loss = model(batch)
#       loss.backward()
#       optimizer.step()
#       planner.maybe_migrate(step, model)  # ← 只加这一行

class SlowPlanner:
    def __init__(self, check_interval=50, imbalance_threshold=1.3):
        self.check_interval      = check_interval
        self.imbalance_threshold = imbalance_threshold
        self.load_history        = []

    def maybe_migrate(self, step, model):
        self.load_history.append(collect_expert_loads(model))

        if step % self.check_interval != 0:
            return

        imbalance = compute_imbalance(self.load_history[-self.check_interval:])
        if imbalance < self.imbalance_threshold:
            return  # 不需要迁移

        plan = self._greedy_plan(self.load_history)
        self._execute_migration(model, plan)  # 异步 P2P（与下一个 step 重叠）

    def _greedy_plan(self, history):
        """贪心：把热点 Expert 迁移到冷点 GPU"""
        # 简单贪心，不需要 ILP
        ...

    def _execute_migration(self, model, plan):
        """
        Primus/Megatron 中：直接用 dist.isend/irecv 做异步 P2P
        MI300X：走 Infinity Fabric（节点内高带宽）
        """
        for src_rank, dst_rank, expert_param in plan:
            dist.isend(expert_param, dst=dst_rank)  # 异步，不阻塞训练
```

### 收益假设与验证

```
预期吞吐提升：10~25%（在 FastRouter 基础上额外，或单独使用）

必做实验：

  实验 A：收敛性验证（红线）
    设置：内部 7B MoE，2000 steps
    对比：Baseline vs SlowPlanner
    指标：loss curve，关注迁移时刻是否有 loss spike
    红线：loss spike 幅度 > 5% of moving average → 收紧触发条件

  实验 B：迁移开销 vs 收益
    指标：单次迁移耗时（MI300X P2P 带宽实测）
          迁移后 step time 降低持续时间
          ROI = 节省计算时间 / 迁移通信时间（预期 > 5x）

  实验 C：与 Feature 1 叠加效果
    对比：Baseline / FastRouter only / SlowPlanner only / 两者叠加
    指标：吞吐、负载均衡系数

  实验 D：规模效果
    设置：8 / 16 / 32 / 64 MI300X
    指标：提升比例是否随规模增大（不均衡问题在大规模下更严重）
```

### 决策门

```
if 实验 A 出现持续 loss spike（不可接受）：
  → 提高 imbalance_threshold（更保守触发）
  → 降低 max_experts_per_migration（每次迁更少）
  → 如果问题依然存在，停止 SlowPlanner
  → 转向 Feature 3（Overlap 调度，不改物理分布，无收敛风险）

if 实验 B ROI < 2x（迁移成本太高）：
  → 减少迁移频率（check_interval 从 50 增加到 100）
  → 或限制到节点内迁移（走 XGMI，不走 ROCEv2）

if 实验 C/D 吞吐提升 ≥ 10%：
  → Feature 1+2 组合有明确收益
  → 继续 Feature 3（Overlap 调度叠加）
```

### 技术产出

```
□ SlowPlanner patch（Primus/Megatron，最小改动）
□ 收敛实验报告 + 迁移 ROI 分析
□ MI300X P2P 带宽实测数据（Infinity Fabric vs ROCEv2）
□ 论文方向：Feature 0+1+2 数据足够支撑一篇 paper
  "Load-Adaptive Expert Parallelism on MI300X"
  对比 LAER-MoE（A100），突出 MI300X 的独特优势
```

### 时间估计
**实现：2~3 周 | 实验：3 周 | 总计 5~6 周**（业余 10~12 周）

---

## 4. Feature 3：静态 Overlap 调度（OverlapScheduler）

### 背景与假设

```
假设：目前 Primus/Megatron 中 A2A 和 Expert FFN 的 overlap 不充分，
      存在通信等待计算或计算等待通信的空泡。
      通过静态分析通信/计算的依赖关系，可以精确找到所有安全的 overlap 点，
      显著提升 GPU 利用率。

前提：Feature 0 CommReport 显示实际 overlap 率 < 理论上界 × 80%
      （说明当前调度有优化空间）

参考：FlowMoE（arXiv:2510.00207）的流水线调度思路，
      但 Axion 的设计是静态生成，FlowMoE 是动态的
```

### 实现方案

```python
# 接入方式：替换 Primus/Megatron 的 MoE dispatch/combine 调用方式
#
# 原始（串行）：
#   dispatched = all_to_all(tokens, routing)      # 阻塞等待
#   output = expert_ffn(dispatched, experts)
#   combined = all_to_all(output, routing)        # 阻塞等待
#
# OverlapScheduler（流水线，分 chunk）：
#   with overlap_scheduler as sched:
#       for i, chunk in enumerate(split_chunks(tokens, N_CHUNKS)):
#           dispatched_i = sched.async_a2a(chunk, routing)     # 非阻塞
#           if i > 0:
#               output_i_minus_1 = expert_ffn(dispatched[i-1]) # 与 A2A 重叠
#           dispatched.append(dispatched_i.wait())

class OverlapScheduler:
    """
    静态 chunk 流水线调度器。
    直接在 Primus/Megatron 的 MoE 前向中使用，
    不需要改 IR 或 Pass 系统。
    """
    def __init__(self, num_chunks=4, stream_mode='rccl'):
        self.num_chunks = num_chunks
        self.compute_stream = torch.cuda.Stream()
        self.comm_stream    = torch.cuda.Stream()

    def dispatch_with_overlap(self, tokens, routing, expert_fn):
        chunks = tokens.chunk(self.num_chunks, dim=0)
        results = []
        pending_comm = None

        for i, chunk in enumerate(chunks):
            # 发出本 chunk 的 A2A（在 comm stream 上，非阻塞）
            with torch.cuda.stream(self.comm_stream):
                next_comm = rccl_a2a_async(chunk, routing)

            # 计算上一个 chunk 的 Expert FFN（与本 chunk A2A 重叠）
            if pending_comm is not None:
                with torch.cuda.stream(self.compute_stream):
                    dispatched = pending_comm.wait()
                    results.append(expert_fn(dispatched))

            pending_comm = next_comm

        # 处理最后一个 chunk
        if pending_comm is not None:
            results.append(expert_fn(pending_comm.wait()))

        return torch.cat(results, dim=0)
```

### 收益假设与验证

```
预期吞吐提升：5~15%（取决于当前 A2A 时间占比和现有 overlap 率）

必做实验：

  实验 A：overlap 率对比
    设置：CommProfiler 在启用前后各跑 100 steps
    指标：实际 overlap 率（A2A 时间中有多少被 Expert FFN 覆盖）
    预期：overlap 率从当前值提升到 ≥ 80%

  实验 B：chunk 数量的影响
    设置：num_chunks = 1（关闭）/ 2 / 4 / 8
    指标：step time vs overlap 率
    预期：存在最优 chunk 数（太多 chunk 增加调度开销）

  实验 C：与 Feature 1+2 叠加
    对比：累计收益 vs 单独各自的收益（是否有叠加效果）

  实验 D：正确性验证
    对比：overlap 版本 vs 非 overlap 版本的输出（数值一致性）
    这是最重要的验证：chunk 拆分不能改变计算结果
```

### 决策门

```
if 实验 A overlap 率提升 < 5%（绝对值）：
  → 当前系统的 A2A 和 FFN 已经有较好 overlap
  → Feature 3 收益边际效应低
  → 评估是否值得继续 Feature 4（CommTensor zero-copy）

if 实验 C 三个 Feature 叠加后总提升 ≥ 20%：
  → 技术路线验证充分，可以开始考虑 Axion 构建阶段
  → 这是"进入 Axion 重写"的关键判断点之一

if 实验 D 发现数值不一致：
  → 立即停止，找 chunk 拆分的边界条件 bug
  → 不上生产，不做更多实验
```

### 技术产出

```
□ OverlapScheduler（Primus/Megatron 的 MoE 层替换方案）
□ chunk 大小自动选择（基于 MI300X 带宽参数的启发式规则）
□ 重叠调度的正确性证明（依赖关系分析文档）
□ 论文方向：Feature 3 是 Paper 2 的核心贡献
  "Static Communication-Computation Overlap for MoE on MI300X"
  关键差异：与 FlowMoE 对比（静态 vs 动态调度）
```

### 时间估计
**实现：2 周 | 实验：2~3 周 | 总计 4~5 周**（业余 8~10 周）

---

## 5. Feature 4：CommTensor zero-copy

### 背景与假设

```
假设：Expert dispatch/combine 的 pack/unpack 操作消耗了显著的内存带宽，
      通过在 A2A 之前直接使用通信友好的物理布局，可以消除这两次拷贝。

前提：Feature 0 CommReport 显示 A2A 时间中有明显的 pack/unpack 开销
      （在 MI300X 上：seq_len × hidden × 2 bytes × 2 = 约 58 MB per A2A for S=4096,H=7168）

注意：这个 Feature 不需要完整的 CommTensor 类型系统，
      在 Primus/Megatron 中只需要改内存分配策略即可验证核心假设。
```

### 实现方案

```python
# 核心思路：
# 传统：hidden [S, H]（按 token 顺序）→ pack（重排）→ A2A → unpack
# zero-copy：直接分配按目标 GPU 分组的内存 → A2A（直接 DMA）→ index map 访问
#
# 在 Primus/Megatron 中不需要完整 CommTensor 系统，
# 只需要在 dispatch 前改内存分配方式：

def dispatch_zero_copy(hidden, routing_table):
    # 直接分配按目标 rank 分组的 buffer
    # 物理内存：[rank0_tokens | rank1_tokens | ... | rankN_tokens]
    sorted_hidden, index_map = sort_by_dst_rank(hidden, routing_table)
    # sort_by_dst_rank 就是原来的 pack 操作，但直接输出到目标 buffer

    # A2A：direct DMA，无需额外 copy
    dispatched = rccl_a2a(sorted_hidden, routing_table.send_counts)

    # 返回 dispatched + index_map（供后续 combine 使用）
    return dispatched, index_map

def combine_zero_copy(expert_output, index_map, routing_table):
    # A2A combine：direct DMA
    combined_sorted = rccl_a2a(expert_output, routing_table.recv_counts)

    # 用 index_map 恢复原始顺序（index_select，仍然是一次 copy）
    # 这一步暂时无法完全消除，但比 unpack 高效
    return combined_sorted[index_map]
```

### 收益假设与验证

```
预期：在高 seq_len（S ≥ 4096）场景下，A2A 端到端时间降低 10~20%
      MI300X HBM3（5.3 TB/s）比 H100（3.35 TB/s）收益更大

必做实验：

  实验 A：pack/unpack 单独开销
    设置：隔离 pack 操作，用 hipperf 测量
    指标：pack 耗时（按 seq_len 扫描：1024/2048/4096/8192）
    预期：seq_len=4096，pack 约占 A2A 总时间 15~25%

  实验 B：zero-copy vs 原始的 A2A 端到端时间
    设置：相同 routing，对比两种实现的 A2A 总时间
    排除变量：使用相同 RCCL 配置

  实验 C：端到端 step time 提升
    设置：64 MI300X，接入 Primus/Megatron
    指标：tok/s 提升百分比

  实验 D：正确性验证
    对比：zero-copy 版本 vs 原始版本的 dispatch/combine 输出
```

### 决策门

```
if 实验 A pack 开销 < 5% of A2A 时间：
  → pack/unpack 不是显著瓶颈
  → Feature 4 的绝对收益有限（< 2% 端到端）
  → 不在 Primus/Megatron 上继续推进
  → 记录这个结论（CommTensor 的价值在 MI300X 上有限制条件）

if 实验 C 端到端提升 ≥ 3%：
  → 有增量价值，继续
  → 但注意：这个 Feature 实现复杂度高，需要 ROI 合理

重要认知：
  CommTensor 最大的价值不一定是 zero-copy 的绝对性能，
  而是它的类型系统设计（编译期保证 layout 正确性）。
  如果 zero-copy 性能收益不显著，这个 Feature 的价值
  主要体现在 Axion 构建阶段的设计严谨性，而非现在的性能。
```

### 技术产出

```
□ zero-copy dispatch/combine patch（Primus/Megatron）
□ MI300X pack/unpack 开销的精确量化（hipperf 数据）
□ CommTensor 设计的性能验证报告
  （证明或证伪：zero-copy 在 MI300X 上是否有显著价值）
□ 论文方向：作为 Paper 3 或 Feature 1~3 的一个 section
```

### 时间估计
**实现：2~3 周 | 实验：2 周 | 总计 4~5 周**（业余 8~10 周）

---

## 6. 后续 Feature（由前序数据驱动）

```
目前还没有足够数据来设计 Feature 5+。
根据 Feature 0~4 的结果，可能的方向：

  方向 A：RaggedShard（Dense 参数分片灵活化）
    适用条件：内部开始使用 Shampoo/Muon 优化器
    参考：veScale-FSDP 的核心贡献

  方向 B：FSEP 热点 Expert 分裂（一个 Expert 分片到多 GPU）
    适用条件：256+ GPU 场景，单个 Expert 参数量大
    参考：LAER-MoE FSEP 的完整版

  方向 C：跨节点通信拓扑优化
    适用条件：Feature 0 显示跨节点 A2A 占比 > 60%
    方案：Expert 初始分配拓扑感知，优先节点内路由

  方向 D：Sequence Parallelism 与 EP 的联合优化
    适用条件：长序列训练（S > 8192）成为主要场景

具体做哪个，等 Feature 0~4 的数据说话。
```

---

## 7. Axion 构建阶段（条件触发）

### 触发条件

```
同时满足以下条件，才开始 Axion 重写：

  □ Feature 0~3 中至少 3 个通过验证（有 ≥ 5% 各自收益）
  □ 累计端到端吞吐提升 ≥ 20%（对比 Primus/Megatron baseline）
  □ Feature 之间的 patch 开始出现耦合（维护成本上升）
  □ 有时间投入（不再是纯业余项目，或获得内部支持）

如果条件始终不满足：
  → Axion 构建阶段无限期推迟
  → 验证阶段的成果（各个 Feature patch）本身已有足够价值
```

### 构建阶段的意义

```
进入 Axion 构建阶段时，每个设计决策都有数据背书：

  ModelGraph + CommInferencePass：
    因为 Feature 0 证明了"通信可见性"有价值
    → 我们知道这个 Pass 应该分析什么

  CommTensor + CommTensorLayoutPass：
    因为 Feature 4 量化了 zero-copy 的收益
    → 我们知道 CommLayout 枚举是否需要 SPARSE_CSR

  OverlapInsertionPass：
    因为 Feature 3 证明了静态 chunk 调度有效
    → 我们知道 num_chunks 的合理范围

  FSEPShardingPass：
    因为 Feature 2 验证了 Expert 迁移的 ROI
    → 我们知道 imbalance_threshold 的合理值

  这些不是猜测，而是从真实 MI300X 训练数据中得到的参数。
  Axion 的设计是"基于证据的设计"，而非"基于直觉的设计"。
```

### 构建阶段概要（仅供参考，届时重新规划）

```
Stage A（约 3 个月）：
  ModelGraph + PassManager + AnalysisPass + FusionPass
  单机 MI300X 跑通 Llama 3.1 8B + DSv3-like MoE

Stage B（约 3 个月）：
  CommInferencePass + FSEPShardingPass
  OverlapInsertionPass + CommTensorLayoutPass
  DistributedExecutablePlan

Stage C（约 3 个月）：
  CommFabric RCCL Driver（完整版）
  CommTensor 运行时（zero-copy，index map）
  FSEP Slow/Fast Planner（用验证阶段的参数）

Stage D（按需）：
  新架构支持、更大规模、开源评估
```

---

## 8. 总时间线与里程碑

```
时间线（业余时间，假设每周 5~10 小时）：

Week 1~4    Feature 0：CommProfiler + CommReport
               ↓ 数据：确认瓶颈类型和优先级

Week 5~8    Feature 1：FastRouter（+收敛实验）
               ↓ 数据：路由均衡是否安全有效？

Week 9~16   Feature 2：SlowPlanner（+收敛实验）
               ↓ 数据：物理迁移 ROI？

Week 17~22  Feature 3：OverlapScheduler
               ↓ 数据：静态调度提升多少？

Week 23~28  Feature 4：CommTensor zero-copy
               ↓ 数据：pack/unpack 开销是否显著？

Week 28 后  决策点：
               □ 累计收益是否 ≥ 20%？
               □ Patch 是否开始耦合？
               □ 是否有内部支持？
               → Yes × 3：进入 Axion 构建阶段
               → 否则：停留在验证阶段，持续迭代
```

---

## 9. 技术产出规划

```
Feature 0~2 完成后（约 Week 16）：
  → 内部技术报告：MI300X MoE 训练的通信优化实践
  → 可选：AMD 技术博客（Feature 0+1+2 的数据）
  → 论文方向：MLSys 2027 投稿窗口（约 Week 20）
    "Load-Balanced MoE Training on MI300X: Observations and Optimizations"

Feature 3~4 完成后（约 Week 28）：
  → 内部报告：完整 Feature 0~4 收益拆分
  → 论文方向：OSDI/EuroSys 2027 投稿窗口
    "Towards Communication-Efficient MoE Training on AMD GPUs"

Axion 构建完成后（如果触发）：
  → 完整系统论文：OSDI/SOSP 2028
  → 开源评估（AMD IP 审查）
```

---

## 10. 每个 Feature 的独立可交付性

```
每个 Feature 是独立的——即使后续 Feature 不做，已完成的 Feature 也有价值：

  仅 Feature 0：
    → 内部工程师第一次能看清 MI300X MoE 通信瓶颈
    → 这本身就是有价值的工具

  Feature 0 + 1：
    → 5~15% 吞吐提升（如果收敛实验通过）
    → 一份有数据的内部技术报告

  Feature 0 + 1 + 2：
    → 预期 15~30% 累计吞吐提升
    → 有充分数据支撑一篇 paper

  Feature 0 + 1 + 2 + 3：
    → 预期 20~40% 累计吞吐提升
    → 足够支撑顶会投稿

  所有 Feature 完成 → 进入 Axion 构建阶段
  或者：所有 Feature 完成但不构建 Axion → 已经足够有价值
```

---

*Axion 整体规划 v0.3 | 2026-03-08*  
*核心思路：Feature-by-Feature 验证 → 数据驱动决策 → 有收益才继续 → Axion 是终点不是起点*
