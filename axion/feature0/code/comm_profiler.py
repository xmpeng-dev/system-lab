"""
axion/feature0/code/comm_profiler.py

AxionCommProfiler: 零侵入挂载在 Megatron-LM MoE 层上的通信 Profiler。

依赖:
    - megatron-core (NVIDIA/Megatron-LM)
    - PyTorch

使用方式:
    profiler = AxionCommProfiler(num_warmup_steps=5, profile_steps=20)
    profiler.attach(model)
    # 正常跑训练...
    report = profiler.report()
    profiler.detach()
"""

import time
import types
import torch
import torch.distributed as dist
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ─────────────────────────────────────────────────────────────
# 运行时动态导入 megatron-core，避免在没有 megatron 环境时 import 失败
# ─────────────────────────────────────────────────────────────

def _import_megatron_moe():
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
    from megatron.core.transformer.moe.router import TopKRouter
    return MoELayer, MoEAlltoAllTokenDispatcher, TopKRouter


# ─────────────────────────────────────────────────────────────
# 原始数据结构：单个 step + 单层的采样快照
# ─────────────────────────────────────────────────────────────

@dataclass
class StepStats:
    """单个 training step、单个 MoE 层的原始统计数据"""
    layer_idx: int
    step_idx: int
    # Expert 负载（来自 TopKRouter.forward() 的 routing_map）
    tokens_per_expert: torch.Tensor        # shape: [num_experts]
    # A2A 时间（wall-clock，单位 ms）
    dispatch_a2a_ms: float = 0.0           # dispatch 阶段 A2A 耗时
    combine_a2a_ms: float = 0.0            # combine 阶段 A2A 耗时
    # MoE 层总时间（包含 A2A + Expert FFN）
    moe_layer_ms: float = 0.0
    # 通信量（字节，用于估算带宽）
    dispatch_bytes: int = 0
    combine_bytes: int = 0


# ─────────────────────────────────────────────────────────────
# 主类
# ─────────────────────────────────────────────────────────────

class AxionCommProfiler:
    """
    零侵入挂载在 Megatron-LM MoE 层上的通信 Profiler。

    接入方式：
      1. MoELayer：register_forward_pre/post_hook → 测量 MoE 层总耗时
      2. TopKRouter：register_forward_hook → 拦截 routing_map，统计 Expert 负载
      3. MoEAlltoAllTokenDispatcher：monkey-patch dispatch/combine → 测量 A2A 时间
         （dispatcher 不走 nn.Module.forward()，无法用标准 hook 捕获）

    示例:
        profiler = AxionCommProfiler(num_warmup_steps=5, profile_steps=20)
        profiler.attach(model)
        for step in range(num_steps):
            train_step(...)
        report = profiler.report()
        profiler.detach()
    """

    def __init__(
        self,
        num_warmup_steps: int = 5,    # 前 N 步 warmup，不计入统计
        profile_steps: int = 20,       # 采样 N 步后停止
        enabled: bool = True,
    ):
        self.num_warmup_steps = num_warmup_steps
        self.profile_steps = profile_steps
        self.enabled = enabled

        self._step_idx: int = 0
        self._stats: List[StepStats] = []
        self._hooks: List = []         # nn.Module hook handles（可 remove）
        self._patched_dispatchers: List = []  # (module, orig_dispatch, orig_combine)

    # ──────────────────────────────────────────────────────────
    # 挂载 / 卸载
    # ──────────────────────────────────────────────────────────

    def attach(self, model: torch.nn.Module) -> "AxionCommProfiler":
        """
        扫描 model 中所有 MoELayer，注册 hook 和 monkey-patch。
        不修改模型参数和 forward 计算逻辑。
        """
        if not self.enabled:
            return self

        MoELayer, MoEAlltoAllTokenDispatcher, TopKRouter = _import_megatron_moe()

        moe_layer_count = 0
        for name, module in model.named_modules():
            if isinstance(module, MoELayer):
                self._register_moe_layer_hooks(module, moe_layer_count)
                moe_layer_count += 1

            elif isinstance(module, MoEAlltoAllTokenDispatcher):
                # layer_idx = moe_layer_count - 1（dispatcher 在对应 MoELayer 内部）
                self._patch_dispatcher(module, moe_layer_count - 1)

            elif isinstance(module, TopKRouter):
                self._register_router_hooks(module, moe_layer_count - 1)

        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            print(f"[AxionCommProfiler] Attached to {moe_layer_count} MoE layers. "
                  f"Warmup={self.num_warmup_steps}, ProfileSteps={self.profile_steps}")
        return self

    def detach(self):
        """移除所有 hook 并恢复 monkey-patched 方法"""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

        for module, orig_dispatch, orig_combine in self._patched_dispatchers:
            module.dispatch = orig_dispatch
            module.combine  = orig_combine
        self._patched_dispatchers.clear()

    # ──────────────────────────────────────────────────────────
    # Hook 实现
    # ──────────────────────────────────────────────────────────

    def _register_moe_layer_hooks(self, module: torch.nn.Module, layer_idx: int):
        """MoELayer 前后：测量整个 MoE 层的 wall-clock 时间"""
        _state: Dict = {}
        profiler = self

        def pre_hook(mod, inputs):
            if not profiler._is_profiling():
                return
            torch.cuda.synchronize()
            _state['t0'] = time.perf_counter()

        def post_hook(mod, inputs, outputs):
            if not profiler._is_profiling():
                # 即使不在 profiling 窗口，也需要驱动 step 计数（layer 0）
                if layer_idx == 0:
                    profiler._step_idx += 1
                return
            torch.cuda.synchronize()
            t0 = _state.get('t0', time.perf_counter())
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            profiler._get_or_create_stats(layer_idx).moe_layer_ms = elapsed_ms
            # layer 0 的 post_hook 驱动 step 计数
            if layer_idx == 0:
                profiler._step_idx += 1

        h1 = module.register_forward_pre_hook(pre_hook)
        h2 = module.register_forward_hook(post_hook)
        self._hooks.extend([h1, h2])

    def _patch_dispatcher(self, module, layer_idx: int):
        """
        Monkey-patch MoEAlltoAllTokenDispatcher.dispatch / combine。

        原因：dispatcher 的 dispatch()/combine() 是普通方法调用，
        不走 nn.Module.forward()，register_forward_hook 无法捕获。
        """
        orig_dispatch = module.dispatch
        orig_combine  = module.combine
        profiler = self

        def patched_dispatch(hidden_states, max_prob, top_indices):
            if not profiler._is_profiling():
                return orig_dispatch(hidden_states, max_prob, top_indices)

            dispatch_bytes = hidden_states.numel() * hidden_states.element_size()

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = orig_dispatch(hidden_states, max_prob, top_indices)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            stats = profiler._get_or_create_stats(layer_idx)
            stats.dispatch_a2a_ms = elapsed_ms
            stats.dispatch_bytes  = dispatch_bytes
            return result

        def patched_combine(expert_output, *args, **kwargs):
            if not profiler._is_profiling():
                return orig_combine(expert_output, *args, **kwargs)

            combine_bytes = expert_output.numel() * expert_output.element_size()

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = orig_combine(expert_output, *args, **kwargs)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            stats = profiler._get_or_create_stats(layer_idx)
            stats.combine_a2a_ms = elapsed_ms
            stats.combine_bytes  = combine_bytes
            return result

        module.dispatch = patched_dispatch
        module.combine  = patched_combine
        # 保存原始方法，供 detach() 恢复
        self._patched_dispatchers.append((module, orig_dispatch, orig_combine))

    def _register_router_hooks(self, module: torch.nn.Module, layer_idx: int):
        """
        TopKRouter post_hook：拦截 routing_map，统计 tokens_per_expert。

        megatron-core TopKRouter.forward() 返回 (probs, routing_map)：
          - probs: [num_tokens, top_k]，路由权重
          - routing_map: [num_tokens, top_k]，expert id（int）
        """
        profiler = self

        def post_hook(mod, inputs, outputs):
            if not profiler._is_profiling():
                return
            if not (isinstance(outputs, (tuple, list)) and len(outputs) >= 2):
                return
            routing_map = outputs[1]   # [num_tokens, top_k]
            if routing_map is None or routing_map.dim() != 2:
                return

            num_experts = getattr(mod.config, 'num_moe_experts', None)
            if num_experts is None:
                return

            tokens_per_expert = torch.bincount(
                routing_map.flatten().long(),
                minlength=num_experts,
            ).float().cpu()
            profiler._get_or_create_stats(layer_idx).tokens_per_expert = tokens_per_expert

        h = module.register_forward_hook(post_hook)
        self._hooks.append(h)

    # ──────────────────────────────────────────────────────────
    # 辅助方法
    # ──────────────────────────────────────────────────────────

    def _is_profiling(self) -> bool:
        """只在 warmup 结束后的 profile_steps 步内采样"""
        return (self.enabled
                and self._step_idx >= self.num_warmup_steps
                and self._step_idx < self.num_warmup_steps + self.profile_steps)

    def _get_or_create_stats(self, layer_idx: int) -> StepStats:
        """懒创建当前 step 当前层的 StepStats"""
        step = self._step_idx
        for s in self._stats:
            if s.step_idx == step and s.layer_idx == layer_idx:
                return s
        s = StepStats(
            layer_idx=layer_idx,
            step_idx=step,
            tokens_per_expert=torch.zeros(1),
        )
        self._stats.append(s)
        return s

    # ──────────────────────────────────────────────────────────
    # 生成报告
    # ──────────────────────────────────────────────────────────

    def report(self):
        """聚合采集到的 StepStats，生成 CommReport"""
        from .comm_report import CommReport
        return CommReport.from_stats(self._stats)

    @property
    def num_collected_steps(self) -> int:
        """已采集的 step 数（不含 warmup）"""
        return max(0, self._step_idx - self.num_warmup_steps)
