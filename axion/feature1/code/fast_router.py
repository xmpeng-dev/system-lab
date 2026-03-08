"""
axion/feature1/code/fast_router.py

FastRouter: 通过 EMA 负载统计动态调整 gate logits，软性均衡 Expert 负载。
不修改模型结构，只在 TopKRouter 的 softmax 之前插入一行偏置。
"""

import torch
import torch.distributed as dist
from dataclasses import dataclass
from typing import Optional


@dataclass
class FastRouterConfig:
    alpha: float = 0.1          # 惩罚强度（越大均衡越激进）
    beta: float = 2.0           # 非线性系数（越大对重载 Expert 惩罚越集中）
    ema_decay: float = 0.9      # EMA 平滑系数
    num_experts: int = 0        # 全局 Expert 数量（attach 时自动填充）
    ep_group: Optional[object] = None  # Expert Parallel 通信组


class FastRouter:
    """
    零侵入的 Expert 负载均衡路由调整器。

    使用方式:
        fast_router = FastRouter(FastRouterConfig(alpha=0.1, beta=2.0))
        fast_router.attach(model)  # monkey-patch TopKRouter

        for step in range(num_steps):
            train_step(model, ...)
            fast_router.step_end()  # 更新 EMA 负载统计

        fast_router.detach()       # 恢复原始 router
    """

    def __init__(self, config: FastRouterConfig):
        self.config = config
        self.load_ema: Optional[torch.Tensor] = None
        self._patched_routers = []   # (module, orig_forward) 列表

    # ──────────────────────────────────────────────────────────
    # 挂载 / 卸载
    # ──────────────────────────────────────────────────────────

    def attach(self, model: torch.nn.Module) -> "FastRouter":
        """扫描 model，monkey-patch 所有 TopKRouter"""
        try:
            from megatron.core.transformer.moe.router import TopKRouter
        except ImportError:
            raise RuntimeError("megatron-core not found. Install NVIDIA/Megatron-LM.")

        router_count = 0
        for name, module in model.named_modules():
            if isinstance(module, TopKRouter):
                # 读取 num_experts（首次）
                if self.config.num_experts == 0:
                    self.config.num_experts = module.config.num_moe_experts
                    self.load_ema = torch.ones(
                        self.config.num_experts, dtype=torch.float32
                    )

                self._patch_router(module)
                router_count += 1

        print(f"[FastRouter] Attached to {router_count} TopKRouters, "
              f"num_experts={self.config.num_experts}, "
              f"alpha={self.config.alpha}, beta={self.config.beta}")
        return self

    def detach(self):
        """恢复所有 TopKRouter 的原始 forward"""
        for module, orig_forward in self._patched_routers:
            module.forward = orig_forward
        self._patched_routers.clear()
        print("[FastRouter] Detached.")

    def _patch_router(self, module):
        import types
        orig_forward = module.forward
        fast_router = self

        def patched_forward(hidden_states):
            # 1. 调用原始 forward 获取 logits
            #    megatron-core TopKRouter 的 forward 内部做 logit 计算 + topk
            #    我们需要在 softmax 之前拦截。
            #
            #    由于 megatron-core 的 TopKRouter 不暴露中间 logits，
            #    当前版本采用 monkey-patch _compute_router_probabilities 的方式。
            #    TODO：如果 megatron-core 版本不同，需要适配接入点。
            return fast_router._forward_with_penalty(module, orig_forward, hidden_states)

        module.forward = types.MethodType(
            lambda self_mod, hidden_states: patched_forward(hidden_states),
            module,
        )
        self._patched_routers.append((module, orig_forward))

    def _forward_with_penalty(self, router_module, orig_forward, hidden_states):
        """
        在 TopKRouter.forward 中插入负载惩罚。

        实现策略：
          megatron-core TopKRouter 的 forward 分两步：
            1. _compute_router_probabilities → 计算 logits + softmax → probs
            2. _get_top_k_logits → topk 选择

          我们 patch _compute_router_probabilities，在 softmax 之前插入惩罚。
        """
        if self.load_ema is None:
            return orig_forward(hidden_states)

        # 临时替换 _compute_router_probabilities
        orig_compute = router_module._compute_router_probabilities
        fast_router = self

        def patched_compute(hidden_states_inner, num_experts, apply_softmax):
            logits, probs = orig_compute(hidden_states_inner, num_experts,
                                         apply_softmax=False)
            # 插入负载惩罚（在 softmax 之前）
            penalty = fast_router.compute_penalty()  # [num_experts]
            logits = logits - penalty.to(logits.device)
            if apply_softmax:
                probs = torch.softmax(logits, dim=-1)
            else:
                probs = logits
            return logits, probs

        router_module._compute_router_probabilities = patched_compute
        result = orig_forward(hidden_states)
        router_module._compute_router_probabilities = orig_compute
        return result

    # ──────────────────────────────────────────────────────────
    # 负载统计 & 惩罚计算
    # ──────────────────────────────────────────────────────────

    def compute_penalty(self) -> torch.Tensor:
        """
        计算 gate logit 惩罚项：
            penalty_i = alpha * log((load_ema_i / mean_load)^beta)
                      = alpha * beta * log(load_ema_i / mean_load)
        """
        mean_load = self.load_ema.mean()
        if mean_load <= 0:
            return torch.zeros_like(self.load_ema)
        ratio = self.load_ema / mean_load
        ratio = ratio.clamp(min=1e-6)   # 避免 log(0)
        penalty = self.config.alpha * self.config.beta * ratio.log()
        return penalty   # shape: [num_experts]

    @torch.no_grad()
    def update_load(self, expert_counts: torch.Tensor):
        """
        用本 step 的 Expert token 计数更新 EMA 负载统计。
        expert_counts: [num_experts]，每个 expert 收到的 token 数。

        调用时机：每个 step 的 dispatch 完成后（在 optimizer.step 之前）。
        """
        counts = expert_counts.float().cpu()

        # EP 场景：all_reduce 获取全局负载
        if self.config.ep_group is not None:
            counts = counts.cuda()
            dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=self.config.ep_group)
            counts = counts.cpu()

        if self.load_ema is None:
            self.load_ema = counts
        else:
            d = self.config.ema_decay
            self.load_ema = d * self.load_ema + (1 - d) * counts

    def step_end(self, expert_counts: Optional[torch.Tensor] = None):
        """
        每个 step 结束时调用。
        如果传入 expert_counts，更新 EMA；否则跳过（保持上一步的统计）。
        """
        if expert_counts is not None:
            self.update_load(expert_counts)

    # ──────────────────────────────────────────────────────────
    # 状态查询
    # ──────────────────────────────────────────────────────────

    @property
    def current_imbalance(self) -> float:
        """当前 EMA 负载的不均衡系数（max / mean）"""
        if self.load_ema is None:
            return 1.0
        mean = self.load_ema.mean().item()
        if mean <= 0:
            return 1.0
        return (self.load_ema.max() / mean).item()
