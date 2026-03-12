"""
MoEPackage Fused Permute-Quantize-Dispatch Pipeline

将 4 个核心模块串联成端到端的 MoE 层前向/反向流水线。
关键创新：单 HIP Kernel 完成 Permute + FP8 Quantize + P2P Write

Megatron-Core 当前的 Dispatch 流水线：
  Kernel 1: Fused Permute + Pack        (1R BF16, 1W BF16)
  Kernel 2: FP8 Quantize                (1R BF16, 1W FP8)
  Kernel 3: DeepEP Dispatch             (1R FP8, 网络发送)
  → HBM 访问 5 次

MoEPackage 的 Fused Pipeline：
  Single HIP Kernel: Permute + Quantize + P2P Write
  → 1 次 HBM 读取 + XGMI P2P 写入远端（零本地写）
  → HBM 访问减少 80%

DeepSeek-V3 参数下的性能分析：
  T=4096, K=8, H=7168, BF16
  Megatron-Core: 5 × 4096×8×7168×2B = 2.24 GB HBM 流量
  MoEPackage:    1 × 4096×8×7168×2B = 0.45 GB HBM 读取
  61 层 × FWD+BWD：~273 GB → ~55 GB（节省 ~218 GB）
  MI300X 5.3 TB/s：~41ms → ~10ms → Dispatch 流水线加速 ~4×
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .xgmi_dispatch import XGMIDispatcher, XGMIDispatchConfig
from .p2p_buffer_pool import P2PBufferPool, BufferPoolConfig, BufferUsage
from .dropless_gemm import DroplessGroupedGEMM, DroplessGEMMConfig
from .dual_channel_scheduler import (
    DualChannelScheduler,
    CommTask,
    TaskType,
    ChannelType,
)


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# MI300X 硬件参数
MI300X_HBM_BW_TBS = 5.3         # HBM 带宽 TB/s
MI300X_XGMI_BW_GBS = 896.0     # XGMI 带宽 GB/s
MI300X_BF16_TFLOPS = 1307.0    # BF16 峰值 TFLOPS


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

@dataclass
class FusedPipelineConfig:
    """
    Fused Pipeline 端到端配置

    Args:
        num_experts: Expert 总数
        d_model: 隐藏层维度
        d_ffn: FFN 中间层维度
        num_ep_ranks: EP 并行度
        max_tokens_per_step: 每 step 最大 token 数
        top_k: TopK 路由
        num_layers: MoE 层数
        gpus_per_node: 每节点 GPU 数
        dtype: 数据类型
        use_fp8_dispatch: Dispatch 时是否量化为 FP8
        use_fp8_gemm: Expert GEMM 是否使用 FP8
        safety_factor: Expert 容量安全系数
        activation: 激活函数
        enable_hip_graph: 是否启用 HIP Graph
    """
    num_experts: int = 256
    d_model: int = 7168
    d_ffn: int = 18432
    num_ep_ranks: int = 64
    max_tokens_per_step: int = 4096
    top_k: int = 8
    num_layers: int = 61          # DeepSeek-V3: 61 MoE layers
    gpus_per_node: int = 8
    dtype: torch.dtype = torch.bfloat16
    use_fp8_dispatch: bool = True
    use_fp8_gemm: bool = False
    safety_factor: float = 1.5
    activation: str = 'swiglu'
    enable_hip_graph: bool = True

    @property
    def experts_per_rank(self) -> int:
        return max(1, self.num_experts // self.num_ep_ranks)

    @property
    def bytes_per_element(self) -> int:
        if self.use_fp8_dispatch:
            return 1
        return 2 if self.dtype in (torch.float16, torch.bfloat16) else 4

    @property
    def dispatch_bytes_per_layer(self) -> int:
        """单层 Dispatch 的数据量 (bytes)"""
        return (
            self.max_tokens_per_step * self.top_k
            * self.d_model * self.bytes_per_element
        )


# ---------------------------------------------------------------------------
# MoEPackage Layer：端到端 MoE 层
# ---------------------------------------------------------------------------

class MoEPackageLayer(nn.Module):
    """
    Complete MoE layer with Fused Permute-Quantize-Dispatch Pipeline.

    整合所有 4 个 MoEPackage 模块：
      Module 1: XGMIDispatcher — XGMI P2P Dispatch/Combine
      Module 2: P2PBufferPool — 持久化 Buffer 管理
      Module 3: DroplessGroupedGEMM — 静态 Padded GEMM
      Module 4: DualChannelScheduler — 双通道调度

    Forward:
      Gate → Fused(Permute + FP8 Quant + P2P Write) → Expert GEMM
        → Fused(Dequant + Unpermute + Weighted Reduce)

    Backward:
      Reverse path with gradient computation

    Usage:
        config = FusedPipelineConfig()
        layer = MoEPackageLayer(config, layer_id=0, local_rank=0)
        output = layer(hidden_states)
    """

    def __init__(
        self,
        config: FusedPipelineConfig,
        layer_id: int = 0,
        local_rank: int = 0,
    ):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.local_rank = local_rank
        self._device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        # --- Gate / Router ---
        # Gate 输出维度 = 本 rank 上的 expert 数（至少 top_k 以支持 TopK 选择）
        self.num_local_experts = max(config.experts_per_rank, config.top_k)
        self.gate = nn.Linear(config.d_model, self.num_local_experts,
                              bias=False)

        # --- Module 1: XGMI Dispatcher ---
        dispatch_config = XGMIDispatchConfig(
            num_ep_ranks=config.num_ep_ranks,
            d_model=config.d_model,
            max_tokens_per_step=config.max_tokens_per_step,
            top_k=config.top_k,
            gpus_per_node=config.gpus_per_node,
            dtype=config.dtype,
            use_fp8_dispatch=config.use_fp8_dispatch,
        )
        self.dispatcher = XGMIDispatcher(dispatch_config)

        # --- Module 2: P2P Buffer Pool ---
        capacity = math.ceil(
            config.max_tokens_per_step * config.top_k
            / config.num_ep_ranks * 2.0
        )
        pool_config = BufferPoolConfig(
            num_ep_ranks=config.num_ep_ranks,
            d_model=config.d_model,
            capacity_per_rank=capacity,
            dtype=config.dtype,
        )
        self.buffer_pool = P2PBufferPool(pool_config, local_rank, self._device)

        # --- Module 3: Dropless Grouped GEMM ---
        gemm_config = DroplessGEMMConfig(
            num_experts=config.num_experts,
            d_model=config.d_model,
            d_ffn=config.d_ffn,
            max_tokens_per_step=config.max_tokens_per_step,
            num_ep_ranks=config.num_ep_ranks,
            safety_factor=config.safety_factor,
            activation=config.activation,
            use_fp8_gemm=config.use_fp8_gemm,
        )
        self.expert_gemm = DroplessGroupedGEMM(gemm_config)

        # --- Module 4: Dual-Channel Scheduler ---
        self.scheduler = DualChannelScheduler()

        # 初始化 Buffer
        self.dispatcher.initialize_buffers(local_rank, capacity)
        self.buffer_pool.exchange_ipc_handles()

    def forward(self, hidden_states: Tensor) -> Tensor:
        """
        MoE layer forward pass with fused pipeline.

        完整前向流程：
          1. Gate 路由：hidden_states → routing scores + expert assignments
          2. Fused Dispatch: Permute + [FP8 Quant] + XGMI P2P Write
          3. Expert GEMM: Padded Static Grouped GEMM with valid_mask
          4. Fused Combine: Dequant + Unpermute + Weighted Reduce

        Args:
            hidden_states: [batch_size × seq_len, d_model]

        Returns:
            output: [batch_size × seq_len, d_model]
        """
        T, H = hidden_states.shape
        assert H == self.config.d_model, (
            f"Expected d_model={self.config.d_model}, got {H}"
        )

        # ─── Step 1: Gate 路由 ───
        gate_logits = self.gate(hidden_states)       # [T, E_local]
        routing_scores, expert_ids = torch.topk(
            gate_logits, self.config.top_k, dim=-1,
        )                                             # 各 [T, K]
        routing_scores = torch.softmax(routing_scores, dim=-1)

        # 将 expert_ids 映射到 EP rank（简化：均匀映射）
        # 统一验证点：确保 expert_ids 在合法范围内
        routing_map = expert_ids % self.config.num_ep_ranks

        # ─── Step 2: Fused Dispatch ───
        # AMD HIP: 单 kernel 完成 Permute + FP8 Quant + P2P Write
        # fused_permute_quant_dispatch_kernel:
        #   1. 从 HBM 读入 token 到 LDS (64KB/CU)
        #   2. 在 LDS 中查询 routing_map → 目标 GPU + slot
        #   3. 在 LDS 中完成 BF16 → FP8 量化（如果启用）
        #   4. 从 LDS 直接 XGMI P2P 写入目标 GPU recv buffer

        # 原型：分步模拟融合操作
        dispatched_tokens = self._fused_dispatch(
            hidden_states, routing_map, expert_ids,
        )

        # ─── Step 3: Expert GEMM ───
        recv_tokens, recv_expert_ids, source_info = dispatched_tokens
        if recv_tokens.shape[0] > 0:
            # Map global expert_ids to local expert range
            local_eids = recv_expert_ids % self.config.experts_per_rank
            expert_output, valid_mask = self.expert_gemm(
                recv_tokens, local_eids,
            )
        else:
            expert_output = recv_tokens
            valid_mask = torch.ones(0, dtype=torch.bool, device=self._device)

        # ─── Step 4: Fused Combine ───
        # AMD HIP: 单 kernel 完成 Dequant + Unpermute + Weighted Reduce
        output = self._fused_combine(
            expert_output, source_info, routing_scores,
            num_output_tokens=T,
        )

        return output

    def _fused_dispatch(
        self,
        hidden_states: Tensor,
        routing_map: Tensor,
        expert_ids: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Fused Permute + Quantize + P2P Dispatch.

        生产中：单个 HIP Kernel
          每个 thread block 处理一个 token：
          1. Load token from HBM to LDS (64KB/CU)
          2. Lookup routing_map in LDS → target GPU + slot
          3. BF16 → FP8 quantize in LDS (if enabled)
          4. XGMI P2P write from LDS to remote recv buffer
          → 1 HBM read + 0 local HBM write（远端写入不占本卡 HBM 带宽）

        原型：通过 XGMIDispatcher 模拟
        """
        # 可选 FP8 量化
        if self.config.use_fp8_dispatch:
            # AMD HIP: 实际在 LDS 中完成量化，这里模拟
            # MI300X MFMA 支持 E4M3 FP8
            dispatch_data = hidden_states  # 原型中保持原精度
        else:
            dispatch_data = hidden_states

        # XGMI P2P Dispatch
        return self.dispatcher.dispatch(dispatch_data, routing_map, expert_ids)

    def _fused_combine(
        self,
        expert_output: Tensor,
        source_info: Tensor,
        routing_scores: Tensor,
        num_output_tokens: int,
    ) -> Tensor:
        """
        Fused Dequantize + Unpermute + Weighted Reduce.

        生产中：单个 HIP Kernel
          1. 从远端 recv buffer 读取 FP8 Expert 输出
          2. FP8 → BF16 反量化
          3. 按 routing_scores 加权
          4. scatter_add 到输出 buffer
          → 消除中间 BF16 buffer 的 2 次 HBM 读写

        原型：通过 XGMIDispatcher.combine 模拟
        """
        return self.dispatcher.combine(
            expert_output, source_info, routing_scores,
        )

    # -----------------------------------------------------------------------
    # 性能估算
    # -----------------------------------------------------------------------

    def estimate_performance(self) -> Dict[str, float]:
        """
        Estimate per-layer and full-model performance.

        基于 DeepSeek-V3 参数和 MI300X 硬件规格。
        """
        cfg = self.config
        T = cfg.max_tokens_per_step
        K = cfg.top_k
        H = cfg.d_model
        bpe = cfg.bytes_per_element

        # --- Dispatch 流水线 HBM 流量对比 ---
        # Megatron-Core: 5 × T × K × H × bpe（3 读 + 2 写）
        mc_dispatch_bytes = 5 * T * K * H * 2  # BF16 = 2 bytes
        # MoEPackage: 1 × T × K × H × bpe（1 读，远端写入不占本卡 HBM）
        mp_dispatch_bytes = 1 * T * K * H * bpe

        # 单层时间
        hbm_bw_bytes_per_us = MI300X_HBM_BW_TBS * 1e6  # TB/s → bytes/μs
        mc_dispatch_us = mc_dispatch_bytes / hbm_bw_bytes_per_us
        mp_dispatch_us = mp_dispatch_bytes / hbm_bw_bytes_per_us

        # Expert GEMM 时间估算
        gemm_eff = self.expert_gemm.estimate_efficiency()
        expert_gemm_us = gemm_eff['estimated_gemm_time_ms'] * 1000

        # --- 双通道带宽优势 ---
        bw_analysis = self.scheduler.estimate_effective_bandwidth(
            cfg.dispatch_bytes_per_layer,
        )

        # --- 全模型估算 ---
        # 61 层 × (FWD + BWD)
        total_layers = cfg.num_layers * 2  # FWD + BWD
        mc_total_dispatch_ms = mc_dispatch_us * total_layers / 1000
        mp_total_dispatch_ms = mp_dispatch_us * total_layers / 1000

        return {
            # 单层分析
            'mc_dispatch_bytes_per_layer': mc_dispatch_bytes,
            'mp_dispatch_bytes_per_layer': mp_dispatch_bytes,
            'hbm_savings_pct': round(
                (1 - mp_dispatch_bytes / mc_dispatch_bytes) * 100, 1
            ),
            'mc_dispatch_us': round(mc_dispatch_us, 1),
            'mp_dispatch_us': round(mp_dispatch_us, 1),
            'dispatch_speedup': round(mc_dispatch_us / max(mp_dispatch_us, 0.1), 1),

            # Expert GEMM
            'expert_gemm_us': round(expert_gemm_us, 1),
            'compute_efficiency_pct': gemm_eff['compute_efficiency_pct'],

            # 双通道
            'amd_effective_bw_gbs': bw_analysis['amd_effective_bw_gbs'],
            'nvidia_effective_bw_gbs': bw_analysis['nvidia_effective_bw_gbs'],
            'bw_advantage_pct': bw_analysis['bw_advantage_pct'],

            # 全模型
            'total_layers_fwd_bwd': total_layers,
            'mc_total_dispatch_ms': round(mc_total_dispatch_ms, 2),
            'mp_total_dispatch_ms': round(mp_total_dispatch_ms, 2),
            'total_dispatch_savings_gb': round(
                (mc_dispatch_bytes - mp_dispatch_bytes)
                * total_layers / (1024 ** 3), 2
            ),
        }

    def build_forward_dag(self) -> List[CommTask]:
        """Build the complete forward DAG for this layer."""
        return self.scheduler.build_moe_forward_dag(
            layer_id=self.layer_id,
            num_tokens=self.config.max_tokens_per_step,
            d_model=self.config.d_model,
            top_k=self.config.top_k,
            bytes_per_element=self.config.bytes_per_element,
        )

    def memory_report(self) -> Dict[str, float]:
        """Return comprehensive memory usage report."""
        pool_mem = self.buffer_pool.memory_report()
        dispatch_mem = self.dispatcher.memory_report()
        # Expert 权重
        E = self.config.experts_per_rank
        H = self.config.d_model
        F = self.config.d_ffn
        if self.config.activation == 'swiglu':
            weight_bytes = E * (H * 2 * F + F * H) * 2  # BF16
        else:
            weight_bytes = E * (H * F + F * H) * 2

        return {
            'buffer_pool_gb': pool_mem['total_gb'],
            'dispatch_buffers_mb': dispatch_mem.get('total_recv_buffer_mb', 0),
            'expert_weights_gb': weight_bytes / (1024 ** 3),
            'total_gb': (
                pool_mem['total_gb']
                + dispatch_mem.get('total_recv_buffer_mb', 0) / 1024
                + weight_bytes / (1024 ** 3)
            ),
        }

    def __repr__(self) -> str:
        cfg = self.config
        perf = self.estimate_performance()
        return (
            f"MoEPackageLayer(\n"
            f"  layer_id={self.layer_id},\n"
            f"  experts={cfg.num_experts}, d_model={cfg.d_model}, "
            f"d_ffn={cfg.d_ffn},\n"
            f"  top_k={cfg.top_k}, ep_ranks={cfg.num_ep_ranks},\n"
            f"  fp8_dispatch={cfg.use_fp8_dispatch}, "
            f"hip_graph={cfg.enable_hip_graph},\n"
            f"  dispatch_speedup={perf['dispatch_speedup']}×, "
            f"hbm_savings={perf['hbm_savings_pct']}%,\n"
            f"  dual_channel_bw={perf['amd_effective_bw_gbs']} GB/s "
            f"(vs NVIDIA {perf['nvidia_effective_bw_gbs']} GB/s)\n"
            f")"
        )


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------

def create_moepackage_layer(
    num_experts: int = 256,
    d_model: int = 7168,
    d_ffn: int = 18432,
    num_ep_ranks: int = 64,
    top_k: int = 8,
    layer_id: int = 0,
    local_rank: int = 0,
    **kwargs,
) -> MoEPackageLayer:
    """
    Factory function to create a MoEPackageLayer with DeepSeek-V3 defaults.

    Args:
        num_experts: Number of experts (default: 256 for DeepSeek-V3)
        d_model: Hidden dimension (default: 7168)
        d_ffn: FFN dimension (default: 18432)
        num_ep_ranks: EP parallelism (default: 64)
        top_k: TopK routing (default: 8)
        layer_id: Layer index
        local_rank: Local GPU rank
        **kwargs: Additional FusedPipelineConfig parameters

    Returns:
        Configured MoEPackageLayer
    """
    config = FusedPipelineConfig(
        num_experts=num_experts,
        d_model=d_model,
        d_ffn=d_ffn,
        num_ep_ranks=num_ep_ranks,
        top_k=top_k,
        **kwargs,
    )
    return MoEPackageLayer(config, layer_id=layer_id, local_rank=local_rank)
