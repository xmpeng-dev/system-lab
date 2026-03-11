"""
MoEX Layer：基于 CommTensor 的端到端 MoE Layer 实现

实现完整的 MoE Layer forward/backward：
  hidden_states [B*L, H]
    → Router（Gate GEMM + TopK，直写 CommTensor meta）
    → CommTensor（dispatch-ready，1次写入替代4次拷贝）
    → Dispatch（零拷贝 RDMA）
    → Expert GEMM（tile-level，Comet风格）
    → Combine（零拷贝 scatter_add）
    → hidden_states [B*L, H]

集成：
  - CommTensorPool：预分配避免 malloc 延迟
  - FSEP：Fully Sharded Expert Parallel（可选）
  - 负载均衡损失：与 LAER-MoE Load Planner 对接
  - Megatron-Core 接口兼容
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor

from .comm_tensor import (
    CommTensor,
    CommTensorConfig,
    CommTensorPool,
    load_balance_loss,
    route_to_comm_tensor,
)


# ---------------------------------------------------------------------------
# Expert FFN（专家前馈网络）
# ---------------------------------------------------------------------------

class ExpertFFN(nn.Module):
    """
    单个 Expert 的 FFN 模块

    结构：Linear(H → 4H) → SwiGLU/GeLU → Linear(4H → H)
    在 GroupedGEMM 中，多个 Expert 的 GEMM 合并执行。
    """

    def __init__(self, d_model: int, d_ffn: int, activation: str = 'swiglu'):
        super().__init__()
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.activation = activation

        if activation == 'swiglu':
            # SwiGLU：W_gate 和 W_up 合并，实际 d_ffn 是 2/3 配置值
            self.w_gate_up = nn.Linear(d_model, 2 * d_ffn, bias=False)
            self.w_down = nn.Linear(d_ffn, d_model, bias=False)
        else:
            self.w_up = nn.Linear(d_model, d_ffn, bias=False)
            self.w_down = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """x: [num_tokens, H] → [num_tokens, H]"""
        if self.activation == 'swiglu':
            gate_up = self.w_gate_up(x)          # [num_tokens, 2*d_ffn]
            gate, up = gate_up.chunk(2, dim=-1)   # [num_tokens, d_ffn] each
            hidden = nn.functional.silu(gate) * up
            return self.w_down(hidden)
        else:
            hidden = nn.functional.gelu(self.w_up(x))
            return self.w_down(hidden)


# ---------------------------------------------------------------------------
# Expert Engine（专家计算引擎，支持传统 EP 和 FSEP）
# ---------------------------------------------------------------------------

class ExpertEngine(nn.Module):
    """
    Expert 计算引擎

    支持两种模式：
    - 传统 EP：每 GPU 持有 experts_per_rank 个完整专家
    - FSEP：每 GPU 持有所有专家的 1/R 分片（需要 ReduceScatter）

    关键优化：GroupedGEMM（将多个小 GEMM 合并为一个 kernel 调用）
    """

    def __init__(
        self,
        config: CommTensorConfig,
        d_ffn: int,
        num_experts_total: int,
        activation: str = 'swiglu',
    ):
        super().__init__()
        self.config = config
        self.d_ffn = d_ffn
        self.num_experts_total = num_experts_total
        self.use_fsep = config.use_fsep

        # 本地专家数量
        self.local_num_experts = config.experts_per_rank

        if config.use_fsep:
            # FSEP 模式：持有所有专家的 1/R 份
            # W_up/W_gate: [num_experts_total, H, d_ffn/R]（按输出维度分片）
            # W_down: [num_experts_total, d_ffn, H/R]（按输入维度分片）
            fsep_dim = max(1, d_ffn // config.num_ep_ranks)
            self.w_up = nn.Parameter(
                torch.empty(num_experts_total, config.d_model, fsep_dim)
            )
            self.w_down = nn.Parameter(
                torch.empty(num_experts_total, fsep_dim, config.d_model)
            )
            nn.init.kaiming_uniform_(self.w_up, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.w_down, a=math.sqrt(5))
        else:
            # 传统 EP 模式：持有 local_num_experts 个完整专家
            self.experts = nn.ModuleList([
                ExpertFFN(config.d_model, d_ffn, activation)
                for _ in range(self.local_num_experts)
            ])

    def forward(
        self,
        input_ct: CommTensor,
        ep_group: Optional[dist.ProcessGroup] = None,
        fsep_group: Optional[dist.ProcessGroup] = None,
    ) -> CommTensor:
        """
        执行 Expert GEMM，返回输出 CommTensor

        Args:
            input_ct: 接收到的输入 CommTensor（来自 dispatch 后的本地数据）
            ep_group: EP 通信组（用于 combine 的 RDMA）
            fsep_group: FSEP 组（用于 ReduceScatter，节点内 NVLink）

        Returns:
            output CommTensor，已填充 expert 计算结果
        """
        output_ct = CommTensor.allocate(self.config, input_ct.data.device)
        output_ct.meta = input_ct.meta  # 复用路由元数据

        if self.use_fsep:
            return self._forward_fsep(input_ct, output_ct, fsep_group)
        else:
            return self._forward_ep(input_ct, output_ct)

    def _forward_ep(self, input_ct: CommTensor, output_ct: CommTensor) -> CommTensor:
        """
        传统 EP 模式的 Expert Forward

        每个 GPU 处理 local_num_experts 个专家，
        对应 input_ct 中 slot_counts[my_rank] 个 token。
        """
        my_rank = dist.get_rank() if dist.is_initialized() else 0
        config = self.config
        H = config.d_model

        # 获取本 rank 接收到的 token（来自其他 rank dispatch 过来的）
        # 注：在 EP 模式下，dispatch 后 input_ct 只有 my_rank 的数据有效
        count = input_ct.meta.slot_counts[my_rank].item()
        if count == 0:
            return output_ct

        # 获取 token 数据（[count, H]）
        tokens = input_ct.view_rank(my_rank).reshape(count, -1)[:, :H]

        # 确定每个 token 对应的 local expert ID
        # expert_id = token_indices[my_rank, :count] % experts_per_rank
        global_expert_ids = input_ct.meta.token_indices[my_rank, :count].long()
        local_expert_ids = global_expert_ids % self.local_num_experts

        # GroupedGEMM：按 expert 分组计算（避免小 GEMM overhead）
        output = self._grouped_gemm(tokens, local_expert_ids)  # [count, H]

        # 写入 output CommTensor
        T, tile = config.num_tiles, config.tile_size
        output_padded = torch.nn.functional.pad(
            output, (0, T * tile - H)
        ) if H < T * tile else output
        output_ct.data[my_rank, :count, :, :] = output_padded.view(count, T, tile)

        return output_ct

    def _forward_fsep(
        self,
        input_ct: CommTensor,
        output_ct: CommTensor,
        fsep_group: Optional[dist.ProcessGroup],
    ) -> CommTensor:
        """
        FSEP 模式的 Expert Forward

        1. 每 GPU 使用本地 W 分片计算所有 token 的 partial output
        2. ReduceScatter（NVLink）汇聚 partial outputs
        3. 写入 output CommTensor
        """
        config = self.config
        H = config.d_model
        total_tokens = input_ct.total_tokens()

        if total_tokens == 0:
            return output_ct

        # 获取所有 token（FSEP 模式下所有 GPU 都有所有 token）
        all_tokens = input_ct.view_all_tokens()  # [total_tokens, H]

        # 确定每个 token 的 expert ID
        all_expert_ids = []
        for r in range(config.num_ep_ranks):
            count = input_ct.meta.slot_counts[r].item()
            if count > 0:
                all_expert_ids.append(input_ct.meta.token_indices[r, :count].long())
        if not all_expert_ids:
            return output_ct
        expert_ids = torch.cat(all_expert_ids, dim=0)  # [total_tokens]

        # FSEP GEMM：使用本地 W 分片计算 partial output
        # W_up: [num_experts, H, d_ffn/R]
        # all_tokens: [total_tokens, H]
        # partial_hidden: [total_tokens, d_ffn/R]
        partial_hidden = self._fsep_gemm(all_tokens, expert_ids, self.w_up)
        partial_hidden = torch.nn.functional.silu(partial_hidden)  # 激活函数

        # partial_down: [total_tokens, H/R]（按 H 维分片）
        # 注：实际 FSEP 中 w_down 按 H 输出维分片
        partial_out = self._fsep_gemm(partial_hidden, expert_ids, self.w_down)

        # ReduceScatter（NVLink，节点内）
        # [total_tokens, H/R] × R → 每 GPU 获得 total_tokens/R 个 token 的完整 H
        if fsep_group is not None and dist.is_initialized():
            # 真实 ReduceScatter 实现
            full_out = torch.zeros(
                total_tokens // config.num_ep_ranks,
                H,
                dtype=partial_out.dtype,
                device=partial_out.device,
            )
            dist.reduce_scatter_tensor(full_out, partial_out, group=fsep_group)
        else:
            # 单 GPU 模式：直接使用 partial_out
            full_out = partial_out

        # 写入 output CommTensor
        T, tile = config.num_tiles, config.tile_size
        my_rank = dist.get_rank() if dist.is_initialized() else 0
        my_count = full_out.shape[0]
        if H < T * tile:
            full_out_padded = torch.nn.functional.pad(full_out, (0, T * tile - H))
        else:
            full_out_padded = full_out
        output_ct.data[my_rank, :my_count, :, :] = full_out_padded.view(my_count, T, tile)
        output_ct.meta.slot_counts[my_rank] = my_count

        return output_ct

    def _grouped_gemm(self, tokens: Tensor, local_expert_ids: Tensor) -> Tensor:
        """
        GroupedGEMM：将多个专家的 GEMM 合并

        朴素实现：按 expert ID 分组，分别计算，再合并
        优化实现：应使用 cuBLAS GroupedGEMM 或 Triton kernel
        """
        output = torch.zeros_like(tokens)

        for local_eid in range(self.local_num_experts):
            mask = local_expert_ids == local_eid
            if not mask.any():
                continue
            expert_tokens = tokens[mask]         # [n_e, H]
            expert_output = self.experts[local_eid](expert_tokens)  # [n_e, H]
            output[mask] = expert_output

        return output

    def _fsep_gemm(
        self,
        tokens: Tensor,          # [total_tokens, in_dim]
        expert_ids: Tensor,      # [total_tokens]
        weight: nn.Parameter,    # [num_experts, in_dim, out_dim]
    ) -> Tensor:
        """
        FSEP 模式的 GroupedGEMM（使用本地 W 分片）

        朴素实现，实际应使用 Triton kernel 优化
        """
        num_experts = weight.shape[0]
        out_dim = weight.shape[-1]
        output = torch.zeros(tokens.shape[0], out_dim,
                             dtype=tokens.dtype, device=tokens.device)

        for eid in range(num_experts):
            mask = expert_ids == eid
            if not mask.any():
                continue
            # tokens[mask]: [n_e, in_dim]
            # weight[eid]: [in_dim, out_dim]
            output[mask] = tokens[mask] @ weight[eid]

        return output


# ---------------------------------------------------------------------------
# MoEX Router（路由器）
# ---------------------------------------------------------------------------

class MoEXRouter(nn.Module):
    """
    MoEX 路由器

    Gate GEMM + TopK + CommTensor meta 填充（零额外 buffer）

    关键特性：
    - 路由结果直接写入 CommTensor meta（无中间 buffer）
    - 兼容 LAER-MoE Load Planner 接口
    - 支持辅助负载均衡损失
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        config: CommTensorConfig,
        capacity_factor: float = 1.25,
        z_loss_coeff: float = 1e-3,
        aux_loss_coeff: float = 1e-2,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.config = config
        self.capacity_factor = capacity_factor
        self.z_loss_coeff = z_loss_coeff
        self.aux_loss_coeff = aux_loss_coeff

        # Gate 权重
        self.gate = nn.Linear(d_model, num_experts, bias=False)

        # 初始化（小方差，稳定路由）
        nn.init.normal_(self.gate.weight, std=0.01)

    def forward(
        self,
        hidden_states: Tensor,         # [B*L, H]
        pool: Optional[CommTensorPool] = None,
    ) -> Tuple[CommTensor, Tensor]:
        """
        Args:
            hidden_states: [B*L, H]
            pool: CommTensor pool（可选，避免 malloc）

        Returns:
            (CommTensor, aux_loss)
            - CommTensor：dispatch-ready
            - aux_loss：负载均衡辅助损失（训练时加入总损失）
        """
        ct, gate_logits = route_to_comm_tensor(
            hidden_states=hidden_states,
            gate_weight=self.gate.weight.T,  # [H, num_experts]
            config=self.config,
            pool=pool,
        )

        # 计算辅助损失
        aux_loss = self._compute_aux_loss(gate_logits, ct)

        return ct, aux_loss

    def _compute_aux_loss(self, gate_logits: Tensor, ct: CommTensor) -> Tensor:
        """计算负载均衡辅助损失"""
        # Z-loss（来自 ST-MoE）：抑制 logit 过大，提升训练稳定性
        z_loss = torch.mean(torch.logsumexp(gate_logits, dim=-1) ** 2)

        # 辅助损失（来自 Switch Transformer）：
        # 从 gate_logits 直接计算，确保 expert_ids 在正确范围内
        routing_probs = torch.softmax(gate_logits, dim=-1)  # [B*L, num_experts]
        # 每个 expert 的平均路由概率
        mean_probs = routing_probs.mean(dim=0)  # [num_experts]
        # 每个 expert 接收的 token 比例（Top-1 用于计算）
        top1_ids = gate_logits.argmax(dim=-1)   # [B*L]
        expert_counts = torch.zeros(self.num_experts, device=gate_logits.device)
        expert_counts.scatter_add_(0, top1_ids, torch.ones_like(top1_ids, dtype=torch.float))
        expert_frac = expert_counts / expert_counts.sum()
        # Switch Transformer 损失：避免某些 expert 完全没有 token
        aux_loss_val = self.num_experts * (expert_frac * mean_probs).sum()

        return self.z_loss_coeff * z_loss + self.aux_loss_coeff * aux_loss_val


# ---------------------------------------------------------------------------
# MoEX Layer（完整 MoE Layer）
# ---------------------------------------------------------------------------

class MoEXLayer(nn.Module):
    """
    MoEX MoE Layer：完整的端到端实现

    以 CommTensor 为中心的 MoE Layer，实现：
    1. 路由（直写 CommTensor meta）
    2. Dispatch（零拷贝 RDMA）
    3. Expert GEMM（tile 级，FSEP 可选）
    4. Combine（零拷贝 scatter_add）

    兼容 Megatron-Core MoE Layer 接口（可直接替换）。
    """

    def __init__(
        self,
        config: CommTensorConfig,
        num_experts: int,
        d_ffn: int,
        activation: str = 'swiglu',
        capacity_factor: float = 1.25,
        aux_loss_coeff: float = 1e-2,
        pool_size: int = 4,
        process_groups: Optional[object] = None,  # MoEXProcessGroups
    ):
        super().__init__()

        self.config = config
        self.num_experts = num_experts
        self.process_groups = process_groups

        # 子模块
        self.router = MoEXRouter(
            d_model=config.d_model,
            num_experts=num_experts,
            config=config,
            capacity_factor=capacity_factor,
            aux_loss_coeff=aux_loss_coeff,
        )

        self.expert_engine = ExpertEngine(
            config=config,
            d_ffn=d_ffn,
            num_experts_total=num_experts,
            activation=activation,
        )

        # CommTensor 预分配池（消除 dispatch 时的 malloc 延迟）
        self.pool = CommTensorPool(config, pool_size=pool_size)

        # CUDA Streams（分离 compute 和 comm）
        if torch.cuda.is_available():
            self.compute_stream = torch.cuda.current_stream()
            self.comm_stream = torch.cuda.Stream()
        else:
            self.compute_stream = None
            self.comm_stream = None

    def forward(
        self,
        hidden_states: Tensor,   # [B*L, H]
        return_aux_loss: bool = True,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        MoEX Layer Forward Pass

        Args:
            hidden_states: [B, L, H] 或 [B*L, H]（自动处理）
            return_aux_loss: 是否返回辅助损失

        Returns:
            (output, aux_loss)
            - output: [B*L, H]，与输入同形状
            - aux_loss: 负载均衡损失（return_aux_loss=True 时）
        """
        # 处理 3D 输入
        original_shape = hidden_states.shape
        if hidden_states.dim() == 3:
            B, L, H = hidden_states.shape
            hidden_states = hidden_states.view(B * L, H)
        else:
            assert hidden_states.dim() == 2

        B_L, H = hidden_states.shape
        output = torch.zeros_like(hidden_states)

        # Step 1: Route（Gate GEMM + TopK，直写 CommTensor meta）
        # 使用 pool 避免 malloc：acquire → fill → release
        ct, aux_loss = self.router(hidden_states, pool=self.pool)

        # Step 2: Dispatch（零拷贝 RDMA，异步）
        # 在 comm_stream 上执行，与 compute 流程并行
        ep_group = getattr(self.process_groups, 'ep_group', None) if self.process_groups else None
        ct.dispatch_async(ep_group, self.comm_stream)

        # Step 3: Expert GEMM（tile-level，Comet 风格）
        # 等待 dispatch 完成（CUDA event sync），然后计算
        # 注：实际中，tile-level GEMM 可以在接收到第一个 tile 后就开始
        fsep_group = getattr(self.process_groups, 'fsep_group', None) if self.process_groups else None
        expert_output_ct = self.expert_engine(ct, ep_group, fsep_group)

        # Step 4: Combine（零拷贝 scatter_add，无中间 buffer）
        output = ct.combine_into(output, expert_output_ct)

        # 归还 CommTensor 到 pool
        self.pool.release(ct)
        # 注：expert_output_ct 是从 ExpertEngine 内部分配的临时 CT，
        # 在真实实现中应从独立的 expert_output_pool 管理。
        # 当前原型中不做额外处理（小批量测试无显著内存影响）。

        # 还原原始形状
        if len(original_shape) == 3:
            output = output.view(original_shape)

        if return_aux_loss:
            return output, aux_loss
        return output, None

    def extra_repr(self) -> str:
        return (
            f'num_experts={self.num_experts}, '
            f'd_model={self.config.d_model}, '
            f'ep_size={self.config.num_ep_ranks}, '
            f'use_fsep={self.config.use_fsep}, '
            f'tile_size={self.config.tile_size}'
        )


# ---------------------------------------------------------------------------
# MoEX Block（含 Attention 的完整 Transformer Block）
# ---------------------------------------------------------------------------

class MoEXBlock(nn.Module):
    """
    MoEX Transformer Block：Attention + MoE FFN

    支持 Parallel Folding（Attention/MoE 独立并行配置）。
    在 Attention 输出后，CommTensor 的构建与 TP All-Reduce 并行执行。
    """

    def __init__(
        self,
        config: CommTensorConfig,
        num_experts: int,
        d_ffn: int,
        num_heads: int,
        activation: str = 'swiglu',
        pool_size: int = 4,
    ):
        super().__init__()
        self.config = config
        H = config.d_model

        # Layer Norms
        self.attn_norm = nn.LayerNorm(H)
        self.ffn_norm = nn.LayerNorm(H)

        # Self-Attention（简化版）
        self.attn = nn.MultiheadAttention(H, num_heads, batch_first=True)

        # MoEX FFN
        self.moe = MoEXLayer(
            config=config,
            num_experts=num_experts,
            d_ffn=d_ffn,
            activation=activation,
            pool_size=pool_size,
        )

    def forward(
        self,
        x: Tensor,              # [B, L, H]
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Block Forward：Pre-Norm + Attention + Pre-Norm + MoE

        Returns:
            (output, aux_loss)
        """
        # Attention（残差连接）
        residual = x
        x = self.attn_norm(x)
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = residual + attn_out

        # MoEX FFN（残差连接）
        residual = x
        ffn_input = self.ffn_norm(x)
        ffn_out, aux_loss = self.moe(ffn_input)
        x = residual + ffn_out

        return x, aux_loss


# ---------------------------------------------------------------------------
# 工厂函数（便于快速创建）
# ---------------------------------------------------------------------------

def create_moex_layer(
    num_experts: int,
    d_model: int,
    d_ffn: int,
    ep_size: int,
    top_k: int = 2,
    tile_size: int = 128,
    max_tokens: int = 4096,
    use_fsep: bool = False,
    activation: str = 'swiglu',
    dtype: torch.dtype = torch.float16,
) -> MoEXLayer:
    """
    便捷工厂函数：创建 MoEX Layer

    Example:
        >>> layer = create_moex_layer(
        ...     num_experts=64,
        ...     d_model=4096,
        ...     d_ffn=14336,
        ...     ep_size=8,
        ...     top_k=2,
        ... )
        >>> hidden = torch.randn(4096, 4096)
        >>> output, aux_loss = layer(hidden)
    """
    config = CommTensorConfig(
        num_ep_ranks=ep_size,
        d_model=d_model,
        max_tokens_per_step=max_tokens,
        top_k=top_k,
        tile_size=tile_size,
        dtype=dtype,
        use_fsep=use_fsep,
        experts_per_rank=max(1, num_experts // ep_size),
    )

    return MoEXLayer(
        config=config,
        num_experts=num_experts,
        d_ffn=d_ffn,
        activation=activation,
    )
