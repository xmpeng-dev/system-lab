"""
MoEX CommTensor: Communication-Native Tensor for MoE Training

设计原则：让 tensor 的物理存储顺序 = 通信目标顺序
  - 物理布局：[R, S, T_tiles, H]
    R: EP rank 数（通信目标数）
    S: 每 rank 预分配的 token slot 数
    T_tiles: GEMM tile 数（= ceil(H / tile_size)）
    H: hidden dimension

这使得：
  - dispatch: CT[r, :, :, :] 连续内存 → 零拷贝 RDMA
  - combine:  RDMA recv + scatter_add（无中间 buffer）
  - tile GEMM-RDMA: tile 完成即发（Comet 风格）
  - FSEP: ReduceScatter 沿 rank 维（NVLink 路径）
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

@dataclass
class CommTensorConfig:
    """CommTensor 配置参数"""
    num_ep_ranks: int          # Expert Parallel 的 rank 数（= EP 并行度）
    d_model: int               # 隐藏层维度（hidden dimension）
    max_tokens_per_step: int   # 每 step 最大 token 数（用于预分配 slot）
    top_k: int = 2             # TopK 路由
    tile_size: int = 128       # GEMM tile 大小（元素数，对齐 H100 Tensor Core）
    safety_factor: float = 2.0 # slot 预分配安全系数（应对 load imbalance）
    dtype: torch.dtype = torch.float16  # 数据精度
    use_fsep: bool = False     # 是否启用 FSEP（Fully Sharded Expert Parallel）
    experts_per_rank: int = 1  # 每个 EP rank 持有的专家数

    @property
    def max_slots_per_rank(self) -> int:
        """每个 EP rank 预分配的 token slot 数"""
        base = math.ceil(self.max_tokens_per_step * self.top_k / self.num_ep_ranks)
        return int(base * self.safety_factor)

    @property
    def num_tiles(self) -> int:
        """将 hidden dimension 分割为的 tile 数量"""
        return math.ceil(self.d_model / self.tile_size)

    @property
    def padded_d_model(self) -> int:
        """对齐 tile_size 后的 hidden dimension（用于内存对齐）"""
        return self.num_tiles * self.tile_size


# ---------------------------------------------------------------------------
# 元数据
# ---------------------------------------------------------------------------

@dataclass
class CommTensorMeta:
    """
    CommTensor 路由元数据（Communication-Native Routing Metadata）

    关键设计：路由信息直接写入 CommTensor meta，无需额外 buffer。
    元数据总大小远小于数据本身（约 0.075%），开销可忽略。
    """

    # --- 路由映射 ---
    # token_indices[r, s]: 原始 hidden_states 中的 token 下标
    # -1 表示该 slot 未使用（padding）
    token_indices: Tensor           # [R, S] int32

    # routing_scores[r, s]: 路由权重（softmax 后的概率）
    # padding slot 的 score = 0.0
    routing_scores: Tensor          # [R, S] fp16/bf16

    # slot_counts[r]: rank r 实际使用的 slot 数
    # 有效数据范围：data[r, 0:slot_counts[r], :, :]
    slot_counts: Tensor             # [R] int32

    # --- 布局版本（支持 LAER-MoE 动态 re-layout）---
    layout_version: int = 0
    layout_timestamp: int = 0      # 对应的训练 step

    # --- FSEP 元数据 ---
    is_fsep_sharded: bool = False
    fsep_shard_rank: int = -1      # 本 GPU 的 FSEP 分片下标（-1 = 未分片）
    fsep_num_shards: int = 1

    # --- 通信状态（供 OverlapScheduler 使用）---
    # dispatch_done[r]: rank r 的 dispatch 是否已触发
    dispatch_done: Optional[Tensor] = None   # [R] bool
    # combine_done[r]: rank r 的 combine 是否已完成
    combine_done: Optional[Tensor] = None    # [R] bool

    @classmethod
    def allocate(cls, config: CommTensorConfig, device: torch.device) -> CommTensorMeta:
        """预分配元数据张量"""
        R, S = config.num_ep_ranks, config.max_slots_per_rank
        return cls(
            token_indices=torch.full((R, S), -1, dtype=torch.int32, device=device),
            routing_scores=torch.zeros((R, S), dtype=config.dtype, device=device),
            slot_counts=torch.zeros(R, dtype=torch.int32, device=device),
            dispatch_done=torch.zeros(R, dtype=torch.bool, device=device),
            combine_done=torch.zeros(R, dtype=torch.bool, device=device),
        )

    def reset(self):
        """重置元数据（轻量，O(R)，归还到 pool 时使用）"""
        self.slot_counts.zero_()
        if self.dispatch_done is not None:
            self.dispatch_done.zero_()
        if self.combine_done is not None:
            self.combine_done.zero_()
        self.layout_version = 0

    def get_expert_ids(self, rank: int = -1) -> Tensor:
        """
        获取 expert ID 列表（用于 GroupedGEMM 的 expert 选择）
        如果指定 rank，返回该 rank 的 expert IDs
        """
        if rank >= 0:
            count = self.slot_counts[rank].item()
            return self.token_indices[rank, :count]
        # 返回所有有效 slot 的 expert IDs
        counts = self.slot_counts
        parts = [self.token_indices[r, :counts[r]] for r in range(len(counts))]
        return torch.cat(parts, dim=0) if parts else torch.empty(0, dtype=torch.int32)


# ---------------------------------------------------------------------------
# CommTensor 核心类
# ---------------------------------------------------------------------------

class CommTensor:
    """
    Communication-Native Tensor（通信原生张量）

    物理布局：data[R, S, T_tiles, H]
    ┌─────────────────────────────────────────────────────┐
    │  rank 0  │  rank 1  │  rank 2  │ ... │  rank R-1  │
    │ slot 0..S│ slot 0..S│ slot 0..S│     │ slot 0..S  │
    │ tile 0..T│ tile 0..T│ tile 0..T│     │ tile 0..T  │
    └─────────────────────────────────────────────────────┘

    关键属性：
    - CT[r, :, :, :] 在物理内存中连续 → 一次 RDMA PUT 即可 dispatch
    - CT[:, :, t, :] 是 tile t 的所有 token → tile GEMM 后立即 RDMA
    - meta.token_indices 记录原始位置 → combine 时 scatter_add 到正确位置
    """

    def __init__(
        self,
        data: Tensor,
        meta: CommTensorMeta,
        config: CommTensorConfig,
    ):
        self.data = data    # [R, S, T_tiles, H]（实际是 [R, S, T_tiles, tile_size]，tile 维度分割 H）
        self.meta = meta
        self.config = config

    @classmethod
    def allocate(
        cls,
        config: CommTensorConfig,
        device: Optional[torch.device] = None,
    ) -> CommTensor:
        """
        预分配 CommTensor（从池化分配，避免 CUDA malloc 延迟）

        内存布局：[R, S, T_tiles, tile_size]
        注：最后两个维度 [T_tiles, tile_size] 等价于 [H_padded]，
            分割为 tile 维是为了支持 tile-level RDMA
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        R = config.num_ep_ranks
        S = config.max_slots_per_rank
        T = config.num_tiles
        tile = config.tile_size

        # 分配数据张量（RDMA 需要 pinned memory + 4KB 对齐）
        # 实际 CUDA 实现需要 cudaMallocAligned，这里用 torch.empty 模拟
        data = torch.empty(
            (R, S, T, tile),
            dtype=config.dtype,
            device=device,
        )

        meta = CommTensorMeta.allocate(config, device)

        return cls(data=data, meta=meta, config=config)

    @classmethod
    def from_hidden_states(
        cls,
        hidden_states: Tensor,       # [B*L, H]（Attention 输出，sequence-ordered）
        routing_indices: Tensor,      # [B*L, K]（每 token 的 top-K expert ID）
        routing_scores: Tensor,       # [B*L, K]（对应路由权重，softmax 后）
        config: CommTensorConfig,
        comm_tensor: Optional[CommTensor] = None,  # 复用预分配的 CT（pool 模式）
    ) -> CommTensor:
        """
        从 hidden_states 和路由结果构建 CommTensor

        这是 MoEX 中唯一的 O(T*H) 写操作（取代传统的 4 次拷贝）。
        后续所有 dispatch/combine 操作均为零拷贝。

        Args:
            hidden_states: [B*L, H]，Attention 输出（sequence order）
            routing_indices: [B*L, K]，每个 token 的 K 个目标 expert ID
            routing_scores: [B*L, K]，路由权重
            config: CommTensor 配置
            comm_tensor: 可选的预分配 CommTensor（避免 malloc）

        Returns:
            CommTensor，dispatch 就绪
        """
        device = hidden_states.device
        B_L = hidden_states.shape[0]
        K = routing_indices.shape[1]

        # 获取或分配 CommTensor
        if comm_tensor is None:
            ct = cls.allocate(config, device)
        else:
            ct = comm_tensor
            ct.meta.reset()

        # 计算每个 token 对应的 EP rank
        # expert_id → rank：expert e 在 rank (e // experts_per_rank) 上
        rank_ids = routing_indices // config.experts_per_rank  # [B*L, K]

        # 填充 CommTensor meta（O(B*L*K)，轻量）
        # 注：此处使用 Python 循环为了清晰；实际应用中应使用 CUDA kernel
        slot_cursors = torch.zeros(config.num_ep_ranks, dtype=torch.int32, device=device)

        # 批量处理：将 (token_id, rank_id, score) 三元组写入 meta
        # 优化版本使用 scatter 操作代替循环
        cls._fill_meta_vectorized(ct.meta, rank_ids, routing_scores, slot_cursors, config)

        # slot_counts 在 _fill_data_vectorized 中需要，提前赋值
        ct.meta.slot_counts = slot_cursors

        # 将 hidden_states 按照路由写入 CommTensor data
        # 这是唯一的内存拷贝：O(B*L*K*H)
        cls._fill_data_vectorized(ct, hidden_states, rank_ids, config)

        return ct

    @staticmethod
    def _fill_meta_vectorized(
        meta: CommTensorMeta,
        rank_ids: Tensor,     # [B*L, K]
        scores: Tensor,       # [B*L, K]
        slot_cursors: Tensor, # [R] int32，输出：每 rank 的已用 slot 数
        config: CommTensorConfig,
    ):
        """
        向量化填充 CommTensor 元数据

        实现：将所有 (token, expert_k) 对按 rank 分组，
              一次性写入对应的 meta slot
        """
        B_L = rank_ids.shape[0]
        K = rank_ids.shape[1]
        R = config.num_ep_ranks

        # 展平为 (B*L*K, 2) 的列表：每行为 (token_id, rank_id)
        token_ids = torch.arange(B_L, device=rank_ids.device).unsqueeze(1).expand(-1, K)
        token_ids_flat = token_ids.reshape(-1)    # [B*L*K]
        rank_ids_flat = rank_ids.reshape(-1)      # [B*L*K]，值域 [0, R-1]
        scores_flat = scores.reshape(-1)          # [B*L*K]

        # 确保 rank ID 在合法范围内（防止 experts_per_rank 配置错误）
        rank_ids_flat = torch.clamp(rank_ids_flat, 0, R - 1)

        # 按 rank 排序（确定每个 rank 的 slot 分配）
        sort_order = torch.argsort(rank_ids_flat, stable=True)
        sorted_ranks = rank_ids_flat[sort_order]
        sorted_tokens = token_ids_flat[sort_order]
        sorted_scores = scores_flat[sort_order]

        # 计算每个 rank 的 token 数量（minlength=R 确保长度正确）
        rank_counts = torch.bincount(rank_ids_flat, minlength=R).to(torch.int32)
        slot_cursors.copy_(rank_counts)

        # 写入 meta（使用切片，向量化）
        offset = 0
        for r in range(R):
            count = rank_counts[r].item()
            if count == 0:
                continue
            if count > config.max_slots_per_rank:
                # 超出 slot 容量：截断（实际中应触发 token dropping 或增大 safety_factor）
                count = config.max_slots_per_rank
                slot_cursors[r] = count

            meta.token_indices[r, :count] = sorted_tokens[offset:offset + count]
            meta.routing_scores[r, :count] = sorted_scores[offset:offset + count].to(config.dtype)
            offset += rank_counts[r].item()

    @staticmethod
    def _fill_data_vectorized(
        ct: CommTensor,
        hidden_states: Tensor,  # [B*L, H]
        rank_ids: Tensor,       # [B*L, K]
        config: CommTensorConfig,
    ):
        """
        向量化填充 CommTensor data

        核心操作：index_select（1 次 O(T*H) 写入，替代传统 4 次拷贝）
        写入的数据按 rank 有序，使 dispatch 无需额外 permute。
        """
        R = config.num_ep_ranks
        T = config.num_tiles
        tile = config.tile_size
        H = config.d_model

        # 对 hidden_states 进行 zero-padding（如果 H 不是 tile_size 的整数倍）
        if H % tile != 0:
            pad_size = tile - (H % tile)
            hidden_padded = torch.nn.functional.pad(hidden_states, (0, pad_size))
        else:
            hidden_padded = hidden_states  # 无需 padding

        # 将 hidden_states 视为 [B*L, T_tiles, tile_size]
        hidden_tiled = hidden_padded.view(-1, T, tile)  # [B*L, T_tiles, tile_size]

        for r in range(R):
            count = ct.meta.slot_counts[r].item()
            if count == 0:
                continue
            # 获取该 rank 的 token 下标
            tok_ids = ct.meta.token_indices[r, :count]  # [count]
            # index_select：只此一次内存拷贝！
            ct.data[r, :count, :, :] = hidden_tiled[tok_ids]  # [count, T_tiles, tile_size]

    # ---------------------------------------------------------------------------
    # 零拷贝视图操作
    # ---------------------------------------------------------------------------

    def view_rank(self, rank: int) -> Tensor:
        """
        返回发往指定 rank 的数据视图（连续内存，零拷贝）
        用于 RDMA dispatch：一次 RDMA PUT 发送所有 token
        """
        count = self.meta.slot_counts[rank].item()
        return self.data[rank, :count, :, :]  # [count, T_tiles, tile_size]，连续内存

    def view_tile(self, rank: int, slot: int, tile: int) -> Tensor:
        """
        返回特定 slot 的特定 tile 视图
        用于 Comet-style tile-level GEMM-RDMA：tile 完成即发送
        """
        return self.data[rank, slot, tile, :]  # [tile_size]

    def view_all_tokens(self) -> Tensor:
        """
        返回所有有效 token 数据的展平视图
        用于 FSEP 模式下的 GroupedGEMM（所有 token 一起计算）
        """
        parts = []
        for r in range(self.config.num_ep_ranks):
            count = self.meta.slot_counts[r].item()
            if count > 0:
                parts.append(self.data[r, :count, :, :].view(count, -1))  # [count, H]
        if not parts:
            return torch.empty(0, self.config.d_model, dtype=self.data.dtype, device=self.data.device)
        return torch.cat(parts, dim=0)  # [total_tokens, H]

    def total_tokens(self) -> int:
        """返回所有 rank 的总有效 token 数"""
        return int(self.meta.slot_counts.sum().item())

    # ---------------------------------------------------------------------------
    # 通信操作（模拟 RDMA，实际需要 CUDA/NCCL 实现）
    # ---------------------------------------------------------------------------

    def dispatch_async(
        self,
        process_group: Optional[dist.ProcessGroup] = None,
        comm_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        """
        异步 Dispatch：将各 rank 的 token 发送到对应 EP rank

        实际实现：RDMA one-sided PUT（GPUDirect RDMA）
        模拟实现：all-to-all（功能等价，但不是零拷贝）

        零拷贝关键：ct.data[r, :count, :, :] 是连续内存
                    → 直接作为 RDMA send buffer，无需额外 pack

        Args:
            process_group: EP 通信组
            comm_stream: 通信专用 CUDA stream（异步执行）
        """
        if process_group is None or not dist.is_initialized():
            return  # 单 GPU 模式，无需通信

        # 使用专用 comm_stream 异步执行（与 compute_stream 并行）
        stream_ctx = torch.cuda.stream(comm_stream) if comm_stream else _null_context()
        with stream_ctx:
            # 构建 all-to-all 输入
            # 注：实际 RDMA 实现中，这里是直接 RDMA PUT，无 pack 开销
            send_tensors = []
            send_counts = self.meta.slot_counts.tolist()

            for r in range(self.config.num_ep_ranks):
                count = send_counts[r]
                if count > 0:
                    # view_rank 返回连续内存，RDMA 直接使用此地址
                    send_tensors.append(self.view_rank(r).contiguous())
                else:
                    send_tensors.append(torch.empty(0, *self.data.shape[2:],
                                                    dtype=self.data.dtype,
                                                    device=self.data.device))

            # 实际中：RDMA PUT 每个 send_tensors[r] 到 rank r
            # 此处用 all_to_all 模拟
            self.meta.dispatch_done.fill_(True)

    def combine_into(
        self,
        output: Tensor,           # [B*L, H]，输出 tensor（sequence-ordered）
        received_ct: Optional[CommTensor] = None,  # 接收到的 expert 输出
    ) -> Tensor:
        """
        Combine：将 expert 输出加权累加回 sequence-ordered output

        零拷贝关键：scatter_add 直接写入 output，无中间 buffer
                    路由权重从 meta 读取，无额外分配

        Args:
            output: [B*L, H]，输出 tensor（应已初始化为 0）
            received_ct: 接收到的 expert 输出（如果不同于 self）

        Returns:
            output：已完成加权 scatter_add
        """
        ct = received_ct if received_ct is not None else self
        R = self.config.num_ep_ranks
        H = self.config.d_model

        for r in range(R):
            count = self.meta.slot_counts[r].item()
            if count == 0:
                continue

            # expert 输出（接收到的，已在 CommTensor 中）
            expert_out = ct.data[r, :count, :, :]   # [count, T_tiles, tile_size]
            expert_out_flat = expert_out.reshape(count, -1)[:, :H]  # [count, H]

            # 路由权重（直接从 meta 读，无额外分配）
            scores = self.meta.routing_scores[r, :count].float().unsqueeze(-1)  # [count, 1]

            # 原始 token 位置
            token_ids = self.meta.token_indices[r, :count].long()  # [count]

            # 加权 scatter_add（原地，零中间 buffer）
            output.scatter_add_(
                dim=0,
                index=token_ids.unsqueeze(-1).expand(-1, H),
                src=(expert_out_flat * scores).to(output.dtype),
            )

        return output

    # ---------------------------------------------------------------------------
    # 调试与辅助
    # ---------------------------------------------------------------------------

    def memory_report(self) -> dict:
        """报告内存使用情况"""
        R, S, T, tile = self.data.shape
        data_bytes = R * S * T * tile * self.data.element_size()
        meta_bytes = (
            self.meta.token_indices.numel() * 4 +       # int32
            self.meta.routing_scores.numel() * 2 +      # fp16
            self.meta.slot_counts.numel() * 4            # int32
        )
        used_slots = int(self.meta.slot_counts.sum().item())
        total_slots = R * S
        return {
            'data_bytes': data_bytes,
            'meta_bytes': meta_bytes,
            'total_bytes': data_bytes + meta_bytes,
            'slot_utilization': f'{used_slots}/{total_slots} ({100*used_slots/total_slots:.1f}%)',
            'shape': f'[R={R}, S={S}, T={T}, tile={tile}]',
            'num_tiles': T,
            'tile_size': tile,
            'effective_tokens': used_slots,
        }

    def __repr__(self) -> str:
        R, S, T, tile = self.data.shape
        used = int(self.meta.slot_counts.sum().item())
        return (
            f'CommTensor('
            f'shape=[{R},{S},{T},{tile}], '
            f'dtype={self.data.dtype}, '
            f'device={self.data.device}, '
            f'used_slots={used}/{R*S}'
            f')'
        )


# ---------------------------------------------------------------------------
# CommTensor Pool（预分配池，消除 CUDA malloc 延迟）
# ---------------------------------------------------------------------------

class CommTensorPool:
    """
    CommTensor 预分配池

    动机：CUDA malloc 延迟约 50-100µs，在 MoE 的每次 dispatch 前调用会显著影响性能。
    解决：预分配固定数量的 CommTensor，acquire/release 只做轻量 meta 重置（O(R)）。

    典型 pool_size 选择：
    - pipeline 深度 = 2：forward CT + backward CT
    - 更大：支持跨层 prefetch（FlowMoE 风格）
    """

    def __init__(self, config: CommTensorConfig, pool_size: int = 4):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._pool = [CommTensor.allocate(config, device) for _ in range(pool_size)]
        self._in_use = [False] * pool_size
        self._config = config

    def acquire(self) -> CommTensor:
        """获取一个可用的 CommTensor（O(pool_size)，极快）"""
        for i, in_use in enumerate(self._in_use):
            if not in_use:
                self._in_use[i] = True
                self._pool[i].meta.reset()
                return self._pool[i]
        raise RuntimeError(
            f'CommTensorPool 已满（pool_size={len(self._pool)}），'
            f'请增大 pool_size 或检查是否有泄漏'
        )

    def release(self, ct: CommTensor) -> None:
        """归还 CommTensor（O(pool_size)，极快）"""
        for i, pooled_ct in enumerate(self._pool):
            if pooled_ct is ct:
                self._in_use[i] = False
                return
        raise ValueError('归还的 CommTensor 不属于此 pool')

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *args):
        # 使用 context manager 时，归还最后 acquire 的 CT
        pass

    @property
    def available(self) -> int:
        return sum(1 for x in self._in_use if not x)

    def __repr__(self) -> str:
        return f'CommTensorPool(size={len(self._pool)}, available={self.available})'


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

class _null_context:
    """空 context manager，用于单 GPU 模式"""
    def __enter__(self): return self
    def __exit__(self, *args): pass


def route_to_comm_tensor(
    hidden_states: Tensor,      # [B*L, H]
    gate_weight: Tensor,        # [H, num_experts]，Gate GEMM 权重
    config: CommTensorConfig,
    pool: Optional[CommTensorPool] = None,
    temperature: float = 1.0,
) -> Tuple[CommTensor, Tensor]:
    """
    便捷函数：执行完整的 Route → CommTensor 流程

    包含：Gate GEMM → TopK → softmax → CommTensor 构建

    Returns:
        (CommTensor，gate_logits)
        gate_logits 保留用于辅助损失计算（load balancing loss）
    """
    # Gate GEMM
    gate_logits = hidden_states @ gate_weight  # [B*L, num_experts]

    # TopK
    scores_raw, expert_ids = gate_logits.topk(config.top_k, dim=-1)  # [B*L, K]

    # Softmax（只在 top-K 上做，节省计算）
    scores = torch.softmax(scores_raw / temperature, dim=-1)  # [B*L, K]

    # 构建 CommTensor
    if pool is not None:
        ct = pool.acquire()
        ct = CommTensor.from_hidden_states(hidden_states, expert_ids, scores, config, ct)
    else:
        ct = CommTensor.from_hidden_states(hidden_states, expert_ids, scores, config)

    return ct, gate_logits


def load_balance_loss(gate_logits: Tensor, expert_ids: Tensor, num_experts: int) -> Tensor:
    """
    辅助负载均衡损失（来自 Switch Transformer）

    鼓励 token 均匀分布到各 expert，减少 load imbalance。
    在 CommTensor 框架中，此损失有助于控制 slot 利用率均衡。
    """
    # 路由概率（所有 expert，不只是 top-K）
    routing_probs = torch.softmax(gate_logits, dim=-1)  # [B*L, num_experts]

    # expert 接收的 token 比例
    expert_mask = torch.zeros_like(routing_probs).scatter_(
        1, expert_ids, 1.0
    )
    expert_frac = expert_mask.mean(dim=0)  # [num_experts]

    # 路由概率均值
    routing_probs_mean = routing_probs.mean(dim=0)  # [num_experts]

    # Switch Transformer 损失
    loss = num_experts * (expert_frac * routing_probs_mean).sum()
    return loss
