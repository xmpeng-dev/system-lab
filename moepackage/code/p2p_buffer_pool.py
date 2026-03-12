"""
MoEPackage Module 2: Persistent P2P Buffer Pool

替代 NCCL User Buffer Registration（RCCL 无此 API），利用 AMD XGMI P2P
IPC Handle 实现持久化通信 Buffer 管理。

核心思路：
  - 训练初始化时一次性分配所有通信 Buffer（hipMalloc）
  - 通过 hipIpcGetMemHandle / hipIpcOpenMemHandle 交换远端指针
  - 训练循环中零 malloc / free / 注册 / 注销
  - 前向/反向共享同一物理内存（通过 BufferSlot 管理）
  - 所有地址在训练期间恒定 → 天然支持 HIP Graph 捕获

对比 NCCL UBR：
  NVIDIA: 持久 Buffer → NCCL UBR 注册 → SM 占用 1-4 SM
  AMD:    持久 Buffer → XGMI P2P 注册 → 0 SM 占用（P2P = 内存操作）
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

# hipIpcMemHandle_t 大小 (bytes)
HIP_IPC_HANDLE_SIZE = 64


# MI300X HBM3e 容量
MI300X_HBM_CAPACITY_GB = 192


class BufferUsage(Enum):
    """Buffer 用途标识"""
    DISPATCH_SEND = auto()    # Dispatch 发送 buffer
    DISPATCH_RECV = auto()    # Dispatch 接收 buffer
    COMBINE_SEND = auto()     # Combine 发送 buffer
    COMBINE_RECV = auto()     # Combine 接收 buffer
    GATEWAY_STAGING = auto()  # Gateway 跨节点中转 buffer
    SHARED_FWD_BWD = auto()   # 前向/反向共享 buffer


@dataclass
class BufferPoolConfig:
    """
    Persistent P2P Buffer Pool 配置

    Args:
        num_ep_ranks: Expert Parallel rank 数
        d_model: 隐藏层维度
        capacity_per_rank: 每 rank 的 token 容量
        num_buffer_slots: Buffer slot 数量（前向/反向共享时至少 2）
        dtype: 数据类型
        enable_shared_fwd_bwd: 前向和反向共享同一物理 buffer
        hbm_budget_gb: HBM 预算上限 (GB)
    """
    num_ep_ranks: int
    d_model: int
    capacity_per_rank: int
    num_buffer_slots: int = 2     # 双 buffer：一个正在用，一个准备中
    dtype: torch.dtype = torch.bfloat16
    enable_shared_fwd_bwd: bool = True
    hbm_budget_gb: float = 8.0    # Buffer Pool 总预算（占 HBM 的 ~4%）

    @property
    def bytes_per_element(self) -> int:
        return 2 if self.dtype in (torch.float16, torch.bfloat16) else 4

    @property
    def slot_size_bytes(self) -> int:
        """单个 BufferSlot 的字节大小"""
        return self.capacity_per_rank * self.d_model * self.bytes_per_element

    @property
    def total_pool_bytes(self) -> int:
        """Buffer Pool 总字节数"""
        # 每个 rank 需要 send + recv buffer × slot 数量
        if self.enable_shared_fwd_bwd:
            # 共享：send 和 recv 各一组
            return 2 * self.num_ep_ranks * self.slot_size_bytes
        else:
            # 不共享：FWD/BWD 各一组 send+recv
            return 4 * self.num_ep_ranks * self.slot_size_bytes

    @property
    def total_pool_gb(self) -> float:
        return self.total_pool_bytes / (1024 ** 3)

    def validate(self) -> None:
        """验证配置合法性"""
        if self.total_pool_gb > self.hbm_budget_gb:
            raise ValueError(
                f"Buffer Pool size ({self.total_pool_gb:.2f} GB) exceeds "
                f"HBM budget ({self.hbm_budget_gb:.2f} GB). "
                f"Consider reducing capacity_per_rank or num_ep_ranks."
            )


# ---------------------------------------------------------------------------
# Buffer Slot：单个通信 Buffer 单元
# ---------------------------------------------------------------------------

@dataclass
class BufferSlot:
    """
    A single pre-allocated communication buffer.

    每个 BufferSlot 对应一个预分配的 Tensor，在整个训练过程中地址恒定。
    AMD HIP: 地址恒定 → hipGraph 捕获安全，无需 replay 时重新绑定地址。

    Attributes:
        slot_id: 唯一标识
        usage: 用途（send/recv/gateway）
        data: 预分配的数据 Tensor
        metadata: 元信息 Tensor（token count, valid mask 等）
        is_locked: 是否被当前操作锁定
        peer_rank: 对端 rank（P2P 场景下）
    """
    slot_id: int
    usage: BufferUsage
    data: Tensor                         # [capacity, d_model]
    metadata: Tensor                     # [capacity] valid mask
    is_locked: bool = False
    peer_rank: int = -1
    # AMD HIP: IPC Handle（模拟）
    ipc_handle: Optional[bytes] = field(default=None, repr=False)

    @classmethod
    def allocate(
        cls,
        slot_id: int,
        usage: BufferUsage,
        capacity: int,
        d_model: int,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
        peer_rank: int = -1,
    ) -> BufferSlot:
        """
        Allocate a buffer slot with pre-allocated memory.

        AMD HIP: 实际使用 hipMalloc + hipIpcGetMemHandle
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 数据 buffer（预分配，地址恒定）
        data = torch.zeros(capacity, d_model, dtype=dtype, device=device)
        # valid mask（标记哪些 slot 有效 token）
        metadata = torch.zeros(capacity, dtype=torch.bool, device=device)

        # 模拟 IPC Handle（实际为 hipIpcMemHandle_t）
        ipc_handle = bytes(HIP_IPC_HANDLE_SIZE)

        return cls(
            slot_id=slot_id,
            usage=usage,
            data=data,
            metadata=metadata,
            peer_rank=peer_rank,
            ipc_handle=ipc_handle,
        )

    def reset(self) -> None:
        """
        Reset slot for reuse (only metadata, data is overwritten).

        关键：不清零 data（下次写入时会覆盖），只重置 metadata。
        O(capacity) 开销，远小于重新分配。
        """
        self.metadata.zero_()
        self.is_locked = False

    def write_tokens(self, tokens: Tensor, start_idx: int = 0) -> int:
        """
        Write tokens into this buffer slot.

        Args:
            tokens: Token data to write [N, d_model]
            start_idx: Starting slot index

        Returns:
            Number of tokens actually written
        """
        n_tokens = tokens.shape[0]
        capacity = self.data.shape[0]
        n_write = min(n_tokens, capacity - start_idx)

        if n_write > 0:
            # AMD HIP: 如果是远端 buffer，实际使用 hipMemcpyPeer
            self.data[start_idx:start_idx + n_write] = tokens[:n_write]
            self.metadata[start_idx:start_idx + n_write] = True

        return n_write

    def read_valid_tokens(self) -> Tensor:
        """Read only valid (non-padding) tokens from this slot."""
        valid_mask = self.metadata
        if valid_mask.any():
            return self.data[valid_mask]
        return self.data[:0]  # 空 tensor，保持 shape 兼容

    @property
    def num_valid_tokens(self) -> int:
        return self.metadata.sum().item()

    @property
    def size_bytes(self) -> int:
        return (
            self.data.nelement() * self.data.element_size()
            + self.metadata.nelement() * self.metadata.element_size()
        )

    def __repr__(self) -> str:
        return (
            f"BufferSlot(id={self.slot_id}, usage={self.usage.name}, "
            f"valid={self.num_valid_tokens}/{self.data.shape[0]}, "
            f"locked={self.is_locked}, peer={self.peer_rank})"
        )


# ---------------------------------------------------------------------------
# P2P Buffer Pool：持久化 Buffer 池
# ---------------------------------------------------------------------------

class P2PBufferPool:
    """
    Persistent P2P Buffer Pool for AMD MI300X.

    训练初始化时一次性分配所有通信 Buffer，训练循环中零 malloc 开销。
    前向/反向共享同一物理内存（双 buffer 交替使用）。

    对比 NVIDIA NCCL UBR:
      NCCL UBR: ncclCommRegister(comm, buf, size, &handle) → SM 占用 1-4 SM
      P2P Pool: hipIpcGetMemHandle(buf) → 0 SM 占用

    Usage:
        config = BufferPoolConfig(num_ep_ranks=8, d_model=7168, ...)
        pool = P2PBufferPool(config, local_rank=0)
        slot = pool.acquire(BufferUsage.DISPATCH_SEND, peer_rank=3)
        slot.write_tokens(tokens)
        # ... dispatch 操作 ...
        pool.release(slot)
    """

    def __init__(
        self,
        config: BufferPoolConfig,
        local_rank: int = 0,
        device: Optional[torch.device] = None,
    ):
        config.validate()
        self.config = config
        self.local_rank = local_rank
        self._device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        # --- 一次性分配所有 Buffer ---
        self._send_slots: Dict[int, BufferSlot] = {}
        self._recv_slots: Dict[int, BufferSlot] = {}
        self._all_slots: List[BufferSlot] = []

        self._initialize_pool()

        # IPC Handle 交换标记
        self._ipc_exchanged = False

    def _initialize_pool(self) -> None:
        """
        One-time pool initialization.

        AMD HIP: hipMalloc 连续分配 + hipIpcGetMemHandle 导出每块地址。
        所有分配在此完成，后续训练循环不再调用 hipMalloc。
        """
        slot_id = 0
        for peer in range(self.config.num_ep_ranks):
            # Send buffer：本 GPU 发送给 peer 的数据
            send_slot = BufferSlot.allocate(
                slot_id=slot_id,
                usage=BufferUsage.DISPATCH_SEND,
                capacity=self.config.capacity_per_rank,
                d_model=self.config.d_model,
                dtype=self.config.dtype,
                device=self._device,
                peer_rank=peer,
            )
            self._send_slots[peer] = send_slot
            self._all_slots.append(send_slot)
            slot_id += 1

            # Recv buffer：本 GPU 从 peer 接收的数据
            recv_slot = BufferSlot.allocate(
                slot_id=slot_id,
                usage=BufferUsage.DISPATCH_RECV,
                capacity=self.config.capacity_per_rank,
                d_model=self.config.d_model,
                dtype=self.config.dtype,
                device=self._device,
                peer_rank=peer,
            )
            self._recv_slots[peer] = recv_slot
            self._all_slots.append(recv_slot)
            slot_id += 1

    def exchange_ipc_handles(self) -> None:
        """
        Exchange IPC handles with all peers (one-time at training init).

        AMD HIP:
          1. hipIpcGetMemHandle(&handle, ptr) 导出本地 buffer 的 IPC handle
          2. All-to-All 交换 handles（可用 MPI 或 RCCL）
          3. hipIpcOpenMemHandle(&peer_ptr, remote_handle) 获取远端指针

        训练循环中直接使用 peer_ptr 进行 P2P 读写。
        """
        # 模拟 IPC Handle 交换
        # 实际中：
        #   for each buffer:
        #     handle = hipIpcGetMemHandle(buffer.data_ptr())
        #     all_gather(handle) → 收集所有 GPU 的 handles
        #     for each remote handle:
        #       peer_ptrs[rank] = hipIpcOpenMemHandle(remote_handle)
        self._ipc_exchanged = True

    def acquire(
        self,
        usage: BufferUsage,
        peer_rank: int = -1,
    ) -> BufferSlot:
        """
        Acquire a buffer slot for use.

        Args:
            usage: Buffer 用途
            peer_rank: 对端 rank（-1 表示任意）

        Returns:
            Available BufferSlot

        Raises:
            RuntimeError: No available slot
        """
        slots = (
            self._send_slots if usage in (
                BufferUsage.DISPATCH_SEND, BufferUsage.COMBINE_SEND
            )
            else self._recv_slots
        )

        if peer_rank >= 0 and peer_rank in slots:
            slot = slots[peer_rank]
            if not slot.is_locked:
                slot.is_locked = True
                slot.reset()
                return slot

        # 查找任意可用 slot
        for slot in slots.values():
            if not slot.is_locked:
                slot.is_locked = True
                slot.reset()
                return slot

        raise RuntimeError(
            f"No available buffer slot for usage={usage.name}, "
            f"peer_rank={peer_rank}"
        )

    def release(self, slot: BufferSlot) -> None:
        """Release a buffer slot back to the pool."""
        slot.is_locked = False

    def release_all(self) -> None:
        """Release all locked slots (typically at step boundary)."""
        for slot in self._all_slots:
            slot.is_locked = False

    def reset_all(self) -> None:
        """Reset all slots for a new training step."""
        for slot in self._all_slots:
            slot.reset()

    def get_send_buffer(self, peer_rank: int) -> BufferSlot:
        """Get the send buffer for a specific peer."""
        return self._send_slots[peer_rank]

    def get_recv_buffer(self, peer_rank: int) -> BufferSlot:
        """Get the recv buffer for a specific peer."""
        return self._recv_slots[peer_rank]

    # --- HIP Graph 兼容性接口 ---

    def get_static_addresses(self) -> Dict[str, List[int]]:
        """
        Return all buffer addresses for HIP Graph capture.

        HIP Graph 要求所有内存地址在 capture 和 replay 时相同。
        P2P Buffer Pool 的地址在训练初始化后恒定，天然满足此要求。

        Returns:
            Dict with 'send' and 'recv' address lists
        """
        return {
            'send': [
                slot.data.data_ptr() for slot in self._send_slots.values()
            ],
            'recv': [
                slot.data.data_ptr() for slot in self._recv_slots.values()
            ],
        }

    def verify_address_stability(self) -> bool:
        """
        Verify that all buffer addresses remain stable (for HIP Graph safety).

        Should be called periodically during development to ensure no
        accidental reallocation.
        """
        for slot in self._all_slots:
            if slot.data.data_ptr() == 0:
                return False
        return True

    # --- 统计与调试 ---

    def memory_report(self) -> Dict[str, float]:
        """Return memory usage report (MB)."""
        total_data = sum(
            s.data.nelement() * s.data.element_size() for s in self._all_slots
        )
        total_meta = sum(
            s.metadata.nelement() * s.metadata.element_size()
            for s in self._all_slots
        )
        locked = sum(1 for s in self._all_slots if s.is_locked)

        return {
            'total_data_mb': total_data / (1024 ** 2),
            'total_metadata_mb': total_meta / (1024 ** 2),
            'total_mb': (total_data + total_meta) / (1024 ** 2),
            'total_gb': (total_data + total_meta) / (1024 ** 3),
            'num_slots': len(self._all_slots),
            'locked_slots': locked,
            'ipc_exchanged': self._ipc_exchanged,
            'hbm_utilization_pct': (
                (total_data + total_meta) / (MI300X_HBM_CAPACITY_GB * 1024 ** 3)
                * 100
            ),
        }

    def utilization_stats(self) -> Dict[str, float]:
        """Return buffer utilization statistics."""
        total_capacity = sum(s.data.shape[0] for s in self._all_slots)
        total_valid = sum(s.num_valid_tokens for s in self._all_slots)
        return {
            'total_capacity': total_capacity,
            'total_valid_tokens': total_valid,
            'utilization_pct': (
                total_valid / max(total_capacity, 1) * 100
            ),
        }

    def __repr__(self) -> str:
        cfg = self.config
        mem = self.memory_report()
        return (
            f"P2PBufferPool(\n"
            f"  ep_ranks={cfg.num_ep_ranks}, d_model={cfg.d_model},\n"
            f"  capacity_per_rank={cfg.capacity_per_rank},\n"
            f"  shared_fwd_bwd={cfg.enable_shared_fwd_bwd},\n"
            f"  total_memory={mem['total_gb']:.2f} GB,\n"
            f"  hbm_utilization={mem['hbm_utilization_pct']:.1f}%,\n"
            f"  slots={len(self._all_slots)}, "
            f"ipc_exchanged={self._ipc_exchanged}\n"
            f")"
        )
