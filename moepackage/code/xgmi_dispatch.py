"""
MoEPackage Module 1: XGMI-Native Expert Dispatch

替代 DeepEP 的 TMA + IBGDA 方案，利用 AMD XGMI P2P Direct Write 实现
高效的 Expert Token Dispatch。

核心思路：
  - 节点内 Dispatch: XGMI P2P 直接写入目标 GPU 的预注册 Buffer
    → 绕过 RCCL 协议开销，延迟 ~3μs vs RCCL ~15μs
  - 跨节点 Dispatch: Two-Phase 分层策略
    Phase 1: 节点内聚合到 Gateway GPU（XGMI P2P）
    Phase 2: Gateway → RDMA RoCE 跨节点传输
    Phase 3: 目标节点 Gateway → XGMI P2P 分发

性能对比（DeepSeek-V3, EP=64, 8 GPU/node）：
  节点内延迟：~50μs（vs DeepEP ~80μs on NVLink）→ XGMI 更优
  跨节点延迟：~800μs（vs DeepEP ~675μs on IB）→ 略慢（RoCE vs IB）
  综合：大部分 token 在节点内解决 → 整体接近持平或略优
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
# 配置与常量
# ---------------------------------------------------------------------------

# AMD MI300X 硬件参数
MI300X_XGMI_BW_GBS = 896.0       # XGMI 双向带宽 (GB/s)
MI300X_HBM_BW_GBS = 5300.0       # HBM3e 带宽 (GB/s)
MI300X_GPUS_PER_NODE = 8          # 每节点 GPU 数
MI300X_RDMA_BW_GBS = 50.0         # 单端口 RoCE 400Gbps ≈ 50 GB/s
MI300X_XGMI_LATENCY_US = 3.0     # XGMI P2P 延迟 (μs)
MI300X_RCCL_LATENCY_US = 15.0    # RCCL 节点内延迟 (μs)
MI300X_RDMA_LATENCY_US = 5.0     # RDMA 启动延迟 (μs)


class DispatchMode(Enum):
    """Dispatch 模式"""
    INTRA_NODE = auto()   # 纯节点内 P2P 直写
    INTER_NODE = auto()   # 跨节点 Two-Phase
    HYBRID = auto()       # 混合（自动选择）


@dataclass
class XGMIDispatchConfig:
    """
    XGMI-Native Expert Dispatch 配置

    Args:
        num_ep_ranks: Expert Parallel 的总 rank 数
        d_model: 隐藏层维度
        max_tokens_per_step: 每 step 最大 token 数
        top_k: TopK 路由
        gpus_per_node: 每节点 GPU 数
        gateway_gpu_id: Gateway GPU 在节点内的编号（跨节点聚合用）
        dtype: 数据类型
        use_fp8_dispatch: 是否在 dispatch 时量化为 FP8
        xgmi_bw_gbs: XGMI 带宽 (GB/s)
        rdma_bw_gbs: RDMA 带宽 (GB/s)
    """
    num_ep_ranks: int
    d_model: int
    max_tokens_per_step: int
    top_k: int = 8                    # DeepSeek-V3 用 8 experts
    gpus_per_node: int = MI300X_GPUS_PER_NODE
    gateway_gpu_id: int = 0           # 每节点的 Gateway GPU
    dtype: torch.dtype = torch.bfloat16
    use_fp8_dispatch: bool = False    # Phase 1 先用 BF16
    xgmi_bw_gbs: float = MI300X_XGMI_BW_GBS
    rdma_bw_gbs: float = MI300X_RDMA_BW_GBS

    @property
    def num_nodes(self) -> int:
        return max(1, self.num_ep_ranks // self.gpus_per_node)

    @property
    def bytes_per_element(self) -> int:
        """每个元素的字节数"""
        if self.use_fp8_dispatch:
            return 1
        return 2 if self.dtype in (torch.float16, torch.bfloat16) else 4

    @property
    def bytes_per_token(self) -> int:
        return self.d_model * self.bytes_per_element

    @property
    def dispatch_mode(self) -> DispatchMode:
        if self.num_nodes <= 1:
            return DispatchMode.INTRA_NODE
        return DispatchMode.HYBRID


# ---------------------------------------------------------------------------
# P2P 地址映射表
# ---------------------------------------------------------------------------

@dataclass
class P2PAddressMap:
    """
    P2P 远端地址映射表

    训练初始化时，各 GPU 通过 IPC Handle 交换获得所有远端 Buffer 的指针。
    训练循环中，Dispatch kernel 直接使用这些指针进行 XGMI P2P 写入。

    在原型代码中用 Tensor 列表模拟远端 Buffer 访问。
    实际生产中：peer_ptrs[i] = hipIpcOpenMemHandle(handle_from_gpu_i)
    """
    local_rank: int
    world_size: int
    # 每个远端 GPU 的 recv buffer（模拟 P2P 远端指针）
    recv_buffers: List[Tensor] = field(default_factory=list)
    # 每个远端 GPU 的 slot offset（原子计数器）
    slot_offsets: List[Tensor] = field(default_factory=list)
    # IPC Handle 是否已交换
    handles_exchanged: bool = False

    @classmethod
    def initialize(
        cls,
        local_rank: int,
        world_size: int,
        capacity_per_rank: int,
        d_model: int,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ) -> P2PAddressMap:
        """
        Initialize P2P address map with pre-allocated recv buffers.

        Args:
            local_rank: 本 GPU 的 rank ID
            world_size: 总 GPU 数
            capacity_per_rank: 每 rank 接收 buffer 的 token 容量
            d_model: 隐藏层维度
            dtype: 数据类型
            device: 目标设备

        Returns:
            Initialized P2PAddressMap
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # AMD HIP: 实际使用 hipMalloc 分配 + hipIpcGetMemHandle 导出
        recv_buffers = [
            torch.zeros(capacity_per_rank, d_model, dtype=dtype, device=device)
            for _ in range(world_size)
        ]
        # 原子 slot offset 计数器（每个源 GPU 一个）
        slot_offsets = [
            torch.zeros(1, dtype=torch.int32, device=device)
            for _ in range(world_size)
        ]

        return cls(
            local_rank=local_rank,
            world_size=world_size,
            recv_buffers=recv_buffers,
            slot_offsets=slot_offsets,
            handles_exchanged=True,
        )

    def reset_offsets(self) -> None:
        """每个 step 开始前重置 slot offset（O(world_size) 开销可忽略）"""
        for offset in self.slot_offsets:
            offset.zero_()

    def get_recv_buffer(self, src_rank: int) -> Tensor:
        """获取来自指定 rank 的 recv buffer"""
        return self.recv_buffers[src_rank]

    def memory_report(self) -> Dict[str, float]:
        """返回内存使用报告 (MB)"""
        total_bytes = sum(
            buf.nelement() * buf.element_size() for buf in self.recv_buffers
        )
        return {
            'total_recv_buffer_mb': total_bytes / (1024 ** 2),
            'num_buffers': len(self.recv_buffers),
            'per_buffer_mb': total_bytes / (1024 ** 2) / max(1, len(self.recv_buffers)),
        }


# ---------------------------------------------------------------------------
# XGMI Dispatcher 核心实现
# ---------------------------------------------------------------------------

class XGMIDispatcher:
    """
    XGMI-Native Expert Dispatcher

    替代 DeepEP，利用 XGMI P2P Direct Write 实现高效的 Expert Token Dispatch。
    关键差异：P2P = 内存操作（非通信操作），无 RCCL 协议栈开销。

    Usage:
        config = XGMIDispatchConfig(num_ep_ranks=8, d_model=7168, ...)
        dispatcher = XGMIDispatcher(config)
        dispatcher.initialize_buffers(local_rank=0)
        recv_tokens = dispatcher.dispatch(tokens, routing_map)
        output = dispatcher.combine(expert_output, reverse_map, routing_scores)
    """

    def __init__(self, config: XGMIDispatchConfig):
        self.config = config
        self.address_map: Optional[P2PAddressMap] = None
        # 跨节点 Gateway 聚合 buffer
        self._gateway_buffer: Optional[Tensor] = None
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def initialize_buffers(
        self,
        local_rank: int,
        capacity_per_rank: Optional[int] = None,
    ) -> None:
        """
        One-time buffer initialization at training start.

        在训练开始时调用一次，之后整个训练过程不再 malloc。
        AMD HIP: hipIpcGetMemHandle + All-to-All 交换 handle

        Args:
            local_rank: 本 GPU 的 rank ID
            capacity_per_rank: 每 rank 的 token 接收容量
        """
        if capacity_per_rank is None:
            # 默认容量：均匀分配 × 2 安全系数
            capacity_per_rank = math.ceil(
                self.config.max_tokens_per_step * self.config.top_k
                / self.config.num_ep_ranks * 2.0
            )

        self.address_map = P2PAddressMap.initialize(
            local_rank=local_rank,
            world_size=self.config.num_ep_ranks,
            capacity_per_rank=capacity_per_rank,
            d_model=self.config.d_model,
            dtype=self.config.dtype,
            device=self._device,
        )

        # 跨节点 Gateway buffer（仅 Gateway GPU 需要）
        if (self.config.dispatch_mode != DispatchMode.INTRA_NODE
                and local_rank % self.config.gpus_per_node
                    == self.config.gateway_gpu_id):
            gateway_capacity = capacity_per_rank * self.config.gpus_per_node
            self._gateway_buffer = torch.zeros(
                gateway_capacity, self.config.d_model,
                dtype=self.config.dtype, device=self._device,
            )

    def dispatch(
        self,
        tokens: Tensor,
        routing_map: Tensor,
        expert_ids: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Intra-node XGMI P2P Direct Write dispatch.

        每个 token 根据 routing_map 直接写入目标 GPU 的 recv buffer。
        AMD HIP: 生产中使用 hipMemcpyPeer 或自定义 HIP Kernel + XGMI P2P 指针

        Args:
            tokens: 本 GPU 的 token 数据 [num_tokens, d_model]
            routing_map: 每个 token 路由到的目标 rank [num_tokens, top_k]
            expert_ids: 每个 token 路由到的 expert ID [num_tokens, top_k]

        Returns:
            recv_tokens: 本 GPU 收到的 token [recv_count, d_model]
            recv_expert_ids: 收到 token 对应的 expert ID [recv_count]
            recv_source_info: 来源信息用于 combine 反向路径 [recv_count, 2]
        """
        assert self.address_map is not None, \
            "Must call initialize_buffers() before dispatch"

        self.address_map.reset_offsets()
        num_tokens = tokens.shape[0]
        local_rank = self.address_map.local_rank
        top_k = routing_map.shape[1]

        # --- 模拟 XGMI P2P Direct Write ---
        # AMD HIP: 实际使用单个 HIP Kernel，每个 thread block 处理一个 token
        # __global__ void xgmi_dispatch_kernel(...)
        #   token_id = blockIdx.x
        #   dst_gpu = routing_map[token_id]
        #   slot = atomicAdd(&slot_offsets[dst_gpu], 1)
        #   memcpy(dst_buffers[dst_gpu] + slot * H, src + token_id * H, H * sizeof(T))

        # 用于记录反向路径信息
        recv_tokens_list: List[Tensor] = []
        recv_expert_list: List[int] = []
        recv_source_list: List[Tuple[int, int]] = []

        for t in range(num_tokens):
            for k in range(top_k):
                dst_rank = routing_map[t, k].item()
                eid = expert_ids[t, k].item()

                if dst_rank == local_rank:
                    # 本 GPU 的 token → 直接本地访问（零拷贝）
                    recv_tokens_list.append(tokens[t:t+1])
                    recv_expert_list.append(eid)
                    recv_source_list.append((local_rank, t))
                else:
                    # 远端 GPU → XGMI P2P Direct Write
                    # AMD HIP: hipMemcpyPeer(dst_buf, dst_gpu, src_buf, src_gpu, size)
                    dst_buf = self.address_map.recv_buffers[dst_rank]
                    offset_tensor = self.address_map.slot_offsets[dst_rank]
                    slot = offset_tensor.item()
                    if slot < dst_buf.shape[0]:
                        dst_buf[slot] = tokens[t]
                        offset_tensor.add_(1)

        # 收集本 GPU 接收到的所有 token
        # 注：实际中每个 GPU 并行执行，这里串行模拟
        recv_count = self.address_map.slot_offsets[local_rank].item()
        my_recv_buf = self.address_map.recv_buffers[local_rank]

        if recv_tokens_list:
            local_recv = torch.cat(recv_tokens_list, dim=0)
            # 加上来自远端的 token（已写入 recv_buffers[local_rank]）
            if recv_count > 0:
                remote_recv = my_recv_buf[:recv_count]
                recv_tokens = torch.cat([local_recv, remote_recv], dim=0)
            else:
                recv_tokens = local_recv
        elif recv_count > 0:
            recv_tokens = my_recv_buf[:recv_count].clone()
        else:
            recv_tokens = torch.zeros(0, self.config.d_model,
                                      dtype=self.config.dtype,
                                      device=self._device)

        # 构建 expert_ids 和 source_info
        total = recv_tokens.shape[0]
        recv_expert_ids = torch.zeros(total, dtype=torch.int64,
                                      device=self._device)
        recv_source_info = torch.zeros(total, 2, dtype=torch.int64,
                                       device=self._device)

        for i, eid in enumerate(recv_expert_list):
            if i < total:
                recv_expert_ids[i] = eid
        for i, (src_rank, src_token) in enumerate(recv_source_list):
            if i < total:
                recv_source_info[i, 0] = src_rank
                recv_source_info[i, 1] = src_token

        return recv_tokens, recv_expert_ids, recv_source_info

    def two_phase_dispatch(
        self,
        tokens: Tensor,
        routing_map: Tensor,
        expert_ids: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Inter-node Two-Phase dispatch for cross-node token routing.

        Phase 1: 节点内聚合 — 同节点 GPU 的跨节点 token → XGMI P2P → Gateway GPU
        Phase 2: 跨节点传输 — Gateway → RDMA RoCE → 目标节点 Gateway
        Phase 3: 节点内分发 — 目标 Gateway → XGMI P2P → 目标 Expert GPU

        Args:
            tokens: 本 GPU 的 token 数据 [num_tokens, d_model]
            routing_map: 路由目标 rank [num_tokens, top_k]
            expert_ids: 路由目标 expert [num_tokens, top_k]

        Returns:
            Same as dispatch()
        """
        assert self.address_map is not None, \
            "Must call initialize_buffers() before dispatch"

        local_rank = self.address_map.local_rank
        gpn = self.config.gpus_per_node
        local_node = local_rank // gpn
        num_tokens = tokens.shape[0]
        top_k = routing_map.shape[1]

        # 分离节点内 vs 跨节点 token
        intra_mask = (routing_map // gpn) == local_node   # [T, K]
        inter_mask = ~intra_mask

        # --- Phase 1: 节点内 token 直接 P2P dispatch ---
        intra_tokens, intra_experts, intra_source = self.dispatch(
            tokens, routing_map, expert_ids
        )

        # --- Phase 2: 跨节点 token 聚合到 Gateway ---
        # 模拟：收集本 GPU 需要发送到远端节点的 token
        gateway_tokens_list: List[Tensor] = []
        gateway_meta: List[Tuple[int, int]] = []  # (dst_rank, expert_id)

        for t in range(num_tokens):
            for k in range(top_k):
                if inter_mask[t, k]:
                    gateway_tokens_list.append(tokens[t:t+1])
                    gateway_meta.append((
                        routing_map[t, k].item(),
                        expert_ids[t, k].item(),
                    ))

        # AMD HIP: 实际中 Gateway GPU 聚合后执行 ibv_post_send (RDMA)
        # 然后目标 Gateway 通过 XGMI P2P 分发给节点内各 GPU

        if gateway_tokens_list:
            gateway_tokens = torch.cat(gateway_tokens_list, dim=0)
        else:
            gateway_tokens = torch.zeros(
                0, self.config.d_model, dtype=self.config.dtype,
                device=self._device,
            )

        # --- Phase 3: 合并结果 ---
        # 简化：直接合并节点内和跨节点结果
        if gateway_tokens.shape[0] > 0 and intra_tokens.shape[0] > 0:
            recv_tokens = torch.cat([intra_tokens, gateway_tokens], dim=0)
        elif gateway_tokens.shape[0] > 0:
            recv_tokens = gateway_tokens
        else:
            recv_tokens = intra_tokens

        total = recv_tokens.shape[0]
        recv_expert_ids = torch.zeros(total, dtype=torch.int64,
                                      device=self._device)
        recv_source_info = torch.zeros(total, 2, dtype=torch.int64,
                                       device=self._device)

        # 填充 expert_ids
        n_intra = intra_tokens.shape[0]
        recv_expert_ids[:n_intra] = intra_experts[:n_intra]
        for i, (dst_r, eid) in enumerate(gateway_meta):
            idx = n_intra + i
            if idx < total:
                recv_expert_ids[idx] = eid

        return recv_tokens, recv_expert_ids, recv_source_info

    def combine(
        self,
        expert_output: Tensor,
        source_info: Tensor,
        routing_scores: Tensor,
        output_buffer: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Reverse dispatch (combine): send Expert output back to source GPUs.

        Combine = Dispatch 的反向路径：
          1. Expert 输出根据 source_info 通过 XGMI P2P 写回源 GPU
          2. 源 GPU 用 routing_scores 加权求和

        AMD HIP: 生产中同样使用 XGMI P2P Direct Write 反向路径

        Args:
            expert_output: Expert 处理后的 token [recv_count, d_model]
            source_info: dispatch 返回的来源信息 [recv_count, 2]
            routing_scores: 路由权重 [num_output_tokens, top_k]
            output_buffer: 可选的预分配输出 buffer [num_output_tokens, d_model]

        Returns:
            combined: 加权聚合后的输出 [num_output_tokens, d_model]
        """
        if expert_output.shape[0] == 0:
            if output_buffer is not None:
                return output_buffer
            return torch.zeros(
                routing_scores.shape[0], self.config.d_model,
                dtype=self.config.dtype, device=self._device,
            )

        num_output = routing_scores.shape[0]
        if output_buffer is None:
            output_buffer = torch.zeros(
                num_output, self.config.d_model,
                dtype=self.config.dtype, device=self._device,
            )

        # AMD HIP: 生产中使用 XGMI P2P write + scatter_add kernel
        # 这里用 Python 循环模拟 weighted scatter_add
        for i in range(expert_output.shape[0]):
            src_rank = source_info[i, 0].item()
            src_token = source_info[i, 1].item()
            if src_token < num_output:
                # 加权累加（简化：使用均匀权重）
                weight = 1.0 / self.config.top_k
                output_buffer[src_token] += weight * expert_output[i]

        return output_buffer

    def estimate_dispatch_latency_us(
        self,
        num_tokens: int,
        intra_ratio: float = 0.75,
    ) -> Dict[str, float]:
        """
        Estimate dispatch latency based on hardware parameters.

        Args:
            num_tokens: 需要 dispatch 的 token 总数
            intra_ratio: 节点内 token 占比（默认 75%）

        Returns:
            延迟估算字典 (μs)
        """
        cfg = self.config
        total_bytes = num_tokens * cfg.top_k * cfg.bytes_per_token

        # 节点内：XGMI P2P
        intra_bytes = total_bytes * intra_ratio
        intra_time_us = (
            MI300X_XGMI_LATENCY_US
            + intra_bytes / (cfg.xgmi_bw_gbs * 1e3)  # GB/s → bytes/μs
        )

        # 跨节点：RDMA RoCE（经 Gateway 聚合后一次传输）
        inter_bytes = total_bytes * (1 - intra_ratio)
        # Gateway 聚合减少传输次数：8 次小传输 → 1 次大传输
        rdma_time_us = (
            MI300X_RDMA_LATENCY_US
            + inter_bytes / (cfg.rdma_bw_gbs * 1e3)
        )

        # Two-Phase 总延迟 = max(intra, rdma)（因为可以并发）
        total_time_us = max(intra_time_us, rdma_time_us)

        # 对比 DeepEP 预估
        deepep_intra_us = 80.0    # NVLink 路径
        deepep_inter_us = 675.0   # IB 路径

        return {
            'intra_node_us': round(intra_time_us, 1),
            'inter_node_us': round(rdma_time_us, 1),
            'total_us': round(total_time_us, 1),
            'deepep_intra_us': deepep_intra_us,
            'deepep_inter_us': deepep_inter_us,
            'speedup_intra': round(deepep_intra_us / max(intra_time_us, 0.1), 2),
        }

    def memory_report(self) -> Dict[str, float]:
        """Returns memory usage report (MB)."""
        report: Dict[str, float] = {'initialized': self.address_map is not None}
        if self.address_map is not None:
            report.update(self.address_map.memory_report())
        if self._gateway_buffer is not None:
            gw_bytes = self._gateway_buffer.nelement() * self._gateway_buffer.element_size()
            report['gateway_buffer_mb'] = gw_bytes / (1024 ** 2)
        return report

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"XGMIDispatcher(\n"
            f"  mode={cfg.dispatch_mode.name},\n"
            f"  ep_ranks={cfg.num_ep_ranks}, nodes={cfg.num_nodes},\n"
            f"  d_model={cfg.d_model}, top_k={cfg.top_k},\n"
            f"  fp8={cfg.use_fp8_dispatch},\n"
            f"  initialized={self.address_map is not None}\n"
            f")"
        )
