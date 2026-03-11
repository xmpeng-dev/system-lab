"""Axion core: Communication-first MoE training primitives.

This package implements the lowest-level tensor abstractions for
communication-native MoE (Mixture of Experts) training.  Everything is built
around :class:`~axion.core.comm_tensor.CommTensor`, a tensor whose *physical
memory layout* is part of its type.

Key classes
-----------
CommLayout
    Enum describing how tokens are arranged in memory.
CommSpec
    AllToAll communication parameters (send/recv counts, process group).
RoutingTable
    Per-step routing decisions (expert assignments, permutation indices).
CommTensor
    Layout-aware tensor wrapper; the central primitive of the system.
MoEDispatcher
    Orchestrates the full dispatch → expert_fn → combine cycle.
OverlapScheduler
    Chunks tokens and pipelines AllToAll with Expert FFN for overlap.
"""

from axion.core.comm_tensor import (
    CommLayout,
    CommSpec,
    CommTensor,
    RoutingTable,
)
from axion.core.moe_dispatcher import MoEDispatchConfig, MoEDispatcher
from axion.core.overlap_scheduler import OverlapConfig, OverlapScheduler

__all__ = [
    # CommTensor type system
    "CommLayout",
    "CommSpec",
    "CommTensor",
    "RoutingTable",
    # Dispatcher
    "MoEDispatchConfig",
    "MoEDispatcher",
    # Overlap scheduler
    "OverlapConfig",
    "OverlapScheduler",
]
