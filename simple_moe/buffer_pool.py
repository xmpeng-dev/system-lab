"""
Buffer Pool for memory-efficient collective operations.

[veScale] Pre-allocate All-Gather and All-to-All buffers and reuse them
across layers via an LRU policy, avoiding repeated malloc/free.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Optional

import torch


class BufferPool:
    """
    A fixed-capacity pool of pre-allocated CUDA buffers.

    Buffers are acquired/released by callers.  When all buffers are in use the
    oldest is forcibly reclaimed (LRU eviction).
    """

    def __init__(
        self,
        max_elem_count: int,
        num_buffers: int = 2,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ) -> None:
        self._max = max_elem_count
        self._dtype = dtype
        self._device = device
        self._buffers = [
            torch.empty(max_elem_count, dtype=dtype, device=device)
            for _ in range(num_buffers)
        ]
        self._available: deque[int] = deque(range(num_buffers))
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def acquire(self, numel: int) -> torch.Tensor:
        """Return a buffer slice of at least *numel* elements."""
        with self._lock:
            if not self._available:
                self._available.append(0)
            idx = self._available.popleft()
        buf = self._buffers[idx]
        if numel > buf.numel():
            buf = torch.empty(numel, dtype=self._dtype, device=self._device)
            self._buffers[idx] = buf
        return buf[:numel]

    def release(self, buf: torch.Tensor) -> None:
        """Return a buffer to the pool (matched by data pointer)."""
        with self._lock:
            for i, b in enumerate(self._buffers):
                if buf.data_ptr() == b.data_ptr():
                    self._available.append(i)
                    return
            # If not found (e.g. resized), just ignore.

    # ------------------------------------------------------------------
    @property
    def num_total(self) -> int:
        return len(self._buffers)

    @property
    def num_available(self) -> int:
        return len(self._available)


class DualBufferPool:
    """
    Convenience wrapper that holds two pools:
    one for All-Gather buffers and one for All-to-All buffers.
    """

    def __init__(
        self,
        ag_max_elem: int,
        a2a_max_elem: int,
        num_buffers: int = 2,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.ag_pool = BufferPool(ag_max_elem, num_buffers, dtype)
        self.a2a_pool = BufferPool(a2a_max_elem, num_buffers, dtype)
