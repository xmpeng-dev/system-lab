"""Unit tests for axion.core.overlap_scheduler.

Validates the OverlapScheduler's chunked pipeline against the serial baseline
(MoEDispatcher.full_forward).  Tests run on CPU without distributed backend.
"""

from __future__ import annotations

import torch
import pytest

from axion.core.comm_tensor import RoutingTable
from axion.core.moe_dispatcher import MoEDispatchConfig, MoEDispatcher
from axion.core.overlap_scheduler import OverlapConfig, OverlapScheduler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dispatch_config():
    return MoEDispatchConfig(
        num_experts=4,
        topk=2,
        hidden_dim=8,
        world_size=1,
    )


@pytest.fixture
def routing_table_8tok():
    """8 tokens, topk=2, 4 experts (8 is divisible by 1/2/4)."""
    torch.manual_seed(99)
    expert_indices = torch.randint(0, 4, (8, 2))
    routing_weights = torch.ones(8, 2) / 2
    return RoutingTable.build(
        expert_indices=expert_indices,
        routing_weights=routing_weights,
        num_experts=4,
        world_size=1,
    )


@pytest.fixture
def hidden_8tok():
    torch.manual_seed(0)
    return torch.randn(8, 8)


def serial_result(dispatch_config, hidden, routing_table, expert_fn):
    """Reference: serial MoEDispatcher."""
    dispatcher = MoEDispatcher(dispatch_config)
    return dispatcher.full_forward(hidden, routing_table, expert_fn)


# ---------------------------------------------------------------------------
# num_chunks=1 fast-path
# ---------------------------------------------------------------------------

class TestNoOverlap:
    def test_num_chunks_1_matches_serial(
        self, dispatch_config, hidden_8tok, routing_table_8tok
    ):
        expert_fn = lambda x: x  # identity
        expected = serial_result(
            dispatch_config, hidden_8tok, routing_table_8tok, expert_fn
        )
        sched = OverlapScheduler(
            OverlapConfig(num_chunks=1, dispatch_config=dispatch_config)
        )
        out = sched.forward(hidden_8tok, routing_table_8tok, expert_fn)
        assert torch.allclose(out, expected, atol=1e-5), (
            f"num_chunks=1 mismatch, max diff={( out - expected).abs().max()}"
        )


# ---------------------------------------------------------------------------
# num_chunks=2 (CPU, no CUDA streams)
# ---------------------------------------------------------------------------

class TestOverlapSchedulerCPU:
    """On CPU, comm_stream is None so the scheduler falls back to serial path."""

    def test_output_shape(self, dispatch_config, hidden_8tok, routing_table_8tok):
        sched = OverlapScheduler(
            OverlapConfig(num_chunks=2, dispatch_config=dispatch_config)
        )
        out = sched.forward(hidden_8tok, routing_table_8tok, lambda x: x)
        assert out.shape == hidden_8tok.shape

    def test_identity_expert_matches_serial(
        self, dispatch_config, hidden_8tok, routing_table_8tok
    ):
        expert_fn = lambda x: x
        expected = serial_result(
            dispatch_config, hidden_8tok, routing_table_8tok, expert_fn
        )
        sched = OverlapScheduler(
            OverlapConfig(num_chunks=2, dispatch_config=dispatch_config)
        )
        out = sched.forward(hidden_8tok, routing_table_8tok, expert_fn)
        assert torch.allclose(out, expected, atol=1e-5), (
            f"num_chunks=2 mismatch, max diff={( out - expected).abs().max()}"
        )

    def test_scale_expert_matches_serial(
        self, dispatch_config, hidden_8tok, routing_table_8tok
    ):
        expert_fn = lambda x: x * 3.0
        expected = serial_result(
            dispatch_config, hidden_8tok, routing_table_8tok, expert_fn
        )
        sched = OverlapScheduler(
            OverlapConfig(num_chunks=2, dispatch_config=dispatch_config)
        )
        out = sched.forward(hidden_8tok, routing_table_8tok, expert_fn)
        assert torch.allclose(out, expected, atol=1e-4), (
            f"scale-expert mismatch, max diff={( out - expected).abs().max()}"
        )

    def test_num_chunks_4(
        self, dispatch_config, hidden_8tok, routing_table_8tok
    ):
        expert_fn = lambda x: x
        expected = serial_result(
            dispatch_config, hidden_8tok, routing_table_8tok, expert_fn
        )
        sched = OverlapScheduler(
            OverlapConfig(num_chunks=4, dispatch_config=dispatch_config)
        )
        out = sched.forward(hidden_8tok, routing_table_8tok, expert_fn)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_uneven_chunk_raises(
        self, dispatch_config, routing_table_8tok
    ):
        """num_tokens not divisible by num_chunks should raise ValueError."""
        hidden_7 = torch.randn(7, 8)
        # Need a 7-token routing table.
        expert_indices = torch.randint(0, 4, (7, 2))
        routing_weights = torch.ones(7, 2) / 2
        rt_7 = RoutingTable.build(
            expert_indices=expert_indices,
            routing_weights=routing_weights,
            num_experts=4,
            world_size=1,
        )
        sched = OverlapScheduler(
            OverlapConfig(num_chunks=4, dispatch_config=dispatch_config)
        )
        with pytest.raises(ValueError, match="divisible"):
            sched.forward(hidden_7, rt_7, lambda x: x)
