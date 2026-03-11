"""Unit tests for axion.core.moe_dispatcher.

Tests the MoEDispatcher dispatch / combine cycle using the single-process
CPU fast-path (no GPU or distributed backend required).
"""

from __future__ import annotations

import torch
import pytest

from axion.core.comm_tensor import CommLayout, CommSpec, CommTensor, RoutingTable
from axion.core.moe_dispatcher import MoEDispatchConfig, MoEDispatcher


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return MoEDispatchConfig(
        num_experts=4,
        topk=2,
        hidden_dim=16,
        world_size=1,
    )


@pytest.fixture
def routing_table():
    expert_indices = torch.tensor([[0, 1], [2, 3], [0, 2], [1, 3]])
    routing_weights = torch.ones(4, 2) / 2
    return RoutingTable.build(
        expert_indices=expert_indices,
        routing_weights=routing_weights,
        num_experts=4,
        world_size=1,
    )


@pytest.fixture
def hidden():
    torch.manual_seed(7)
    return torch.randn(4, 16)


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

class TestDispatch:
    def test_returns_comm_tensor(self, config, hidden, routing_table):
        dispatcher = MoEDispatcher(config)
        ct = dispatcher.dispatch(hidden, routing_table)
        assert isinstance(ct, CommTensor)

    def test_dispatch_layout_is_blocked_by_src(self, config, hidden, routing_table):
        dispatcher = MoEDispatcher(config)
        ct = dispatcher.dispatch(hidden, routing_table)
        assert ct.layout == CommLayout.BLOCKED_BY_SRC

    def test_dispatch_shape(self, config, hidden, routing_table):
        dispatcher = MoEDispatcher(config)
        ct = dispatcher.dispatch(hidden, routing_table)
        num_tokens, hidden_dim = hidden.shape
        topk = routing_table.expert_indices.shape[1]
        assert ct.shape == (num_tokens * topk, hidden_dim)


# ---------------------------------------------------------------------------
# combine
# ---------------------------------------------------------------------------

class TestCombine:
    def test_combine_shape(self, config, hidden, routing_table):
        dispatcher = MoEDispatcher(config)
        dispatched_ct = dispatcher.dispatch(hidden, routing_table)

        # Identity expert_fn.
        expert_ct = CommTensor(
            dispatched_ct.data.clone(),
            CommLayout.BLOCKED_BY_SRC,
            dispatched_ct.comm_spec,
        )
        out = dispatcher.combine(expert_ct, routing_table)
        assert out.shape == hidden.shape

    def test_combine_wrong_layout_raises(self, config, hidden, routing_table):
        dispatcher = MoEDispatcher(config)
        wrong_ct = CommTensor(hidden, CommLayout.BLOCKED_BY_DST)
        with pytest.raises(ValueError, match="BLOCKED_BY_SRC"):
            dispatcher.combine(wrong_ct, routing_table)


# ---------------------------------------------------------------------------
# full_forward
# ---------------------------------------------------------------------------

class TestFullForward:
    def test_output_shape(self, config, hidden, routing_table):
        dispatcher = MoEDispatcher(config)
        out = dispatcher.full_forward(
            hidden, routing_table, expert_fn=lambda x: x
        )
        assert out.shape == hidden.shape

    def test_identity_expert_recovers_input(self, config, hidden, routing_table):
        """With identity expert_fn and uniform weights, output == input."""
        dispatcher = MoEDispatcher(config)
        out = dispatcher.full_forward(
            hidden, routing_table, expert_fn=lambda x: x
        )
        assert torch.allclose(out, hidden, atol=1e-5), (
            f"Identity expert round-trip failed. Max diff: {(out - hidden).abs().max()}"
        )

    def test_scale_expert_changes_output(self, config, hidden, routing_table):
        """Expert_fn that doubles all values should produce 2× output."""
        dispatcher = MoEDispatcher(config)
        out = dispatcher.full_forward(
            hidden, routing_table, expert_fn=lambda x: x * 2.0
        )
        assert torch.allclose(out, hidden * 2.0, atol=1e-5)

    def test_dtype_preserved(self, config, routing_table):
        """Output dtype should match input dtype."""
        dispatcher = MoEDispatcher(config)
        hidden_bf16 = torch.randn(4, 16).to(torch.bfloat16)
        out = dispatcher.full_forward(
            hidden_bf16, routing_table, expert_fn=lambda x: x
        )
        assert out.dtype == torch.bfloat16

    def test_non_uniform_weights(self, config):
        """Non-uniform routing weights should produce weighted sum."""
        # Token 0 goes to expert 0 with weight 0.8 and expert 1 with weight 0.2.
        expert_indices = torch.tensor([[0, 1], [2, 3]])
        routing_weights = torch.tensor([[0.8, 0.2], [0.6, 0.4]])
        rt = RoutingTable.build(
            expert_indices=expert_indices,
            routing_weights=routing_weights,
            num_experts=4,
            world_size=1,
        )
        hidden = torch.ones(2, 4)
        dispatcher = MoEDispatcher(config)
        out = dispatcher.full_forward(hidden, rt, expert_fn=lambda x: x)
        # With identity expert_fn: output[i] = sum_k(w_ik * hidden[i]) = hidden[i].
        assert torch.allclose(out, hidden, atol=1e-5)
