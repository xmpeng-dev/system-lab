"""Unit tests for axion.core.comm_tensor.

These tests run on CPU (no GPU or distributed backend required) to validate
the core CommTensor type system: layout semantics, from_dense / to_dense
round-trips, alltoall single-process fast-paths, and RoutingTable.build.
"""

from __future__ import annotations

import torch
import pytest

from axion.core.comm_tensor import (
    CommLayout,
    CommSpec,
    CommTensor,
    RoutingTable,
    _scale_comm_spec,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_routing_table():
    """4 tokens, topk=2, 4 experts, world_size=1.

    Expert assignments:
        token 0 → experts [0, 1]
        token 1 → experts [2, 3]
        token 2 → experts [0, 2]
        token 3 → experts [1, 3]

    With world_size=1 all tokens stay on the same GPU.
    All send_counts / recv_counts are [8] (4 tokens × topk=2).
    """
    num_tokens, topk, num_experts, world_size = 4, 2, 4, 1
    expert_indices = torch.tensor([[0, 1], [2, 3], [0, 2], [1, 3]])
    routing_weights = torch.ones(num_tokens, topk) / topk  # uniform weights

    rt = RoutingTable.build(
        expert_indices=expert_indices,
        routing_weights=routing_weights,
        num_experts=num_experts,
        world_size=world_size,
    )
    return rt


@pytest.fixture
def hidden(simple_routing_table):
    """Hidden states for the 4-token fixture: shape [4, 8]."""
    torch.manual_seed(42)
    return torch.randn(4, 8)


# ---------------------------------------------------------------------------
# CommLayout
# ---------------------------------------------------------------------------

class TestCommLayout:
    def test_all_variants_exist(self):
        assert CommLayout.INTERLEAVED is not None
        assert CommLayout.BLOCKED_BY_DST is not None
        assert CommLayout.BLOCKED_BY_SRC is not None
        assert CommLayout.BLOCKED_BY_EXPERT is not None
        assert CommLayout.SPARSE_CSR is not None

    def test_enum_values_are_strings(self):
        for layout in CommLayout:
            assert isinstance(layout.value, str)

    def test_distinct_values(self):
        values = [layout.value for layout in CommLayout]
        assert len(values) == len(set(values)), "CommLayout enum values must be unique"


# ---------------------------------------------------------------------------
# CommSpec
# ---------------------------------------------------------------------------

class TestCommSpec:
    def test_total_send(self):
        spec = CommSpec(send_counts=[3, 5], recv_counts=[4, 4])
        assert spec.total_send == 8

    def test_total_recv(self):
        spec = CommSpec(send_counts=[3, 5], recv_counts=[2, 6])
        assert spec.total_recv == 8

    def test_scale_comm_spec(self):
        spec = CommSpec(send_counts=[8, 8], recv_counts=[4, 12])
        half = _scale_comm_spec(spec, 0.5)
        assert half.send_counts == [4, 4]
        assert half.recv_counts == [2, 6]

    def test_scale_comm_spec_quarter(self):
        spec = CommSpec(send_counts=[8, 8], recv_counts=[8, 8])
        quarter = _scale_comm_spec(spec, 0.25)
        assert quarter.send_counts == [2, 2]
        assert quarter.recv_counts == [2, 2]


# ---------------------------------------------------------------------------
# RoutingTable.build
# ---------------------------------------------------------------------------

class TestRoutingTableBuild:
    def test_shape(self, simple_routing_table):
        rt = simple_routing_table
        num_tokens, topk = 4, 2
        assert rt.expert_indices.shape == (num_tokens, topk)
        assert rt.routing_weights.shape == (num_tokens, topk)
        assert rt.src_indices.shape == (num_tokens * topk,)
        assert rt.inverse_src_indices.shape == (num_tokens * topk,)
        assert rt.send_counts.shape[0] == 1   # world_size=1
        assert rt.recv_counts.shape[0] == 1

    def test_permutation_is_valid(self, simple_routing_table):
        rt = simple_routing_table
        n = rt.src_indices.numel()
        assert set(rt.src_indices.tolist()) == set(range(n))
        assert set(rt.inverse_src_indices.tolist()) == set(range(n))

    def test_inverse_undoes_permutation(self, simple_routing_table):
        rt = simple_routing_table
        x = torch.arange(rt.src_indices.numel()).float()
        sorted_x = x[rt.src_indices]
        recovered = torch.empty_like(x)
        recovered[rt.inverse_src_indices] = sorted_x
        assert torch.allclose(x, recovered)

    def test_send_counts_sum(self, simple_routing_table):
        rt = simple_routing_table
        # With world_size=1, all tokens go to rank 0.
        assert rt.send_counts.sum().item() == 4 * 2  # num_tokens * topk

    def test_custom_expert_to_rank(self):
        """Two experts per rank, world_size=2."""
        expert_indices = torch.tensor([[0, 1], [2, 3], [0, 3]])
        routing_weights = torch.ones(3, 2) / 2
        expert_to_rank = torch.tensor([0, 0, 1, 1])  # expert 0,1 → rank 0; 2,3 → rank 1
        rt = RoutingTable.build(
            expert_indices=expert_indices,
            routing_weights=routing_weights,
            num_experts=4,
            expert_to_rank=expert_to_rank,
            world_size=2,
        )
        # Tokens going to rank 0: (0,0), (0,1), (2,0) → experts 0,1,0 → all rank 0 → 3 slots
        # Tokens going to rank 1: (1,0), (1,1), (2,1) → experts 2,3,3 → all rank 1 → 3 slots
        assert rt.send_counts[0].item() == 3
        assert rt.send_counts[1].item() == 3


# ---------------------------------------------------------------------------
# CommTensor.from_dense
# ---------------------------------------------------------------------------

class TestFromDense:
    def test_layout_is_blocked_by_dst(self, hidden, simple_routing_table):
        ct = CommTensor.from_dense(hidden, simple_routing_table)
        assert ct.layout == CommLayout.BLOCKED_BY_DST

    def test_shape_preserved(self, hidden, simple_routing_table):
        ct = CommTensor.from_dense(hidden, simple_routing_table)
        num_tokens, hidden_dim = hidden.shape
        topk = simple_routing_table.expert_indices.shape[1]
        assert ct.shape == (num_tokens * topk, hidden_dim)

    def test_no_data_loss(self, hidden, simple_routing_table):
        """All token embeddings should be present in the BLOCKED_BY_DST buffer."""
        ct = CommTensor.from_dense(hidden, simple_routing_table)
        topk = simple_routing_table.expert_indices.shape[1]
        # Expand hidden to [S * topk, H].
        expanded = hidden.unsqueeze(1).expand(-1, topk, -1).reshape(-1, hidden.shape[-1])
        # After sorting by dst, the set of rows should be the same.
        sorted_rows = set(tuple(row.tolist()) for row in ct.data)
        expanded_rows = set(tuple(row.tolist()) for row in expanded)
        assert sorted_rows == expanded_rows

    def test_comm_spec_attached(self, hidden, simple_routing_table):
        spec = CommSpec(send_counts=[8], recv_counts=[8])
        ct = CommTensor.from_dense(hidden, simple_routing_table, spec)
        assert ct.comm_spec is spec


# ---------------------------------------------------------------------------
# CommTensor.to_dense
# ---------------------------------------------------------------------------

class TestToDense:
    def test_round_trip_shape(self, hidden, simple_routing_table):
        """from_dense → to_dense round-trip must restore original shape."""
        ct = CommTensor.from_dense(hidden, simple_routing_table)
        out = ct.to_dense(simple_routing_table)
        assert out.shape == hidden.shape

    def test_round_trip_values_uniform_weights(self, hidden, simple_routing_table):
        """With uniform weights (0.5 each), round-trip recovers original hidden."""
        ct = CommTensor.from_dense(hidden, simple_routing_table)
        out = ct.to_dense(simple_routing_table)
        # Each token's output = 0.5 * hidden[i] + 0.5 * hidden[i] = hidden[i].
        assert torch.allclose(out, hidden, atol=1e-5), (
            f"Round-trip mismatch: max diff = {(out - hidden).abs().max().item()}"
        )

    def test_wrong_layout_raises(self, simple_routing_table):
        dummy = torch.randn(4, 8)
        ct = CommTensor(dummy, CommLayout.INTERLEAVED)
        with pytest.raises(ValueError, match="BLOCKED_BY_DST"):
            ct.to_dense(simple_routing_table)


# ---------------------------------------------------------------------------
# CommTensor.alltoall (single-process fast-path)
# ---------------------------------------------------------------------------

class TestAlltoallSingleProcess:
    """Validates the single-process (world_size=1) identity fast-path."""

    def test_dispatch_layout_transition(self, hidden, simple_routing_table):
        spec = CommSpec(send_counts=[8], recv_counts=[8], world_size=1)
        ct = CommTensor.from_dense(hidden, simple_routing_table, spec)
        assert ct.layout == CommLayout.BLOCKED_BY_DST

        dispatched = ct.alltoall_dispatch()
        assert dispatched.layout == CommLayout.BLOCKED_BY_SRC

    def test_dispatch_data_preserved(self, hidden, simple_routing_table):
        spec = CommSpec(send_counts=[8], recv_counts=[8], world_size=1)
        ct = CommTensor.from_dense(hidden, simple_routing_table, spec)
        dispatched = ct.alltoall_dispatch()
        # Single-process: identity → data should be equal.
        assert torch.allclose(ct.data, dispatched.data)

    def test_combine_layout_transition(self, hidden, simple_routing_table):
        spec = CommSpec(send_counts=[8], recv_counts=[8], world_size=1)
        ct_send = CommTensor.from_dense(hidden, simple_routing_table, spec)
        ct_recv = ct_send.alltoall_dispatch()
        combined = ct_recv.alltoall_combine()
        assert combined.layout == CommLayout.BLOCKED_BY_DST

    def test_full_cycle_roundtrip(self, hidden, simple_routing_table):
        """dispatch → combine → to_dense should recover original hidden."""
        spec = CommSpec(send_counts=[8], recv_counts=[8], world_size=1)
        ct = CommTensor.from_dense(hidden, simple_routing_table, spec)
        dispatched = ct.alltoall_dispatch()
        combined = dispatched.alltoall_combine()
        out = combined.to_dense(simple_routing_table)
        assert torch.allclose(out, hidden, atol=1e-5), (
            f"Full-cycle mismatch: max diff = {(out - hidden).abs().max().item()}"
        )

    def test_dispatch_without_comm_spec_raises(self, hidden, simple_routing_table):
        ct = CommTensor.from_dense(hidden, simple_routing_table, comm_spec=None)
        with pytest.raises(RuntimeError, match="CommSpec"):
            ct.alltoall_dispatch()

    def test_combine_wrong_layout_raises(self, hidden, simple_routing_table):
        spec = CommSpec(send_counts=[8], recv_counts=[8], world_size=1)
        ct = CommTensor.from_dense(hidden, simple_routing_table, spec)
        # ct is BLOCKED_BY_DST; combine expects BLOCKED_BY_SRC.
        with pytest.raises(ValueError, match="BLOCKED_BY_SRC"):
            ct.alltoall_combine()


# ---------------------------------------------------------------------------
# CommTensor.chunk
# ---------------------------------------------------------------------------

class TestChunk:
    def test_chunk_count(self, hidden, simple_routing_table):
        spec = CommSpec(send_counts=[8], recv_counts=[8], world_size=1)
        ct = CommTensor.from_dense(hidden, simple_routing_table, spec)
        chunks = ct.chunk(2)
        assert len(chunks) == 2

    def test_chunk_layout_preserved(self, hidden, simple_routing_table):
        spec = CommSpec(send_counts=[8], recv_counts=[8], world_size=1)
        ct = CommTensor.from_dense(hidden, simple_routing_table, spec)
        for chunk in ct.chunk(2):
            assert chunk.layout == CommLayout.BLOCKED_BY_DST

    def test_chunk_shape(self, hidden, simple_routing_table):
        spec = CommSpec(send_counts=[8], recv_counts=[8], world_size=1)
        ct = CommTensor.from_dense(hidden, simple_routing_table, spec)
        chunks = ct.chunk(2)
        total_tokens = sum(c.shape[0] for c in chunks)
        assert total_tokens == ct.shape[0]

    def test_chunk_uneven_raises(self):
        data = torch.randn(7, 8)
        ct = CommTensor(data, CommLayout.BLOCKED_BY_DST)
        with pytest.raises(ValueError, match="equal chunks"):
            ct.chunk(3)


# ---------------------------------------------------------------------------
# CommTensor repr
# ---------------------------------------------------------------------------

class TestRepr:
    def test_repr_contains_layout(self, hidden, simple_routing_table):
        ct = CommTensor.from_dense(hidden, simple_routing_table)
        r = repr(ct)
        assert "blocked_by_dst" in r
        assert "CommTensor" in r
