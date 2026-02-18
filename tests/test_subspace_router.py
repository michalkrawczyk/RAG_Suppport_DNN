"""Tests for SubspaceRouter."""

from __future__ import annotations

import math
from typing import List

import pytest
import torch

from RAG_supporters.nn.models.subspace_router import SubspaceRouter, SubspaceRouterConfig


# ---------------------------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------------------------

B, D, H, K = 8, 32, 16, 4


def _make_router(
    K: int = K,
    temperature: float = 1.0,
    gumbel_hard: bool = False,
    normalize_input: bool = True,
) -> SubspaceRouter:
    return SubspaceRouter(
        SubspaceRouterConfig(
            embedding_dim=D,
            hidden_dim=H,
            num_subspaces=K,
            num_layers=2,
            temperature=temperature,
            gumbel_hard=gumbel_hard,
            normalize_input=normalize_input,
        )
    )


def _make_inputs(seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    return torch.randn(B, D), torch.randn(B, D)


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------


class TestInit:
    def test_from_config_object(self):
        cfg = SubspaceRouterConfig(embedding_dim=D, hidden_dim=H, num_subspaces=K)
        router = SubspaceRouter(cfg)
        assert router.config is cfg

    def test_from_dict(self):
        router = SubspaceRouter({"embedding_dim": D, "hidden_dim": H, "num_subspaces": K})
        assert router.num_subspaces == K
        assert router.embedding_dim == D

    def test_router_mlp_present(self):
        router = _make_router()
        assert hasattr(router, "router_mlp")

    def test_invalid_temperature_raises(self):
        with pytest.raises(ValueError, match="temperature"):
            SubspaceRouterConfig(embedding_dim=D, hidden_dim=H, num_subspaces=K, temperature=0.0)

    def test_invalid_num_subspaces_raises(self):
        with pytest.raises(ValueError, match="num_subspaces"):
            SubspaceRouterConfig(embedding_dim=D, hidden_dim=H, num_subspaces=1)

    def test_invalid_num_layers_raises(self):
        with pytest.raises(ValueError, match="num_layers"):
            SubspaceRouterConfig(embedding_dim=D, hidden_dim=H, num_subspaces=K, num_layers=0)

    def test_invalid_activation_raises(self):
        with pytest.raises(ValueError, match="activation"):
            SubspaceRouterConfig(
                embedding_dim=D, hidden_dim=H, num_subspaces=K, activation="NonExistent"
            )

    def test_get_model_summary(self):
        router = _make_router()
        summary = router.get_model_summary()
        assert "SubspaceRouter" in summary
        assert str(K) in summary

    def test_repr(self):
        router = _make_router()
        r = repr(router)
        assert "SubspaceRouter" in r


# ---------------------------------------------------------------------------
# TestForward
# ---------------------------------------------------------------------------


class TestForward:
    def test_output_shapes(self):
        router = _make_router()
        q, s = _make_inputs()
        weights, logits = router(q, s, training=False)
        assert weights.shape == (B, K), f"Expected ({B},{K}), got {weights.shape}"
        assert logits.shape == (B, K), f"Expected ({B},{K}), got {logits.shape}"

    def test_routing_weights_sum_to_one(self):
        router = _make_router()
        q, s = _make_inputs()
        weights, _ = router(q, s, training=False)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(B), atol=1e-5), f"Row sums: {sums}"

    def test_routing_weights_non_negative(self):
        router = _make_router()
        q, s = _make_inputs()
        weights, _ = router(q, s, training=False)
        assert (weights >= 0).all()

    def test_no_nan_in_output(self):
        router = _make_router()
        q, s = _make_inputs()
        weights, logits = router(q, s, training=False)
        assert not torch.isnan(weights).any()
        assert not torch.isnan(logits).any()

    def test_no_nan_in_training_output(self):
        router = _make_router()
        q, s = _make_inputs()
        weights, logits = router(q, s, training=True)
        assert not torch.isnan(weights).any()
        assert not torch.isnan(logits).any()

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_different_batch_sizes(self, batch_size: int):
        router = _make_router()
        torch.manual_seed(0)
        q = torch.randn(batch_size, D)
        s = torch.randn(batch_size, D)
        weights, logits = router(q, s, training=False)
        assert weights.shape == (batch_size, K)
        assert logits.shape == (batch_size, K)

    def test_training_weights_sum_to_one(self):
        router = _make_router()
        q, s = _make_inputs()
        weights, _ = router(q, s, training=True)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(B), atol=1e-5)


# ---------------------------------------------------------------------------
# TestGumbel
# ---------------------------------------------------------------------------


class TestGumbel:
    def test_training_mode_is_stochastic(self):
        """Different Gumbel noise seeds should produce different routing weights."""
        router = _make_router()
        q, s = _make_inputs()
        torch.manual_seed(0)
        w1, _ = router(q, s, training=True)
        torch.manual_seed(99)
        w2, _ = router(q, s, training=True)
        assert not torch.allclose(w1, w2), "Training mode should produce stochastic outputs"

    def test_inference_mode_is_deterministic(self):
        """Without Gumbel noise, same inputs → same outputs."""
        router = _make_router()
        router.eval()
        q, s = _make_inputs()
        w1, _ = router(q, s, training=False)
        w2, _ = router(q, s, training=False)
        assert torch.allclose(w1, w2)

    def test_hard_gumbel_produces_one_hot(self):
        """With gumbel_hard=True each routing row should have a single 1.0."""
        router = _make_router(gumbel_hard=True)
        q, s = _make_inputs()
        torch.manual_seed(7)
        weights, _ = router(q, s, training=True)
        # Each row should have exactly one entry ≈ 1.0
        max_vals, _ = weights.max(dim=-1)
        assert torch.allclose(max_vals, torch.ones(B), atol=1e-5)
        # And all other values should be ≈ 0
        sorted_weights, _ = weights.sort(dim=-1, descending=True)
        assert torch.allclose(sorted_weights[:, 1:], torch.zeros(B, K - 1), atol=1e-5)


# ---------------------------------------------------------------------------
# TestRoutingCollapse
# ---------------------------------------------------------------------------


class TestRoutingCollapse:
    def test_routing_not_always_same_subspace(self):
        """Fresh initialisation should not immediately collapse to one subspace."""
        router = _make_router(K=K)
        torch.manual_seed(0)
        q = torch.randn(32, D)
        s = torch.randn(32, D)
        with torch.no_grad():
            weights, _ = router(q, s, training=False)
        assigned = weights.argmax(dim=-1)
        unique = assigned.unique()
        assert unique.numel() > 1, (
            f"All {len(assigned)} samples routed to the same subspace "
            f"(subspace {assigned[0].item()}); may indicate collapsed init."
        )


# ---------------------------------------------------------------------------
# TestGetPrimarySubspace
# ---------------------------------------------------------------------------


class TestGetPrimarySubspace:
    def test_output_shapes(self):
        router = _make_router()
        q, s = _make_inputs()
        cluster_ids, confidences = router.get_primary_subspace(q, s)
        assert cluster_ids.shape == (B,)
        assert confidences.shape == (B,)

    def test_cluster_ids_in_range(self):
        router = _make_router()
        q, s = _make_inputs()
        cluster_ids, _ = router.get_primary_subspace(q, s)
        assert (cluster_ids >= 0).all()
        assert (cluster_ids < K).all()

    def test_confidence_in_range(self):
        router = _make_router()
        q, s = _make_inputs()
        _, confidences = router.get_primary_subspace(q, s)
        assert (confidences >= 0).all()
        assert (confidences <= 1.0 + 1e-6).all()

    def test_no_grad_on_output(self):
        router = _make_router()
        q, s = _make_inputs()
        cluster_ids, confidences = router.get_primary_subspace(q, s)
        assert not confidences.requires_grad


# ---------------------------------------------------------------------------
# TestExplain
# ---------------------------------------------------------------------------


class TestExplain:
    def test_required_keys_present(self):
        router = _make_router()
        q, s = _make_inputs()
        names = [f"c{i}" for i in range(K)]
        result = router.explain(q, s, names)
        for key in ("routing_weights", "primary_subspace", "primary_confidence", "entropy", "cluster_names"):
            assert key in result, f"Missing key: {key}"

    def test_primary_subspace_in_names(self):
        router = _make_router()
        q, s = _make_inputs()
        names = [f"cluster_{i}" for i in range(K)]
        result = router.explain(q, s, names)
        assert result["primary_subspace"] in names

    def test_routing_weights_length(self):
        router = _make_router()
        q, s = _make_inputs()
        names = [f"c{i}" for i in range(K)]
        result = router.explain(q, s, names)
        assert len(result["routing_weights"]) == K

    def test_wrong_cluster_names_length_raises(self):
        router = _make_router()
        q, s = _make_inputs()
        with pytest.raises(ValueError):
            router.explain(q, s, ["only_one"])

    def test_single_sample_input(self):
        router = _make_router()
        q_single = torch.randn(D)
        s_single = torch.randn(D)
        names = [f"c{i}" for i in range(K)]
        result = router.explain(q_single, s_single, names)
        assert result["primary_subspace"] in names


# ---------------------------------------------------------------------------
# TestGradients
# ---------------------------------------------------------------------------


class TestGradients:
    def test_gradients_flow_to_router_mlp(self):
        router = _make_router()
        q, s = _make_inputs()
        weights, _ = router(q, s, training=True)
        loss = weights.sum()
        loss.backward()
        any_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in router.router_mlp.parameters()
        )
        assert any_grad, "No gradient reached router_mlp parameters"

    def test_no_gradients_without_training(self):
        """get_primary_subspace uses torch.no_grad — returned tensors have no grad."""
        router = _make_router()
        q, s = _make_inputs()
        _, confidences = router.get_primary_subspace(q, s)
        assert not confidences.requires_grad
