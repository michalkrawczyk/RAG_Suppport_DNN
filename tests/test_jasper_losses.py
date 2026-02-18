"""Tests for JASPER multi-objective loss functions."""

import pytest
import torch
import torch.nn as nn

from RAG_supporters.nn.losses.jasper_losses import (
    JASPERLoss,
    ContrastiveLoss,
    CentroidLoss,
    VICRegLoss,
    JASPERMultiObjectiveLoss,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rand(B: int, D: int) -> torch.Tensor:
    return torch.randn(B, D)


def _make_batch(B: int = 8, D: int = 32, K: int = 4, C: int = 6):
    """Return a consistent dummy batch."""
    torch.manual_seed(99)
    predicted = _rand(B, D)
    ema_target = _rand(B, D)
    negatives = _rand(B * K, D).view(B, K, D)
    centroids = _rand(C, D)
    cluster_ids = torch.randint(0, C, (B,))
    return predicted, ema_target, negatives, centroids, cluster_ids


# ---------------------------------------------------------------------------
# JASPERLoss
# ---------------------------------------------------------------------------


class TestJASPERLoss:
    def test_returns_dict(self):
        loss_fn = JASPERLoss()
        p, t, *_ = _make_batch()
        result = loss_fn(p, t)
        assert isinstance(result, dict)
        assert "jasper" in result

    def test_loss_is_scalar(self):
        loss_fn = JASPERLoss()
        p, t, *_ = _make_batch()
        result = loss_fn(p, t)
        assert result["jasper"].shape == ()

    def test_loss_is_non_negative(self):
        loss_fn = JASPERLoss()
        p, t, *_ = _make_batch()
        result = loss_fn(p, t)
        assert result["jasper"].item() >= 0

    def test_zero_loss_for_identical_inputs(self):
        loss_fn = JASPERLoss()
        p = torch.ones(4, 16)
        result = loss_fn(p, p.clone())
        assert result["jasper"].item() < 1e-6

    def test_gradients_flow(self):
        loss_fn = JASPERLoss()
        p = _rand(4, 16).requires_grad_(True)
        t = _rand(4, 16)
        result = loss_fn(p, t)
        result["jasper"].backward()
        assert p.grad is not None

    def test_invalid_reduction_raises(self):
        with pytest.raises(ValueError):
            JASPERLoss(reduction="invalid")


# ---------------------------------------------------------------------------
# ContrastiveLoss
# ---------------------------------------------------------------------------


class TestContrastiveLoss:
    def test_returns_dict(self):
        loss_fn = ContrastiveLoss()
        p, t, neg, *_ = _make_batch()
        result = loss_fn(p, t, neg)
        assert "contrastive" in result

    def test_output_is_scalar(self):
        loss_fn = ContrastiveLoss()
        p, t, neg, *_ = _make_batch()
        result = loss_fn(p, t, neg)
        assert result["contrastive"].shape == ()

    def test_non_negative_loss(self):
        loss_fn = ContrastiveLoss()
        p, t, neg, *_ = _make_batch()
        assert loss_fn(p, t, neg)["contrastive"].item() >= 0

    def test_lower_temperature_gives_larger_loss_variance(self):
        """Lower temperature should generally sharpen predictions."""
        p, t, neg, *_ = _make_batch(B=16)
        loss_hi = ContrastiveLoss(temperature=1.0)(p, t, neg)["contrastive"]
        loss_lo = ContrastiveLoss(temperature=0.01)(p, t, neg)["contrastive"]
        # With random embeddings and low temperature the loss should be higher
        assert loss_lo.item() != loss_hi.item()

    def test_gradients_flow(self):
        loss_fn = ContrastiveLoss()
        p, t, neg, *_ = _make_batch(B=4)
        p = p.requires_grad_(True)
        result = loss_fn(p, t, neg)
        result["contrastive"].backward()
        assert p.grad is not None

    def test_invalid_temperature_raises(self):
        with pytest.raises(ValueError):
            ContrastiveLoss(temperature=0.0)

    def test_single_negative(self):
        loss_fn = ContrastiveLoss()
        p = _rand(4, 32)
        t = _rand(4, 32)
        neg = torch.randn(4, 1, 32)
        result = loss_fn(p, t, neg)
        assert result["contrastive"].shape == ()


# ---------------------------------------------------------------------------
# CentroidLoss
# ---------------------------------------------------------------------------


class TestCentroidLoss:
    def test_returns_dict(self):
        loss_fn = CentroidLoss()
        p, _, _, centroids, cluster_ids = _make_batch()
        result = loss_fn(p, centroids, cluster_ids)
        assert "centroid" in result
        assert "centroid_acc" in result

    def test_loss_is_scalar(self):
        loss_fn = CentroidLoss()
        p, _, _, centroids, cluster_ids = _make_batch()
        result = loss_fn(p, centroids, cluster_ids)
        assert result["centroid"].shape == ()

    def test_accuracy_in_range(self):
        loss_fn = CentroidLoss()
        p, _, _, centroids, cluster_ids = _make_batch()
        acc = loss_fn(p, centroids, cluster_ids)["centroid_acc"].item()
        assert 0.0 <= acc <= 1.0

    def test_perfect_prediction_gives_high_accuracy(self):
        """If predicted == centroid for each sample, accuracy should be 1.0."""
        C, D = 4, 32
        centroids = torch.eye(D)[:C]   # orthogonal centroids
        cluster_ids = torch.arange(C)
        # Predicted = exact centroids
        loss_fn = CentroidLoss(temperature=0.01)
        result = loss_fn(centroids.clone(), centroids, cluster_ids)
        assert result["centroid_acc"].item() > 0.9

    def test_gradients_flow(self):
        loss_fn = CentroidLoss()
        p, _, _, centroids, cluster_ids = _make_batch(B=4)
        p = p.requires_grad_(True)
        result = loss_fn(p, centroids, cluster_ids)
        result["centroid"].backward()
        assert p.grad is not None

    def test_accuracy_is_detached(self):
        loss_fn = CentroidLoss()
        p, _, _, centroids, cluster_ids = _make_batch(B=4)
        p = p.requires_grad_(True)
        result = loss_fn(p, centroids, cluster_ids)
        assert not result["centroid_acc"].requires_grad


# ---------------------------------------------------------------------------
# VICRegLoss
# ---------------------------------------------------------------------------


class TestVICRegLoss:
    def test_returns_dict_with_components(self):
        loss_fn = VICRegLoss()
        p, t, *_ = _make_batch()
        result = loss_fn(p, t)
        assert "vicreg" in result
        assert "vicreg_v" in result
        assert "vicreg_i" in result
        assert "vicreg_c" in result

    def test_total_loss_is_scalar(self):
        loss_fn = VICRegLoss()
        p, t, *_ = _make_batch()
        result = loss_fn(p, t)
        assert result["vicreg"].shape == ()

    def test_non_negative_total_loss(self):
        loss_fn = VICRegLoss()
        p, t, *_ = _make_batch()
        assert loss_fn(p, t)["vicreg"].item() >= 0

    def test_gradients_flow(self):
        loss_fn = VICRegLoss()
        p = _rand(8, 32).requires_grad_(True)
        t = _rand(8, 32)
        result = loss_fn(p, t)
        result["vicreg"].backward()
        assert p.grad is not None

    def test_vicreg_prevents_collapse(self):
        """Constant embeddings (collapsed) should produce high variance loss."""
        loss_fn = VICRegLoss(lambda_v=25.0, lambda_i=0.0, lambda_c=0.0)
        B, D = 32, 16
        # Collapsed: all embeddings identical → variance ≈ 0 → high variance penalty
        collapsed = torch.ones(B, D) * 3.0
        result_collapsed = loss_fn(collapsed, collapsed)

        # Normal: random embeddings → variance ≈ 1 → low variance penalty
        random = torch.randn(B, D)
        result_random = loss_fn(random, random)

        assert result_collapsed["vicreg"].item() > result_random["vicreg"].item()

    def test_sub_components_are_detached(self):
        loss_fn = VICRegLoss()
        p = _rand(8, 32).requires_grad_(True)
        t = _rand(8, 32)
        result = loss_fn(p, t)
        for key in ("vicreg_v", "vicreg_i", "vicreg_c"):
            assert not result[key].requires_grad


# ---------------------------------------------------------------------------
# JASPERMultiObjectiveLoss
# ---------------------------------------------------------------------------


class TestJASPERMultiObjectiveLoss:
    @pytest.fixture
    def loss_fn(self):
        return JASPERMultiObjectiveLoss(
            lambda_jasper=1.0,
            lambda_contrastive=0.5,
            lambda_centroid=0.1,
            lambda_vicreg=0.1,
        )

    @pytest.fixture
    def batch_inputs(self):
        return _make_batch(B=8, D=32, K=4, C=6)

    def test_returns_total_and_dict(self, loss_fn, batch_inputs):
        p, t, neg, c, ids = batch_inputs
        total, loss_dict = loss_fn(p, t, neg, c, ids)
        assert isinstance(total, torch.Tensor)
        assert isinstance(loss_dict, dict)
        assert "total" in loss_dict

    def test_total_is_scalar(self, loss_fn, batch_inputs):
        p, t, neg, c, ids = batch_inputs
        total, _ = loss_fn(p, t, neg, c, ids)
        assert total.shape == ()

    def test_all_components_present(self, loss_fn, batch_inputs):
        p, t, neg, c, ids = batch_inputs
        _, loss_dict = loss_fn(p, t, neg, c, ids)
        for key in ("jasper", "contrastive", "centroid", "vicreg"):
            assert key in loss_dict, f"Missing key: {key}"

    def test_total_gradients_flow(self, loss_fn, batch_inputs):
        _, t, neg, c, ids = batch_inputs
        p = _rand(8, 32).requires_grad_(True)
        total, _ = loss_fn(p, t, neg, c, ids)
        total.backward()
        assert p.grad is not None

    def test_sub_components_detached(self, loss_fn, batch_inputs):
        _, t, neg, c, ids = batch_inputs
        p = _rand(8, 32).requires_grad_(True)
        total, loss_dict = loss_fn(p, t, neg, c, ids)
        for key, val in loss_dict.items():
            if key != "total":
                assert not val.requires_grad, f"{key} should be detached"

    def test_zero_weights_produce_zero_loss(self, batch_inputs):
        loss_fn = JASPERMultiObjectiveLoss(
            lambda_jasper=0.0,
            lambda_contrastive=0.0,
            lambda_centroid=0.0,
            lambda_vicreg=0.0,
        )
        p, t, neg, c, ids = batch_inputs
        total, _ = loss_fn(p, t, neg, c, ids)
        assert abs(total.item()) < 1e-6

    def test_repr(self, loss_fn):
        r = repr(loss_fn)
        assert "JASPERMultiObjectiveLoss" in r
