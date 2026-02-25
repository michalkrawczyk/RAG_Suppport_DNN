"""Tests for JASPER and routing loss functions.

Merged from:
- test_jasper_losses.py  (JASPERLoss, ContrastiveLoss, CentroidLoss, VICRegLoss, JASPERMultiObjectiveLoss)
- test_routing_losses.py (RoutingLoss, EntropyRegularization, ResidualPenalty, DisentanglementLoss)
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from RAG_supporters.nn.losses.jasper_losses import (
    JASPERLoss,
    ContrastiveLoss,
    CentroidLoss,
    VICRegLoss,
    JASPERMultiObjectiveLoss,
)
from RAG_supporters.nn.losses.routing_losses import (
    DisentanglementLoss,
    EntropyRegularization,
    ResidualPenalty,
    RoutingLoss,
)


# ===========================================================================
# Shared helpers
# ===========================================================================


def _rand(B: int, D: int) -> torch.Tensor:
    return torch.randn(B, D)


def _make_jasper_batch(B: int = 8, D: int = 32, K: int = 4, C: int = 6):
    """Return a consistent dummy batch for JASPER losses."""
    torch.manual_seed(99)
    predicted = _rand(B, D)
    ema_target = _rand(B, D)
    negatives = _rand(B * K, D).view(B, K, D)
    centroids = _rand(C, D)
    cluster_ids = torch.randint(0, C, (B,))
    return predicted, ema_target, negatives, centroids, cluster_ids


_B, _K, _D = 8, 4, 32


def _make_routing_inputs(seed: int = 99):
    torch.manual_seed(seed)
    logits = torch.randn(_B, _K)
    weights = F.softmax(logits, dim=-1)
    cluster_ids = torch.randint(0, _K, (_B,))
    fine_vector = torch.randn(_B, _D)
    return logits, weights, cluster_ids, fine_vector


# ===========================================================================
# JASPER Losses
# ===========================================================================


class TestJASPERLoss:
    def test_output_format(self):
        loss_fn = JASPERLoss()
        p, t, *_ = _make_jasper_batch()
        result = loss_fn(p, t)
        assert isinstance(result, dict), "JASPERLoss should return a dict"
        assert "jasper" in result, "Result dict must contain 'jasper' key"
        assert result["jasper"].shape == (), "JASPERLoss should return a scalar tensor"
        assert result["jasper"].item() >= 0, "JASPERLoss should always be non-negative"

    def test_zero_loss_for_identical_inputs(self):
        loss_fn = JASPERLoss()
        p = torch.ones(4, 16)
        result = loss_fn(p, p.clone())
        assert (
            result["jasper"].item() < 1e-6
        ), "JASPERLoss for identical predicted and target should be ~0"

    def test_gradients_flow(self):
        loss_fn = JASPERLoss()
        p = _rand(4, 16).requires_grad_(True)
        t = _rand(4, 16)
        result = loss_fn(p, t)
        result["jasper"].backward()
        assert p.grad is not None, "Gradient must flow back to predicted embeddings"

    def test_invalid_reduction_raises(self):
        with pytest.raises(ValueError):
            JASPERLoss(reduction="invalid")


class TestContrastiveLoss:
    def test_output_format(self):
        loss_fn = ContrastiveLoss()
        p, t, neg, *_ = _make_jasper_batch()
        result = loss_fn(p, t, neg)
        assert "contrastive" in result, "ContrastiveLoss result dict must contain 'contrastive' key"
        assert result["contrastive"].shape == (), "ContrastiveLoss should return a scalar tensor"
        assert result["contrastive"].item() >= 0, "ContrastiveLoss should always be non-negative"

    def test_lower_temperature_gives_larger_loss_variance(self):
        """Lower temperature should generally sharpen predictions."""
        p, t, neg, *_ = _make_jasper_batch(B=16)
        loss_hi = ContrastiveLoss(temperature=1.0)(p, t, neg)["contrastive"]
        loss_lo = ContrastiveLoss(temperature=0.01)(p, t, neg)["contrastive"]
        assert (
            loss_lo.item() != loss_hi.item()
        ), "ContrastiveLoss at different temperatures should yield different values"

    def test_gradients_flow(self):
        loss_fn = ContrastiveLoss()
        p, t, neg, *_ = _make_jasper_batch(B=4)
        p = p.requires_grad_(True)
        result = loss_fn(p, t, neg)
        result["contrastive"].backward()
        assert (
            p.grad is not None
        ), "Gradient must flow back to predicted embeddings in ContrastiveLoss"

    def test_invalid_temperature_raises(self):
        with pytest.raises(ValueError):
            ContrastiveLoss(temperature=0.0)

    def test_single_negative(self):
        loss_fn = ContrastiveLoss()
        p = _rand(4, 32)
        t = _rand(4, 32)
        neg = torch.randn(4, 1, 32)
        result = loss_fn(p, t, neg)
        assert (
            result["contrastive"].shape == ()
        ), "ContrastiveLoss with single negative should still return a scalar"


class TestCentroidLoss:
    def test_output_format(self):
        loss_fn = CentroidLoss()
        p, _, _, centroids, cluster_ids = _make_jasper_batch()
        result = loss_fn(p, centroids, cluster_ids)
        assert "centroid" in result, "CentroidLoss result dict must contain 'centroid' key"
        assert "centroid_acc" in result, "CentroidLoss result dict must contain 'centroid_acc' key"
        assert result["centroid"].shape == (), "CentroidLoss should return a scalar tensor"

    def test_accuracy_in_range(self):
        loss_fn = CentroidLoss()
        p, _, _, centroids, cluster_ids = _make_jasper_batch()
        acc = loss_fn(p, centroids, cluster_ids)["centroid_acc"].item()
        assert 0.0 <= acc <= 1.0, f"centroid_acc={acc} is outside [0, 1]"

    def test_perfect_prediction_gives_high_accuracy(self):
        """If predicted == centroid for each sample, accuracy should be 1.0."""
        C, D = 4, 32
        centroids = torch.eye(D)[:C]
        cluster_ids = torch.arange(C)
        loss_fn = CentroidLoss(temperature=0.01)
        result = loss_fn(centroids.clone(), centroids, cluster_ids)
        assert (
            result["centroid_acc"].item() > 0.9
        ), "CentroidLoss accuracy should be near 1.0 when predicted == centroids"

    def test_gradients_flow(self):
        loss_fn = CentroidLoss()
        p, _, _, centroids, cluster_ids = _make_jasper_batch(B=4)
        p = p.requires_grad_(True)
        result = loss_fn(p, centroids, cluster_ids)
        result["centroid"].backward()
        assert p.grad is not None, "Gradient must flow back to predicted embeddings in CentroidLoss"

    def test_accuracy_is_detached(self):
        loss_fn = CentroidLoss()
        p, _, _, centroids, cluster_ids = _make_jasper_batch(B=4)
        p = p.requires_grad_(True)
        result = loss_fn(p, centroids, cluster_ids)
        assert not result[
            "centroid_acc"
        ].requires_grad, (
            "centroid_acc must be detached (no gradient) to avoid interfering with backprop"
        )


class TestVICRegLoss:
    def test_output_format(self):
        loss_fn = VICRegLoss()
        p, t, *_ = _make_jasper_batch()
        result = loss_fn(p, t)
        assert "vicreg" in result, "VICRegLoss result must contain 'vicreg' key"
        assert "vicreg_v" in result, "VICRegLoss result must contain 'vicreg_v' (variance) key"
        assert "vicreg_i" in result, "VICRegLoss result must contain 'vicreg_i' (invariance) key"
        assert "vicreg_c" in result, "VICRegLoss result must contain 'vicreg_c' (covariance) key"
        assert result["vicreg"].shape == (), "VICRegLoss total should be a scalar tensor"
        assert result["vicreg"].item() >= 0, "VICRegLoss total should always be non-negative"

    def test_gradients_flow(self):
        loss_fn = VICRegLoss()
        p = _rand(8, 32).requires_grad_(True)
        t = _rand(8, 32)
        result = loss_fn(p, t)
        result["vicreg"].backward()
        assert p.grad is not None, "Gradient must flow back to predicted embeddings in VICRegLoss"

    def test_vicreg_prevents_collapse(self):
        """Constant embeddings (collapsed) should produce high variance loss."""
        loss_fn = VICRegLoss(lambda_v=25.0, lambda_i=0.0, lambda_c=0.0)
        B, D = 32, 16
        collapsed = torch.ones(B, D) * 3.0
        result_collapsed = loss_fn(collapsed, collapsed)
        random = torch.randn(B, D)
        result_random = loss_fn(random, random)
        assert (
            result_collapsed["vicreg"].item() > result_random["vicreg"].item()
        ), "Collapsed embeddings should produce higher VICReg variance loss than random embeddings"

    def test_sub_components_are_detached(self):
        loss_fn = VICRegLoss()
        p = _rand(8, 32).requires_grad_(True)
        t = _rand(8, 32)
        result = loss_fn(p, t)
        for key in ("vicreg_v", "vicreg_i", "vicreg_c"):
            assert not result[
                key
            ].requires_grad, f"VICReg component '{key}' must be detached from the computation graph"


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
        return _make_jasper_batch(B=8, D=32, K=4, C=6)

    def test_output_format(self, loss_fn, batch_inputs):
        p, t, neg, c, ids = batch_inputs
        total, loss_dict = loss_fn(p, t, neg, c, ids)
        assert isinstance(total, torch.Tensor), "JASPERMultiObjectiveLoss should return a Tensor as first output"
        assert isinstance(loss_dict, dict), "JASPERMultiObjectiveLoss should return a dict as second output"
        assert "total" in loss_dict, "Loss dict must contain 'total' key"
        assert total.shape == (), "JASPERMultiObjectiveLoss total should be a scalar tensor"
        for key in ("jasper", "contrastive", "centroid", "vicreg"):
            assert key in loss_dict, f"Missing key: {key}"

    def test_total_gradients_flow(self, loss_fn, batch_inputs):
        _, t, neg, c, ids = batch_inputs
        p = _rand(8, 32).requires_grad_(True)
        total, _ = loss_fn(p, t, neg, c, ids)
        total.backward()
        assert (
            p.grad is not None
        ), "Gradient must flow back to predicted embeddings through combined JASPER loss"

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
        assert abs(total.item()) < 1e-6, "All-zero lambda weights should produce ~0 total loss"

    def test_repr(self, loss_fn):
        r = repr(loss_fn)
        assert "JASPERMultiObjectiveLoss" in r, "repr should contain 'JASPERMultiObjectiveLoss'"


# ===========================================================================
# Routing Losses
# ===========================================================================


class TestRoutingLoss:
    def test_output_format(self):
        loss_fn = RoutingLoss()
        logits, _, ids, _ = _make_routing_inputs()
        result = loss_fn(logits, ids)
        assert "routing" in result, "RoutingLoss result must contain 'routing' key"
        assert "routing_acc" in result, "RoutingLoss result must contain 'routing_acc' key"
        assert result["routing"].shape == (), "RoutingLoss should return a scalar tensor"
        assert result["routing"].item() >= 0, "RoutingLoss should always be non-negative"

    def test_accuracy_in_range(self):
        loss_fn = RoutingLoss()
        logits, _, ids, _ = _make_routing_inputs()
        result = loss_fn(logits, ids)
        acc = result["routing_acc"].item()
        assert 0.0 <= acc <= 1.0, f"routing_acc={acc} is outside [0, 1]"

    def test_accuracy_is_detached(self):
        loss_fn = RoutingLoss()
        logits, _, ids, _ = _make_routing_inputs()
        result = loss_fn(logits, ids)
        assert not result[
            "routing_acc"
        ].requires_grad, (
            "routing_acc must be detached (no gradient) to avoid interfering with backprop"
        )

    def test_gradients_flow(self):
        loss_fn = RoutingLoss()
        logits = torch.randn(_B, _K, requires_grad=True)
        ids = torch.randint(0, _K, (_B,))
        result = loss_fn(logits, ids)
        result["routing"].backward()
        assert logits.grad is not None, "Gradient must flow back to routing logits"

    def test_perfect_logits_give_perfect_accuracy(self):
        loss_fn = RoutingLoss()
        ids = torch.arange(_B) % _K
        logits = torch.zeros(_B, _K)
        for i, cid in enumerate(ids):
            logits[i, cid] = 10.0
        result = loss_fn(logits, ids)
        assert result["routing_acc"].item() == pytest.approx(1.0, abs=1e-5)

    def test_invalid_label_smoothing_raises(self):
        with pytest.raises(ValueError, match="label_smoothing"):
            RoutingLoss(label_smoothing=1.0)

    def test_weight_applied(self):
        logits, _, ids, _ = _make_routing_inputs()
        r1 = RoutingLoss(weight=1.0)(logits, ids)
        r2 = RoutingLoss(weight=2.0)(logits, ids)
        assert r2["routing"].item() == pytest.approx(2.0 * r1["routing"].item(), rel=1e-5)


class TestEntropyRegularization:
    def test_output_format(self):
        loss_fn = EntropyRegularization()
        _, weights, _, _ = _make_routing_inputs()
        result = loss_fn(weights, current_epoch=0)
        assert "entropy_reg" in result, "EntropyRegularization result must contain 'entropy_reg' key"
        assert "routing_entropy" in result, "EntropyRegularization result must contain 'routing_entropy' key"
        assert result["entropy_reg"].shape == (), "EntropyRegularization should return a scalar tensor"

    def test_entropy_is_detached(self):
        loss_fn = EntropyRegularization()
        _, weights, _, _ = _make_routing_inputs()
        result = loss_fn(weights, current_epoch=0)
        assert not result[
            "routing_entropy"
        ].requires_grad, "routing_entropy must be detached to avoid interfering with backprop"

    def test_uniform_weights_give_max_entropy(self):
        """Uniform weights should give entropy ≈ log(K)."""
        loss_fn = EntropyRegularization()
        uniform = torch.ones(_B, _K) / _K
        result = loss_fn(uniform, current_epoch=0)
        expected_entropy = math.log(_K)
        actual_entropy = result["routing_entropy"].item()
        assert (
            abs(actual_entropy - expected_entropy) < 0.01
        ), f"Expected H≈{expected_entropy:.3f}, got {actual_entropy:.3f}"

    def test_entropy_schedule(self):
        loss_fn_10 = EntropyRegularization(entropy_high=2.0, entropy_low=0.1, anneal_epochs=10)
        t0 = loss_fn_10.get_target_entropy(0)
        t5 = loss_fn_10.get_target_entropy(5)
        t10 = loss_fn_10.get_target_entropy(10)
        assert t0 > t5 > t10, f"Expected t0({t0:.3f}) > t5({t5:.3f}) > t10({t10:.3f})"
        loss_fn_5 = EntropyRegularization(entropy_high=2.0, entropy_low=0.1, anneal_epochs=5)
        assert loss_fn_5.get_target_entropy(5) == pytest.approx(
            0.1, abs=1e-6
        ), "After anneal_epochs the schedule should clamp at entropy_low"
        assert loss_fn_5.get_target_entropy(100) == pytest.approx(
            0.1, abs=1e-6
        ), "Well after anneal_epochs the schedule should stay clamped at entropy_low"

    def test_entropy_high_defaults_to_log_k(self):
        loss_fn = EntropyRegularization(entropy_high=None)
        uniform = torch.ones(_B, _K) / _K
        loss_fn(uniform, current_epoch=0)
        assert (
            abs(loss_fn._entropy_high - math.log(_K)) < 1e-5
        ), f"_entropy_high should default to log(K)={math.log(_K):.4f} for K={_K} subspaces"

    def test_gradients_flow(self):
        loss_fn = EntropyRegularization()
        logits = torch.randn(_B, _K, requires_grad=True)
        weights = F.softmax(logits, dim=-1)
        result = loss_fn(weights, current_epoch=0)
        result["entropy_reg"].backward()
        assert (
            logits.grad is not None
        ), "Gradient must flow back through EntropyRegularization to routing logits"

    def test_invalid_entropy_low_raises(self):
        with pytest.raises(ValueError, match="entropy_low"):
            EntropyRegularization(entropy_low=-0.1)

    def test_invalid_anneal_epochs_raises(self):
        with pytest.raises(ValueError, match="anneal_epochs"):
            EntropyRegularization(anneal_epochs=0)


class TestResidualPenalty:
    def test_output_format(self):
        loss_fn = ResidualPenalty()
        _, _, _, fine = _make_routing_inputs()
        result = loss_fn(fine)
        assert "residual_penalty" in result, "ResidualPenalty result must contain 'residual_penalty' key"
        assert "residual_norm_mean" in result, "ResidualPenalty result must contain 'residual_norm_mean' key"
        assert result["residual_penalty"].shape == (), "ResidualPenalty should return a scalar tensor"

    def test_zero_loss_below_margin(self):
        """If all norms are below margin, penalty should be exactly 0."""
        loss_fn = ResidualPenalty(margin=100.0)
        fine = torch.randn(_B, _D) * 0.01
        result = loss_fn(fine)
        assert result["residual_penalty"].item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_loss_above_margin(self):
        """Large fine vectors should incur a positive penalty."""
        loss_fn = ResidualPenalty(margin=0.1)
        fine = torch.ones(_B, _D) * 10.0
        result = loss_fn(fine)
        assert (
            result["residual_penalty"].item() > 0
        ), "Residual penalty should be positive when fine-vector norms exceed the margin"

    def test_hinge_behaviour(self):
        """Only samples exceeding margin should contribute to the loss."""
        margin = 1.0
        loss_fn = ResidualPenalty(margin=margin, weight=1.0)
        fine = torch.zeros(_B, _D)
        fine[: _B // 2, 0] = 0.5
        fine[_B // 2 :, 0] = 5.0
        result = loss_fn(fine)
        assert (
            result["residual_penalty"].item() > 0
        ), "Hinge loss should be positive when half the batch exceeds the margin"

    def test_norm_mean_is_detached(self):
        loss_fn = ResidualPenalty()
        _, _, _, fine = _make_routing_inputs()
        result = loss_fn(fine)
        assert not result[
            "residual_norm_mean"
        ].requires_grad, "residual_norm_mean must be detached (monitoring metric, not for backprop)"

    def test_gradients_flow(self):
        loss_fn = ResidualPenalty(margin=0.01)
        fine = torch.randn(_B, _D, requires_grad=True)
        result = loss_fn(fine)
        result["residual_penalty"].backward()
        assert (
            fine.grad is not None
        ), "Gradient must flow back to fine residual vector through ResidualPenalty"

    def test_invalid_margin_raises(self):
        with pytest.raises(ValueError, match="margin"):
            ResidualPenalty(margin=0.0)

    def test_weight_applied(self):
        _, _, _, fine = _make_routing_inputs()
        r1 = ResidualPenalty(margin=0.01, weight=1.0)(fine)
        r2 = ResidualPenalty(margin=0.01, weight=3.0)(fine)
        assert r2["residual_penalty"].item() == pytest.approx(
            3.0 * r1["residual_penalty"].item(), rel=1e-5
        )


class TestDisentanglementLoss:
    def test_output_format(self):
        loss_fn = DisentanglementLoss()
        _, weights, _, _ = _make_routing_inputs()
        result = loss_fn(weights)
        assert "disentanglement" in result, "DisentanglementLoss result must contain 'disentanglement' key"
        assert result["disentanglement"].shape == (), "DisentanglementLoss should return a scalar tensor"

    def test_identical_rows_give_zero_loss(self):
        """Identical rows center to zero: off-diagonal covariance is 0, so loss is 0."""
        loss_fn = DisentanglementLoss(weight=1.0)
        torch.manual_seed(0)
        single_row = F.softmax(torch.randn(1, _K), dim=-1)
        correlated = single_row.expand(_B, -1)
        random_weights = F.softmax(torch.randn(_B, _K), dim=-1)
        loss_corr = loss_fn(correlated)["disentanglement"].item()
        loss_rand = loss_fn(random_weights)["disentanglement"].item()
        assert loss_corr == pytest.approx(0.0, abs=1e-6), (
            "Identical rows center to zero so off-diagonal covariance is 0 and loss should be 0"
        )
        assert loss_rand >= 0, "DisentanglementLoss should be non-negative for random inputs"

    def test_gradients_flow(self):
        loss_fn = DisentanglementLoss()
        logits = torch.randn(_B, _K, requires_grad=True)
        weights = F.softmax(logits, dim=-1)
        result = loss_fn(weights)
        result["disentanglement"].backward()
        assert (
            logits.grad is not None
        ), "Gradient must flow back through DisentanglementLoss to routing logits"

    def test_single_sample_returns_zero(self):
        """Covariance undefined for B=1; should return zero tensor."""
        loss_fn = DisentanglementLoss()
        weights = F.softmax(torch.randn(1, _K), dim=-1)
        result = loss_fn(weights)
        assert result["disentanglement"].item() == pytest.approx(0.0, abs=1e-8)

    def test_weight_applied(self):
        _, weights, _, _ = _make_routing_inputs()
        r1 = DisentanglementLoss(weight=1.0)(weights)
        r2 = DisentanglementLoss(weight=5.0)(weights)
        assert r2["disentanglement"].item() == pytest.approx(
            5.0 * r1["disentanglement"].item(), rel=1e-5
        )
