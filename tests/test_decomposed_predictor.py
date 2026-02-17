"""Tests for DecomposedJASPERPredictor."""

from __future__ import annotations

import pytest
import torch

from RAG_supporters.nn.models.decomposed_predictor import (
    DecomposedJASPERPredictor,
    DecomposedJASPERConfig,
)


# ---------------------------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------------------------

B, D, H, K = 8, 32, 16, 4


def _make_model(fine_input_mode: str = "concat") -> DecomposedJASPERPredictor:
    return DecomposedJASPERPredictor(
        DecomposedJASPERConfig(
            embedding_dim=D,
            hidden_dim=H,
            num_subspaces=K,
            num_layers=2,
            router_hidden_dim=H,
            fine_input_mode=fine_input_mode,
        )
    )


def _make_inputs(seed: int = 42):
    torch.manual_seed(seed)
    q = torch.randn(B, D)
    s = torch.randn(B, D)
    c = torch.randn(K, D)
    return q, s, c


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------


class TestInit:
    def test_sub_modules_present(self):
        model = _make_model()
        assert hasattr(model, "question_encoder")
        assert hasattr(model, "steering_encoder")
        assert hasattr(model, "router")
        assert hasattr(model, "fine_mlp")

    def test_num_subspaces(self):
        model = _make_model()
        assert model.num_subspaces == K

    def test_embedding_dim_property(self):
        model = _make_model()
        assert model.embedding_dim == D

    def test_hidden_dim_property(self):
        model = _make_model()
        assert model.hidden_dim == H

    def test_from_dict(self):
        model = DecomposedJASPERPredictor({
            "embedding_dim": D,
            "hidden_dim": H,
            "num_subspaces": K,
            "num_layers": 2,
            "router_hidden_dim": H,
        })
        assert model.num_subspaces == K

    def test_invalid_fine_input_mode_raises(self):
        with pytest.raises(ValueError, match="fine_input_mode"):
            DecomposedJASPERConfig(
                embedding_dim=D, hidden_dim=H, num_subspaces=K, fine_input_mode="invalid"
            )

    def test_add_mode_has_coarse_projector(self):
        model = _make_model(fine_input_mode="add")
        assert model.coarse_projector is not None

    def test_concat_mode_no_coarse_projector(self):
        model = _make_model(fine_input_mode="concat")
        assert model.coarse_projector is None

    def test_get_model_summary(self):
        model = _make_model()
        summary = model.get_model_summary()
        assert "DecomposedJASPERPredictor" in summary

    def test_repr(self):
        model = _make_model()
        r = repr(model)
        assert "DecomposedJASPERPredictor" in r


# ---------------------------------------------------------------------------
# TestForward
# ---------------------------------------------------------------------------


class TestForward:
    def test_prediction_shape(self):
        model = _make_model()
        q, s, c = _make_inputs()
        pred, _ = model(q, s, c)
        assert pred.shape == (B, D)

    def test_explanation_dict_keys(self):
        model = _make_model()
        q, s, c = _make_inputs()
        _, xai = model(q, s, c)
        for key in ("routing_weights", "concept_logits", "coarse", "fine", "atypicality"):
            assert key in xai, f"Missing key: {key}"

    def test_explanation_shapes(self):
        model = _make_model()
        q, s, c = _make_inputs()
        _, xai = model(q, s, c)
        assert xai["routing_weights"].shape == (B, K)
        assert xai["concept_logits"].shape == (B, K)
        assert xai["coarse"].shape == (B, D)
        assert xai["fine"].shape == (B, D)
        assert xai["atypicality"].shape == (B,)

    def test_explanation_all_detached(self):
        model = _make_model()
        q, s, c = _make_inputs()
        _, xai = model(q, s, c)
        for key, val in xai.items():
            assert not val.requires_grad, f"Tensor '{key}' should be detached"

    def test_no_nan_in_prediction(self):
        model = _make_model()
        q, s, c = _make_inputs()
        pred, xai = model(q, s, c)
        assert not torch.isnan(pred).any()
        for key, val in xai.items():
            assert not torch.isnan(val).any(), f"NaN in '{key}'"

    def test_wrong_centroid_K_raises(self):
        model = _make_model()
        q, s, _ = _make_inputs()
        wrong_c = torch.randn(K + 1, D)
        with pytest.raises(ValueError, match="centroid_embs"):
            model(q, s, wrong_c)

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_different_batch_sizes(self, batch_size: int):
        model = _make_model()
        torch.manual_seed(0)
        q = torch.randn(batch_size, D)
        s = torch.randn(batch_size, D)
        c = torch.randn(K, D)
        pred, xai = model(q, s, c)
        assert pred.shape == (batch_size, D)
        assert xai["routing_weights"].shape == (batch_size, K)

    def test_routing_weights_sum_to_one(self):
        model = _make_model()
        q, s, c = _make_inputs()
        _, xai = model(q, s, c)
        sums = xai["routing_weights"].sum(dim=-1)
        assert torch.allclose(sums, torch.ones(B), atol=1e-5)


# ---------------------------------------------------------------------------
# TestArithmetic
# ---------------------------------------------------------------------------


class TestArithmetic:
    def test_prediction_equals_coarse_plus_fine(self):
        """Verify that prediction = coarse + fine numerically."""
        model = _make_model()
        model.eval()
        q, s, c = _make_inputs()

        with torch.no_grad():
            prediction, xai = model(q, s, c, training=False)

        # Reconstruct from detached components (detach loses the addition graph,
        # but the values should match because the values were computed that way)
        reconstructed = xai["coarse"] + xai["fine"]
        assert torch.allclose(prediction.detach(), reconstructed, atol=1e-5), (
            f"Max diff: {(prediction.detach() - reconstructed).abs().max().item()}"
        )

    def test_atypicality_equals_fine_norm(self):
        model = _make_model()
        q, s, c = _make_inputs()
        with torch.no_grad():
            _, xai = model(q, s, c, training=False)
        expected = xai["fine"].norm(dim=-1)
        assert torch.allclose(xai["atypicality"], expected, atol=1e-5)


# ---------------------------------------------------------------------------
# TestFineModes
# ---------------------------------------------------------------------------


class TestFineModes:
    def test_add_mode_produces_valid_output(self):
        model = _make_model(fine_input_mode="add")
        q, s, c = _make_inputs()
        pred, xai = model(q, s, c)
        assert pred.shape == (B, D)
        assert not torch.isnan(pred).any()

    def test_concat_mode_same_shapes_as_add(self):
        m_concat = _make_model(fine_input_mode="concat")
        m_add = _make_model(fine_input_mode="add")
        q, s, c = _make_inputs()
        pred_c, _ = m_concat(q, s, c)
        pred_a, _ = m_add(q, s, c)
        assert pred_c.shape == pred_a.shape


# ---------------------------------------------------------------------------
# TestGradients
# ---------------------------------------------------------------------------


class TestGradients:
    def test_all_params_receive_gradient(self):
        model = _make_model()
        q, s, c = _make_inputs()
        pred, _ = model(q, s, c)
        pred.sum().backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for param: {name}"

    def test_centroid_embs_receives_gradient(self):
        model = _make_model()
        q, s, _ = _make_inputs()
        c = torch.randn(K, D, requires_grad=True)
        pred, _ = model(q, s, c)
        pred.sum().backward()
        assert c.grad is not None
        assert c.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# TestLatentRepresentations
# ---------------------------------------------------------------------------


class TestLatents:
    def test_cached_after_forward(self):
        model = _make_model()
        q, s, c = _make_inputs()
        model(q, s, c)
        latents = model.get_latent_representations()
        assert "question_latent" in latents
        assert "steering_latent" in latents
        assert "routing_weights" in latents

    def test_latents_are_detached(self):
        model = _make_model()
        q, s, c = _make_inputs()
        model(q, s, c)
        latents = model.get_latent_representations()
        for key, val in latents.items():
            assert not val.requires_grad, f"Latent '{key}' should not require grad"

    def test_latents_on_cpu(self):
        model = _make_model()
        q, s, c = _make_inputs()
        model(q, s, c)
        latents = model.get_latent_representations()
        for val in latents.values():
            assert val.device.type == "cpu"

    def test_get_routing_weights(self):
        model = _make_model()
        q, s, c = _make_inputs()
        # get_routing_weights does not need centroid_embs
        weights = model.get_routing_weights(q, s)
        assert weights.shape == (B, K)
