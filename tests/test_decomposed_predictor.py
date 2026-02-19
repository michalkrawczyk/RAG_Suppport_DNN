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
        assert hasattr(model, "question_encoder"), \
            "DecomposedJASPERPredictor must have a 'question_encoder' attribute"
        assert hasattr(model, "steering_encoder"), \
            "DecomposedJASPERPredictor must have a 'steering_encoder' attribute"
        assert hasattr(model, "router"), \
            "DecomposedJASPERPredictor must have a 'router' attribute"
        assert hasattr(model, "fine_mlp"), \
            "DecomposedJASPERPredictor must have a 'fine_mlp' attribute"

    def test_num_subspaces(self):
        model = _make_model()
        assert model.num_subspaces == K, \
            f"num_subspaces property should be {K}"

    def test_embedding_dim_property(self):
        model = _make_model()
        assert model.embedding_dim == D, \
            f"embedding_dim property should be {D}"

    def test_hidden_dim_property(self):
        model = _make_model()
        assert model.hidden_dim == H, \
            f"hidden_dim property should be {H}"

    def test_from_dict(self):
        model = DecomposedJASPERPredictor({
            "embedding_dim": D,
            "hidden_dim": H,
            "num_subspaces": K,
            "num_layers": 2,
            "router_hidden_dim": H,
        })
        assert model.num_subspaces == K, \
            "num_subspaces should be set correctly from dict config"

    def test_invalid_fine_input_mode_raises(self):
        with pytest.raises(ValueError, match="fine_input_mode"):
            DecomposedJASPERConfig(
                embedding_dim=D, hidden_dim=H, num_subspaces=K, fine_input_mode="invalid"
            )

    def test_add_mode_has_coarse_projector(self):
        model = _make_model(fine_input_mode="add")
        assert model.coarse_projector is not None, \
            "'add' fine_input_mode should create a coarse_projector"

    def test_concat_mode_no_coarse_projector(self):
        model = _make_model(fine_input_mode="concat")
        assert model.coarse_projector is None, \
            "'concat' fine_input_mode should not create a coarse_projector"

    def test_get_model_summary(self):
        model = _make_model()
        summary = model.get_model_summary()
        assert "DecomposedJASPERPredictor" in summary, \
            "Model summary should contain 'DecomposedJASPERPredictor'"

    def test_repr(self):
        model = _make_model()
        r = repr(model)
        assert "DecomposedJASPERPredictor" in r, "repr should contain 'DecomposedJASPERPredictor'"


# ---------------------------------------------------------------------------
# TestForward
# ---------------------------------------------------------------------------


class TestForward:
    def test_prediction_shape(self):
        model = _make_model()
        q, s, c = _make_inputs()
        pred, _ = model(q, s, c)
        assert pred.shape == (B, D), \
            f"Prediction shape should be ({B}, {D}), got {pred.shape}"

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
        assert xai["routing_weights"].shape == (B, K), \
            f"routing_weights shape should be ({B}, {K})"
        assert xai["concept_logits"].shape == (B, K), \
            f"concept_logits shape should be ({B}, {K})"
        assert xai["coarse"].shape == (B, D), \
            f"coarse vector shape should be ({B}, {D})"
        assert xai["fine"].shape == (B, D), \
            f"fine vector shape should be ({B}, {D})"
        assert xai["atypicality"].shape == (B,), \
            f"atypicality shape should be ({B},)"

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
        assert not torch.isnan(pred).any(), "Prediction must not contain NaN values"
        for key, val in xai.items():
            assert not torch.isnan(val).any(), f"NaN in '{key}'""

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
        assert torch.allclose(sums, torch.ones(B), atol=1e-5), \
            f"Routing weights should sum to 1.0 per sample, got: {sums}"


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
        assert torch.allclose(xai["atypicality"], expected, atol=1e-5), \
            "atypicality should equal the L2 norm of the fine residual vector"


# ---------------------------------------------------------------------------
# TestFineModes
# ---------------------------------------------------------------------------


class TestFineModes:
    def test_add_mode_produces_valid_output(self):
        model = _make_model(fine_input_mode="add")
        q, s, c = _make_inputs()
        pred, xai = model(q, s, c)
        assert pred.shape == (B, D), \
            f"'add' mode prediction shape should be ({B}, {D})"
        assert not torch.isnan(pred).any(), \
            "'add' mode prediction must not contain NaN"

    def test_concat_mode_same_shapes_as_add(self):
        m_concat = _make_model(fine_input_mode="concat")
        m_add = _make_model(fine_input_mode="add")
        q, s, c = _make_inputs()
        pred_c, _ = m_concat(q, s, c)
        pred_a, _ = m_add(q, s, c)
        assert pred_c.shape == pred_a.shape, \
            "'concat' and 'add' modes should produce predictions of the same shape"


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
        assert c.grad is not None, \
            "Gradient should flow back to centroid_embs when requires_grad=True"
        assert c.grad.abs().sum() > 0, \
            "centroid_embs gradient should be non-zero"


# ---------------------------------------------------------------------------
# TestLatentRepresentations
# ---------------------------------------------------------------------------


class TestLatents:
    def test_cached_after_forward(self):
        model = _make_model()
        q, s, c = _make_inputs()
        model(q, s, c)
        latents = model.get_latent_representations()
        assert "question_latent" in latents, \
            "Latents should contain 'question_latent' after forward pass"
        assert "steering_latent" in latents, \
            "Latents should contain 'steering_latent' after forward pass"
        assert "routing_weights" in latents, \
            "Latents should contain 'routing_weights' after forward pass"

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
            assert val.device.type == "cpu", \
                "All cached latents should reside on CPU (no device transfer overhead)"

    def test_get_routing_weights(self):
        model = _make_model()
        q, s, c = _make_inputs()
        # get_routing_weights does not need centroid_embs
        weights = model.get_routing_weights(q, s)
        assert weights.shape == (B, K), \
            f"get_routing_weights output shape should be ({B}, {K})"
