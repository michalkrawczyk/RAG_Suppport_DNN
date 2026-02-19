"""Tests for JASPERPredictor model."""

import math
import pytest
import torch
import torch.nn as nn

from RAG_supporters.nn.models.jasper_predictor import JASPERPredictor, JASPERPredictorConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config():
    return JASPERPredictorConfig(embedding_dim=64, hidden_dim=32, num_layers=2)


@pytest.fixture
def model(default_config):
    return JASPERPredictor(default_config)


@pytest.fixture
def batch():
    """Small deterministic batch."""
    torch.manual_seed(42)
    B, D = 4, 64
    return {
        "question_emb": torch.randn(B, D),
        "steering_emb": torch.randn(B, D),
    }


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestInit:
    def test_from_config_object(self, default_config):
        model = JASPERPredictor(default_config)
        assert model.config is default_config, \
            "JASPERPredictor.config should be the same object as the config passed in"

    def test_from_dict(self):
        model = JASPERPredictor({"embedding_dim": 32, "hidden_dim": 16, "num_layers": 1})
        assert model.embedding_dim == 32, \
            "embedding_dim property should reflect the value from the dict config"

    def test_invalid_num_layers_raises(self):
        with pytest.raises(ValueError, match="num_layers"):
            JASPERPredictorConfig(embedding_dim=64, hidden_dim=32, num_layers=0)

    def test_invalid_dropout_raises(self):
        with pytest.raises(ValueError, match="dropout"):
            JASPERPredictorConfig(embedding_dim=64, hidden_dim=32, num_layers=1, dropout=1.5)

    def test_invalid_activation_raises(self):
        with pytest.raises(ValueError, match="activation"):
            JASPERPredictorConfig(embedding_dim=64, hidden_dim=32, activation="NonExistent")

    def test_sub_networks_present(self, model):
        assert hasattr(model, "question_encoder"), \
            "JASPERPredictor must have a 'question_encoder' attribute"
        assert hasattr(model, "steering_encoder"), \
            "JASPERPredictor must have a 'steering_encoder' attribute"
        assert hasattr(model, "predictor_head"), \
            "JASPERPredictor must have a 'predictor_head' attribute"
        assert isinstance(model.question_encoder, nn.Sequential), \
            "question_encoder should be an nn.Sequential"
        assert isinstance(model.steering_encoder, nn.Sequential), \
            "steering_encoder should be an nn.Sequential"
        assert isinstance(model.predictor_head, nn.Sequential), \
            "predictor_head should be an nn.Sequential"

    def test_properties(self, default_config, model):
        assert model.embedding_dim == default_config.embedding_dim, \
            "model.embedding_dim should match config.embedding_dim"
        assert model.hidden_dim == default_config.hidden_dim, \
            "model.hidden_dim should match config.hidden_dim"

    def test_has_parameters(self, model):
        total = sum(p.numel() for p in model.parameters())
        assert total > 0, "JASPERPredictor should have trainable parameters"


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------


class TestForward:
    def test_output_shape(self, model, batch):
        out = model(batch["question_emb"], batch["steering_emb"])
        B, D = batch["question_emb"].shape
        assert out.shape == (B, D), \
            f"Expected output shape ({B}, {D}), got {out.shape}"

    def test_output_dtype_float32(self, model, batch):
        out = model(batch["question_emb"], batch["steering_emb"])
        assert out.dtype == torch.float32, \
            f"Expected float32 output, got {out.dtype}"

    def test_different_batch_sizes(self, model):
        D = 64
        for B in (1, 8, 32):
            q = torch.randn(B, D)
            s = torch.randn(B, D)
            out = model(q, s)
            assert out.shape == (B, D), f"Failed for B={B}"

    def test_no_nan_in_output(self, model, batch):
        out = model(batch["question_emb"], batch["steering_emb"])
        assert not torch.isnan(out).any(), "JASPERPredictor output must not contain NaN values"
        assert not torch.isinf(out).any(), "JASPERPredictor output must not contain Inf values"

    def test_normalize_output_flag(self):
        model = JASPERPredictor(
            JASPERPredictorConfig(embedding_dim=32, hidden_dim=16, num_layers=1, normalize_output=True)
        )
        q = torch.randn(4, 32)
        s = torch.randn(4, 32)
        out = model(q, s)
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
            "With normalize_output=True, all output vectors should have unit norm"

    def test_single_layer_model(self):
        model = JASPERPredictor(JASPERPredictorConfig(embedding_dim=16, hidden_dim=8, num_layers=1))
        q = torch.randn(2, 16)
        s = torch.randn(2, 16)
        out = model(q, s)
        assert out.shape == (2, 16), "Single-layer model should produce output of shape (2, 16)"

    def test_deep_model(self):
        model = JASPERPredictor(JASPERPredictorConfig(embedding_dim=32, hidden_dim=16, num_layers=5))
        q = torch.randn(3, 32)
        s = torch.randn(3, 32)
        out = model(q, s)
        assert out.shape == (3, 32), "Deep model should produce output of shape (3, 32)"


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


class TestGradients:
    def test_gradients_flow_to_all_params(self, model, batch):
        out = model(batch["question_emb"], batch["steering_emb"])
        loss = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_no_grad_context_produces_no_grad(self, model, batch):
        with torch.no_grad():
            out = model(batch["question_emb"], batch["steering_emb"])
        assert not out.requires_grad, \
            "Output inside torch.no_grad() context should not require grad"

    def test_grad_enabled_produces_grad(self, model, batch):
        q = batch["question_emb"].requires_grad_(True)
        s = batch["steering_emb"].requires_grad_(True)
        out = model(q, s)
        assert out.requires_grad, \
            "Output from grad-enabled inputs should require grad"


# ---------------------------------------------------------------------------
# Latent representations
# ---------------------------------------------------------------------------


class TestLatentRepresentations:
    def test_empty_before_forward(self, model):
        model._latents = {}
        latents = model.get_latent_representations()
        assert isinstance(latents, dict), \
            "get_latent_representations() should return a dict"
        assert len(latents) == 0, \
            "Latents should be empty when _latents is cleared before forward"

    def test_populated_after_forward(self, model, batch):
        model(batch["question_emb"], batch["steering_emb"])
        latents = model.get_latent_representations()
        assert "question_latent" in latents, \
            "Latents should contain 'question_latent' after forward pass"
        assert "steering_latent" in latents, \
            "Latents should contain 'steering_latent' after forward pass"

    def test_latent_shapes(self, model, batch):
        model(batch["question_emb"], batch["steering_emb"])
        latents = model.get_latent_representations()
        B = batch["question_emb"].shape[0]
        H = model.config.hidden_dim
        assert latents["question_latent"].shape == (B, H), \
            f"question_latent shape should be ({B}, {H})"
        assert latents["steering_latent"].shape == (B, H), \
            f"steering_latent shape should be ({B}, {H})"

    def test_latents_are_detached(self, model, batch):
        q = batch["question_emb"].requires_grad_(True)
        model(q, batch["steering_emb"])
        latents = model.get_latent_representations()
        for v in latents.values():
            assert not v.requires_grad, \
                "All cached latent representations must be detached from the computation graph"


# ---------------------------------------------------------------------------
# Config instantiation
# ---------------------------------------------------------------------------


class TestConfigInstantiation:
    def test_from_dict_with_extra_keys(self):
        """Extra keys should be silently ignored."""
        model = JASPERPredictor(
            {"embedding_dim": 32, "hidden_dim": 16, "num_layers": 1, "unknown_key": "ignored"}
        )
        assert model.embedding_dim == 32, \
            "Extra unknown keys in config dict should be silently ignored"

    def test_config_from_dict_classmethod(self):
        cfg = JASPERPredictorConfig.from_dict({"embedding_dim": 128, "hidden_dim": 64, "num_layers": 2})
        assert cfg.embedding_dim == 128, \
            "from_dict class method should correctly set embedding_dim"

    def test_summary_string(self, model):
        summary = model.get_model_summary()
        assert "JASPERPredictor" in summary, \
            "Model summary should contain 'JASPERPredictor'"
        assert "params" in summary, \
            "Model summary should contain parameter count info ('params')"

    def test_repr(self, model):
        r = repr(model)
        assert "JASPERPredictor" in r, "repr should contain 'JASPERPredictor'"
        assert str(model.config.embedding_dim) in r, \
            "repr should contain the embedding_dim value"
