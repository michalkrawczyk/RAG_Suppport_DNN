"""Tests for neural network model components.

Merged from:
- test_jasper_predictor.py   (JASPERPredictor, JASPERPredictorConfig)
- test_decomposed_predictor.py (DecomposedJASPERPredictor, DecomposedJASPERConfig)
- test_subspace_router.py    (SubspaceRouter, SubspaceRouterConfig)
- test_ema_encoder.py        (EMAEncoder)
"""

from __future__ import annotations

import copy
import math
from typing import List

import pytest
import torch
import torch.nn as nn

from RAG_supporters.nn.models.jasper_predictor import JASPERPredictor, JASPERPredictorConfig
from RAG_supporters.nn.models.decomposed_predictor import (
    DecomposedJASPERPredictor,
    DecomposedJASPERConfig,
)
from RAG_supporters.nn.models.subspace_router import SubspaceRouter, SubspaceRouterConfig
from RAG_supporters.nn.models.ema_encoder import EMAEncoder


# ===========================================================================
# JASPERPredictor
# ===========================================================================


@pytest.fixture
def predictor_config():
    return JASPERPredictorConfig(embedding_dim=64, hidden_dim=32, num_layers=2)


@pytest.fixture
def predictor(predictor_config):
    return JASPERPredictor(predictor_config)


@pytest.fixture
def predictor_batch():
    torch.manual_seed(42)
    B, D = 4, 64
    return {
        "question_emb": torch.randn(B, D),
        "steering_emb": torch.randn(B, D),
    }


class TestJASPERPredictorInit:
    def test_from_config_object(self, predictor_config):
        model = JASPERPredictor(predictor_config)
        assert (
            model.config is predictor_config
        ), "JASPERPredictor.config should be the same object as the config passed in"

    def test_from_dict(self):
        model = JASPERPredictor({"embedding_dim": 32, "hidden_dim": 16, "num_layers": 1})
        assert (
            model.embedding_dim == 32
        ), "embedding_dim property should reflect the value from the dict config"

    def test_invalid_num_layers_raises(self):
        with pytest.raises(ValueError, match="num_layers"):
            JASPERPredictorConfig(embedding_dim=64, hidden_dim=32, num_layers=0)

    def test_invalid_dropout_raises(self):
        with pytest.raises(ValueError, match="dropout"):
            JASPERPredictorConfig(embedding_dim=64, hidden_dim=32, num_layers=1, dropout=1.5)

    def test_invalid_activation_raises(self):
        with pytest.raises(ValueError, match="activation"):
            JASPERPredictorConfig(embedding_dim=64, hidden_dim=32, activation="NonExistent")

    def test_sub_networks_present(self, predictor):
        assert hasattr(
            predictor, "question_encoder"
        ), "JASPERPredictor must have a 'question_encoder' attribute"
        assert hasattr(
            predictor, "steering_encoder"
        ), "JASPERPredictor must have a 'steering_encoder' attribute"
        assert hasattr(
            predictor, "predictor_head"
        ), "JASPERPredictor must have a 'predictor_head' attribute"
        assert isinstance(
            predictor.question_encoder, nn.Sequential
        ), "question_encoder should be an nn.Sequential"
        assert isinstance(
            predictor.steering_encoder, nn.Sequential
        ), "steering_encoder should be an nn.Sequential"
        assert isinstance(
            predictor.predictor_head, nn.Sequential
        ), "predictor_head should be an nn.Sequential"

    def test_properties(self, predictor_config, predictor):
        assert (
            predictor.embedding_dim == predictor_config.embedding_dim
        ), "model.embedding_dim should match config.embedding_dim"
        assert (
            predictor.hidden_dim == predictor_config.hidden_dim
        ), "model.hidden_dim should match config.hidden_dim"

    def test_has_parameters(self, predictor):
        total = sum(p.numel() for p in predictor.parameters())
        assert total > 0, "JASPERPredictor should have trainable parameters"

    def test_from_dict_with_extra_keys(self):
        model = JASPERPredictor(
            {"embedding_dim": 32, "hidden_dim": 16, "num_layers": 1, "unknown_key": "ignored"}
        )
        assert (
            model.embedding_dim == 32
        ), "Extra unknown keys in config dict should be silently ignored"

    def test_config_from_dict_classmethod(self):
        cfg = JASPERPredictorConfig.from_dict(
            {"embedding_dim": 128, "hidden_dim": 64, "num_layers": 2}
        )
        assert cfg.embedding_dim == 128, "from_dict class method should correctly set embedding_dim"

    def test_summary_string(self, predictor):
        summary = predictor.get_model_summary()
        assert "JASPERPredictor" in summary, "Model summary should contain 'JASPERPredictor'"
        assert "params" in summary, "Model summary should contain parameter count info ('params')"

    def test_repr(self, predictor):
        r = repr(predictor)
        assert "JASPERPredictor" in r, "repr should contain 'JASPERPredictor'"
        assert (
            str(predictor.config.embedding_dim) in r
        ), "repr should contain the embedding_dim value"


class TestJASPERPredictorForward:
    def test_output_shape(self, predictor, predictor_batch):
        out = predictor(predictor_batch["question_emb"], predictor_batch["steering_emb"])
        B, D = predictor_batch["question_emb"].shape
        assert out.shape == (B, D), f"Expected output shape ({B}, {D}), got {out.shape}"

    def test_output_dtype_float32(self, predictor, predictor_batch):
        out = predictor(predictor_batch["question_emb"], predictor_batch["steering_emb"])
        assert out.dtype == torch.float32, f"Expected float32 output, got {out.dtype}"

    def test_different_batch_sizes(self, predictor):
        D = 64
        for B in (1, 8, 32):
            q = torch.randn(B, D)
            s = torch.randn(B, D)
            out = predictor(q, s)
            assert out.shape == (B, D), f"Failed for B={B}"

    def test_no_nan_in_output(self, predictor, predictor_batch):
        out = predictor(predictor_batch["question_emb"], predictor_batch["steering_emb"])
        assert not torch.isnan(out).any(), "JASPERPredictor output must not contain NaN values"
        assert not torch.isinf(out).any(), "JASPERPredictor output must not contain Inf values"

    def test_normalize_output_flag(self):
        model = JASPERPredictor(
            JASPERPredictorConfig(
                embedding_dim=32, hidden_dim=16, num_layers=1, normalize_output=True
            )
        )
        q = torch.randn(4, 32)
        s = torch.randn(4, 32)
        out = model(q, s)
        norms = out.norm(dim=-1)
        assert torch.allclose(
            norms, torch.ones_like(norms), atol=1e-5
        ), "With normalize_output=True, all output vectors should have unit norm"

    def test_single_layer_model(self):
        model = JASPERPredictor(JASPERPredictorConfig(embedding_dim=16, hidden_dim=8, num_layers=1))
        out = model(torch.randn(2, 16), torch.randn(2, 16))
        assert out.shape == (2, 16), "Single-layer model should produce output of shape (2, 16)"

    def test_deep_model(self):
        model = JASPERPredictor(
            JASPERPredictorConfig(embedding_dim=32, hidden_dim=16, num_layers=5)
        )
        out = model(torch.randn(3, 32), torch.randn(3, 32))
        assert out.shape == (3, 32), "Deep model should produce output of shape (3, 32)"


class TestJASPERPredictorGradients:
    def test_gradients_flow_to_all_params(self, predictor, predictor_batch):
        out = predictor(predictor_batch["question_emb"], predictor_batch["steering_emb"])
        out.sum().backward()
        for name, param in predictor.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_no_grad_context_produces_no_grad(self, predictor, predictor_batch):
        with torch.no_grad():
            out = predictor(predictor_batch["question_emb"], predictor_batch["steering_emb"])
        assert (
            not out.requires_grad
        ), "Output inside torch.no_grad() context should not require grad"

    def test_grad_enabled_produces_grad(self, predictor, predictor_batch):
        q = predictor_batch["question_emb"].requires_grad_(True)
        s = predictor_batch["steering_emb"].requires_grad_(True)
        out = predictor(q, s)
        assert out.requires_grad, "Output from grad-enabled inputs should require grad"


class TestJASPERPredictorLatents:
    def test_empty_before_forward(self, predictor):
        predictor._latents = {}
        latents = predictor.get_latent_representations()
        assert isinstance(latents, dict), "get_latent_representations() should return a dict"
        assert len(latents) == 0, "Latents should be empty when _latents is cleared before forward"

    def test_populated_after_forward(self, predictor, predictor_batch):
        predictor(predictor_batch["question_emb"], predictor_batch["steering_emb"])
        latents = predictor.get_latent_representations()
        assert (
            "question_latent" in latents
        ), "Latents should contain 'question_latent' after forward pass"
        assert (
            "steering_latent" in latents
        ), "Latents should contain 'steering_latent' after forward pass"

    def test_latent_shapes(self, predictor, predictor_batch):
        predictor(predictor_batch["question_emb"], predictor_batch["steering_emb"])
        latents = predictor.get_latent_representations()
        B = predictor_batch["question_emb"].shape[0]
        H = predictor.config.hidden_dim
        assert latents["question_latent"].shape == (
            B,
            H,
        ), f"question_latent shape should be ({B}, {H})"
        assert latents["steering_latent"].shape == (
            B,
            H,
        ), f"steering_latent shape should be ({B}, {H})"

    def test_latents_are_detached(self, predictor, predictor_batch):
        q = predictor_batch["question_emb"].requires_grad_(True)
        predictor(q, predictor_batch["steering_emb"])
        latents = predictor.get_latent_representations()
        for v in latents.values():
            assert (
                not v.requires_grad
            ), "All cached latent representations must be detached from the computation graph"


# ===========================================================================
# DecomposedJASPERPredictor
# ===========================================================================

_D_D, _H_D, _K_D = 32, 16, 4  # decomposed predictor constants
_B_D = 8


def _make_decomposed(fine_input_mode: str = "concat") -> DecomposedJASPERPredictor:
    return DecomposedJASPERPredictor(
        DecomposedJASPERConfig(
            embedding_dim=_D_D,
            hidden_dim=_H_D,
            num_subspaces=_K_D,
            num_layers=2,
            router_hidden_dim=_H_D,
            fine_input_mode=fine_input_mode,
        )
    )


def _make_decomposed_inputs(seed: int = 42):
    torch.manual_seed(seed)
    q = torch.randn(_B_D, _D_D)
    s = torch.randn(_B_D, _D_D)
    c = torch.randn(_K_D, _D_D)
    return q, s, c


class TestDecomposedPredictorInit:
    def test_sub_modules_present(self):
        model = _make_decomposed()
        for attr in ("question_encoder", "steering_encoder", "router", "fine_mlp"):
            assert hasattr(model, attr), f"DecomposedJASPERPredictor must have a '{attr}' attribute"

    def test_num_subspaces(self):
        model = _make_decomposed()
        assert model.num_subspaces == _K_D, f"num_subspaces property should be {_K_D}"

    def test_embedding_dim_property(self):
        model = _make_decomposed()
        assert model.embedding_dim == _D_D, f"embedding_dim property should be {_D_D}"

    def test_hidden_dim_property(self):
        model = _make_decomposed()
        assert model.hidden_dim == _H_D, f"hidden_dim property should be {_H_D}"

    def test_from_dict(self):
        model = DecomposedJASPERPredictor(
            {
                "embedding_dim": _D_D,
                "hidden_dim": _H_D,
                "num_subspaces": _K_D,
                "num_layers": 2,
                "router_hidden_dim": _H_D,
            }
        )
        assert model.num_subspaces == _K_D, "num_subspaces should be set correctly from dict config"

    def test_invalid_fine_input_mode_raises(self):
        with pytest.raises(ValueError, match="fine_input_mode"):
            DecomposedJASPERConfig(
                embedding_dim=_D_D, hidden_dim=_H_D, num_subspaces=_K_D, fine_input_mode="invalid"
            )

    def test_add_mode_has_coarse_projector(self):
        model = _make_decomposed(fine_input_mode="add")
        assert (
            model.coarse_projector is not None
        ), "'add' fine_input_mode should create a coarse_projector"

    def test_concat_mode_no_coarse_projector(self):
        model = _make_decomposed(fine_input_mode="concat")
        assert (
            model.coarse_projector is None
        ), "'concat' fine_input_mode should not create a coarse_projector"

    def test_get_model_summary(self):
        model = _make_decomposed()
        summary = model.get_model_summary()
        assert (
            "DecomposedJASPERPredictor" in summary
        ), "Model summary should contain 'DecomposedJASPERPredictor'"

    def test_repr(self):
        model = _make_decomposed()
        assert "DecomposedJASPERPredictor" in repr(
            model
        ), "repr should contain 'DecomposedJASPERPredictor'"


class TestDecomposedPredictorForward:
    def test_prediction_shape(self):
        model = _make_decomposed()
        q, s, c = _make_decomposed_inputs()
        pred, _ = model(q, s, c)
        assert pred.shape == (
            _B_D,
            _D_D,
        ), f"Prediction shape should be ({_B_D}, {_D_D}), got {pred.shape}"

    def test_explanation_dict_keys(self):
        model = _make_decomposed()
        q, s, c = _make_decomposed_inputs()
        _, xai = model(q, s, c)
        for key in ("routing_weights", "concept_logits", "coarse", "fine", "atypicality"):
            assert key in xai, f"Missing key: {key}"

    def test_explanation_shapes(self):
        model = _make_decomposed()
        q, s, c = _make_decomposed_inputs()
        _, xai = model(q, s, c)
        assert xai["routing_weights"].shape == (
            _B_D,
            _K_D,
        ), f"routing_weights shape should be ({_B_D}, {_K_D})"
        assert xai["coarse"].shape == (
            _B_D,
            _D_D,
        ), f"coarse vector shape should be ({_B_D}, {_D_D})"
        assert xai["fine"].shape == (_B_D, _D_D), f"fine vector shape should be ({_B_D}, {_D_D})"
        assert xai["atypicality"].shape == (_B_D,), f"atypicality shape should be ({_B_D},)"

    def test_explanation_all_detached(self):
        model = _make_decomposed()
        q, s, c = _make_decomposed_inputs()
        _, xai = model(q, s, c)
        for key, val in xai.items():
            assert not val.requires_grad, f"Tensor '{key}' should be detached"

    def test_no_nan_in_prediction(self):
        model = _make_decomposed()
        q, s, c = _make_decomposed_inputs()
        pred, xai = model(q, s, c)
        assert not torch.isnan(pred).any(), "Prediction must not contain NaN values"
        for key, val in xai.items():
            assert not torch.isnan(val).any(), f"NaN in '{key}'"

    def test_wrong_centroid_K_raises(self):
        model = _make_decomposed()
        q, s, _ = _make_decomposed_inputs()
        with pytest.raises(ValueError, match="centroid_embs"):
            model(q, s, torch.randn(_K_D + 1, _D_D))

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_different_batch_sizes(self, batch_size: int):
        model = _make_decomposed()
        torch.manual_seed(0)
        q = torch.randn(batch_size, _D_D)
        s = torch.randn(batch_size, _D_D)
        c = torch.randn(_K_D, _D_D)
        pred, xai = model(q, s, c)
        assert pred.shape == (batch_size, _D_D)
        assert xai["routing_weights"].shape == (batch_size, _K_D)

    def test_routing_weights_sum_to_one(self):
        model = _make_decomposed()
        q, s, c = _make_decomposed_inputs()
        _, xai = model(q, s, c)
        sums = xai["routing_weights"].sum(dim=-1)
        assert torch.allclose(
            sums, torch.ones(_B_D), atol=1e-5
        ), f"Routing weights should sum to 1.0 per sample, got: {sums}"


class TestDecomposedPredictorArithmetic:
    def test_prediction_equals_coarse_plus_fine(self):
        model = _make_decomposed()
        model.eval()
        q, s, c = _make_decomposed_inputs()
        with torch.no_grad():
            prediction, xai = model(q, s, c, training=False)
        reconstructed = xai["coarse"] + xai["fine"]
        assert torch.allclose(
            prediction.detach(), reconstructed, atol=1e-5
        ), f"Max diff: {(prediction.detach() - reconstructed).abs().max().item()}"

    def test_atypicality_equals_fine_norm(self):
        model = _make_decomposed()
        q, s, c = _make_decomposed_inputs()
        with torch.no_grad():
            _, xai = model(q, s, c, training=False)
        expected = xai["fine"].norm(dim=-1)
        assert torch.allclose(
            xai["atypicality"], expected, atol=1e-5
        ), "atypicality should equal the L2 norm of the fine residual vector"


class TestDecomposedPredictorFineModes:
    def test_add_mode_produces_valid_output(self):
        model = _make_decomposed(fine_input_mode="add")
        q, s, c = _make_decomposed_inputs()
        pred, xai = model(q, s, c)
        assert pred.shape == (_B_D, _D_D), f"'add' mode prediction shape should be ({_B_D}, {_D_D})"
        assert not torch.isnan(pred).any(), "'add' mode prediction must not contain NaN"

    def test_concat_mode_same_shapes_as_add(self):
        m_concat = _make_decomposed(fine_input_mode="concat")
        m_add = _make_decomposed(fine_input_mode="add")
        q, s, c = _make_decomposed_inputs()
        pred_c, _ = m_concat(q, s, c)
        pred_a, _ = m_add(q, s, c)
        assert (
            pred_c.shape == pred_a.shape
        ), "'concat' and 'add' modes should produce predictions of the same shape"


class TestDecomposedPredictorGradients:
    def test_all_params_receive_gradient(self):
        model = _make_decomposed()
        q, s, c = _make_decomposed_inputs()
        pred, _ = model(q, s, c)
        pred.sum().backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for param: {name}"

    def test_centroid_embs_receives_gradient(self):
        model = _make_decomposed()
        q, s, _ = _make_decomposed_inputs()
        c = torch.randn(_K_D, _D_D, requires_grad=True)
        pred, _ = model(q, s, c)
        pred.sum().backward()
        assert (
            c.grad is not None
        ), "Gradient should flow back to centroid_embs when requires_grad=True"


class TestDecomposedPredictorLatents:
    def test_cached_after_forward(self):
        model = _make_decomposed()
        q, s, c = _make_decomposed_inputs()
        model(q, s, c)
        latents = model.get_latent_representations()
        for key in ("question_latent", "steering_latent", "routing_weights"):
            assert key in latents, f"Latents should contain '{key}' after forward pass"

    def test_latents_are_detached(self):
        model = _make_decomposed()
        q, s, c = _make_decomposed_inputs()
        model(q, s, c)
        latents = model.get_latent_representations()
        for key, val in latents.items():
            assert not val.requires_grad, f"Latent '{key}' should not require grad"

    def test_get_routing_weights(self):
        model = _make_decomposed()
        q, s, _ = _make_decomposed_inputs()
        weights = model.get_routing_weights(q, s)
        assert weights.shape == (
            _B_D,
            _K_D,
        ), f"get_routing_weights output shape should be ({_B_D}, {_K_D})"


# ===========================================================================
# SubspaceRouter
# ===========================================================================

_D_R, _H_R, _K_R = 32, 16, 4
_B_R = 8


def _make_router(
    K: int = _K_R,
    temperature: float = 1.0,
    gumbel_hard: bool = False,
    normalize_input: bool = True,
) -> SubspaceRouter:
    return SubspaceRouter(
        SubspaceRouterConfig(
            embedding_dim=_D_R,
            hidden_dim=_H_R,
            num_subspaces=K,
            num_layers=2,
            temperature=temperature,
            gumbel_hard=gumbel_hard,
            normalize_input=normalize_input,
        )
    )


def _make_router_inputs(seed: int = 42):
    torch.manual_seed(seed)
    return torch.randn(_B_R, _D_R), torch.randn(_B_R, _D_R)


class TestSubspaceRouterInit:
    def test_from_config_object(self):
        cfg = SubspaceRouterConfig(embedding_dim=_D_R, hidden_dim=_H_R, num_subspaces=_K_R)
        router = SubspaceRouter(cfg)
        assert (
            router.config is cfg
        ), "SubspaceRouter.config should be the same object as the config passed in"

    def test_from_dict(self):
        router = SubspaceRouter({"embedding_dim": _D_R, "hidden_dim": _H_R, "num_subspaces": _K_R})
        assert router.num_subspaces == _K_R, "num_subspaces property should reflect the dict config"
        assert router.embedding_dim == _D_R, "embedding_dim property should reflect the dict config"

    def test_router_mlp_present(self):
        router = _make_router()
        assert hasattr(router, "router_mlp"), "SubspaceRouter must have a 'router_mlp' attribute"

    def test_invalid_temperature_raises(self):
        with pytest.raises(ValueError, match="temperature"):
            SubspaceRouterConfig(
                embedding_dim=_D_R, hidden_dim=_H_R, num_subspaces=_K_R, temperature=0.0
            )

    def test_invalid_num_subspaces_raises(self):
        with pytest.raises(ValueError, match="num_subspaces"):
            SubspaceRouterConfig(embedding_dim=_D_R, hidden_dim=_H_R, num_subspaces=1)

    def test_invalid_num_layers_raises(self):
        with pytest.raises(ValueError, match="num_layers"):
            SubspaceRouterConfig(
                embedding_dim=_D_R, hidden_dim=_H_R, num_subspaces=_K_R, num_layers=0
            )

    def test_invalid_activation_raises(self):
        with pytest.raises(ValueError, match="activation"):
            SubspaceRouterConfig(
                embedding_dim=_D_R, hidden_dim=_H_R, num_subspaces=_K_R, activation="NonExistent"
            )

    def test_get_model_summary(self):
        router = _make_router()
        summary = router.get_model_summary()
        assert "SubspaceRouter" in summary, "Model summary should contain 'SubspaceRouter'"
        assert str(_K_R) in summary, f"Model summary should contain num_subspaces={_K_R}"

    def test_repr(self):
        assert "SubspaceRouter" in repr(_make_router()), "repr should contain 'SubspaceRouter'"


class TestSubspaceRouterForward:
    def test_output_shapes(self):
        router = _make_router()
        q, s = _make_router_inputs()
        weights, logits = router(q, s, training=False)
        assert weights.shape == (_B_R, _K_R), f"Expected ({_B_R},{_K_R}), got {weights.shape}"
        assert logits.shape == (_B_R, _K_R), f"Expected ({_B_R},{_K_R}), got {logits.shape}"

    def test_routing_weights_sum_to_one(self):
        router = _make_router()
        q, s = _make_router_inputs()
        weights, _ = router(q, s, training=False)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(_B_R), atol=1e-5), f"Row sums: {sums}"

    def test_routing_weights_non_negative(self):
        router = _make_router()
        q, s = _make_router_inputs()
        weights, _ = router(q, s, training=False)
        assert (weights >= 0).all(), "All routing weights must be non-negative"

    def test_no_nan_in_output(self):
        router = _make_router()
        q, s = _make_router_inputs()
        weights, logits = router(q, s, training=False)
        assert not torch.isnan(weights).any(), "Routing weights must not contain NaN"
        assert not torch.isnan(logits).any(), "Routing logits must not contain NaN"

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_different_batch_sizes(self, batch_size: int):
        router = _make_router()
        torch.manual_seed(0)
        q = torch.randn(batch_size, _D_R)
        s = torch.randn(batch_size, _D_R)
        weights, logits = router(q, s, training=False)
        assert weights.shape == (batch_size, _K_R)
        assert logits.shape == (batch_size, _K_R)

    def test_training_weights_sum_to_one(self):
        router = _make_router()
        q, s = _make_router_inputs()
        weights, _ = router(q, s, training=True)
        sums = weights.sum(dim=-1)
        assert torch.allclose(
            sums, torch.ones(_B_R), atol=1e-5
        ), f"Routing weights should sum to 1.0 per sample in training mode, got: {sums}"


class TestSubspaceRouterGumbel:
    def test_training_mode_is_stochastic(self):
        router = _make_router()
        q, s = _make_router_inputs()
        torch.manual_seed(0)
        w1, _ = router(q, s, training=True)
        torch.manual_seed(99)
        w2, _ = router(q, s, training=True)
        assert not torch.allclose(w1, w2), "Training mode should produce stochastic outputs"

    def test_inference_mode_is_deterministic(self):
        router = _make_router()
        router.eval()
        q, s = _make_router_inputs()
        w1, _ = router(q, s, training=False)
        w2, _ = router(q, s, training=False)
        assert torch.allclose(
            w1, w2
        ), "Inference mode (training=False) should produce deterministic routing weights"

    def test_hard_gumbel_produces_one_hot(self):
        router = _make_router(gumbel_hard=True)
        q, s = _make_router_inputs()
        torch.manual_seed(7)
        weights, _ = router(q, s, training=True)
        max_vals, _ = weights.max(dim=-1)
        assert torch.allclose(max_vals, torch.ones(_B_R), atol=1e-5)
        sorted_weights, _ = weights.sort(dim=-1, descending=True)
        assert torch.allclose(sorted_weights[:, 1:], torch.zeros(_B_R, _K_R - 1), atol=1e-5)


class TestSubspaceRouterGetPrimarySubspace:
    def test_output_shapes(self):
        router = _make_router()
        q, s = _make_router_inputs()
        cluster_ids, confidences = router.get_primary_subspace(q, s)
        assert cluster_ids.shape == (_B_R,), f"cluster_ids shape should be ({_B_R},)"
        assert confidences.shape == (_B_R,), f"confidences shape should be ({_B_R},)"

    def test_cluster_ids_in_range(self):
        router = _make_router()
        q, s = _make_router_inputs()
        cluster_ids, _ = router.get_primary_subspace(q, s)
        assert (cluster_ids >= 0).all(), "All cluster IDs should be >= 0"
        assert (cluster_ids < _K_R).all(), f"All cluster IDs should be < num_subspaces={_K_R}"

    def test_confidence_in_range(self):
        router = _make_router()
        q, s = _make_router_inputs()
        _, confidences = router.get_primary_subspace(q, s)
        assert (confidences >= 0).all(), "All routing confidences should be >= 0"
        assert (confidences <= 1.0 + 1e-6).all(), "All routing confidences should be <= 1.0"

    def test_no_grad_on_output(self):
        router = _make_router()
        q, s = _make_router_inputs()
        _, confidences = router.get_primary_subspace(q, s)
        assert (
            not confidences.requires_grad
        ), "get_primary_subspace confidences should be detached (no grad)"


class TestSubspaceRouterExplain:
    def test_required_keys_present(self):
        router = _make_router()
        q, s = _make_router_inputs()
        names = [f"c{i}" for i in range(_K_R)]
        result = router.explain(q, s, names)
        for key in (
            "routing_weights",
            "primary_subspace",
            "primary_confidence",
            "entropy",
            "cluster_names",
        ):
            assert key in result, f"Missing key: {key}"

    def test_primary_subspace_in_names(self):
        router = _make_router()
        q, s = _make_router_inputs()
        names = [f"cluster_{i}" for i in range(_K_R)]
        result = router.explain(q, s, names)
        assert (
            result["primary_subspace"] in names
        ), "primary_subspace should be one of the provided cluster names"

    def test_wrong_cluster_names_length_raises(self):
        router = _make_router()
        q, s = _make_router_inputs()
        with pytest.raises(ValueError):
            router.explain(q, s, ["only_one"])

    def test_single_sample_input(self):
        router = _make_router()
        q_single = torch.randn(_D_R)
        s_single = torch.randn(_D_R)
        names = [f"c{i}" for i in range(_K_R)]
        result = router.explain(q_single, s_single, names)
        assert (
            result["primary_subspace"] in names
        ), "explain() with single-sample input should return a valid primary_subspace name"


class TestSubspaceRouterGradients:
    def test_gradients_flow_to_router_mlp(self):
        router = _make_router()
        q, s = _make_router_inputs()
        weights, _ = router(q, s, training=True)
        weights.sum().backward()
        any_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in router.router_mlp.parameters()
        )
        assert any_grad, "No gradient reached router_mlp parameters"


# ===========================================================================
# EMAEncoder
# ===========================================================================


@pytest.fixture
def small_encoder():
    torch.manual_seed(0)
    return nn.Linear(16, 16)


@pytest.fixture
def ema(small_encoder):
    return EMAEncoder(small_encoder, tau_min=0.9, tau_max=0.99)


@pytest.fixture
def ema_x():
    torch.manual_seed(1)
    return torch.randn(4, 16)


class TestEMAEncoderInit:
    def test_creates_target_copy(self, ema, small_encoder):
        assert (
            ema.target_encoder is not ema.online_encoder
        ), "Target encoder must be a separate object from online encoder"
        for op, tp in zip(ema.online_encoder.parameters(), ema.target_encoder.parameters()):
            assert torch.equal(
                op.data, tp.data
            ), "Target and online encoder should start with identical weights"

    def test_target_requires_no_grad(self, ema):
        for p in ema.target_encoder.parameters():
            assert not p.requires_grad, "Target encoder parameters must be frozen"

    def test_online_requires_grad(self, ema):
        for p in ema.online_encoder.parameters():
            assert p.requires_grad, "Online encoder parameters must be trainable"

    def test_invalid_tau_raises(self, small_encoder):
        with pytest.raises(ValueError):
            EMAEncoder(small_encoder, tau_min=0.99, tau_max=0.9)
        with pytest.raises(ValueError):
            EMAEncoder(small_encoder, tau_min=0.0, tau_max=0.9)
        with pytest.raises(ValueError):
            EMAEncoder(small_encoder, tau_min=0.9, tau_max=1.0)

    def test_invalid_schedule_raises(self, small_encoder):
        with pytest.raises(ValueError, match="Unsupported schedule"):
            EMAEncoder(small_encoder, schedule="linear")


class TestEMAEncoderTauSchedule:
    def test_tau_at_step_zero(self, ema):
        tau = ema.get_tau(step=0, max_steps=100)
        assert (
            abs(tau - ema.tau_min) < 1e-6
        ), f"tau at step 0 should equal tau_min={ema.tau_min}, got {tau}"

    def test_tau_at_last_step(self, ema):
        tau = ema.get_tau(step=100, max_steps=100)
        assert (
            abs(tau - ema.tau_max) < 1e-6
        ), f"tau at final step should equal tau_max={ema.tau_max}, got {tau}"

    def test_tau_monotonically_increasing(self, ema):
        taus = [ema.get_tau(s, 100) for s in range(0, 101, 10)]
        for i in range(len(taus) - 1):
            assert taus[i] <= taus[i + 1] + 1e-9, f"tau not increasing at step {i * 10}"

    def test_tau_within_bounds(self, ema):
        for step in range(0, 110, 10):
            tau = ema.get_tau(step, 100)
            assert (
                ema.tau_min - 1e-7 <= tau <= ema.tau_max + 1e-7
            ), f"tau={tau} at step={step} is outside [{ema.tau_min}, {ema.tau_max}]"

    def test_tau_clamps_beyond_max_steps(self, ema):
        tau = ema.get_tau(step=9999, max_steps=100)
        assert (
            abs(tau - ema.tau_max) < 1e-6
        ), f"tau beyond max_steps should clamp to tau_max={ema.tau_max}, got {tau}"

    def test_zero_max_steps_returns_tau_max(self, ema):
        tau = ema.get_tau(0, 0)
        assert tau == ema.tau_max, f"get_tau with max_steps=0 should return tau_max={ema.tau_max}"

    def test_tau_info_dict(self, ema):
        info = ema.get_tau_info(10, 100)
        assert "tau" in info, "get_tau_info should include 'tau' key"
        assert info["tau_min"] == ema.tau_min, f"tau_info['tau_min'] should be {ema.tau_min}"
        assert info["tau_max"] == ema.tau_max, f"tau_info['tau_max'] should be {ema.tau_max}"


class TestEMAEncoderUpdate:
    def test_target_params_change_after_update(self, ema):
        original_target = {n: p.data.clone() for n, p in ema.target_encoder.named_parameters()}
        with torch.no_grad():
            for p in ema.online_encoder.parameters():
                p.data += 0.1
        ema.update_target(step=0, max_steps=100)
        for name, param in ema.target_encoder.named_parameters():
            assert not torch.equal(
                param.data, original_target[name]
            ), f"Target param '{name}' should have changed after EMA update"

    def test_target_drifts_slower_than_online(self, small_encoder):
        ema = EMAEncoder(copy.deepcopy(small_encoder), tau_min=0.99, tau_max=0.999)
        initial_online = {n: p.data.clone() for n, p in ema.online_encoder.named_parameters()}
        initial_target = {n: p.data.clone() for n, p in ema.target_encoder.named_parameters()}
        with torch.no_grad():
            for p in ema.online_encoder.parameters():
                p.data += 1.0
        ema.update_target(0, 100)
        for name in initial_online:
            online_delta = (
                (ema.online_encoder.state_dict()[name] - initial_online[name]).abs().mean()
            )
            target_delta = (
                (ema.target_encoder.state_dict()[name] - initial_target[name]).abs().mean()
            )
            assert (
                target_delta.item() < online_delta.item()
            ), f"Target ({target_delta:.4f}) drifted more than online ({online_delta:.4f}) for {name}"

    def test_update_does_not_restore_grad_to_target(self, ema):
        ema.update_target(0, 100)
        for p in ema.target_encoder.parameters():
            assert not p.requires_grad, "EMA update must not re-enable gradients on target encoder"

    def test_multiple_updates_converge(self, small_encoder):
        ema = EMAEncoder(copy.deepcopy(small_encoder), tau_min=0.5, tau_max=0.5)
        with torch.no_grad():
            for p in ema.online_encoder.parameters():
                p.data.fill_(10.0)
        for step in range(200):
            ema.update_target(step, 200)
        for op, tp in zip(ema.online_encoder.parameters(), ema.target_encoder.parameters()):
            diff = (op.data - tp.data).abs().mean()
            assert diff.item() < 1.0, f"Target not converging: diff={diff:.4f}"


class TestEMAEncoderForward:
    def test_forward_uses_online_encoder(self, ema, ema_x):
        out_ema = ema(ema_x)
        out_online = ema.online_encoder(ema_x)
        assert torch.allclose(
            out_ema, out_online
        ), "EMAEncoder.forward() should route through the online encoder"

    def test_encode_target_uses_target_encoder(self, ema, ema_x):
        out = ema.encode_target(ema_x)
        expected = ema.target_encoder(ema_x)
        assert torch.allclose(
            out, expected
        ), "encode_target() output must match target_encoder(x) directly"

    def test_encode_target_returns_detached(self, ema, ema_x):
        out = ema.encode_target(ema_x)
        assert not out.requires_grad, "encode_target() must return a detached tensor"

    def test_online_forward_produces_grad(self, ema, ema_x):
        x_grad = ema_x.requires_grad_(True)
        out = ema(x_grad)
        assert out.requires_grad, "EMAEncoder.forward() should produce a grad-enabled output"

    def test_target_does_not_participate_in_backprop(self, ema, ema_x):
        ema.encode_target(ema_x)
        for p in ema.target_encoder.parameters():
            assert p.grad is None, "Target encoder parameters must not accumulate gradients"


class TestEMAEncoderStateDict:
    def test_state_dict_contains_tau_keys(self, ema):
        sd = ema.state_dict()
        assert "_ema_tau_min" in sd, "state_dict must contain '_ema_tau_min'"
        assert "_ema_tau_max" in sd, "state_dict must contain '_ema_tau_max'"
        assert "_ema_schedule" in sd, "state_dict must contain '_ema_schedule'"

    def test_state_dict_round_trip(self, small_encoder):
        ema1 = EMAEncoder(copy.deepcopy(small_encoder), tau_min=0.95, tau_max=0.98)
        with torch.no_grad():
            for p in ema1.online_encoder.parameters():
                p.data += 0.5
        ema1.update_target(10, 100)
        sd = ema1.state_dict()
        ema2 = EMAEncoder(copy.deepcopy(small_encoder), tau_min=0.0001, tau_max=0.9999)
        ema2.load_state_dict(sd)
        assert ema2.tau_min == ema1.tau_min, "Loaded tau_min should match saved tau_min"
        assert ema2.tau_max == ema1.tau_max, "Loaded tau_max should match saved tau_max"
        for p1, p2 in zip(ema1.online_encoder.parameters(), ema2.online_encoder.parameters()):
            assert torch.equal(p1.data, p2.data), "Loaded online weights should match saved"
        for p1, p2 in zip(ema1.target_encoder.parameters(), ema2.target_encoder.parameters()):
            assert torch.equal(p1.data, p2.data), "Loaded target weights should match saved"

    def test_load_restores_frozen_target(self, small_encoder, ema):
        sd = ema.state_dict()
        ema2 = EMAEncoder(copy.deepcopy(small_encoder))
        ema2.load_state_dict(sd)
        for p in ema2.target_encoder.parameters():
            assert not p.requires_grad, "load_state_dict must keep target encoder parameters frozen"

    def test_repr(self, ema):
        r = repr(ema)
        assert "EMAEncoder" in r, "repr should contain 'EMAEncoder'"
        assert str(ema.tau_min) in r, "repr should contain the tau_min value"
