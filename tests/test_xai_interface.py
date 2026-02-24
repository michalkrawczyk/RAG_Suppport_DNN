"""Tests for XAIInterface."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from RAG_supporters.nn.models.decomposed_predictor import (
    DecomposedJASPERPredictor,
    DecomposedJASPERConfig,
)
from RAG_supporters.nn.models.jasper_predictor import JASPERPredictor, JASPERPredictorConfig
from RAG_supporters.nn.inference.xai_interface import XAIInterface


# ---------------------------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------------------------

D, K = 32, 4
CLUSTER_NAMES = [f"cluster_{i}" for i in range(K)]


def _make_decomposed_interface(
    training_pairs=None,
) -> tuple[XAIInterface, DecomposedJASPERPredictor, torch.Tensor]:
    torch.manual_seed(0)
    model = DecomposedJASPERPredictor(
        DecomposedJASPERConfig(
            embedding_dim=D, hidden_dim=16, num_subspaces=K, num_layers=2, router_hidden_dim=16
        )
    )
    centroid_embs = torch.randn(K, D)
    iface = XAIInterface(model, centroid_embs, CLUSTER_NAMES, training_pairs=training_pairs)
    return iface, model, centroid_embs


def _make_jasper_interface() -> tuple[XAIInterface, JASPERPredictor, torch.Tensor]:
    torch.manual_seed(1)
    model = JASPERPredictor(JASPERPredictorConfig(embedding_dim=D, hidden_dim=16, num_layers=2))
    centroid_embs = torch.randn(K, D)
    iface = XAIInterface(model, centroid_embs, CLUSTER_NAMES)
    return iface, model, centroid_embs


def _make_query(seed: int = 42):
    torch.manual_seed(seed)
    return torch.randn(D), torch.randn(D)


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------


class TestInit:
    def test_is_decomposed_flag_true_for_decomposed(self):
        iface, _, _ = _make_decomposed_interface()
        assert (
            iface._is_decomposed is True
        ), "XAIInterface._is_decomposed should be True for DecomposedJASPERPredictor"

    def test_is_decomposed_flag_false_for_jasper(self):
        iface, _, _ = _make_jasper_interface()
        assert (
            iface._is_decomposed is False
        ), "XAIInterface._is_decomposed should be False for plain JASPERPredictor"

    def test_model_in_eval_mode(self):
        iface, model, _ = _make_decomposed_interface()
        assert (
            not model.training
        ), "XAIInterface should switch the model to eval mode on construction"

    def test_wrong_cluster_names_length_raises(self):
        torch.manual_seed(0)
        model = DecomposedJASPERPredictor(
            DecomposedJASPERConfig(embedding_dim=D, hidden_dim=16, num_subspaces=K, num_layers=2)
        )
        centroid_embs = torch.randn(K, D)
        with pytest.raises(ValueError, match="cluster_names"):
            XAIInterface(model, centroid_embs, ["only_one_name"])

    def test_training_pairs_stored(self):
        pairs = [(torch.randn(D), torch.randn(D), torch.randn(D)) for _ in range(5)]
        iface, _, _ = _make_decomposed_interface(training_pairs=pairs)
        assert (
            iface._train_questions is not None
        ), "_train_questions should be populated when training_pairs are provided"
        assert iface._train_questions.shape == (
            5,
            D,
        ), f"_train_questions should have shape (5, {D}) for 5 training pairs"

    def test_repr(self):
        iface, _, _ = _make_decomposed_interface()
        r = repr(iface)
        assert "XAIInterface" in r, "repr() should include the class name 'XAIInterface'"


# ---------------------------------------------------------------------------
# TestExplainPrediction
# ---------------------------------------------------------------------------


class TestExplainPrediction:
    REQUIRED_KEYS = (
        "prediction",
        "primary_subspace",
        "primary_confidence",
        "routing_distribution",
        "routing_entropy",
        "atypicality",
        "coarse_vector",
        "fine_vector",
        "steering_influence",
        "similar_pairs",
        "actionable_signal",
    )

    def test_all_required_keys_present(self):
        iface, _, _ = _make_decomposed_interface()
        q, s = _make_query()
        result = iface.explain_prediction(q, s)
        for key in self.REQUIRED_KEYS:
            assert key in result, f"Missing key: {key}"

    def test_primary_subspace_in_cluster_names(self):
        iface, _, _ = _make_decomposed_interface()
        q, s = _make_query()
        result = iface.explain_prediction(q, s)
        assert (
            result["primary_subspace"] in CLUSTER_NAMES
        ), f"primary_subspace '{result['primary_subspace']}' must be one of {CLUSTER_NAMES}"

    def test_primary_confidence_in_range(self):
        iface, _, _ = _make_decomposed_interface()
        q, s = _make_query()
        result = iface.explain_prediction(q, s)
        assert (
            0.0 <= result["primary_confidence"] <= 1.0
        ), f"primary_confidence={result['primary_confidence']} is outside [0, 1]"

    def test_routing_distribution_sums_to_one(self):
        iface, _, _ = _make_decomposed_interface()
        q, s = _make_query()
        result = iface.explain_prediction(q, s)
        total = sum(result["routing_distribution"].values())
        assert abs(total - 1.0) < 1e-5, f"Routing dist sums to {total}"

    def test_routing_distribution_has_all_clusters(self):
        iface, _, _ = _make_decomposed_interface()
        q, s = _make_query()
        result = iface.explain_prediction(q, s)
        assert set(result["routing_distribution"].keys()) == set(
            CLUSTER_NAMES
        ), "routing_distribution keys must match the provided cluster names"

    def test_atypicality_non_negative(self):
        iface, _, _ = _make_decomposed_interface()
        q, s = _make_query()
        result = iface.explain_prediction(q, s)
        assert (
            result["atypicality"] >= 0
        ), "Atypicality score should always be non-negative (it is a distance)"

    def test_prediction_is_list(self):
        iface, _, _ = _make_decomposed_interface()
        q, s = _make_query()
        result = iface.explain_prediction(q, s)
        assert isinstance(
            result["prediction"], list
        ), "Prediction should be returned as a Python list for JSON serialisability"
        assert len(result["prediction"]) == D, f"Prediction should have length {D} (embedding_dim)"

    def test_coarse_and_fine_vectors_present_for_decomposed(self):
        iface, _, _ = _make_decomposed_interface()
        q, s = _make_query()
        result = iface.explain_prediction(q, s)
        assert (
            result["coarse_vector"] is not None
        ), "coarse_vector should be present for DecomposedJASPERPredictor"
        assert (
            result["fine_vector"] is not None
        ), "fine_vector should be present for DecomposedJASPERPredictor"
        assert (
            len(result["coarse_vector"]) == D
        ), f"coarse_vector should have length {D} (embedding_dim)"

    def test_coarse_and_fine_none_for_jasper(self):
        iface, _, _ = _make_jasper_interface()
        q, s = _make_query()
        result = iface.explain_prediction(q, s)
        assert (
            result["coarse_vector"] is None
        ), "coarse_vector should be None for plain JASPERPredictor"
        assert result["fine_vector"] is None, "fine_vector should be None for plain JASPERPredictor"

    def test_works_with_1D_input(self):
        """Accepts [D] shaped (non-batched) inputs."""
        iface, _, _ = _make_decomposed_interface()
        q = torch.randn(D)
        s = torch.randn(D)
        result = iface.explain_prediction(q, s)
        assert "prediction" in result

    def test_works_with_2D_input(self):
        """Accepts [1, D] shaped inputs."""
        iface, _, _ = _make_decomposed_interface()
        q = torch.randn(1, D)
        s = torch.randn(1, D)
        result = iface.explain_prediction(q, s)
        assert "prediction" in result

    def test_works_with_jasper_predictor(self):
        iface, _, _ = _make_jasper_interface()
        q, s = _make_query()
        result = iface.explain_prediction(q, s)
        for key in self.REQUIRED_KEYS:
            assert key in result, f"Missing key: {key}"

    def test_actionable_signal_is_string(self):
        iface, _, _ = _make_decomposed_interface()
        q, s = _make_query()
        result = iface.explain_prediction(q, s)
        assert isinstance(
            result["actionable_signal"], str
        ), "actionable_signal should be a non-empty string"
        assert (
            len(result["actionable_signal"]) > 0
        ), "actionable_signal should not be an empty string"

    def test_similar_pairs_empty_without_training_pairs(self):
        iface, _, _ = _make_decomposed_interface(training_pairs=None)
        q, s = _make_query()
        result = iface.explain_prediction(q, s)
        assert (
            result["similar_pairs"] == []
        ), "similar_pairs should be empty list when no training_pairs were provided"

    def test_similar_pairs_with_training_data(self):
        torch.manual_seed(5)
        pairs = [(torch.randn(D), torch.randn(D), torch.randn(D)) for _ in range(10)]
        iface, _, _ = _make_decomposed_interface(training_pairs=pairs)
        q, s = _make_query()
        result = iface.explain_prediction(q, s)
        assert (
            len(result["similar_pairs"]) > 0
        ), "similar_pairs should be non-empty when training_pairs were provided"
        assert (
            "similarity" in result["similar_pairs"][0]
        ), "Each item in similar_pairs should contain a 'similarity' key"


# ---------------------------------------------------------------------------
# TestCompareSteeringInfluence
# ---------------------------------------------------------------------------


class TestCompareSteeringInfluence:
    def test_returns_correct_list_lengths(self):
        iface, _, _ = _make_decomposed_interface()
        q, _ = _make_query()
        N = 3
        torch.manual_seed(10)
        steerings = [torch.randn(D) for _ in range(N)]
        result = iface.compare_steering_influence(q, steerings)
        assert (
            len(result["labels"]) == N
        ), f"compare_steering_influence should return {N} labels for {N} steering vectors"
        assert (
            len(result["routing_matrices"]) == N
        ), f"compare_steering_influence should return {N} routing_matrices"
        assert (
            len(result["predictions"]) == N
        ), f"compare_steering_influence should return {N} predictions"
        assert (
            len(result["atypicalities"]) == N
        ), f"compare_steering_influence should return {N} atypicalities"
        assert (
            len(result["routing_kl_from_first"]) == N - 1
        ), f"routing_kl_from_first should have {N - 1} entries (comparison vs first steering)"

    def test_single_steering_has_empty_kl(self):
        iface, _, _ = _make_decomposed_interface()
        q, _ = _make_query()
        steerings = [torch.randn(D)]
        result = iface.compare_steering_influence(q, steerings)
        assert (
            result["routing_kl_from_first"] == []
        ), "routing_kl_from_first should be empty list when only one steering vector is given"

    def test_custom_labels(self):
        iface, _, _ = _make_decomposed_interface()
        q, _ = _make_query()
        steerings = [torch.randn(D), torch.randn(D)]
        labels = ["steer_A", "steer_B"]
        result = iface.compare_steering_influence(q, steerings, labels=labels)
        assert (
            result["labels"] == labels
        ), "Custom labels should be preserved in compare_steering_influence result"

    def test_routing_matrices_sum_to_one(self):
        iface, _, _ = _make_decomposed_interface()
        q, _ = _make_query()
        steerings = [torch.randn(D), torch.randn(D)]
        result = iface.compare_steering_influence(q, steerings)
        for rm in result["routing_matrices"]:
            assert (
                abs(sum(rm) - 1.0) < 1e-5
            ), f"Each routing_matrix should sum to 1.0 (got {sum(rm):.6f})"


# ---------------------------------------------------------------------------
# TestVisualization
# ---------------------------------------------------------------------------


class TestVisualization:
    def _make_xai_dict(self) -> dict:
        iface, _, _ = _make_decomposed_interface()
        q, s = _make_query()
        return iface.explain_prediction(q, s)

    def test_returns_figure_when_matplotlib_available(self):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        iface, _, _ = _make_decomposed_interface()
        xai_dict = self._make_xai_dict()
        fig = iface.visualize_explanation(xai_dict)
        assert fig is not None
        plt.close(fig)

    def test_returns_none_when_matplotlib_unavailable(self):
        import RAG_supporters.nn.inference.xai_interface as xai_module

        original = xai_module._MATPLOTLIB_AVAILABLE
        xai_module._MATPLOTLIB_AVAILABLE = False
        try:
            iface, _, _ = _make_decomposed_interface()
            xai_dict = self._make_xai_dict()
            result = iface.visualize_explanation(xai_dict)
            assert result is None
        finally:
            xai_module._MATPLOTLIB_AVAILABLE = original

    def test_saves_to_file_when_save_path_given(self, tmp_path):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        iface, _, _ = _make_decomposed_interface()
        xai_dict = self._make_xai_dict()
        save_path = str(tmp_path / "xai_plot.png")
        fig = iface.visualize_explanation(xai_dict, save_path=save_path)
        if fig is not None:
            plt.close(fig)
            assert Path(
                save_path
            ).exists(), f"visualize_explanation should save plot to '{save_path}'"


# ---------------------------------------------------------------------------
# TestSaveXAIOutputs
# ---------------------------------------------------------------------------


class TestSaveXAIOutputs:
    def test_saves_valid_json(self, tmp_path):
        iface, _, _ = _make_decomposed_interface()
        q, s = _make_query()
        xai_results = [iface.explain_prediction(q, s)]

        out_path = str(tmp_path / "xai_outputs.json")
        iface.save_xai_outputs(xai_results, out_path)

        assert Path(out_path).exists(), f"save_xai_outputs should create file at '{out_path}'"
        with open(out_path) as f:
            loaded = json.load(f)
        assert isinstance(loaded, list), "save_xai_outputs should write a JSON array"
        assert len(loaded) == 1, "Saved JSON should contain exactly 1 item"

    def test_saved_json_has_required_keys(self, tmp_path):
        iface, _, _ = _make_decomposed_interface()
        q, s = _make_query()
        xai_results = [iface.explain_prediction(q, s)]
        out_path = str(tmp_path / "xai.json")
        iface.save_xai_outputs(xai_results, out_path)

        with open(out_path) as f:
            loaded = json.load(f)
        item = loaded[0]
        for key in ("prediction", "primary_subspace", "routing_distribution"):
            assert key in item, f"Missing key: {key}"

    def test_saves_multiple_results(self, tmp_path):
        iface, _, _ = _make_decomposed_interface()
        xai_results = []
        for i in range(5):
            q, s = _make_query(seed=i)
            xai_results.append(iface.explain_prediction(q, s))

        out_path = str(tmp_path / "xai_multi.json")
        iface.save_xai_outputs(xai_results, out_path)

        with open(out_path) as f:
            loaded = json.load(f)
        assert len(loaded) == 5, "Saved JSON should contain 5 items for 5 XAI results"
