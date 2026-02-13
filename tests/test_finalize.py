"""Tests for dataset finalization and config writing (Task 8)."""

import json

import pytest
import torch

from RAG_supporters.jasper import BuildConfig, DatasetFinalizer, finalize_dataset


def _write_valid_pt_dataset(output_dir, embedding_dim=8, n_neg=3):
    """Create a minimal valid PT dataset artifact set for finalizer tests."""
    n_questions = 3
    n_sources = 5
    n_keywords = 4
    n_clusters = 2
    n_pairs = 6

    question_embs = torch.randn(n_questions, embedding_dim)
    source_embs = torch.randn(n_sources, embedding_dim)
    keyword_embs = torch.randn(n_keywords, embedding_dim)
    centroid_embs = torch.randn(n_clusters, embedding_dim)

    pair_index = torch.tensor(
        [
            [0, 0],
            [0, 1],
            [1, 2],
            [1, 3],
            [2, 4],
            [2, 0],
        ],
        dtype=torch.long,
    )
    pair_cluster_id = torch.tensor([0, 0, 1, 1, 0, 1], dtype=torch.long)
    pair_relevance = torch.tensor([0.9, 0.8, 1.0, 0.7, 0.6, 0.95], dtype=torch.float32)
    pair_keyword_ids = [[0, 1], [1], [2], [2, 3], [], [0, 3]]

    steering_centroid = torch.randn(n_pairs, embedding_dim)
    steering_keyword_weighted = torch.randn(n_pairs, embedding_dim)
    steering_residual = torch.randn(n_pairs, embedding_dim)
    centroid_distances = torch.tensor([0.2, 0.5, 0.8, 1.1, 0.4, 0.9], dtype=torch.float32)

    hard_negatives = torch.tensor(
        [
            [2, 3, 4],
            [2, 3, 4],
            [0, 1, 4],
            [0, 1, 4],
            [1, 2, 3],
            [1, 2, 3],
        ],
        dtype=torch.long,
    )
    if n_neg != 3:
        hard_negatives = hard_negatives[:, :n_neg]

    negative_tiers = torch.tensor(
        [
            [1, 2, 4],
            [1, 3, 4],
            [1, 2, 3],
            [2, 3, 4],
            [1, 2, 4],
            [1, 3, 4],
        ],
        dtype=torch.long,
    )
    if n_neg != 3:
        negative_tiers = negative_tiers[:, :n_neg]

    train_idx = torch.tensor([0, 1, 2], dtype=torch.long)
    val_idx = torch.tensor([3], dtype=torch.long)
    test_idx = torch.tensor([4, 5], dtype=torch.long)

    torch.save(question_embs, output_dir / "question_embs.pt")
    torch.save(source_embs, output_dir / "source_embs.pt")
    torch.save(keyword_embs, output_dir / "keyword_embs.pt")
    torch.save(centroid_embs, output_dir / "centroid_embs.pt")

    torch.save(pair_index, output_dir / "pair_index.pt")
    torch.save(pair_cluster_id, output_dir / "pair_cluster_id.pt")
    torch.save(pair_relevance, output_dir / "pair_relevance.pt")
    torch.save(pair_keyword_ids, output_dir / "pair_keyword_ids.pt")

    torch.save(steering_centroid, output_dir / "steering_centroid.pt")
    torch.save(steering_keyword_weighted, output_dir / "steering_keyword_weighted.pt")
    torch.save(steering_residual, output_dir / "steering_residual.pt")
    torch.save(centroid_distances, output_dir / "centroid_distances.pt")

    torch.save(hard_negatives, output_dir / "hard_negatives.pt")
    torch.save(negative_tiers, output_dir / "negative_tiers.pt")

    torch.save(train_idx, output_dir / "train_idx.pt")
    torch.save(val_idx, output_dir / "val_idx.pt")
    torch.save(test_idx, output_dir / "test_idx.pt")

    return {
        "embedding_dim": embedding_dim,
        "n_neg": n_neg,
        "n_pairs": n_pairs,
        "n_questions": n_questions,
        "n_sources": n_sources,
        "n_keywords": n_keywords,
        "n_clusters": n_clusters,
    }


@pytest.fixture
def valid_output_dir(tmp_path):
    """Create a temp output directory with valid PT artifacts."""
    stats = _write_valid_pt_dataset(tmp_path)
    return tmp_path, stats


class TestDatasetFinalizerImport:
    """Import sanity checks."""

    def test_import(self):
        """Test that finalizer symbols are importable."""
        assert DatasetFinalizer is not None, "DatasetFinalizer should be importable"
        assert finalize_dataset is not None, "finalize_dataset should be importable"


class TestDatasetFinalizerInit:
    """Initialization tests."""

    def test_init_valid_path(self, valid_output_dir):
        """Test finalizer initialization on valid directory."""
        output_dir, _ = valid_output_dir
        finalizer = DatasetFinalizer(output_dir)

        assert finalizer.output_dir == output_dir, "Finalizer should store output_dir"

    def test_init_missing_path(self, tmp_path):
        """Test initialization fails for missing directory."""
        missing = tmp_path / "does-not-exist"
        with pytest.raises(FileNotFoundError, match="Output directory not found"):
            DatasetFinalizer(missing)


class TestDatasetFinalizerValidation:
    """Validation behavior tests."""

    def test_finalize_valid_with_config(self, valid_output_dir):
        """Test successful finalization with explicit config."""
        output_dir, stats = valid_output_dir

        config = BuildConfig(
            embedding_dim=stats["embedding_dim"],
            n_neg=stats["n_neg"],
            clustering_source="clusters.json",
            storage_format="pt",
        )

        finalizer = DatasetFinalizer(output_dir)
        final_config = finalizer.finalize(config=config, save=True)

        assert final_config.n_pairs == stats["n_pairs"], "Config should contain n_pairs"
        assert final_config.n_questions == stats["n_questions"], "Config should contain n_questions"
        assert final_config.n_sources == stats["n_sources"], "Config should contain n_sources"
        assert final_config.n_keywords == stats["n_keywords"], "Config should contain n_keywords"
        assert final_config.n_clusters == stats["n_clusters"], "Config should contain n_clusters"

        saved_config_path = output_dir / "config.json"
        assert saved_config_path.exists(), "Finalization should save config.json"

        with open(saved_config_path, "r", encoding="utf-8") as file:
            saved_data = json.load(file)

        assert saved_data["n_pairs"] == stats["n_pairs"], "Saved config should include n_pairs"
        assert (
            saved_data["embedding_dim"] == stats["embedding_dim"]
        ), "Saved config should include embedding_dim"

    def test_finalize_via_convenience_function(self, valid_output_dir):
        """Test convenience function finalization API."""
        output_dir, stats = valid_output_dir

        config = BuildConfig(
            embedding_dim=stats["embedding_dim"],
            n_neg=stats["n_neg"],
            clustering_source="clusters.json",
            storage_format="pt",
        )

        final_config = finalize_dataset(output_dir=output_dir, config=config, save=False)

        assert isinstance(final_config, BuildConfig), "Convenience API should return BuildConfig"
        assert final_config.n_pairs == stats["n_pairs"], "Convenience API should compute n_pairs"

    def test_finalize_missing_required_file(self, valid_output_dir):
        """Test error when one required artifact is missing."""
        output_dir, stats = valid_output_dir
        (output_dir / "hard_negatives.pt").unlink()

        config = BuildConfig(
            embedding_dim=stats["embedding_dim"],
            n_neg=stats["n_neg"],
            clustering_source="clusters.json",
            storage_format="pt",
        )

        finalizer = DatasetFinalizer(output_dir)
        with pytest.raises(FileNotFoundError, match="Missing required dataset files"):
            finalizer.finalize(config=config)

    def test_finalize_detects_pair_shape_mismatch(self, valid_output_dir):
        """Test error when pair_index has invalid shape."""
        output_dir, stats = valid_output_dir
        torch.save(torch.tensor([0, 1, 2], dtype=torch.long), output_dir / "pair_index.pt")

        config = BuildConfig(
            embedding_dim=stats["embedding_dim"],
            n_neg=stats["n_neg"],
            clustering_source="clusters.json",
            storage_format="pt",
        )

        finalizer = DatasetFinalizer(output_dir)
        with pytest.raises(ValueError, match="pair_index.*has 1 dimensions, expected 2 dimensions"):
            finalizer.finalize(config=config)

    def test_finalize_detects_true_source_in_negatives(self, valid_output_dir):
        """Test error when hard negatives include true source."""
        output_dir, stats = valid_output_dir

        hard_negatives = torch.load(output_dir / "hard_negatives.pt", weights_only=True)
        hard_negatives[0, 0] = 0
        torch.save(hard_negatives, output_dir / "hard_negatives.pt")

        config = BuildConfig(
            embedding_dim=stats["embedding_dim"],
            n_neg=stats["n_neg"],
            clustering_source="clusters.json",
            storage_format="pt",
        )

        finalizer = DatasetFinalizer(output_dir)
        with pytest.raises(ValueError, match="hard_negatives contains true source"):
            finalizer.finalize(config=config)

    def test_finalize_detects_split_overlap(self, valid_output_dir):
        """Test error when splits overlap."""
        output_dir, stats = valid_output_dir

        torch.save(torch.tensor([0, 1, 2], dtype=torch.long), output_dir / "train_idx.pt")
        torch.save(torch.tensor([2], dtype=torch.long), output_dir / "val_idx.pt")

        config = BuildConfig(
            embedding_dim=stats["embedding_dim"],
            n_neg=stats["n_neg"],
            clustering_source="clusters.json",
            storage_format="pt",
        )

        finalizer = DatasetFinalizer(output_dir)
        with pytest.raises(ValueError, match="overlap"):
            finalizer.finalize(config=config)

    def test_finalize_detects_embedding_dim_mismatch_with_config(self, valid_output_dir):
        """Test error when config embedding_dim differs from artifacts."""
        output_dir, stats = valid_output_dir

        config = BuildConfig(
            embedding_dim=stats["embedding_dim"] + 1,
            n_neg=stats["n_neg"],
            clustering_source="clusters.json",
            storage_format="pt",
        )

        finalizer = DatasetFinalizer(output_dir)
        with pytest.raises(ValueError, match="Config embedding_dim"):
            finalizer.finalize(config=config)

    def test_finalize_detects_n_neg_mismatch_with_config(self, valid_output_dir):
        """Test error when config n_neg differs from artifacts."""
        output_dir, stats = valid_output_dir

        config = BuildConfig(
            embedding_dim=stats["embedding_dim"],
            n_neg=stats["n_neg"] + 1,
            clustering_source="clusters.json",
            storage_format="pt",
        )

        finalizer = DatasetFinalizer(output_dir)
        with pytest.raises(ValueError, match="Config n_neg"):
            finalizer.finalize(config=config)
