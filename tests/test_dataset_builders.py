"""Tests for dataset builder components: BuildConfig, build_dataset, DatasetFinalizer."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from RAG_supporters.jasper import BuildConfig, DatasetFinalizer, build_dataset, finalize_dataset

# ---------------------------------------------------------------------------
# Shared helpers & fixtures
# ---------------------------------------------------------------------------


class MockEmbeddingModel:
    """Lightweight deterministic embedding model for tests."""

    def __init__(self, dim: int = 8):
        self.dim = dim
        self._model_name = "mock-build-model"

    def encode(
        self,
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
        **kwargs,
    ):
        """Return deterministic embeddings for provided text input."""
        if isinstance(texts, str):
            texts = [texts]

        vectors = []
        for text in texts:
            seed = sum(ord(char) for char in text) % (2**32)
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(self.dim).astype(np.float32)

            if normalize_embeddings:
                vec = vec / (np.linalg.norm(vec) + 1e-8)

            vectors.append(vec)

        return np.array(vectors)


@pytest.fixture
def cluster_json_file(tmp_path):
    """Create minimal KeywordClusterer-compatible JSON file (3 clusters, dim=8)."""
    cluster_payload = {
        "metadata": {
            "algorithm": "kmeans",
            "n_clusters": 3,
            "n_keywords": 9,
            "random_state": 42,
            "embedding_dim": 8,
            "assignment_config": {
                "default_mode": "hard",
                "default_threshold": 0.15,
                "default_metric": "cosine",
            },
        },
        "cluster_assignments": {
            "python": 0,
            "programming": 0,
            "code": 0,
            "sql": 1,
            "database": 1,
            "storage": 1,
            "biology": 2,
            "cell": 2,
            "gene": 2,
        },
        "clusters": {
            "0": ["python", "programming", "code"],
            "1": ["sql", "database", "storage"],
            "2": ["biology", "cell", "gene"],
        },
        "cluster_stats": {
            "0": {"size": 3, "topic_descriptors": ["programming"]},
            "1": {"size": 3, "topic_descriptors": ["database"]},
            "2": {"size": 3, "topic_descriptors": ["biology"]},
        },
        "centroids": [
            [0.10] * 8,
            [0.20] * 8,
            [0.30] * 8,
        ],
        "embeddings": {
            "python": [0.11] * 8,
            "programming": [0.12] * 8,
            "code": [0.13] * 8,
            "sql": [0.21] * 8,
            "database": [0.22] * 8,
            "storage": [0.23] * 8,
            "biology": [0.31] * 8,
            "cell": [0.32] * 8,
            "gene": [0.33] * 8,
        },
    }

    cluster_path = tmp_path / "clusters.json"
    with open(cluster_path, "w", encoding="utf-8") as file:
        json.dump(cluster_payload, file, indent=2)

    return cluster_path


@pytest.fixture
def csv_paths(tmp_path):
    """Create two CSV inputs with 9 unique pairs across 3 clusters."""
    import pandas as pd

    df_one = pd.DataFrame(
        {
            "question": [
                "What is Python?",
                "How to write code?",
                "What is programming?",
                "What is SQL?",
                "What is a database?",
            ],
            "source": [
                "Python is a language.",
                "Code is written in many languages.",
                "Programming solves problems.",
                "SQL queries relational databases.",
                "Databases store structured data.",
            ],
            "keywords": [
                "python, programming",
                "code, programming",
                "python, code",
                "sql, database",
                "database, storage",
            ],
            "relevance_score": [0.9, 0.8, 0.85, 0.95, 0.88],
        }
    )

    df_two = pd.DataFrame(
        {
            "question": [
                "What is storage?",
                "What is biology?",
                "What is a cell?",
                "What is a gene?",
            ],
            "source": [
                "Storage systems keep persistent data.",
                "Biology studies living organisms.",
                "Cells are units of life.",
                "Genes store hereditary information.",
            ],
            "keywords": [
                "sql, storage",
                "biology, cell",
                "cell, gene",
                "gene, biology",
            ],
            "relevance_score": [0.82, 0.91, 0.87, 0.93],
        }
    )

    csv_one = tmp_path / "part1.csv"
    csv_two = tmp_path / "part2.csv"
    df_one.to_csv(csv_one, index=False)
    df_two.to_csv(csv_two, index=False)

    return [csv_one, csv_two]


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
        [[0, 0], [0, 1], [1, 2], [1, 3], [2, 4], [2, 0]],
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
        [[2, 3, 4], [2, 3, 4], [0, 1, 4], [0, 1, 4], [1, 2, 3], [1, 2, 3]],
        dtype=torch.long,
    )
    if n_neg != 3:
        hard_negatives = hard_negatives[:, :n_neg]

    negative_tiers = torch.tensor(
        [[1, 2, 4], [1, 3, 4], [1, 2, 3], [2, 3, 4], [1, 2, 4], [1, 3, 4]],
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


# ===========================================================================
# Section 1 — BuildConfig
# ===========================================================================


class TestBuildConfigInit:
    """Test BuildConfig initialization and validation."""

    def test_valid_initialization(self):
        """Test BuildConfig initializes with valid parameters."""
        config = BuildConfig(
            embedding_dim=384,
            n_neg=12,
            clustering_source="clusters.json",
            split_ratios=[0.8, 0.1, 0.1],
            steering_probabilities={
                "zero": 0.25,
                "centroid": 0.25,
                "keyword": 0.25,
                "residual": 0.25,
            },
            curriculum={
                "mode": "linear",
                "start_distance": 0.3,
                "end_distance": 0.7,
                "warmup_epochs": 10,
            },
        )

        assert config.embedding_dim == 384, "Embedding dimension should be set"
        assert config.n_neg == 12, "Number of negatives should be set"
        assert config.clustering_source == "clusters.json", "Clustering source should be set"
        assert config.storage_format == "pt", "Default storage format should be 'pt'"
        assert config.random_seed == 42, "Default random seed should be 42"

    def test_default_values(self):
        """Test BuildConfig uses correct default values."""
        config = BuildConfig(embedding_dim=384, n_neg=12, clustering_source="clusters.json")

        assert config.split_ratios == [
            0.8,
            0.1,
            0.1,
        ], "Default split ratios should be [0.8, 0.1, 0.1]"
        assert len(config.steering_probabilities) == 4, "Should have 4 steering probability keys"
        assert sum(config.steering_probabilities.values()) == pytest.approx(
            1.0
        ), "Steering probabilities should sum to 1.0"
        assert config.curriculum["mode"] == "linear", "Default curriculum mode should be 'linear'"
        assert (
            config.include_inspection_file is False
        ), "Default inspection file flag should be False"

    def test_invalid_split_ratios_sum(self):
        """Test BuildConfig raises error when split ratios don't sum to 1.0."""
        with pytest.raises(ValueError, match="split_ratios must sum to 1.0"):
            BuildConfig(
                embedding_dim=384,
                n_neg=12,
                clustering_source="clusters.json",
                split_ratios=[0.7, 0.2, 0.2],
            )

    def test_invalid_split_ratios_count(self):
        """Test BuildConfig raises error when split ratios don't have 3 values."""
        with pytest.raises(ValueError, match="split_ratios must have exactly 3 values"):
            BuildConfig(
                embedding_dim=384,
                n_neg=12,
                clustering_source="clusters.json",
                split_ratios=[0.8, 0.2],
            )

    def test_invalid_split_ratios_range(self):
        """Test BuildConfig raises error when split ratios are out of range."""
        with pytest.raises(ValueError, match="split_ratios must sum to 1.0"):
            BuildConfig(
                embedding_dim=384,
                n_neg=12,
                clustering_source="clusters.json",
                split_ratios=[0.8, 0.1, -0.1],
            )

    def test_invalid_steering_probabilities_sum(self):
        """Test BuildConfig raises error when steering probabilities don't sum to 1.0."""
        with pytest.raises(ValueError, match="steering_probabilities must sum to 1.0"):
            BuildConfig(
                embedding_dim=384,
                n_neg=12,
                clustering_source="clusters.json",
                steering_probabilities={
                    "zero": 0.3,
                    "centroid": 0.3,
                    "keyword": 0.3,
                    "residual": 0.3,
                },
            )

    def test_invalid_steering_probabilities_keys(self):
        """Test BuildConfig raises error when steering probabilities have wrong keys."""
        with pytest.raises(ValueError, match="steering_probabilities must have keys"):
            BuildConfig(
                embedding_dim=384,
                n_neg=12,
                clustering_source="clusters.json",
                steering_probabilities={
                    "zero": 0.25,
                    "centroid": 0.25,
                    "keyword": 0.25,
                    "wrong_key": 0.25,
                },
            )

    def test_invalid_storage_format(self):
        """Test BuildConfig raises error for invalid storage format."""
        with pytest.raises(ValueError, match="storage_format must be 'pt' or 'hdf5'"):
            BuildConfig(
                embedding_dim=384,
                n_neg=12,
                clustering_source="clusters.json",
                storage_format="invalid",
            )

    def test_invalid_curriculum_mode(self):
        """Test BuildConfig raises error for invalid curriculum mode."""
        with pytest.raises(ValueError, match="curriculum mode must be"):
            BuildConfig(
                embedding_dim=384,
                n_neg=12,
                clustering_source="clusters.json",
                curriculum={"mode": "invalid"},
            )

    def test_missing_curriculum_mode(self):
        """Test BuildConfig raises error when curriculum mode is missing."""
        with pytest.raises(ValueError, match="curriculum must have 'mode' key"):
            BuildConfig(
                embedding_dim=384, n_neg=12, clustering_source="clusters.json", curriculum={}
            )

    def test_invalid_embedding_dim(self):
        """Test BuildConfig raises error for non-positive embedding dimension."""
        with pytest.raises(ValueError, match="embedding_dim must be positive"):
            BuildConfig(embedding_dim=0, n_neg=12, clustering_source="clusters.json")

    def test_invalid_n_neg(self):
        """Test BuildConfig raises error for non-positive n_neg."""
        with pytest.raises(ValueError, match="n_neg must be positive"):
            BuildConfig(embedding_dim=384, n_neg=0, clustering_source="clusters.json")


class TestBuildConfigSerialization:
    """Test BuildConfig save/load functionality."""

    def test_save_and_load(self):
        """Test BuildConfig can be saved and loaded correctly."""
        config = BuildConfig(
            embedding_dim=384, n_neg=12, clustering_source="clusters.json", random_seed=123
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"

            config.save(config_path)
            assert config_path.exists(), "Config file should be created"

            loaded = BuildConfig.load(config_path)

            assert loaded.embedding_dim == config.embedding_dim, "Embedding dimension should match"
            assert loaded.n_neg == config.n_neg, "Number of negatives should match"
            assert (
                loaded.clustering_source == config.clustering_source
            ), "Clustering source should match"
            assert loaded.random_seed == config.random_seed, "Random seed should match"
            assert loaded.split_ratios == config.split_ratios, "Split ratios should match"

    def test_save_creates_directories(self):
        """Test save() creates parent directories if they don't exist."""
        config = BuildConfig(embedding_dim=384, n_neg=12, clustering_source="clusters.json")

        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "a" / "b" / "c" / "config.json"

            config.save(nested_path)
            assert nested_path.exists(), "Config file should be created with nested directories"

    def test_load_nonexistent_file(self):
        """Test load() raises error for non-existent file."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            BuildConfig.load("nonexistent.json")

    def test_to_dict(self):
        """Test to_dict() returns correct dictionary representation."""
        config = BuildConfig(embedding_dim=384, n_neg=12, clustering_source="clusters.json")

        config_dict = config.to_dict()

        assert isinstance(config_dict, dict), "Should return dictionary"
        assert config_dict["embedding_dim"] == 384, "Dictionary should contain embedding_dim"
        assert config_dict["n_neg"] == 12, "Dictionary should contain n_neg"
        assert (
            config_dict["clustering_source"] == "clusters.json"
        ), "Dictionary should contain clustering_source"

    def test_json_format(self):
        """Test saved JSON has correct format and indentation."""
        config = BuildConfig(embedding_dim=384, n_neg=12, clustering_source="clusters.json")

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config.save(config_path)

            with open(config_path, "r") as f:
                content = f.read()

            parsed = json.loads(content)
            assert isinstance(parsed, dict), "Saved file should be valid JSON dictionary"
            assert "\n" in content, "JSON should be pretty-printed with newlines"


class TestBuildConfigUpdate:
    """Test BuildConfig update_post_build functionality."""

    def test_update_post_build(self):
        """Test update_post_build() sets computed statistics correctly."""
        config = BuildConfig(embedding_dim=384, n_neg=12, clustering_source="clusters.json")

        assert config.n_pairs is None, "n_pairs should initially be None"
        assert config.n_questions is None, "n_questions should initially be None"

        config.update_post_build(
            n_pairs=10000, n_questions=5000, n_sources=8000, n_keywords=200, n_clusters=20
        )

        assert config.n_pairs == 10000, "n_pairs should be updated"
        assert config.n_questions == 5000, "n_questions should be updated"
        assert config.n_sources == 8000, "n_sources should be updated"
        assert config.n_keywords == 200, "n_keywords should be updated"
        assert config.n_clusters == 20, "n_clusters should be updated"

    def test_update_persists_after_save(self):
        """Test updated values persist after save/load cycle."""
        config = BuildConfig(embedding_dim=384, n_neg=12, clustering_source="clusters.json")

        config.update_post_build(
            n_pairs=10000, n_questions=5000, n_sources=8000, n_keywords=200, n_clusters=20
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"

            config.save(config_path)
            loaded = BuildConfig.load(config_path)

            assert loaded.n_pairs == 10000, "n_pairs should persist after save/load"
            assert loaded.n_questions == 5000, "n_questions should persist after save/load"
            assert loaded.n_sources == 8000, "n_sources should persist after save/load"


class TestBuildConfigEdgeCases:
    """Test BuildConfig edge cases and boundary conditions."""

    def test_zero_split_ratio_allowed(self):
        """Test BuildConfig allows zero split ratio (e.g., no test set)."""
        config = BuildConfig(
            embedding_dim=384,
            n_neg=12,
            clustering_source="clusters.json",
            split_ratios=[0.9, 0.1, 0.0],
        )

        assert config.split_ratios[2] == 0.0, "Zero split ratio should be allowed"

    def test_hdf5_storage_format(self):
        """Test BuildConfig accepts 'hdf5' storage format."""
        config = BuildConfig(
            embedding_dim=384, n_neg=12, clustering_source="clusters.json", storage_format="hdf5"
        )

        assert config.storage_format == "hdf5", "Should accept 'hdf5' storage format"

    def test_different_curriculum_modes(self):
        """Test BuildConfig accepts all valid curriculum modes."""
        for mode in ["fixed", "linear", "cosine"]:
            config = BuildConfig(
                embedding_dim=384,
                n_neg=12,
                clustering_source="clusters.json",
                curriculum={"mode": mode},
            )

            assert config.curriculum["mode"] == mode, f"Should accept '{mode}' curriculum mode"

    def test_large_embedding_dim(self):
        """Test BuildConfig accepts large embedding dimensions."""
        config = BuildConfig(embedding_dim=4096, n_neg=12, clustering_source="clusters.json")

        assert config.embedding_dim == 4096, "Should accept large embedding dimensions"

    def test_many_negatives(self):
        """Test BuildConfig accepts large number of negatives."""
        config = BuildConfig(embedding_dim=384, n_neg=100, clustering_source="clusters.json")

        assert config.n_neg == 100, "Should accept large number of negatives"


# ===========================================================================
# Section 2 — build_dataset orchestrator
# ===========================================================================


class TestDatasetBuildOrchestrator:
    """Task 9 orchestrator coverage."""

    def test_build_dataset_end_to_end(self, tmp_path, csv_paths, cluster_json_file):
        """Should run Tasks 1-8 and persist final PT dataset artifacts."""
        output_dir = tmp_path / "dataset_out"
        model = MockEmbeddingModel(dim=8)

        config = {
            "n_neg": 2,
            "split_ratios": [0.6, 0.2, 0.2],
            "include_inspection_file": True,
            "random_seed": 123,
        }

        final_config = build_dataset(
            csv_paths=csv_paths,
            cluster_json_path=cluster_json_file,
            embedding_model=model,
            output_dir=output_dir,
            config=config,
            show_progress=False,
            embedding_batch_size=4,
        )

        assert isinstance(final_config, BuildConfig), "build_dataset should return BuildConfig"
        assert final_config.n_pairs == 9, "Final config should report 9 merged pairs"
        assert final_config.n_questions == 9, "Final config should report 9 unique questions"
        assert final_config.n_sources == 9, "Final config should report 9 unique sources"
        assert final_config.n_clusters == 3, "Final config should report 3 clusters"
        assert (
            final_config.embedding_dim == 8
        ), "Final config embedding_dim should match model output"
        assert final_config.n_neg == 2, "Final config n_neg should match requested value"

        required_files = [
            "config.json",
            "inspection.json",
            "question_embs.pt",
            "source_embs.pt",
            "keyword_embs.pt",
            "centroid_embs.pt",
            "pair_index.pt",
            "pair_cluster_id.pt",
            "pair_relevance.pt",
            "pair_keyword_ids.pt",
            "steering_centroid.pt",
            "steering_keyword_weighted.pt",
            "steering_residual.pt",
            "centroid_distances.pt",
            "hard_negatives.pt",
            "negative_tiers.pt",
            "train_idx.pt",
            "val_idx.pt",
            "test_idx.pt",
        ]

        for file_name in required_files:
            assert (output_dir / file_name).exists(), f"Build should create {file_name}"

    def test_build_dataset_rejects_non_pt_storage(self, tmp_path, csv_paths, cluster_json_file):
        """Should raise clear error for unsupported storage format."""
        output_dir = tmp_path / "dataset_out_hdf5"
        model = MockEmbeddingModel(dim=8)

        with pytest.raises(NotImplementedError, match="storage_format='pt' only"):
            build_dataset(
                csv_paths=csv_paths,
                cluster_json_path=cluster_json_file,
                embedding_model=model,
                output_dir=output_dir,
                storage_format="hdf5",
                show_progress=False,
            )

    def test_build_dataset_smoke_with_buildconfig(self, tmp_path, csv_paths, cluster_json_file):
        """Should support BuildConfig input and produce a usable config.json in a smoke run."""
        output_dir = tmp_path / "dataset_out_smoke"
        model = MockEmbeddingModel(dim=8)

        build_config = BuildConfig(
            embedding_dim=1,
            n_neg=2,
            clustering_source=str(cluster_json_file),
            split_ratios=[0.7, 0.2, 0.1],
            storage_format="pt",
            include_inspection_file=False,
            random_seed=7,
        )

        final_config = build_dataset(
            csv_paths=csv_paths,
            cluster_json_path=cluster_json_file,
            embedding_model=model,
            output_dir=output_dir,
            config=build_config,
            show_progress=False,
            embedding_batch_size=3,
        )

        assert (
            final_config.embedding_dim == 8
        ), "Final config should be updated to actual embedding dimension"
        assert final_config.n_neg == 2, "Final config should preserve configured n_neg"
        assert final_config.n_pairs == 9, "Smoke run should produce expected pair count"
        assert (output_dir / "config.json").exists(), "Smoke run should persist config.json"


# ===========================================================================
# Section 3 — DatasetFinalizer
# ===========================================================================


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
