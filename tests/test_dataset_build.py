"""Tests for Task 9 dataset build orchestrator."""

import json

import numpy as np
import pandas as pd
import pytest

from RAG_supporters.jasper import build_dataset, BuildConfig


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
    """Create minimal KeywordClusterer-compatible JSON file."""
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
        assert final_config.embedding_dim == 8, "Final config embedding_dim should match model output"
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
        """Should raise clear error for unsupported storage format in Task 9."""
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

        assert final_config.embedding_dim == 8, "Final config should be updated to actual embedding dimension"
        assert final_config.n_neg == 2, "Final config should preserve configured n_neg"
        assert final_config.n_pairs == 9, "Smoke run should produce expected pair count"
        assert (output_dir / "config.json").exists(), "Smoke run should persist config.json"
