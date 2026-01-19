"""Tests for topic distance calculator utility."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_topic_distance_calculator_basic():
    """Test basic functionality of TopicDistanceCalculator."""
    from RAG_supporters.utils.topic_distance_calculator import TopicDistanceCalculator

    # Create mock KeywordClusterer data
    mock_clusterer_data = {
        "metadata": {
            "n_clusters": 3,
            "embedding_dim": 384,
        },
        "centroids": [
            np.random.rand(384).tolist(),
            np.random.rand(384).tolist(),
            np.random.rand(384).tolist(),
        ],
        "cluster_stats": {
            "0": {
                "topic_descriptors": ["machine learning", "AI", "neural networks"],
                "size": 10,
            },
            "1": {
                "topic_descriptors": ["database", "SQL", "storage"],
                "size": 8,
            },
            "2": {
                "topic_descriptors": ["web development", "frontend", "javascript"],
                "size": 12,
            },
        },
    }

    # Test initialization with dict
    calculator = TopicDistanceCalculator(
        keyword_clusterer_json=mock_clusterer_data,
        metric="cosine",
    )

    assert calculator.centroids.shape == (3, 384)
    assert len(calculator.topic_descriptors) == 3


def test_topic_distance_calculator_with_csv():
    """Test CSV processing with topic distance calculator."""
    from RAG_supporters.utils.topic_distance_calculator import calculate_topic_distances_from_csv

    # Create temporary files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create mock CSV
        csv_path = tmpdir / "test_data.csv"
        df = pd.DataFrame({
            "question_text": [
                "What is machine learning?",
                "How do databases work?",
            ],
            "source_text": [
                "Machine learning is a subset of AI",
                "Databases store and manage data",
            ],
        })
        df.to_csv(csv_path, index=False)

        # Create mock KeywordClusterer JSON
        clusterer_path = tmpdir / "clusters.json"
        mock_clusterer_data = {
            "metadata": {
                "n_clusters": 2,
                "embedding_dim": 384,
            },
            "centroids": [
                np.random.rand(384).tolist(),
                np.random.rand(384).tolist(),
            ],
            "cluster_stats": {
                "0": {
                    "topic_descriptors": ["machine learning", "AI"],
                    "size": 10,
                },
                "1": {
                    "topic_descriptors": ["database", "SQL"],
                    "size": 8,
                },
            },
        }
        with open(clusterer_path, "w") as f:
            json.dump(mock_clusterer_data, f)

        # Create mock embedder
        class MockEmbedder:
            def create_embeddings(self, texts):
                return {text: np.random.rand(384) for text in texts}

        embedder = MockEmbedder()

        # Test processing
        output_path = tmpdir / "results.csv"
        result_df = calculate_topic_distances_from_csv(
            csv_path=csv_path,
            keyword_clusterer_json=clusterer_path,
            embedder=embedder,
            output_path=output_path,
            show_progress=False,
        )

        # Verify output
        assert len(result_df) == 2
        assert "question_topic_distances" in result_df.columns
        assert "source_topic_distances" in result_df.columns
        assert "question_closest_topic" in result_df.columns
        assert "source_closest_topic" in result_df.columns
        assert output_path.exists()


def test_compute_distances():
    """Test distance computation."""
    from RAG_supporters.utils.topic_distance_calculator import TopicDistanceCalculator

    # Create mock data
    mock_clusterer_data = {
        "centroids": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "cluster_stats": {},
    }

    calculator = TopicDistanceCalculator(
        keyword_clusterer_json=mock_clusterer_data,
        metric="euclidean",
    )

    # Test embedding close to first centroid
    embedding = np.array([0.9, 0.1, 0.1])
    distances = calculator._compute_distances_to_centroids(embedding)

    assert len(distances) == 3
    assert distances[0] < distances[1]
    assert distances[0] < distances[2]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
