"""Tests for topic distance calculator utility."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_topic_distance_calculator_basic():
    """Test basic functionality of TopicDistanceCalculator."""
    from RAG_supporters.clustering.topic_distance_calculator import TopicDistanceCalculator

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
    from RAG_supporters.clustering.topic_distance_calculator import calculate_topic_distances_from_csv

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

        # Verify output - check all 6 output columns
        assert len(result_df) == 2
        assert "question_topic_distances" in result_df.columns
        assert "source_topic_distances" in result_df.columns
        assert "question_closest_topic" in result_df.columns
        assert "source_closest_topic" in result_df.columns
        assert "question_closest_topic_keywords" in result_df.columns
        assert "source_closest_topic_keywords" in result_df.columns
        assert output_path.exists()


def test_compute_distances():
    """Test distance computation."""
    from RAG_supporters.clustering.topic_distance_calculator import TopicDistanceCalculator

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


def test_invalid_metric():
    """Test that invalid metric raises ValueError."""
    from RAG_supporters.clustering.topic_distance_calculator import TopicDistanceCalculator

    mock_clusterer_data = {
        "centroids": [[1.0, 0.0, 0.0]],
        "cluster_stats": {},
    }

    with pytest.raises(ValueError, match="Invalid metric"):
        TopicDistanceCalculator(
            keyword_clusterer_json=mock_clusterer_data,
            metric="invalid_metric",
        )


def test_missing_centroids():
    """Test that missing centroids in JSON raises ValueError."""
    from RAG_supporters.clustering.topic_distance_calculator import TopicDistanceCalculator

    mock_clusterer_data = {
        "cluster_stats": {},
        # Missing 'centroids' key
    }

    with pytest.raises(ValueError, match="missing 'centroids'"):
        TopicDistanceCalculator(
            keyword_clusterer_json=mock_clusterer_data,
            metric="cosine",
        )


def test_missing_required_columns():
    """Test that missing required columns in CSV raises ValueError."""
    from RAG_supporters.clustering.topic_distance_calculator import calculate_topic_distances_from_csv

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create CSV with missing columns
        csv_path = tmpdir / "test_data.csv"
        df = pd.DataFrame({
            "wrong_column": ["some text"],
        })
        df.to_csv(csv_path, index=False)

        # Create mock KeywordClusterer JSON
        clusterer_path = tmpdir / "clusters.json"
        mock_clusterer_data = {
            "centroids": [np.random.rand(384).tolist()],
            "cluster_stats": {},
        }
        with open(clusterer_path, "w") as f:
            json.dump(mock_clusterer_data, f)

        # Create mock embedder
        class MockEmbedder:
            def create_embeddings(self, texts):
                return {text: np.random.rand(384) for text in texts}

        embedder = MockEmbedder()

        with pytest.raises(ValueError, match="Missing required columns"):
            calculate_topic_distances_from_csv(
                csv_path=csv_path,
                keyword_clusterer_json=clusterer_path,
                embedder=embedder,
                show_progress=False,
            )


def test_embedder_none_when_needed():
    """Test that missing embedder raises ValueError when text embedding is needed."""
    from RAG_supporters.clustering.topic_distance_calculator import calculate_topic_distances_from_csv

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create CSV with text columns
        csv_path = tmpdir / "test_data.csv"
        df = pd.DataFrame({
            "question_text": ["What is ML?"],
            "source_text": ["ML is AI"],
        })
        df.to_csv(csv_path, index=False)

        # Create mock KeywordClusterer JSON
        clusterer_path = tmpdir / "clusters.json"
        mock_clusterer_data = {
            "centroids": [np.random.rand(384).tolist()],
            "cluster_stats": {},
        }
        with open(clusterer_path, "w") as f:
            json.dump(mock_clusterer_data, f)

        # Don't provide embedder
        with pytest.raises(ValueError, match="Embedder is required"):
            result_df = calculate_topic_distances_from_csv(
                csv_path=csv_path,
                keyword_clusterer_json=clusterer_path,
                embedder=None,  # No embedder provided
                show_progress=False,
            )


def test_database_none_when_using_ids():
    """Test that missing database raises ValueError when using ID columns."""
    from RAG_supporters.clustering.topic_distance_calculator import calculate_topic_distances_from_csv

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create CSV with ID columns
        csv_path = tmpdir / "test_data.csv"
        df = pd.DataFrame({
            "question_id": ["q1"],
            "source_id": ["s1"],
            "question_text": ["What is ML?"],
            "source_text": ["ML is AI"],
        })
        df.to_csv(csv_path, index=False)

        # Create mock KeywordClusterer JSON
        clusterer_path = tmpdir / "clusters.json"
        mock_clusterer_data = {
            "centroids": [np.random.rand(384).tolist()],
            "cluster_stats": {},
        }
        with open(clusterer_path, "w") as f:
            json.dump(mock_clusterer_data, f)

        # Use ID columns but don't provide database
        with pytest.raises(ValueError, match="Database required"):
            result_df = calculate_topic_distances_from_csv(
                csv_path=csv_path,
                keyword_clusterer_json=clusterer_path,
                question_id_col="question_id",
                database=None,  # No database provided
                show_progress=False,
            )


def test_unsupported_embedder_interface():
    """Test that embedder with unsupported interface raises ValueError."""
    from RAG_supporters.clustering.topic_distance_calculator import TopicDistanceCalculator

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create mock KeywordClusterer JSON
        mock_clusterer_data = {
            "centroids": [np.random.rand(384).tolist()],
            "cluster_stats": {},
        }

        # Create embedder with no supported methods
        class UnsupportedEmbedder:
            def wrong_method(self, texts):
                return {}

        embedder = UnsupportedEmbedder()
        calculator = TopicDistanceCalculator(
            keyword_clusterer_json=mock_clusterer_data,
            embedder=embedder,
            metric="cosine",
        )

        # Should raise ValueError when trying to use unsupported embedder
        with pytest.raises(ValueError, match="Embedder must have"):
            calculator._get_embedding_from_text("test text")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
