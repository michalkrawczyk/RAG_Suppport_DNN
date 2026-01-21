"""Tests for topic distance calculator utility."""

import json
import logging
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

        # Verify output - check all 2 output columns
        assert len(result_df) == 2
        assert "question_term_distance_scores" in result_df.columns
        assert "source_term_distance_scores" in result_df.columns
        assert output_path.exists()
        
        # Verify JSON mapping format
        for idx in range(len(result_df)):
            if pd.notna(result_df.at[idx, "question_term_distance_scores"]):
                question_mapping = json.loads(result_df.at[idx, "question_term_distance_scores"])
                assert isinstance(question_mapping, dict)
                # Should have topic descriptors as keys
                assert all(isinstance(k, str) for k in question_mapping.keys())
                # Should have distance values
                assert all(isinstance(v, (int, float)) for v in question_mapping.values())
            
            if pd.notna(result_df.at[idx, "source_term_distance_scores"]):
                source_mapping = json.loads(result_df.at[idx, "source_term_distance_scores"])
                assert isinstance(source_mapping, dict)
                assert all(isinstance(k, str) for k in source_mapping.keys())
                assert all(isinstance(v, (int, float)) for v in source_mapping.values())


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

    # Create mock KeywordClusterer JSON
    mock_clusterer_data = {
        "centroids": [np.random.rand(384).tolist()],
        "cluster_stats": {},
    }

    # Create embedder with no supported methods (no encode or embed_documents)
    class UnsupportedEmbedder:
        def wrong_method(self, texts):
            return {}

    embedder = UnsupportedEmbedder()
    
    # Should raise ValueError when trying to wrap unsupported embedder
    with pytest.raises(ValueError, match="Unable to detect model type"):
        calculator = TopicDistanceCalculator(
            keyword_clusterer_json=mock_clusterer_data,
            embedder=embedder,
            metric="cosine",
        )


def test_create_distance_json_mapping():
    """Test creation of JSON mapping from distances to topic descriptors."""
    from RAG_supporters.clustering.topic_distance_calculator import TopicDistanceCalculator

    # Create mock data with topic descriptors
    mock_clusterer_data = {
        "centroids": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "cluster_stats": {
            "0": {
                "topic_descriptors": ["machine learning", "AI", "neural networks"],
                "size": 10,
            },
            "1": {
                "topic_descriptors": ["database", "SQL"],
                "size": 8,
            },
            "2": {
                "topic_descriptors": ["web development", "javascript"],
                "size": 12,
            },
        },
    }

    calculator = TopicDistanceCalculator(
        keyword_clusterer_json=mock_clusterer_data,
        metric="cosine",
    )

    # Test with mock distances
    distances = np.array([0.2, 0.5, 0.8])
    json_mapping_str = calculator._create_distance_json_mapping(distances)
    
    # Parse JSON
    mapping = json.loads(json_mapping_str)
    
    # Verify structure
    assert isinstance(mapping, dict)
    
    # Check that all topic descriptors are present
    expected_topics = {
        "machine learning", "AI", "neural networks",
        "database", "SQL",
        "web development", "javascript"
    }
    assert set(mapping.keys()) == expected_topics
    
    # Check that distances are correctly assigned
    # Cluster 0 topics should have distance 0.2
    assert mapping["machine learning"] == 0.2
    assert mapping["AI"] == 0.2
    assert mapping["neural networks"] == 0.2
    
    # Cluster 1 topics should have distance 0.5
    assert mapping["database"] == 0.5
    assert mapping["SQL"] == 0.5
    
    # Cluster 2 topics should have distance 0.8
    assert mapping["web development"] == 0.8
    assert mapping["javascript"] == 0.8


def test_create_distance_json_mapping_duplicate_topics():
    """Test JSON mapping when topic descriptor appears in multiple clusters (should use minimum)."""
    from RAG_supporters.clustering.topic_distance_calculator import TopicDistanceCalculator

    # Create mock data where "AI" appears in two clusters with different distances
    mock_clusterer_data = {
        "centroids": [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        "cluster_stats": {
            "0": {
                "topic_descriptors": ["machine learning", "AI"],
                "size": 10,
            },
            "1": {
                "topic_descriptors": ["AI", "deep learning"],  # "AI" appears again
                "size": 8,
            },
        },
    }

    calculator = TopicDistanceCalculator(
        keyword_clusterer_json=mock_clusterer_data,
        metric="cosine",
    )

    # Distances: cluster 0 is closer (0.1) than cluster 1 (0.9)
    distances = np.array([0.1, 0.9])
    json_mapping_str = calculator._create_distance_json_mapping(distances)
    
    mapping = json.loads(json_mapping_str)
    
    # "AI" should have the minimum distance (0.1 from cluster 0)
    assert mapping["AI"] == 0.1
    assert mapping["machine learning"] == 0.1
    assert mapping["deep learning"] == 0.9


def test_empty_text_handling():
    """Test handling of empty or None text values."""
    from RAG_supporters.clustering.topic_distance_calculator import calculate_topic_distances_from_csv

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create CSV with empty and None-like values
        csv_path = tmpdir / "test_data.csv"
        df = pd.DataFrame({
            "question_text": ["What is ML?", "", None, "Valid question"],
            "source_text": ["ML is AI", None, "", "Valid source"],
        })
        df.to_csv(csv_path, index=False)

        # Create mock KeywordClusterer JSON
        clusterer_path = tmpdir / "clusters.json"
        mock_clusterer_data = {
            "centroids": [np.random.rand(384).tolist()],
            "cluster_stats": {
                "0": {
                    "topic_descriptors": ["machine learning", "AI"],
                    "size": 10,
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

        # Test processing - should handle empty/None values gracefully
        result_df = calculate_topic_distances_from_csv(
            csv_path=csv_path,
            keyword_clusterer_json=clusterer_path,
            embedder=embedder,
            show_progress=False,
        )

        # Verify output
        assert len(result_df) == 4
        
        # First row should have both scores
        assert pd.notna(result_df.at[0, "question_term_distance_scores"])
        assert pd.notna(result_df.at[0, "source_term_distance_scores"])
        
        # Second row: question is empty, source is None
        # Both should be None or handled gracefully
        assert pd.isna(result_df.at[1, "question_term_distance_scores"])
        
        # Third row: question is None, source is empty
        assert pd.isna(result_df.at[2, "source_term_distance_scores"])
        
        # Fourth row should have both scores
        assert pd.notna(result_df.at[3, "question_term_distance_scores"])
        assert pd.notna(result_df.at[3, "source_term_distance_scores"])


def test_embedding_dimension_mismatch():
    """Test error when embedding dimension doesn't match centroids."""
    from RAG_supporters.clustering.topic_distance_calculator import TopicDistanceCalculator

    # Create mock data with 384-dim centroids
    mock_clusterer_data = {
        "centroids": [np.random.rand(384).tolist()],
        "cluster_stats": {},
    }

    calculator = TopicDistanceCalculator(
        keyword_clusterer_json=mock_clusterer_data,
        metric="cosine",
    )

    # Test with wrong dimension embedding (128 instead of 384)
    wrong_dim_embedding = np.random.rand(128)
    
    # Should raise error when computing distances
    with pytest.raises((ValueError, IndexError)):
        calculator._compute_distances_to_centroids(wrong_dim_embedding)


def test_database_embedding_not_found(caplog):
    """Test handling when database doesn't return embedding."""
    from RAG_supporters.clustering.topic_distance_calculator import TopicDistanceCalculator

    # Create mock database that returns None
    class MockDatabase:
        def get_embedding(self, item_id, collection=None):
            return None  # Simulate not found

    mock_clusterer_data = {
        "centroids": [np.random.rand(384).tolist()],
        "cluster_stats": {},
    }

    calculator = TopicDistanceCalculator(
        keyword_clusterer_json=mock_clusterer_data,
        metric="cosine",
    )

    database = MockDatabase()
    
    # Should return None and log warning
    with caplog.at_level(logging.WARNING):
        result = calculator._get_embedding_from_database(
            item_id="test_id",
            database=database,
            collection_name="questions"
        )
    
    assert result is None
    # Check that warning was logged
    assert any("Embedding not found" in record.message for record in caplog.records)
    assert any("test_id" in record.message for record in caplog.records)


def test_embedder_wrapping():
    """Test that non-KeywordEmbedder embedders are automatically wrapped."""
    from RAG_supporters.clustering.topic_distance_calculator import TopicDistanceCalculator

    # Create mock sentence-transformers-like embedder
    class MockSentenceTransformer:
        def encode(self, texts, **kwargs):
            if isinstance(texts, str):
                texts = [texts]
            return np.array([np.random.rand(384) for _ in texts])

    mock_clusterer_data = {
        "centroids": [np.random.rand(384).tolist()],
        "cluster_stats": {},
    }

    # Pass non-KeywordEmbedder instance
    st_embedder = MockSentenceTransformer()
    calculator = TopicDistanceCalculator(
        keyword_clusterer_json=mock_clusterer_data,
        embedder=st_embedder,
        metric="cosine",
    )

    # Should be wrapped in KeywordEmbedder
    from RAG_supporters.embeddings.keyword_embedder import KeywordEmbedder
    assert isinstance(calculator.embedder, KeywordEmbedder)
    
    # Should still work for embedding text
    embedding = calculator._get_embedding_from_text("test text")
    assert embedding.shape == (384,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
