"""
Unit tests for keyword_clustering module.

Tests cover:
- KeywordClusterer initialization
- Clustering operations (fit, predict)
- Topic descriptor extraction
- Assignment operations (hard/soft modes)
- Configuration persistence (save/load)
- State management
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from .keyword_clustering import KeywordClusterer, cluster_keywords_from_embeddings


@pytest.fixture
def sample_embeddings():
    """Create sample keyword embeddings for testing."""
    np.random.seed(42)
    n_keywords = 20
    embedding_dim = 10

    # Create embeddings with 3 distinct clusters
    embeddings = {}
    for i in range(n_keywords):
        if i < 7:
            base = np.array([1.0, 0.0, 0.0] + [0.0] * (embedding_dim - 3))
        elif i < 14:
            base = np.array([0.0, 1.0, 0.0] + [0.0] * (embedding_dim - 3))
        else:
            base = np.array([0.0, 0.0, 1.0] + [0.0] * (embedding_dim - 3))

        embedding = base + np.random.normal(0, 0.05, embedding_dim)
        embedding = embedding / np.linalg.norm(embedding)
        embeddings[f"keyword_{i}"] = embedding

    return embeddings


@pytest.fixture
def fitted_clusterer(sample_embeddings):
    """Create a fitted clusterer for testing."""
    clusterer = KeywordClusterer(n_clusters=3, random_state=42)
    clusterer.fit(sample_embeddings)
    return clusterer


class TestKeywordClustererInit:
    """Test KeywordClusterer initialization."""

    def test_init_parameters(self):
        """Test initialization with default and custom parameters."""
        # Test default parameters
        clusterer = KeywordClusterer()
        assert clusterer.algorithm == "kmeans"
        assert clusterer.n_clusters == 8
        assert clusterer.random_state == 42
        assert clusterer.keywords is None
        assert clusterer.embeddings_matrix is None
        assert clusterer.cluster_labels is None

        # Test default assignment config
        assert clusterer._default_assignment_mode == "soft"
        assert clusterer._default_threshold == 0.1
        assert clusterer._default_metric == "cosine"

        # Test custom parameters
        clusterer = KeywordClusterer(
            algorithm="bisecting_kmeans", n_clusters=5, random_state=123
        )
        assert clusterer.algorithm == "bisecting_kmeans"
        assert clusterer.n_clusters == 5
        assert clusterer.random_state == 123

    def test_init_invalid_algorithm(self):
        """Test initialization with invalid algorithm."""
        with pytest.raises(ValueError):
            clusterer = KeywordClusterer(algorithm="invalid")
            clusterer._create_model()


class TestKeywordClustererFitting:
    """Test KeywordClusterer fitting operations."""

    def test_fit(self, sample_embeddings):
        """Test successful fitting and state verification."""
        clusterer = KeywordClusterer(n_clusters=3, random_state=42)
        result = clusterer.fit(sample_embeddings)

        # Check return value is self for chaining
        assert result is clusterer

        # Check state after fitting
        assert clusterer.cluster_labels is not None
        assert len(clusterer.keywords) == len(sample_embeddings)
        assert clusterer.embeddings_matrix.shape[0] == len(sample_embeddings)
        assert clusterer.embeddings_matrix.shape[1] == 10

    def test_get_cluster_assignments(self, fitted_clusterer, sample_embeddings):
        """Test getting cluster assignments."""
        assignments = fitted_clusterer.get_cluster_assignments()
        assert len(assignments) == len(sample_embeddings)
        assert "keyword_0" in assignments
        assert isinstance(assignments["keyword_0"], int)

    def test_get_cluster_assignments_unfitted(self):
        """Test getting assignments before fitting raises error."""
        clusterer = KeywordClusterer(n_clusters=3)
        with pytest.raises(ValueError):
            clusterer.get_cluster_assignments()

    def test_get_clusters_and_centroids(self, fitted_clusterer):
        """Test getting clusters and centroids."""
        # Test get_clusters
        clusters = fitted_clusterer.get_clusters()
        assert isinstance(clusters, dict)

        # Check all keywords are assigned
        total_keywords = sum(len(kws) for kws in clusters.values())
        assert total_keywords == 20

        # Test get_centroids
        centroids = fitted_clusterer.get_centroids()
        assert centroids.shape[0] == 3
        assert centroids.shape[1] == 10


class TestTopicDescriptorExtraction:
    """Test topic descriptor extraction."""

    def test_extract_topic_descriptors(self, fitted_clusterer):
        """Test extracting topic descriptors with different metrics."""
        # Test with default (euclidean)
        topics = fitted_clusterer.extract_topic_descriptors(n_descriptors=5)
        assert len(topics) == 3
        for topic_id, descriptors in topics.items():
            assert len(descriptors) == 5
            assert isinstance(descriptors[0], str)

        # Test with euclidean metric explicitly
        topics_euclidean = fitted_clusterer.extract_topic_descriptors(
            n_descriptors=3, metric="euclidean"
        )
        assert len(topics_euclidean) == 3

        # Test with cosine metric
        topics_cosine = fitted_clusterer.extract_topic_descriptors(
            n_descriptors=3, metric="cosine"
        )
        assert len(topics_cosine) == 3

    def test_extract_topic_descriptors_invalid_metric(self, fitted_clusterer):
        """Test topic extraction with invalid metric."""
        with pytest.raises(ValueError):
            fitted_clusterer.extract_topic_descriptors(metric="invalid")

    def test_extract_topic_descriptors_unfitted(self):
        """Test extracting topics before fitting raises error."""
        unfitted_clusterer = KeywordClusterer(n_clusters=3)
        with pytest.raises(ValueError):
            unfitted_clusterer.extract_topic_descriptors()


class TestAssignmentConfiguration:
    """Test assignment configuration."""

    def test_configure_assignment(self, fitted_clusterer):
        """Test configuring assignment parameters."""
        result = fitted_clusterer.configure_assignment(
            assignment_mode="hard", threshold=0.5, metric="euclidean"
        )

        # Check return value is self for chaining
        assert result is fitted_clusterer

        # Check configuration was set
        assert fitted_clusterer._default_assignment_mode == "hard"
        assert fitted_clusterer._default_threshold == 0.5
        assert fitted_clusterer._default_metric == "euclidean"

    def test_configure_assignment_invalid_mode(self, fitted_clusterer):
        """Test configuring with invalid mode."""
        with pytest.raises(ValueError):
            fitted_clusterer.configure_assignment(assignment_mode="invalid")

    def test_configure_assignment_invalid_metric(self, fitted_clusterer):
        """Test configuring with invalid metric."""
        with pytest.raises(ValueError):
            fitted_clusterer.configure_assignment(metric="invalid")


class TestAssignmentOperations:
    """Test assignment operations."""

    @pytest.fixture
    def test_embedding(self):
        """Create a test embedding."""
        np.random.seed(42)
        embedding = np.random.randn(10)
        return embedding / np.linalg.norm(embedding)

    def test_assign_modes(self, fitted_clusterer, test_embedding):
        """Test assignment with default, hard, and soft modes with thresholds."""
        # Test default assignment
        result = fitted_clusterer.assign(test_embedding)
        assert "mode" in result
        assert "assigned_clusters" in result
        assert "primary_cluster" in result
        assert "probabilities" in result

        # Check probabilities sum to 1
        probs = list(result["probabilities"].values())
        assert abs(sum(probs) - 1.0) < 1e-5

        # Test hard mode
        result_hard = fitted_clusterer.assign(test_embedding, mode="hard")
        assert result_hard["mode"] == "hard"
        assert len(result_hard["assigned_clusters"]) <= 1

        # Test soft mode
        result_soft = fitted_clusterer.assign(
            test_embedding, mode="soft", threshold=0.1
        )
        assert result_soft["mode"] == "soft"
        assert len(result_soft["assigned_clusters"]) > 0

        # Test threshold filtering
        result_threshold = fitted_clusterer.assign(
            test_embedding, mode="soft", threshold=0.3
        )
        # Only clusters above threshold should be included
        for cluster_id in result_threshold["assigned_clusters"]:
            assert result_threshold["probabilities"][cluster_id] >= 0.3

    def test_assign_invalid_mode(self, fitted_clusterer, test_embedding):
        """Test assignment with invalid mode."""
        with pytest.raises(ValueError):
            fitted_clusterer.assign(test_embedding, mode="invalid")

    def test_assign_unfitted(self, test_embedding):
        """Test assignment before fitting raises error."""
        unfitted_clusterer = KeywordClusterer(n_clusters=3)
        with pytest.raises(ValueError):
            unfitted_clusterer.assign(test_embedding)

    def test_assign_batch(self, fitted_clusterer):
        """Test batch assignment with custom parameters."""
        np.random.seed(42)
        embeddings = {
            f"source_{i}": np.random.randn(10) / np.linalg.norm(np.random.randn(10))
            for i in range(5)
        }

        # Test default batch assignment
        results = fitted_clusterer.assign_batch(embeddings)
        assert len(results) == 5
        for source_id, result in results.items():
            assert "assigned_clusters" in result
            assert "primary_cluster" in result

        # Test with custom parameters
        results_custom = fitted_clusterer.assign_batch(
            embeddings, mode="hard", threshold=0.5
        )
        for result in results_custom.values():
            assert result["mode"] == "hard"


class TestPersistence:
    """Test save/load operations."""

    @pytest.fixture
    def configured_clusterer(self, fitted_clusterer):
        """Create a configured clusterer with topics."""
        fitted_clusterer.extract_topic_descriptors(n_descriptors=5)
        fitted_clusterer.configure_assignment(
            assignment_mode="soft", threshold=0.2, metric="cosine"
        )
        return fitted_clusterer

    def test_save_and_load_results(self, configured_clusterer):
        """Test saving results and loading with all configurations restored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "clusters.json"

            # Save results
            configured_clusterer.save_results(str(output_path), include_topics=True)
            assert output_path.exists()

            # Check file content
            with open(output_path) as f:
                data = json.load(f)

            assert "metadata" in data
            assert "cluster_assignments" in data
            assert "clusters" in data
            assert "centroids" in data
            assert "assignment_config" in data["metadata"]

            # Verify assignment config is saved
            config = data["metadata"]["assignment_config"]
            assert config["default_mode"] == "soft"
            assert config["default_threshold"] == 0.2
            assert config["default_metric"] == "cosine"

            # Load results
            loaded = KeywordClusterer.from_results(str(output_path))

            # Verify basic properties
            assert loaded.n_clusters == 3
            assert loaded.algorithm == "kmeans"
            assert loaded._is_fitted

            # Verify assignment config is restored
            assert loaded._default_assignment_mode == "soft"
            assert loaded._default_threshold == 0.2
            assert loaded._default_metric == "cosine"

            # Verify topics are restored
            assert len(loaded.topics) == 3
            for topic_descriptors in loaded.topics.values():
                assert len(topic_descriptors) == 5

            # Verify loaded clusterer can perform assignments
            test_embedding = np.random.randn(10)
            test_embedding = test_embedding / np.linalg.norm(test_embedding)

            result = loaded.assign(test_embedding)
            assert "assigned_clusters" in result
            assert "primary_cluster" in result


class TestConvenienceFunction:
    """Test cluster_keywords_from_embeddings convenience function."""

    def test_cluster_keywords_from_embeddings(self, sample_embeddings):
        """Test convenience function with and without saving."""
        # Test without saving
        clusterer, topics = cluster_keywords_from_embeddings(
            sample_embeddings, n_clusters=3, n_descriptors=5, random_state=42
        )

        assert isinstance(clusterer, KeywordClusterer)
        assert len(topics) == 3
        for descriptors in topics.values():
            assert len(descriptors) == 5

        # Test with saving
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "clusters.json"

            clusterer, topics = cluster_keywords_from_embeddings(
                sample_embeddings,
                n_clusters=3,
                n_descriptors=5,
                output_path=str(output_path),
                random_state=42,
            )

            assert output_path.exists()


class TestComputeDistances:
    """Test distance computation methods."""

    @pytest.fixture
    def test_embedding(self):
        """Create a test embedding."""
        np.random.seed(42)
        embedding = np.random.randn(10)
        return embedding / np.linalg.norm(embedding)

    def test_compute_distances(self, fitted_clusterer, test_embedding):
        """Test Euclidean and cosine distance computation."""
        # Test Euclidean distance
        distances_euclidean = fitted_clusterer.compute_distances(
            test_embedding, metric="euclidean"
        )
        assert len(distances_euclidean) == 3
        assert all(d >= 0 for d in distances_euclidean)

        # Test cosine distance
        distances_cosine = fitted_clusterer.compute_distances(
            test_embedding, metric="cosine"
        )
        assert len(distances_cosine) == 3
        assert all(0 <= d <= 2 for d in distances_cosine)

    def test_compute_distances_invalid_metric(self, fitted_clusterer, test_embedding):
        """Test distance computation with invalid metric."""
        with pytest.raises(ValueError):
            fitted_clusterer.compute_distances(test_embedding, metric="invalid")
