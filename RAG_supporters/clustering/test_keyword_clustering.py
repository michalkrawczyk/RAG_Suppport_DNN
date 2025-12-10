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
    """
    Create sample keyword embeddings for testing.

    This fixture generates 20 synthetic embeddings designed to form 3 distinct clusters:
    - Cluster 1 (keywords 0-6): Base vector [1.0, 0.0, 0.0, ...] + small noise
    - Cluster 2 (keywords 7-13): Base vector [0.0, 1.0, 0.0, ...] + small noise
    - Cluster 3 (keywords 14-19): Base vector [0.0, 0.0, 1.0, ...] + small noise

    Each embedding is normalized to unit length after adding Gaussian noise (σ=0.05)
    to simulate realistic embedding variations within clusters.

    Returns
    -------
    dict
        Dictionary mapping keyword names to 10-dimensional normalized embeddings
    """
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
        assert (
            clusterer.algorithm == "kmeans"
        ), f"Expected algorithm 'kmeans', got '{clusterer.algorithm}'"
        assert (
            clusterer.n_clusters == 8
        ), f"Expected n_clusters 8, got {clusterer.n_clusters}"
        assert (
            clusterer.random_state == 42
        ), f"Expected random_state 42, got {clusterer.random_state}"
        assert (
            clusterer.keywords is None
        ), f"Expected keywords None, got {clusterer.keywords}"
        assert (
            clusterer.embeddings_matrix is None
        ), f"Expected embeddings_matrix None, got {clusterer.embeddings_matrix}"
        assert (
            clusterer.cluster_labels is None
        ), f"Expected cluster_labels None, got {clusterer.cluster_labels}"

        # Test default assignment config
        assert (
            clusterer._default_assignment_mode == "soft"
        ), f"Expected default mode 'soft', got '{clusterer._default_assignment_mode}'"
        assert (
            clusterer._default_threshold == 0.1
        ), f"Expected default threshold 0.1, got {clusterer._default_threshold}"
        assert (
            clusterer._default_metric == "cosine"
        ), f"Expected default metric 'cosine', got '{clusterer._default_metric}'"

        # Test custom parameters
        clusterer = KeywordClusterer(
            algorithm="bisecting_kmeans", n_clusters=5, random_state=123
        )
        assert (
            clusterer.algorithm == "bisecting_kmeans"
        ), f"Expected algorithm 'bisecting_kmeans', got '{clusterer.algorithm}'"
        assert (
            clusterer.n_clusters == 5
        ), f"Expected n_clusters 5, got {clusterer.n_clusters}"
        assert (
            clusterer.random_state == 123
        ), f"Expected random_state 123, got {clusterer.random_state}"

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
        assert result is clusterer, "fit() should return self for method chaining"

        # Check state after fitting
        assert (
            clusterer.cluster_labels is not None
        ), "cluster_labels should be set after fitting"
        assert len(clusterer.keywords) == len(
            sample_embeddings
        ), f"Expected {len(sample_embeddings)} keywords, got {len(clusterer.keywords)}"
        assert clusterer.embeddings_matrix.shape[0] == len(
            sample_embeddings
        ), f"Expected embeddings_matrix with {len(sample_embeddings)} rows, got {clusterer.embeddings_matrix.shape[0]}"
        assert (
            clusterer.embeddings_matrix.shape[1] == 10
        ), f"Expected embeddings with 10 dimensions, got {clusterer.embeddings_matrix.shape[1]}"

    def test_get_cluster_assignments(self, fitted_clusterer, sample_embeddings):
        """Test getting cluster assignments."""
        assignments = fitted_clusterer.get_cluster_assignments()
        assert len(assignments) == len(
            sample_embeddings
        ), f"Expected {len(sample_embeddings)} assignments, got {len(assignments)}"
        assert "keyword_0" in assignments, "Expected 'keyword_0' in assignments"
        assert isinstance(
            assignments["keyword_0"], int
        ), f"Expected int assignment, got {type(assignments['keyword_0'])}"

    def test_get_cluster_assignments_unfitted(self):
        """Test getting assignments before fitting raises error."""
        clusterer = KeywordClusterer(n_clusters=3)
        with pytest.raises(ValueError):
            clusterer.get_cluster_assignments()

    def test_get_clusters_and_centroids(self, fitted_clusterer):
        """Test getting clusters and centroids."""
        # Test get_clusters
        clusters = fitted_clusterer.get_clusters()
        assert isinstance(clusters, dict), f"Expected dict, got {type(clusters)}"

        # Check all keywords are assigned
        total_keywords = sum(len(kws) for kws in clusters.values())
        assert (
            total_keywords == 20
        ), f"Expected 20 total keywords assigned, got {total_keywords}"

        # Test get_centroids
        centroids = fitted_clusterer.get_centroids()
        assert (
            centroids.shape[0] == 3
        ), f"Expected 3 centroids, got {centroids.shape[0]}"
        assert (
            centroids.shape[1] == 10
        ), f"Expected 10-dimensional centroids, got {centroids.shape[1]}"


class TestTopicDescriptorExtraction:
    """Test topic descriptor extraction."""

    def test_extract_topic_descriptors(self, fitted_clusterer):
        """Test extracting topic descriptors with different metrics."""
        # Test with default (euclidean)
        topics = fitted_clusterer.extract_topic_descriptors(n_descriptors=5)
        assert len(topics) == 3, f"Expected 3 topics, got {len(topics)}"
        for topic_id, descriptors in topics.items():
            assert (
                len(descriptors) == 5
            ), f"Expected 5 descriptors for topic {topic_id}, got {len(descriptors)}"
            assert isinstance(
                descriptors[0], str
            ), f"Expected str descriptor, got {type(descriptors[0])}"

        # Test with euclidean metric explicitly
        topics_euclidean = fitted_clusterer.extract_topic_descriptors(
            n_descriptors=3, metric="euclidean"
        )
        assert (
            len(topics_euclidean) == 3
        ), f"Expected 3 topics with euclidean, got {len(topics_euclidean)}"

        # Test with cosine metric
        topics_cosine = fitted_clusterer.extract_topic_descriptors(
            n_descriptors=3, metric="cosine"
        )
        assert (
            len(topics_cosine) == 3
        ), f"Expected 3 topics with cosine, got {len(topics_cosine)}"

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
        assert (
            result is fitted_clusterer
        ), "configure_assignment() should return self for chaining"

        # Check configuration was set
        assert (
            fitted_clusterer._default_assignment_mode == "hard"
        ), f"Expected mode 'hard', got '{fitted_clusterer._default_assignment_mode}'"
        assert (
            fitted_clusterer._default_threshold == 0.5
        ), f"Expected threshold 0.5, got {fitted_clusterer._default_threshold}"
        assert (
            fitted_clusterer._default_metric == "euclidean"
        ), f"Expected metric 'euclidean', got '{fitted_clusterer._default_metric}'"

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
        assert "mode" in result, "Expected 'mode' in result"
        assert "assigned_clusters" in result, "Expected 'assigned_clusters' in result"
        assert "primary_cluster" in result, "Expected 'primary_cluster' in result"
        assert "probabilities" in result, "Expected 'probabilities' in result"

        # Check probabilities sum to 1
        probs = list(result["probabilities"].values())
        assert (
            abs(sum(probs) - 1.0) < 1e-5
        ), f"Expected probabilities to sum to 1.0, got {sum(probs)}"

        # Test hard mode
        result_hard = fitted_clusterer.assign(test_embedding, mode="hard")
        assert (
            result_hard["mode"] == "hard"
        ), f"Expected mode 'hard', got '{result_hard['mode']}'"
        assert (
            len(result_hard["assigned_clusters"]) <= 1
        ), f"Hard mode should assign to at most 1 cluster, got {len(result_hard['assigned_clusters'])}"

        # Test soft mode
        result_soft = fitted_clusterer.assign(
            test_embedding, mode="soft", threshold=0.1
        )
        assert (
            result_soft["mode"] == "soft"
        ), f"Expected mode 'soft', got '{result_soft['mode']}'"
        assert (
            len(result_soft["assigned_clusters"]) > 0
        ), "Soft mode should assign to at least 1 cluster"

        # Test threshold filtering
        result_threshold = fitted_clusterer.assign(
            test_embedding, mode="soft", threshold=0.3
        )
        # Only clusters above threshold should be included
        for cluster_id in result_threshold["assigned_clusters"]:
            prob = result_threshold["probabilities"][cluster_id]
            assert (
                prob >= 0.3
            ), f"Cluster {cluster_id} has probability {prob} below threshold 0.3"

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
        assert len(results) == 5, f"Expected 5 results, got {len(results)}"
        for source_id, result in results.items():
            assert (
                "assigned_clusters" in result
            ), f"Expected 'assigned_clusters' for {source_id}"
            assert (
                "primary_cluster" in result
            ), f"Expected 'primary_cluster' for {source_id}"

        # Test with custom parameters
        results_custom = fitted_clusterer.assign_batch(
            embeddings, mode="hard", threshold=0.5
        )
        for source_id, result in results_custom.items():
            assert (
                result["mode"] == "hard"
            ), f"Expected mode 'hard' for {source_id}, got '{result['mode']}'"


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
            assert output_path.exists(), f"Expected output file at {output_path}"

            # Check file content
            with open(output_path) as f:
                data = json.load(f)

            assert "metadata" in data, "Expected 'metadata' in saved data"
            assert (
                "cluster_assignments" in data
            ), "Expected 'cluster_assignments' in saved data"
            assert "clusters" in data, "Expected 'clusters' in saved data"
            assert "centroids" in data, "Expected 'centroids' in saved data"
            assert (
                "assignment_config" in data["metadata"]
            ), "Expected 'assignment_config' in metadata"

            # Verify assignment config is saved
            config = data["metadata"]["assignment_config"]
            assert (
                config["default_mode"] == "soft"
            ), f"Expected saved mode 'soft', got '{config['default_mode']}'"
            assert (
                config["default_threshold"] == 0.2
            ), f"Expected saved threshold 0.2, got {config['default_threshold']}"
            assert (
                config["default_metric"] == "cosine"
            ), f"Expected saved metric 'cosine', got '{config['default_metric']}'"

            # Load results
            loaded = KeywordClusterer.from_results(str(output_path))

            # Verify basic properties
            assert (
                loaded.n_clusters == 3
            ), f"Expected loaded n_clusters 3, got {loaded.n_clusters}"
            assert (
                loaded.algorithm == "kmeans"
            ), f"Expected loaded algorithm 'kmeans', got '{loaded.algorithm}'"
            assert loaded._is_fitted, "Expected loaded clusterer to be in fitted state"

            # Verify assignment config is restored
            assert (
                loaded._default_assignment_mode == "soft"
            ), f"Expected loaded mode 'soft', got '{loaded._default_assignment_mode}'"
            assert (
                loaded._default_threshold == 0.2
            ), f"Expected loaded threshold 0.2, got {loaded._default_threshold}"
            assert (
                loaded._default_metric == "cosine"
            ), f"Expected loaded metric 'cosine', got '{loaded._default_metric}'"

            # Verify topics are restored
            assert (
                len(loaded.topics) == 3
            ), f"Expected 3 loaded topics, got {len(loaded.topics)}"
            for topic_id, topic_descriptors in loaded.topics.items():
                assert (
                    len(topic_descriptors) == 5
                ), f"Expected 5 descriptors for topic {topic_id}, got {len(topic_descriptors)}"

            # Verify loaded clusterer can perform assignments
            test_embedding = np.random.randn(10)
            test_embedding = test_embedding / np.linalg.norm(test_embedding)

            result = loaded.assign(test_embedding)
            assert (
                "assigned_clusters" in result
            ), "Expected 'assigned_clusters' in assignment result"
            assert (
                "primary_cluster" in result
            ), "Expected 'primary_cluster' in assignment result"


class TestConvenienceFunction:
    """Test cluster_keywords_from_embeddings convenience function."""

    def test_cluster_keywords_from_embeddings(self, sample_embeddings):
        """Test convenience function with and without saving."""
        # Test without saving
        clusterer, topics = cluster_keywords_from_embeddings(
            sample_embeddings, n_clusters=3, n_descriptors=5, random_state=42
        )

        assert isinstance(
            clusterer, KeywordClusterer
        ), f"Expected KeywordClusterer instance, got {type(clusterer)}"
        assert len(topics) == 3, f"Expected 3 topics, got {len(topics)}"
        for topic_id, descriptors in topics.items():
            assert (
                len(descriptors) == 5
            ), f"Expected 5 descriptors for topic {topic_id}, got {len(descriptors)}"

        # Test with saving
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "clusters.json"

            clusterer, _ = cluster_keywords_from_embeddings(
                sample_embeddings,
                n_clusters=3,
                n_descriptors=5,
                output_path=str(output_path),
                random_state=42,
            )

            assert output_path.exists(), f"Expected saved file at {output_path}"


class TestComputeDistances:
    """Test distance computation methods."""

    @pytest.fixture
    def test_embedding(self):
        """Create a test embedding."""
        np.random.seed(42)
        embedding = np.random.randn(10)
        return embedding / np.linalg.norm(embedding)

    @pytest.mark.parametrize(
        "metric,min_value,max_value",
        [
            ("euclidean", 0, float("inf")),
            ("cosine", 0, 2),
        ],
    )
    def test_compute_distances(
        self, fitted_clusterer, test_embedding, metric, min_value, max_value
    ):
        """Test distance computation with different metrics."""
        distances = fitted_clusterer.compute_distances(test_embedding, metric=metric)

        assert (
            len(distances) == 3
        ), f"Expected 3 distances with {metric} metric, got {len(distances)}"
        for i, d in enumerate(distances):
            assert (
                min_value <= d <= max_value
            ), f"Distance {i} with {metric} metric ({d}) outside expected range [{min_value}, {max_value}]"

    def test_compute_distances_invalid_metric(self, fitted_clusterer, test_embedding):
        """Test distance computation with invalid metric."""
        with pytest.raises(ValueError):
            fitted_clusterer.compute_distances(test_embedding, metric="invalid")
