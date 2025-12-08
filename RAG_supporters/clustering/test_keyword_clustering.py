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
import unittest
from pathlib import Path

import numpy as np

from .keyword_clustering import (KeywordClusterer,
                                 cluster_keywords_from_embeddings)


class TestKeywordClustererInit(unittest.TestCase):
    """Test KeywordClusterer initialization."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        clusterer = KeywordClusterer()
        self.assertEqual(clusterer.algorithm, "kmeans")
        self.assertEqual(clusterer.n_clusters, 8)
        self.assertEqual(clusterer.random_state, 42)
        self.assertIsNone(clusterer.keywords)
        self.assertIsNone(clusterer.embeddings_matrix)
        self.assertIsNone(clusterer.cluster_labels)

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        clusterer = KeywordClusterer(
            algorithm="bisecting_kmeans", n_clusters=5, random_state=123
        )
        self.assertEqual(clusterer.algorithm, "bisecting_kmeans")
        self.assertEqual(clusterer.n_clusters, 5)
        self.assertEqual(clusterer.random_state, 123)

    def test_init_invalid_algorithm(self):
        """Test initialization with invalid algorithm."""
        with self.assertRaises(ValueError):
            clusterer = KeywordClusterer(algorithm="invalid")
            # Trigger model creation
            clusterer._create_model()

    def test_default_assignment_config(self):
        """Test default assignment configuration."""
        clusterer = KeywordClusterer()
        self.assertEqual(clusterer._default_assignment_mode, "soft")
        self.assertEqual(clusterer._default_threshold, 0.1)
        self.assertEqual(clusterer._default_metric, "cosine")


class TestKeywordClustererFitting(unittest.TestCase):
    """Test KeywordClusterer fitting operations."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n_keywords = 20
        self.embedding_dim = 10

        # Create test embeddings with 3 distinct clusters
        self.keyword_embeddings = {}
        for i in range(self.n_keywords):
            if i < 7:
                base = np.array([1.0, 0.0, 0.0] + [0.0] * (self.embedding_dim - 3))
            elif i < 14:
                base = np.array([0.0, 1.0, 0.0] + [0.0] * (self.embedding_dim - 3))
            else:
                base = np.array([0.0, 0.0, 1.0] + [0.0] * (self.embedding_dim - 3))

            embedding = base + np.random.normal(0, 0.05, self.embedding_dim)
            embedding = embedding / np.linalg.norm(embedding)
            self.keyword_embeddings[f"keyword_{i}"] = embedding

    def test_fit_success(self):
        """Test successful fitting."""
        clusterer = KeywordClusterer(n_clusters=3, random_state=42)
        result = clusterer.fit(self.keyword_embeddings)

        # Check return value is self for chaining
        self.assertIs(result, clusterer)

        # Check state after fitting
        self.assertIsNotNone(clusterer.cluster_labels)
        self.assertEqual(len(clusterer.keywords), self.n_keywords)
        self.assertEqual(clusterer.embeddings_matrix.shape[0], self.n_keywords)
        self.assertEqual(clusterer.embeddings_matrix.shape[1], self.embedding_dim)

    def test_fit_sets_fitted_state(self):
        """Test that fitting sets the fitted state."""
        clusterer = KeywordClusterer(n_clusters=3)
        self.assertIsNone(clusterer.cluster_labels)

        clusterer.fit(self.keyword_embeddings)
        self.assertIsNotNone(clusterer.cluster_labels)

    def test_get_cluster_assignments(self):
        """Test getting cluster assignments."""
        clusterer = KeywordClusterer(n_clusters=3, random_state=42)
        clusterer.fit(self.keyword_embeddings)

        assignments = clusterer.get_cluster_assignments()
        self.assertEqual(len(assignments), self.n_keywords)
        self.assertIn("keyword_0", assignments)
        self.assertIsInstance(assignments["keyword_0"], int)

    def test_get_cluster_assignments_unfitted(self):
        """Test getting assignments before fitting raises error."""
        clusterer = KeywordClusterer(n_clusters=3)
        with self.assertRaises(ValueError):
            clusterer.get_cluster_assignments()

    def test_get_clusters(self):
        """Test getting clusters."""
        clusterer = KeywordClusterer(n_clusters=3, random_state=42)
        clusterer.fit(self.keyword_embeddings)

        clusters = clusterer.get_clusters()
        self.assertIsInstance(clusters, dict)

        # Check all keywords are assigned
        total_keywords = sum(len(kws) for kws in clusters.values())
        self.assertEqual(total_keywords, self.n_keywords)

    def test_get_centroids(self):
        """Test getting centroids."""
        clusterer = KeywordClusterer(n_clusters=3, random_state=42)
        clusterer.fit(self.keyword_embeddings)

        centroids = clusterer.get_centroids()
        self.assertEqual(centroids.shape[0], 3)
        self.assertEqual(centroids.shape[1], self.embedding_dim)


class TestTopicDescriptorExtraction(unittest.TestCase):
    """Test topic descriptor extraction."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n_keywords = 30
        self.embedding_dim = 10

        self.keyword_embeddings = {}
        for i in range(self.n_keywords):
            embedding = np.random.randn(self.embedding_dim)
            embedding = embedding / np.linalg.norm(embedding)
            self.keyword_embeddings[f"keyword_{i}"] = embedding

        self.clusterer = KeywordClusterer(n_clusters=3, random_state=42)
        self.clusterer.fit(self.keyword_embeddings)

    def test_extract_topic_descriptors(self):
        """Test extracting topic descriptors."""
        topics = self.clusterer.extract_topic_descriptors(n_descriptors=5)

        self.assertEqual(len(topics), 3)
        for topic_id, descriptors in topics.items():
            self.assertEqual(len(descriptors), 5)
            self.assertIsInstance(descriptors[0], str)

    def test_extract_topic_descriptors_euclidean(self):
        """Test topic extraction with Euclidean metric."""
        topics = self.clusterer.extract_topic_descriptors(
            n_descriptors=3, metric="euclidean"
        )
        self.assertEqual(len(topics), 3)

    def test_extract_topic_descriptors_cosine(self):
        """Test topic extraction with cosine metric."""
        topics = self.clusterer.extract_topic_descriptors(
            n_descriptors=3, metric="cosine"
        )
        self.assertEqual(len(topics), 3)

    def test_extract_topic_descriptors_invalid_metric(self):
        """Test topic extraction with invalid metric."""
        with self.assertRaises(ValueError):
            self.clusterer.extract_topic_descriptors(metric="invalid")

    def test_extract_topic_descriptors_unfitted(self):
        """Test extracting topics before fitting raises error."""
        unfitted_clusterer = KeywordClusterer(n_clusters=3)
        with self.assertRaises(ValueError):
            unfitted_clusterer.extract_topic_descriptors()


class TestAssignmentConfiguration(unittest.TestCase):
    """Test assignment configuration."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.keyword_embeddings = {
            f"keyword_{i}": np.random.randn(10) / np.linalg.norm(np.random.randn(10))
            for i in range(20)
        }
        self.clusterer = KeywordClusterer(n_clusters=3, random_state=42)
        self.clusterer.fit(self.keyword_embeddings)

    def test_configure_assignment(self):
        """Test configuring assignment parameters."""
        result = self.clusterer.configure_assignment(
            assignment_mode="hard", threshold=0.5, metric="euclidean"
        )

        # Check return value is self for chaining
        self.assertIs(result, self.clusterer)

        # Check configuration was set
        self.assertEqual(self.clusterer._default_assignment_mode, "hard")
        self.assertEqual(self.clusterer._default_threshold, 0.5)
        self.assertEqual(self.clusterer._default_metric, "euclidean")

    def test_configure_assignment_invalid_mode(self):
        """Test configuring with invalid mode."""
        with self.assertRaises(ValueError):
            self.clusterer.configure_assignment(assignment_mode="invalid")

    def test_configure_assignment_invalid_metric(self):
        """Test configuring with invalid metric."""
        with self.assertRaises(ValueError):
            self.clusterer.configure_assignment(metric="invalid")


class TestAssignmentOperations(unittest.TestCase):
    """Test assignment operations."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.keyword_embeddings = {
            f"keyword_{i}": np.random.randn(10) / np.linalg.norm(np.random.randn(10))
            for i in range(20)
        }
        self.clusterer = KeywordClusterer(n_clusters=3, random_state=42)
        self.clusterer.fit(self.keyword_embeddings)

        self.test_embedding = np.random.randn(10)
        self.test_embedding = self.test_embedding / np.linalg.norm(self.test_embedding)

    def test_assign_default(self):
        """Test assignment with default configuration."""
        result = self.clusterer.assign(self.test_embedding)

        self.assertIn("mode", result)
        self.assertIn("assigned_clusters", result)
        self.assertIn("primary_cluster", result)
        self.assertIn("probabilities", result)

        # Check probabilities sum to 1
        probs = list(result["probabilities"].values())
        self.assertAlmostEqual(sum(probs), 1.0, places=5)

    def test_assign_hard_mode(self):
        """Test hard assignment mode."""
        result = self.clusterer.assign(self.test_embedding, mode="hard")

        self.assertEqual(result["mode"], "hard")
        self.assertLessEqual(len(result["assigned_clusters"]), 1)

    def test_assign_soft_mode(self):
        """Test soft assignment mode."""
        result = self.clusterer.assign(self.test_embedding, mode="soft", threshold=0.1)

        self.assertEqual(result["mode"], "soft")
        # Soft mode can assign to multiple clusters
        self.assertGreater(len(result["assigned_clusters"]), 0)

    def test_assign_with_threshold(self):
        """Test assignment with threshold filtering."""
        result = self.clusterer.assign(self.test_embedding, mode="soft", threshold=0.3)

        # Only clusters above threshold should be included
        for cluster_id in result["assigned_clusters"]:
            self.assertGreaterEqual(result["probabilities"][cluster_id], 0.3)

    def test_assign_invalid_mode(self):
        """Test assignment with invalid mode."""
        with self.assertRaises(ValueError):
            self.clusterer.assign(self.test_embedding, mode="invalid")

    def test_assign_unfitted(self):
        """Test assignment before fitting raises error."""
        unfitted_clusterer = KeywordClusterer(n_clusters=3)
        with self.assertRaises(ValueError):
            unfitted_clusterer.assign(self.test_embedding)

    def test_assign_batch(self):
        """Test batch assignment."""
        embeddings = {
            f"source_{i}": np.random.randn(10) / np.linalg.norm(np.random.randn(10))
            for i in range(5)
        }

        results = self.clusterer.assign_batch(embeddings)

        self.assertEqual(len(results), 5)
        for source_id, result in results.items():
            self.assertIn("assigned_clusters", result)
            self.assertIn("primary_cluster", result)

    def test_assign_batch_with_custom_params(self):
        """Test batch assignment with custom parameters."""
        embeddings = {
            f"source_{i}": np.random.randn(10) / np.linalg.norm(np.random.randn(10))
            for i in range(3)
        }

        results = self.clusterer.assign_batch(embeddings, mode="hard", threshold=0.5)

        for result in results.values():
            self.assertEqual(result["mode"], "hard")


class TestPersistence(unittest.TestCase):
    """Test save/load operations."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.keyword_embeddings = {
            f"keyword_{i}": np.random.randn(10) / np.linalg.norm(np.random.randn(10))
            for i in range(20)
        }
        self.clusterer = KeywordClusterer(n_clusters=3, random_state=42)
        self.clusterer.fit(self.keyword_embeddings)
        self.clusterer.extract_topic_descriptors(n_descriptors=5)
        self.clusterer.configure_assignment(
            assignment_mode="soft", threshold=0.2, metric="cosine"
        )

    def test_save_results(self):
        """Test saving clustering results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "clusters.json"
            self.clusterer.save_results(str(output_path), include_topics=True)

            self.assertTrue(output_path.exists())

            # Check file content
            with open(output_path) as f:
                data = json.load(f)

            self.assertIn("metadata", data)
            self.assertIn("cluster_assignments", data)
            self.assertIn("clusters", data)
            self.assertIn("centroids", data)
            self.assertIn("assignment_config", data["metadata"])

    def test_save_results_with_assignment_config(self):
        """Test that assignment config is persisted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "clusters.json"
            self.clusterer.save_results(str(output_path))

            with open(output_path) as f:
                data = json.load(f)

            config = data["metadata"]["assignment_config"]
            self.assertEqual(config["default_mode"], "soft")
            self.assertEqual(config["default_threshold"], 0.2)
            self.assertEqual(config["default_metric"], "cosine")

    def test_from_results(self):
        """Test loading from saved results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "clusters.json"
            self.clusterer.save_results(str(output_path), include_topics=True)

            loaded = KeywordClusterer.from_results(str(output_path))

            self.assertEqual(loaded.n_clusters, 3)
            self.assertEqual(loaded.algorithm, "kmeans")
            # Loaded model is in fitted state for assignment but doesn't have cluster_labels
            self.assertTrue(loaded._is_fitted)

    def test_from_results_restores_assignment_config(self):
        """Test that assignment config is restored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "clusters.json"
            self.clusterer.save_results(str(output_path))

            loaded = KeywordClusterer.from_results(str(output_path))

            self.assertEqual(loaded._default_assignment_mode, "soft")
            self.assertEqual(loaded._default_threshold, 0.2)
            self.assertEqual(loaded._default_metric, "cosine")

    def test_from_results_restores_topics(self):
        """Test that topics are restored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "clusters.json"
            self.clusterer.save_results(str(output_path), include_topics=True)

            loaded = KeywordClusterer.from_results(str(output_path))

            self.assertEqual(len(loaded.topics), 3)
            for topic_descriptors in loaded.topics.values():
                self.assertEqual(len(topic_descriptors), 5)

    def test_loaded_clusterer_can_assign(self):
        """Test that loaded clusterer can perform assignments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "clusters.json"
            self.clusterer.save_results(str(output_path))

            loaded = KeywordClusterer.from_results(str(output_path))

            test_embedding = np.random.randn(10)
            test_embedding = test_embedding / np.linalg.norm(test_embedding)

            result = loaded.assign(test_embedding)
            self.assertIn("assigned_clusters", result)
            self.assertIn("primary_cluster", result)


class TestConvenienceFunction(unittest.TestCase):
    """Test cluster_keywords_from_embeddings convenience function."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.keyword_embeddings = {
            f"keyword_{i}": np.random.randn(10) / np.linalg.norm(np.random.randn(10))
            for i in range(20)
        }

    def test_cluster_keywords_from_embeddings(self):
        """Test convenience function."""
        clusterer, topics = cluster_keywords_from_embeddings(
            self.keyword_embeddings, n_clusters=3, n_descriptors=5, random_state=42
        )

        self.assertIsInstance(clusterer, KeywordClusterer)
        self.assertEqual(len(topics), 3)
        for descriptors in topics.values():
            self.assertEqual(len(descriptors), 5)

    def test_cluster_keywords_from_embeddings_with_save(self):
        """Test convenience function with saving."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "clusters.json"

            clusterer, topics = cluster_keywords_from_embeddings(
                self.keyword_embeddings,
                n_clusters=3,
                n_descriptors=5,
                output_path=str(output_path),
                random_state=42,
            )

            self.assertTrue(output_path.exists())


class TestComputeDistances(unittest.TestCase):
    """Test distance computation methods."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.keyword_embeddings = {
            f"keyword_{i}": np.random.randn(10) / np.linalg.norm(np.random.randn(10))
            for i in range(20)
        }
        self.clusterer = KeywordClusterer(n_clusters=3, random_state=42)
        self.clusterer.fit(self.keyword_embeddings)

        self.test_embedding = np.random.randn(10)
        self.test_embedding = self.test_embedding / np.linalg.norm(self.test_embedding)

    def test_compute_distances_euclidean(self):
        """Test Euclidean distance computation."""
        distances = self.clusterer.compute_distances(
            self.test_embedding, metric="euclidean"
        )

        self.assertEqual(len(distances), 3)
        self.assertTrue(all(d >= 0 for d in distances))

    def test_compute_distances_cosine(self):
        """Test cosine distance computation."""
        distances = self.clusterer.compute_distances(
            self.test_embedding, metric="cosine"
        )

        self.assertEqual(len(distances), 3)
        self.assertTrue(all(0 <= d <= 2 for d in distances))

    def test_compute_distances_invalid_metric(self):
        """Test distance computation with invalid metric."""
        with self.assertRaises(ValueError):
            self.clusterer.compute_distances(self.test_embedding, metric="invalid")


if __name__ == "__main__":
    unittest.main()
