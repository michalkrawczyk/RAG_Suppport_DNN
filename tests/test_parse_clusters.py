"""Tests for Cluster Parser."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from RAG_supporters.clustering_ops import ClusterParser, parse_clusters


@pytest.fixture
def sample_clustering_json():
    """Create sample KeywordClusterer JSON format."""
    return {
        "metadata": {
            "algorithm": "kmeans",
            "n_clusters": 3,
            "n_keywords": 10,
            "random_state": 42,
            "embedding_dim": 384,
            "assignment_config": {
                "default_mode": "hard",
                "default_threshold": 0.15,
                "default_metric": "cosine",
            },
        },
        "cluster_assignments": {
            "machine learning": 0,
            "deep learning": 0,
            "neural networks": 0,
            "python": 1,
            "java": 1,
            "programming": 1,
            "database": 2,
            "sql": 2,
            "nosql": 2,
            "mongodb": 2,
        },
        "clusters": {
            "0": ["machine learning", "deep learning", "neural networks"],
            "1": ["python", "java", "programming"],
            "2": ["database", "sql", "nosql", "mongodb"],
        },
        "cluster_stats": {
            "0": {
                "size": 3,
                "keywords_sample": ["machine learning", "deep learning", "neural networks"],
                "topic_descriptors": ["machine learning", "deep learning", "AI"],
            },
            "1": {
                "size": 3,
                "keywords_sample": ["python", "java", "programming"],
                "topic_descriptors": ["python", "programming", "languages"],
            },
            "2": {
                "size": 4,
                "keywords_sample": ["database", "sql", "nosql", "mongodb"],
                "topic_descriptors": ["database", "sql", "storage"],
            },
        },
        "centroids": [
            [0.1] * 384,  # Cluster 0 centroid
            [0.2] * 384,  # Cluster 1 centroid
            [0.3] * 384,  # Cluster 2 centroid
        ],
        "embeddings": {
            "machine learning": [0.11] * 384,
            "deep learning": [0.12] * 384,
            "neural networks": [0.13] * 384,
            "python": [0.21] * 384,
            "java": [0.22] * 384,
            "programming": [0.23] * 384,
            "database": [0.31] * 384,
            "sql": [0.32] * 384,
            "nosql": [0.33] * 384,
            "mongodb": [0.34] * 384,
        },
    }


@pytest.fixture
def clustering_json_file(sample_clustering_json, tmp_path):
    """Create temporary clustering JSON file."""
    json_path = tmp_path / "clusters.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sample_clustering_json, f, indent=2)
    return json_path


class TestClusterParserInit:
    """Test ClusterParser initialization."""

    def test_init_valid_file(self, clustering_json_file):
        """Test initialization with valid JSON file."""
        parser = ClusterParser(clustering_json_file)

        assert parser.clustering_data is not None, "Should load clustering data"
        assert parser.clustering_data.n_clusters == 3, "Should have correct number of clusters"
        assert len(parser.keyword_to_cluster) == 10, "Should load all keywords"
        assert len(parser.cluster_keywords) == 3, "Should have all cluster keyword mappings"

    def test_init_file_not_found(self):
        """Test initialization with non-existent file."""
        with pytest.raises(FileNotFoundError, match="Clustering JSON not found"):
            ClusterParser("nonexistent.json")

    def test_init_loads_embeddings(self, clustering_json_file):
        """Test initialization loads embeddings if available."""
        parser = ClusterParser(clustering_json_file)

        assert parser.keyword_embeddings is not None, "Should load embeddings"
        assert len(parser.keyword_embeddings) == 10, "Should have embeddings for all keywords"

        # Check embedding shape
        for emb in parser.keyword_embeddings.values():
            assert emb.shape == (384,), "Embeddings should have correct shape"

    def test_init_without_embeddings(self, sample_clustering_json, tmp_path):
        """Test initialization without embeddings in JSON."""
        # Remove embeddings
        json_data = sample_clustering_json.copy()
        del json_data["embeddings"]

        json_path = tmp_path / "no_embs.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f)

        parser = ClusterParser(json_path)

        assert parser.keyword_embeddings is None, "Should have no embeddings"

    def test_init_custom_cosine_threshold(self, clustering_json_file):
        """Test initialization with custom cosine threshold."""
        parser = ClusterParser(clustering_json_file, cosine_threshold=0.8)

        assert parser.cosine_threshold == 0.8, "Should use custom threshold"


class TestClusterParserKeywordMatching:
    """Test keyword matching functionality."""

    def test_match_keyword_exact_match(self, clustering_json_file):
        """Test exact keyword matching."""
        parser = ClusterParser(clustering_json_file)

        cluster_id = parser.match_keyword("machine learning")

        assert cluster_id == 0, "Should match to cluster 0"

    def test_match_keyword_case_insensitive(self, clustering_json_file):
        """Test case-insensitive keyword matching."""
        parser = ClusterParser(clustering_json_file)

        cluster_id = parser.match_keyword("MACHINE LEARNING")

        assert cluster_id == 0, "Should match case-insensitively"

    def test_match_keyword_with_whitespace(self, clustering_json_file):
        """Test keyword matching with extra whitespace."""
        parser = ClusterParser(clustering_json_file)

        cluster_id = parser.match_keyword("  machine learning  ")

        assert cluster_id == 0, "Should match after stripping whitespace"

    def test_match_keyword_no_match(self, clustering_json_file):
        """Test keyword matching when no match found."""
        parser = ClusterParser(clustering_json_file)

        cluster_id = parser.match_keyword("unknown keyword")

        assert cluster_id is None, "Should return None for unmatched keyword"

    def test_match_keyword_all_keywords(self, clustering_json_file):
        """Test matching all keywords in dataset."""
        parser = ClusterParser(clustering_json_file)

        # Cluster 0
        assert parser.match_keyword("machine learning") == 0, "ML should be in cluster 0"
        assert parser.match_keyword("deep learning") == 0, "DL should be in cluster 0"
        assert parser.match_keyword("neural networks") == 0, "NN should be in cluster 0"

        # Cluster 1
        assert parser.match_keyword("python") == 1, "Python should be in cluster 1"
        assert parser.match_keyword("java") == 1, "Java should be in cluster 1"
        assert parser.match_keyword("programming") == 1, "Programming should be in cluster 1"

        # Cluster 2
        assert parser.match_keyword("database") == 2, "Database should be in cluster 2"
        assert parser.match_keyword("sql") == 2, "SQL should be in cluster 2"
        assert parser.match_keyword("nosql") == 2, "NoSQL should be in cluster 2"
        assert parser.match_keyword("mongodb") == 2, "MongoDB should be in cluster 2"


class TestClusterParserBatchMatching:
    """Test batch keyword matching."""

    def test_match_keywords_batch(self, clustering_json_file):
        """Test batch keyword matching."""
        parser = ClusterParser(clustering_json_file)

        keywords = ["machine learning", "python", "database", "unknown"]
        cluster_ids = parser.match_keywords_batch(keywords)

        assert cluster_ids == [0, 1, 2, None], "Should match all keywords correctly"

    def test_match_keywords_batch_empty(self, clustering_json_file):
        """Test batch matching with empty list."""
        parser = ClusterParser(clustering_json_file)

        cluster_ids = parser.match_keywords_batch([])

        assert cluster_ids == [], "Should return empty list for empty input"


class TestClusterParserFuzzyMatching:
    """Test fuzzy keyword matching."""

    def test_match_keyword_fuzzy_exact_match(self, clustering_json_file):
        """Test fuzzy matching with exact match."""
        parser = ClusterParser(clustering_json_file)

        cluster_id, similarity = parser.match_keyword_fuzzy("machine learning")

        assert cluster_id == 0, "Should match to cluster 0"
        assert similarity == 1.0, "Exact match should have similarity 1.0"

    def test_match_keyword_fuzzy_no_match(self, clustering_json_file):
        """Test fuzzy matching with no match."""
        parser = ClusterParser(clustering_json_file)

        cluster_id, similarity = parser.match_keyword_fuzzy("completely unknown term")

        assert cluster_id is None, "Should not match unknown term"
        assert similarity == 0.0, "No match should have similarity 0.0"

    def test_match_keyword_fuzzy_without_similarity(self, clustering_json_file):
        """Test fuzzy matching without returning similarity."""
        parser = ClusterParser(clustering_json_file)

        cluster_id = parser.match_keyword_fuzzy("machine learning", return_similarity=False)

        assert cluster_id == 0, "Should return cluster ID only"


class TestClusterParserDescriptors:
    """Test cluster descriptor retrieval."""

    def test_get_cluster_descriptors(self, clustering_json_file):
        """Test retrieving cluster descriptors."""
        parser = ClusterParser(clustering_json_file)

        descriptors = parser.get_cluster_descriptors(0)

        assert descriptors is not None, "Should have descriptors for cluster 0"
        assert "machine learning" in descriptors, "Should include ML in descriptors"
        assert len(descriptors) == 3, "Should have 3 descriptors"

    def test_get_cluster_descriptors_all_clusters(self, clustering_json_file):
        """Test retrieving descriptors for all clusters."""
        parser = ClusterParser(clustering_json_file)

        desc_0 = parser.get_cluster_descriptors(0)
        desc_1 = parser.get_cluster_descriptors(1)
        desc_2 = parser.get_cluster_descriptors(2)

        assert desc_0 is not None, "Cluster 0 should have descriptors"
        assert desc_1 is not None, "Cluster 1 should have descriptors"
        assert desc_2 is not None, "Cluster 2 should have descriptors"


class TestClusterParserCentroids:
    """Test centroid retrieval."""

    def test_get_centroid(self, clustering_json_file):
        """Test retrieving cluster centroid."""
        parser = ClusterParser(clustering_json_file)

        centroid = parser.get_centroid(0)

        assert centroid is not None, "Should have centroid for cluster 0"
        assert centroid.shape == (384,), "Centroid should have correct shape"
        assert np.allclose(centroid, 0.1), "Centroid should have expected values"

    def test_get_centroid_all_clusters(self, clustering_json_file):
        """Test retrieving all centroids."""
        parser = ClusterParser(clustering_json_file)

        centroid_0 = parser.get_centroid(0)
        centroid_1 = parser.get_centroid(1)
        centroid_2 = parser.get_centroid(2)

        assert centroid_0 is not None, "Cluster 0 should have centroid"
        assert centroid_1 is not None, "Cluster 1 should have centroid"
        assert centroid_2 is not None, "Cluster 2 should have centroid"

        assert np.allclose(centroid_0, 0.1), "Cluster 0 centroid should be 0.1"
        assert np.allclose(centroid_1, 0.2), "Cluster 1 centroid should be 0.2"
        assert np.allclose(centroid_2, 0.3), "Cluster 2 centroid should be 0.3"


class TestClusterParserClusterKeywords:
    """Test cluster keyword retrieval."""

    def test_get_cluster_keywords(self, clustering_json_file):
        """Test retrieving keywords for a cluster."""
        parser = ClusterParser(clustering_json_file)

        keywords = parser.get_cluster_keywords(0)

        assert keywords is not None, "Should have keywords for cluster 0"
        assert len(keywords) == 3, "Cluster 0 should have 3 keywords"
        assert "machine learning" in keywords, "Should include ML"
        assert "deep learning" in keywords, "Should include DL"

    def test_get_cluster_keywords_all_clusters(self, clustering_json_file):
        """Test retrieving keywords for all clusters."""
        parser = ClusterParser(clustering_json_file)

        kw_0 = parser.get_cluster_keywords(0)
        kw_1 = parser.get_cluster_keywords(1)
        kw_2 = parser.get_cluster_keywords(2)

        assert len(kw_0) == 3, "Cluster 0 should have 3 keywords"
        assert len(kw_1) == 3, "Cluster 1 should have 3 keywords"
        assert len(kw_2) == 4, "Cluster 2 should have 4 keywords"


class TestClusterParserUtilities:
    """Test utility methods."""

    def test_get_all_keywords(self, clustering_json_file):
        """Test retrieving all keywords."""
        parser = ClusterParser(clustering_json_file)

        all_keywords = parser.get_all_keywords()

        assert len(all_keywords) == 10, "Should have 10 unique keywords"
        assert "machine learning" in all_keywords, "Should include ML"
        assert "python" in all_keywords, "Should include Python"

    def test_get_clusters_for_keywords(self, clustering_json_file):
        """Test getting cluster assignments for keywords."""
        parser = ClusterParser(clustering_json_file)

        keywords = ["machine learning", "python", "sql", "unknown"]
        assignments = parser.get_clusters_for_keywords(keywords)

        assert assignments["machine learning"] == 0, "ML should be in cluster 0"
        assert assignments["python"] == 1, "Python should be in cluster 1"
        assert assignments["sql"] == 2, "SQL should be in cluster 2"
        assert assignments["unknown"] is None, "Unknown should be None"

    def test_get_clusters_for_keywords_ignore_missing(self, clustering_json_file):
        """Test getting clusters with ignore_missing=False."""
        parser = ClusterParser(clustering_json_file)

        keywords = ["machine learning", "unknown"]
        assignments = parser.get_clusters_for_keywords(keywords, ignore_missing=False)

        assert "machine learning" in assignments, "Should include matched keyword"
        assert "unknown" not in assignments, "Should exclude unmatched keyword"

    def test_compute_cluster_coverage(self, clustering_json_file):
        """Test computing cluster coverage statistics."""
        parser = ClusterParser(clustering_json_file)

        keywords = ["machine learning", "python", "sql", "unknown"]
        matched, total, ratio, clusters = parser.compute_cluster_coverage(keywords)

        assert matched == 3, "Should match 3 keywords"
        assert total == 4, "Should have 4 total keywords"
        assert ratio == 0.75, "Coverage ratio should be 0.75"
        assert clusters == {0, 1, 2}, "Should cover all 3 clusters"

    def test_compute_cluster_coverage_empty(self, clustering_json_file):
        """Test coverage with empty keyword list."""
        parser = ClusterParser(clustering_json_file)

        matched, total, ratio, clusters = parser.compute_cluster_coverage([])

        assert matched == 0, "Should match 0 keywords"
        assert total == 0, "Should have 0 total keywords"
        assert ratio == 0.0, "Coverage ratio should be 0.0"
        assert len(clusters) == 0, "Should cover 0 clusters"


class TestClusterParserValidation:
    """Test validation functionality."""

    def test_validate_valid_data(self, clustering_json_file):
        """Test validation with valid data."""
        parser = ClusterParser(clustering_json_file)

        # Should not raise
        parser.validate()

    def test_validate_invalid_cluster_id(self, sample_clustering_json, tmp_path):
        """Test validation catches invalid cluster IDs."""
        # Create invalid JSON with cluster ID >= n_clusters
        json_data = sample_clustering_json.copy()
        json_data["cluster_assignments"]["invalid_keyword"] = 999

        json_path = tmp_path / "invalid.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f)

        parser = ClusterParser(json_path)

        with pytest.raises(ValueError, match="Invalid cluster ID"):
            parser.validate()


class TestClusterParserMetadata:
    """Test metadata retrieval."""

    def test_get_metadata(self, clustering_json_file):
        """Test retrieving metadata for config.json."""
        parser = ClusterParser(clustering_json_file)

        metadata = parser.get_metadata()

        assert metadata["n_clusters"] == 3, "Should have correct cluster count"
        assert metadata["n_keywords"] == 10, "Should have correct keyword count"
        assert metadata["embedding_dim"] == 384, "Should have correct embedding dim"
        assert "clustering_source" in metadata, "Should include source path"
        assert metadata["algorithm"] == "kmeans", "Should include algorithm"
        assert metadata["random_state"] == 42, "Should include random state"


class TestConvenienceFunction:
    """Test parse_clusters convenience function."""

    def test_parse_clusters_with_validation(self, clustering_json_file):
        """Test parse_clusters() convenience function with validation."""
        parser = parse_clusters(clustering_json_file, validate=True)

        assert parser is not None, "Should return parser"
        assert parser.clustering_data.n_clusters == 3, "Should have loaded data"

    def test_parse_clusters_without_validation(self, clustering_json_file):
        """Test parse_clusters() without validation."""
        parser = parse_clusters(clustering_json_file, validate=False)

        assert parser is not None, "Should return parser without validation"

    def test_parse_clusters_custom_threshold(self, clustering_json_file):
        """Test parse_clusters() with custom threshold."""
        parser = parse_clusters(clustering_json_file, cosine_threshold=0.85)

        assert parser.cosine_threshold == 0.85, "Should use custom threshold"


class TestClusterParserEdgeCases:
    """Test edge cases and error handling."""

    def test_normalize_keyword_various_formats(self, clustering_json_file):
        """Test keyword normalization with various formats."""
        parser = ClusterParser(clustering_json_file)

        # All should normalize to same value
        keywords = [
            "Machine Learning",
            "MACHINE LEARNING",
            "  machine learning  ",
            "machine learning",
        ]

        normalized_set = {parser._normalize_keyword(kw) for kw in keywords}

        assert len(normalized_set) == 1, "All should normalize to same value"
        assert "machine learning" in normalized_set, "Should normalize to lowercase"

    def test_empty_keyword_list_batch_match(self, clustering_json_file):
        """Test batch matching with empty list."""
        parser = ClusterParser(clustering_json_file)

        result = parser.match_keywords_batch([])

        assert result == [], "Should return empty list"

    def test_get_centroid_invalid_cluster(self, clustering_json_file):
        """Test getting centroid for invalid cluster ID."""
        parser = ClusterParser(clustering_json_file)

        centroid = parser.get_centroid(999)

        assert centroid is None, "Should return None for invalid cluster ID"

    def test_get_cluster_keywords_invalid_cluster(self, clustering_json_file):
        """Test getting keywords for invalid cluster ID."""
        parser = ClusterParser(clustering_json_file)

        keywords = parser.get_cluster_keywords(999)

        assert keywords is None, "Should return None for invalid cluster ID"
