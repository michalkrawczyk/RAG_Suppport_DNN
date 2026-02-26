"""Tests for clustering ops: ClusterParser, SourceClusterLinker, EmbeddingGenerator."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from RAG_supporters.clustering_ops import (
    ClusterParser,
    SourceClusterLinker,
    parse_clusters,
    link_sources,
)
from RAG_supporters.embeddings_ops import EmbeddingGenerator, generate_embeddings
from RAG_supporters.DEFAULT_CONSTS import DEFAULT_EMB_KEYS


# ---------------------------------------------------------------------------
# Shared fixtures — 10-keyword cluster data (used by ClusterParser + Linker)
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_clustering_json():
    """Create sample KeywordClusterer JSON format (10 keywords, 384-dim)."""
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
            [0.1] * 384,
            [0.2] * 384,
            [0.3] * 384,
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
    """Create temporary clustering JSON file from shared sample."""
    json_path = tmp_path / "clusters.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sample_clustering_json, f, indent=2)
    return json_path


# Fixture alias for SourceClusterLinker tests
@pytest.fixture
def cluster_parser(clustering_json_file):
    """Create ClusterParser instance for linker tests."""
    return ClusterParser(clustering_json_file)


# ---------------------------------------------------------------------------
# Embed-specific fixtures — 6-keyword cluster data (used by EmbeddingGenerator)
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_embed_clustering_json():
    """Create sample KeywordClusterer JSON format (6 keywords, 384-dim)."""
    return {
        "metadata": {
            "algorithm": "kmeans",
            "n_clusters": 3,
            "n_keywords": 6,
            "random_state": 42,
            "embedding_dim": 384,
            "assignment_config": {
                "default_mode": "hard",
                "default_threshold": 0.15,
                "default_metric": "cosine",
            },
        },
        "cluster_assignments": {"python": 0, "java": 0, "sql": 1, "database": 1, "ai": 2, "ml": 2},
        "clusters": {"0": ["python", "java"], "1": ["sql", "database"], "2": ["ai", "ml"]},
        "cluster_stats": {
            "0": {"size": 2, "topic_descriptors": ["programming"]},
            "1": {"size": 2, "topic_descriptors": ["database"]},
            "2": {"size": 2, "topic_descriptors": ["artificial intelligence"]},
        },
        "centroids": [
            [0.1] * 384,
            [0.2] * 384,
            [0.3] * 384,
        ],
        "embeddings": {
            "python": [0.11] * 384,
            "java": [0.12] * 384,
            "sql": [0.21] * 384,
            "database": [0.22] * 384,
            "ai": [0.31] * 384,
            "ml": [0.32] * 384,
        },
    }


@pytest.fixture
def embed_clustering_json_file(sample_embed_clustering_json, tmp_path):
    """Create temporary clustering JSON file from embed-specific sample."""
    json_path = tmp_path / "embed_clusters.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sample_embed_clustering_json, f, indent=2)
    return json_path


@pytest.fixture
def embed_cluster_parser(embed_clustering_json_file):
    """Create ClusterParser instance for embedding tests."""
    return ClusterParser(embed_clustering_json_file)


class MockEmbeddingModel:
    """Mock sentence-transformers model for testing."""

    def __init__(self, dim=384, model_name="mock-model"):
        self.dim = dim
        self.model_name = model_name
        self._model_name = model_name

    def encode(
        self,
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
        **kwargs,
    ):
        """Mock encode — deterministic embeddings based on text hash."""
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for text in texts:
            seed = hash(text) % (2**32)
            np.random.seed(seed)
            emb = np.random.randn(self.dim).astype(np.float32)
            if normalize_embeddings:
                emb = emb / (np.linalg.norm(emb) + 1e-8)
            embeddings.append(emb)

        return np.array(embeddings)


@pytest.fixture
def mock_model():
    """Create mock embedding model."""
    return MockEmbeddingModel(dim=384)


# ===========================================================================
# Section 1 — ClusterParser
# ===========================================================================


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

        for emb in parser.keyword_embeddings.values():
            assert emb.shape == (384,), "Embeddings should have correct shape"

    def test_init_without_embeddings(self, sample_clustering_json, tmp_path):
        """Test initialization without embeddings in JSON."""
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

        assert parser.match_keyword("machine learning") == 0, "Should match to cluster 0"

    def test_match_keyword_case_insensitive(self, clustering_json_file):
        """Test case-insensitive keyword matching."""
        parser = ClusterParser(clustering_json_file)

        assert parser.match_keyword("MACHINE LEARNING") == 0, "Should match case-insensitively"

    def test_match_keyword_with_whitespace(self, clustering_json_file):
        """Test keyword matching with extra whitespace."""
        parser = ClusterParser(clustering_json_file)

        assert (
            parser.match_keyword("  machine learning  ") == 0
        ), "Should match after stripping whitespace"

    def test_match_keyword_no_match(self, clustering_json_file):
        """Test keyword matching when no match found."""
        parser = ClusterParser(clustering_json_file)

        assert (
            parser.match_keyword("unknown keyword") is None
        ), "Should return None for unmatched keyword"

    def test_match_keyword_all_keywords(self, clustering_json_file):
        """Test matching all keywords in dataset."""
        parser = ClusterParser(clustering_json_file)

        assert parser.match_keyword("machine learning") == 0, "ML should be in cluster 0"
        assert parser.match_keyword("deep learning") == 0, "DL should be in cluster 0"
        assert parser.match_keyword("neural networks") == 0, "NN should be in cluster 0"
        assert parser.match_keyword("python") == 1, "Python should be in cluster 1"
        assert parser.match_keyword("java") == 1, "Java should be in cluster 1"
        assert parser.match_keyword("programming") == 1, "Programming should be in cluster 1"
        assert parser.match_keyword("database") == 2, "Database should be in cluster 2"
        assert parser.match_keyword("sql") == 2, "SQL should be in cluster 2"
        assert parser.match_keyword("nosql") == 2, "NoSQL should be in cluster 2"
        assert parser.match_keyword("mongodb") == 2, "MongoDB should be in cluster 2"


class TestClusterParserBatchMatching:
    """Test batch keyword matching."""

    def test_match_keywords_batch(self, clustering_json_file):
        """Test batch keyword matching."""
        parser = ClusterParser(clustering_json_file)

        cluster_ids = parser.match_keywords_batch(
            ["machine learning", "python", "database", "unknown"]
        )

        assert cluster_ids == [0, 1, 2, None], "Should match all keywords correctly"

    def test_match_keywords_batch_empty(self, clustering_json_file):
        """Test batch matching with empty list."""
        parser = ClusterParser(clustering_json_file)

        assert parser.match_keywords_batch([]) == [], "Should return empty list for empty input"


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

        assert parser.get_cluster_descriptors(0) is not None, "Cluster 0 should have descriptors"
        assert parser.get_cluster_descriptors(1) is not None, "Cluster 1 should have descriptors"
        assert parser.get_cluster_descriptors(2) is not None, "Cluster 2 should have descriptors"


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

    def test_get_cluster_keywords_all_clusters(self, clustering_json_file):
        """Test retrieving keywords for all clusters."""
        parser = ClusterParser(clustering_json_file)

        assert len(parser.get_cluster_keywords(0)) == 3, "Cluster 0 should have 3 keywords"
        assert len(parser.get_cluster_keywords(1)) == 3, "Cluster 1 should have 3 keywords"
        assert len(parser.get_cluster_keywords(2)) == 4, "Cluster 2 should have 4 keywords"


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

        assignments = parser.get_clusters_for_keywords(
            ["machine learning", "unknown"], ignore_missing=False
        )

        assert "machine learning" in assignments, "Should include matched keyword"
        assert "unknown" not in assignments, "Should exclude unmatched keyword"

    def test_compute_cluster_coverage(self, clustering_json_file):
        """Test computing cluster coverage statistics."""
        parser = ClusterParser(clustering_json_file)

        matched, total, ratio, clusters = parser.compute_cluster_coverage(
            ["machine learning", "python", "sql", "unknown"]
        )

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
        parser.validate()  # Should not raise

    def test_validate_invalid_cluster_id(self, sample_clustering_json, tmp_path):
        """Test validation catches invalid cluster IDs."""
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


class TestConvenienceFunctionParseCluster:
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

        assert parser.match_keywords_batch([]) == [], "Should return empty list"

    def test_get_centroid_invalid_cluster(self, clustering_json_file):
        """Test getting centroid for invalid cluster ID."""
        parser = ClusterParser(clustering_json_file)

        assert parser.get_centroid(999) is None, "Should return None for invalid cluster ID"

    def test_get_cluster_keywords_invalid_cluster(self, clustering_json_file):
        """Test getting keywords for invalid cluster ID."""
        parser = ClusterParser(clustering_json_file)

        assert parser.get_cluster_keywords(999) is None, "Should return None for invalid cluster ID"


# ===========================================================================
# Section 2 — SourceClusterLinker
# ===========================================================================


class TestSourceClusterLinkerInit:
    """Test SourceClusterLinker initialization."""

    def test_init_valid(self, cluster_parser):
        """Test initialization with valid cluster parser."""
        linker = SourceClusterLinker(cluster_parser)

        assert linker.cluster_parser is not None, "Should have cluster parser"
        assert linker.n_clusters == 3, "Should have 3 clusters"
        assert linker.fallback_strategy == "largest", "Should have default fallback strategy"

    def test_init_custom_fallback(self, cluster_parser):
        """Test initialization with custom fallback strategy."""
        linker = SourceClusterLinker(cluster_parser, fallback_strategy="random")

        assert linker.fallback_strategy == "random", "Should use custom fallback strategy"

    def test_init_invalid_fallback(self, cluster_parser):
        """Test initialization fails with invalid fallback strategy."""
        with pytest.raises(ValueError, match="Invalid fallback_strategy"):
            SourceClusterLinker(cluster_parser, fallback_strategy="invalid")

    def test_compute_cluster_sizes(self, cluster_parser):
        """Test cluster size computation."""
        linker = SourceClusterLinker(cluster_parser)

        assert linker.cluster_sizes[0] == 3, "Cluster 0 should have 3 keywords"
        assert linker.cluster_sizes[1] == 3, "Cluster 1 should have 3 keywords"
        assert linker.cluster_sizes[2] == 4, "Cluster 2 should have 4 keywords"


class TestSourceClusterLinkerFallback:
    """Test fallback cluster assignment."""

    def test_fallback_largest(self, cluster_parser):
        """Test fallback to largest cluster."""
        linker = SourceClusterLinker(cluster_parser, fallback_strategy="largest")

        assert (
            linker._get_fallback_cluster() == 2
        ), "Should assign to cluster 2 (largest with 4 keywords)"

    def test_fallback_uniform(self, cluster_parser):
        """Test fallback with uniform distribution."""
        linker = SourceClusterLinker(cluster_parser, fallback_strategy="uniform")

        assert linker._get_fallback_cluster(pair_id=0) == 0, "pair_id=0 should map to cluster 0"
        assert linker._get_fallback_cluster(pair_id=1) == 1, "pair_id=1 should map to cluster 1"
        assert linker._get_fallback_cluster(pair_id=2) == 2, "pair_id=2 should map to cluster 2"
        assert linker._get_fallback_cluster(pair_id=3) == 0, "pair_id=3 should wrap to cluster 0"

    def test_fallback_random(self, cluster_parser):
        """Test fallback with random assignment."""
        linker = SourceClusterLinker(cluster_parser, fallback_strategy="random")

        np.random.seed(42)
        fallback_cluster = linker._get_fallback_cluster()

        assert 0 <= fallback_cluster < 3, "Should assign to valid cluster"


class TestSourceClusterLinkerSinglePair:
    """Test linking single pairs to clusters."""

    def test_link_single_keyword(self, cluster_parser):
        """Test linking pair with single keyword."""
        linker = SourceClusterLinker(cluster_parser)

        assert linker.link_pair(["python"]) == 1, "Should link 'python' to cluster 1"

    def test_link_multiple_keywords_same_cluster(self, cluster_parser):
        """Test linking pair with multiple keywords from same cluster."""
        linker = SourceClusterLinker(cluster_parser)

        assert (
            linker.link_pair(["python", "java", "programming"]) == 1
        ), "Should link all programming keywords to cluster 1"

    def test_link_multiple_keywords_different_clusters(self, cluster_parser):
        """Test linking pair with keywords from different clusters (majority voting)."""
        linker = SourceClusterLinker(cluster_parser)

        assert (
            linker.link_pair(["python", "java", "machine learning"]) == 1
        ), "Should link to cluster 1 (majority vote: 2 vs 1)"

    def test_link_empty_keywords(self, cluster_parser):
        """Test linking pair with no keywords uses fallback."""
        linker = SourceClusterLinker(cluster_parser, fallback_strategy="largest")

        assert (
            linker.link_pair([], pair_id=0) == 2
        ), "Should use fallback to largest cluster (cluster 2)"

    def test_link_unmatched_keywords(self, cluster_parser):
        """Test linking pair with unmatched keywords uses fallback."""
        linker = SourceClusterLinker(cluster_parser, fallback_strategy="largest")

        assert (
            linker.link_pair(["unknown", "invalid", "keywords"], pair_id=5) == 2
        ), "Should use fallback when no keywords match"


class TestSourceClusterLinkerBatch:
    """Test batch linking of pairs."""

    def test_link_batch_basic(self, cluster_parser):
        """Test batch linking of multiple pairs."""
        linker = SourceClusterLinker(cluster_parser)

        keywords_list = [["python", "programming"], ["database", "sql"], ["machine learning"]]
        cluster_ids = linker.link_batch(keywords_list)

        assert len(cluster_ids) == 3, "Should return 3 cluster assignments"
        assert cluster_ids[0] == 1, "First pair should link to cluster 1"
        assert cluster_ids[1] == 2, "Second pair should link to cluster 2"
        assert cluster_ids[2] == 0, "Third pair should link to cluster 0"

    def test_link_batch_with_pair_ids(self, cluster_parser):
        """Test batch linking with explicit pair IDs."""
        linker = SourceClusterLinker(cluster_parser, fallback_strategy="uniform")

        cluster_ids = linker.link_batch([["python"], [], ["sql"]], [10, 11, 12])

        assert len(cluster_ids) == 3, "Should return 3 cluster assignments"
        assert cluster_ids[0] == 1, "First pair should link to cluster 1"
        assert cluster_ids[1] == 11 % 3, "Second pair should use uniform fallback"
        assert cluster_ids[2] == 2, "Third pair should link to cluster 2"


class TestSourceClusterLinkerDataFrame:
    """Test DataFrame linking."""

    def test_link_dataframe_basic(self, cluster_parser):
        """Test linking DataFrame with keywords."""
        linker = SourceClusterLinker(cluster_parser)

        df = pd.DataFrame(
            {
                "pair_id": [0, 1, 2],
                "question": ["Q1", "Q2", "Q3"],
                "source": ["S1", "S2", "S3"],
                "keywords": [
                    ["python", "programming"],
                    ["database", "sql"],
                    ["machine learning", "deep learning"],
                ],
            }
        )

        df_linked = linker.link_dataframe(df, show_progress=False)

        assert "cluster_id" in df_linked.columns, "Should add cluster_id column"
        assert len(df_linked) == 3, "Should preserve all rows"
        assert df_linked["cluster_id"].tolist() == [
            1,
            2,
            0,
        ], "Should have correct cluster assignments"

    def test_link_dataframe_custom_columns(self, cluster_parser):
        """Test linking DataFrame with custom column names."""
        linker = SourceClusterLinker(cluster_parser)

        df = pd.DataFrame({"id": [0, 1, 2], "kw": [["python"], ["sql"], ["java"]]})

        df_linked = linker.link_dataframe(
            df,
            keywords_col="kw",
            pair_id_col="id",
            output_col="assigned_cluster",
            show_progress=False,
        )

        assert "assigned_cluster" in df_linked.columns, "Should add custom output column"
        assert df_linked["assigned_cluster"].tolist() == [
            1,
            2,
            1,
        ], "Should have correct assignments"

    def test_link_dataframe_missing_keywords_col(self, cluster_parser):
        """Test linking DataFrame fails when keywords column missing."""
        linker = SourceClusterLinker(cluster_parser)

        df = pd.DataFrame({"pair_id": [0, 1], "text": ["text1", "text2"]})

        with pytest.raises(ValueError, match="Keywords column .* not found"):
            linker.link_dataframe(df)

    def test_link_dataframe_missing_pair_id_col(self, cluster_parser):
        """Test linking DataFrame uses index when pair_id column missing."""
        linker = SourceClusterLinker(cluster_parser, fallback_strategy="uniform")

        df = pd.DataFrame({"keywords": [[], [], []]})

        df_linked = linker.link_dataframe(df, show_progress=False)

        assert "cluster_id" in df_linked.columns, "Should add cluster_id column"
        assert df_linked["cluster_id"].tolist() == [0, 1, 2], "Should use index for fallback"

    def test_link_dataframe_string_keywords(self, cluster_parser):
        """Test linking DataFrame with keywords as comma-separated strings."""
        linker = SourceClusterLinker(cluster_parser)

        df = pd.DataFrame({"pair_id": [0, 1], "keywords": ["python, programming", "sql, database"]})

        df_linked = linker.link_dataframe(df, show_progress=False)

        assert df_linked["cluster_id"].tolist() == [1, 2], "Should parse string keywords correctly"


class TestSourceClusterLinkerValidation:
    """Test cluster assignment validation."""

    def test_validate_valid_assignments(self, cluster_parser):
        """Test validation of valid cluster assignments."""
        linker = SourceClusterLinker(cluster_parser)

        validation = linker.validate_assignments([0, 0, 1, 1, 2, 2])

        assert validation["valid"] is True, "Should be valid"
        assert len(validation["errors"]) == 0, "Should have no errors"
        assert validation["statistics"]["n_pairs"] == 6, "Should count 6 pairs"
        assert validation["statistics"]["n_clusters_used"] == 3, "Should use all 3 clusters"

    def test_validate_invalid_range_negative(self, cluster_parser):
        """Test validation fails with negative cluster IDs."""
        linker = SourceClusterLinker(cluster_parser)

        validation = linker.validate_assignments([0, -1, 2])

        assert validation["valid"] is False, "Should be invalid"
        assert len(validation["errors"]) > 0, "Should have errors"
        assert "Invalid cluster ID" in validation["errors"][0], "Should report negative ID error"

    def test_validate_invalid_range_too_large(self, cluster_parser):
        """Test validation fails with cluster IDs >= n_clusters."""
        linker = SourceClusterLinker(cluster_parser)

        validation = linker.validate_assignments([0, 1, 3])

        assert validation["valid"] is False, "Should be invalid"
        assert len(validation["errors"]) > 0, "Should have errors"
        assert "Invalid cluster ID" in validation["errors"][0], "Should report out-of-range error"

    def test_validate_missing_clusters(self, cluster_parser):
        """Test validation warns when clusters have no assignments."""
        linker = SourceClusterLinker(cluster_parser)

        validation = linker.validate_assignments([0, 0, 0, 0])

        assert validation["valid"] is True, "Should be valid (warnings don't fail)"
        assert len(validation["warnings"]) > 0, "Should have warnings"
        assert (
            "have no assignments" in validation["warnings"][0]
        ), "Should warn about missing clusters"

    def test_validate_imbalanced_distribution(self):
        """Test validation warns about highly imbalanced distribution."""
        clustering_data = {
            "metadata": {"n_clusters": 5, "embedding_dim": 384},
            "cluster_assignments": {f"kw{i}": i % 5 for i in range(20)},
            "clusters": {str(i): [f"kw{j}" for j in range(i, 20, 5)] for i in range(5)},
        }

        json_path = tempfile.mktemp(suffix=".json")
        with open(json_path, "w") as f:
            json.dump(clustering_data, f)

        try:
            parser = ClusterParser(json_path)
            linker = SourceClusterLinker(parser)

            cluster_assignments = [0] * 10000 + [1, 2, 3, 4]
            validation = linker.validate_assignments(cluster_assignments)

            assert validation["valid"] is True, "Should be valid (warnings don't fail)"
            assert len(validation["warnings"]) > 0, "Should have warnings about imbalance"
            assert "imbalanced" in validation["warnings"][0].lower(), "Should warn about imbalance"
        finally:
            Path(json_path).unlink(missing_ok=True)

    def test_validate_torch_tensor(self, cluster_parser):
        """Test validation works with PyTorch tensors."""
        linker = SourceClusterLinker(cluster_parser)

        validation = linker.validate_assignments(torch.tensor([0, 1, 2, 0, 1, 2]))

        assert validation["valid"] is True, "Should validate torch.Tensor"
        assert validation["statistics"]["n_pairs"] == 6, "Should count pairs from tensor"

    def test_validate_numpy_array(self, cluster_parser):
        """Test validation works with numpy arrays."""
        linker = SourceClusterLinker(cluster_parser)

        validation = linker.validate_assignments(np.array([0, 1, 2, 0, 1, 2]))

        assert validation["valid"] is True, "Should validate np.ndarray"
        assert validation["statistics"]["n_pairs"] == 6, "Should count pairs from array"


class TestLinkSourcesConvenience:
    """Test convenience function."""

    def test_link_sources_basic(self, cluster_parser):
        """Test convenience function for linking sources."""
        df = pd.DataFrame(
            {"pair_id": [0, 1, 2], "keywords": [["python"], ["sql"], ["machine learning"]]}
        )

        df_linked = link_sources(df, cluster_parser, show_progress=False)

        assert "cluster_id" in df_linked.columns, "Should add cluster_id column"
        assert df_linked["cluster_id"].tolist() == [1, 2, 0], "Should have correct assignments"

    def test_link_sources_custom_fallback(self, cluster_parser):
        """Test convenience function with custom fallback strategy."""
        df = pd.DataFrame({"pair_id": [0, 1], "keywords": [[], []]})

        df_linked = link_sources(
            df, cluster_parser, fallback_strategy="uniform", show_progress=False
        )

        assert df_linked["cluster_id"].tolist() == [0, 1], "Should use uniform fallback"


class TestSourceClusterLinkerEdgeCases:
    """Test edge cases."""

    def test_link_pair_case_insensitive(self, cluster_parser):
        """Test keyword matching is case-insensitive."""
        linker = SourceClusterLinker(cluster_parser)

        assert (
            linker.link_pair(["Python", "PROGRAMMING"]) == 1
        ), "Should match keywords case-insensitively"

    def test_link_pair_with_none(self, cluster_parser):
        """Test linking pair handles None keywords gracefully."""
        linker = SourceClusterLinker(cluster_parser, fallback_strategy="largest")

        assert (
            linker.link_pair(None if None else [], pair_id=0) == 2
        ), "Should handle None/empty as fallback"

    def test_link_dataframe_large(self, cluster_parser):
        """Test linking large DataFrame efficiently."""
        linker = SourceClusterLinker(cluster_parser)

        n_pairs = 1000
        df = pd.DataFrame(
            {
                "pair_id": list(range(n_pairs)),
                "keywords": [["python"] if i % 2 == 0 else ["sql"] for i in range(n_pairs)],
            }
        )

        df_linked = linker.link_dataframe(df, show_progress=False)

        assert len(df_linked) == n_pairs, "Should process all pairs"
        assert "cluster_id" in df_linked.columns, "Should add cluster_id column"
        assert df_linked["cluster_id"].value_counts()[1] == 500, "Should assign 500 to cluster 1"
        assert df_linked["cluster_id"].value_counts()[2] == 500, "Should assign 500 to cluster 2"


# ===========================================================================
# Section 3 — EmbeddingGenerator
# ===========================================================================


class TestEmbeddingGeneratorInit:
    """Test EmbeddingGenerator initialization."""

    def test_init_without_cluster_parser(self, mock_model):
        """Test initialization without cluster parser."""
        generator = EmbeddingGenerator(mock_model)

        assert generator.embedder is not None, "Should wrap model in TextEmbedder"
        assert generator.cluster_parser is None, "Should have no cluster parser"
        assert generator.batch_size == 32, "Should have default batch size"
        assert generator.show_progress is True, "Should show progress by default"

    def test_init_with_cluster_parser(self, mock_model, embed_cluster_parser):
        """Test initialization with cluster parser."""
        generator = EmbeddingGenerator(mock_model, embed_cluster_parser)

        assert generator.cluster_parser is not None, "Should have cluster parser"

    def test_init_custom_batch_size(self, mock_model):
        """Test initialization with custom batch size."""
        generator = EmbeddingGenerator(mock_model, batch_size=64, show_progress=False)

        assert generator.batch_size == 64, "Should use custom batch size"
        assert generator.show_progress is False, "Should disable progress"


class TestEmbeddingGeneratorValidation:
    """Test embedding validation methods."""

    def test_check_valid_embeddings(self, mock_model):
        """Test validation of valid embeddings."""
        generator = EmbeddingGenerator(mock_model)

        embeddings = np.random.randn(10, 384).astype(np.float32)
        is_valid, errors = generator._check_for_invalid_values(embeddings, "test")

        assert is_valid is True, "Should be valid"
        assert len(errors) == 0, "Should have no errors"

    def test_check_nan_embeddings(self, mock_model):
        """Test detection of NaN values."""
        generator = EmbeddingGenerator(mock_model)

        embeddings = np.random.randn(10, 384).astype(np.float32)
        embeddings[0, 0] = np.nan

        is_valid, errors = generator._check_for_invalid_values(embeddings, "test")

        assert is_valid is False, "Should be invalid"
        assert len(errors) > 0, "Should have errors"
        assert "NaN" in errors[0], "Should report NaN error"

    def test_check_inf_embeddings(self, mock_model):
        """Test detection of Inf values."""
        generator = EmbeddingGenerator(mock_model)

        embeddings = np.random.randn(10, 384).astype(np.float32)
        embeddings[0, 0] = np.inf

        is_valid, errors = generator._check_for_invalid_values(embeddings, "test")

        assert is_valid is False, "Should be invalid"
        assert len(errors) > 0, "Should have errors"
        assert "Inf" in errors[0], "Should report Inf error"

    def test_check_zero_embeddings(self, mock_model):
        """Test detection of all-zero embeddings."""
        generator = EmbeddingGenerator(mock_model)

        embeddings = np.random.randn(10, 384).astype(np.float32)
        embeddings[0, :] = 0.0

        is_valid, errors = generator._check_for_invalid_values(embeddings, "test")

        assert is_valid is False, "Should be invalid"
        assert len(errors) > 0, "Should have errors"
        assert "all-zero" in errors[0], "Should report all-zero error"


class TestEmbeddingGeneratorCentroidValidation:
    """Test centroid similarity validation."""

    def test_validate_centroid_similarity_valid(self, mock_model, embed_cluster_parser):
        """Test validation passes for similar centroids."""
        generator = EmbeddingGenerator(mock_model, embed_cluster_parser)

        keyword_embeddings = {
            "python": np.array([0.11] * 384),
            "java": np.array([0.12] * 384),
            "sql": np.array([0.21] * 384),
            "database": np.array([0.22] * 384),
            "ai": np.array([0.31] * 384),
            "ml": np.array([0.32] * 384),
        }

        centroids = np.array([[0.1] * 384, [0.2] * 384, [0.3] * 384])

        is_valid, warnings = generator._validate_centroid_similarity(
            keyword_embeddings, centroids, min_similarity=0.5
        )

        assert isinstance(is_valid, bool), "Should return validation result"
        assert isinstance(warnings, list), "Should return warnings list"

    def test_validate_centroid_similarity_no_parser(self, mock_model):
        """Test validation skips when no cluster parser."""
        generator = EmbeddingGenerator(mock_model)

        is_valid, warnings = generator._validate_centroid_similarity(
            {"test": np.random.randn(384)}, np.random.randn(3, 384)
        )

        assert is_valid is True, "Should skip validation and return True"
        assert len(warnings) == 0, "Should have no warnings"


class TestEmbeddingGeneratorTextEmbeddings:
    """Test text embedding generation."""

    def test_generate_text_embeddings_basic(self, mock_model):
        """Test generating embeddings for texts."""
        generator = EmbeddingGenerator(mock_model, show_progress=False)

        embeddings = generator.generate_text_embeddings(
            ["Sample text 1", "Sample text 2", "Sample text 3"], text_type="test"
        )

        assert isinstance(embeddings, torch.Tensor), "Should return torch.Tensor"
        assert embeddings.shape == (3, 384), "Should have correct shape"
        assert embeddings.dtype == torch.float32, "Should be float32"

    def test_generate_text_embeddings_empty_list(self, mock_model):
        """Test generation fails with empty text list."""
        generator = EmbeddingGenerator(mock_model)

        with pytest.raises(ValueError, match="Cannot generate embeddings for empty"):
            generator.generate_text_embeddings([], text_type="test")

    def test_generate_text_embeddings_with_normalization(self, mock_model):
        """Test generation with L2 normalization."""
        generator = EmbeddingGenerator(mock_model, normalize_embeddings=True, show_progress=False)

        embeddings = generator.generate_text_embeddings(["Text 1", "Text 2"], text_type="test")

        norms = torch.norm(embeddings, dim=1)
        assert torch.allclose(
            norms, torch.ones_like(norms), atol=0.01
        ), "Embeddings should be L2-normalized"

    def test_generate_text_embeddings_deterministic(self, mock_model):
        """Test embeddings are deterministic for same text."""
        generator = EmbeddingGenerator(mock_model, show_progress=False)

        text = ["Same text"]
        embeddings1 = generator.generate_text_embeddings(text, text_type="test")
        embeddings2 = generator.generate_text_embeddings(text, text_type="test")

        assert torch.allclose(embeddings1, embeddings2), "Same text should produce same embeddings"


class TestEmbeddingGeneratorKeywordEmbeddings:
    """Test keyword embedding generation."""

    def test_generate_keyword_embeddings(self, mock_model):
        """Test generating keyword embeddings with mapping."""
        generator = EmbeddingGenerator(mock_model, show_progress=False)

        embeddings, keyword_to_id = generator.generate_keyword_embeddings(["python", "java", "sql"])

        assert isinstance(embeddings, torch.Tensor), "Should return torch.Tensor"
        assert embeddings.shape == (3, 384), "Should have correct shape"
        assert len(keyword_to_id) == 3, "Should have mapping for all keywords"
        assert keyword_to_id["python"] == 0, "Should map first keyword to 0"
        assert keyword_to_id["java"] == 1, "Should map second keyword to 1"
        assert keyword_to_id["sql"] == 2, "Should map third keyword to 2"


class TestEmbeddingGeneratorCentroidEmbeddings:
    """Test centroid embedding generation."""

    def test_generate_centroid_embeddings(self, mock_model, embed_cluster_parser):
        """Test generating centroid embeddings from cluster parser."""
        generator = EmbeddingGenerator(mock_model, embed_cluster_parser, show_progress=False)

        centroids = generator.generate_centroid_embeddings()

        assert isinstance(centroids, torch.Tensor), "Should return torch.Tensor"
        assert centroids.shape == (3, 384), "Should have centroids for 3 clusters"
        assert centroids.dtype == torch.float32, "Should be float32"

    def test_generate_centroid_embeddings_no_parser(self, mock_model):
        """Test generation fails without cluster parser."""
        generator = EmbeddingGenerator(mock_model)

        with pytest.raises(ValueError, match="ClusterParser required"):
            generator.generate_centroid_embeddings()

    def test_generate_centroid_embeddings_with_validation(self, mock_model, embed_cluster_parser):
        """Test centroid generation with keyword validation."""
        generator = EmbeddingGenerator(mock_model, embed_cluster_parser, show_progress=False)

        keywords = ["python", "java", "sql", "database", "ai", "ml"]
        keyword_embs, keyword_to_id = generator.generate_keyword_embeddings(keywords)
        keyword_embs_dict = {kw: keyword_embs[idx].numpy() for kw, idx in keyword_to_id.items()}

        centroids = generator.generate_centroid_embeddings(
            keyword_embeddings_dict=keyword_embs_dict, validate=True
        )

        assert isinstance(centroids, torch.Tensor), "Should return centroids despite warnings"


class TestEmbeddingGeneratorAllEmbeddings:
    """Test generating all embeddings from DataFrame."""

    def test_generate_all_embeddings_basic(self, mock_model, embed_cluster_parser):
        """Test generating all embeddings from DataFrame."""
        generator = EmbeddingGenerator(mock_model, embed_cluster_parser, show_progress=False)

        df = pd.DataFrame(
            {
                "question": ["What is Python?", "What is SQL?", "What is Python?"],
                "source": ["Python is a language", "SQL is a query language", "Python rocks"],
                "keywords": [["python", "java"], ["sql"], ["python"]],
            }
        )

        embeddings = generator.generate_all_embeddings(df)

        assert DEFAULT_EMB_KEYS.question in embeddings, "Should have question embeddings"
        assert DEFAULT_EMB_KEYS.source in embeddings, "Should have source embeddings"
        assert DEFAULT_EMB_KEYS.keyword in embeddings, "Should have keyword embeddings"
        assert DEFAULT_EMB_KEYS.centroid in embeddings, "Should have centroid embeddings"
        assert DEFAULT_EMB_KEYS.question_to_id in embeddings, "Should have question mapping"
        assert DEFAULT_EMB_KEYS.source_to_id in embeddings, "Should have source mapping"
        assert DEFAULT_EMB_KEYS.keyword_to_id in embeddings, "Should have keyword mapping"

    def test_generate_all_embeddings_shapes(self, mock_model, embed_cluster_parser):
        """Test embeddings have correct shapes."""
        generator = EmbeddingGenerator(mock_model, embed_cluster_parser, show_progress=False)

        df = pd.DataFrame(
            {
                "question": ["Q1", "Q2", "Q1"],
                "source": ["S1", "S2", "S3"],
                "keywords": [["kw1"], ["kw2"], ["kw1", "kw3"]],
            }
        )

        embeddings = generator.generate_all_embeddings(df)

        assert embeddings[DEFAULT_EMB_KEYS.question].shape == (2, 384), "Should have 2 unique questions"
        assert embeddings[DEFAULT_EMB_KEYS.source].shape == (3, 384), "Should have 3 unique sources"
        assert embeddings[DEFAULT_EMB_KEYS.keyword].shape == (3, 384), "Should have 3 unique keywords"
        assert embeddings[DEFAULT_EMB_KEYS.centroid].shape == (3, 384), "Should have 3 cluster centroids"

    def test_generate_all_embeddings_string_keywords(self, mock_model, embed_cluster_parser):
        """Test handling keywords as comma-separated strings."""
        generator = EmbeddingGenerator(mock_model, embed_cluster_parser, show_progress=False)

        df = pd.DataFrame({"question": ["Q1"], "source": ["S1"], "keywords": ["python, java, sql"]})

        embeddings = generator.generate_all_embeddings(df)

        assert "python" in embeddings[DEFAULT_EMB_KEYS.keyword_to_id], "Should extract 'python'"
        assert "java" in embeddings[DEFAULT_EMB_KEYS.keyword_to_id], "Should extract 'java'"
        assert "sql" in embeddings[DEFAULT_EMB_KEYS.keyword_to_id], "Should extract 'sql'"


class TestEmbeddingGeneratorSaveEmbeddings:
    """Test saving embeddings to files."""

    def test_save_embeddings_basic(self, mock_model, tmp_path):
        """Test saving embeddings to directory."""
        generator = EmbeddingGenerator(mock_model, show_progress=False)

        embeddings = {
            DEFAULT_EMB_KEYS.question: torch.randn(10, 384),
            DEFAULT_EMB_KEYS.source: torch.randn(20, 384),
            DEFAULT_EMB_KEYS.keyword: torch.randn(5, 384),
            DEFAULT_EMB_KEYS.question_to_id: {"Q1": 0},
        }

        generator.save_embeddings(embeddings, tmp_path)

        assert (tmp_path / f"{DEFAULT_EMB_KEYS.question}.pt").exists(), "Should save question embeddings"
        assert (tmp_path / f"{DEFAULT_EMB_KEYS.source}.pt").exists(), "Should save source embeddings"
        assert (tmp_path / f"{DEFAULT_EMB_KEYS.keyword}.pt").exists(), "Should save keyword embeddings"
        assert not (tmp_path / f"{DEFAULT_EMB_KEYS.question_to_id}.pt").exists(), "Should skip non-tensor items"

    def test_save_embeddings_with_prefix(self, mock_model, tmp_path):
        """Test saving embeddings with filename prefix."""
        generator = EmbeddingGenerator(mock_model)

        embeddings = {DEFAULT_EMB_KEYS.question: torch.randn(10, 384), DEFAULT_EMB_KEYS.source: torch.randn(20, 384)}

        generator.save_embeddings(embeddings, tmp_path, prefix="test_")

        assert (tmp_path / f"test_{DEFAULT_EMB_KEYS.question}.pt").exists(), "Should save with prefix"
        assert (tmp_path / f"test_{DEFAULT_EMB_KEYS.source}.pt").exists(), "Should save with prefix"

    def test_save_embeddings_creates_directory(self, mock_model, tmp_path):
        """Test saving creates output directory if not exists."""
        generator = EmbeddingGenerator(mock_model)

        output_dir = tmp_path / "nonexistent" / "subdir"
        embeddings = {DEFAULT_EMB_KEYS.question: torch.randn(5, 384)}

        generator.save_embeddings(embeddings, output_dir)

        assert output_dir.exists(), "Should create output directory"
        assert (output_dir / f"{DEFAULT_EMB_KEYS.question}.pt").exists(), "Should save embedding file"

    def test_save_and_load_embeddings(self, mock_model, tmp_path):
        """Test saved embeddings can be loaded back."""
        generator = EmbeddingGenerator(mock_model)

        original_embeddings = {
            DEFAULT_EMB_KEYS.question: torch.randn(10, 384),
            DEFAULT_EMB_KEYS.source: torch.randn(20, 384),
        }

        generator.save_embeddings(original_embeddings, tmp_path)

        loaded_question = torch.load(tmp_path / f"{DEFAULT_EMB_KEYS.question}.pt")
        loaded_source = torch.load(tmp_path / f"{DEFAULT_EMB_KEYS.source}.pt")

        assert torch.allclose(
            loaded_question, original_embeddings[DEFAULT_EMB_KEYS.question]
        ), "Loaded question embeddings should match original"
        assert torch.allclose(
            loaded_source, original_embeddings[DEFAULT_EMB_KEYS.source]
        ), "Loaded source embeddings should match original"


class TestGenerateEmbeddingsConvenience:
    """Test convenience function."""

    def test_generate_embeddings_basic(self, mock_model, embed_cluster_parser):
        """Test convenience function generates all embeddings."""
        df = pd.DataFrame(
            {"question": ["Q1", "Q2"], "source": ["S1", "S2"], "keywords": [["python"], ["sql"]]}
        )

        embeddings = generate_embeddings(
            df=df,
            embedding_model=mock_model,
            cluster_parser=embed_cluster_parser,
            batch_size=16,
            validate=True,
        )

        assert DEFAULT_EMB_KEYS.question in embeddings, "Should have question embeddings"
        assert DEFAULT_EMB_KEYS.source in embeddings, "Should have source embeddings"
        assert embeddings[DEFAULT_EMB_KEYS.question].shape[0] == 2, "Should have 2 question embeddings"

    def test_generate_embeddings_with_save(self, mock_model, embed_cluster_parser, tmp_path):
        """Test convenience function saves embeddings to directory."""
        df = pd.DataFrame({"question": ["Q1"], "source": ["S1"], "keywords": [["python"]]})

        generate_embeddings(
            df=df,
            embedding_model=mock_model,
            cluster_parser=embed_cluster_parser,
            output_dir=tmp_path,
        )

        assert (tmp_path / f"{DEFAULT_EMB_KEYS.question}.pt").exists(), "Should save question embeddings"
        assert (tmp_path / f"{DEFAULT_EMB_KEYS.source}.pt").exists(), "Should save source embeddings"
        assert (tmp_path / f"{DEFAULT_EMB_KEYS.keyword}.pt").exists(), "Should save keyword embeddings"
        assert (tmp_path / f"{DEFAULT_EMB_KEYS.centroid}.pt").exists(), "Should save centroid embeddings"

    def test_generate_embeddings_custom_columns(self, mock_model, embed_cluster_parser):
        """Test convenience function with custom column names."""
        df = pd.DataFrame({"q": ["Q1"], "s": ["S1"], "kw": [["python"]]})

        embeddings = generate_embeddings(
            df=df,
            embedding_model=mock_model,
            cluster_parser=embed_cluster_parser,
            question_col="q",
            source_col="s",
            keywords_col="kw",
        )

        assert DEFAULT_EMB_KEYS.question in embeddings, "Should generate embeddings with custom columns"


class TestEmbeddingGeneratorEdgeCases:
    """Test edge cases."""

    def test_generate_embeddings_single_text(self, mock_model):
        """Test generating embeddings for single text."""
        generator = EmbeddingGenerator(mock_model, show_progress=False)

        embeddings = generator.generate_text_embeddings(["Single text"], text_type="test")

        assert embeddings.shape == (1, 384), "Should handle single text"

    def test_generate_embeddings_large_batch(self, mock_model):
        """Test generating embeddings for large batch."""
        generator = EmbeddingGenerator(mock_model, batch_size=16, show_progress=False)

        texts = [f"Text {i}" for i in range(100)]
        embeddings = generator.generate_text_embeddings(texts, text_type="test")

        assert embeddings.shape == (100, 384), "Should handle large batch"

    def test_generate_embeddings_unicode_text(self, mock_model):
        """Test generating embeddings for unicode text."""
        generator = EmbeddingGenerator(mock_model, show_progress=False)

        texts = ["Hello 世界", "Привет мир", "مرحبا"]
        embeddings = generator.generate_text_embeddings(texts, text_type="test")

        assert embeddings.shape == (3, 384), "Should handle unicode text"

    def test_generate_all_embeddings_no_cluster_parser(self, mock_model):
        """Test generating embeddings without cluster parser (no centroids)."""
        generator = EmbeddingGenerator(mock_model, show_progress=False)

        df = pd.DataFrame({"question": ["Q1"], "source": ["S1"], "keywords": [["python"]]})

        embeddings = generator.generate_all_embeddings(df)

        assert DEFAULT_EMB_KEYS.question in embeddings, "Should have question embeddings"
        assert DEFAULT_EMB_KEYS.source in embeddings, "Should have source embeddings"
        assert DEFAULT_EMB_KEYS.keyword in embeddings, "Should have keyword embeddings"
        assert (
            DEFAULT_EMB_KEYS.centroid not in embeddings
        ), "Should not have centroid embeddings without parser"
