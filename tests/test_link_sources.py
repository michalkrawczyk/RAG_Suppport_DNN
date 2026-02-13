"""Tests for Source-Cluster Linker."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from RAG_supporters.clustering_ops import (
    SourceClusterLinker,
    link_sources,
)
from RAG_supporters.clustering_ops import ClusterParser


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
                "default_metric": "cosine"
            }
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
            "mongodb": 2
        },
        "clusters": {
            "0": ["machine learning", "deep learning", "neural networks"],
            "1": ["python", "java", "programming"],
            "2": ["database", "sql", "nosql", "mongodb"]
        },
        "cluster_stats": {
            "0": {
                "size": 3,
                "keywords_sample": ["machine learning", "deep learning", "neural networks"],
                "topic_descriptors": ["machine learning", "deep learning", "AI"]
            },
            "1": {
                "size": 3,
                "keywords_sample": ["python", "java", "programming"],
                "topic_descriptors": ["python", "programming", "languages"]
            },
            "2": {
                "size": 4,
                "keywords_sample": ["database", "sql", "nosql", "mongodb"],
                "topic_descriptors": ["database", "sql", "storage"]
            }
        },
        "centroids": [
            [0.1] * 384,  # Cluster 0 centroid
            [0.2] * 384,  # Cluster 1 centroid
            [0.3] * 384   # Cluster 2 centroid
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
            "mongodb": [0.34] * 384
        }
    }


@pytest.fixture
def clustering_json_file(sample_clustering_json, tmp_path):
    """Create temporary clustering JSON file."""
    json_path = tmp_path / "clusters.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sample_clustering_json, f, indent=2)
    return json_path


@pytest.fixture
def cluster_parser(clustering_json_file):
    """Create ClusterParser instance."""
    return ClusterParser(clustering_json_file)


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
        
        fallback_cluster = linker._get_fallback_cluster()
        
        assert fallback_cluster == 2, "Should assign to cluster 2 (largest with 4 keywords)"
    
    def test_fallback_uniform(self, cluster_parser):
        """Test fallback with uniform distribution."""
        linker = SourceClusterLinker(cluster_parser, fallback_strategy="uniform")
        
        # Test deterministic uniform assignment
        assert linker._get_fallback_cluster(pair_id=0) == 0, "pair_id=0 should map to cluster 0"
        assert linker._get_fallback_cluster(pair_id=1) == 1, "pair_id=1 should map to cluster 1"
        assert linker._get_fallback_cluster(pair_id=2) == 2, "pair_id=2 should map to cluster 2"
        assert linker._get_fallback_cluster(pair_id=3) == 0, "pair_id=3 should wrap to cluster 0"
    
    def test_fallback_random(self, cluster_parser):
        """Test fallback with random assignment."""
        linker = SourceClusterLinker(cluster_parser, fallback_strategy="random")
        
        # Set seed for reproducibility
        np.random.seed(42)
        fallback_cluster = linker._get_fallback_cluster()
        
        assert 0 <= fallback_cluster < 3, "Should assign to valid cluster"


class TestSourceClusterLinkerSinglePair:
    """Test linking single pairs to clusters."""
    
    def test_link_single_keyword(self, cluster_parser):
        """Test linking pair with single keyword."""
        linker = SourceClusterLinker(cluster_parser)
        
        cluster_id = linker.link_pair(["python"])
        
        assert cluster_id == 1, "Should link 'python' to cluster 1"
    
    def test_link_multiple_keywords_same_cluster(self, cluster_parser):
        """Test linking pair with multiple keywords from same cluster."""
        linker = SourceClusterLinker(cluster_parser)
        
        cluster_id = linker.link_pair(["python", "java", "programming"])
        
        assert cluster_id == 1, "Should link all programming keywords to cluster 1"
    
    def test_link_multiple_keywords_different_clusters(self, cluster_parser):
        """Test linking pair with keywords from different clusters (majority voting)."""
        linker = SourceClusterLinker(cluster_parser)
        
        # 2 from cluster 1, 1 from cluster 0 -> should select cluster 1
        cluster_id = linker.link_pair(["python", "java", "machine learning"])
        
        assert cluster_id == 1, "Should link to cluster 1 (majority vote: 2 vs 1)"
    
    def test_link_empty_keywords(self, cluster_parser):
        """Test linking pair with no keywords uses fallback."""
        linker = SourceClusterLinker(cluster_parser, fallback_strategy="largest")
        
        cluster_id = linker.link_pair([], pair_id=0)
        
        assert cluster_id == 2, "Should use fallback to largest cluster (cluster 2)"
    
    def test_link_unmatched_keywords(self, cluster_parser):
        """Test linking pair with unmatched keywords uses fallback."""
        linker = SourceClusterLinker(cluster_parser, fallback_strategy="largest")
        
        cluster_id = linker.link_pair(["unknown", "invalid", "keywords"], pair_id=5)
        
        assert cluster_id == 2, "Should use fallback when no keywords match"


class TestSourceClusterLinkerBatch:
    """Test batch linking of pairs."""
    
    def test_link_batch_basic(self, cluster_parser):
        """Test batch linking of multiple pairs."""
        linker = SourceClusterLinker(cluster_parser)
        
        keywords_list = [
            ["python", "programming"],
            ["database", "sql"],
            ["machine learning"]
        ]
        
        cluster_ids = linker.link_batch(keywords_list)
        
        assert len(cluster_ids) == 3, "Should return 3 cluster assignments"
        assert cluster_ids[0] == 1, "First pair should link to cluster 1"
        assert cluster_ids[1] == 2, "Second pair should link to cluster 2"
        assert cluster_ids[2] == 0, "Third pair should link to cluster 0"
    
    def test_link_batch_with_pair_ids(self, cluster_parser):
        """Test batch linking with explicit pair IDs."""
        linker = SourceClusterLinker(cluster_parser, fallback_strategy="uniform")
        
        keywords_list = [
            ["python"],
            [],  # Empty keywords
            ["sql"]
        ]
        pair_ids = [10, 11, 12]
        
        cluster_ids = linker.link_batch(keywords_list, pair_ids)
        
        assert len(cluster_ids) == 3, "Should return 3 cluster assignments"
        assert cluster_ids[0] == 1, "First pair should link to cluster 1"
        assert cluster_ids[1] == 11 % 3, "Second pair should use uniform fallback"
        assert cluster_ids[2] == 2, "Third pair should link to cluster 2"


class TestSourceClusterLinkerDataFrame:
    """Test DataFrame linking."""
    
    def test_link_dataframe_basic(self, cluster_parser):
        """Test linking DataFrame with keywords."""
        linker = SourceClusterLinker(cluster_parser)
        
        df = pd.DataFrame({
            "pair_id": [0, 1, 2],
            "question": ["Q1", "Q2", "Q3"],
            "source": ["S1", "S2", "S3"],
            "keywords": [
                ["python", "programming"],
                ["database", "sql"],
                ["machine learning", "deep learning"]
            ]
        })
        
        df_linked = linker.link_dataframe(df, show_progress=False)
        
        assert "cluster_id" in df_linked.columns, "Should add cluster_id column"
        assert len(df_linked) == 3, "Should preserve all rows"
        assert df_linked["cluster_id"].tolist() == [1, 2, 0], "Should have correct cluster assignments"
    
    def test_link_dataframe_custom_columns(self, cluster_parser):
        """Test linking DataFrame with custom column names."""
        linker = SourceClusterLinker(cluster_parser)
        
        df = pd.DataFrame({
            "id": [0, 1, 2],
            "kw": [["python"], ["sql"], ["java"]]
        })
        
        df_linked = linker.link_dataframe(
            df,
            keywords_col="kw",
            pair_id_col="id",
            output_col="assigned_cluster",
            show_progress=False
        )
        
        assert "assigned_cluster" in df_linked.columns, "Should add custom output column"
        assert df_linked["assigned_cluster"].tolist() == [1, 2, 1], "Should have correct assignments"
    
    def test_link_dataframe_missing_keywords_col(self, cluster_parser):
        """Test linking DataFrame fails when keywords column missing."""
        linker = SourceClusterLinker(cluster_parser)
        
        df = pd.DataFrame({"pair_id": [0, 1], "text": ["text1", "text2"]})
        
        with pytest.raises(ValueError, match="Keywords column .* not found"):
            linker.link_dataframe(df)
    
    def test_link_dataframe_missing_pair_id_col(self, cluster_parser):
        """Test linking DataFrame uses index when pair_id column missing."""
        linker = SourceClusterLinker(cluster_parser, fallback_strategy="uniform")
        
        df = pd.DataFrame({
            "keywords": [[], [], []]  # Empty keywords to test fallback
        })
        
        df_linked = linker.link_dataframe(df, show_progress=False)
        
        assert "cluster_id" in df_linked.columns, "Should add cluster_id column"
        # Should use DataFrame index as pair_id for uniform fallback
        assert df_linked["cluster_id"].tolist() == [0, 1, 2], "Should use index for fallback"
    
    def test_link_dataframe_string_keywords(self, cluster_parser):
        """Test linking DataFrame with keywords as comma-separated strings."""
        linker = SourceClusterLinker(cluster_parser)
        
        df = pd.DataFrame({
            "pair_id": [0, 1],
            "keywords": ["python, programming", "sql, database"]
        })
        
        df_linked = linker.link_dataframe(df, show_progress=False)
        
        assert df_linked["cluster_id"].tolist() == [1, 2], "Should parse string keywords correctly"


class TestSourceClusterLinkerValidation:
    """Test cluster assignment validation."""
    
    def test_validate_valid_assignments(self, cluster_parser):
        """Test validation of valid cluster assignments."""
        linker = SourceClusterLinker(cluster_parser)
        
        cluster_assignments = [0, 0, 1, 1, 2, 2]
        validation = linker.validate_assignments(cluster_assignments)
        
        assert validation["valid"] is True, "Should be valid"
        assert len(validation["errors"]) == 0, "Should have no errors"
        assert validation["statistics"]["n_pairs"] == 6, "Should count 6 pairs"
        assert validation["statistics"]["n_clusters_used"] == 3, "Should use all 3 clusters"
    
    def test_validate_invalid_range_negative(self, cluster_parser):
        """Test validation fails with negative cluster IDs."""
        linker = SourceClusterLinker(cluster_parser)
        
        cluster_assignments = [0, -1, 2]
        validation = linker.validate_assignments(cluster_assignments)
        
        assert validation["valid"] is False, "Should be invalid"
        assert len(validation["errors"]) > 0, "Should have errors"
        assert "Invalid cluster ID" in validation["errors"][0], "Should report negative ID error"
    
    def test_validate_invalid_range_too_large(self, cluster_parser):
        """Test validation fails with cluster IDs >= n_clusters."""
        linker = SourceClusterLinker(cluster_parser)
        
        cluster_assignments = [0, 1, 3]  # 3 >= n_clusters (3)
        validation = linker.validate_assignments(cluster_assignments)
        
        assert validation["valid"] is False, "Should be invalid"
        assert len(validation["errors"]) > 0, "Should have errors"
        assert "Invalid cluster ID" in validation["errors"][0], "Should report out-of-range error"
    
    def test_validate_missing_clusters(self, cluster_parser):
        """Test validation warns when clusters have no assignments."""
        linker = SourceClusterLinker(cluster_parser)
        
        cluster_assignments = [0, 0, 0, 0]  # Only cluster 0, missing 1 and 2
        validation = linker.validate_assignments(cluster_assignments)
        
        assert validation["valid"] is True, "Should be valid (warnings don't fail)"
        assert len(validation["warnings"]) > 0, "Should have warnings"
        assert "have no assignments" in validation["warnings"][0], "Should warn about missing clusters"
    
    def test_validate_imbalanced_distribution(self):
        """Test validation warns about highly imbalanced distribution."""
        # Create a parser with 5 clusters to allow CV > 2.0
        clustering_data = {
            "metadata": {"n_clusters": 5, "embedding_dim": 384},
            "cluster_assignments": {f"kw{i}": i % 5 for i in range(20)},
            "clusters": {str(i): [f"kw{j}" for j in range(i, 20, 5)] for i in range(5)}
        }
        
        json_path = tempfile.mktemp(suffix=".json")
        with open(json_path, "w") as f:
            json.dump(clustering_data, f)
        
        try:
            parser = ClusterParser(json_path)
            linker = SourceClusterLinker(parser)
            
            # Highly imbalanced: 10000 in cluster 0, 1 each in clusters 1-4
            # With 5 clusters: mean=2000.8, std≈4000, CV≈2.0
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
        
        cluster_assignments = torch.tensor([0, 1, 2, 0, 1, 2])
        validation = linker.validate_assignments(cluster_assignments)
        
        assert validation["valid"] is True, "Should validate torch.Tensor"
        assert validation["statistics"]["n_pairs"] == 6, "Should count pairs from tensor"
    
    def test_validate_numpy_array(self, cluster_parser):
        """Test validation works with numpy arrays."""
        linker = SourceClusterLinker(cluster_parser)
        
        cluster_assignments = np.array([0, 1, 2, 0, 1, 2])
        validation = linker.validate_assignments(cluster_assignments)
        
        assert validation["valid"] is True, "Should validate np.ndarray"
        assert validation["statistics"]["n_pairs"] == 6, "Should count pairs from array"


class TestLinkSourcesConvenience:
    """Test convenience function."""
    
    def test_link_sources_basic(self, cluster_parser):
        """Test convenience function for linking sources."""
        df = pd.DataFrame({
            "pair_id": [0, 1, 2],
            "keywords": [["python"], ["sql"], ["machine learning"]]
        })
        
        df_linked = link_sources(df, cluster_parser, show_progress=False)
        
        assert "cluster_id" in df_linked.columns, "Should add cluster_id column"
        assert df_linked["cluster_id"].tolist() == [1, 2, 0], "Should have correct assignments"
    
    def test_link_sources_custom_fallback(self, cluster_parser):
        """Test convenience function with custom fallback strategy."""
        df = pd.DataFrame({
            "pair_id": [0, 1],
            "keywords": [[], []]  # Empty keywords
        })
        
        df_linked = link_sources(
            df,
            cluster_parser,
            fallback_strategy="uniform",
            show_progress=False
        )
        
        assert df_linked["cluster_id"].tolist() == [0, 1], "Should use uniform fallback"


class TestSourceClusterLinkerEdgeCases:
    """Test edge cases."""
    
    def test_link_pair_case_insensitive(self, cluster_parser):
        """Test keyword matching is case-insensitive."""
        linker = SourceClusterLinker(cluster_parser)
        
        # Keywords are stored lowercase in cluster JSON
        cluster_id = linker.link_pair(["Python", "PROGRAMMING"])
        
        assert cluster_id == 1, "Should match keywords case-insensitively"
    
    def test_link_pair_with_none(self, cluster_parser):
        """Test linking pair handles None keywords gracefully."""
        linker = SourceClusterLinker(cluster_parser, fallback_strategy="largest")
        
        # Should treat None as empty list
        cluster_id = linker.link_pair(None if None else [], pair_id=0)
        
        assert cluster_id == 2, "Should handle None/empty as fallback"
    
    def test_link_dataframe_large(self, cluster_parser):
        """Test linking large DataFrame efficiently."""
        linker = SourceClusterLinker(cluster_parser)
        
        # Create large DataFrame
        n_pairs = 1000
        df = pd.DataFrame({
            "pair_id": list(range(n_pairs)),
            "keywords": [["python"] if i % 2 == 0 else ["sql"] for i in range(n_pairs)]
        })
        
        df_linked = linker.link_dataframe(df, show_progress=False)
        
        assert len(df_linked) == n_pairs, "Should process all pairs"
        assert "cluster_id" in df_linked.columns, "Should add cluster_id column"
        # Half should be cluster 1 (python), half should be cluster 2 (sql)
        assert df_linked["cluster_id"].value_counts()[1] == 500, "Should assign 500 to cluster 1"
        assert df_linked["cluster_id"].value_counts()[2] == 500, "Should assign 500 to cluster 2"
