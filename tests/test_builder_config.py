"""Tests for BuildConfig dataclass."""

import json
import tempfile
from pathlib import Path

import pytest

from RAG_supporters.dataset.builder_config import BuildConfig


class TestBuildConfigInit:
    """Test BuildConfig initialization and validation."""
    
    def test_valid_initialization(self):
        """Test BuildConfig initializes with valid parameters."""
        config = BuildConfig(
            embedding_dim=384,
            n_neg=12,
            clustering_source="clusters.json",
            split_ratios=[0.8, 0.1, 0.1],
            steering_probabilities={
                "zero": 0.25,
                "centroid": 0.25,
                "keyword": 0.25,
                "residual": 0.25
            },
            curriculum={
                "mode": "linear",
                "start_distance": 0.3,
                "end_distance": 0.7,
                "warmup_epochs": 10
            }
        )
        
        assert config.embedding_dim == 384, "Embedding dimension should be set"
        assert config.n_neg == 12, "Number of negatives should be set"
        assert config.clustering_source == "clusters.json", "Clustering source should be set"
        assert config.storage_format == "pt", "Default storage format should be 'pt'"
        assert config.random_seed == 42, "Default random seed should be 42"
    
    def test_default_values(self):
        """Test BuildConfig uses correct default values."""
        config = BuildConfig(
            embedding_dim=384,
            n_neg=12,
            clustering_source="clusters.json"
        )
        
        assert config.split_ratios == [0.8, 0.1, 0.1], "Default split ratios should be [0.8, 0.1, 0.1]"
        assert len(config.steering_probabilities) == 4, "Should have 4 steering probability keys"
        assert sum(config.steering_probabilities.values()) == pytest.approx(1.0), "Steering probabilities should sum to 1.0"
        assert config.curriculum["mode"] == "linear", "Default curriculum mode should be 'linear'"
        assert config.include_inspection_file is False, "Default inspection file flag should be False"
    
    def test_invalid_split_ratios_sum(self):
        """Test BuildConfig raises error when split ratios don't sum to 1.0."""
        with pytest.raises(ValueError, match="split_ratios must sum to 1.0"):
            BuildConfig(
                embedding_dim=384,
                n_neg=12,
                clustering_source="clusters.json",
                split_ratios=[0.7, 0.2, 0.2]  # Sums to 1.1
            )
    
    def test_invalid_split_ratios_count(self):
        """Test BuildConfig raises error when split ratios don't have 3 values."""
        with pytest.raises(ValueError, match="split_ratios must have exactly 3 values"):
            BuildConfig(
                embedding_dim=384,
                n_neg=12,
                clustering_source="clusters.json",
                split_ratios=[0.8, 0.2]  # Only 2 values
            )
    
    def test_invalid_split_ratios_range(self):
        """Test BuildConfig raises error when split ratios are out of range."""
        with pytest.raises(ValueError, match="split_ratios must be in range"):
            BuildConfig(
                embedding_dim=384,
                n_neg=12,
                clustering_source="clusters.json",
                split_ratios=[0.8, 0.1, -0.1]  # Negative value
            )
    
    def test_invalid_steering_probabilities_sum(self):
        """Test BuildConfig raises error when steering probabilities don't sum to 1.0."""
        with pytest.raises(ValueError, match="steering_probabilities must sum to 1.0"):
            BuildConfig(
                embedding_dim=384,
                n_neg=12,
                clustering_source="clusters.json",
                steering_probabilities={
                    "zero": 0.3,
                    "centroid": 0.3,
                    "keyword": 0.3,
                    "residual": 0.3  # Sums to 1.2
                }
            )
    
    def test_invalid_steering_probabilities_keys(self):
        """Test BuildConfig raises error when steering probabilities have wrong keys."""
        with pytest.raises(ValueError, match="steering_probabilities must have keys"):
            BuildConfig(
                embedding_dim=384,
                n_neg=12,
                clustering_source="clusters.json",
                steering_probabilities={
                    "zero": 0.25,
                    "centroid": 0.25,
                    "keyword": 0.25,
                    "wrong_key": 0.25  # Invalid key
                }
            )
    
    def test_invalid_storage_format(self):
        """Test BuildConfig raises error for invalid storage format."""
        with pytest.raises(ValueError, match="storage_format must be 'pt' or 'hdf5'"):
            BuildConfig(
                embedding_dim=384,
                n_neg=12,
                clustering_source="clusters.json",
                storage_format="invalid"
            )
    
    def test_invalid_curriculum_mode(self):
        """Test BuildConfig raises error for invalid curriculum mode."""
        with pytest.raises(ValueError, match="curriculum mode must be"):
            BuildConfig(
                embedding_dim=384,
                n_neg=12,
                clustering_source="clusters.json",
                curriculum={"mode": "invalid"}
            )
    
    def test_missing_curriculum_mode(self):
        """Test BuildConfig raises error when curriculum mode is missing."""
        with pytest.raises(ValueError, match="curriculum must have 'mode' key"):
            BuildConfig(
                embedding_dim=384,
                n_neg=12,
                clustering_source="clusters.json",
                curriculum={}
            )
    
    def test_invalid_embedding_dim(self):
        """Test BuildConfig raises error for non-positive embedding dimension."""
        with pytest.raises(ValueError, match="embedding_dim must be positive"):
            BuildConfig(
                embedding_dim=0,
                n_neg=12,
                clustering_source="clusters.json"
            )
    
    def test_invalid_n_neg(self):
        """Test BuildConfig raises error for non-positive n_neg."""
        with pytest.raises(ValueError, match="n_neg must be positive"):
            BuildConfig(
                embedding_dim=384,
                n_neg=0,
                clustering_source="clusters.json"
            )


class TestBuildConfigSerialization:
    """Test BuildConfig save/load functionality."""
    
    def test_save_and_load(self):
        """Test BuildConfig can be saved and loaded correctly."""
        config = BuildConfig(
            embedding_dim=384,
            n_neg=12,
            clustering_source="clusters.json",
            random_seed=123
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            
            # Save
            config.save(config_path)
            assert config_path.exists(), "Config file should be created"
            
            # Load
            loaded = BuildConfig.load(config_path)
            
            assert loaded.embedding_dim == config.embedding_dim, "Embedding dimension should match"
            assert loaded.n_neg == config.n_neg, "Number of negatives should match"
            assert loaded.clustering_source == config.clustering_source, "Clustering source should match"
            assert loaded.random_seed == config.random_seed, "Random seed should match"
            assert loaded.split_ratios == config.split_ratios, "Split ratios should match"
    
    def test_save_creates_directories(self):
        """Test save() creates parent directories if they don't exist."""
        config = BuildConfig(
            embedding_dim=384,
            n_neg=12,
            clustering_source="clusters.json"
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "a" / "b" / "c" / "config.json"
            
            config.save(nested_path)
            assert nested_path.exists(), "Config file should be created with nested directories"
    
    def test_load_nonexistent_file(self):
        """Test load() raises error for non-existent file."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            BuildConfig.load("nonexistent.json")
    
    def test_to_dict(self):
        """Test to_dict() returns correct dictionary representation."""
        config = BuildConfig(
            embedding_dim=384,
            n_neg=12,
            clustering_source="clusters.json"
        )
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict), "Should return dictionary"
        assert config_dict["embedding_dim"] == 384, "Dictionary should contain embedding_dim"
        assert config_dict["n_neg"] == 12, "Dictionary should contain n_neg"
        assert config_dict["clustering_source"] == "clusters.json", "Dictionary should contain clustering_source"
    
    def test_json_format(self):
        """Test saved JSON has correct format and indentation."""
        config = BuildConfig(
            embedding_dim=384,
            n_neg=12,
            clustering_source="clusters.json"
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config.save(config_path)
            
            # Read raw JSON
            with open(config_path, "r") as f:
                content = f.read()
            
            # Verify it's valid JSON
            parsed = json.loads(content)
            assert isinstance(parsed, dict), "Saved file should be valid JSON dictionary"
            
            # Verify indentation (pretty-printed)
            assert "\n" in content, "JSON should be pretty-printed with newlines"


class TestBuildConfigUpdate:
    """Test BuildConfig update_post_build functionality."""
    
    def test_update_post_build(self):
        """Test update_post_build() sets computed statistics correctly."""
        config = BuildConfig(
            embedding_dim=384,
            n_neg=12,
            clustering_source="clusters.json"
        )
        
        assert config.n_pairs is None, "n_pairs should initially be None"
        assert config.n_questions is None, "n_questions should initially be None"
        
        config.update_post_build(
            n_pairs=10000,
            n_questions=5000,
            n_sources=8000,
            n_keywords=200,
            n_clusters=20
        )
        
        assert config.n_pairs == 10000, "n_pairs should be updated"
        assert config.n_questions == 5000, "n_questions should be updated"
        assert config.n_sources == 8000, "n_sources should be updated"
        assert config.n_keywords == 200, "n_keywords should be updated"
        assert config.n_clusters == 20, "n_clusters should be updated"
    
    def test_update_persists_after_save(self):
        """Test updated values persist after save/load cycle."""
        config = BuildConfig(
            embedding_dim=384,
            n_neg=12,
            clustering_source="clusters.json"
        )
        
        config.update_post_build(
            n_pairs=10000,
            n_questions=5000,
            n_sources=8000,
            n_keywords=200,
            n_clusters=20
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            
            config.save(config_path)
            loaded = BuildConfig.load(config_path)
            
            assert loaded.n_pairs == 10000, "n_pairs should persist after save/load"
            assert loaded.n_questions == 5000, "n_questions should persist after save/load"
            assert loaded.n_sources == 8000, "n_sources should persist after save/load"


class TestBuildConfigEdgeCases:
    """Test BuildConfig edge cases and boundary conditions."""
    
    def test_zero_split_ratio_allowed(self):
        """Test BuildConfig allows zero split ratio (e.g., no test set)."""
        config = BuildConfig(
            embedding_dim=384,
            n_neg=12,
            clustering_source="clusters.json",
            split_ratios=[0.9, 0.1, 0.0]  # No test set
        )
        
        assert config.split_ratios[2] == 0.0, "Zero split ratio should be allowed"
    
    def test_hdf5_storage_format(self):
        """Test BuildConfig accepts 'hdf5' storage format."""
        config = BuildConfig(
            embedding_dim=384,
            n_neg=12,
            clustering_source="clusters.json",
            storage_format="hdf5"
        )
        
        assert config.storage_format == "hdf5", "Should accept 'hdf5' storage format"
    
    def test_different_curriculum_modes(self):
        """Test BuildConfig accepts all valid curriculum modes."""
        for mode in ["fixed", "linear", "cosine"]:
            config = BuildConfig(
                embedding_dim=384,
                n_neg=12,
                clustering_source="clusters.json",
                curriculum={"mode": mode}
            )
            
            assert config.curriculum["mode"] == mode, f"Should accept '{mode}' curriculum mode"
    
    def test_large_embedding_dim(self):
        """Test BuildConfig accepts large embedding dimensions."""
        config = BuildConfig(
            embedding_dim=4096,
            n_neg=12,
            clustering_source="clusters.json"
        )
        
        assert config.embedding_dim == 4096, "Should accept large embedding dimensions"
    
    def test_many_negatives(self):
        """Test BuildConfig accepts large number of negatives."""
        config = BuildConfig(
            embedding_dim=384,
            n_neg=100,
            clustering_source="clusters.json"
        )
        
        assert config.n_neg == 100, "Should accept large number of negatives"
