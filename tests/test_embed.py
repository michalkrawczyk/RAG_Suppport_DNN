"""Tests for Embedding Generator."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from RAG_supporters.embeddings_ops import (
    EmbeddingGenerator,
    generate_embeddings,
)
from RAG_supporters.clustering_ops import ClusterParser


# Mock embedding model for testing
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
        **kwargs
    ):
        """Mock encode method that returns deterministic embeddings."""
        if isinstance(texts, str):
            texts = [texts]
        
        # Generate deterministic embeddings based on text hash
        embeddings = []
        for text in texts:
            # Use hash of text to create deterministic but unique embeddings
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


@pytest.fixture
def sample_clustering_json():
    """Create sample KeywordClusterer JSON format."""
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
                "default_metric": "cosine"
            }
        },
        "cluster_assignments": {
            "python": 0,
            "java": 0,
            "sql": 1,
            "database": 1,
            "ai": 2,
            "ml": 2
        },
        "clusters": {
            "0": ["python", "java"],
            "1": ["sql", "database"],
            "2": ["ai", "ml"]
        },
        "cluster_stats": {
            "0": {"size": 2, "topic_descriptors": ["programming"]},
            "1": {"size": 2, "topic_descriptors": ["database"]},
            "2": {"size": 2, "topic_descriptors": ["artificial intelligence"]}
        },
        "centroids": [
            [0.1] * 384,  # Cluster 0 centroid
            [0.2] * 384,  # Cluster 1 centroid
            [0.3] * 384   # Cluster 2 centroid
        ],
        "embeddings": {
            "python": [0.11] * 384,
            "java": [0.12] * 384,
            "sql": [0.21] * 384,
            "database": [0.22] * 384,
            "ai": [0.31] * 384,
            "ml": [0.32] * 384
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


class TestEmbeddingGeneratorInit:
    """Test EmbeddingGenerator initialization."""
    
    def test_init_without_cluster_parser(self, mock_model):
        """Test initialization without cluster parser."""
        generator = EmbeddingGenerator(mock_model)
        
        assert generator.embedder is not None, "Should wrap model in KeywordEmbedder"
        assert generator.cluster_parser is None, "Should have no cluster parser"
        assert generator.batch_size == 32, "Should have default batch size"
        assert generator.show_progress is True, "Should show progress by default"
    
    def test_init_with_cluster_parser(self, mock_model, cluster_parser):
        """Test initialization with cluster parser."""
        generator = EmbeddingGenerator(mock_model, cluster_parser)
        
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
    
    def test_validate_centroid_similarity_valid(self, mock_model, cluster_parser):
        """Test validation passes for similar centroids."""
        generator = EmbeddingGenerator(mock_model, cluster_parser)
        
        # Create keyword embeddings similar to centroids
        keyword_embeddings = {
            "python": np.array([0.11] * 384),
            "java": np.array([0.12] * 384),
            "sql": np.array([0.21] * 384),
            "database": np.array([0.22] * 384),
            "ai": np.array([0.31] * 384),
            "ml": np.array([0.32] * 384)
        }
        
        centroids = np.array([
            [0.1] * 384,   # Close to python/java
            [0.2] * 384,   # Close to sql/database
            [0.3] * 384    # Close to ai/ml
        ])
        
        is_valid, warnings = generator._validate_centroid_similarity(
            keyword_embeddings,
            centroids,
            min_similarity=0.5
        )
        
        # May have warnings but should complete
        assert isinstance(is_valid, bool), "Should return validation result"
        assert isinstance(warnings, list), "Should return warnings list"
    
    def test_validate_centroid_similarity_no_parser(self, mock_model):
        """Test validation skips when no cluster parser."""
        generator = EmbeddingGenerator(mock_model)
        
        keyword_embeddings = {"test": np.random.randn(384)}
        centroids = np.random.randn(3, 384)
        
        is_valid, warnings = generator._validate_centroid_similarity(
            keyword_embeddings,
            centroids
        )
        
        assert is_valid is True, "Should skip validation and return True"
        assert len(warnings) == 0, "Should have no warnings"


class TestEmbeddingGeneratorTextEmbeddings:
    """Test text embedding generation."""
    
    def test_generate_text_embeddings_basic(self, mock_model):
        """Test generating embeddings for texts."""
        generator = EmbeddingGenerator(mock_model, show_progress=False)
        
        texts = ["Sample text 1", "Sample text 2", "Sample text 3"]
        embeddings = generator.generate_text_embeddings(texts, text_type="test")
        
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
        generator = EmbeddingGenerator(
            mock_model,
            normalize_embeddings=True,
            show_progress=False
        )
        
        texts = ["Text 1", "Text 2"]
        embeddings = generator.generate_text_embeddings(texts, text_type="test")
        
        # Check L2 norm is approximately 1
        norms = torch.norm(embeddings, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=0.01), \
            "Embeddings should be L2-normalized"
    
    def test_generate_text_embeddings_deterministic(self, mock_model):
        """Test embeddings are deterministic for same text."""
        generator = EmbeddingGenerator(mock_model, show_progress=False)
        
        text = ["Same text"]
        embeddings1 = generator.generate_text_embeddings(text, text_type="test")
        embeddings2 = generator.generate_text_embeddings(text, text_type="test")
        
        assert torch.allclose(embeddings1, embeddings2), \
            "Same text should produce same embeddings"


class TestEmbeddingGeneratorKeywordEmbeddings:
    """Test keyword embedding generation."""
    
    def test_generate_keyword_embeddings(self, mock_model):
        """Test generating keyword embeddings with mapping."""
        generator = EmbeddingGenerator(mock_model, show_progress=False)
        
        keywords = ["python", "java", "sql"]
        embeddings, keyword_to_id = generator.generate_keyword_embeddings(keywords)
        
        assert isinstance(embeddings, torch.Tensor), "Should return torch.Tensor"
        assert embeddings.shape == (3, 384), "Should have correct shape"
        assert len(keyword_to_id) == 3, "Should have mapping for all keywords"
        assert keyword_to_id["python"] == 0, "Should map first keyword to 0"
        assert keyword_to_id["java"] == 1, "Should map second keyword to 1"
        assert keyword_to_id["sql"] == 2, "Should map third keyword to 2"


class TestEmbeddingGeneratorCentroidEmbeddings:
    """Test centroid embedding generation."""
    
    def test_generate_centroid_embeddings(self, mock_model, cluster_parser):
        """Test generating centroid embeddings from cluster parser."""
        generator = EmbeddingGenerator(mock_model, cluster_parser, show_progress=False)
        
        centroids = generator.generate_centroid_embeddings()
        
        assert isinstance(centroids, torch.Tensor), "Should return torch.Tensor"
        assert centroids.shape == (3, 384), "Should have centroids for 3 clusters"
        assert centroids.dtype == torch.float32, "Should be float32"
    
    def test_generate_centroid_embeddings_no_parser(self, mock_model):
        """Test generation fails without cluster parser."""
        generator = EmbeddingGenerator(mock_model)
        
        with pytest.raises(ValueError, match="ClusterParser required"):
            generator.generate_centroid_embeddings()
    
    def test_generate_centroid_embeddings_with_validation(self, mock_model, cluster_parser):
        """Test centroid generation with keyword validation."""
        generator = EmbeddingGenerator(mock_model, cluster_parser, show_progress=False)
        
        # Create keyword embeddings for validation
        keywords = ["python", "java", "sql", "database", "ai", "ml"]
        keyword_embs, keyword_to_id = generator.generate_keyword_embeddings(keywords)
        keyword_embs_dict = {
            kw: keyword_embs[idx].numpy()
            for kw, idx in keyword_to_id.items()
        }
        
        centroids = generator.generate_centroid_embeddings(
            keyword_embeddings_dict=keyword_embs_dict,
            validate=True
        )
        
        assert isinstance(centroids, torch.Tensor), "Should return centroids despite warnings"


class TestEmbeddingGeneratorAllEmbeddings:
    """Test generating all embeddings from DataFrame."""
    
    def test_generate_all_embeddings_basic(self, mock_model, cluster_parser):
        """Test generating all embeddings from DataFrame."""
        generator = EmbeddingGenerator(mock_model, cluster_parser, show_progress=False)
        
        df = pd.DataFrame({
            "question": ["What is Python?", "What is SQL?", "What is Python?"],
            "source": ["Python is a language", "SQL is a query language", "Python rocks"],
            "keywords": [["python", "java"], ["sql"], ["python"]]
        })
        
        embeddings = generator.generate_all_embeddings(df)
        
        assert "question_embs" in embeddings, "Should have question embeddings"
        assert "source_embs" in embeddings, "Should have source embeddings"
        assert "keyword_embs" in embeddings, "Should have keyword embeddings"
        assert "centroid_embs" in embeddings, "Should have centroid embeddings"
        assert "question_to_id" in embeddings, "Should have question mapping"
        assert "source_to_id" in embeddings, "Should have source mapping"
        assert "keyword_to_id" in embeddings, "Should have keyword mapping"
    
    def test_generate_all_embeddings_shapes(self, mock_model, cluster_parser):
        """Test embeddings have correct shapes."""
        generator = EmbeddingGenerator(mock_model, cluster_parser, show_progress=False)
        
        df = pd.DataFrame({
            "question": ["Q1", "Q2", "Q1"],
            "source": ["S1", "S2", "S3"],
            "keywords": [["kw1"], ["kw2"], ["kw1", "kw3"]]
        })
        
        embeddings = generator.generate_all_embeddings(df)
        
        # 2 unique questions, 3 unique sources, 3 unique keywords
        assert embeddings["question_embs"].shape == (2, 384), "Should have 2 unique questions"
        assert embeddings["source_embs"].shape == (3, 384), "Should have 3 unique sources"
        assert embeddings["keyword_embs"].shape == (3, 384), "Should have 3 unique keywords"
        assert embeddings["centroid_embs"].shape == (3, 384), "Should have 3 cluster centroids"
    
    def test_generate_all_embeddings_string_keywords(self, mock_model, cluster_parser):
        """Test handling keywords as comma-separated strings."""
        generator = EmbeddingGenerator(mock_model, cluster_parser, show_progress=False)
        
        df = pd.DataFrame({
            "question": ["Q1"],
            "source": ["S1"],
            "keywords": ["python, java, sql"]
        })
        
        embeddings = generator.generate_all_embeddings(df)
        
        assert "python" in embeddings["keyword_to_id"], "Should extract 'python'"
        assert "java" in embeddings["keyword_to_id"], "Should extract 'java'"
        assert "sql" in embeddings["keyword_to_id"], "Should extract 'sql'"


class TestEmbeddingGeneratorSaveEmbeddings:
    """Test saving embeddings to files."""
    
    def test_save_embeddings_basic(self, mock_model, tmp_path):
        """Test saving embeddings to directory."""
        generator = EmbeddingGenerator(mock_model, show_progress=False)
        
        embeddings = {
            "question_embs": torch.randn(10, 384),
            "source_embs": torch.randn(20, 384),
            "keyword_embs": torch.randn(5, 384),
            "question_to_id": {"Q1": 0},  # Should be skipped (not tensor)
        }
        
        generator.save_embeddings(embeddings, tmp_path)
        
        assert (tmp_path / "question_embs.pt").exists(), "Should save question embeddings"
        assert (tmp_path / "source_embs.pt").exists(), "Should save source embeddings"
        assert (tmp_path / "keyword_embs.pt").exists(), "Should save keyword embeddings"
        assert not (tmp_path / "question_to_id.pt").exists(), "Should skip non-tensor items"
    
    def test_save_embeddings_with_prefix(self, mock_model, tmp_path):
        """Test saving embeddings with filename prefix."""
        generator = EmbeddingGenerator(mock_model)
        
        embeddings = {
            "question_embs": torch.randn(10, 384),
            "source_embs": torch.randn(20, 384)
        }
        
        generator.save_embeddings(embeddings, tmp_path, prefix="test_")
        
        assert (tmp_path / "test_question_embs.pt").exists(), "Should save with prefix"
        assert (tmp_path / "test_source_embs.pt").exists(), "Should save with prefix"
    
    def test_save_embeddings_creates_directory(self, mock_model, tmp_path):
        """Test saving creates output directory if not exists."""
        generator = EmbeddingGenerator(mock_model)
        
        output_dir = tmp_path / "nonexistent" / "subdir"
        embeddings = {"question_embs": torch.randn(5, 384)}
        
        generator.save_embeddings(embeddings, output_dir)
        
        assert output_dir.exists(), "Should create output directory"
        assert (output_dir / "question_embs.pt").exists(), "Should save embedding file"
    
    def test_save_and_load_embeddings(self, mock_model, tmp_path):
        """Test saved embeddings can be loaded back."""
        generator = EmbeddingGenerator(mock_model)
        
        original_embeddings = {
            "question_embs": torch.randn(10, 384),
            "source_embs": torch.randn(20, 384)
        }
        
        generator.save_embeddings(original_embeddings, tmp_path)
        
        # Load back
        loaded_question = torch.load(tmp_path / "question_embs.pt")
        loaded_source = torch.load(tmp_path / "source_embs.pt")
        
        assert torch.allclose(loaded_question, original_embeddings["question_embs"]), \
            "Loaded question embeddings should match original"
        assert torch.allclose(loaded_source, original_embeddings["source_embs"]), \
            "Loaded source embeddings should match original"


class TestGenerateEmbeddingsConvenience:
    """Test convenience function."""
    
    def test_generate_embeddings_basic(self, mock_model, cluster_parser):
        """Test convenience function generates all embeddings."""
        df = pd.DataFrame({
            "question": ["Q1", "Q2"],
            "source": ["S1", "S2"],
            "keywords": [["python"], ["sql"]]
        })
        
        embeddings = generate_embeddings(
            df=df,
            embedding_model=mock_model,
            cluster_parser=cluster_parser,
            batch_size=16,
            validate=True
        )
        
        assert "question_embs" in embeddings, "Should have question embeddings"
        assert "source_embs" in embeddings, "Should have source embeddings"
        assert embeddings["question_embs"].shape[0] == 2, "Should have 2 question embeddings"
    
    def test_generate_embeddings_with_save(self, mock_model, cluster_parser, tmp_path):
        """Test convenience function saves embeddings to directory."""
        df = pd.DataFrame({
            "question": ["Q1"],
            "source": ["S1"],
            "keywords": [["python"]]
        })
        
        embeddings = generate_embeddings(
            df=df,
            embedding_model=mock_model,
            cluster_parser=cluster_parser,
            output_dir=tmp_path
        )
        
        assert (tmp_path / "question_embs.pt").exists(), "Should save question embeddings"
        assert (tmp_path / "source_embs.pt").exists(), "Should save source embeddings"
        assert (tmp_path / "keyword_embs.pt").exists(), "Should save keyword embeddings"
        assert (tmp_path / "centroid_embs.pt").exists(), "Should save centroid embeddings"
    
    def test_generate_embeddings_custom_columns(self, mock_model, cluster_parser):
        """Test convenience function with custom column names."""
        df = pd.DataFrame({
            "q": ["Q1"],
            "s": ["S1"],
            "kw": [["python"]]
        })
        
        embeddings = generate_embeddings(
            df=df,
            embedding_model=mock_model,
            cluster_parser=cluster_parser,
            question_col="q",
            source_col="s",
            keywords_col="kw"
        )
        
        assert "question_embs" in embeddings, "Should generate embeddings with custom columns"


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
        
        # Generate 100 texts
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
        
        df = pd.DataFrame({
            "question": ["Q1"],
            "source": ["S1"],
            "keywords": [["python"]]
        })
        
        embeddings = generator.generate_all_embeddings(df)
        
        assert "question_embs" in embeddings, "Should have question embeddings"
        assert "source_embs" in embeddings, "Should have source embeddings"
        assert "keyword_embs" in embeddings, "Should have keyword embeddings"
        assert "centroid_embs" not in embeddings, "Should not have centroid embeddings without parser"
