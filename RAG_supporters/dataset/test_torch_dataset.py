"""
Unit tests for torch_dataset module.

Tests cover:
- SteeringMode enum
- BaseDomainAssignDataset with steering modes
- Triplet generation (base_embedding, steering_embedding, target)
- Multi-label target generation (hard/soft)
- Metadata tracking
- Cache persistence with steering data
- CachedDomainAssignDataset with steering support
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from .torch_dataset import (
    BaseDomainAssignDataset,
    CachedDomainAssignDataset,
    SteeringMode,
)


class MockEmbeddingModel:
    """Mock embedding model for testing."""

    def __init__(self, dim=128):
        self.dim = dim
        np.random.seed(42)

    def embed_query(self, text: str) -> list:
        """
        Generate deterministic embedding based on text hash.
        
        Uses hash-based seeding to ensure same text always produces
        same embedding, making tests reproducible.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding
        """
        # Use hash for deterministic results
        seed = hash(text) % (2**32)
        np.random.seed(seed)
        return np.random.randn(self.dim).tolist()

    def embed_documents(self, texts: list) -> list:
        """
        Generate embeddings for multiple texts.
        
        Delegates to embed_query for batch processing, ensuring
        consistent behavior between single and batch operations.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings (each is a list of floats)
        """
        return [self.embed_query(text) for text in texts]


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "source": ["Source A", "Source B", "Source C", "Source D"],
            "question": [
                "What is machine learning?",
                "How does deep learning work?",
                "Explain neural networks",
                "What is natural language processing?",
            ],
            "suggestions": [
                '[{"term": "machine learning", "confidence": 0.9, "type": "keyword"}]',
                '[{"term": "deep learning", "confidence": 0.8, "type": "keyword"}]',
                '[{"term": "neural network", "confidence": 0.85, "type": "keyword"}]',
                '[{"term": "nlp", "confidence": 0.95, "type": "keyword"}]',
            ],
        }
    )


@pytest.fixture
def mock_embeddings():
    """Create mock embedding model."""
    return MockEmbeddingModel(dim=128)


@pytest.fixture
def cluster_labels_hard():
    """Hard cluster assignments."""
    return {0: 0, 1: 0, 2: 1, 3: 1}


@pytest.fixture
def cluster_labels_soft():
    """Soft cluster assignments (probability distributions)."""
    return {
        0: [0.8, 0.2],
        1: [0.7, 0.3],
        2: [0.3, 0.7],
        3: [0.1, 0.9],
    }


@pytest.fixture
def cluster_descriptors():
    """Cluster descriptors."""
    return {
        0: ["machine learning", "artificial intelligence"],
        1: ["neural networks", "deep learning"],
    }


@pytest.fixture
def llm_steering_texts():
    """LLM-generated steering texts."""
    return {
        0: "Focus on machine learning fundamentals",
        1: "Emphasize deep learning architectures",
        2: "Explain neural network structures",
        3: "Cover NLP techniques and applications",
    }


class TestSteeringMode:
    """Test SteeringMode enum."""

    def test_enum_values(self):
        """Test enum has all expected values."""
        assert SteeringMode.SUGGESTION.value == "suggestion"
        assert SteeringMode.LLM_GENERATED.value == "llm_generated"
        assert SteeringMode.CLUSTER_DESCRIPTOR.value == "cluster_descriptor"
        assert SteeringMode.ZERO.value == "zero"
        assert SteeringMode.MIXED.value == "mixed"

    def test_from_string(self):
        """Test creating enum from string."""
        mode = SteeringMode("suggestion")
        assert mode == SteeringMode.SUGGESTION


class TestBaseDomainAssignDatasetStandardMode:
    """Test BaseDomainAssignDataset in standard mode (backward compatibility)."""

    def test_standard_mode_without_embeddings(self, sample_df):
        """Test standard mode without embeddings."""
        dataset = BaseDomainAssignDataset(
            df=sample_df,
            return_embeddings=False,
        )

        assert len(dataset) == 4
        sample = dataset[0]
        assert "source" in sample
        assert "question" in sample
        assert "suggestion_texts" in sample
        assert sample["source"] == "Source A"

    def test_standard_mode_with_embeddings(self, sample_df, mock_embeddings):
        """Test standard mode with embeddings."""
        dataset = BaseDomainAssignDataset(
            df=sample_df,
            embedding_model=mock_embeddings,
            return_embeddings=True,
        )

        sample = dataset[0]
        assert "source" in sample
        assert "question" in sample
        assert "suggestions" in sample
        assert isinstance(sample["source"], torch.Tensor)
        assert isinstance(sample["question"], torch.Tensor)
        assert sample["source"].shape == (128,)


class TestBaseDomainAssignDatasetTripletMode:
    """Test BaseDomainAssignDataset in triplet mode."""

    def test_triplet_mode_validation(self, sample_df, mock_embeddings):
        """Test that triplet mode requires steering_mode."""
        with pytest.raises(ValueError, match="steering_mode must be specified"):
            BaseDomainAssignDataset(
                df=sample_df,
                embedding_model=mock_embeddings,
                return_embeddings=True,
                return_triplets=True,
            )

    def test_triplet_mode_zero_steering(
        self, sample_df, mock_embeddings, cluster_labels_hard
    ):
        """Test triplet mode with zero steering."""
        dataset = BaseDomainAssignDataset(
            df=sample_df,
            embedding_model=mock_embeddings,
            return_embeddings=True,
            return_triplets=True,
            steering_mode=SteeringMode.ZERO,
            cluster_labels=cluster_labels_hard,
        ).build(save_to_cache=False)

        sample = dataset[0]

        # Check triplet structure
        assert "base_embedding" in sample
        assert "steering_embedding" in sample
        assert "target" in sample
        assert "metadata" in sample

        # Check embeddings
        assert isinstance(sample["base_embedding"], torch.Tensor)
        assert isinstance(sample["steering_embedding"], torch.Tensor)
        assert sample["base_embedding"].shape == (128,)
        assert sample["steering_embedding"].shape == (128,)

        # Check zero steering
        assert torch.allclose(
            sample["steering_embedding"], torch.zeros(128), atol=1e-6
        )

        # Check target
        assert isinstance(sample["target"], torch.Tensor)
        assert sample["target"].item() == 0

        # Check metadata
        assert sample["metadata"]["steering_mode"] == "zero"
        assert "suggestion_texts" in sample["metadata"]
        assert "cluster_assignment" in sample["metadata"]

    def test_triplet_mode_suggestion_steering(
        self, sample_df, mock_embeddings, cluster_labels_hard
    ):
        """Test triplet mode with suggestion steering."""
        dataset = BaseDomainAssignDataset(
            df=sample_df,
            embedding_model=mock_embeddings,
            return_embeddings=True,
            return_triplets=True,
            steering_mode=SteeringMode.SUGGESTION,
            cluster_labels=cluster_labels_hard,
        ).build(save_to_cache=False)

        sample = dataset[0]

        assert "steering_embedding" in sample
        assert isinstance(sample["steering_embedding"], torch.Tensor)
        assert sample["steering_embedding"].shape == (128,)
        # Should not be zero
        assert not torch.allclose(
            sample["steering_embedding"], torch.zeros(128), atol=1e-6
        )

        # Check metadata
        assert sample["metadata"]["steering_mode"] == "suggestion"
        assert len(sample["metadata"]["suggestion_texts"]) > 0

    def test_triplet_mode_cluster_descriptor_steering(
        self, sample_df, mock_embeddings, cluster_labels_hard, cluster_descriptors
    ):
        """Test triplet mode with cluster descriptor steering."""
        dataset = BaseDomainAssignDataset(
            df=sample_df,
            embedding_model=mock_embeddings,
            return_embeddings=True,
            return_triplets=True,
            steering_mode=SteeringMode.CLUSTER_DESCRIPTOR,
            cluster_labels=cluster_labels_hard,
            cluster_descriptors=cluster_descriptors,
        ).build(save_to_cache=False)

        sample = dataset[0]

        assert "steering_embedding" in sample
        assert isinstance(sample["steering_embedding"], torch.Tensor)
        assert sample["steering_embedding"].shape == (128,)

        # Check metadata includes cluster descriptors
        assert "cluster_descriptors" in sample["metadata"]
        assert len(sample["metadata"]["cluster_descriptors"]) > 0

    def test_triplet_mode_llm_steering(
        self, sample_df, mock_embeddings, cluster_labels_hard, llm_steering_texts
    ):
        """Test triplet mode with LLM-generated steering."""
        dataset = BaseDomainAssignDataset(
            df=sample_df,
            embedding_model=mock_embeddings,
            return_embeddings=True,
            return_triplets=True,
            steering_mode=SteeringMode.LLM_GENERATED,
            cluster_labels=cluster_labels_hard,
            llm_steering_texts=llm_steering_texts,
        ).build(save_to_cache=False)

        sample = dataset[0]

        assert "steering_embedding" in sample
        assert isinstance(sample["steering_embedding"], torch.Tensor)

        # Check metadata includes LLM text
        assert "llm_steering_text" in sample["metadata"]
        assert sample["metadata"]["llm_steering_text"] == llm_steering_texts[0]


class TestMultiLabelTargets:
    """Test multi-label target generation."""

    def test_hard_target_from_int(
        self, sample_df, mock_embeddings, cluster_labels_hard
    ):
        """Test hard target from integer cluster label."""
        dataset = BaseDomainAssignDataset(
            df=sample_df,
            embedding_model=mock_embeddings,
            return_embeddings=True,
            return_triplets=True,
            steering_mode=SteeringMode.ZERO,
            cluster_labels=cluster_labels_hard,
            multi_label_mode="hard",
        ).build(save_to_cache=False)

        sample = dataset[0]
        assert isinstance(sample["target"], torch.Tensor)
        assert sample["target"].dtype == torch.long
        assert sample["target"].item() == 0

    def test_soft_target_from_probabilities(
        self, sample_df, mock_embeddings, cluster_labels_soft
    ):
        """Test soft target from probability distribution."""
        dataset = BaseDomainAssignDataset(
            df=sample_df,
            embedding_model=mock_embeddings,
            return_embeddings=True,
            return_triplets=True,
            steering_mode=SteeringMode.ZERO,
            cluster_labels=cluster_labels_soft,
            multi_label_mode="soft",
        ).build(save_to_cache=False)

        sample = dataset[0]
        assert isinstance(sample["target"], torch.Tensor)
        assert sample["target"].dtype == torch.float32
        assert sample["target"].shape == (2,)
        # Check probabilities sum to 1
        assert torch.allclose(sample["target"].sum(), torch.tensor(1.0), atol=1e-5)

    def test_soft_target_from_int_converts_to_onehot(
        self, sample_df, mock_embeddings, cluster_labels_hard
    ):
        """Test soft mode converts int labels to one-hot."""
        dataset = BaseDomainAssignDataset(
            df=sample_df,
            embedding_model=mock_embeddings,
            return_embeddings=True,
            return_triplets=True,
            steering_mode=SteeringMode.ZERO,
            cluster_labels=cluster_labels_hard,
            multi_label_mode="soft",
        ).build(save_to_cache=False)

        sample = dataset[0]
        assert isinstance(sample["target"], torch.Tensor)
        assert sample["target"].dtype == torch.float32
        # Should be one-hot: [1.0, 0.0] for cluster 0
        assert sample["target"][0].item() == 1.0
        assert sample["target"][1].item() == 0.0


class TestMetadata:
    """Test metadata tracking."""

    def test_metadata_completeness(
        self,
        sample_df,
        mock_embeddings,
        cluster_labels_hard,
        cluster_descriptors,
        llm_steering_texts,
    ):
        """Test metadata contains all expected fields."""
        dataset = BaseDomainAssignDataset(
            df=sample_df,
            embedding_model=mock_embeddings,
            return_embeddings=True,
            return_triplets=True,
            steering_mode=SteeringMode.LLM_GENERATED,
            cluster_labels=cluster_labels_hard,
            cluster_descriptors=cluster_descriptors,
            llm_steering_texts=llm_steering_texts,
        ).build(save_to_cache=False)

        sample = dataset[0]
        metadata = sample["metadata"]

        # Check all expected fields
        assert "steering_mode" in metadata
        assert "suggestion_texts" in metadata
        assert "source_text" in metadata
        assert "question_text" in metadata
        assert "sample_index" in metadata
        assert "cluster_assignment" in metadata
        assert "cluster_descriptors" in metadata
        assert "llm_steering_text" in metadata

        # Verify values
        assert metadata["steering_mode"] == "llm_generated"
        assert metadata["sample_index"] == 0
        assert metadata["source_text"] == "Source A"
        assert metadata["question_text"] == "What is machine learning?"


class TestCachePersistence:
    """Test cache saving and loading with steering data."""

    def test_save_and_load_with_steering(
        self, sample_df, mock_embeddings, cluster_labels_hard, cluster_descriptors
    ):
        """Test complete save/load cycle with steering data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"

            # Build and save
            dataset = BaseDomainAssignDataset(
                df=sample_df,
                embedding_model=mock_embeddings,
                cache_dir=cache_dir,
                return_embeddings=True,
                return_triplets=True,
                steering_mode=SteeringMode.CLUSTER_DESCRIPTOR,
                cluster_labels=cluster_labels_hard,
                cluster_descriptors=cluster_descriptors,
                multi_label_mode="hard",
            ).build(save_to_cache=True)

            # Load from cache
            loaded_dataset = CachedDomainAssignDataset(
                cache_dir=cache_dir, return_embeddings=True
            )

            # Verify metadata
            assert loaded_dataset.steering_mode == SteeringMode.CLUSTER_DESCRIPTOR
            assert loaded_dataset.return_triplets is True
            assert loaded_dataset.multi_label_mode == "hard"

            # Verify data integrity
            assert len(loaded_dataset) == len(dataset)

            # Compare samples
            original_sample = dataset[0]
            loaded_sample = loaded_dataset[0]

            assert torch.allclose(
                original_sample["base_embedding"],
                loaded_sample["base_embedding"],
                atol=1e-5,
            )
            assert torch.allclose(
                original_sample["steering_embedding"],
                loaded_sample["steering_embedding"],
                atol=1e-5,
            )
            assert original_sample["target"] == loaded_sample["target"]

    def test_metadata_persistence(
        self, sample_df, mock_embeddings, cluster_labels_hard
    ):
        """Test that steering metadata is saved and loaded correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"

            # Build with steering
            BaseDomainAssignDataset(
                df=sample_df,
                embedding_model=mock_embeddings,
                cache_dir=cache_dir,
                return_embeddings=True,
                return_triplets=True,
                steering_mode=SteeringMode.ZERO,
                cluster_labels=cluster_labels_hard,
            ).build(save_to_cache=True)

            # Load and check metadata
            loaded = CachedDomainAssignDataset(
                cache_dir=cache_dir, return_embeddings=True
            )

            assert loaded.metadata["steering_mode"] == "zero"
            assert loaded.metadata["return_triplets"] is True
            assert loaded.metadata["has_cluster_labels"] is True


class TestMixedSteering:
    """Test mixed steering mode."""

    def test_mixed_steering_with_weights(
        self, sample_df, mock_embeddings, cluster_labels_hard, cluster_descriptors
    ):
        """Test mixed steering with custom weights."""
        dataset = BaseDomainAssignDataset(
            df=sample_df,
            embedding_model=mock_embeddings,
            return_embeddings=True,
            return_triplets=True,
            steering_mode=SteeringMode.MIXED,
            cluster_labels=cluster_labels_hard,
            cluster_descriptors=cluster_descriptors,
            steering_weights={"suggestion": 0.6, "cluster_descriptor": 0.4},
        ).build(save_to_cache=False)

        sample = dataset[0]

        assert "steering_embedding" in sample
        assert isinstance(sample["steering_embedding"], torch.Tensor)
        # Mixed embedding should not be all zeros
        assert not torch.allclose(
            sample["steering_embedding"], torch.zeros(128), atol=1e-6
        )


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_suggestions(self, mock_embeddings):
        """Test handling of samples with no suggestions."""
        df = pd.DataFrame(
            {
                "source": ["Source A"],
                "question": ["What is AI?"],
                "suggestions": ["[]"],
            }
        )

        dataset = BaseDomainAssignDataset(
            df=df,
            embedding_model=mock_embeddings,
            return_embeddings=True,
            return_triplets=True,
            steering_mode=SteeringMode.SUGGESTION,
            cluster_labels={0: 0},
        ).build(save_to_cache=False)

        sample = dataset[0]
        # Should handle gracefully, possibly with zero vector
        assert "steering_embedding" in sample

    def test_missing_cluster_assignment(self, sample_df, mock_embeddings):
        """Test handling when cluster assignment is missing."""
        dataset = BaseDomainAssignDataset(
            df=sample_df,
            embedding_model=mock_embeddings,
            return_embeddings=True,
            return_triplets=True,
            steering_mode=SteeringMode.ZERO,
            cluster_labels={0: 0, 1: 1},  # Only partial assignments
        ).build(save_to_cache=False)

        sample = dataset[2]  # Index 2 has no assignment
        # Should still return valid sample
        assert "base_embedding" in sample
        # Target might be None or not included
        if "target" in sample:
            # If included, should handle gracefully
            pass

    def test_index_out_of_range(self, sample_df, mock_embeddings):
        """Test index bounds checking."""
        dataset = BaseDomainAssignDataset(
            df=sample_df,
            embedding_model=mock_embeddings,
            return_embeddings=True,
        )

        with pytest.raises(IndexError):
            dataset[100]

        with pytest.raises(IndexError):
            dataset[-10]


class TestBuildProcess:
    """Test the build process with steering."""

    def test_build_steps_logging(
        self, sample_df, mock_embeddings, cluster_labels_hard, caplog
    ):
        """Test that build process logs appropriate steps for steering."""
        dataset = BaseDomainAssignDataset(
            df=sample_df,
            embedding_model=mock_embeddings,
            return_embeddings=True,
            return_triplets=True,
            steering_mode=SteeringMode.ZERO,
            cluster_labels=cluster_labels_hard,
        )

        with caplog.at_level("INFO"):
            dataset.build(save_to_cache=False)

        # Check that steering mode is logged
        log_text = caplog.text
        assert "Steering mode: zero" in log_text

    def test_validation_warnings(
        self, sample_df, mock_embeddings, cluster_labels_hard, caplog
    ):
        """Test that validation issues are logged."""
        # Create incomplete cluster labels
        incomplete_labels = {0: 0, 1: 1}  # Only 2 out of 4 samples

        dataset = BaseDomainAssignDataset(
            df=sample_df,
            embedding_model=mock_embeddings,
            return_embeddings=True,
            return_triplets=True,
            steering_mode=SteeringMode.ZERO,
            cluster_labels=incomplete_labels,
        )

        with caplog.at_level("WARNING"):
            dataset.build(save_to_cache=False)

        # Should warn about missing assignments
        log_text = caplog.text
        assert "Missing cluster assignments" in log_text or "doesn't match" in log_text
