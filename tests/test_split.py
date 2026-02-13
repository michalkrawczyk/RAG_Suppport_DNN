"""Tests for Dataset Splitter."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from RAG_supporters.data_prep import (
    DatasetSplitter,
    split_dataset,
)


@pytest.fixture
def sample_data():
    """Create sample data for testing.
    
    Creates:
    - 100 questions
    - 500 pairs distributed across questions
    - 5 clusters with varying distributions
    """
    torch.manual_seed(42)
    np.random.seed(42)
    
    n_questions = 100
    n_pairs = 500
    n_clusters = 5
    
    # Create pair indices with varying numbers of sources per question
    pair_indices = []
    question_id = 0
    
    while len(pair_indices) < n_pairs:
        # Random number of sources per question (1-10)
        n_sources = np.random.randint(1, 11)
        for _ in range(n_sources):
            if len(pair_indices) >= n_pairs:
                break
            source_id = np.random.randint(0, 200)
            pair_indices.append([question_id, source_id])
        question_id += 1
    
    pair_indices = torch.tensor(pair_indices[:n_pairs], dtype=torch.long)
    
    # Assign clusters to pairs
    pair_cluster_ids = torch.randint(0, n_clusters, (n_pairs,))
    
    # Count actual unique questions
    unique_questions = pair_indices[:, 0].unique()
    n_questions_actual = len(unique_questions)
    
    return {
        "pair_indices": pair_indices,
        "pair_cluster_ids": pair_cluster_ids,
        "n_pairs": n_pairs,
        "n_questions": n_questions_actual,
        "n_clusters": n_clusters
    }


@pytest.fixture
def small_data():
    """Create small dataset for edge case testing.
    
    Creates:
    - 10 questions
    - 30 pairs
    - 3 clusters
    """
    torch.manual_seed(42)
    
    # 10 questions, each with 3 sources
    pair_indices = []
    for q in range(10):
        for s in range(3):
            pair_indices.append([q, s + q * 3])
    
    pair_indices = torch.tensor(pair_indices, dtype=torch.long)
    pair_cluster_ids = torch.randint(0, 3, (30,))
    
    return {
        "pair_indices": pair_indices,
        "pair_cluster_ids": pair_cluster_ids,
        "n_pairs": 30,
        "n_questions": 10,
        "n_clusters": 3
    }


class TestDatasetSplitterInit:
    """Test DatasetSplitter initialization."""
    
    def test_init_valid(self, sample_data):
        """Test initialization with valid inputs."""
        splitter = DatasetSplitter(
            pair_indices=sample_data["pair_indices"],
            pair_cluster_ids=sample_data["pair_cluster_ids"]
        )
        
        assert splitter.n_pairs == sample_data["n_pairs"], \
            "Should have correct number of pairs"
        assert splitter.n_questions == sample_data["n_questions"], \
            "Should have correct number of questions"
        assert splitter.train_ratio == 0.7, \
            "Should have default train ratio"
        assert splitter.val_ratio == 0.15, \
            "Should have default val ratio"
        assert splitter.test_ratio == 0.15, \
            "Should have default test ratio"
    
    def test_init_custom_ratios(self, sample_data):
        """Test initialization with custom ratios."""
        splitter = DatasetSplitter(
            pair_indices=sample_data["pair_indices"],
            pair_cluster_ids=sample_data["pair_cluster_ids"],
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2
        )
        
        assert splitter.train_ratio == 0.6, \
            "Should have custom train ratio"
        assert splitter.val_ratio == 0.2, \
            "Should have custom val ratio"
        assert splitter.test_ratio == 0.2, \
            "Should have custom test ratio"
    
    def test_init_invalid_pair_indices_type(self, sample_data):
        """Test initialization with invalid pair indices type."""
        with pytest.raises(TypeError, match="pair_indices must be torch.Tensor"):
            DatasetSplitter(
                pair_indices=[[0, 1], [1, 2]],  # List instead of tensor
                pair_cluster_ids=sample_data["pair_cluster_ids"]
            )
    
    def test_init_invalid_pair_cluster_ids_type(self, sample_data):
        """Test initialization with invalid cluster IDs type."""
        with pytest.raises(TypeError, match="pair_cluster_ids must be torch.Tensor"):
            DatasetSplitter(
                pair_indices=sample_data["pair_indices"],
                pair_cluster_ids=[0, 1, 2]  # List instead of tensor
            )
    
    def test_init_invalid_pair_indices_shape(self, sample_data):
        """Test initialization with invalid pair indices shape."""
        with pytest.raises(ValueError, match="pair_indices must be"):
            DatasetSplitter(
                pair_indices=torch.tensor([0, 1, 2]),  # 1D instead of 2D
                pair_cluster_ids=sample_data["pair_cluster_ids"]
            )
    
    def test_init_invalid_pair_cluster_ids_shape(self, sample_data):
        """Test initialization with invalid cluster IDs shape."""
        with pytest.raises(ValueError, match="pair_cluster_ids must be 1D"):
            DatasetSplitter(
                pair_indices=sample_data["pair_indices"],
                pair_cluster_ids=torch.tensor([[0, 1], [2, 3]])  # 2D instead of 1D
            )
    
    def test_init_length_mismatch(self, sample_data):
        """Test initialization with mismatched tensor lengths."""
        with pytest.raises(ValueError, match="pair_cluster_ids length.*must equal"):
            DatasetSplitter(
                pair_indices=sample_data["pair_indices"],
                pair_cluster_ids=torch.randint(0, 5, (100,))  # Wrong length
            )
    
    def test_init_empty_tensors(self):
        """Test initialization with empty tensors."""
        with pytest.raises(ValueError, match="must have at least 1 rows"):
            DatasetSplitter(
                pair_indices=torch.empty((0, 2), dtype=torch.long),
                pair_cluster_ids=torch.empty((0,), dtype=torch.long)
            )
    
    def test_init_invalid_train_ratio(self, sample_data):
        """Test initialization with invalid train ratio."""
        with pytest.raises(ValueError, match="train_ratio must be"):
            DatasetSplitter(
                pair_indices=sample_data["pair_indices"],
                pair_cluster_ids=sample_data["pair_cluster_ids"],
                train_ratio=1.5  # > 1
            )
    
    def test_init_invalid_val_ratio(self, sample_data):
        """Test initialization with invalid val ratio."""
        with pytest.raises(ValueError, match="val_ratio must be"):
            DatasetSplitter(
                pair_indices=sample_data["pair_indices"],
                pair_cluster_ids=sample_data["pair_cluster_ids"],
                val_ratio=0.0  # Not > 0
            )
    
    def test_init_ratios_not_sum_to_one(self, sample_data):
        """Test initialization with ratios that don't sum to 1."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            DatasetSplitter(
                pair_indices=sample_data["pair_indices"],
                pair_cluster_ids=sample_data["pair_cluster_ids"],
                train_ratio=0.6,
                val_ratio=0.2,
                test_ratio=0.3  # Sum = 1.1
            )


class TestDatasetSplitterSplit:
    """Test split functionality."""
    
    def test_split_basic(self, sample_data):
        """Test basic split operation."""
        splitter = DatasetSplitter(
            pair_indices=sample_data["pair_indices"],
            pair_cluster_ids=sample_data["pair_cluster_ids"]
        )
        
        results = splitter.split()
        
        assert 'train_idx' in results, "Should return train_idx"
        assert 'val_idx' in results, "Should return val_idx"
        assert 'test_idx' in results, "Should return test_idx"
        
        assert isinstance(results['train_idx'], torch.Tensor), \
            "train_idx should be torch.Tensor"
        assert isinstance(results['val_idx'], torch.Tensor), \
            "val_idx should be torch.Tensor"
        assert isinstance(results['test_idx'], torch.Tensor), \
            "test_idx should be torch.Tensor"
    
    def test_split_covers_all_pairs(self, sample_data):
        """Test that split covers all pairs exactly once."""
        splitter = DatasetSplitter(
            pair_indices=sample_data["pair_indices"],
            pair_cluster_ids=sample_data["pair_cluster_ids"]
        )
        
        results = splitter.split()
        
        # Convert to sets
        train_set = set(results['train_idx'].tolist())
        val_set = set(results['val_idx'].tolist())
        test_set = set(results['test_idx'].tolist())
        
        # Check no overlap
        assert len(train_set & val_set) == 0, \
            "Train and val should not overlap"
        assert len(train_set & test_set) == 0, \
            "Train and test should not overlap"
        assert len(val_set & test_set) == 0, \
            "Val and test should not overlap"
        
        # Check all pairs covered
        all_indices = train_set | val_set | test_set
        assert len(all_indices) == sample_data["n_pairs"], \
            "All pairs should be covered exactly once"
    
    def test_split_no_question_leakage(self, sample_data):
        """Test that no questions appear in multiple splits."""
        splitter = DatasetSplitter(
            pair_indices=sample_data["pair_indices"],
            pair_cluster_ids=sample_data["pair_cluster_ids"]
        )
        
        results = splitter.split()
        
        # Extract question IDs from each split
        train_questions = set(
            sample_data["pair_indices"][results['train_idx'], 0].tolist()
        )
        val_questions = set(
            sample_data["pair_indices"][results['val_idx'], 0].tolist()
        )
        test_questions = set(
            sample_data["pair_indices"][results['test_idx'], 0].tolist()
        )
        
        # Check no overlap
        assert len(train_questions & val_questions) == 0, \
            "Questions should not leak between train and val"
        assert len(train_questions & test_questions) == 0, \
            "Questions should not leak between train and test"
        assert len(val_questions & test_questions) == 0, \
            "Questions should not leak between val and test"
    
    def test_split_non_empty_splits(self, sample_data):
        """Test that all splits are non-empty."""
        splitter = DatasetSplitter(
            pair_indices=sample_data["pair_indices"],
            pair_cluster_ids=sample_data["pair_cluster_ids"]
        )
        
        results = splitter.split()
        
        assert len(results['train_idx']) > 0, \
            "Train split should not be empty"
        assert len(results['val_idx']) > 0, \
            "Val split should not be empty"
        assert len(results['test_idx']) > 0, \
            "Test split should not be empty"
    
    def test_split_approximate_ratios(self, sample_data):
        """Test that split sizes approximately match ratios."""
        splitter = DatasetSplitter(
            pair_indices=sample_data["pair_indices"],
            pair_cluster_ids=sample_data["pair_cluster_ids"],
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15
        )
        
        results = splitter.split()
        
        n_pairs = sample_data["n_pairs"]
        train_ratio_actual = len(results['train_idx']) / n_pairs
        val_ratio_actual = len(results['val_idx']) / n_pairs
        test_ratio_actual = len(results['test_idx']) / n_pairs
        
        # Allow 10% tolerance due to question-level grouping
        assert 0.6 <= train_ratio_actual <= 0.8, \
            f"Train ratio should be ~0.7, got {train_ratio_actual:.3f}"
        assert 0.05 <= val_ratio_actual <= 0.25, \
            f"Val ratio should be ~0.15, got {val_ratio_actual:.3f}"
        assert 0.05 <= test_ratio_actual <= 0.25, \
            f"Test ratio should be ~0.15, got {test_ratio_actual:.3f}"
    
    def test_split_deterministic(self, sample_data):
        """Test that split is deterministic with same seed."""
        splitter1 = DatasetSplitter(
            pair_indices=sample_data["pair_indices"],
            pair_cluster_ids=sample_data["pair_cluster_ids"],
            random_seed=42
        )
        results1 = splitter1.split()
        
        splitter2 = DatasetSplitter(
            pair_indices=sample_data["pair_indices"],
            pair_cluster_ids=sample_data["pair_cluster_ids"],
            random_seed=42
        )
        results2 = splitter2.split()
        
        assert torch.equal(results1['train_idx'], results2['train_idx']), \
            "Train split should be deterministic"
        assert torch.equal(results1['val_idx'], results2['val_idx']), \
            "Val split should be deterministic"
        assert torch.equal(results1['test_idx'], results2['test_idx']), \
            "Test split should be deterministic"
    
    def test_split_different_seeds(self, sample_data):
        """Test that different seeds produce different splits."""
        splitter1 = DatasetSplitter(
            pair_indices=sample_data["pair_indices"],
            pair_cluster_ids=sample_data["pair_cluster_ids"],
            random_seed=42
        )
        results1 = splitter1.split()
        
        splitter2 = DatasetSplitter(
            pair_indices=sample_data["pair_indices"],
            pair_cluster_ids=sample_data["pair_cluster_ids"],
            random_seed=123
        )
        results2 = splitter2.split()
        
        assert not torch.equal(results1['train_idx'], results2['train_idx']), \
            "Different seeds should produce different train splits"
    
    def test_split_small_dataset(self, small_data):
        """Test split on small dataset."""
        splitter = DatasetSplitter(
            pair_indices=small_data["pair_indices"],
            pair_cluster_ids=small_data["pair_cluster_ids"],
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2
        )
        
        results = splitter.split()
        
        # Check all splits non-empty
        assert len(results['train_idx']) >= 6, \
            "Train split should have at least 6 pairs (2 questions × 3 pairs)"
        assert len(results['val_idx']) >= 3, \
            "Val split should have at least 3 pairs (1 question × 3 pairs)"
        assert len(results['test_idx']) >= 3, \
            "Test split should have at least 3 pairs (1 question × 3 pairs)"
        
        # Check no leakage
        train_questions = set(small_data["pair_indices"][results['train_idx'], 0].tolist())
        val_questions = set(small_data["pair_indices"][results['val_idx'], 0].tolist())
        test_questions = set(small_data["pair_indices"][results['test_idx'], 0].tolist())
        
        assert len(train_questions & val_questions) == 0, \
            "No question leakage between train and val"
        assert len(train_questions & test_questions) == 0, \
            "No question leakage between train and test"
        assert len(val_questions & test_questions) == 0, \
            "No question leakage between val and test"
    
    def test_split_single_cluster(self):
        """Test split when all pairs belong to single cluster."""
        torch.manual_seed(42)
        
        # 20 questions, 100 pairs, all in cluster 0
        pair_indices = []
        for q in range(20):
            for s in range(5):
                pair_indices.append([q, s + q * 5])
        
        pair_indices = torch.tensor(pair_indices, dtype=torch.long)
        pair_cluster_ids = torch.zeros(100, dtype=torch.long)  # All cluster 0
        
        splitter = DatasetSplitter(
            pair_indices=pair_indices,
            pair_cluster_ids=pair_cluster_ids
        )
        
        results = splitter.split()
        
        # Should still produce valid splits
        assert len(results['train_idx']) > 0, "Train split should not be empty"
        assert len(results['val_idx']) > 0, "Val split should not be empty"
        assert len(results['test_idx']) > 0, "Test split should not be empty"
        
        # Check no question leakage
        train_q = set(pair_indices[results['train_idx'], 0].tolist())
        val_q = set(pair_indices[results['val_idx'], 0].tolist())
        test_q = set(pair_indices[results['test_idx'], 0].tolist())
        
        assert len(train_q & val_q) == 0, "No leakage with single cluster"


class TestSplitDatasetFunction:
    """Test split_dataset convenience function."""
    
    def test_split_dataset_creates_files(self, sample_data):
        """Test that split_dataset creates output files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "splits"
            
            results = split_dataset(
                pair_indices=sample_data["pair_indices"],
                pair_cluster_ids=sample_data["pair_cluster_ids"],
                output_dir=output_dir,
                show_progress=False
            )
            
            # Check return value
            assert 'train_idx' in results, "Should return train_idx"
            assert 'val_idx' in results, "Should return val_idx"
            assert 'test_idx' in results, "Should return test_idx"
            
            # Check files created
            assert (output_dir / 'train_idx.pt').exists(), \
                "Should create train_idx.pt"
            assert (output_dir / 'val_idx.pt').exists(), \
                "Should create val_idx.pt"
            assert (output_dir / 'test_idx.pt').exists(), \
                "Should create test_idx.pt"
    
    def test_split_dataset_files_loadable(self, sample_data):
        """Test that saved files can be loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "splits"
            
            results = split_dataset(
                pair_indices=sample_data["pair_indices"],
                pair_cluster_ids=sample_data["pair_cluster_ids"],
                output_dir=output_dir,
                show_progress=False
            )
            
            # Load saved files
            train_loaded = torch.load(output_dir / 'train_idx.pt')
            val_loaded = torch.load(output_dir / 'val_idx.pt')
            test_loaded = torch.load(output_dir / 'test_idx.pt')
            
            # Check equality
            assert torch.equal(train_loaded, results['train_idx']), \
                "Loaded train_idx should match returned value"
            assert torch.equal(val_loaded, results['val_idx']), \
                "Loaded val_idx should match returned value"
            assert torch.equal(test_loaded, results['test_idx']), \
                "Loaded test_idx should match returned value"
    
    def test_split_dataset_custom_ratios(self, sample_data):
        """Test split_dataset with custom ratios."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "splits"
            
            results = split_dataset(
                pair_indices=sample_data["pair_indices"],
                pair_cluster_ids=sample_data["pair_cluster_ids"],
                output_dir=output_dir,
                train_ratio=0.8,
                val_ratio=0.1,
                test_ratio=0.1,
                show_progress=False
            )
            
            n_pairs = sample_data["n_pairs"]
            train_ratio = len(results['train_idx']) / n_pairs
            
            # Train should be larger with 0.8 ratio
            assert train_ratio >= 0.7, \
                f"Train ratio should be ~0.8, got {train_ratio:.3f}"


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_many_sources_per_question(self):
        """Test with questions having many sources."""
        torch.manual_seed(42)
        
        # 10 questions, each with 50 sources = 500 pairs
        pair_indices = []
        for q in range(10):
            for s in range(50):
                pair_indices.append([q, s])
        
        pair_indices = torch.tensor(pair_indices, dtype=torch.long)
        pair_cluster_ids = torch.randint(0, 3, (500,))
        
        splitter = DatasetSplitter(
            pair_indices=pair_indices,
            pair_cluster_ids=pair_cluster_ids
        )
        
        results = splitter.split()
        
        # Check that all pairs from same question stay together
        for split_idx in [results['train_idx'], results['val_idx'], results['test_idx']]:
            questions_in_split = pair_indices[split_idx, 0].unique()
            for q in questions_in_split:
                # Count pairs for this question in this split
                pairs_in_split = (pair_indices[split_idx, 0] == q).sum().item()
                # Should be exactly 50 (all sources for this question)
                assert pairs_in_split == 50, \
                    f"All 50 pairs for question {q} should be in same split"
    
    def test_unbalanced_cluster_distribution(self):
        """Test with highly unbalanced cluster distribution."""
        torch.manual_seed(42)
        
        # 100 pairs: 80 in cluster 0, 10 in cluster 1, 10 in cluster 2
        pair_indices = torch.stack([
            torch.arange(100),
            torch.arange(100)
        ], dim=1)
        
        pair_cluster_ids = torch.cat([
            torch.zeros(80, dtype=torch.long),
            torch.ones(10, dtype=torch.long),
            torch.full((10,), 2, dtype=torch.long)
        ])
        
        splitter = DatasetSplitter(
            pair_indices=pair_indices,
            pair_cluster_ids=pair_cluster_ids
        )
        
        results = splitter.split()
        
        # Should still produce valid splits
        assert len(results['train_idx']) > 0, "Train split not empty"
        assert len(results['val_idx']) > 0, "Val split not empty"
        assert len(results['test_idx']) > 0, "Test split not empty"
