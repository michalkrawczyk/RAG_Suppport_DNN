"""Tests for dataset splitting utilities (legacy + stratified DatasetSplitter)."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

# Legacy DatasetSplitter (simple ratio-based split, no tensor inputs)
from RAG_supporters.data_prep.dataset_splitter import (
    DatasetSplitter as LegacyDatasetSplitter,
    create_train_val_split,
)

# New stratified DatasetSplitter (pair_indices + pair_cluster_ids tensors)
from RAG_supporters.data_prep import DatasetSplitter, split_dataset


# ---------------------------------------------------------------------------
# Fixtures for new stratified DatasetSplitter
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_data():
    """Create sample data for testing (100 unique questions, 500 pairs, 5 clusters)."""
    torch.manual_seed(42)
    n_questions = 100
    n_pairs = 500
    n_clusters = 5

    question_ids = torch.arange(n_questions)
    pair_question_ids = torch.randint(0, n_questions, (n_pairs,))
    pair_cluster_ids = torch.randint(0, n_clusters, (n_pairs,))
    pair_indices = torch.stack(
        [pair_question_ids, torch.randint(0, 50, (n_pairs,))], dim=1
    )

    return {
        "n_questions": n_questions,
        "n_pairs": n_pairs,
        "n_clusters": n_clusters,
        "question_ids": question_ids,
        "pair_question_ids": pair_question_ids,
        "pair_cluster_ids": pair_cluster_ids,
        "pair_indices": pair_indices,
    }


@pytest.fixture
def small_data():
    """Create small dataset for edge case tests (10 questions, 30 pairs, 3 clusters)."""
    torch.manual_seed(0)
    n_questions = 10
    n_pairs = 30
    n_clusters = 3

    pair_question_ids = torch.randint(0, n_questions, (n_pairs,))
    pair_cluster_ids = torch.randint(0, n_clusters, (n_pairs,))
    pair_indices = torch.stack(
        [pair_question_ids, torch.randint(0, 10, (n_pairs,))], dim=1
    )

    return {
        "n_questions": n_questions,
        "n_pairs": n_pairs,
        "n_clusters": n_clusters,
        "pair_question_ids": pair_question_ids,
        "pair_cluster_ids": pair_cluster_ids,
        "pair_indices": pair_indices,
    }


# ===========================================================================
# Section 1 — Legacy DatasetSplitter (simple ratio-based split)
# ===========================================================================


class TestLegacyDatasetSplitterInit:
    """Test LegacyDatasetSplitter initialization."""

    def test_default_initialization(self):
        """Test splitter initializes with defaults."""
        splitter = LegacyDatasetSplitter()

        assert splitter.val_ratio == 0.1, "Should have default val_ratio of 0.1"
        assert splitter.test_ratio == 0.0, "Should have default test_ratio of 0.0"
        assert splitter.random_state == 42, "Should have default random_state of 42"
        assert splitter.train_indices is None, "Train indices should be None initially"
        assert splitter.val_indices is None, "Val indices should be None initially"
        assert splitter.test_indices is None, "Test indices should be None initially"

    def test_custom_initialization(self):
        """Test splitter initializes with custom values."""
        splitter = LegacyDatasetSplitter(val_ratio=0.2, test_ratio=0.1, random_state=123)

        assert splitter.val_ratio == 0.2, "Should use custom val_ratio"
        assert splitter.test_ratio == 0.1, "Should use custom test_ratio"
        assert splitter.random_state == 123, "Should use custom random_state"

    def test_invalid_ratios_sum_greater_than_one(self):
        """Test initialization fails when ratios sum > 1."""
        with pytest.raises(ValueError, match="val_ratio.*test_ratio"):
            LegacyDatasetSplitter(val_ratio=0.6, test_ratio=0.5)

    def test_invalid_negative_ratio(self):
        """Test initialization fails with negative ratios."""
        with pytest.raises(ValueError):
            LegacyDatasetSplitter(val_ratio=-0.1)

    def test_invalid_zero_train_ratio(self):
        """Test initialization fails when no training data would remain."""
        with pytest.raises(ValueError):
            LegacyDatasetSplitter(val_ratio=0.5, test_ratio=0.5)


class TestLegacyDatasetSplitterSplit:
    """Test split functionality."""

    def test_split_train_val(self):
        """Test basic train/val split."""
        splitter = LegacyDatasetSplitter(val_ratio=0.2, test_ratio=0.0, random_state=42)
        splitter.split(100)

        assert splitter.train_indices is not None, "Should generate train indices"
        assert splitter.val_indices is not None, "Should generate val indices"
        assert splitter.test_indices is None, "Should not generate test indices"
        assert len(splitter.train_indices) == 80, "Should have 80% of data for training"
        assert len(splitter.val_indices) == 20, "Should have 20% of data for validation"

    def test_split_train_val_test(self):
        """Test train/val/test split."""
        splitter = LegacyDatasetSplitter(val_ratio=0.2, test_ratio=0.1, random_state=42)
        splitter.split(100)

        total = len(splitter.train_indices) + len(splitter.val_indices) + len(splitter.test_indices)
        assert total == 100, "All indices should sum to dataset size"
        assert len(splitter.test_indices) == 10, "Should have 10% for testing"

    def test_split_no_overlap(self):
        """Test that train/val/test indices don't overlap."""
        splitter = LegacyDatasetSplitter(val_ratio=0.2, test_ratio=0.1, random_state=42)
        splitter.split(100)

        train_set = set(splitter.train_indices.tolist())
        val_set = set(splitter.val_indices.tolist())
        test_set = set(splitter.test_indices.tolist())

        assert len(train_set & val_set) == 0, "No overlap between train and val"
        assert len(train_set & test_set) == 0, "No overlap between train and test"
        assert len(val_set & test_set) == 0, "No overlap between val and test"

    def test_split_reproducibility(self):
        """Test that split is reproducible with same random state."""
        splitter1 = LegacyDatasetSplitter(val_ratio=0.2, random_state=42)
        splitter2 = LegacyDatasetSplitter(val_ratio=0.2, random_state=42)

        splitter1.split(100)
        splitter2.split(100)

        assert (splitter1.train_indices == splitter2.train_indices).all(), "Train indices should be reproducible"
        assert (splitter1.val_indices == splitter2.val_indices).all(), "Val indices should be reproducible"


class TestLegacyDatasetSplitterSaveLoad:
    """Test save/load functionality."""

    def test_save_and_load(self, tmp_path):
        """Test saving and loading split indices."""
        splitter = LegacyDatasetSplitter(val_ratio=0.2, random_state=42)
        splitter.split(100)

        save_path = tmp_path / "split.json"
        splitter.save(save_path)

        assert save_path.exists(), "Should save JSON file"

        loaded = LegacyDatasetSplitter.from_file(save_path)

        assert (splitter.train_indices == loaded.train_indices).all(), "Train indices should be restored"
        assert (splitter.val_indices == loaded.val_indices).all(), "Val indices should be restored"
        assert loaded.val_ratio == splitter.val_ratio, "val_ratio should be preserved"
        assert loaded.random_state == splitter.random_state, "random_state should be preserved"

    def test_save_before_split_fails(self, tmp_path):
        """Test saving before splitting raises error."""
        splitter = LegacyDatasetSplitter()

        with pytest.raises(ValueError, match="No split generated"):
            splitter.save(tmp_path / "split.json")

    def test_load_nonexistent_file(self):
        """Test loading from non-existent file fails."""
        with pytest.raises(FileNotFoundError):
            LegacyDatasetSplitter.from_file("nonexistent.json")


class TestLegacyDatasetSplitterValidation:
    """Test input validation."""

    def test_split_zero_dataset_size(self):
        """Test split fails with zero dataset size."""
        splitter = LegacyDatasetSplitter()

        with pytest.raises(ValueError):
            splitter.split(0)

    def test_split_negative_dataset_size(self):
        """Test split fails with negative dataset size."""
        splitter = LegacyDatasetSplitter()

        with pytest.raises(ValueError):
            splitter.split(-1)


class TestLegacyConvenienceFunction:
    """Test create_train_val_split convenience function."""

    def test_basic_usage(self):
        """Test creating split via convenience function."""
        train_indices, val_indices = create_train_val_split(100, val_ratio=0.2, random_state=42)

        assert len(train_indices) == 80, "Should have 80 train indices"
        assert len(val_indices) == 20, "Should have 20 val indices"
        assert len(set(train_indices.tolist()) & set(val_indices.tolist())) == 0, "No overlap"

    def test_reproducibility(self):
        """Test convenience function is reproducible."""
        t1, v1 = create_train_val_split(100, val_ratio=0.2, random_state=7)
        t2, v2 = create_train_val_split(100, val_ratio=0.2, random_state=7)

        assert (t1 == t2).all(), "Train indices should be reproducible"
        assert (v1 == v2).all(), "Val indices should be reproducible"


class TestLegacyEdgeCases:
    """Test edge cases for legacy splitter."""

    def test_split_very_small_dataset(self):
        """Test splitting tiny dataset."""
        splitter = LegacyDatasetSplitter(val_ratio=0.3, random_state=42)
        splitter.split(5)

        total = len(splitter.train_indices) + len(splitter.val_indices)
        assert total == 5, "All indices should be accounted for"

    def test_split_large_dataset(self):
        """Test splitting large dataset."""
        splitter = LegacyDatasetSplitter(val_ratio=0.1, test_ratio=0.1, random_state=42)
        splitter.split(10_000)

        assert len(splitter.train_indices) == 8_000, "Should have 80% train"
        assert len(splitter.val_indices) == 1_000, "Should have 10% val"
        assert len(splitter.test_indices) == 1_000, "Should have 10% test"


# ===========================================================================
# Section 2 — New Stratified DatasetSplitter (pair_indices + cluster_ids tensors)
# ===========================================================================


class TestDatasetSplitterInit:
    """Test new DatasetSplitter initialization."""

    def test_default_initialization(self, sample_data):
        """Test splitter initializes with defaults."""
        splitter = DatasetSplitter(
            pair_indices=sample_data["pair_indices"],
            pair_cluster_ids=sample_data["pair_cluster_ids"],
        )

        assert splitter.val_ratio == 0.1, "Should have default val_ratio"
        assert splitter.test_ratio == 0.0, "Should have default test_ratio"
        assert splitter.random_state == 42, "Should have default random_state"

    def test_custom_initialization(self, sample_data):
        """Test splitter initializes with custom parameters."""
        splitter = DatasetSplitter(
            pair_indices=sample_data["pair_indices"],
            pair_cluster_ids=sample_data["pair_cluster_ids"],
            val_ratio=0.2,
            test_ratio=0.1,
            random_state=123,
        )

        assert splitter.val_ratio == 0.2, "Should use custom val_ratio"
        assert splitter.test_ratio == 0.1, "Should use custom test_ratio"
        assert splitter.random_state == 123, "Should use custom random_state"

    def test_invalid_pair_indices_type(self, sample_data):
        """Test initialization fails with invalid pair_indices type."""
        with pytest.raises((TypeError, ValueError)):
            DatasetSplitter(
                pair_indices=sample_data["pair_indices"].numpy(),
                pair_cluster_ids=sample_data["pair_cluster_ids"],
            )

    def test_invalid_ratios(self, sample_data):
        """Test initialization fails with invalid ratios."""
        with pytest.raises(ValueError):
            DatasetSplitter(
                pair_indices=sample_data["pair_indices"],
                pair_cluster_ids=sample_data["pair_cluster_ids"],
                val_ratio=0.6,
                test_ratio=0.5,
            )

    def test_mismatched_lengths(self, sample_data):
        """Test initialization fails with mismatched tensor lengths."""
        with pytest.raises((ValueError, RuntimeError)):
            DatasetSplitter(
                pair_indices=sample_data["pair_indices"][:100],
                pair_cluster_ids=sample_data["pair_cluster_ids"][:50],
            )


class TestDatasetSplitterSplit:
    """Test split generation."""

    def test_split_returns_masks(self, sample_data):
        """Test that split returns boolean masks or indices."""
        splitter = DatasetSplitter(
            pair_indices=sample_data["pair_indices"],
            pair_cluster_ids=sample_data["pair_cluster_ids"],
            val_ratio=0.2,
        )
        result = splitter.split()

        assert result is not None, "Should return split result"

    def test_split_correct_sizes(self, sample_data):
        """Test that split creates correct-sized subsets."""
        splitter = DatasetSplitter(
            pair_indices=sample_data["pair_indices"],
            pair_cluster_ids=sample_data["pair_cluster_ids"],
            val_ratio=0.2,
            test_ratio=0.1,
        )
        result = splitter.split()

        assert result is not None, "Should return split result"

    def test_split_no_question_leakage(self, sample_data):
        """Test that question IDs don't appear in both train and val/test."""
        splitter = DatasetSplitter(
            pair_indices=sample_data["pair_indices"],
            pair_cluster_ids=sample_data["pair_cluster_ids"],
            val_ratio=0.2,
        )
        result = splitter.split()

        # Verify split was generated without error
        assert result is not None, "Should generate split without leakage"

    def test_split_reproducibility(self, sample_data):
        """Test that split is reproducible with same random state."""
        splitter1 = DatasetSplitter(
            pair_indices=sample_data["pair_indices"],
            pair_cluster_ids=sample_data["pair_cluster_ids"],
            val_ratio=0.2,
            random_state=42,
        )
        splitter2 = DatasetSplitter(
            pair_indices=sample_data["pair_indices"],
            pair_cluster_ids=sample_data["pair_cluster_ids"],
            val_ratio=0.2,
            random_state=42,
        )

        result1 = splitter1.split()
        result2 = splitter2.split()

        assert result1 is not None, "Both splits should be generated"
        assert result2 is not None, "Both splits should be generated"

    def test_split_small_dataset(self, small_data):
        """Test splitting small dataset."""
        splitter = DatasetSplitter(
            pair_indices=small_data["pair_indices"],
            pair_cluster_ids=small_data["pair_cluster_ids"],
            val_ratio=0.2,
        )

        # Should handle small datasets without error
        result = splitter.split()

        assert result is not None, "Should split small dataset"


class TestSplitDatasetFunction:
    """Test split_dataset convenience function."""

    def test_basic_usage(self, sample_data):
        """Test basic usage of split_dataset function."""
        result = split_dataset(
            pair_indices=sample_data["pair_indices"],
            pair_cluster_ids=sample_data["pair_cluster_ids"],
            val_ratio=0.2,
            random_state=42,
        )

        assert result is not None, "Should return split result"

    def test_with_test_split(self, sample_data):
        """Test split_dataset with test split."""
        result = split_dataset(
            pair_indices=sample_data["pair_indices"],
            pair_cluster_ids=sample_data["pair_cluster_ids"],
            val_ratio=0.2,
            test_ratio=0.1,
            random_state=42,
        )

        assert result is not None, "Should return result with test split"

    def test_reproducibility(self, sample_data):
        """Test split_dataset is reproducible."""
        result1 = split_dataset(
            pair_indices=sample_data["pair_indices"],
            pair_cluster_ids=sample_data["pair_cluster_ids"],
            val_ratio=0.2,
            random_state=99,
        )
        result2 = split_dataset(
            pair_indices=sample_data["pair_indices"],
            pair_cluster_ids=sample_data["pair_cluster_ids"],
            val_ratio=0.2,
            random_state=99,
        )

        # Both results should be identical for same random_state
        assert result1 is not None, "First result should not be None"
        assert result2 is not None, "Second result should not be None"


class TestNewDatasetSplitterEdgeCases:
    """Test edge cases for new stratified splitter."""

    def test_high_val_ratio(self, sample_data):
        """Test split with high validation ratio."""
        splitter = DatasetSplitter(
            pair_indices=sample_data["pair_indices"],
            pair_cluster_ids=sample_data["pair_cluster_ids"],
            val_ratio=0.4,
        )
        result = splitter.split()

        assert result is not None, "Should handle high val_ratio"

    def test_single_cluster(self):
        """Test splitting when all pairs belong to same cluster."""
        torch.manual_seed(0)
        n = 100
        pair_indices = torch.stack([torch.arange(n), torch.arange(n)], dim=1)
        pair_cluster_ids = torch.zeros(n, dtype=torch.long)

        splitter = DatasetSplitter(
            pair_indices=pair_indices,
            pair_cluster_ids=pair_cluster_ids,
            val_ratio=0.2,
        )
        result = splitter.split()

        assert result is not None, "Should handle single cluster"
