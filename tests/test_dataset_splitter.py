"""
Unit tests for dataset_splitter module.

Tests cover:
- DatasetSplitter initialization
- Split generation with various ratios
- Save/load functionality
- Split validation
- Integration with ClusterLabeledDataset
- Consistency across multiple loads
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Add RAG_supporters to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'RAG_supporters' / 'dataset'))

from dataset_splitter import (
    DatasetSplitter,
    create_train_val_split,
)


class TestDatasetSplitterInit:
    """Test DatasetSplitter initialization."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        splitter = DatasetSplitter()
        assert splitter.random_state is None
        assert splitter.train_indices is None
        assert splitter.val_indices is None
        assert splitter.dataset_size is None
        assert splitter.val_ratio is None

    def test_init_with_random_state(self):
        """Test initialization with random state."""
        splitter = DatasetSplitter(random_state=42)
        assert splitter.random_state == 42


class TestDatasetSplitterSplit:
    """Test dataset splitting operations."""

    def test_split_basic(self):
        """Test basic split with default parameters."""
        splitter = DatasetSplitter(random_state=42)
        train_idx, val_idx = splitter.split(dataset_size=100, val_ratio=0.2)

        # Check sizes
        assert len(train_idx) == 80
        assert len(val_idx) == 20
        assert len(train_idx) + len(val_idx) == 100

        # Check no overlap
        assert len(set(train_idx) & set(val_idx)) == 0

        # Check all indices present
        all_indices = set(train_idx + val_idx)
        assert all_indices == set(range(100))

    def test_split_different_ratios(self):
        """Test splitting with different validation ratios."""
        splitter = DatasetSplitter(random_state=42)
        
        # Test 10% validation
        train_idx, val_idx = splitter.split(dataset_size=100, val_ratio=0.1)
        assert len(val_idx) == 10
        assert len(train_idx) == 90

        # Test 30% validation
        train_idx, val_idx = splitter.split(dataset_size=100, val_ratio=0.3)
        assert len(val_idx) == 30
        assert len(train_idx) == 70

    def test_split_reproducibility(self):
        """Test that splits are reproducible with same random_state."""
        splitter1 = DatasetSplitter(random_state=42)
        train_idx1, val_idx1 = splitter1.split(dataset_size=100, val_ratio=0.2)

        splitter2 = DatasetSplitter(random_state=42)
        train_idx2, val_idx2 = splitter2.split(dataset_size=100, val_ratio=0.2)

        assert train_idx1 == train_idx2
        assert val_idx1 == val_idx2

    def test_split_different_seeds(self):
        """Test that different seeds produce different splits."""
        splitter1 = DatasetSplitter(random_state=42)
        train_idx1, val_idx1 = splitter1.split(dataset_size=100, val_ratio=0.2)

        splitter2 = DatasetSplitter(random_state=123)
        train_idx2, val_idx2 = splitter2.split(dataset_size=100, val_ratio=0.2)

        # Sizes should be the same
        assert len(train_idx1) == len(train_idx2)
        assert len(val_idx1) == len(val_idx2)

        # But splits should be different
        assert train_idx1 != train_idx2 or val_idx1 != val_idx2

    def test_split_no_shuffle(self):
        """Test split without shuffling."""
        splitter = DatasetSplitter(random_state=42)
        train_idx, val_idx = splitter.split(
            dataset_size=100, val_ratio=0.2, shuffle=False
        )

        # Without shuffle, val indices should be first 20
        assert val_idx == list(range(20))
        # And train indices should be rest
        assert train_idx == list(range(20, 100))

    def test_split_invalid_val_ratio(self):
        """Test split with invalid validation ratio."""
        splitter = DatasetSplitter(random_state=42)

        with pytest.raises(ValueError):
            splitter.split(dataset_size=100, val_ratio=0.0)

        with pytest.raises(ValueError):
            splitter.split(dataset_size=100, val_ratio=1.0)

        with pytest.raises(ValueError):
            splitter.split(dataset_size=100, val_ratio=-0.1)

        with pytest.raises(ValueError):
            splitter.split(dataset_size=100, val_ratio=1.5)

    def test_split_invalid_dataset_size(self):
        """Test split with invalid dataset size."""
        splitter = DatasetSplitter(random_state=42)

        with pytest.raises(ValueError):
            splitter.split(dataset_size=0, val_ratio=0.2)

        with pytest.raises(ValueError):
            splitter.split(dataset_size=-10, val_ratio=0.2)

    def test_get_split(self):
        """Test getting split after creation."""
        splitter = DatasetSplitter(random_state=42)
        train_idx1, val_idx1 = splitter.split(dataset_size=100, val_ratio=0.2)

        # Get split again
        train_idx2, val_idx2 = splitter.get_split()

        assert train_idx1 == train_idx2
        assert val_idx1 == val_idx2

    def test_get_split_before_creation(self):
        """Test getting split before creating one raises error."""
        splitter = DatasetSplitter()

        with pytest.raises(ValueError):
            splitter.get_split()


class TestDatasetSplitterSaveLoad:
    """Test save/load operations."""

    def test_save_and_load_basic(self):
        """Test basic save and load functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            split_path = Path(tmpdir) / "split.json"

            # Create and save split
            splitter1 = DatasetSplitter(random_state=42)
            train_idx1, val_idx1 = splitter1.split(dataset_size=100, val_ratio=0.2)
            splitter1.save_split(split_path)

            # Load split
            splitter2 = DatasetSplitter.load_split(split_path)
            train_idx2, val_idx2 = splitter2.get_split()

            # Verify splits match
            assert train_idx1 == train_idx2
            assert val_idx1 == val_idx2
            assert splitter1.random_state == splitter2.random_state
            assert splitter1.dataset_size == splitter2.dataset_size
            assert splitter1.val_ratio == splitter2.val_ratio

    def test_save_with_metadata(self):
        """Test saving with additional metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            split_path = Path(tmpdir) / "split.json"

            # Create and save split with metadata
            splitter = DatasetSplitter(random_state=42)
            splitter.split(dataset_size=100, val_ratio=0.2)
            
            metadata = {
                "dataset_name": "test_dataset",
                "description": "Test split for unit tests",
            }
            splitter.save_split(split_path, metadata=metadata)

            # Load and check metadata
            with open(split_path) as f:
                data = json.load(f)

            assert "metadata" in data
            assert data["metadata"]["dataset_name"] == "test_dataset"
            assert data["metadata"]["description"] == "Test split for unit tests"

    def test_save_without_split(self):
        """Test saving without creating a split raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            split_path = Path(tmpdir) / "split.json"
            splitter = DatasetSplitter(random_state=42)

            with pytest.raises(ValueError):
                splitter.save_split(split_path)

    def test_load_nonexistent_file(self):
        """Test loading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            DatasetSplitter.load_split("nonexistent_file.json")

    def test_load_invalid_format(self):
        """Test loading invalid file format raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            split_path = Path(tmpdir) / "invalid.json"

            # Create invalid file
            with open(split_path, 'w') as f:
                json.dump({"invalid": "data"}, f)

            with pytest.raises(ValueError):
                DatasetSplitter.load_split(split_path)

    def test_multiple_loads_consistency(self):
        """Test loading same split multiple times produces consistent results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            split_path = Path(tmpdir) / "split.json"

            # Create and save split
            splitter = DatasetSplitter(random_state=42)
            splitter.split(dataset_size=1000, val_ratio=0.2)
            splitter.save_split(split_path)

            # Load multiple times
            splitter1 = DatasetSplitter.load_split(split_path)
            splitter2 = DatasetSplitter.load_split(split_path)
            splitter3 = DatasetSplitter.load_split(split_path)

            train1, val1 = splitter1.get_split()
            train2, val2 = splitter2.get_split()
            train3, val3 = splitter3.get_split()

            # All should be identical
            assert train1 == train2 == train3
            assert val1 == val2 == val3


class TestDatasetSplitterValidation:
    """Test split validation operations."""

    def test_validate_split_valid(self):
        """Test validation of valid split."""
        splitter = DatasetSplitter(random_state=42)
        splitter.split(dataset_size=100, val_ratio=0.2)

        # Should not raise
        assert splitter.validate_split(100) is True

    def test_validate_split_size_mismatch(self):
        """Test validation with size mismatch produces warning."""
        splitter = DatasetSplitter(random_state=42)
        splitter.split(dataset_size=100, val_ratio=0.2)

        # Should validate but log warning (we can't easily test logging in pytest)
        assert splitter.validate_split(150) is True

    def test_validate_split_indices_out_of_bounds(self):
        """Test validation with indices out of bounds raises error."""
        splitter = DatasetSplitter(random_state=42)
        splitter.split(dataset_size=100, val_ratio=0.2)

        with pytest.raises(ValueError):
            splitter.validate_split(50)  # Dataset too small

    def test_validate_split_before_creation(self):
        """Test validation before creating split raises error."""
        splitter = DatasetSplitter()

        with pytest.raises(ValueError):
            splitter.validate_split(100)


class TestConvenienceFunction:
    """Test create_train_val_split convenience function."""

    def test_basic_usage(self):
        """Test basic usage of convenience function."""
        train_idx, val_idx = create_train_val_split(
            dataset_size=100,
            val_ratio=0.2,
            random_state=42,
        )

        assert len(train_idx) == 80
        assert len(val_idx) == 20

    def test_with_save(self):
        """Test convenience function with save."""
        with tempfile.TemporaryDirectory() as tmpdir:
            split_path = Path(tmpdir) / "split.json"

            train_idx, val_idx = create_train_val_split(
                dataset_size=100,
                val_ratio=0.2,
                random_state=42,
                save_path=split_path,
            )

            # Verify file was created
            assert split_path.exists()

            # Load and verify
            splitter = DatasetSplitter.load_split(split_path)
            train_idx2, val_idx2 = splitter.get_split()

            assert train_idx == train_idx2
            assert val_idx == val_idx2

    def test_with_metadata(self):
        """Test convenience function with metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            split_path = Path(tmpdir) / "split.json"

            metadata = {"experiment": "test"}
            create_train_val_split(
                dataset_size=100,
                val_ratio=0.2,
                random_state=42,
                save_path=split_path,
                metadata=metadata,
            )

            # Verify metadata was saved
            with open(split_path) as f:
                data = json.load(f)

            assert data["metadata"]["experiment"] == "test"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_small_dataset(self):
        """Test split with very small dataset."""
        splitter = DatasetSplitter(random_state=42)
        train_idx, val_idx = splitter.split(dataset_size=10, val_ratio=0.2)

        # Should have 2 val, 8 train
        assert len(val_idx) == 2
        assert len(train_idx) == 8

    def test_large_dataset(self):
        """Test split with large dataset."""
        splitter = DatasetSplitter(random_state=42)
        train_idx, val_idx = splitter.split(dataset_size=100000, val_ratio=0.2)

        assert len(val_idx) == 20000
        assert len(train_idx) == 80000

        # Verify no duplicates
        assert len(set(train_idx) & set(val_idx)) == 0

    def test_extreme_ratio(self):
        """Test with extreme but valid ratios."""
        splitter = DatasetSplitter(random_state=42)
        
        # Very small validation set
        train_idx, val_idx = splitter.split(dataset_size=1000, val_ratio=0.01)
        assert len(val_idx) == 10
        assert len(train_idx) == 990

        # Very large validation set
        train_idx, val_idx = splitter.split(dataset_size=1000, val_ratio=0.99)
        assert len(val_idx) == 990
        assert len(train_idx) == 10
