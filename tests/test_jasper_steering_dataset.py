"""Tests for JASPERSteeringDataset."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from RAG_supporters.pytorch_datasets import JASPERSteeringDataset


@pytest.fixture
def mock_dataset_dir():
    """Create a mock dataset directory with all required files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Configuration
        config = {
            "embedding_dim": 64,
            "n_neg": 4,
            "n_pairs": 100,
            "n_questions": 50,
            "n_sources": 80,
            "n_keywords": 30,
            "n_clusters": 5,
            "steering_probabilities": {
                "zero": 0.25,
                "centroid": 0.25,
                "keyword": 0.25,
                "residual": 0.25,
            },
            "curriculum": {
                "zero_prob_start": 0.5,
                "zero_prob_end": 0.1,
                "epochs_total": 100,
            },
        }

        with open(tmpdir / "config.json", "w") as f:
            json.dump(config, f)

        # Create embeddings
        torch.save(torch.randn(50, 64), tmpdir / "question_embs.pt")
        torch.save(torch.randn(80, 64), tmpdir / "source_embs.pt")
        torch.save(torch.randn(30, 64), tmpdir / "keyword_embs.pt")
        torch.save(torch.randn(5, 64), tmpdir / "centroid_embs.pt")

        # Pair data
        pair_index = torch.stack(
            [
                torch.randint(0, 50, (100,)),  # q_idx
                torch.randint(0, 80, (100,)),  # s_idx
            ],
            dim=1,
        )
        torch.save(pair_index, tmpdir / "pair_index.pt")
        torch.save(torch.randint(0, 5, (100,)), tmpdir / "pair_cluster_id.pt")
        torch.save(torch.rand(100), tmpdir / "pair_relevance.pt")

        # Pair keyword IDs (list of lists)
        pair_keyword_ids = [
            [0, 1, 2],
            [1, 3],
            [2, 4, 5],
        ] * 34  # Repeat to get 102, then truncate
        pair_keyword_ids = pair_keyword_ids[:100]
        torch.save(pair_keyword_ids, tmpdir / "pair_keyword_ids.pt")

        # Steering tensors
        torch.save(torch.randn(100, 64), tmpdir / "steering_centroid.pt")
        torch.save(torch.randn(100, 64), tmpdir / "steering_keyword_weighted.pt")
        torch.save(torch.randn(100, 64), tmpdir / "steering_residual.pt")
        torch.save(torch.rand(100) * 2, tmpdir / "centroid_distances.pt")

        # Hard negatives
        torch.save(torch.randint(0, 80, (100, 4)), tmpdir / "hard_negatives.pt")
        torch.save(torch.randint(1, 5, (100, 4)), tmpdir / "negative_tiers.pt")

        # Split indices
        train_idx = torch.arange(0, 70)
        val_idx = torch.arange(70, 85)
        test_idx = torch.arange(85, 100)

        torch.save(train_idx, tmpdir / "train_idx.pt")
        torch.save(val_idx, tmpdir / "val_idx.pt")
        torch.save(test_idx, tmpdir / "test_idx.pt")

        yield tmpdir


def test_dataset_init(mock_dataset_dir):
    """Test dataset initialization."""
    dataset = JASPERSteeringDataset(mock_dataset_dir, split="train", epoch=0)

    assert len(dataset) == 70, "Train split should have 70 samples"
    assert dataset.embedding_dim == 64, "Embedding dimension should be 64"
    assert dataset.n_neg == 4, "Number of negatives should be 4"


def test_dataset_splits(mock_dataset_dir):
    """Test that different splits load correct number of samples."""
    train_dataset = JASPERSteeringDataset(mock_dataset_dir, split="train")
    val_dataset = JASPERSteeringDataset(mock_dataset_dir, split="val")
    test_dataset = JASPERSteeringDataset(mock_dataset_dir, split="test")

    assert len(train_dataset) == 70, "Train split should have 70 samples"
    assert len(val_dataset) == 15, "Val split should have 15 samples"
    assert len(test_dataset) == 15, "Test split should have 15 samples"


def test_getitem_schema(mock_dataset_dir):
    """Test that __getitem__ returns correct schema and shapes."""
    dataset = JASPERSteeringDataset(mock_dataset_dir, split="train")

    sample = dataset[0]

    # Check all expected keys present
    expected_keys = {
        "question_emb",
        "target_source_emb",
        "steering",
        "negative_embs",
        "cluster_id",
        "relevance",
        "centroid_distance",
        "steering_variant",
        "negative_tiers",
    }
    assert set(sample.keys()) == expected_keys, "Sample should have all expected keys"

    # Check shapes
    assert sample["question_emb"].shape == (64,), "Question embedding should be [D]"
    assert sample["target_source_emb"].shape == (64,), "Target source embedding should be [D]"
    assert sample["steering"].shape == (64,), "Steering should be [D]"
    assert sample["negative_embs"].shape == (4, 64), "Negative embeddings should be [N_neg, D]"
    assert sample["cluster_id"].shape == (), "Cluster ID should be scalar"
    assert sample["relevance"].shape == (), "Relevance should be scalar"
    assert sample["centroid_distance"].shape == (), "Centroid distance should be scalar"
    assert sample["steering_variant"].shape == (), "Steering variant should be scalar"
    assert sample["negative_tiers"].shape == (4,), "Negative tiers should be [N_neg]"


def test_steering_variant_distribution(mock_dataset_dir):
    """Test that steering variant distribution matches configuration over many samples."""
    dataset = JASPERSteeringDataset(mock_dataset_dir, split="train", epoch=0)

    # Sample many times to estimate distribution
    n_samples = 1000
    variant_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    for i in range(n_samples):
        sample = dataset[i % len(dataset)]
        variant = sample["steering_variant"].item()
        variant_counts[variant] += 1

    # Convert to proportions
    variant_probs = {k: v / n_samples for k, v in variant_counts.items()}

    # With curriculum at epoch 0, zero should be ~0.5, others ~0.166 each
    # Allow ±0.05 tolerance
    expected_zero_prob = 0.5
    expected_other_prob = 0.5 / 3  # ~0.166

    assert (
        abs(variant_probs[0] - expected_zero_prob) < 0.05
    ), f"Zero steering probability should be ~{expected_zero_prob}, got {variant_probs[0]}"

    for variant_id in [1, 2, 3]:
        assert (
            abs(variant_probs[variant_id] - expected_other_prob) < 0.05
        ), f"Variant {variant_id} probability should be ~{expected_other_prob}, got {variant_probs[variant_id]}"


def test_set_epoch_changes_probs(mock_dataset_dir):
    """Test that set_epoch changes steering probabilities."""
    dataset = JASPERSteeringDataset(mock_dataset_dir, split="train", epoch=0)

    # Initial probabilities (epoch 0)
    initial_probs = dataset.steering_probs.copy()

    # Set to later epoch
    dataset.set_epoch(50)
    mid_probs = dataset.steering_probs.copy()

    # Set to final epoch
    dataset.set_epoch(100)
    final_probs = dataset.steering_probs.copy()

    # Zero probability should decrease over epochs
    assert (
        initial_probs["zero"] > mid_probs["zero"]
    ), "Zero probability should decrease from epoch 0 to 50"
    assert (
        mid_probs["zero"] > final_probs["zero"]
    ), "Zero probability should decrease from epoch 50 to 100"

    # Other probabilities should increase
    assert (
        initial_probs["centroid"] < mid_probs["centroid"]
    ), "Centroid probability should increase from epoch 0 to 50"


def test_force_steering_zero(mock_dataset_dir):
    """Test forcing steering to zero variant."""
    dataset = JASPERSteeringDataset(mock_dataset_dir, split="train")

    dataset.force_steering("zero")

    # Sample multiple times, all should be zero
    for i in range(10):
        sample = dataset[i]
        assert sample["steering_variant"].item() == 0, "Forced steering should be zero"
        assert torch.allclose(
            sample["steering"], torch.zeros(64)
        ), "Steering tensor should be zeros"


def test_force_steering_centroid(mock_dataset_dir):
    """Test forcing steering to centroid variant."""
    dataset = JASPERSteeringDataset(mock_dataset_dir, split="train")

    dataset.force_steering("centroid")

    # Sample multiple times, all should be centroid
    for i in range(10):
        sample = dataset[i]
        assert sample["steering_variant"].item() == 1, "Forced steering should be centroid"


def test_force_steering_restore(mock_dataset_dir):
    """Test restoring stochastic steering after forcing."""
    dataset = JASPERSteeringDataset(mock_dataset_dir, split="train")

    # Force to zero
    dataset.force_steering("zero")
    sample1 = dataset[0]
    assert sample1["steering_variant"].item() == 0, "Should be forced to zero"

    # Restore stochastic
    dataset.force_steering(None)

    # Sample multiple times, should see variety
    variants_seen = set()
    for i in range(100):
        sample = dataset[i % len(dataset)]
        variants_seen.add(sample["steering_variant"].item())

    # Should see more than one variant over 100 samples
    assert len(variants_seen) > 1, "Should see multiple variants after restoring stochastic mode"


def test_reload_negatives(mock_dataset_dir):
    """Test hot-reloading hard negatives."""
    dataset = JASPERSteeringDataset(mock_dataset_dir, split="train")

    # Get initial negatives
    initial_sample = dataset[0]
    initial_negatives = initial_sample["negative_embs"].clone()

    # Modify negatives on disk
    new_negatives = torch.randint(0, 80, (100, 4))
    torch.save(new_negatives, mock_dataset_dir / "hard_negatives.pt")
    torch.save(torch.randint(1, 5, (100, 4)), mock_dataset_dir / "negative_tiers.pt")

    # Reload
    dataset.reload_negatives()

    # Get new sample
    new_sample = dataset[0]

    # Negatives should be different (with very high probability)
    assert not torch.equal(
        initial_negatives, new_sample["negative_embs"]
    ), "Negatives should change after reload"


def test_no_nan_inf(mock_dataset_dir):
    """Test that dataset never returns NaN or Inf values."""
    dataset = JASPERSteeringDataset(mock_dataset_dir, split="train")

    for i in range(min(20, len(dataset))):
        sample = dataset[i]

        # Check embeddings for NaN/Inf
        for key in ["question_emb", "target_source_emb", "steering", "negative_embs"]:
            tensor = sample[key]
            assert not torch.isnan(tensor).any(), f"NaN detected in {key} for sample {i}"
            assert not torch.isinf(tensor).any(), f"Inf detected in {key} for sample {i}"


def test_relevance_in_range(mock_dataset_dir):
    """Test that relevance scores are in [0, 1] range."""
    dataset = JASPERSteeringDataset(mock_dataset_dir, split="train")

    for i in range(min(20, len(dataset))):
        sample = dataset[i]
        relevance = sample["relevance"].item()

        assert 0 <= relevance <= 1, f"Relevance {relevance} out of range [0, 1] for sample {i}"


def test_centroid_distance_in_range(mock_dataset_dir):
    """Test that centroid distances are in [0, 2] range."""
    dataset = JASPERSteeringDataset(mock_dataset_dir, split="train")

    for i in range(min(20, len(dataset))):
        sample = dataset[i]
        distance = sample["centroid_distance"].item()

        assert (
            0 <= distance <= 2
        ), f"Centroid distance {distance} out of range [0, 2] for sample {i}"


def test_cluster_id_valid(mock_dataset_dir):
    """Test that cluster IDs are valid indices."""
    dataset = JASPERSteeringDataset(mock_dataset_dir, split="train")

    n_clusters = len(dataset.centroid_embs)

    for i in range(min(20, len(dataset))):
        sample = dataset[i]
        cluster_id = sample["cluster_id"].item()

        assert (
            0 <= cluster_id < n_clusters
        ), f"Cluster ID {cluster_id} out of valid range [0, {n_clusters}) for sample {i}"


def test_steering_variant_valid(mock_dataset_dir):
    """Test that steering variants are in valid range [0, 3]."""
    dataset = JASPERSteeringDataset(mock_dataset_dir, split="train")

    for i in range(min(20, len(dataset))):
        sample = dataset[i]
        variant = sample["steering_variant"].item()

        assert (
            0 <= variant <= 3
        ), f"Steering variant {variant} out of valid range [0, 3] for sample {i}"


def test_determinism_with_same_epoch(mock_dataset_dir):
    """Test that same epoch gives same samples (deterministic RNG)."""
    dataset1 = JASPERSteeringDataset(mock_dataset_dir, split="train", epoch=5)
    dataset2 = JASPERSteeringDataset(mock_dataset_dir, split="train", epoch=5)

    # Sample from both
    samples1 = [dataset1[i]["steering_variant"].item() for i in range(10)]
    samples2 = [dataset2[i]["steering_variant"].item() for i in range(10)]

    assert samples1 == samples2, "Same epoch should produce same steering variants (determinism)"


def test_different_epochs_different_samples(mock_dataset_dir):
    """Test that different epochs give different samples."""
    dataset1 = JASPERSteeringDataset(mock_dataset_dir, split="train", epoch=0)
    dataset2 = JASPERSteeringDataset(mock_dataset_dir, split="train", epoch=10)

    # Sample from both (many times to increase chance of difference)
    samples1 = [dataset1[i % len(dataset1)]["steering_variant"].item() for i in range(100)]
    samples2 = [dataset2[i % len(dataset2)]["steering_variant"].item() for i in range(100)]

    # Should be different with high probability
    assert samples1 != samples2, "Different epochs should produce different steering variants"


def test_invalid_split_raises(mock_dataset_dir):
    """Test that invalid split raises error."""
    with pytest.raises(ValueError, match="Split file not found"):
        JASPERSteeringDataset(mock_dataset_dir, split="invalid")


def test_missing_config_raises():
    """Test that missing config.json raises error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="Config file not found"):
            JASPERSteeringDataset(tmpdir, split="train")


def test_missing_dataset_dir_raises():
    """Test that missing dataset directory raises error."""
    with pytest.raises(ValueError, match="Dataset directory not found"):
        JASPERSteeringDataset("/nonexistent/path", split="train")


def test_index_out_of_bounds_raises(mock_dataset_dir):
    """Test that out-of-bounds index raises IndexError."""
    dataset = JASPERSteeringDataset(mock_dataset_dir, split="train")

    # Valid indices: 0 to len(dataset) - 1
    with pytest.raises(IndexError, match="Index .* out of range"):
        _ = dataset[len(dataset)]

    with pytest.raises(IndexError, match="Index .* out of range"):
        _ = dataset[-1]


def test_context_manager_support(mock_dataset_dir):
    """Test that dataset works as context manager."""
    with JASPERSteeringDataset(mock_dataset_dir, split="train") as dataset:
        assert len(dataset) == 70, "Dataset should be accessible within context"
        sample = dataset[0]
        assert "question_emb" in sample, "Should be able to access samples"

    # After exiting, tensors should be deleted (basic check)
    assert (
        not hasattr(dataset, "question_embs") or dataset.question_embs is None
    ), "Tensors should be cleaned up after context exit"


def test_device_placement_cpu(mock_dataset_dir):
    """Test dataset loads to CPU by default."""
    dataset = JASPERSteeringDataset(mock_dataset_dir, split="train")

    assert dataset.device == torch.device("cpu"), "Default device should be CPU"
    assert dataset.question_embs.device == torch.device(
        "cpu"
    ), "Question embeddings should be on CPU"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_placement_cuda(mock_dataset_dir):
    """Test dataset loads to CUDA when requested."""
    device = torch.device("cuda")
    dataset = JASPERSteeringDataset(mock_dataset_dir, split="train", device=device)

    assert dataset.device == device, "Device should be CUDA"
    assert dataset.question_embs.device == device, "Question embeddings should be on CUDA"
    assert dataset.source_embs.device == device, "Source embeddings should be on CUDA"


def test_create_combined_splits(mock_dataset_dir):
    """Test loading all splits at once."""
    splits = JASPERSteeringDataset.create_combined_splits(mock_dataset_dir, epoch=0)

    assert "train" in splits, "Should have train split"
    assert "val" in splits, "Should have val split"
    assert "test" in splits, "Should have test split"

    assert len(splits["train"]) == 70, "Train split should have 70 samples"
    assert len(splits["val"]) == 15, "Val split should have 15 samples"
    assert len(splits["test"]) == 15, "Test split should have 15 samples"


def test_referential_integrity_invalid_question_idx(mock_dataset_dir):
    """Test that invalid question index in pair_index raises error."""
    # Corrupt pair_index with out-of-bounds question index
    pair_index = torch.load(mock_dataset_dir / "pair_index.pt")
    pair_index[0, 0] = 999  # Invalid question index
    torch.save(pair_index, mock_dataset_dir / "pair_index.pt")

    with pytest.raises(ValueError, match="pair_index references question"):
        JASPERSteeringDataset(mock_dataset_dir, split="train")


def test_referential_integrity_invalid_source_idx(mock_dataset_dir):
    """Test that invalid source index in pair_index raises error."""
    # Corrupt pair_index with out-of-bounds source index
    pair_index = torch.load(mock_dataset_dir / "pair_index.pt")
    pair_index[0, 1] = 999  # Invalid source index
    torch.save(pair_index, mock_dataset_dir / "pair_index.pt")

    with pytest.raises(ValueError, match="pair_index references source"):
        JASPERSteeringDataset(mock_dataset_dir, split="train")


def test_referential_integrity_invalid_negative_idx(mock_dataset_dir):
    """Test that invalid index in hard_negatives raises error."""
    # Corrupt hard_negatives with out-of-bounds source index
    hard_negatives = torch.load(mock_dataset_dir / "hard_negatives.pt")
    hard_negatives[0, 0] = 999  # Invalid source index
    torch.save(hard_negatives, mock_dataset_dir / "hard_negatives.pt")

    with pytest.raises(ValueError, match="hard_negatives references source"):
        JASPERSteeringDataset(mock_dataset_dir, split="train")


def test_referential_integrity_split_indices(mock_dataset_dir):
    """Test that split indices reference valid pairs."""
    # Corrupt train split indices
    train_idx = torch.tensor([0, 1, 999])  # Invalid pair index
    torch.save(train_idx, mock_dataset_dir / "train_idx.pt")

    with pytest.raises(ValueError, match="Split indices reference pair"):
        JASPERSteeringDataset(mock_dataset_dir, split="train")


def test_memory_usage_logging(mock_dataset_dir, caplog):
    """Test that memory usage is logged during initialization."""
    import logging

    caplog.set_level(logging.INFO)

    dataset = JASPERSteeringDataset(mock_dataset_dir, split="train")

    # Check that memory was logged
    assert any(
        "memory=" in record.message.lower() for record in caplog.records
    ), "Memory usage should be logged during initialization"


def test_close_method(mock_dataset_dir):
    """Test explicit close method."""
    dataset = JASPERSteeringDataset(mock_dataset_dir, split="train")

    # Verify dataset is usable
    _ = dataset[0]

    # Close explicitly
    dataset.close()

    # Verify tensors are deleted
    with pytest.raises(AttributeError):
        _ = dataset.question_embs


def test_storage_format_detection_pt(mock_dataset_dir):
    """Test storage format detection defaults to PT."""
    dataset = JASPERSteeringDataset(mock_dataset_dir, split="train", storage_format="auto")

    assert dataset.storage_format == "pt", "Should detect PT format when no HDF5 file exists"


def test_storage_format_explicit_pt(mock_dataset_dir):
    """Test explicit PT format specification."""
    dataset = JASPERSteeringDataset(mock_dataset_dir, split="train", storage_format="pt")

    assert dataset.storage_format == "pt", "Should use PT format when explicitly requested"
    assert len(dataset) == 70, "Should load data correctly with PT format"


def test_memory_mapping_disabled_by_default(mock_dataset_dir):
    """Test that memory mapping is disabled for small datasets."""
    dataset = JASPERSteeringDataset(mock_dataset_dir, split="train")

    assert dataset.use_mmap is False, "Memory mapping should be disabled for small datasets"


def test_memory_mapping_explicit_enable(mock_dataset_dir):
    """Test explicit memory mapping enable."""
    dataset = JASPERSteeringDataset(mock_dataset_dir, split="train", use_mmap=True)

    assert dataset.use_mmap is True, "Memory mapping should be enabled when explicitly requested"
    assert len(dataset) == 70, "Should load data correctly with memory mapping"


def test_create_combined_splits_with_storage_format(mock_dataset_dir):
    """Test create_combined_splits with storage format parameter."""
    splits = JASPERSteeringDataset.create_combined_splits(mock_dataset_dir, storage_format="pt")

    assert "train" in splits, "Should have train split"
    assert "val" in splits, "Should have val split"
    assert "test" in splits, "Should have test split"
    assert all(s.storage_format == "pt" for s in splits.values()), "All splits should use PT format"


@pytest.mark.skipif(
    not hasattr(JASPERSteeringDataset, "convert_pt_to_hdf5"), reason="HDF5 conversion not available"
)
def test_hdf5_conversion_requires_h5py(mock_dataset_dir):
    """Test that HDF5 conversion checks for h5py."""
    try:
        import h5py

        pytest.skip("h5py is installed, cannot test error case")
    except ImportError:
        with pytest.raises(ImportError, match="h5py is required"):
            JASPERSteeringDataset.convert_pt_to_hdf5(mock_dataset_dir)


@pytest.mark.skipif(
    not hasattr(JASPERSteeringDataset, "convert_pt_to_hdf5"), reason="HDF5 conversion not available"
)
def test_hdf5_storage_format_error_without_h5py(mock_dataset_dir):
    """Test that requesting HDF5 format without h5py raises proper error."""
    try:
        import h5py

        pytest.skip("h5py is installed, cannot test error case")
    except ImportError:
        with pytest.raises(ValueError, match="h5py is not installed"):
            JASPERSteeringDataset(mock_dataset_dir, split="train", storage_format="hdf5")


def test_storage_format_auto_no_files():
    """Test storage format detection with no valid files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create only config (no data files)
        config = {"embedding_dim": 64, "n_neg": 4}
        with open(tmpdir / "config.json", "w") as f:
            json.dump(config, f)

        with pytest.raises(ValueError, match="No dataset files found"):
            JASPERSteeringDataset(tmpdir, split="train", storage_format="auto")


def test_invalid_storage_format():
    """Test invalid storage format raises error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        config = {"embedding_dim": 64, "n_neg": 4}
        with open(tmpdir / "config.json", "w") as f:
            json.dump(config, f)

        # Create minimal PT file
        torch.save(torch.randn(1, 64), tmpdir / "question_embs.pt")
        torch.save(torch.arange(1), tmpdir / "train_idx.pt")

        with pytest.raises(ValueError, match="Invalid storage_format"):
            JASPERSteeringDataset(tmpdir, split="train", storage_format="invalid")


# Integration tests with HDF5 (only run if h5py installed)
@pytest.mark.skipif(
    not hasattr(JASPERSteeringDataset, "convert_pt_to_hdf5"), reason="HDF5 conversion not available"
)
def test_hdf5_integration(mock_dataset_dir):
    """Test full HDF5 workflow: convert and load."""
    try:
        import h5py
    except ImportError:
        pytest.skip("h5py not installed")

    # Convert PT to HDF5
    JASPERSteeringDataset.convert_pt_to_hdf5(mock_dataset_dir)

    # Verify HDF5 file exists
    hdf5_path = Path(mock_dataset_dir) / "dataset.h5"
    assert hdf5_path.exists(), "HDF5 file should be created"

    # Load from HDF5
    dataset = JASPERSteeringDataset(mock_dataset_dir, split="train", storage_format="hdf5")

    assert dataset.storage_format == "hdf5", "Should use HDF5 format"
    assert len(dataset) == 70, "Should load correct number of samples"

    # Verify data can be accessed
    sample = dataset[0]
    assert "question_emb" in sample, "Sample should contain question embedding"
    assert sample["question_emb"].shape == (64,), "Question embedding shape should be correct"

    # Close dataset properly
    dataset.close()


@pytest.mark.skipif(
    not hasattr(JASPERSteeringDataset, "convert_pt_to_hdf5"), reason="HDF5 conversion not available"
)
def test_hdf5_auto_detection(mock_dataset_dir):
    """Test that storage format auto-detection prefers HDF5 when available."""
    try:
        import h5py
    except ImportError:
        pytest.skip("h5py not installed")

    # Convert to HDF5
    JASPERSteeringDataset.convert_pt_to_hdf5(mock_dataset_dir)

    # Auto-detect should prefer HDF5
    dataset = JASPERSteeringDataset(mock_dataset_dir, split="train", storage_format="auto")

    assert dataset.storage_format == "hdf5", "Auto-detection should prefer HDF5 when available"
    dataset.close()


def test_memory_mapping_auto_enable_large_dataset():
    """Test that memory mapping auto-enables for large datasets."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Simulate large dataset metadata (>10GB)
        config = {
            "embedding_dim": 768,
            "n_neg": 20,
            "n_pairs": 1000000,  # 1 million pairs -> >10GB
            "n_questions": 500000,
            "n_sources": 800000,
            "n_keywords": 50000,
            "n_clusters": 100,
            "steering_probabilities": {
                "zero": 0.25,
                "centroid": 0.25,
                "keyword": 0.25,
                "residual": 0.25,
            },
        }

        with open(tmpdir / "config.json", "w") as f:
            json.dump(config, f)

        # Create minimal PT files
        torch.save(torch.randn(10, 768), tmpdir / "question_embs.pt")
        torch.save(torch.randn(10, 768), tmpdir / "source_embs.pt")
        torch.save(torch.randn(5, 768), tmpdir / "keyword_embs.pt")
        torch.save(torch.randn(2, 768), tmpdir / "centroid_embs.pt")
        torch.save(
            torch.stack(
                [torch.zeros(5, dtype=torch.long), torch.zeros(5, dtype=torch.long)], dim=1
            ),
            tmpdir / "pair_index.pt",
        )
        torch.save(torch.zeros(5, dtype=torch.long), tmpdir / "pair_cluster_id.pt")
        torch.save(torch.zeros(5), tmpdir / "pair_relevance.pt")
        torch.save([[0, 1]] * 5, tmpdir / "pair_keyword_ids.pt")
        torch.save(torch.zeros((5, 20), dtype=torch.long), tmpdir / "hard_negatives.pt")
        torch.save(torch.ones((5, 20), dtype=torch.long), tmpdir / "negative_tiers.pt")
        torch.save(torch.randn(5, 768), tmpdir / "steering_centroid.pt")
        torch.save(torch.randn(5, 768), tmpdir / "steering_keyword_weighted.pt")
        torch.save(torch.randn(5, 768), tmpdir / "steering_residual.pt")
        torch.save(torch.rand(5), tmpdir / "centroid_distances.pt")
        torch.save(torch.arange(3), tmpdir / "train_idx.pt")
        torch.save(torch.arange(3, 4), tmpdir / "val_idx.pt")
        torch.save(torch.arange(4, 5), tmpdir / "test_idx.pt")

        # Auto-detect should enable mmap for large config
        dataset = JASPERSteeringDataset(tmpdir, split="train", device=torch.device("cpu"))

        assert (
            dataset.use_mmap is True
        ), "Memory mapping should auto-enable for large datasets (>10GB estimated)"
        dataset.close()
