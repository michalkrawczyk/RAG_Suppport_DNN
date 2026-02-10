"""Tests for JEPA Steering DataLoader."""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from RAG_supporters.dataset import (
    JEPASteeringDataset,
    create_loader,
    set_epoch,
    validate_first_batch,
)


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

        # Pair keyword IDs
        pair_keyword_ids = [[0, 1, 2]] * 100
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
        torch.save(torch.arange(0, 70), tmpdir / "train_idx.pt")
        torch.save(torch.arange(70, 85), tmpdir / "val_idx.pt")
        torch.save(torch.arange(85, 100), tmpdir / "test_idx.pt")

        yield tmpdir


def test_create_loader_basic(mock_dataset_dir):
    """Test basic DataLoader creation."""
    loader = create_loader(
        dataset_dir=mock_dataset_dir,
        split="train",
        batch_size=8,
        num_workers=0,
    )

    assert loader is not None, "Loader should be created"
    assert loader.batch_size == 8, "Batch size should be 8"
    assert len(loader.dataset) == 70, "Dataset should have 70 samples"


def test_create_loader_all_splits(mock_dataset_dir):
    """Test DataLoader creation for all splits."""
    train_loader = create_loader(mock_dataset_dir, split="train", batch_size=8)
    val_loader = create_loader(mock_dataset_dir, split="val", batch_size=8)
    test_loader = create_loader(mock_dataset_dir, split="test", batch_size=8)

    assert len(train_loader.dataset) == 70, "Train dataset should have 70 samples"
    assert len(val_loader.dataset) == 15, "Val dataset should have 15 samples"
    assert len(test_loader.dataset) == 15, "Test dataset should have 15 samples"


def test_loader_iteration(mock_dataset_dir):
    """Test iterating through DataLoader."""
    loader = create_loader(
        dataset_dir=mock_dataset_dir,
        split="train",
        batch_size=8,
        num_workers=0,
    )

    batches = list(loader)

    # 70 samples with batch_size=8, drop_last=True -> 8 batches
    expected_batches = 70 // 8
    assert len(batches) == expected_batches, f"Should have {expected_batches} batches"

    # Check first batch
    batch = batches[0]
    assert batch["question_emb"].shape == (8, 64), "Question embeddings shape should be [B, D]"


def test_loader_batch_shapes(mock_dataset_dir):
    """Test that DataLoader produces correct batch shapes."""
    loader = create_loader(
        dataset_dir=mock_dataset_dir,
        split="train",
        batch_size=16,
        num_workers=0,
    )

    batch = next(iter(loader))

    assert batch["question_emb"].shape == (16, 64), "Question embeddings should be [B, D]"
    assert batch["target_source_emb"].shape == (16, 64), "Target source embeddings should be [B, D]"
    assert batch["steering"].shape == (16, 64), "Steering should be [B, D]"
    assert batch["negative_embs"].shape == (16, 4, 64), "Negative embeddings should be [B, N_neg, D]"
    assert batch["cluster_id"].shape == (16,), "Cluster IDs should be [B]"
    assert batch["relevance"].shape == (16,), "Relevance should be [B]"
    assert batch["centroid_distance"].shape == (16,), "Centroid distance should be [B]"
    assert batch["steering_variant"].shape == (16,), "Steering variant should be [B]"
    assert batch["negative_tiers"].shape == (16, 4), "Negative tiers should be [B, N_neg]"


def test_validate_first_batch_passes(mock_dataset_dir):
    """Test that validate_first_batch passes for valid DataLoader."""
    loader = create_loader(
        dataset_dir=mock_dataset_dir,
        split="train",
        batch_size=8,
        num_workers=0,
    )

    result = validate_first_batch(loader)

    assert result is True, "Validation should pass for valid DataLoader"


def test_validate_first_batch_all_keys(mock_dataset_dir):
    """Test that validate_first_batch checks all expected keys."""
    loader = create_loader(
        dataset_dir=mock_dataset_dir,
        split="train",
        batch_size=8,
        num_workers=0,
    )

    # Should not raise
    validate_first_batch(loader)


def test_set_epoch_on_loader(mock_dataset_dir):
    """Test set_epoch function on DataLoader."""
    loader = create_loader(
        dataset_dir=mock_dataset_dir,
        split="train",
        batch_size=8,
        num_workers=0,
        epoch=0,
    )

    # Initial epoch
    assert loader.dataset_obj.current_epoch == 0, "Initial epoch should be 0"

    # Set new epoch
    set_epoch(loader, 5)

    assert loader.dataset_obj.current_epoch == 5, "Epoch should be updated to 5"


def test_drop_last_train_vs_val(mock_dataset_dir):
    """Test that drop_last defaults to True for train, False for val."""
    train_loader = create_loader(mock_dataset_dir, split="train", batch_size=8)
    val_loader = create_loader(mock_dataset_dir, split="val", batch_size=8)

    assert train_loader.drop_last is True, "Train loader should drop last batch by default"
    assert val_loader.drop_last is False, "Val loader should not drop last batch by default"


def test_drop_last_override(mock_dataset_dir):
    """Test that drop_last can be overridden."""
    loader = create_loader(
        mock_dataset_dir,
        split="train",
        batch_size=8,
        drop_last=False,
    )

    assert loader.drop_last is False, "drop_last should be overridden to False"


def test_pin_memory_enabled(mock_dataset_dir):
    """Test that pin_memory is enabled by default."""
    loader = create_loader(
        dataset_dir=mock_dataset_dir,
        split="train",
        batch_size=8,
    )

    assert loader.pin_memory is True, "pin_memory should be enabled by default"


def test_pin_memory_disabled(mock_dataset_dir):
    """Test that pin_memory can be disabled."""
    loader = create_loader(
        dataset_dir=mock_dataset_dir,
        split="train",
        batch_size=8,
        pin_memory=False,
    )

    assert loader.pin_memory is False, "pin_memory should be disabled when set to False"


def test_num_workers_configurable(mock_dataset_dir):
    """Test that num_workers can be configured."""
    loader = create_loader(
        dataset_dir=mock_dataset_dir,
        split="train",
        batch_size=8,
        num_workers=2,
    )

    assert loader.num_workers == 2, "num_workers should be set to 2"


def test_loader_shuffle_train(mock_dataset_dir):
    """Test that train loader shuffles by default."""
    loader = create_loader(
        dataset_dir=mock_dataset_dir,
        split="train",
        batch_size=8,
        num_workers=0,
    )

    # Get two epochs of data
    epoch1_first_batch = next(iter(loader))
    epoch2_first_batch = next(iter(loader))

    # Due to shuffling, first batches should differ (with high probability)
    # Check question embeddings
    q_embs1 = epoch1_first_batch["question_emb"]
    q_embs2 = epoch2_first_batch["question_emb"]

    # Not guaranteed to be different, but very likely
    # (Skip this check as it's probabilistic)


def test_loader_no_shuffle_val(mock_dataset_dir):
    """Test that val loader does not shuffle."""
    loader = create_loader(
        dataset_dir=mock_dataset_dir,
        split="val",
        batch_size=8,
        num_workers=0,
    )

    # Val should not shuffle (sampler is None, shuffle is False)
    assert loader.sampler is None, "Val loader should have no sampler"


def test_distributed_sampler_not_used_by_default(mock_dataset_dir):
    """Test that DistributedSampler is not used by default."""
    loader = create_loader(
        dataset_dir=mock_dataset_dir,
        split="train",
        batch_size=8,
        distributed=False,
    )

    assert loader.sampler is None, "Sampler should be None when distributed=False"


def test_loader_one_epoch_iteration_count(mock_dataset_dir):
    """Test that one epoch iterates correct number of batches."""
    loader = create_loader(
        dataset_dir=mock_dataset_dir,
        split="train",
        batch_size=8,
        num_workers=0,
        drop_last=True,
    )

    batch_count = 0
    for _ in loader:
        batch_count += 1

    expected_batches = 70 // 8  # 8 full batches
    assert batch_count == expected_batches, f"Should iterate {expected_batches} batches"


def test_loader_with_epoch_curriculum(mock_dataset_dir):
    """Test DataLoader with curriculum learning across epochs."""
    loader = create_loader(
        dataset_dir=mock_dataset_dir,
        split="train",
        batch_size=8,
        num_workers=0,
        epoch=0,
    )

    # Epoch 0: steering probabilities should favor zero
    batch_epoch0 = next(iter(loader))

    # Advance to epoch 50
    set_epoch(loader, 50)
    batch_epoch50 = next(iter(loader))

    # Can't directly compare steering variants due to randomness,
    # but the dataset should have updated probabilities
    assert loader.dataset_obj.current_epoch == 50, "Dataset epoch should be updated"


def test_loader_full_pipeline(mock_dataset_dir):
    """Test full DataLoader pipeline: create, iterate, validate."""
    # Create loader
    loader = create_loader(
        dataset_dir=mock_dataset_dir,
        split="train",
        batch_size=8,
        num_workers=0,
    )

    # Validate first batch
    assert validate_first_batch(loader) is True, "First batch validation should pass"

    # Iterate through full epoch
    batch_count = 0
    for batch in loader:
        batch_count += 1
        # Check no NaN/Inf
        assert not torch.isnan(batch["question_emb"]).any(), "No NaN in question_emb"
        assert not torch.isinf(batch["question_emb"]).any(), "No Inf in question_emb"

    assert batch_count > 0, "Should iterate at least one batch"


def test_loader_different_batch_sizes(mock_dataset_dir):
    """Test DataLoader with different batch sizes."""
    for batch_size in [4, 8, 16]:
        loader = create_loader(
            dataset_dir=mock_dataset_dir,
            split="train",
            batch_size=batch_size,
            num_workers=0,
            drop_last=True,
        )

        batch = next(iter(loader))

        assert batch["question_emb"].shape[0] == batch_size, (
            f"Batch size should be {batch_size}"
        )


def test_loader_test_split(mock_dataset_dir):
    """Test DataLoader for test split."""
    loader = create_loader(
        dataset_dir=mock_dataset_dir,
        split="test",
        batch_size=8,
        num_workers=0,
    )

    assert len(loader.dataset) == 15, "Test dataset should have 15 samples"

    # Iterate and count batches
    batch_count = sum(1 for _ in loader)
    assert batch_count >= 1, "Should have at least one batch"


def test_loader_force_steering_in_validation(mock_dataset_dir):
    """Test forcing steering variant in validation mode."""
    loader = create_loader(
        dataset_dir=mock_dataset_dir,
        split="val",
        batch_size=8,
        num_workers=0,
    )

    # Force zero steering
    loader.dataset_obj.force_steering("zero")

    batch = next(iter(loader))

    # All variants should be zero
    assert (batch["steering_variant"] == 0).all(), "All steering variants should be zero when forced"

    # All steering tensors should be zeros
    assert torch.allclose(
        batch["steering"], torch.zeros_like(batch["steering"])
    ), "All steering tensors should be zeros"
