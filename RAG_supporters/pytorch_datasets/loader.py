"""
DataLoader factory and utilities for JASPER Steering Dataset.

Provides functions to create DataLoaders with proper configuration for
distributed training, validation, and testing.
"""

import logging
from pathlib import Path
from typing import Literal, Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from RAG_supporters.pytorch_datasets import JASPERSteeringDataset

LOGGER = logging.getLogger(__name__)


def create_loader(
    dataset_dir: str | Path,
    split: Literal["train", "val", "test"],
    batch_size: int,
    num_workers: int = 0,
    distributed: bool = False,
    epoch: int = 0,
    drop_last: Optional[bool] = None,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for JASPER Steering Dataset.

    Parameters
    ----------
    dataset_dir : str or Path
        Directory containing dataset files
    split : Literal["train", "val", "test"]
        Which split to load
    batch_size : int
        Batch size for the DataLoader
    num_workers : int, optional
        Number of worker processes for data loading, by default 0
    distributed : bool, optional
        Whether to use DistributedSampler for multi-GPU training, by default False
    epoch : int, optional
        Initial epoch for curriculum learning, by default 0
    drop_last : Optional[bool], optional
        Whether to drop last incomplete batch.
        If None, defaults to True for train, False for val/test
    pin_memory : bool, optional
        Whether to pin memory for faster GPU transfer, by default True

    Returns
    -------
    DataLoader
        Configured PyTorch DataLoader

    Examples
    --------
    >>> loader = create_loader(
    ...     dataset_dir="/path/to/dataset",
    ...     split="train",
    ...     batch_size=32,
    ...     num_workers=4,
    ... )
    >>> for batch in loader:
    ...     # Train model
    ...     pass
    """
    # Create dataset
    dataset = JASPERSteeringDataset(dataset_dir=dataset_dir, split=split, epoch=epoch)

    # Default drop_last based on split
    if drop_last is None:
        drop_last = split == "train"

    # Create sampler
    if distributed:
        sampler = DistributedSampler(
            dataset,
            shuffle=(split == "train"),
            drop_last=drop_last,
        )
        shuffle = None  # Shuffle handled by sampler
    else:
        sampler = None
        shuffle = split == "train"

    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    # Attach dataset for easy access to set_epoch
    loader.dataset_obj = dataset
    loader.sampler_obj = sampler

    LOGGER.info(
        f"Created {split} DataLoader: {len(dataset)} samples, "
        f"batch_size={batch_size}, num_workers={num_workers}, "
        f"distributed={distributed}"
    )

    return loader


def set_epoch(loader: DataLoader, epoch: int):
    """
    Set epoch for both dataset and sampler (for curriculum and distributed training).

    Parameters
    ----------
    loader : DataLoader
        DataLoader created by create_loader
    epoch : int
        Epoch number to set

    Examples
    --------
    >>> loader = create_loader(...)
    >>> for epoch in range(100):
    ...     set_epoch(loader, epoch)
    ...     for batch in loader:
    ...         # Train
    ...         pass
    """
    # Set epoch on dataset for curriculum learning
    if hasattr(loader, "dataset_obj"):
        loader.dataset_obj.set_epoch(epoch)
    elif isinstance(loader.dataset, JASPERSteeringDataset):
        loader.dataset.set_epoch(epoch)

    # Set epoch on sampler for distributed training
    if hasattr(loader, "sampler_obj") and loader.sampler_obj is not None:
        loader.sampler_obj.set_epoch(epoch)
    elif isinstance(loader.sampler, DistributedSampler):
        loader.sampler.set_epoch(epoch)


def validate_first_batch(loader: DataLoader) -> bool:
    """
    Validate that the first batch has correct shapes and no NaN/Inf values.

    Parameters
    ----------
    loader : DataLoader
        DataLoader to validate

    Returns
    -------
    bool
        True if validation passes

    Raises
    ------
    AssertionError
        If any validation check fails

    Examples
    --------
    >>> loader = create_loader(...)
    >>> validate_first_batch(loader)
    True
    """
    LOGGER.info("Validating first batch from DataLoader...")

    # Get first batch
    batch = next(iter(loader))

    # Expected keys
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

    # Check all keys present
    actual_keys = set(batch.keys())
    assert actual_keys == expected_keys, (
        f"Batch keys mismatch. Expected: {expected_keys}, Got: {actual_keys}"
    )

    B = batch["question_emb"].size(0)
    D = batch["question_emb"].size(1)
    N_neg = batch["negative_embs"].size(1)

    LOGGER.info(f"Batch size: {B}, Embedding dim: {D}, Num negatives: {N_neg}")

    # Validate shapes
    assert batch["question_emb"].shape == (B, D), (
        f"question_emb shape mismatch: {batch['question_emb'].shape} != ({B}, {D})"
    )
    assert batch["target_source_emb"].shape == (B, D), (
        f"target_source_emb shape mismatch: {batch['target_source_emb'].shape} != ({B}, {D})"
    )
    assert batch["steering"].shape == (B, D), (
        f"steering shape mismatch: {batch['steering'].shape} != ({B}, {D})"
    )
    assert batch["negative_embs"].shape == (B, N_neg, D), (
        f"negative_embs shape mismatch: {batch['negative_embs'].shape} != ({B}, {N_neg}, {D})"
    )
    assert batch["cluster_id"].shape == (B,), (
        f"cluster_id shape mismatch: {batch['cluster_id'].shape} != ({B},)"
    )
    assert batch["relevance"].shape == (B,), (
        f"relevance shape mismatch: {batch['relevance'].shape} != ({B},)"
    )
    assert batch["centroid_distance"].shape == (B,), (
        f"centroid_distance shape mismatch: {batch['centroid_distance'].shape} != ({B},)"
    )
    assert batch["steering_variant"].shape == (B,), (
        f"steering_variant shape mismatch: {batch['steering_variant'].shape} != ({B},)"
    )
    assert batch["negative_tiers"].shape == (B, N_neg), (
        f"negative_tiers shape mismatch: {batch['negative_tiers'].shape} != ({B}, {N_neg})"
    )

    # Check for NaN/Inf in embeddings
    for key in ["question_emb", "target_source_emb", "steering", "negative_embs"]:
        tensor = batch[key]
        assert not torch.isnan(tensor).any(), f"NaN detected in {key}"
        assert not torch.isinf(tensor).any(), f"Inf detected in {key}"

    # Check cluster_id range (must be valid indices)
    dataset = loader.dataset
    if isinstance(dataset, JASPERSteeringDataset):
        n_clusters = len(dataset.centroid_embs)
        assert (batch["cluster_id"] >= 0).all(), "cluster_id must be non-negative"
        assert (batch["cluster_id"] < n_clusters).all(), (
            f"cluster_id must be < {n_clusters}, got max {batch['cluster_id'].max()}"
        )

    # Check relevance range [0, 1]
    assert (batch["relevance"] >= 0).all(), "relevance must be >= 0"
    assert (batch["relevance"] <= 1).all(), "relevance must be <= 1"

    # Check centroid_distance range [0, 2]
    assert (batch["centroid_distance"] >= 0).all(), "centroid_distance must be >= 0"
    assert (batch["centroid_distance"] <= 2).all(), "centroid_distance must be <= 2"

    # Check steering_variant range [0, 3]
    assert (batch["steering_variant"] >= 0).all(), "steering_variant must be >= 0"
    assert (batch["steering_variant"] <= 3).all(), "steering_variant must be <= 3"

    # Print shape summary
    LOGGER.info("Batch validation passed!")
    LOGGER.info("Shape summary:")
    LOGGER.info(f"  question_emb:        {batch['question_emb'].shape}")
    LOGGER.info(f"  target_source_emb:   {batch['target_source_emb'].shape}")
    LOGGER.info(f"  steering:            {batch['steering'].shape}")
    LOGGER.info(f"  negative_embs:       {batch['negative_embs'].shape}")
    LOGGER.info(f"  cluster_id:          {batch['cluster_id'].shape}")
    LOGGER.info(f"  relevance:           {batch['relevance'].shape}")
    LOGGER.info(f"  centroid_distance:   {batch['centroid_distance'].shape}")
    LOGGER.info(f"  steering_variant:    {batch['steering_variant'].shape}")
    LOGGER.info(f"  negative_tiers:      {batch['negative_tiers'].shape}")

    return True
