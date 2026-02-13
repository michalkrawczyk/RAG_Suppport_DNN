"""PyTorch Dataset implementations for training.

This module provides PyTorch Dataset classes for different training scenarios:
- JASPER steering dataset with curriculum learning
- General RAG datasets
- Cluster-labeled datasets
- DataLoader factories with validation

Key Features:
- Pre-loaded embeddings (zero I/O during training)
- Curriculum learning support
- Hard negative sampling
- Distributed training support
- Device placement and context managers
- Batch validation utilities

Examples
--------
>>> from RAG_supporters.pytorch_datasets import JASPERSteeringDataset, create_loader
>>>
>>> # Create dataset
>>> dataset = JASPERSteeringDataset(
...     dataset_dir="output/dataset",
...     split="train",
...     epoch=0
... )
>>>
>>> # Create DataLoader with curriculum learning
>>> loader = create_loader(
...     dataset_dir="output/dataset",
...     split="train",
...     batch_size=32,
...     num_workers=4
... )
>>>
>>> # Training loop
>>> for epoch in range(100):
...     loader.dataset_obj.set_epoch(epoch)
...     for batch in loader:
...         # Train model
...         pass
"""

from .jasper_steering_dataset import JASPERSteeringDataset
from .rag_dataset import (
    BaseRAGDatasetGenerator,
    SamplePairingType,
    SampleTripletRAGChroma,
)
from .cluster_labeled_dataset import ClusterLabeledDataset
from .loader import create_loader, set_epoch, validate_first_batch

__all__ = [
    # JASPER dataset
    "JASPERSteeringDataset",
    # RAG datasets
    "BaseRAGDatasetGenerator",
    "SamplePairingType",
    "SampleTripletRAGChroma",
    # Cluster-labeled dataset
    "ClusterLabeledDataset",
    # DataLoader utilities
    "create_loader",
    "set_epoch",
    "validate_first_batch",
]
