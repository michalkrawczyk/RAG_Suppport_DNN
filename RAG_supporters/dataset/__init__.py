"""
Dataset module for RAG Support DNN.

This module provides PyTorch datasets for RAG-based question answering
and cluster steering.
"""

# ClusteringData from clustering module
from ..clustering import ClusteringData

# Legacy imports for backward compatibility
from .torch_dataset import (
    BaseDomainAssignDataset,
    CachedDomainAssignDataset,
    build_and_load_dataset,
)

# New modular components
from .steering import (
    CacheManager,
    DatasetBuilder,
    SteeringConfig,
    SteeringDataset,
    SteeringGenerator,
    SteeringMode,
)

# Export both old and new interfaces
__all__ = [
    # Legacy interface (backward compatibility)
    "BaseDomainAssignDataset",
    "CachedDomainAssignDataset",
    "build_and_load_dataset",
    # New modular components
    "SteeringMode",
    "ClusteringData",
    "SteeringConfig",
    "CacheManager",
    "SteeringGenerator",
    "DatasetBuilder",
    "SteeringDataset",
]
