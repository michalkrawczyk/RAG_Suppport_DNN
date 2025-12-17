"""
Dataset module for RAG Support DNN.

This module provides PyTorch datasets for RAG-based question answering
and cluster steering.
"""

# Legacy imports for backward compatibility
from .torch_dataset import (
    BaseDomainAssignDataset,
    CachedDomainAssignDataset,
    SteeringMode as LegacySteeringMode,
    build_and_load_dataset,
)

# New modular components
from .steering import (
    CacheManager,
    ClusteringData,
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
