"""
Dataset module for RAG Support DNN.

This module provides PyTorch datasets for RAG-based question answering
and cluster steering.

New Domain Assessment Approach (Recommended):
- ClusterLabeledDataset: PyTorch dataset with CSV domain assessment + clustering JSON
- DomainAssessmentDatasetBuilder: Build dataset from CSV files
- DomainAssessmentParser: Parse domain assessment CSVs
- LabelCalculator: Calculate 3-type labels (source, steering, combined)
- SteeringEmbeddingGenerator: Generate steering embeddings with augmentations
- SQLiteStorageManager: SQLite + numpy memmap storage

Legacy Steering Approach (Backward Compatible):
- SteeringDataset: Original steering dataset
- SteeringDatasetBuilder: Original builder
- BaseDomainAssignDataset: Deprecated, use ClusterLabeledDataset
"""

# ClusteringData from clustering module
from ..clustering import ClusteringData

# Legacy imports for backward compatibility
from .torch_dataset import (
    BaseDomainAssignDataset,
    CachedDomainAssignDataset,
    build_and_load_dataset,
)

# Original modular components (still available)
from .steering import (
    CacheManager,
    SteeringDatasetBuilder,
    SteeringConfig,
    SteeringDataset,
    SteeringGenerator,
    SteeringMode,
)

# New domain assessment components (RECOMMENDED)
from .cluster_labeled_dataset import ClusterLabeledDataset
from .domain_assessment_dataset_builder import DomainAssessmentDatasetBuilder
from .domain_assessment_parser import DomainAssessmentParser
from .label_calculator import LabelCalculator
from .steering_embedding_generator import SteeringEmbeddingGenerator
from .sqlite_storage import SQLiteStorageManager

# Export both old and new interfaces
__all__ = [
    # Legacy interface (backward compatibility)
    "BaseDomainAssignDataset",
    "CachedDomainAssignDataset",
    "build_and_load_dataset",
    # Original modular components
    "SteeringMode",
    "ClusteringData",
    "SteeringConfig",
    "CacheManager",
    "SteeringGenerator",
    "SteeringDatasetBuilder",
    "SteeringDataset",
    # New domain assessment approach (RECOMMENDED)
    "ClusterLabeledDataset",
    "DomainAssessmentDatasetBuilder",
    "DomainAssessmentParser",
    "LabelCalculator",
    "SteeringEmbeddingGenerator",
    "SQLiteStorageManager",
]
