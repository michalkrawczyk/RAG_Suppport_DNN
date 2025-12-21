"""
Dataset module for RAG Support DNN.

This module provides PyTorch datasets for RAG-based question answering
and cluster steering.

Domain Assessment Dataset (Recommended):
- ClusterLabeledDataset: PyTorch dataset with CSV domain assessment + clustering JSON
- DomainAssessmentDatasetBuilder: Build dataset from CSV files
- DomainAssessmentParser: Parse domain assessment CSVs
- LabelCalculator: Calculate 3-type labels (source, steering, combined)
- SteeringEmbeddingGenerator: Generate steering embeddings with augmentations
- SQLiteStorageManager: SQLite + numpy memmap storage
- SteeringMode: Enum for steering embedding modes
- SteeringConfig: Configuration for steering generation
"""

# ClusteringData from clustering module
from ..clustering import ClusteringData

# (Removed legacy backward-compatibility imports)

# Steering configuration components
from .steering import (
    SteeringConfig,
    SteeringMode,
)

# Domain assessment components
from .cluster_labeled_dataset import ClusterLabeledDataset
from .domain_assessment_dataset_builder import DomainAssessmentDatasetBuilder
from .domain_assessment_parser import DomainAssessmentParser
from .label_calculator import LabelCalculator
from .steering_embedding_generator import SteeringEmbeddingGenerator
from .sqlite_storage import SQLiteStorageManager

__all__ = [
    # Steering configuration
    "SteeringMode",
    "SteeringConfig",
    "ClusteringData",
    # Domain assessment approach (RECOMMENDED)
    "ClusterLabeledDataset",
    "DomainAssessmentDatasetBuilder",
    "DomainAssessmentParser",
    "LabelCalculator",
    "SteeringEmbeddingGenerator",
    "SQLiteStorageManager",
]
