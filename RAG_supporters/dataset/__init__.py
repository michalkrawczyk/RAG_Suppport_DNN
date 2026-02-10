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
- DatasetSplitter: Split datasets with persistent sample tracking
- create_train_val_split: Convenience function for creating splits

JEPA Steering Dataset (Pre-computed embeddings):
- JEPASteeringDataset: PyTorch dataset for pre-computed embedding triplets
- create_loader: DataLoader factory with distributed training support
- set_epoch: Set epoch for curriculum and distributed training
- validate_first_batch: Validate DataLoader output shapes
"""

# ClusteringData from clustering module
from RAG_supporters.clustering import ClusteringData

# Domain assessment components
from .cluster_labeled_dataset import ClusterLabeledDataset

# Dataset splitting utilities
from .dataset_splitter import DatasetSplitter, create_train_val_split
from .domain_assessment_dataset_builder import DomainAssessmentDatasetBuilder
from .domain_assessment_parser import DomainAssessmentParser
from .label_calculator import LabelCalculator
from .sqlite_storage import SQLiteStorageManager

# Steering configuration components
from .steering import SteeringConfig, SteeringMode
from .steering_embedding_generator import SteeringEmbeddingGenerator

# JEPA Steering Dataset components
from .jepa_steering_dataset import JEPASteeringDataset
from .loader import create_loader, set_epoch, validate_first_batch

# (Removed legacy backward-compatibility imports)


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
    # Dataset splitting
    "DatasetSplitter",
    "create_train_val_split",
    # JEPA Steering Dataset
    "JEPASteeringDataset",
    "create_loader",
    "set_epoch",
    "validate_first_batch",
]
