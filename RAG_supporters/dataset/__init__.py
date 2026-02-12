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

JASPER Steering Dataset (Pre-computed embeddings):
- JASPERSteeringDataset: PyTorch dataset for pre-computed embedding triplets
  (JASPER: Joint Architecture for Subspace Prediction with Explainable Routing)
- create_loader: DataLoader factory with distributed training support
- set_epoch: Set epoch for curriculum and distributed training
- validate_first_batch: Validate DataLoader output shapes

JASPER Dataset Builder (Dataset construction pipeline):
- BuildConfig: Configuration dataclass for dataset building
- CSVMerger: Merge and normalize multiple CSV files
- merge_csv_files: Convenience function for CSV merging
- ClusterParser: Parse KeywordClusterer JSON and match keywords to clusters
- parse_clusters: Convenience function for cluster parsing
- SourceClusterLinker: Link pairs to clusters via keyword intersection
- link_sources: Convenience function for source-cluster linking
- EmbeddingGenerator: Batch embedding generation with validation
- generate_embeddings: Convenience function for embedding generation
- SteeringBuilder: Generate steering signals for curriculum learning
- build_steering: Convenience function for steering generation
- NegativeMiner: Mine hard negatives for contrastive learning
- mine_negatives: Convenience function for negative mining
- DatasetSplitter: Question-level stratified splitting
- split_dataset: Convenience function for dataset splitting

Note: JEPA has been renamed to JASPER. Legacy code using JEPASteeringDataset
will receive an ImportError with migration instructions.
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

# JASPER Steering Dataset components
from .jasper_steering_dataset import JASPERSteeringDataset
from .loader import create_loader, set_epoch, validate_first_batch

# JASPER Dataset Builder components
from .builder_config import BuildConfig
from .merge_csv import CSVMerger, merge_csv_files
from .parse_clusters import ClusterParser, parse_clusters
from .link_sources import SourceClusterLinker, link_sources
from .embed import EmbeddingGenerator, generate_embeddings
from .build_steering import SteeringBuilder, build_steering
from .mine_negatives import NegativeMiner, mine_negatives
from .split import DatasetSplitter as JASPERDatasetSplitter, split_dataset

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
    # JASPER Steering Dataset
    "JASPERSteeringDataset",
    "create_loader",
    "set_epoch",
    "validate_first_batch",
    # JASPER Dataset Builder
    "BuildConfig",
    "CSVMerger",
    "merge_csv_files",
    "ClusterParser",
    "parse_clusters",
    "SourceClusterLinker",
    "link_sources",
    "EmbeddingGenerator",
    "generate_embeddings",
    "SteeringBuilder",
    "build_steering",
    "NegativeMiner",
    "mine_negatives",
    "JASPERDatasetSplitter",
    "split_dataset",
]
