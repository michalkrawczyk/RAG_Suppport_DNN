"""
Dataset module for RAG Support DNN.

Legacy module that now only contains domain assessment dataset builders
and steering configuration. Most functionality has been moved to:

- PyTorch datasets: RAG_supporters.pytorch_datasets
- Dataset building: RAG_supporters.jasper
- Data preprocessing: RAG_supporters.data_prep
- Contrastive learning: RAG_supporters.contrastive
- Data validation: RAG_supporters.data_validation
- Embedding operations: RAG_supporters.embeddings_ops
- Clustering operations: RAG_supporters.clustering_ops

This module provides:
- DomainAssessmentDatasetBuilder: Build domain assessment datasets
- DomainAssessmentParser: Parse domain assessment CSVs
- SteeringConfig: Configuration for steering generation
- SteeringMode: Enum for steering modes
"""

# Domain assessment components (only things still in this directory)
from .domain_assessment_dataset_builder import DomainAssessmentDatasetBuilder
from .domain_assessment_parser import DomainAssessmentParser

# Steering configuration components
from .steering import SteeringConfig, SteeringMode

__all__ = [
    # Domain assessment approach
    "DomainAssessmentDatasetBuilder",
    "DomainAssessmentParser",
    # Steering configuration
    "SteeringMode",
    "SteeringConfig",
]
