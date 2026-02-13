"""
Dataset module for RAG Support DNN.

Legacy module that now only contains domain assessment dataset builders.
Most functionality has been moved to specialized modules:

- PyTorch datasets: RAG_supporters.pytorch_datasets
- Dataset building: RAG_supporters.jasper
- Data preprocessing: RAG_supporters.data_prep
- Contrastive learning: RAG_supporters.contrastive
- Data validation: RAG_supporters.data_validation
- Embedding operations: RAG_supporters.embeddings_ops (includes SteeringConfig, SteeringMode)
- Clustering operations: RAG_supporters.clustering_ops

This module provides:
- DomainAssessmentDatasetBuilder: Build domain assessment datasets
- DomainAssessmentParser: Parse domain assessment CSVs

Note: SteeringConfig and SteeringMode have been moved to RAG_supporters.embeddings_ops.
Import them directly from there instead of from this module.
"""

# Domain assessment components
from .domain_assessment_dataset_builder import DomainAssessmentDatasetBuilder
from .domain_assessment_parser import DomainAssessmentParser

__all__ = [
    "DomainAssessmentDatasetBuilder",
    "DomainAssessmentParser",
]
