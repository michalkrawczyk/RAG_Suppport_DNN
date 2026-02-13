# Dataset Documentation

This directory contains documentation related to dataset functionality in the RAG Supporters project.

## Contents

### PyTorch Datasets
- [JASPER_STEERING_DATASET.md](JASPER_STEERING_DATASET.md) - **JASPER Steering Dataset** - PyTorch dataset for pre-computed embedding triplets with curriculum learning, hard negatives, GPU preloading, and context manager support
- [JASPER_TRAINING_EXAMPLE.md](JASPER_TRAINING_EXAMPLE.md) - Complete training examples for JASPER including GPU preloading and distributed training

### Dataset Utilities
- [DATASET_SPLITTING.md](DATASET_SPLITTING.md) - Dataset splitting with persistent sample tracking
- [DOMAIN_ASSESSMENT_EXAMPLES.md](DOMAIN_ASSESSMENT_EXAMPLES.md) - Domain assessment dataset usage examples
- [DOMAIN_ASSESSMENT_CLUSTERING_DATASET_EXAMPLE.md](DOMAIN_ASSESSMENT_CLUSTERING_DATASET_EXAMPLE.md) - Complete workflow example from domain assessment to PyTorch dataset

## Overview

These documents cover:
- **JASPER dataset**: Zero-I/O training, curriculum learning, device placement, referential integrity
- Train/Val split generation and restoration
- Domain assessment dataset building
- PyTorch dataset integration
- Label types and steering modes
- Augmentation strategies

## Related Code

- `RAG_supporters/dataset/` - Dataset implementation
- `RAG_supporters/agents/domain_assesment.py` - Domain assessment agent

## See Also

- [Main Documentation](../README.md)
- [Clustering Documentation](../clustering/README.md)
- [Agents Documentation](../agents/README.md)
