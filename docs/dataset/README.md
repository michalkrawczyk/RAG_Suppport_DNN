# Dataset Documentation

This directory contains documentation related to dataset functionality in the RAG Supporters project.

## Contents

- [DATASET_SPLITTING.md](DATASET_SPLITTING.md) - Dataset splitting with persistent sample tracking
- [DOMAIN_ASSESSMENT_EXAMPLES.md](DOMAIN_ASSESSMENT_EXAMPLES.md) - Domain assessment dataset usage examples
- [DOMAIN_ASSESSMENT_CLUSTERING_DATASET_EXAMPLE.md](DOMAIN_ASSESSMENT_CLUSTERING_DATASET_EXAMPLE.md) - Complete workflow example from domain assessment to PyTorch dataset

## Overview

These documents cover:
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
