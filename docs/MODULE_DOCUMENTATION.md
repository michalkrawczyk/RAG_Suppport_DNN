# Module Documentation Index

**Note**: This documentation has been split into separate files by module category for better organization.

## Module Categories

Documentation is now organized by module category:

- **[Contrastive Learning](CONTRASTIVE_LEARNING.md)** 
  - NegativeMiner - 4-tier hard negative sampling
  - SteeringBuilder - Steering signal generation

- **[Data Preparation](DATA_PREPARATION.md)**
  - CSVMerger - CSV merging with deduplication
  - DatasetSplitter (Simple) - Basic train/val/test splitting
  - DatasetSplitter (Stratified) - Question-level stratified splitting

- **[Data Validation](DATA_VALIDATION.md)**
  - validation_utils - Tensor validation, bounds checking, NaN detection
  - tensor_utils - Tensor I/O with validation
  - label_calculator - Label normalization (softmax, L1, minmax)

- **[JASPER Builder](JASPER_BUILDER.md)**
  - BuildConfig - Configuration dataclass
  - DatasetFinalizer - Cross-validation and integrity checks
  - build_dataset - Task orchestrator

- **[Embeddings Operations](EMBEDDINGS_OPERATIONS.md)**
  - EmbeddingGenerator - Batched embedding generation
  - SteeringEmbeddingGenerator - Steering embedding generation
  - SteeringConfig/SteeringMode - Configuration and modes

- **[Clustering Operations](CLUSTERING_OPERATIONS.md)**
  - ClusterParser - Parses cluster JSON with keyword matching
  - SourceClusterLinker - Links pairs to clusters

- **[General Utilities](GENERAL_UTILITIES.md)**
  - text_utils - Text validation and processing
  - suggestion_processing - LLM suggestion processing
  - text_splitters - Text chunking

## Quick Navigation

For the complete user guide to building JASPER datasets, see:
- **[JASPER Builder Guide](dataset/JASPER_BUILDER_GUIDE.md)**

For the dataset runtime documentation, see:
- **[JASPER Steering Dataset](pytorch_datasets/JASPER_STEERING_DATASET.md)**

## Documentation Structure

Each module category documentation file includes:
- Purpose and overview
- Usage examples
- Parameter descriptions
- Method documentation
- Related links

---

*Documentation last updated: Split from monolithic MODULE_DOCUMENTATION.md into separate files by module category*
