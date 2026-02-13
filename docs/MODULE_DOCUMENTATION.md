# Module Documentation Index

**Note**: Module documentation has been reorganized into subdirectories by module category for improved structure and maintainability.

## Module Categories

Documentation is now organized in module-specific subdirectories:

- **[Contrastive Learning](contrastive_learning/CONTRASTIVE_LEARNING.md)** 
  - NegativeMiner - 4-tier hard negative sampling
  - SteeringBuilder - Steering signal generation

- **[Data Preparation](data_preparation/DATA_PREPARATION.md)**
  - CSVMerger - CSV merging with deduplication
  - DatasetSplitter (Simple) - Basic train/val/test splitting
  - DatasetSplitter (Stratified) - Question-level stratified splitting

- **[Data Validation](data_validation/DATA_VALIDATION.md)**
  - validation_utils - Tensor validation, bounds checking, NaN detection
  - tensor_utils - Tensor I/O with validation
  - label_calculator - Label normalization (softmax, L1, minmax)

- **[JASPER Builder](jasper_builder/JASPER_BUILDER.md)**
  - BuildConfig - Configuration dataclass
  - DatasetFinalizer - Cross-validation and integrity checks
  - build_dataset - Task orchestrator

- **[Embeddings Operations](embeddings/EMBEDDINGS_OPERATIONS.md)**
  - EmbeddingGenerator - Batched embedding generation
  - SteeringEmbeddingGenerator - Steering embedding generation
  - SteeringConfig/SteeringMode - Configuration and modes

- **[Clustering Operations](clustering/CLUSTERING_OPERATIONS.md)**
  - ClusterParser - Parses cluster JSON with keyword matching
  - SourceClusterLinker - Links pairs to clusters

- **[General Utilities](utilities/GENERAL_UTILITIES.md)**
  - text_utils - Text validation and processing
  - suggestion_processing - LLM suggestion processing
  - text_splitters - Text chunking

## Quick Navigation

For the complete user guide to building JASPER datasets, see:
- **[JASPER Builder Guide](dataset/JASPER_BUILDER_GUIDE.md)**

For the dataset runtime documentation, see:
- **[JASPER Steering Dataset](dataset/JASPER_STEERING_DATASET.md)**

## Documentation Structure

Each module category documentation file includes:
- Purpose and overview
- Usage examples with code
- Parameter tables
- Method documentation
- Related documentation links
- Cross-references between related modules

## Related Documentation

- **[Documentation Index](README.md)** - Main documentation index
- **[Agents Overview](agents/AGENTS_OVERVIEW.md)** - LLM-powered agents
- **[Developer Guidelines](../AGENTS.md)** - Technical guidelines and architecture

---

*Documentation last updated: Reorganized into module-specific subdirectories*
