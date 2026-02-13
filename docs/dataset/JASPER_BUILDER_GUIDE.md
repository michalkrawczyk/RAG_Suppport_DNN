# JASPER Builder Guide

Complete user guide for building JASPER Steering Datasets from CSV files and cluster JSON.

## Overview

The JASPER Builder Pipeline transforms CSV files with question-source pairs and cluster JSON into a ready-to-train dataset with pre-computed embeddings, hard negatives, and steering signals.

**Input**:
- Multiple CSV files with question-source pairs
- Cluster JSON with keyword embeddings and centroids
- Embedding model for text encoding

**Output**:
- Self-contained dataset directory with all `.pt` tensors and metadata
- Zero text processing or model inference at training time

---

## Quick Start

### Basic Usage

```python
from RAG_supporters.jasper import build_dataset, BuildConfig
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Build dataset
build_dataset(
    csv_paths=["data/pairs1.csv", "data/pairs2.csv"],
    cluster_json_path="data/clusters.json",
    embedding_model=model,
    output_dir="./my_dataset",
    config=BuildConfig(
        embedding_dim=384,
        n_neg=12,
        tier_proportions=[3, 4, 3, 2],
        split_ratios=[0.8, 0.1, 0.1]
    )
)
```

### Using the Dataset

```python
from RAG_supporters.pytorch_datasets import create_loader

loader = create_loader(
    dataset_dir="./my_dataset",
    split="train",
    batch_size=128,
    num_workers=4
)

for batch in loader:
    # Train your model
    question_emb = batch["question_emb"]
    target_emb = batch["target_source_emb"]
    steering = batch["steering"]
    negatives = batch["negative_embs"]
```

---

## Configuration

### BuildConfig Parameters

```python
from RAG_supporters.jasper import BuildConfig

config = BuildConfig(
    embedding_dim=384,              # Must match model
    n_neg=12,                       # Negatives per pair
    tier_proportions=[3, 4, 3, 2],  # Tier distribution
    split_ratios=[0.8, 0.1, 0.1],   # Train/Val/Test
    steering_probabilities={
        "zero": 0.25, "centroid": 0.25,
        "keyword": 0.25, "residual": 0.25
    },
    curriculum={
        "zero_prob_start": 0.5,
        "zero_prob_end": 0.1,
        "epochs_total": 100
    }
)
```

---

## Pipeline Steps

1. **CSV Merger** - Normalizes, merges, assigns IDs
2. **Cluster Parser** - Matches keywords, extracts centroids
3. **Source-Cluster Linker** - Links sources to clusters
4. **Embedding Generator** - Embeds questions, sources, keywords
5. **Steering Builder** - Creates steering signals
6. **Negative Miner** - Samples 4-tier hard negatives
7. **Dataset Splitter** - Stratified train/val/test splits
8. **Config Writer** - Validates and finalizes
9. **Assembly** - Logs summary

---

## Output Structure

```
output_dir/
├── config.json                   # Configuration
├── *_embs.pt                     # Embeddings
├── steering_*.pt                 # Steering signals
├── hard_negatives.pt             # Negative indices
├── train_idx.pt, val_idx.pt, test_idx.pt  # Splits
└── *.json                        # Metadata mappings
```

---

## Input Requirements

**CSV Format** - Must have:
- `question` / `question_text` (required)
- `source` / `source_text` (required)
- `keywords`, `score` (optional)

**Cluster JSON** - Must have:
- `metadata.n_clusters`, `metadata.embedding_dim`
- `cluster_assignments` - keyword → cluster mapping
- `clusters.{id}.centroid` - centroid vectors
- `embeddings` - keyword embeddings

---

## Related Documentation

- [Module Documentation](MODULE_DOCUMENTATION.md) - Detailed component docs
- [JASPER Dataset](../pytorch_datasets/JASPER_STEERING_DATASET.md) - Runtime usage
- [Training Examples](JASPER_TRAINING_EXAMPLE.md) - Training loops

For full pipeline details, see [Module Documentation](MODULE_DOCUMENTATION.md#jasper-builder).
