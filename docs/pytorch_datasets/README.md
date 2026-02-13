# PyTorch Datasets Documentation

This directory documents PyTorch Dataset implementations for training neural networks in the RAG Support DNN project.

## Available Datasets

### Production Datasets

- **[JASPERSteeringDataset](JASPER_STEERING_DATASET.md)** - JASPER steering with curriculum learning and hard negatives
- **[ClusterLabeledDataset](CLUSTER_LABELED_DATASET.md)** - Domain assessment with cluster-based labels

### Base Classes

- **[RAG Dataset](RAG_DATASET.md)** - Abstract base for RAG dataset generation and triplet sampling

### Utilities

- **[Loader Utilities](LOADER_UTILITIES.md)** - DataLoader factory and batch validation
- **[Storage Formats](STORAGE_FORMATS.md)** - PT, HDF5, and memory-mapped loading for large datasets

## Quick Start

### JASPER Steering Dataset

```python
from RAG_supporters.pytorch_datasets import JASPERSteeringDataset, create_loader

# Create DataLoader with curriculum learning
loader = create_loader(
    dataset_dir="output/dataset",
    split="train",
    batch_size=32,
    num_workers=4
)

# Training loop with curriculum learning
for epoch in range(100):
    loader.dataset_obj.set_epoch(epoch)
    for batch in loader:
        # Train JASPER model
        question_emb = batch["question_emb"]
        target_source_emb = batch["target_source_emb"]
        steering = batch["steering"]
        negative_embs = batch["negative_embs"]
```

### Cluster Labeled Dataset

```python
from RAG_supporters.pytorch_datasets import ClusterLabeledDataset
from torch.utils.data import DataLoader

# Load dataset with combined labels
dataset = ClusterLabeledDataset(
    dataset_dir="output/clusters",
    label_type="combined",
    cache_size=1000
)

# Create DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for base_emb, steering_emb, label in loader:
    # Train domain classifier
    pass
```

## Key Features

### Zero I/O During Training

All datasets preload embeddings into memory during initialization:
- No disk access in `__getitem__`
- Predictable batch timing
- Optimal GPU utilization

### Curriculum Learning

JASPERSteeringDataset supports curriculum learning through `set_epoch()`:
- Progressive steering difficulty
- Deterministic per-epoch sampling
- Coordinated with distributed training

### Distributed Training

All datasets support PyTorch DistributedDataParallel:
- `DistributedSampler` integration
- Epoch-synchronized shuffling
- Consistent splits across GPUs

### Memory Management

- **Memmap**: ClusterLabeledDataset uses numpy memmap for large embeddings
- **LRU Cache**: Metadata cached with configurable size
- **Device Placement**: JASPERSteeringDataset supports GPU preloading

## Dataset Comparison

| Feature | JASPER Steering | Cluster Labeled | RAG Base |
|---------|----------------|-----------------|----------|
| **Use Case** | JASPER training | Domain classification | Dataset generation |
| **Output** | (question, steering, target, negatives) | (base_emb, steering_emb, label) | Triplets for labeling |
| **Preloading** | Full (CPU/GPU) | Memmap + cache | N/A |
| **Curriculum** | Yes | No | N/A |
| **Hard Negatives** | Yes (tiered) | No | Configurable |
| **Storage** | PT tensors | SQLite + memmap | ChromaDB |

## File Formats

### JASPER Steering Dataset

```
dataset_dir/
├── config.json              # Dataset configuration
├── train_idx.pt             # Train split indices
├── val_idx.pt               # Validation split indices
├── test_idx.pt              # Test split indices
├── question_embs.pt         # Question embeddings [N_q, D]
├── source_embs.pt           # Source embeddings [N_s, D]
├── keyword_embs.pt          # Keyword embeddings [N_k, D]
├── centroid_embs.pt         # Cluster centroids [N_c, D]
├── pair_index.pt            # (question_idx, source_idx) pairs [N_p, 2]
├── pair_cluster_id.pt       # Cluster assignments [N_p]
├── pair_relevance.pt        # Relevance scores [N_p]
├── pair_keyword_ids.pt      # Keyword IDs per pair (list-of-lists)
├── hard_negatives.pt        # Negative source indices [N_p, n_neg]
├── negative_tiers.pt        # Negative difficulty tiers [N_p, n_neg]
├── steering_centroid.pt     # Centroid steering [N_p, D]
├── steering_keyword_weighted.pt  # Keyword-weighted steering [N_p, D]
├── steering_residual.pt     # Residual steering [N_p, D]
└── centroid_distances.pt    # Distances to centroids [N_p]
```

### Cluster Labeled Dataset

```
dataset_dir/
├── dataset.db              # SQLite database (metadata)
├── base_embeddings.npy     # Base embeddings (memmap) [N, D]
└── steering_embeddings.npy # Steering embeddings (memmap) [N, D]
```

## Performance Tips

### JASPER Steering

- **GPU Preloading**: Pass `device=torch.device("cuda")` to constructor
- **Workers**: Use `num_workers=4` for CPU preprocessing
- **Batch Size**: Larger batches benefit from preloaded negatives

### Cluster Labeled

- **Cache Size**: Set `cache_size` to 10-20% of dataset size
- **Memmap Mode**: Use `mmap_mode="r"` (read-only) for shared memory
- **Label Type**: Choose based on training objective (source/steering/combined)

## Validation

Use `validate_first_batch()` to check DataLoader configuration:

```python
from RAG_supporters.pytorch_datasets.loader import create_loader, validate_first_batch

loader = create_loader("output/dataset", "train", batch_size=32)
validate_first_batch(loader)  # Raises AssertionError if issues found
```

Validates:
- Batch key presence
- Tensor shapes
- NaN/Inf detection
- Value ranges (cluster IDs, relevance, etc.)

## See Also

- [Dataset Building](../dataset/README.md) - How to create datasets
- [JASPER Training](../dataset/JASPER_TRAINING_EXAMPLE.md) - Training workflow
- [Dataset Splitting](../dataset/DATASET_SPLITTING.md) - Train/val/test splits
- [Clustering](../clustering/README.md) - Cluster creation
