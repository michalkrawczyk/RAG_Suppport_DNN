# JASPER Steering Dataset

PyTorch Dataset for JASPER (Joint Architecture for Subspace Prediction with Explainable Routing) training with pre-computed embeddings, hard negatives, and curriculum learning.

## Overview

`JASPERSteeringDataset` serves pre-computed embedding triplets with hard negatives for training JASPER models. All embeddings are preloaded into memory (CPU or GPU) for zero I/O during training.

**Module**: [RAG_supporters/pytorch_datasets/jasper_steering_dataset.py](../../RAG_supporters/pytorch_datasets/jasper_steering_dataset.py)

## Architecture

### Data Flow

```
Dataset directory (disk)
  ↓ [Initialization - one-time]
Pre-loaded tensors (memory)
  ↓ [__getitem__ - zero I/O]
Batch assembly (collate)
  ↓
Training loop (GPU)
```

### Memory Layout

All tensors preloaded:
- **Embeddings**: question_embs, source_embs, keyword_embs, centroid_embs
- **Pairs**: pair_index, pair_cluster_id, pair_relevance, pair_keyword_ids
- **Negatives**: hard_negatives, negative_tiers
- **Steering**: steering_centroid, steering_keyword_weighted, steering_residual, centroid_distances

## API

### Constructor

```python
JASPERSteeringDataset(
    dataset_dir: str | Path,
    split: Literal["train", "val", "test"],
    epoch: int = 0,
    device: Optional[torch.device] = None,
    storage_format: Literal["auto", "pt", "hdf5"] = "auto",
    use_mmap: Optional[bool] = None
)
```

**Parameters**:
- `dataset_dir`: Directory containing dataset files (config.json, *.pt tensors, or dataset.h5)
- `split`: Which split to use ("train", "val", "test")
- `epoch`: Initial epoch number for curriculum learning (default: 0)
- `device`: Device to transfer tensors to during initialization (default: CPU)
- `storage_format`: Format to use - "auto" (default), "pt", or "hdf5"
- `use_mmap`: Enable memory-mapping for large datasets (default: None for auto-detect)

**Raises**:
- `ValueError`: If dataset_dir or split files not found
- `ValueError`: If referential integrity validation fails
- `ImportError`: If HDF5 requested but h5py not installed

### Batch Output

```python
batch = {
    "question_emb": Tensor[B, D],          # Question embeddings
    "target_source_emb": Tensor[B, D],     # Target source embeddings
    "steering": Tensor[B, D],              # Steering vectors (curriculum-selected)
    "negative_embs": Tensor[B, n_neg, D],  # Hard negative embeddings
    "cluster_id": Tensor[B],               # Cluster assignments (int)
    "relevance": Tensor[B],                # Relevance scores [0, 1]
    "centroid_distance": Tensor[B],        # Distance to cluster centroid [0, 2]
    "steering_variant": Tensor[B],         # Which steering was used (0-3)
    "negative_tiers": Tensor[B, n_neg]     # Negative difficulty (0-2)
}
```

### Methods

#### `set_epoch(epoch: int)`

Set epoch for curriculum learning (updates steering probabilities).

```python
for epoch in range(100):
    dataset.set_epoch(epoch)
    for batch in dataloader:
        # Train
        pass
```

#### `force_steering(variant: Optional[str])`

Override curriculum to force specific steering variant (for debugging).

```python
dataset.force_steering("centroid")  # Always use centroid steering
dataset.force_steering(None)         # Resume curriculum
```

Valid variants: `"none"`, `"centroid"`, `"keyword_weighted"`, `"residual"`

#### `get_steering_stats() -> Dict[str, Any]`

Get current curriculum statistics.

```python
stats = dataset.get_steering_stats()
# {'epoch': 5, 'probs': [0.5, 0.3, 0.15, 0.05], 'variant_names': [...]}
```

#### `subset_by_cluster(cluster_ids: List[int]) -> Subset`

Create subset containing only samples from specified clusters.

```python
# Train only on biology and chemistry clusters
subset = dataset.subset_by_cluster([0, 3, 7])
loader = DataLoader(subset, batch_size=32)
```

## Curriculum Learning

### Steering Variants

1. **None** (variant 0): No steering, question embedding only
2. **Centroid** (variant 1): Cluster centroid as steering
3. **Keyword Weighted** (variant 2): Keyword-weighted steering
4. **Residual** (variant 3): Residual steering (most complex)

### Progression Schedule

```python
def _compute_steering_probs(epoch: int) -> np.ndarray:
    """
    Epoch   | None | Centroid | Keyword | Residual
    --------|------|----------|---------|----------
    0-9     | 0.7  | 0.2      | 0.1     | 0.0
    10-19   | 0.5  | 0.3      | 0.15    | 0.05
    20-29   | 0.3  | 0.3      | 0.25    | 0.15
    30-39   | 0.1  | 0.3      | 0.35    | 0.25
    40+     | 0.0  | 0.2      | 0.4     | 0.4
    """
```

### Rationale

- **Early epochs**: Simple steering (none/centroid) to learn basic routing
- **Middle epochs**: Introduce keyword-weighted steering
- **Late epochs**: Full residual steering for complex domain relationships

## Hard Negatives

### Tier System

Each sample has `n_neg` negatives with difficulty tiers:
- **Tier 0**: Easy negatives (distant in embedding space)
- **Tier 1**: Medium negatives (moderate similarity)
- **Tier 2**: Hard negatives (very similar, challenging)

### Usage in Training

```python
for batch in loader:
    question = batch["question_emb"]
    target = batch["target_source_emb"]
    negatives = batch["negative_embs"]  # [B, n_neg, D]
    tiers = batch["negative_tiers"]     # [B, n_neg]
    
    # Weight loss by difficulty
    weights = 1.0 + 0.5 * (tiers / 2.0)  # Tier 2 gets 1.5x weight
    loss = contrastive_loss(question, target, negatives, weights)
```

## Example Usage

### Basic Training Loop

```python
from RAG_supporters.pytorch_datasets import JASPERSteeringDataset
from torch.utils.data import DataLoader

# Create dataset
dataset = JASPERSteeringDataset(
    dataset_dir="output/dataset",
    split="train",
    epoch=0
)

# Create DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Training loop with curriculum
for epoch in range(100):
    dataset.set_epoch(epoch)
    
    for batch in loader:
        question = batch["question_emb"]
        target = batch["target_source_emb"]
        steering = batch["steering"]
        negatives = batch["negative_embs"]
        
        # Forward pass
        output = model(question, steering)
        loss = contrastive_loss(output, target, negatives)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Distributed Training

```python
from RAG_supporters.pytorch_datasets.loader import create_loader, set_epoch

# Create distributed DataLoader
loader = create_loader(
    dataset_dir="output/dataset",
    split="train",
    batch_size=32,
    num_workers=4,
    distributed=True  # Uses DistributedSampler
)

# Training loop
for epoch in range(100):
    set_epoch(loader, epoch)  # Syncs both dataset and sampler
    
    for batch in loader:
        # Train
        pass
```

### GPU Preloading

```python
# Preload all tensors to GPU during initialization
device = torch.device("cuda:0")
dataset = JASPERSteeringDataset(
    dataset_dir="output/dataset",
    split="train",
    device=device  # All tensors moved to GPU
)

# Zero-copy batching (tensors already on GPU)
loader = DataLoader(dataset, batch_size=32)
```

### Validation with Frozen Steering

```python
# Validation: fix steering to "centroid" for consistency
val_dataset = JASPERSteeringDataset("output/dataset", split="val")
val_dataset.force_steering("centroid")

val_loader = DataLoader(val_dataset, batch_size=64)
```

## Storage Formats

The dataset supports three storage formats for flexibility and performance optimization:

### PT Format (Default)

Standard PyTorch tensor format (`.pt` files). Fast loading, widely compatible.

```python
# Explicit PT format
dataset = JASPERSteeringDataset(
    "output/dataset",
    split="train",
    storage_format="pt"
)
```

**Pros**: Fast loading, no extra dependencies  
**Cons**: Larger file size, no compression

### HDF5 Format

Compressed storage with lazy loading support. Requires `h5py`.

```python
# First, convert PT to HDF5 (one-time operation)
JASPERSteeringDataset.convert_pt_to_hdf5("output/dataset")

# Load from HDF5
dataset = JASPERSteeringDataset(
    "output/dataset",
    split="train",
    storage_format="hdf5"
)
```

**Pros**: 30-50% smaller files (gzip compression), lazy loading  
**Cons**: Requires h5py, slightly slower initial load

**When to use**: Large datasets (>5GB), limited disk space, cloud storage

### Auto-Detection

Automatically selects best available format (prefers HDF5 if available):

```python
# Auto-detect: tries HDF5 first, falls back to PT
dataset = JASPERSteeringDataset(
    "output/dataset",
    split="train",
    storage_format="auto"  # Default
)
```

### Converting PT to HDF5

```python
from RAG_supporters.pytorch_datasets import JASPERSteeringDataset

# Convert existing PT dataset to HDF5
JASPERSteeringDataset.convert_pt_to_hdf5(
    "output/dataset",
    compression="gzip"  # Options: gzip, lzf, None
)

# Result: dataset.h5 created alongside *.pt files
# Original PT files preserved for compatibility
```

**Conversion output**:
```
Converting PT format to HDF5 in output/dataset
Converting embeddings...
Converting pair data...
Converting negatives...
Converting steering tensors...
Converting split indices...
Conversion complete. HDF5 file saved to output/dataset/dataset.h5
File sizes: PT format=2500.0 MB, HDF5=1650.0 MB (compression: 34.0%)
```

## Memory-Mapped Loading

For very large datasets (>10GB), enable memory-mapped loading to avoid loading entire dataset into RAM.

### Auto-Enable

Memory-mapping automatically enables for datasets >10GB:

```python
# Auto-enables mmap for large datasets
dataset = JASPERSteeringDataset(
    "output/large_dataset",
    split="train"
    # use_mmap=None (default) -> auto-detects size
)

# Check if mmap is enabled
print(f"Memory-mapping: {dataset.use_mmap}")  # True for >10GB
```

### Explicit Control

```python
# Force enable memory-mapping (small dataset)
dataset = JASPERSteeringDataset(
    "output/dataset",
    split="train",
    use_mmap=True  # Explicit enable
)

# Force disable memory-mapping (large dataset, have enough RAM)
dataset = JASPERSteeringDataset(
    "output/large_dataset",
    split="train",
    use_mmap=False  # Load all to RAM
)
```

### Memory-Mapping Behavior

**Without mmap** (default for <10GB):
- All tensors loaded to RAM during `__init__`
- Fast `__getitem__` (pure memory access)
- High memory usage (entire dataset in RAM)

**With mmap** (auto for >10GB):
- Tensors memory-mapped from disk
- Lazy loading on first access
- Low memory usage (OS pages in/out as needed)
- Slightly slower `__getitem__` (disk I/O on cache miss)

**Best for**:
- Datasets with >10GB compressed size
- Limited RAM environments
- Training with small batch sizes
- Infrequent epoch iteration

**Not compatible with**:
- GPU device preloading (`device != "cpu"`)
- Pinned memory DataLoaders
- High I/O throughput requirements

### Storage Format Comparison

| Format | File Size | Load Speed | RAM Usage | Dependencies |
|--------|-----------|------------|-----------|--------------|
| PT (default) | 100% | Fast | High | torch |
| PT + mmap | 100% | Medium | Low | torch |
| HDF5 | 50-70% | Medium | High | torch, h5py |
| HDF5 + mmap | 50-70% | Slow | Low | torch, h5py |

**Recommendation**:
- **Small datasets (<5GB)**: PT format, no mmap
- **Medium datasets (5-10GB)**: HDF5 format, no mmap  
- **Large datasets (10-100GB)**: HDF5 format, auto mmap
- **Huge datasets (>100GB)**: HDF5 format, explicit mmap

### Combined Example

```python
from RAG_supporters.pytorch_datasets import JASPERSteeringDataset

# Convert to HDF5 once
JASPERSteeringDataset.convert_pt_to_hdf5("datasets/bioasq_large")

# Load with optimal settings
dataset = JASPERSteeringDataset(
    "datasets/bioasq_large",
    split="train",
    storage_format="auto",  # Uses HDF5
    use_mmap=None,          # Auto-enables if >10GB
    device=torch.device("cpu")  # Required for mmap
)

print(f"Format: {dataset.storage_format}")  # "hdf5"
print(f"Memory-mapping: {dataset.use_mmap}")  # True if >10GB
print(f"Memory usage: {dataset._compute_memory_usage():.1f} MB")
```

## Dataset Directory Structure

### PT Format (Default)

```
output/dataset/
├── config.json                      # Dataset metadata
├── train_idx.pt                     # Train split indices [N_train]
├── val_idx.pt                       # Validation split indices [N_val]
├── test_idx.pt                      # Test split indices [N_test]
├── question_embs.pt                 # [N_questions, D]
├── source_embs.pt                   # [N_sources, D]
├── keyword_embs.pt                  # [N_keywords, D]
├── centroid_embs.pt                 # [N_clusters, D]
├── pair_index.pt                    # [N_pairs, 2] (q_idx, s_idx)
├── pair_cluster_id.pt               # [N_pairs] cluster assignments
├── pair_relevance.pt                # [N_pairs] relevance [0, 1]
├── pair_keyword_ids.pt              # List[List[int]] keywords per pair
├── hard_negatives.pt                # [N_pairs, n_neg] negative indices
├── negative_tiers.pt                # [N_pairs, n_neg] difficulty 0-2
├── steering_centroid.pt             # [N_pairs, D]
├── steering_keyword_weighted.pt     # [N_pairs, D]
├── steering_residual.pt             # [N_pairs, D]
└── centroid_distances.pt            # [N_pairs] distance to centroid
```

### HDF5 Format (Optional)

```
output/dataset/
├── config.json                      # Dataset metadata (shared)
├── dataset.h5                       # All tensors in compressed HDF5
│   ├── embeddings/
│   │   ├── questions                # [N_questions, D] compressed
│   │   ├── sources                  # [N_sources, D] compressed
│   │   ├── keywords                 # [N_keywords, D] compressed
│   │   └── centroids                # [N_clusters, D] compressed
│   ├── pair_data/
│   │   ├── index                    # [N_pairs, 2]
│   │   ├── cluster_id               # [N_pairs]
│   │   ├── relevance                # [N_pairs]
│   │   └── keyword_ids/             # Variable-length lists
│   ├── negatives/
│   │   ├── hard                     # [N_pairs, n_neg]
│   │   └── tiers                    # [N_pairs, n_neg]
│   ├── steering/
│   │   ├── centroid                 # [N_pairs, D]
│   │   ├── keyword_weighted         # [N_pairs, D]
│   │   ├── residual                 # [N_pairs, D]
│   │   └── distances                # [N_pairs]
│   └── splits/
│       ├── train                    # [N_train]
│       ├── val                      # [N_val]
│       └── test                     # [N_test]
└── *.pt files (optional, preserved) # Original PT files kept for compatibility
```

### config.json Format

```json
{
  "embedding_dim": 384,
  "n_neg": 5,
  "n_clusters": 50,
  "dataset_name": "bioasq_training",
  "created_at": "2026-02-13T10:30:00",
  "split_sizes": {
    "train": 10000,
    "val": 2000,
    "test": 2000
  }
}
```

## Performance Characteristics

### Memory Usage

For a typical dataset:
- **10K train pairs**, embedding_dim=384, n_neg=5
- **Memory**: ~150 MB total
  - Embeddings: ~50 MB (questions + sources + keywords + centroids)
  - Pairs: ~20 MB (pair_index, cluster_ids, relevance)
  - Negatives: ~10 MB (hard_negatives, tiers)
  - Steering: ~60 MB (3 steering variants × pairs)

### Throughput

- **Zero I/O**: `__getitem__` is pure indexing (~1μs per sample)
- **Batch assembly**: Dominated by DataLoader collation
- **Bottleneck**: Model forward pass, not data loading

### Scaling

- **GPU preloading**: ~2x speedup for small models (reduces CPU→GPU transfer)
- **Workers**: Minimal benefit (data loading is not the bottleneck)
- **Batch size**: Scale freely (no I/O constraint)

## Validation

### Referential Integrity

Dataset validates on initialization:
- Split indices reference valid pairs
- Pair indices reference valid questions/sources
- Hard negatives reference valid sources
- Cluster IDs reference valid centroids
- Keyword IDs reference valid keywords

Raises `ValueError` if any validation fails.

### Batch Validation

```python
from RAG_supporters.pytorch_datasets.loader import validate_first_batch

loader = DataLoader(dataset, batch_size=32)
validate_first_batch(loader)  # Checks shapes, ranges, NaN/Inf
```

## Troubleshooting

### Out of Memory (GPU)

```python
# Don't preload to GPU
dataset = JASPERSteeringDataset("...", split="train")  # device=None (CPU)

# Or reduce batch size
loader = DataLoader(dataset, batch_size=16)  # Instead of 32
```

### Out of Memory (CPU)

```python
# Use memory mapping (not implemented yet - file feature request)
# Current workaround: Reduce dataset size or split into chunks
```

### Slow DataLoader

```python
# Check workers (but likely not the issue)
loader = DataLoader(dataset, num_workers=4)

# Profile to confirm data loading is fast
import time
start = time.time()
batch = next(iter(loader))
print(f"Batch time: {time.time() - start:.3f}s")  # Should be <0.01s
```

## See Also

- [Loader Utilities](LOADER_UTILITIES.md) - DataLoader factory functions
- [JASPER Training Example](../dataset/JASPER_TRAINING_EXAMPLE.md) - Full training workflow
- [Dataset Building](../dataset/JASPER_STEERING_DATASET.md) - How to create datasets
- [Hard Negative Mining](../../RAG_supporters/data_prep/mine_negatives.py) - Negative selection algorithm
