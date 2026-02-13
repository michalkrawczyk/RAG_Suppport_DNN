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
    device: Optional[torch.device] = None
)
```

**Parameters**:
- `dataset_dir`: Directory containing dataset files (config.json, *.pt tensors)
- `split`: Which split to use ("train", "val", "test")
- `epoch`: Initial epoch number for curriculum learning (default: 0)
- `device`: Device to transfer tensors to during initialization (default: CPU)

**Raises**:
- `ValueError`: If dataset_dir or split files not found
- `ValueError`: If referential integrity validation fails

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

## Dataset Directory Structure

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
