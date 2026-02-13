# Cluster Labeled Dataset

PyTorch Dataset for domain assessment training with cluster-based labels and efficient memmap storage.

## Overview

`ClusterLabeledDataset` provides triplets of (base_embedding, steering_embedding, label) for training domain classifiers. Uses SQLite for metadata and numpy memmap for efficient embedding access.

**Module**: [RAG_supporters/pytorch_datasets/cluster_labeled_dataset.py](../../RAG_supporters/pytorch_datasets/cluster_labeled_dataset.py)

## Architecture

### Storage Design

```
Dataset directory
├── dataset.db               # SQLite (metadata, labels, indices)
├── base_embeddings.npy      # Memmap [N, D] (base embeddings)
└── steering_embeddings.npy  # Memmap [N, D] (steering embeddings)
```

**Design rationale**:
- **SQLite**: Fast structured queries for metadata
- **Memmap**: Efficient random access without loading full arrays
- **LRU Cache**: Frequently accessed samples cached in memory

### Label Types

Each sample has 3 label types:

1. **source_label**: Label for base embedding (from source/question text)
2. **steering_label**: Label for steering embedding (from domain suggestions)
3. **combined_label**: Weighted average for augmentation masking

Labels are one-hot or soft (probability distributions over clusters).

## API

### Constructor

```python
ClusterLabeledDataset(
    dataset_dir: Union[str, Path],
    label_type: str = "combined",
    return_metadata: bool = False,
    mmap_mode: str = "r",
    cache_size: int = 1000
)
```

**Parameters**:
- `dataset_dir`: Directory containing dataset.db and embedding files
- `label_type`: Which label to return ("source", "steering", "combined")
- `return_metadata`: Whether to return metadata dict with embeddings
- `mmap_mode`: Numpy memmap mode ("r", "r+", "w+", "c")
- `cache_size`: Maximum number of samples in LRU cache

**Raises**:
- `ValueError`: If label_type invalid
- `FileNotFoundError`: If dataset.db not found
- `ValueError`: If embedding shapes mismatch or size validation fails

### Output Formats

#### Without Metadata

```python
base_emb, steering_emb, label = dataset[idx]
# base_emb: Tensor[D]
# steering_emb: Tensor[D]
# label: Tensor[n_clusters]
```

#### With Metadata

```python
base_emb, steering_emb, label, metadata = dataset[idx]
# metadata: {
#     "sample_id": int,
#     "sample_type": str,
#     "text": str,
#     "chroma_id": str,
#     "suggestions": str,
#     "steering_mode": str,
#     "source_label": Tensor[n_clusters],
#     "steering_label": Tensor[n_clusters],
#     "combined_label": Tensor[n_clusters]
# }
```

### Methods

#### `get_sample_by_id(sample_id: int) -> Optional[Dict[str, Any]]`

Retrieve sample by database ID (bypasses index).

```python
sample = dataset.get_sample_by_id(42)
if sample:
    print(sample["text"], sample["steering_mode"])
```

#### `update_labels(...)`

Update labels for manual correction (invalidates cache).

```python
dataset.update_labels(
    sample_id=42,
    source_label=np.array([1.0, 0.0, 0.0]),  # Override source label
    combined_label=None  # Keep existing combined label
)
```

#### `get_cache_stats() -> Dict[str, int]`

Get cache performance statistics.

```python
stats = dataset.get_cache_stats()
# {'hits': 8500, 'misses': 1500, 'hit_rate': 0.85}
```

#### `prefill_cache(indices: List[int])`

Pre-populate cache with specific samples (for warmup).

```python
# Prefill cache with first 1000 samples
dataset.prefill_cache(list(range(1000)))
```

#### `clear_cache()`

Clear the LRU cache (useful after label updates).

```python
dataset.clear_cache()
```

## Label Types Explained

### Source Label

Derived from base text (question or source):
```python
dataset = ClusterLabeledDataset("...", label_type="source")
base_emb, steering_emb, label = dataset[0]
# label represents cluster distribution for the base text
```

**Use case**: Train classifier on original text semantics.

### Steering Label

Derived from domain suggestions:
```python
dataset = ClusterLabeledDataset("...", label_type="steering")
base_emb, steering_emb, label = dataset[0]
# label represents cluster distribution for steering guidance
```

**Use case**: Train classifier on augmented/steered semantics.

### Combined Label

Weighted average of source and steering:
```python
dataset = ClusterLabeledDataset("...", label_type="combined")
base_emb, steering_emb, label = dataset[0]
# label = alpha * source_label + (1 - alpha) * steering_label
```

**Use case**: Train classifier balancing original and augmented semantics.

## Example Usage

### Basic Training Loop

```python
from RAG_supporters.pytorch_datasets import ClusterLabeledDataset
from torch.utils.data import DataLoader
import torch.nn as nn

# Load dataset with combined labels
dataset = ClusterLabeledDataset(
    dataset_dir="output/clusters",
    label_type="combined",
    cache_size=1000
)

loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

# Simple domain classifier
model = nn.Sequential(
    nn.Linear(dataset.embedding_dim * 2, 512),
    nn.ReLU(),
    nn.Linear(512, dataset.n_clusters),
    nn.Softmax(dim=-1)
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(10):
    for base_emb, steering_emb, label in loader:
        # Concatenate base and steering embeddings
        combined = torch.cat([base_emb, steering_emb], dim=-1)
        
        # Forward pass
        output = model(combined)
        loss = criterion(output, label)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### With Metadata

```python
dataset = ClusterLabeledDataset(
    dataset_dir="output/clusters",
    label_type="combined",
    return_metadata=True
)

loader = DataLoader(dataset, batch_size=16, collate_fn=custom_collate)

for base_emb, steering_emb, label, metadata in loader:
    # Access metadata for interpretability
    texts = [m["text"] for m in metadata]
    steering_modes = [m["steering_mode"] for m in metadata]
    
    # Train with metadata-aware loss
    output = model(base_emb, steering_emb)
    loss = weighted_loss(output, label, steering_modes)
```

### Debugging with Cache Stats

```python
dataset = ClusterLabeledDataset("...", cache_size=500)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train one epoch
for batch in loader:
    pass

# Check cache performance
stats = dataset.get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate']:.2%}")
print(f"Hits: {stats['hits']}, Misses: {stats['misses']}")

# Adjust cache size if hit rate is low
if stats['hit_rate'] < 0.7:
    print("Consider increasing cache_size")
```

### Label Correction

```python
import numpy as np

# Train model and identify mislabeled samples
mislabeled_ids = find_mislabeled_samples(model, dataset)

# Correct labels
for sample_id in mislabeled_ids:
    sample = dataset.get_sample_by_id(sample_id)
    
    # Manual review
    print(f"Text: {sample['text']}")
    print(f"Current label: {sample['source_label']}")
    
    # Update if incorrect
    new_label = get_corrected_label()
    dataset.update_labels(sample_id, source_label=new_label)

# Clear cache to reflect updates
dataset.clear_cache()
```

## Performance Optimization

### Cache Size Tuning

```python
# Rule of thumb: cache_size = 10-20% of dataset size
dataset_size = len(dataset)
optimal_cache = max(100, dataset_size // 10)

dataset = ClusterLabeledDataset("...", cache_size=optimal_cache)
```

### Memmap Mode

```python
# Read-only (default) - safe for multi-process
dataset = ClusterLabeledDataset("...", mmap_mode="r")

# Copy-on-write - safe for multi-process with modifications
dataset = ClusterLabeledDataset("...", mmap_mode="c")

# Read-write - NOT safe for multi-process
dataset = ClusterLabeledDataset("...", mmap_mode="r+")
```

### Multi-Worker DataLoader

```python
# Safe with mmap_mode="r" (default)
loader = DataLoader(dataset, batch_size=32, num_workers=4)

# Each worker has separate cache and memmap view
# Cache is NOT shared across workers
```

### Prefilling Cache

```python
# Warmup phase: prefill cache before training
train_indices = list(range(len(dataset)))
random.shuffle(train_indices)

# Prefill with first epoch's samples
dataset.prefill_cache(train_indices[:dataset._cache_size])

# Start training (first epoch has warm cache)
for epoch in range(100):
    for idx in train_indices:
        batch = dataset[idx]
        # Train
```

## Storage Format

### SQLite Schema

```sql
CREATE TABLE samples (
    sample_id INTEGER PRIMARY KEY,
    embedding_idx INTEGER NOT NULL,           -- Index in memmap arrays
    sample_type TEXT NOT NULL,                -- "question" or "source"
    text TEXT NOT NULL,                       -- Original text
    chroma_id TEXT NOT NULL,                  -- ChromaDB identifier
    suggestions TEXT,                         -- Domain suggestions (JSON)
    steering_mode TEXT NOT NULL,              -- Steering method used
    source_label BLOB NOT NULL,               -- Numpy array (pickled)
    steering_label BLOB NOT NULL,             -- Numpy array (pickled)
    combined_label BLOB NOT NULL              -- Numpy array (pickled)
);

CREATE TABLE dataset_info (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
-- Stores: n_clusters, embedding_dim, dataset_size
```

### Embedding Files

```python
# base_embeddings.npy format
np.memmap(
    "base_embeddings.npy",
    dtype=np.float32,
    mode="r",
    shape=(N, D)  # N samples, D dimensions
)

# steering_embeddings.npy format
np.memmap(
    "steering_embeddings.npy",
    dtype=np.float32,
    mode="r",
    shape=(N, D)
)
```

## Thread Safety

### LRU Cache

The cache is thread-safe:
```python
# Safe for DataLoader with multiple workers
self._cache_lock = threading.Lock()

with self._cache_lock:
    # Access cache
    pass
```

### Label Updates

Updates are atomic (DB updated first, then cache invalidated):
```python
# Thread-safe update
dataset.update_labels(sample_id=42, source_label=new_label)
```

## Memory Usage

Typical dataset:
- **10K samples**, 384-dim embeddings, 50 clusters
- **SQLite DB**: ~5 MB (metadata, labels)
- **Memmap files**: 2 × 10K × 384 × 4 bytes = ~30 MB
- **Cache (1K samples)**: ~1K × (768 + overhead) ≈ 1 MB
- **Total**: ~36 MB

Memmap files are not fully loaded:
- Only accessed pages are in memory
- OS manages paging automatically
- Multiple processes share read-only mmap

## Troubleshooting

### High Memory Usage

```python
# Reduce cache size
dataset = ClusterLabeledDataset("...", cache_size=100)

# Check cache stats
stats = dataset.get_cache_stats()
print(f"Cache entries: {len(dataset._sample_cache)}")
```

### Slow Random Access

```python
# Increase cache size
dataset = ClusterLabeledDataset("...", cache_size=2000)

# Check hit rate after one epoch
for batch in loader:
    pass

stats = dataset.get_cache_stats()
if stats['hit_rate'] < 0.7:
    # Increase cache_size further
    pass
```

### Label Update Not Reflected

```python
# Clear cache after bulk updates
for sample_id in updated_ids:
    dataset.update_labels(sample_id, ...)

dataset.clear_cache()  # Force reload from DB
```

### Out of Memory (Multi-Worker)

```python
# Each worker has its own cache
# Total memory = num_workers × cache_size × sample_size

# Reduce per-worker cache
dataset = ClusterLabeledDataset("...", cache_size=200)
loader = DataLoader(dataset, num_workers=4)  # 4 × 200 = 800 cached
```

## See Also

- [JASPER Steering Dataset](JASPER_STEERING_DATASET.md) - Alternative dataset for JASPER training
- [Clustering](../clustering/CLUSTERING_AND_ASSIGNMENT.md) - How clusters are created
- [Domain Assessment](../dataset/DOMAIN_ASSESSMENT_EXAMPLES.md) - Domain labeling workflow
- [SQLite Storage](../../RAG_supporters/jasper/sqlite_storage.py) - Storage backend implementation
