# DataLoader Utilities

Factory functions and validation utilities for creating PyTorch DataLoaders with JASPER Steering Dataset.

## Overview

The `loader` module provides high-level functions for creating properly configured DataLoaders with support for distributed training, curriculum learning, and batch validation.

**Module**: [RAG_supporters/pytorch_datasets/loader.py](../../RAG_supporters/pytorch_datasets/loader.py)

## Functions

### `create_loader()`

Create a DataLoader with automatic configuration.

```python
def create_loader(
    dataset_dir: str | Path,
    split: Literal["train", "val", "test"],
    batch_size: int,
    num_workers: int = 0,
    distributed: bool = False,
    epoch: int = 0,
    drop_last: Optional[bool] = None,
    pin_memory: bool = True,
) -> DataLoader
```

**Parameters**:
- `dataset_dir`: Directory containing dataset files
- `split`: Which split to load ("train", "val", "test")
- `batch_size`: Batch size for the DataLoader
- `num_workers`: Number of worker processes (default: 0)
- `distributed`: Whether to use DistributedSampler (default: False)
- `epoch`: Initial epoch for curriculum learning (default: 0)
- `drop_last`: Drop last incomplete batch (default: True for train, False for val/test)
- `pin_memory`: Pin memory for faster GPU transfer (default: True)

**Returns**: Configured PyTorch DataLoader

**Attached Attributes**:
- `loader.dataset_obj`: Reference to underlying JASPERSteeringDataset
- `loader.sampler_obj`: Reference to sampler (if distributed)

**Example**:
```python
from RAG_supporters.pytorch_datasets.loader import create_loader

# Single-GPU training
train_loader = create_loader(
    dataset_dir="output/dataset",
    split="train",
    batch_size=32,
    num_workers=4
)

# Multi-GPU training
train_loader = create_loader(
    dataset_dir="output/dataset",
    split="train",
    batch_size=32,
    num_workers=4,
    distributed=True  # Uses DistributedSampler
)

# Validation
val_loader = create_loader(
    dataset_dir="output/dataset",
    split="val",
    batch_size=64,
    num_workers=2
)
```

### `set_epoch()`

Set epoch for both dataset and sampler (curriculum + distributed training).

```python
def set_epoch(loader: DataLoader, epoch: int)
```

**Parameters**:
- `loader`: DataLoader created by `create_loader()`
- `epoch`: Epoch number to set

**Effect**:
- Updates **dataset** curriculum (steering probabilities)
- Updates **sampler** epoch (distributed shuffling seed)

**Example**:
```python
from RAG_supporters.pytorch_datasets.loader import create_loader, set_epoch

loader = create_loader("output/dataset", "train", batch_size=32, distributed=True)

for epoch in range(100):
    # Update both curriculum and distributed sampler
    set_epoch(loader, epoch)
    
    for batch in loader:
        # Train model
        pass
```

**Why Both?**
- **Dataset epoch**: Controls curriculum learning (steering probability schedule)
- **Sampler epoch**: Seeds shuffling for distributed training (ensures different order per epoch)

### `validate_first_batch()`

Validate that the first batch has correct structure and values.

```python
def validate_first_batch(loader: DataLoader) -> bool
```

**Parameters**:
- `loader`: DataLoader to validate

**Returns**: True if validation passes

**Raises**: AssertionError with descriptive message if any check fails

**Checks Performed**:
- All expected keys present
- Tensor shapes match expectations
- No NaN or Inf values in embeddings
- Cluster IDs in valid range [0, n_clusters)
- Relevance in range [0, 1]
- Centroid distance in range [0, 2]
- Steering variant in range [0, 3]

**Example**:
```python
from RAG_supporters.pytorch_datasets.loader import create_loader, validate_first_batch

loader = create_loader("output/dataset", "train", batch_size=32)

# Validate before training
try:
    validate_first_batch(loader)
    print("✓ DataLoader validation passed")
except AssertionError as e:
    print(f"✗ Validation failed: {e}")
    exit(1)

# Start training
for epoch in range(100):
    for batch in loader:
        # Train
        pass
```

## Usage Patterns

### Single-GPU Training

```python
from RAG_supporters.pytorch_datasets.loader import create_loader, set_epoch

# Create loaders
train_loader = create_loader("output/dataset", "train", batch_size=32, num_workers=4)
val_loader = create_loader("output/dataset", "val", batch_size=64, num_workers=2)

# Training loop
for epoch in range(100):
    # Update curriculum
    set_epoch(train_loader, epoch)
    
    # Train
    model.train()
    for batch in train_loader:
        # Forward + backward
        pass
    
    # Validate
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            # Evaluate
            pass
```

### Multi-GPU Training (DDP)

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from RAG_supporters.pytorch_datasets.loader import create_loader, set_epoch

def main():
    # Initialize distributed training
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")
    
    # Create model with DDP
    model = MyModel().to(device)
    model = DDP(model, device_ids=[rank])
    
    # Create distributed loaders
    train_loader = create_loader(
        dataset_dir="output/dataset",
        split="train",
        batch_size=32,
        num_workers=4,
        distributed=True  # Important!
    )
    
    # Training loop
    for epoch in range(100):
        # Sync epoch across all processes
        set_epoch(train_loader, epoch)
        
        for batch in train_loader:
            # Each GPU gets different samples (sharded by DistributedSampler)
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward + backward
            output = model(batch["question_emb"], batch["steering"])
            loss = compute_loss(output, batch)
            loss.backward()
            optimizer.step()
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

### Debugging with Validation

```python
from RAG_supporters.pytorch_datasets.loader import create_loader, validate_first_batch

# Create loader
loader = create_loader("output/dataset", "train", batch_size=32)

# Validate structure
print("Validating DataLoader...")
validate_first_batch(loader)

# Inspect batch manually
batch = next(iter(loader))
print(f"Batch keys: {batch.keys()}")
print(f"question_emb shape: {batch['question_emb'].shape}")
print(f"steering shape: {batch['steering'].shape}")
print(f"negative_embs shape: {batch['negative_embs'].shape}")
print(f"cluster_id range: [{batch['cluster_id'].min()}, {batch['cluster_id'].max()}]")
print(f"relevance range: [{batch['relevance'].min():.3f}, {batch['relevance'].max():.3f}]")
```

### Progressive Batch Size

```python
from RAG_supporters.pytorch_datasets.loader import create_loader, set_epoch

def get_batch_size(epoch: int) -> int:
    """Progressive batch size schedule."""
    if epoch < 10:
        return 16
    elif epoch < 30:
        return 32
    else:
        return 64

# Training with dynamic batch size
for epoch in range(100):
    batch_size = get_batch_size(epoch)
    
    # Recreate loader with new batch size
    train_loader = create_loader(
        "output/dataset", "train",
        batch_size=batch_size,
        num_workers=4
    )
    
    set_epoch(train_loader, epoch)
    
    for batch in train_loader:
        # Train
        pass
```

### Validation with Fixed Steering

```python
from RAG_supporters.pytorch_datasets import JASPERSteeringDataset
from torch.utils.data import DataLoader

# For validation: use fixed steering for consistency
val_dataset = JASPERSteeringDataset("output/dataset", split="val")
val_dataset.force_steering("centroid")  # Fix to centroid steering

val_loader = DataLoader(val_dataset, batch_size=64, num_workers=2)

# All batches use centroid steering (no curriculum randomness)
for batch in val_loader:
    # Validate
    pass
```

## Configuration Guidelines

### `batch_size`

- **Train**: 32-64 (depends on GPU memory and embedding dim)
- **Val/Test**: 64-128 (can be larger, no gradients)

### `num_workers`

- **CPU preloading**: 2-4 workers (diminishing returns beyond 4)
- **GPU preloading**: 0 workers (data already on GPU)
- **Debugging**: 0 workers (easier to trace errors)

Rule of thumb: `num_workers = min(4, num_cpu_cores // 2)`

### `distributed`

- **Single GPU**: `distributed=False`
- **Multi-GPU (DDP)**: `distributed=True`
- **DataParallel**: Not recommended (use DDP instead)

### `drop_last`

- **Train**: `drop_last=True` (consistent batch size for batch norm)
- **Val/Test**: `drop_last=False` (evaluate all samples)

### `pin_memory`

- **GPU training**: `pin_memory=True` (faster CPU→GPU transfer)
- **CPU training**: `pin_memory=False` (no benefit)

## Performance Optimization

### Throughput

For JASPERSteeringDataset:
- **Bottleneck**: Model forward pass, not data loading
- **Workers**: 2-4 is optimal (dataset preloads all data)
- **Batch size**: Scale to GPU memory limit

### Profiling

```python
import time

loader = create_loader("output/dataset", "train", batch_size=32, num_workers=4)

# Profile data loading
times = []
for i, batch in enumerate(loader):
    start = time.time()
    batch = {k: v.to("cuda") for k, v in batch.items()}
    times.append(time.time() - start)
    
    if i >= 100:
        break

print(f"Avg batch time: {np.mean(times) * 1000:.2f} ms")
print(f"Std batch time: {np.std(times) * 1000:.2f} ms")
# Should be < 5ms for JASPERSteeringDataset
```

### Memory Usage

```python
import torch

loader = create_loader("output/dataset", "train", batch_size=32)

# Check memory before/after first batch
torch.cuda.reset_peak_memory_stats()
batch = next(iter(loader))
batch = {k: v.to("cuda") for k, v in batch.items()}

peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
print(f"Peak memory: {peak_mb:.1f} MB for batch")
```

## Error Handling

### Common Errors

#### "Split file not found"

```python
# Error: val_idx.pt not found
loader = create_loader("output/dataset", "val", batch_size=32)

# Solution: Check dataset directory structure
# Should contain: train_idx.pt, val_idx.pt, test_idx.pt
```

#### "Referential integrity failure"

```python
# Error: "pair_index references source 500, but only 400 sources exist"

# Solution: Rebuild dataset with correct indices
# Check dataset building script
```

#### "NaN detected in embeddings"

```python
# Error from validate_first_batch()
loader = create_loader("output/dataset", "train", batch_size=32)
validate_first_batch(loader)  # AssertionError: NaN detected in question_emb

# Solution: Check embedding generation
# Likely issue: Division by zero in normalization
```

### Debugging Tips

```python
# 1. Validate first
loader = create_loader("...", "train", batch_size=32)
validate_first_batch(loader)

# 2. Inspect dataset directly
from RAG_supporters.pytorch_datasets import JASPERSteeringDataset
dataset = JASPERSteeringDataset("...", split="train")
sample = dataset[0]
print({k: v.shape if torch.is_tensor(v) else v for k, v in sample.items()})

# 3. Check config
import json
with open("output/dataset/config.json") as f:
    config = json.load(f)
print(config)

# 4. Test with small batch
loader = create_loader("...", "train", batch_size=2, num_workers=0)
batch = next(iter(loader))
```

## See Also

- [JASPER Steering Dataset](JASPER_STEERING_DATASET.md) - Dataset implementation
- [JASPER Training Example](../dataset/JASPER_TRAINING_EXAMPLE.md) - Full training script
- [PyTorch DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) - Official docs
- [Distributed Training](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) - DDP tutorial
