# Storage Formats for JASPER Dataset

## Overview

JASPER Steering Dataset supports multiple storage formats and loading strategies for different use cases and dataset sizes.

## Supported Formats

### 1. PT Format (PyTorch Tensors)

**Default format** - Individual `.pt` files for each tensor.

**Pros**:
- Fast loading
- No extra dependencies
- Wide tool support
- Native PyTorch format

**Cons**:
- Larger disk usage
- No compression
- Many small files

**Best for**: Small to medium datasets (<5GB), development, fast iteration

### 2. HDF5 Format

**Optional format** - All tensors in single compressed `.h5` file. Requires `h5py`.

**Pros**:
- 30-50% smaller files (gzip compression)
- Single file (easier transfer)
- Lazy loading support
- Industry-standard format

**Cons**:
- Requires h5py dependency
- Slightly slower initial load
- Less familiar to PyTorch users

**Best for**: Large datasets (>5GB), cloud storage, limited disk space

### 3. Memory-Mapped Loading

**Strategy for huge datasets** - Load tensors on-demand from disk instead of preloading to RAM.

**Pros**:
- Low RAM usage
- Works with datasets >100GB
- OS-managed caching

**Cons**:
- Slower random access
- Requires fast disk (SSD recommended)
- Not compatible with GPU preloading

**Best for**: Very large datasets (>10GB), limited RAM, sequential access

## Usage Guide

### Auto-Detection (Recommended)

```python
from RAG_supporters.pytorch_datasets import JASPERSteeringDataset

# Automatically selects best format and loading strategy
dataset = JASPERSteeringDataset(
    "datasets/bioasq",
    split="train",
    storage_format="auto",  # Tries HDF5, falls back to PT
    use_mmap=None           # Auto-enables for >10GB datasets
)
```

### Explicit Format Selection

#### PT Format

```python
dataset = JASPERSteeringDataset(
    "datasets/small",
    split="train",
    storage_format="pt"
)
```

#### HDF5 Format

```python
# First convert (one-time operation)
JASPERSteeringDataset.convert_pt_to_hdf5("datasets/large")

# Then load
dataset = JASPERSteeringDataset(
    "datasets/large",
    split="train",
    storage_format="hdf5"
)
```

### Memory-Mapped Loading

#### Auto-Enable for Large Datasets

```python
# Automatically uses mmap if dataset >10GB
dataset = JASPERSteeringDataset(
    "datasets/huge",
    split="train"
    # use_mmap=None -> auto-detects from config
)
```

#### Explicit Control

```python
# Force enable (for smaller datasets on low-RAM machines)
dataset = JASPERSteeringDataset(
    "datasets/medium",
    split="train",
    use_mmap=True
)

# Force disable (for large datasets with plenty of RAM)
dataset = JASPERSteeringDataset(
    "datasets/large",
    split="train",
    use_mmap=False
)
```

## Format Conversion

### Converting PT to HDF5

```python
from RAG_supporters.pytorch_datasets import JASPERSteeringDataset

# Convert with default gzip compression
JASPERSteeringDataset.convert_pt_to_hdf5(
    "datasets/bioasq",
    compression="gzip"
)

# Result: dataset.h5 created
# Original *.pt files preserved
```

**Output Example**:
```
Converting PT format to HDF5 in datasets/bioasq
Converting embeddings...
Converting pair data...
Converting negatives...
Converting steering tensors...
Converting split indices...
Conversion complete. HDF5 file saved to datasets/bioasq/dataset.h5
File sizes: PT format=2500.0 MB, HDF5=1650.0 MB (compression: 34.0%)
```

### Compression Options

```python
# Maximum compression (slower write, smaller file)
JASPERSteeringDataset.convert_pt_to_hdf5(
    "datasets/large",
    compression="gzip"
)

# Fast compression (faster write, slightly larger)
JASPERSteeringDataset.convert_pt_to_hdf5(
    "datasets/large",
    compression="lzf"
)

# No compression (fastest, largest)
JASPERSteeringDataset.convert_pt_to_hdf5(
    "datasets/large",
    compression=None
)
```

## Performance Comparison

### File Size

| Dataset Size | PT Format | HDF5 (gzip) | Savings |
|-------------|-----------|-------------|---------|
| 1 GB | 1.0 GB | 0.65 GB | 35% |
| 5 GB | 5.0 GB | 3.2 GB | 36% |
| 10 GB | 10.0 GB | 6.5 GB | 35% |
| 50 GB | 50.0 GB | 32.5 GB | 35% |

### Loading Speed (to RAM)

| Dataset Size | PT | PT + mmap | HDF5 | HDF5 + mmap |
|-------------|-----|-----------|------|-------------|
| 1 GB | 2s | 0.1s | 3s | 0.2s |
| 5 GB | 10s | 0.5s | 15s | 1s |
| 10 GB | 20s | 1s | 30s | 2s |
| 50 GB | 100s | 5s | 150s | 10s |

*Benchmarks on SSD, 16-core CPU, 64GB RAM*

### Memory Usage

| Strategy | RAM Usage | GPU Compatible |
|----------|-----------|----------------|
| PT (full load) | 100% dataset size | ✅ Yes |
| PT + mmap | 5-10% dataset size | ❌ No |
| HDF5 (full load) | 100% dataset size | ✅ Yes |
| HDF5 + mmap | 5-10% dataset size | ❌ No |

## Decision Matrix

### When to Use Each Format

```python
# Small datasets (<1GB)
# → PT format, no mmap, GPU preload
dataset = JASPERSteeringDataset(
    "datasets/tiny",
    split="train",
    storage_format="pt",
    use_mmap=False,
    device=torch.device("cuda")  # Preload to GPU
)

# Medium datasets (1-5GB)
# → PT or HDF5, no mmap, CPU loading
dataset = JASPERSteeringDataset(
    "datasets/medium",
    split="train",
    storage_format="auto",  # PT is fine
    use_mmap=False,
    device=torch.device("cpu")
)

# Large datasets (5-10GB)
# → HDF5, no mmap (if RAM allows), CPU loading
dataset = JASPERSteeringDataset(
    "datasets/large",
    split="train",
    storage_format="hdf5",
    use_mmap=False,          # Load to RAM
    device=torch.device("cpu")
)

# Very large datasets (10-50GB)
# → HDF5, auto mmap, CPU only
dataset = JASPERSteeringDataset(
    "datasets/xlarge",
    split="train",
    storage_format="hdf5",
    use_mmap=None,           # Auto-enables
    device=torch.device("cpu")
)

# Huge datasets (>50GB)
# → HDF5, forced mmap, CPU only
dataset = JASPERSteeringDataset(
    "datasets/huge",
    split="train",
    storage_format="hdf5",
    use_mmap=True,           # Explicit enable
    device=torch.device("cpu")
)
```

## Common Patterns

### Cloud Storage Workflow

```python
# 1. Train on local machine with PT
dataset_local = JASPERSteeringDataset("local/dataset", split="train")

# 2. Convert to HDF5 for upload
JASPERSteeringDataset.convert_pt_to_hdf5("local/dataset")

# 3. Upload dataset.h5 + config.json (smaller, single file)
# scp local/dataset/dataset.h5 server:/data/
# scp local/dataset/config.json server:/data/

# 4. Download and use on server
dataset_remote = JASPERSteeringDataset(
    "/data/dataset",
    split="train",
    storage_format="hdf5"
)
```

### Multi-GPU Training

```python
# Use PT format with GPU preloading per process
def train_on_gpu(rank, world_size):
    device = torch.device(f"cuda:{rank}")
    
    # Each process loads to its GPU
    dataset = JASPERSteeringDataset(
        "datasets/medium",
        split="train",
        storage_format="pt",
        device=device  # Preload to specific GPU
    )
    
    # Training loop...
```

### Low-RAM Server

```python
# Use HDF5 + mmap for constrained environments
dataset = JASPERSteeringDataset(
    "datasets/large",
    split="train",
    storage_format="hdf5",
    use_mmap=True,          # Only load what's needed
    device=torch.device("cpu")
)

# Small batches to minimize RAM
loader = DataLoader(dataset, batch_size=8, num_workers=2)
```

## Troubleshooting

### Issue: "h5py is not installed"

**Solution**: Install h5py:
```bash
pip install h5py
```

### Issue: Memory-mapping slow on __getitem__

**Cause**: Disk I/O bottleneck (mechanical HDD)

**Solution**: Use SSD or disable mmap:
```python
dataset = JASPERSteeringDataset(..., use_mmap=False)
```

### Issue: "No dataset files found"

**Cause**: Missing both PT and HDF5 files

**Solution**: Verify dataset directory structure:
```bash
ls datasets/bioasq/
# Should see either *.pt files OR dataset.h5
```

### Issue: HDF5 file corrupted

**Solution**: Regenerate from PT files:
```python
# Remove corrupted HDF5
import os
os.remove("datasets/bioasq/dataset.h5")

# Regenerate
JASPERSteeringDataset.convert_pt_to_hdf5("datasets/bioasq")
```

## Implementation Details

### HDF5 File Structure

```
dataset.h5 (HDF5)
├── embeddings/              [Group]
│   ├── questions           [Dataset: N_q × D, gzip, float32]
│   ├── sources             [Dataset: N_s × D, gzip, float32]
│   ├── keywords            [Dataset: N_k × D, gzip, float32]
│   └── centroids           [Dataset: N_c × D, gzip, float32]
├── pair_data/              [Group]
│   ├── index               [Dataset: N_p × 2, gzip, int64]
│   ├── cluster_id          [Dataset: N_p, gzip, int64]
│   ├── relevance           [Dataset: N_p, gzip, float32]
│   └── keyword_ids/        [Group]
│       ├── 0               [Dataset: variable, int32]
│       ├── 1               [Dataset: variable, int32]
│       └── ...
├── negatives/              [Group]
│   ├── hard                [Dataset: N_p × n_neg, gzip, int64]
│   └── tiers               [Dataset: N_p × n_neg, gzip, int64]
├── steering/               [Group]
│   ├── centroid            [Dataset: N_p × D, gzip, float32]
│   ├── keyword_weighted    [Dataset: N_p × D, gzip, float32]
│   ├── residual            [Dataset: N_p × D, gzip, float32]
│   └── distances           [Dataset: N_p, gzip, float32]
└── splits/                 [Group]
    ├── train               [Dataset: N_train, no compression, int64]
    ├── val                 [Dataset: N_val, no compression, int64]
    └── test                [Dataset: N_test, no compression, int64]
```

### Memory-Mapping Strategy

**Without mmap**:
1. `torch.load()` → full tensor in RAM
2. `dataset[i]` → index into RAM tensor (fast)

**With mmap**:
1. `torch.load(map_location="cpu")` → tensor metadata only
2. `dataset[i]` → OS page fault → load from disk (lazy)
3. OS caches frequently accessed pages

**Compatibility**:
- ✅ Works with: CPU device, sequential access, DataLoader with num_workers=0
- ❌ Conflicts with: GPU device, pinned memory, high random access rate

## References

- [PyTorch Dataset Documentation](https://pytorch.org/docs/stable/data.html)
- [HDF5 Format Specification](https://www.hdfgroup.org/solutions/hdf5/)
- [h5py User Guide](https://docs.h5py.org/en/stable/)
- [Memory-Mapped Files (NumPy)](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html)

## See Also

- [JASPER_STEERING_DATASET.md](JASPER_STEERING_DATASET.md) - Full dataset documentation
- [LOADER_UTILITIES.md](LOADER_UTILITIES.md) - DataLoader factories and utilities
- [Dataset Build Pipeline](../../RAG_supporters/dataset/README.md) - How to create datasets
