# Dataset Splitting with Persistent Sample Tracking

**Note**: This documents `data_prep.dataset_splitter.DatasetSplitter` (simple splitter). For stratified splitting with no-leakage guarantees, use `data_prep.dataset_splitter.DatasetSplitter` (stratified mode with `pair_indices` and `pair_cluster_ids`).

Persistent, reproducible train/val splits with JSON-based tracking.

## Quick Start

```python
from RAG_supporters.data_prep.dataset_splitter import DatasetSplitter

# Create and save split
splitter = DatasetSplitter(random_state=42)
train_idx, val_idx = splitter.split(dataset_size=1000, val_ratio=0.2)
splitter.save_split('split.json')

# Load and reuse
splitter2 = DatasetSplitter.load_split('split.json')
train_idx2, val_idx2 = splitter2.get_split()  # Identical to train_idx, val_idx

# One-liner
from RAG_supporters.data_prep.dataset_splitter import create_train_val_split
train_idx, val_idx = create_train_val_split(
    dataset_size=1000, val_ratio=0.2, random_state=42, save_path='split.json'
)
```

## Integration with ClusterLabeledDataset

```python
from RAG_supporters.dataset import ClusterLabeledDataset

# Method 1: Built-in split creation and save
dataset = ClusterLabeledDataset('path/to/dataset')
train_ds, val_ds = dataset.split_dataset(
    val_ratio=0.2, random_state=42, save_path='split.json'
)

# Method 2: Load previously saved split
train_ds, val_ds = ClusterLabeledDataset.load_split(
    dataset_dir='path/to/dataset', split_path='split.json'
)

# Method 3: Manual split creation
splitter = DatasetSplitter(random_state=42)
train_idx, val_idx = splitter.split(len(dataset), val_ratio=0.2)
train_ds = dataset.create_subset(train_idx)
val_ds = dataset.create_subset(val_idx)
```

## Advanced Usage

**Custom validation ratios:**
```python
train_idx, val_idx = splitter.split(dataset_size=1000, val_ratio=0.1)  # 10%
train_idx, val_idx = splitter.split(dataset_size=1000, val_ratio=0.3)  # 30%
```

**Sequential (non-shuffled) splits:**
```python
train_idx, val_idx = splitter.split(dataset_size=100, val_ratio=0.2, shuffle=False)
# val_idx = [0, 1, ..., 19], train_idx = [20, 21, ..., 99]
```

**Add metadata:**
```python
metadata = {"experiment": "baseline", "model": "bert", "date": "2024-01-15"}
splitter.save_split('split.json', metadata=metadata)
```

**Validate compatibility:**
```python
splitter = DatasetSplitter.load_split('split.json')
splitter.validate_split(dataset_size=1000)  # Raises ValueError if incompatible
```

## Complete Training Example

```python
from RAG_supporters.dataset import ClusterLabeledDataset
from torch.utils.data import DataLoader
import torch

# Create and save split
dataset = ClusterLabeledDataset('path/to/dataset', label_type='combined')
train_ds, val_ds = dataset.split_dataset(
    val_ratio=0.2, random_state=42, save_path='model_v1_split.json'
)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

# Training loop
model = YourModel()
for epoch in range(10):
    for base_emb, steering_emb, label in train_loader:
        loss = train_step(model, base_emb, steering_emb, label)
    
    val_loss = sum(validate_step(model, *batch) for batch in val_loader)
    print(f"Epoch {epoch}: val_loss={val_loss:.4f}")

# Later: Resume with exact same split
train_ds, val_ds = ClusterLabeledDataset.load_split(
    dataset_dir='path/to/dataset', split_path='model_v1_split.json'
)
```

## API Reference

**DatasetSplitter(random_state=None)**
- `random_state`: Random seed for reproducibility (None = non-deterministic)

**Methods:**
- `split(dataset_size, val_ratio=0.2, shuffle=True)` → `(train_idx, val_idx)`
  - Raises ValueError if val_ratio ∉ (0,1) or dataset_size ≤ 0
- `save_split(output_path, metadata=None)` - Save split to JSON (raises if no split created)
- `load_split(input_path)` [classmethod] - Load from JSON (raises FileNotFoundError/ValueError)
- `get_split()` → `(train_idx, val_idx)` (raises if no split exists)
- `validate_split(dataset_size)` → bool (raises if incompatible)

**Convenience Function:**
- `create_train_val_split(dataset_size, val_ratio=0.2, random_state=None, shuffle=True, save_path=None, metadata=None)` → `(train_idx, val_idx)`

**ClusterLabeledDataset Methods:**
- `split_dataset(val_ratio=0.2, random_state=None, shuffle=True, save_path=None)` → `(train_subset, val_subset)`
- `load_split(dataset_dir, split_path, label_type='combined', **kwargs)` [static] → `(train_subset, val_subset)`
- `create_subset(indices)` → Subset

**File Format (JSON):**
```json
{
  "train_indices": [4, 8, 15, ...],
  "val_indices": [0, 1, 2, ...],
  "dataset_size": 1000,
  "val_ratio": 0.2,
  "random_state": 42,
  "metadata": {"experiment": "baseline"}
}
```

## Best Practices

1. **Always use random seeds**: `DatasetSplitter(random_state=42)` for reproducibility
2. **Save splits early**: Include `save_path='split.json'` when creating splits
3. **Validate before use**: `splitter.validate_split(len(dataset))` catches incompatibility
4. **Use meaningful metadata**: Include experiment, model, date in metadata dict
5. **Ensure adequate size**: For small datasets, ensure 1/N ≤ val_ratio ≤ (N-1)/N (both sets need ≥1 sample)
6. **Consistent naming**: Use patterns like `{experiment}_{seed}_val{ratio}.json`

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Empty set error | val_ratio too extreme | Ensure 1/N ≤ val_ratio ≤ (N-1)/N |
| ImportError | numpy missing | `pip install numpy` |
| Indices out of bounds | Split from different dataset | `splitter.validate_split(len(dataset))` |
| Non-reproducible | No random seed | Use `random_state=42` |
| File not found | Incorrect path | Use absolute paths or verify location |

## Related Documentation

- [ClusterLabeledDataset](./DOMAIN_ASSESSMENT_CLUSTERING_DATASET_EXAMPLE.md) • [Domain Assessment Examples](./DOMAIN_ASSESSMENT_EXAMPLES.md) • [Dataset README](./README.md)

