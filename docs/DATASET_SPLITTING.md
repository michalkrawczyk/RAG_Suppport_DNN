# Dataset Splitting with Persistent Sample Tracking

This guide demonstrates how to split datasets into training and validation sets with persistent sample tracking, ensuring consistency across runs and experiments.

## Overview

The dataset splitting functionality provides:
- **Train/Val Split Generation**: Create reproducible splits with configurable ratios
- **Persistent Tracking**: Save split configurations to JSON files
- **Split Restoration**: Load and reuse exact same splits across experiments
- **Dataset Integration**: Seamlessly integrate with `ClusterLabeledDataset`
- **Validation**: Ensure split compatibility with dataset size

## Installation

The dataset splitting functionality is included with the core package:

```python
from RAG_supporters.dataset import DatasetSplitter, create_train_val_split
```

**Dependencies**: `numpy`

## Quick Start

### Basic Usage

```python
from RAG_supporters.dataset import DatasetSplitter

# Create a splitter with a random seed for reproducibility
splitter = DatasetSplitter(random_state=42)

# Split your dataset (100 samples with 20% validation)
train_indices, val_indices = splitter.split(
    dataset_size=100,
    val_ratio=0.2,
    shuffle=True  # Randomly shuffle before splitting
)

print(f"Train samples: {len(train_indices)}")  # 80
print(f"Val samples: {len(val_indices)}")      # 20
```

### Saving and Loading Splits

```python
from RAG_supporters.dataset import DatasetSplitter

# Create and save a split
splitter = DatasetSplitter(random_state=42)
train_idx, val_idx = splitter.split(dataset_size=1000, val_ratio=0.2)
splitter.save_split('my_split.json')

# Later, restore the exact same split
splitter2 = DatasetSplitter.load_split('my_split.json')
train_idx2, val_idx2 = splitter2.get_split()

# Indices will be identical
assert train_idx == train_idx2
assert val_idx == val_idx2
```

### Convenience Function

```python
from RAG_supporters.dataset import create_train_val_split

# One-liner to create and optionally save a split
train_idx, val_idx = create_train_val_split(
    dataset_size=1000,
    val_ratio=0.2,
    random_state=42,
    save_path='split_config.json'  # Optional
)
```

## Integration with ClusterLabeledDataset

### Method 1: Using Built-in Split Methods

```python
from RAG_supporters.dataset import ClusterLabeledDataset

# Load your dataset
dataset = ClusterLabeledDataset('path/to/dataset')

# Split the dataset and save the configuration
train_dataset, val_dataset = dataset.split_dataset(
    val_ratio=0.2,
    random_state=42,
    save_path='experiment_split.json'
)

print(f"Train size: {len(train_dataset)}")
print(f"Val size: {len(val_dataset)}")

# Use with PyTorch DataLoader
from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
```

### Method 2: Loading Previously Saved Splits

```python
from RAG_supporters.dataset import ClusterLabeledDataset

# Load dataset and apply saved split
train_dataset, val_dataset = ClusterLabeledDataset.load_split(
    dataset_dir='path/to/dataset',
    split_path='experiment_split.json'
)

# The same split is restored across runs
for epoch in range(num_epochs):
    for batch in train_loader:
        # Training with consistent train/val split
        ...
```

### Method 3: Manual Split Creation

```python
from RAG_supporters.dataset import ClusterLabeledDataset, DatasetSplitter

# Load dataset
dataset = ClusterLabeledDataset('path/to/dataset')

# Create custom split
splitter = DatasetSplitter(random_state=42)
train_idx, val_idx = splitter.split(len(dataset), val_ratio=0.15)

# Apply split to dataset
train_dataset = dataset.create_subset(train_idx)
val_dataset = dataset.create_subset(val_idx)
```

## Advanced Usage

### Custom Validation Ratios

```python
from RAG_supporters.dataset import DatasetSplitter

splitter = DatasetSplitter(random_state=42)

# Small validation set (10%)
train_idx, val_idx = splitter.split(dataset_size=1000, val_ratio=0.1)

# Large validation set (30%)
train_idx, val_idx = splitter.split(dataset_size=1000, val_ratio=0.3)
```

### Non-Shuffled Splits

```python
# Sequential split (first 20% = validation, rest = training)
splitter = DatasetSplitter()
train_idx, val_idx = splitter.split(
    dataset_size=100,
    val_ratio=0.2,
    shuffle=False
)

# val_idx = [0, 1, ..., 19]
# train_idx = [20, 21, ..., 99]
```

### Adding Metadata to Splits

```python
from RAG_supporters.dataset import DatasetSplitter

splitter = DatasetSplitter(random_state=42)
train_idx, val_idx = splitter.split(dataset_size=1000, val_ratio=0.2)

# Save with experiment metadata
metadata = {
    "experiment_name": "bert_baseline",
    "dataset_version": "v1.2",
    "date": "2024-01-15",
    "notes": "Initial baseline experiment"
}

splitter.save_split('experiment_split.json', metadata=metadata)
```

### Validating Splits

```python
from RAG_supporters.dataset import DatasetSplitter

# Load a saved split
splitter = DatasetSplitter.load_split('experiment_split.json')

# Validate it's compatible with your dataset
try:
    splitter.validate_split(dataset_size=1000)
    print("Split is valid!")
except ValueError as e:
    print(f"Split validation failed: {e}")
```

## Complete Examples

### Example 1: Training with Consistent Splits

```python
from RAG_supporters.dataset import ClusterLabeledDataset
from torch.utils.data import DataLoader
import torch

# First run: Create and save split
dataset = ClusterLabeledDataset('path/to/dataset', label_type='combined')
train_dataset, val_dataset = dataset.split_dataset(
    val_ratio=0.2,
    random_state=42,
    save_path='model_v1_split.json'
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Training loop
model = YourModel()
for epoch in range(10):
    # Train
    for base_emb, steering_emb, label in train_loader:
        loss = train_step(model, base_emb, steering_emb, label)
    
    # Validate
    val_loss = 0
    for base_emb, steering_emb, label in val_loader:
        val_loss += validate_step(model, base_emb, steering_emb, label)
    
    print(f"Epoch {epoch}: val_loss={val_loss:.4f}")

# Save model
torch.save(model.state_dict(), 'model_v1.pth')

# Later: Resume training with exact same split
train_dataset, val_dataset = ClusterLabeledDataset.load_split(
    dataset_dir='path/to/dataset',
    split_path='model_v1_split.json'
)
# Continue training...
```

### Example 2: Hyperparameter Tuning with Fixed Split

```python
from RAG_supporters.dataset import ClusterLabeledDataset
import itertools

# Create split once
dataset = ClusterLabeledDataset('path/to/dataset')
train_dataset, val_dataset = dataset.split_dataset(
    val_ratio=0.2,
    random_state=42,
    save_path='hyperparam_search_split.json'
)

# Try different hyperparameters with same split
learning_rates = [1e-4, 1e-3, 1e-2]
batch_sizes = [16, 32, 64]

best_val_loss = float('inf')
best_params = None

for lr, batch_size in itertools.product(learning_rates, batch_sizes):
    # Load the same split each time
    train_ds, val_ds = ClusterLabeledDataset.load_split(
        dataset_dir='path/to/dataset',
        split_path='hyperparam_search_split.json'
    )
    
    # Train and evaluate
    model = train_model(train_ds, lr=lr, batch_size=batch_size)
    val_loss = evaluate_model(model, val_ds)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_params = {'lr': lr, 'batch_size': batch_size}
    
    print(f"LR={lr}, BS={batch_size}: val_loss={val_loss:.4f}")

print(f"Best params: {best_params}")
```

### Example 3: Cross-Dataset Evaluation

```python
from RAG_supporters.dataset import DatasetSplitter

# Create split for dataset A
splitter_a = DatasetSplitter(random_state=42)
train_a, val_a = splitter_a.split(dataset_size=1000, val_ratio=0.2)
splitter_a.save_split('dataset_a_split.json')

# Create split for dataset B (different random state)
splitter_b = DatasetSplitter(random_state=123)
train_b, val_b = splitter_b.split(dataset_size=800, val_ratio=0.2)
splitter_b.save_split('dataset_b_split.json')

# Later: Load both splits for consistent evaluation
splitter_a = DatasetSplitter.load_split('dataset_a_split.json')
splitter_b = DatasetSplitter.load_split('dataset_b_split.json')

# Use consistent splits across experiments
train_a, val_a = splitter_a.get_split()
train_b, val_b = splitter_b.get_split()
```

## Runnable Examples

Below are practical, self-contained code snippets you can run to test the functionality:

### Example 1: Basic Train/Val Split

```python
from RAG_supporters.dataset import DatasetSplitter

# Create a splitter with reproducible random seed
splitter = DatasetSplitter(random_state=42)

# Split a dataset of 100 samples with 20% validation
train_indices, val_indices = splitter.split(
    dataset_size=100,
    val_ratio=0.2,
    shuffle=True
)

print(f"Dataset size: 100")
print(f"Train samples: {len(train_indices)}")  # 80
print(f"Val samples: {len(val_indices)}")      # 20
print(f"First 5 train indices: {train_indices[:5]}")
print(f"First 5 val indices: {val_indices[:5]}")

# Verify no overlap
overlap = set(train_indices) & set(val_indices)
print(f"Overlap between train and val: {len(overlap)} (should be 0)")
```

### Example 2: Save and Load Split

```python
from RAG_supporters.dataset import DatasetSplitter
from pathlib import Path
import tempfile

with tempfile.TemporaryDirectory() as tmpdir:
    split_path = Path(tmpdir) / "my_split.json"
    
    # Create and save a split
    print("Creating and saving split...")
    splitter1 = DatasetSplitter(random_state=42)
    train_idx1, val_idx1 = splitter1.split(dataset_size=100, val_ratio=0.2)
    
    # Add metadata
    metadata = {
        "experiment": "baseline",
        "model": "bert",
        "date": "2024-01-15"
    }
    splitter1.save_split(split_path, metadata=metadata)
    print(f"Split saved to: {split_path}")
    
    # Load the split
    print("\nLoading split...")
    splitter2 = DatasetSplitter.load_split(split_path)
    train_idx2, val_idx2 = splitter2.get_split()
    
    # Verify they're identical
    print(f"Splits are identical: {train_idx1 == train_idx2 and val_idx1 == val_idx2}")
    print(f"Train indices match: {train_idx1[:5]} == {train_idx2[:5]}")
```

### Example 3: Using Convenience Function

```python
from RAG_supporters.dataset import create_train_val_split

# One-liner to create a split
train_idx, val_idx = create_train_val_split(
    dataset_size=100,
    val_ratio=0.2,
    random_state=42,
    shuffle=True
)

print(f"Created split with convenience function")
print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
```

### Example 4: Different Validation Ratios

```python
from RAG_supporters.dataset import DatasetSplitter

dataset_size = 1000
ratios = [0.1, 0.2, 0.3]

for ratio in ratios:
    splitter = DatasetSplitter(random_state=42)
    train_idx, val_idx = splitter.split(
        dataset_size=dataset_size,
        val_ratio=ratio
    )
    print(f"Val ratio {ratio:.1f}: Train={len(train_idx)}, Val={len(val_idx)}")
```

### Example 5: Reproducibility Test

```python
from RAG_supporters.dataset import DatasetSplitter

# Create two splitters with same seed
splitter1 = DatasetSplitter(random_state=42)
train1, val1 = splitter1.split(dataset_size=100, val_ratio=0.2)

splitter2 = DatasetSplitter(random_state=42)
train2, val2 = splitter2.split(dataset_size=100, val_ratio=0.2)

print(f"Same random seed (42):")
print(f"  Splits are identical: {train1 == train2 and val1 == val2}")

# Create splitter with different seed
splitter3 = DatasetSplitter(random_state=123)
train3, val3 = splitter3.split(dataset_size=100, val_ratio=0.2)

print(f"Different random seed (123):")
print(f"  Splits are different: {train1 != train3 or val1 != val3}")
print(f"  But sizes are same: Train={len(train3)}, Val={len(val3)}")
```

### Example 6: Split Validation

```python
from RAG_supporters.dataset import DatasetSplitter

# Create a split for dataset of size 100
splitter = DatasetSplitter(random_state=42)
train_idx, val_idx = splitter.split(dataset_size=100, val_ratio=0.2)

# Validate against correct size
try:
    splitter.validate_split(dataset_size=100)
    print("✓ Split is valid for dataset size 100")
except ValueError as e:
    print(f"✗ Validation failed: {e}")

# Try to validate against wrong size
try:
    splitter.validate_split(dataset_size=50)
    print("✓ Split is valid for dataset size 50")
except ValueError as e:
    print(f"✗ Validation failed for size 50 (expected): {str(e)[:60]}...")
```

## API Reference

### DatasetSplitter

#### `__init__(random_state=None)`

Initialize a DatasetSplitter.

**Parameters:**
- `random_state` (int, optional): Random seed for reproducibility. If None, splits are non-deterministic.

#### `split(dataset_size, val_ratio=0.2, shuffle=True)`

Split dataset indices into training and validation sets.

**Parameters:**
- `dataset_size` (int): Total number of samples in the dataset
- `val_ratio` (float): Ratio of validation samples (0 < val_ratio < 1). Default: 0.2
- `shuffle` (bool): Whether to shuffle indices before splitting. Default: True

**Returns:**
- `Tuple[List[int], List[int]]`: (train_indices, val_indices)

**Raises:**
- `ValueError`: If val_ratio is not between 0 and 1, or dataset_size is not positive

#### `save_split(output_path, metadata=None)`

Save split configuration to a JSON file.

**Parameters:**
- `output_path` (str or Path): Path where the split configuration will be saved
- `metadata` (dict, optional): Additional metadata to save with the split

**Raises:**
- `ValueError`: If no split has been created

#### `load_split(input_path)` [classmethod]

Load split configuration from a JSON file.

**Parameters:**
- `input_path` (str or Path): Path to the split configuration file

**Returns:**
- `DatasetSplitter`: DatasetSplitter instance with loaded split

**Raises:**
- `FileNotFoundError`: If the split configuration file does not exist
- `ValueError`: If the file format is invalid

#### `get_split()`

Get the current split.

**Returns:**
- `Tuple[List[int], List[int]]`: (train_indices, val_indices)

**Raises:**
- `ValueError`: If no split has been created or loaded

#### `validate_split(dataset_size)`

Validate that the current split is compatible with a dataset.

**Parameters:**
- `dataset_size` (int): Size of the dataset to validate against

**Returns:**
- `bool`: True if split is valid

**Raises:**
- `ValueError`: If split is invalid or incompatible

### Convenience Function

#### `create_train_val_split(dataset_size, val_ratio=0.2, random_state=None, shuffle=True, save_path=None, metadata=None)`

Convenience function to create a train/val split.

**Parameters:**
- `dataset_size` (int): Total number of samples
- `val_ratio` (float): Ratio of validation samples. Default: 0.2
- `random_state` (int, optional): Random seed for reproducibility
- `shuffle` (bool): Whether to shuffle before splitting. Default: True
- `save_path` (str or Path, optional): If provided, save split to this path
- `metadata` (dict, optional): Additional metadata to save

**Returns:**
- `Tuple[List[int], List[int]]`: (train_indices, val_indices)

### ClusterLabeledDataset Methods

#### `split_dataset(val_ratio=0.2, random_state=None, shuffle=True, save_path=None)`

Split this dataset into training and validation subsets.

**Returns:**
- `Tuple[Subset, Subset]`: (train_subset, val_subset)

#### `load_split(dataset_dir, split_path, label_type='combined', **dataset_kwargs)` [staticmethod]

Load a dataset and apply a saved split configuration.

**Returns:**
- `Tuple[Subset, Subset]`: (train_subset, val_subset)

#### `create_subset(indices)`

Create a subset of this dataset using the provided indices.

**Returns:**
- `Subset`: PyTorch Subset wrapping this dataset

## Best Practices

### 1. Always Use Random Seeds for Reproducibility

```python
# Good: Reproducible splits
splitter = DatasetSplitter(random_state=42)

# Avoid: Non-reproducible splits
splitter = DatasetSplitter(random_state=None)
```

### 2. Ensure Adequate Dataset Size

```python
# Both train and validation sets must have at least one sample
# For small datasets, choose val_ratio carefully:

# Minimum dataset size is 2
dataset_size = 10
min_val_ratio = 1.0 / dataset_size  # 0.1 - ensures val set has >= 1 sample
max_val_ratio = (dataset_size - 1) / dataset_size  # 0.9 - ensures train set has >= 1 sample

# Good: Appropriate ratio for small dataset
splitter.split(dataset_size=10, val_ratio=0.2)  # 8 train, 2 val

# Will raise ValueError: Empty validation set
# splitter.split(dataset_size=10, val_ratio=0.05)  # Would give 0 val samples
```

### 3. Save Splits Early in Your Experiment

```python
# Save split immediately after creation
train_dataset, val_dataset = dataset.split_dataset(
    val_ratio=0.2,
    random_state=42,
    save_path='experiment_split.json'  # Save it!
)
```

### 4. Validate Splits Before Using

```python
# Load and validate before using
splitter = DatasetSplitter.load_split('split.json')
splitter.validate_split(len(dataset))  # Ensure compatibility
train_idx, val_idx = splitter.get_split()
```

### 5. Use Meaningful Metadata

```python
metadata = {
    "experiment_id": "exp_001",
    "model": "transformer_base",
    "dataset_version": "v2.1",
    "created_by": "researcher_name",
    "purpose": "baseline_comparison"
}
splitter.save_split('split.json', metadata=metadata)
```

### 5. Consistent Naming Convention

```python
# Use descriptive, consistent names
'experiment_{name}_split.json'
'model_{version}_split.json'
'dataset_{name}_seed{seed}_val{val_ratio}.json'
```

## Troubleshooting

### Issue: Empty validation or training set

**Problem**: ValueError raised: "Validation set would be empty" or "Training set would be empty"

**Cause**: Dataset is too small for the chosen `val_ratio`, resulting in one set having zero samples.

**Solution**: Adjust `val_ratio` to ensure both sets have at least one sample:
```python
# For small datasets, calculate appropriate ratio
dataset_size = 5
min_val_ratio = 1.0 / dataset_size  # 0.2 - minimum to get 1 val sample
max_val_ratio = (dataset_size - 1) / dataset_size  # 0.8 - maximum to keep 1 train sample

# Use a ratio within valid range
splitter = DatasetSplitter(random_state=42)
train_idx, val_idx = splitter.split(dataset_size=5, val_ratio=0.2)  # Works: 4 train, 1 val
```

### Issue: ImportError when using DatasetSplitter

**Solution**: Ensure numpy is installed:
```bash
pip install numpy
```

### Issue: Split indices out of bounds

**Problem**: Loaded split configuration has indices larger than dataset size.

**Solution**: Validate the split before use:
```python
splitter = DatasetSplitter.load_split('split.json')
try:
    splitter.validate_split(len(dataset))
except ValueError as e:
    print(f"Split is incompatible: {e}")
    # Create a new split
```

### Issue: Different results each run

**Problem**: Not using a random seed.

**Solution**: Always specify `random_state`:
```python
splitter = DatasetSplitter(random_state=42)  # Reproducible
```

### Issue: Can't find split file

**Problem**: Split file path is incorrect or file was moved.

**Solution**: Use absolute paths or check file location:
```python
from pathlib import Path

split_path = Path('experiments/exp_001/split.json').absolute()
splitter = DatasetSplitter.load_split(split_path)
```

## File Format

Split configurations are saved as JSON files with the following structure:

```json
{
  "train_indices": [4, 8, 15, 16, 23, ...],
  "val_indices": [0, 1, 2, 3, ...],
  "dataset_size": 1000,
  "val_ratio": 0.2,
  "random_state": 42,
  "metadata": {
    "experiment_name": "baseline",
    "date": "2024-01-15"
  }
}
```

## Related Documentation

- [ClusterLabeledDataset Documentation](./DOMAIN_ASSESSMENT_CLUSTERING_DATASET_EXAMPLE.md)
- [Domain Assessment Examples](./DOMAIN_ASSESSMENT_EXAMPLES.md)
- [README](./README.md)

## Support

For questions or issues:
- Check this documentation
- Review the API reference above
- Ensure all dependencies are installed
- Validate your split configurations
