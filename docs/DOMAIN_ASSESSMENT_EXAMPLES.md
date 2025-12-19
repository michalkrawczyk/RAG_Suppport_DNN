# Domain Assessment Dataset - Usage Examples

This document provides practical examples for the new **domain assessment dataset approach** using CSV files from `domain_assessment.py` and clustering JSON from `keyword_clustering.py`.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Dataset Building](#dataset-building)
3. [Dataset Loading](#dataset-loading)
4. [Label Types](#label-types)
5. [Steering Modes](#steering-modes)
6. [Augmentation](#augmentation)
7. [Manual Label Correction](#manual-label-correction)
8. [Training Loop Integration](#training-loop-integration)

---

## Quick Start

### One-Line Dataset Creation

```python
from RAG_supporters.dataset import ClusterLabeledDataset
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Build and load dataset in one step
dataset = ClusterLabeledDataset.create_from_csvs(
    csv_paths='data/domain_assessment.csv',
    clustering_json_path='data/clustering_results.json',
    output_dir='datasets/my_dataset',
    embedding_model=model
)

# Use in training
for base_emb, steering_emb, label in dataset:
    # Train your model...
    pass
```

---

## Dataset Building

### Basic Build

```python
from RAG_supporters.dataset import DomainAssessmentDatasetBuilder
from RAG_supporters.dataset.steering import SteeringConfig, SteeringMode
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Configure steering (optional - defaults to ZERO mode)
steering_config = SteeringConfig(
    mode=[(SteeringMode.SUGGESTION, 0.6), (SteeringMode.CLUSTER_DESCRIPTOR, 0.4)],
    random_seed=42
)

# Build dataset
builder = DomainAssessmentDatasetBuilder(
    csv_paths=['data/sources.csv', 'data/questions.csv'],  # Can be multiple files
    clustering_json_path='data/clustering_results.json',
    output_dir='datasets/my_dataset',
    embedding_model=model,
    steering_config=steering_config,
    label_normalizer='softmax',  # Options: 'softmax', 'l1', 'l2'
    label_temp=1.0,  # Temperature for softmax
    combined_label_weight=0.5  # Weight for combined labels
)

builder.build()
builder.close()

print("Dataset built successfully!")
```

### With Augmentation

```python
builder = DomainAssessmentDatasetBuilder(
    csv_paths='data/domain_assessment.csv',
    clustering_json_path='data/clustering_results.json',
    output_dir='datasets/augmented_dataset',
    embedding_model=model,
    augment_noise_prob=0.2,  # 20% chance of noise
    augment_zero_prob=0.1,   # 10% chance of zeroing steering
    augment_noise_level=0.01  # Noise std deviation
)

builder.build()
builder.close()
```

### Chunked CSV Reading (Large Datasets)

```python
builder = DomainAssessmentDatasetBuilder(
    csv_paths='data/large_dataset.csv',
    clustering_json_path='data/clustering_results.json',
    output_dir='datasets/large_dataset',
    embedding_model=model,
    chunk_size=10000  # Read 10k rows at a time
)

builder.build()
builder.close()
```

---

## Dataset Loading

### Basic Loading

```python
from RAG_supporters.dataset import ClusterLabeledDataset

# Load pre-built dataset
dataset = ClusterLabeledDataset(
    dataset_dir='datasets/my_dataset',
    label_type='combined',  # Options: 'source', 'steering', 'combined'
    return_metadata=False
)

print(f"Dataset size: {len(dataset)}")

# Get a sample
base_emb, steering_emb, label = dataset[0]
print(f"Base embedding shape: {base_emb.shape}")
print(f"Steering embedding shape: {steering_emb.shape}")
print(f"Label shape: {label.shape}")  # Shape: (n_clusters,)
```

### With Metadata

```python
dataset = ClusterLabeledDataset(
    dataset_dir='datasets/my_dataset',
    label_type='combined',
    return_metadata=True  # Include metadata
)

base_emb, steering_emb, label, metadata = dataset[0]
print(f"Sample ID: {metadata['sample_id']}")
print(f"Sample type: {metadata['sample_type']}")  # 'source' or 'question'
print(f"Text: {metadata['text']}")
print(f"Suggestions: {metadata['suggestions']}")
print(f"Steering mode: {metadata['steering_mode']}")
print(f"Source label: {metadata['source_label']}")
print(f"Steering label: {metadata['steering_label']}")
print(f"Combined label: {metadata['combined_label']}")
```

---

## Label Types

The dataset provides **3 types of labels** for flexibility:

### 1. Source/Question Labels

Labels based on the source text or question itself.

```python
dataset = ClusterLabeledDataset(
    dataset_dir='datasets/my_dataset',
    label_type='source'  # Use source/question labels
)

for base_emb, steering_emb, source_label in dataset:
    # source_label: probability distribution over clusters
    # Based on CSV cluster_probabilities or suggestion embeddings
    pass
```

**Use case**: When steering embedding quality is uncertain or for baseline comparison.

### 2. Steering Labels

Labels based on steering embedding distances to cluster centroids.

```python
dataset = ClusterLabeledDataset(
    dataset_dir='datasets/my_dataset',
    label_type='steering'  # Use steering labels
)

for base_emb, steering_emb, steering_label in dataset:
    # steering_label: probability distribution from steering embedding
    pass
```

**Use case**: When you want to explicitly train on steering signals.

### 3. Combined Labels (Recommended)

Weighted average of source and steering labels.

```python
dataset = ClusterLabeledDataset(
    dataset_dir='datasets/my_dataset',
    label_type='combined'  # Use combined labels (DEFAULT)
)

for base_emb, steering_emb, combined_label in dataset:
    # combined_label = (1-w)*source_label + w*steering_label
    # w = combined_label_weight from builder (default 0.5)
    pass
```

**Use case**: Balanced approach that considers both base text and steering. Best for augmentation masking.

---

## Steering Modes

Configure different steering embedding modes:

### ZERO Mode (No Steering)

```python
from RAG_supporters.dataset.steering import SteeringConfig, SteeringMode

config = SteeringConfig(
    mode=[(SteeringMode.ZERO, 1.0)]  # Always use zero embedding
)
```

**Use case**: Baseline / ablation studies. Zero steering uses only source/question labels.

### SUGGESTION Mode

```python
config = SteeringConfig(
    mode=[(SteeringMode.SUGGESTION, 1.0)]
)
```

Uses suggestion embeddings from CSV (averaged if multiple).

### CLUSTER_DESCRIPTOR Mode

```python
config = SteeringConfig(
    mode=[(SteeringMode.CLUSTER_DESCRIPTOR, 1.0)]
)
```

Uses cluster topic descriptors from clustering JSON.

### LLM_GENERATED Mode

```python
config = SteeringConfig(
    mode=[(SteeringMode.LLM_GENERATED, 1.0)]
)

# Provide LLM-generated steering texts
builder = DomainAssessmentDatasetBuilder(
    ...,
    steering_config=config,
    llm_steering_texts={'0': 'Generated text for sample 0', ...}
)
```

### MIXED Mode

```python
config = SteeringConfig(
    mode=[(SteeringMode.MIXED, 1.0)],
    mixed_weights={
        'suggestion': 0.5,
        'cluster_descriptor': 0.3,
        'llm_generated': 0.2
    }
)
```

Weighted combination of multiple modes.

### Probabilistic Mode Selection

```python
config = SteeringConfig(
    mode=[
        (SteeringMode.SUGGESTION, 0.5),
        (SteeringMode.CLUSTER_DESCRIPTOR, 0.3),
        (SteeringMode.ZERO, 0.2)
    ],
    random_seed=42
)
```

Randomly selects mode per sample based on probabilities.

---

## Augmentation

Integrate augmentations from `RAG_supporters/augmentations/embedding.py`:

### Noise Augmentation

```python
builder = DomainAssessmentDatasetBuilder(
    ...,
    augment_noise_prob=0.3,  # 30% chance
    augment_noise_level=0.01  # Std deviation
)
```

Adds Gaussian noise to steering embeddings using `random_noise_embedding()`.

### Zero Augmentation

```python
builder = DomainAssessmentDatasetBuilder(
    ...,
    augment_zero_prob=0.1  # 10% chance of zeroing steering
)
```

Randomly zeros steering embeddings using `random_zero_embedding()`.

**Important**: When steering is zeroed, the combined label automatically uses more weight from source label.

### Both Augmentations

```python
builder = DomainAssessmentDatasetBuilder(
    ...,
    augment_noise_prob=0.2,
    augment_zero_prob=0.1,
    augment_noise_level=0.005
)
```

Zero augmentation has higher priority (if applied, skip noise).

---

## Manual Label Correction

The SQLite storage allows manual label correction:

### Inspect and Correct Labels

```python
from RAG_supporters.dataset import ClusterLabeledDataset
import numpy as np

dataset = ClusterLabeledDataset('datasets/my_dataset')

# Get sample by ID
sample = dataset.get_sample_by_id(sample_id=42)
print(f"Current source label: {sample['source_label']}")

# Correct label (e.g., based on manual review)
new_source_label = np.array([0.1, 0.8, 0.05, 0.05], dtype=np.float32)

dataset.update_labels(
    sample_id=42,
    source_label=new_source_label
)

print("Label updated!")

# Can also update steering and combined labels
dataset.update_labels(
    sample_id=42,
    source_label=new_source_label,
    steering_label=new_steering_label,
    combined_label=new_combined_label
)
```

### Direct SQLite Access

```python
from RAG_supporters.dataset import SQLiteStorageManager

with SQLiteStorageManager('datasets/my_dataset/dataset.db') as storage:
    # Get all samples
    samples = storage.get_all_samples()
    
    # Update labels
    storage.update_labels(sample_id=42, source_label=new_label)
```

Or use any SQLite GUI tool (e.g., DB Browser for SQLite) to edit labels directly in the `samples` table.

---

## Training Loop Integration

### Basic Training Loop

```python
import torch
from torch.utils.data import DataLoader
from RAG_supporters.dataset import ClusterLabeledDataset

# Load dataset
dataset = ClusterLabeledDataset(
    dataset_dir='datasets/my_dataset',
    label_type='combined'
)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Training loop
model = YourModel()
optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for base_emb, steering_emb, label in dataloader:
        base_emb = base_emb.to(device)
        steering_emb = steering_emb.to(device)
        label = label.to(device)
        
        # Forward pass
        output = model(base_emb, steering_emb)
        
        # Compute loss
        loss = criterion(output, label)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### With Augmentation Masking

Use `combined_label` with random masking:

```python
import random

for base_emb, steering_emb, combined_label in dataloader:
    # Randomly mask steering embedding
    mask_steering = random.random() < 0.3  # 30% masking
    
    if mask_steering:
        steering_emb = torch.zeros_like(steering_emb)
        # combined_label automatically handles zero steering
    
    output = model(base_emb, steering_emb)
    loss = criterion(output, combined_label)
    ...
```

### Multi-Task Learning

Use different label types for multi-task:

```python
# Load with metadata to access all label types
dataset = ClusterLabeledDataset(
    dataset_dir='datasets/my_dataset',
    label_type='combined',
    return_metadata=True
)

for base_emb, steering_emb, combined_label, metadata in dataloader:
    # Task 1: Source classification
    source_output = model.source_head(base_emb)
    source_loss = criterion(source_output, metadata['source_label'])
    
    # Task 2: Steering classification
    steering_output = model.steering_head(steering_emb)
    steering_loss = criterion(steering_output, metadata['steering_label'])
    
    # Task 3: Combined classification
    combined_output = model.combined_head(base_emb, steering_emb)
    combined_loss = criterion(combined_output, combined_label)
    
    # Total loss
    loss = source_loss + steering_loss + combined_loss
    ...
```

---

## Summary

**Key Advantages**:

1. **3 Label Types**: Flexible training strategies
2. **SQLite Storage**: Human-editable, supports corrections
3. **Memory-Mapped Embeddings**: Efficient for large datasets
4. **Augmentation Support**: Built-in noise and zero augmentations
5. **Multiple Steering Modes**: SUGGESTION, CLUSTER_DESCRIPTOR, LLM, ZERO, MIXED
6. **CSV Integration**: Direct support for `domain_assesment.py` outputs
7. **Clustering Integration**: Uses `keyword_clustering.py` centroids and descriptors

**Recommended Workflow**:
1. Run `domain_assessment.py` → Get CSV with suggestions + cluster probabilities
2. Run `keyword_clustering.py` → Get clustering JSON with centroids
3. Use `DomainAssessmentDatasetBuilder` → Build dataset
4. Use `ClusterLabeledDataset` → Train model
5. Manually correct labels if needed → Update SQLite
6. Retrain with corrected labels

**Next Steps**:
- See `tests/test_domain_assessment_dataset.py` for unit tests
- See `docs/DOMAIN_ASSESSMENT_ARCHITECTURE.md` for architecture details
