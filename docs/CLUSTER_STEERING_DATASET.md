# Flexible Cluster Steering Dataset

## Overview

The `BaseDomainAssignDataset` has been extended to support **flexible cluster steering** with multiple steering embedding modes. This enables the dataset to return triplets suitable for training models that use steering embeddings to guide predictions toward specific clusters or subspaces.

## Key Features

- **Multiple Steering Modes**: Support for suggestion-based, LLM-generated, cluster descriptor, zero baseline, and mixed steering embeddings
- **Triplet Output**: Returns (base_embedding, steering_embedding, target) for each sample
- **Multi-Label Targets**: Support for both hard (one-hot) and soft (probabilistic) cluster assignments
- **Rich Metadata**: Comprehensive metadata tracking for review, editing, and auditing
- **Backward Compatible**: Standard mode remains unchanged for existing workflows
- **Cache Support**: Full persistence of steering data and embeddings

## Steering Modes

### 1. SUGGESTION
Uses the first suggestion embedding as the steering signal.
```python
dataset = BaseDomainAssignDataset(
    df=df,
    embedding_model=embeddings,
    steering_mode=SteeringMode.SUGGESTION,
    cluster_labels=cluster_assignments,
    return_triplets=True
)
```

### 2. LLM_GENERATED
Uses LLM-generated steering text embeddings.
```python
llm_texts = {
    0: "Focus on machine learning fundamentals",
    1: "Emphasize deep learning architectures",
    # ... one per sample
}

dataset = BaseDomainAssignDataset(
    df=df,
    embedding_model=embeddings,
    steering_mode=SteeringMode.LLM_GENERATED,
    llm_steering_texts=llm_texts,
    cluster_labels=cluster_assignments,
    return_triplets=True
)
```

### 3. CLUSTER_DESCRIPTOR
Uses cluster/topic descriptor embeddings as steering.
```python
cluster_descriptors = {
    0: ["machine learning", "artificial intelligence"],
    1: ["neural networks", "deep learning"],
    # ... one per cluster
}

dataset = BaseDomainAssignDataset(
    df=df,
    embedding_model=embeddings,
    steering_mode=SteeringMode.CLUSTER_DESCRIPTOR,
    cluster_labels=cluster_assignments,
    cluster_descriptors=cluster_descriptors,
    return_triplets=True
)
```

### 4. ZERO
Zero baseline for ablation studies.
```python
dataset = BaseDomainAssignDataset(
    df=df,
    embedding_model=embeddings,
    steering_mode=SteeringMode.ZERO,
    cluster_labels=cluster_assignments,
    return_triplets=True
)
```

### 5. MIXED
Weighted combination of multiple steering modes.
```python
dataset = BaseDomainAssignDataset(
    df=df,
    embedding_model=embeddings,
    steering_mode=SteeringMode.MIXED,
    steering_weights={
        "suggestion": 0.6,
        "cluster_descriptor": 0.4
    },
    cluster_labels=cluster_assignments,
    cluster_descriptors=cluster_descriptors,
    return_triplets=True
)
```

## Multi-Label Targets

### Hard Assignment (One-Hot)
Each sample assigned to exactly one cluster.
```python
cluster_labels = {
    0: 0,  # Sample 0 -> Cluster 0
    1: 1,  # Sample 1 -> Cluster 1
    2: 0,  # Sample 2 -> Cluster 0
    # ...
}

dataset = BaseDomainAssignDataset(
    df=df,
    embedding_model=embeddings,
    steering_mode=SteeringMode.ZERO,
    cluster_labels=cluster_labels,
    multi_label_mode="hard",  # Default
    return_triplets=True
)

# Returns target as integer: 0, 1, 2, etc.
sample = dataset[0]
print(sample['target'])  # tensor(0, dtype=torch.long)
```

### Soft Assignment (Probabilistic)
Each sample can belong to multiple clusters with probabilities.
```python
cluster_labels = {
    0: [0.8, 0.2],      # Sample 0: 80% cluster 0, 20% cluster 1
    1: [0.3, 0.7],      # Sample 1: 30% cluster 0, 70% cluster 1
    2: [0.5, 0.5],      # Sample 2: equal membership
    # ...
}

dataset = BaseDomainAssignDataset(
    df=df,
    embedding_model=embeddings,
    steering_mode=SteeringMode.ZERO,
    cluster_labels=cluster_labels,
    multi_label_mode="soft",
    return_triplets=True
)

# Returns target as probability distribution
sample = dataset[0]
print(sample['target'])  # tensor([0.8, 0.2], dtype=torch.float32)
```

## Sample Structure

### Standard Mode (Backward Compatible)
```python
dataset = BaseDomainAssignDataset(
    df=df,
    embedding_model=embeddings,
    return_embeddings=True
)

sample = dataset[0]
# {
#     'source': Tensor[embedding_dim],
#     'question': Tensor[embedding_dim],
#     'suggestions': [Tensor[embedding_dim], ...],
#     'suggestion_texts': ['term1', 'term2', ...],
#     'idx': 0
# }
```

### Triplet Mode (Cluster Steering)
```python
dataset = BaseDomainAssignDataset(
    df=df,
    embedding_model=embeddings,
    steering_mode=SteeringMode.CLUSTER_DESCRIPTOR,
    cluster_labels=cluster_assignments,
    cluster_descriptors=cluster_descriptors,
    return_triplets=True
)

sample = dataset[0]
# {
#     'base_embedding': Tensor[embedding_dim],      # Question embedding
#     'steering_embedding': Tensor[embedding_dim],   # Steering signal
#     'target': Tensor[n_clusters] or int,          # Cluster assignment
#     'source_embedding': Tensor[embedding_dim],     # Source embedding
#     'metadata': {
#         'steering_mode': 'cluster_descriptor',
#         'suggestion_texts': ['term1', 'term2', ...],
#         'source_text': 'Source A',
#         'question_text': 'What is machine learning?',
#         'sample_index': 0,
#         'cluster_assignment': 0 or [0.8, 0.2],
#         'cluster_descriptors': ['term1', 'term2', ...],
#         'llm_steering_text': '...' (if applicable)
#     },
#     'idx': 0
# }
```

## Complete Workflow Example

### Step 1: Prepare Data
```python
import pandas as pd
from RAG_supporters.dataset import BaseDomainAssignDataset, SteeringMode
from RAG_supporters.clustering import KeywordClusterer

# Load your data
df = pd.read_csv("data/questions.csv")

# Load or create cluster assignments
clusterer = KeywordClusterer.from_results("results/keyword_clusters.json")
# Assign samples to clusters (Phase 2 from CLUSTERING_AND_ASSIGNMENT.md)
```

### Step 2: Create Dataset with Steering
```python
# Option A: Use cluster descriptors
cluster_descriptors = {
    0: ["machine learning", "AI"],
    1: ["neural networks", "deep learning"],
    2: ["NLP", "language models"]
}

dataset = BaseDomainAssignDataset(
    df=df,
    embedding_model=embeddings,
    steering_mode=SteeringMode.CLUSTER_DESCRIPTOR,
    cluster_labels=cluster_assignments,
    cluster_descriptors=cluster_descriptors,
    multi_label_mode="soft",
    return_triplets=True,
    cache_dir="cache/steering_dataset"
).build()
```

### Step 3: Use in Training
```python
from torch.utils.data import DataLoader

# Create DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
for batch in loader:
    base_emb = batch['base_embedding']       # [batch_size, embedding_dim]
    steering_emb = batch['steering_embedding']  # [batch_size, embedding_dim]
    targets = batch['target']                # [batch_size] or [batch_size, n_clusters]
    
    # Your model forward pass
    predictions = model(base_emb, steering_emb)
    loss = criterion(predictions, targets)
    # ...
```

### Step 4: Load from Cache
```python
from RAG_supporters.dataset import CachedDomainAssignDataset

# Later sessions can load pre-computed dataset
dataset = CachedDomainAssignDataset(
    cache_dir="cache/steering_dataset",
    return_embeddings=True
)

# Works identically to the built dataset
sample = dataset[0]
```

## Use Cases

### 1. Multi-Topic Questions
For questions that span multiple topics, use soft assignments:
```python
# Question could be about both ML and NLP
cluster_labels = {
    0: [0.6, 0.4],  # 60% ML, 40% NLP
}
```

### 2. RL Training
Use different steering modes to explore policy space:
```python
# Zero baseline
dataset_zero = BaseDomainAssignDataset(..., steering_mode=SteeringMode.ZERO)

# With guidance
dataset_guided = BaseDomainAssignDataset(..., steering_mode=SteeringMode.CLUSTER_DESCRIPTOR)
```

### 3. LLM-Guided Steering
Generate context-specific steering:
```python
llm_texts = {}
for idx, row in df.iterrows():
    question = row['question']
    llm_texts[idx] = llm.generate_steering_prompt(question)

dataset = BaseDomainAssignDataset(
    ...,
    steering_mode=SteeringMode.LLM_GENERATED,
    llm_steering_texts=llm_texts
)
```

### 4. Ablation Studies
Compare model performance with different steering:
```python
results = {}
for mode in [SteeringMode.ZERO, SteeringMode.SUGGESTION, SteeringMode.CLUSTER_DESCRIPTOR]:
    dataset = BaseDomainAssignDataset(..., steering_mode=mode)
    results[mode.value] = evaluate_model(dataset)
```

## API Reference

### BaseDomainAssignDataset

**New Parameters:**
- `steering_mode` (Optional[Union[SteeringMode, str]]): Steering embedding mode
- `cluster_labels` (Optional[Dict[int, Union[int, List[float]]]]): Cluster assignments
- `cluster_descriptors` (Optional[Dict[int, List[str]]]): Cluster descriptor texts
- `llm_steering_texts` (Optional[Dict[int, str]]): LLM-generated steering texts
- `return_triplets` (bool): Return triplet format instead of standard
- `multi_label_mode` (str): "hard" or "soft" for target format
- `steering_weights` (Optional[Dict[str, float]]): Weights for mixed mode

**Key Methods:**
- `build(save_to_cache=True)`: Build and optionally cache dataset
- `__getitem__(idx)`: Get sample in triplet or standard format
- `__len__()`: Get dataset length

### CachedDomainAssignDataset

Automatically detects and loads steering data from cache. No additional parameters needed.

## Integration with Clustering

This dataset integrates seamlessly with the clustering module:

```python
from RAG_supporters.embeddings import KeywordEmbedder
from RAG_supporters.clustering import KeywordClusterer

# Phase 1: Cluster suggestions
embedder = KeywordEmbedder()
suggestion_embeddings = embedder.process_csv_to_embeddings(...)
clusterer = KeywordClusterer(n_clusters=5)
clusterer.fit(suggestion_embeddings)

# Phase 2: Assign sources
source_embeddings = embedder.create_embeddings(sources)
assignments = clusterer.assign_batch(source_embeddings, mode="soft")

# Phase 3: Create steering dataset
cluster_labels = {idx: result['probabilities'] for idx, result in assignments.items()}
cluster_descriptors = clusterer.extract_topic_descriptors(n_descriptors=10)

dataset = BaseDomainAssignDataset(
    df=df,
    embedding_model=embedder.embedding_model,
    steering_mode=SteeringMode.CLUSTER_DESCRIPTOR,
    cluster_labels=cluster_labels,
    cluster_descriptors=cluster_descriptors,
    multi_label_mode="soft",
    return_triplets=True
)
```

## Best Practices

1. **Cache Everything**: Build and cache datasets to avoid recomputing embeddings
2. **Match Modes**: Use same embedding model for clustering and dataset
3. **Validate Assignments**: Check that cluster_labels cover all samples
4. **Start Simple**: Begin with ZERO or SUGGESTION modes before complex steering
5. **Monitor Metadata**: Use metadata for debugging and analysis
6. **Soft Labels**: Use soft assignments for ambiguous or multi-topic samples

## Troubleshooting

**Issue: ValueError about steering_mode**
- Solution: Must set `steering_mode` when `return_triplets=True`

**Issue: Missing steering embeddings**
- Solution: Ensure cluster_descriptors or llm_steering_texts are provided for respective modes

**Issue: Cluster label mismatch**
- Solution: Check that cluster_labels keys match sample indices (0 to len(df)-1)

**Issue: Cache loading errors**
- Solution: Rebuild cache if dataset format changed

## See Also

- `CLUSTERING_AND_ASSIGNMENT.md`: Complete guide to clustering and assignment
- `KeywordClusterer`: Clustering and assignment functionality
- `KeywordEmbedder`: Embedding creation
# Steering Dataset Examples

This document provides practical examples of using the `BaseDomainAssignDataset` with different steering modes.

## Prerequisites

```bash
pip install pandas numpy torch
# Plus your embedding model (e.g., langchain, sentence-transformers)
```

## Example 1: Zero Baseline Steering

Zero baseline is useful for ablation studies to compare model performance with and without steering.

```python
from RAG_supporters.dataset import BaseDomainAssignDataset, SteeringMode
import pandas as pd

# Sample data
df = pd.DataFrame({
    "source": ["Source A", "Source B", "Source C"],
    "question": [
        "What is machine learning?",
        "How does deep learning work?",
        "Explain neural networks"
    ],
    "suggestions": [
        '[{"term": "ml", "confidence": 0.9, "type": "keyword"}]',
        '[{"term": "dl", "confidence": 0.8, "type": "keyword"}]',
        '[{"term": "nn", "confidence": 0.85, "type": "keyword"}]',
    ],
})

# Hard cluster assignments
cluster_labels = {0: 0, 1: 0, 2: 1}

# Create dataset with zero steering
dataset = BaseDomainAssignDataset(
    df=df,
    embedding_model=your_embedding_model,
    steering_mode=SteeringMode.ZERO,
    cluster_labels=cluster_labels,
    return_triplets=True,
)
dataset.build(save_to_cache=False)

# Get a sample
sample = dataset[0]
print(f"Base embedding shape: {sample['base_embedding'].shape}")
print(f"Steering is zeros: {(sample['steering_embedding'] == 0).all()}")
```

## Example 2: Cluster Descriptor Steering

Use cluster descriptors to guide the model toward specific topics.

```python
# Cluster assignments and descriptors
cluster_labels = {0: 0, 1: 0, 2: 1}
cluster_descriptors = {
    0: ["machine learning", "artificial intelligence"],
    1: ["neural networks", "deep learning"]
}

dataset = BaseDomainAssignDataset(
    df=df,
    embedding_model=your_embedding_model,
    steering_mode=SteeringMode.CLUSTER_DESCRIPTOR,
    cluster_labels=cluster_labels,
    cluster_descriptors=cluster_descriptors,
    return_triplets=True,
)
dataset.build()

# Examine results
for i in range(len(dataset)):
    sample = dataset[i]
    print(f"Sample {i}:")
    print(f"  Cluster: {sample['target'].item()}")
    print(f"  Descriptors: {sample['metadata']['cluster_descriptors']}")
```

## Example 3: Soft Multi-Label Assignments

Handle ambiguous questions that span multiple topics.

```python
# Soft cluster assignments (probabilities)
cluster_labels = {
    0: [0.8, 0.2],  # 80% cluster 0, 20% cluster 1
    1: [0.7, 0.3],  # 70% cluster 0, 30% cluster 1
    2: [0.3, 0.7],  # 30% cluster 0, 70% cluster 1
}

dataset = BaseDomainAssignDataset(
    df=df,
    embedding_model=your_embedding_model,
    steering_mode=SteeringMode.ZERO,
    cluster_labels=cluster_labels,
    multi_label_mode="soft",
    return_triplets=True,
)
dataset.build()

# Check soft targets
for i in range(len(dataset)):
    sample = dataset[i]
    print(f"Sample {i} target (soft): {sample['target'].numpy()}")
    print(f"  Sum: {sample['target'].sum().item():.4f}")
```

## Example 4: LLM-Generated Steering

Use LLM-generated context to steer predictions.

```python
# LLM-generated steering texts
llm_steering_texts = {
    0: "Focus on fundamental ML concepts and algorithms",
    1: "Emphasize deep learning architectures and training"
}

cluster_labels = {0: 0, 1: 1}

dataset = BaseDomainAssignDataset(
    df=df,
    embedding_model=your_embedding_model,
    steering_mode=SteeringMode.LLM_GENERATED,
    llm_steering_texts=llm_steering_texts,
    cluster_labels=cluster_labels,
    return_triplets=True,
)
dataset.build()

# View LLM steering
for i in range(len(dataset)):
    sample = dataset[i]
    print(f"Sample {i}:")
    print(f"  LLM Steering: {sample['metadata']['llm_steering_text']}")
```

## Example 5: Mixed Steering with Weights

Combine multiple steering signals with custom weights.

```python
cluster_labels = {0: 0, 1: 1}
cluster_descriptors = {
    0: ["machine learning", "AI"],
    1: ["deep learning", "neural nets"]
}

dataset = BaseDomainAssignDataset(
    df=df,
    embedding_model=your_embedding_model,
    steering_mode=SteeringMode.MIXED,
    steering_weights={
        "suggestion": 0.6,
        "cluster_descriptor": 0.4
    },
    cluster_labels=cluster_labels,
    cluster_descriptors=cluster_descriptors,
    return_triplets=True,
)
dataset.build()

sample = dataset[0]
print(f"Steering combines suggestion (60%) + cluster desc (40%)")
print(f"Steering shape: {sample['steering_embedding'].shape}")
```

## Example 6: Random Steering Mode Selection

Allow the dataset to randomly select steering modes per sample. This is useful for data augmentation and exploring different steering strategies.

```python
# Define multiple steering modes with probabilities
steering_modes = [
    (SteeringMode.SUGGESTION, 0.4),
    (SteeringMode.CLUSTER_DESCRIPTOR, 0.4),
    (SteeringMode.ZERO, 0.2)
]

# Note: Probabilities will be automatically normalized if they don't sum to 1.0
# For example, [(mode1, 0.3), (mode2, 0.5)] will be normalized to [(mode1, 0.375), (mode2, 0.625)]

dataset = BaseDomainAssignDataset(
    df=df,
    embedding_model=your_embedding_model,
    steering_mode=steering_modes,  # Pass list of (mode, probability)
    cluster_labels=cluster_labels,
    cluster_descriptors=cluster_descriptors,
    return_triplets=True,
)
dataset.build()

# Each sample will use a randomly selected mode based on probabilities
# Selection is deterministic per sample index for reproducibility
```

## Example 7: Loading from Clustering JSON

Load cluster information directly from KeywordClusterer output.

```python
# Assuming you have clustering results from KeywordClusterer
clustering_results_path = "results/keyword_clusters.json"

dataset = BaseDomainAssignDataset(
    df=df,
    embedding_model=your_embedding_model,
    steering_mode=SteeringMode.CLUSTER_DESCRIPTOR,
    clustering_results_path=clustering_results_path,  # Automatically loads descriptors
    cluster_labels=cluster_labels,
    return_triplets=True,
)
dataset.build()
```

## Example 8: Sample Weighting for Class Balancing

Balance training across clusters using automatic or manual weights.

```python
# Automatic balancing
dataset = BaseDomainAssignDataset(
    df=df,
    embedding_model=your_embedding_model,
    steering_mode=SteeringMode.CLUSTER_DESCRIPTOR,
    cluster_labels=cluster_labels,
    cluster_descriptors=cluster_descriptors,
    balance_clusters=True,  # Automatically compute balanced weights
    return_triplets=True,
)
dataset.build()

# Or provide manual weights
sample_weights = {0: 1.5, 1: 1.0, 2: 2.0}  # Higher weight for rare samples
dataset = BaseDomainAssignDataset(
    df=df,
    embedding_model=your_embedding_model,
    sample_weights=sample_weights,
    # ... other params
)
```

## Example 9: Chunked CSV Reading for Large Files

Avoid memory issues with large CSV files.

```python
dataset = BaseDomainAssignDataset(
    df="large_dataset.csv",
    embedding_model=your_embedding_model,
    read_csv_chunksize=10000,  # Read 10k rows at a time
    # ... other params
)
```

## Training Loop Example

```python
from torch.utils.data import DataLoader

# Create dataset
dataset = BaseDomainAssignDataset(
    df=df,
    embedding_model=embeddings,
    steering_mode=SteeringMode.CLUSTER_DESCRIPTOR,
    cluster_labels=cluster_labels,
    cluster_descriptors=cluster_descriptors,
    multi_label_mode="soft",
    balance_clusters=True,
    return_triplets=True,
)
dataset.build()

# Report statistics
dataset.report_statistics()

# Create DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    for batch in loader:
        base_emb = batch['base_embedding']
        steering_emb = batch['steering_embedding']
        targets = batch['target']
        weights = batch.get('weight', None)  # Optional
        
        # Forward pass
        predictions = model(base_emb, steering_emb)
        
        # Loss with optional weighting
        if weights is not None:
            loss = (criterion(predictions, targets) * weights).mean()
        else:
            loss = criterion(predictions, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## See Also

- **Complete Guide**: `docs/CLUSTER_STEERING_DATASET.md`
- **Quick Reference**: `RAG_supporters/dataset/README_CLUSTER_STEERING.md`
- **Clustering Guide**: `docs/CLUSTERING_AND_ASSIGNMENT.md`
