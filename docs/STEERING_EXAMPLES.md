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
