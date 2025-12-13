# Flexible Cluster Steering Dataset

## Quick Start

The `BaseDomainAssignDataset` now supports **cluster steering** with multiple steering embedding modes for advanced RAG and neural network training.

### Installation Requirements

```bash
pip install torch pandas numpy
# Plus your embedding model (e.g., langchain, sentence-transformers)
```

### Basic Usage

```python
from RAG_supporters.dataset import BaseDomainAssignDataset, SteeringMode
import pandas as pd

# Your data
df = pd.DataFrame({
    "source": ["Source text 1", "Source text 2"],
    "question": ["Question 1", "Question 2"],
    "suggestions": ['[{"term": "ml", "confidence": 0.9}]', ...]
})

# Cluster assignments (from KeywordClusterer or manual)
cluster_labels = {0: 0, 1: 1}  # sample_idx -> cluster_id

# Create dataset
dataset = BaseDomainAssignDataset(
    df=df,
    embedding_model=your_embedding_model,
    steering_mode=SteeringMode.ZERO,
    cluster_labels=cluster_labels,
    return_triplets=True
).build()

# Get a sample
sample = dataset[0]
# Returns: base_embedding, steering_embedding, target, metadata
```

## Steering Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `ZERO` | Zero vector baseline | Ablation studies, baseline comparison |
| `SUGGESTION` | First suggestion embedding | Keyword-guided steering |
| `CLUSTER_DESCRIPTOR` | Topic descriptor embedding | Topic-aware steering |
| `LLM_GENERATED` | LLM-generated context | Context-rich steering |
| `MIXED` | Weighted combination | Hybrid approaches |

## Examples

See `example_usage.py` for complete examples of each mode.

### Example 1: Zero Baseline (Ablation)

```python
dataset = BaseDomainAssignDataset(
    df=df,
    embedding_model=embeddings,
    steering_mode=SteeringMode.ZERO,
    cluster_labels={0: 0, 1: 1},
    return_triplets=True
)
```

### Example 2: Cluster Descriptor Steering

```python
cluster_descriptors = {
    0: ["machine learning", "AI"],
    1: ["neural networks", "deep learning"]
}

dataset = BaseDomainAssignDataset(
    df=df,
    embedding_model=embeddings,
    steering_mode=SteeringMode.CLUSTER_DESCRIPTOR,
    cluster_labels={0: 0, 1: 1},
    cluster_descriptors=cluster_descriptors,
    return_triplets=True
)
```

### Example 3: Soft Multi-Label

```python
# Probabilistic assignments
cluster_labels = {
    0: [0.8, 0.2],  # 80% cluster 0, 20% cluster 1
    1: [0.3, 0.7],  # 30% cluster 0, 70% cluster 1
}

dataset = BaseDomainAssignDataset(
    df=df,
    embedding_model=embeddings,
    steering_mode=SteeringMode.ZERO,
    cluster_labels=cluster_labels,
    multi_label_mode="soft",
    return_triplets=True
)
```

## Integration with Clustering

```python
from RAG_supporters.clustering import KeywordClusterer
from RAG_supporters.embeddings import KeywordEmbedder

# Phase 1: Cluster suggestions
embedder = KeywordEmbedder()
suggestion_embeddings = embedder.process_csv_to_embeddings(...)
clusterer = KeywordClusterer(n_clusters=5).fit(suggestion_embeddings)

# Phase 2: Assign sources
source_embeddings = embedder.create_embeddings(sources)
assignments = clusterer.assign_batch(source_embeddings, mode="soft")

# Phase 3: Create steering dataset
cluster_labels = {
    idx: result['probabilities'] 
    for idx, result in assignments.items()
}
cluster_descriptors = clusterer.extract_topic_descriptors(n_descriptors=10)

dataset = BaseDomainAssignDataset(
    df=df,
    embedding_model=embedder.embedding_model,
    steering_mode=SteeringMode.CLUSTER_DESCRIPTOR,
    cluster_labels=cluster_labels,
    cluster_descriptors=cluster_descriptors,
    multi_label_mode="soft",
    return_triplets=True,
    cache_dir="cache/steering_dataset"
).build()
```

## Training Usage

```python
from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in loader:
    base_emb = batch['base_embedding']       # [batch_size, dim]
    steering_emb = batch['steering_embedding']  # [batch_size, dim]
    targets = batch['target']                # [batch_size] or [batch_size, n_clusters]
    
    # Your model
    predictions = model(base_emb, steering_emb)
    loss = criterion(predictions, targets)
    # ... training step
```

## Sample Structure

### Triplet Mode
```python
{
    'base_embedding': Tensor[embedding_dim],      # Question embedding
    'steering_embedding': Tensor[embedding_dim],   # Steering signal
    'target': Tensor[n_clusters] or int,          # Cluster assignment
    'source_embedding': Tensor[embedding_dim],     # Source embedding
    'metadata': {
        'steering_mode': 'cluster_descriptor',
        'suggestion_texts': ['term1', 'term2'],
        'source_text': 'Source A',
        'question_text': 'What is ML?',
        'sample_index': 0,
        'cluster_assignment': 0 or [0.8, 0.2],
        'cluster_descriptors': ['ml', 'ai'],
        'llm_steering_text': '...' (optional)
    },
    'idx': 0
}
```

## Caching

```python
# Build and cache
dataset = BaseDomainAssignDataset(
    df=df,
    embedding_model=embeddings,
    cache_dir="cache/my_dataset",
    ...
).build(save_to_cache=True)

# Later: Load from cache
from RAG_supporters.dataset import CachedDomainAssignDataset

dataset = CachedDomainAssignDataset(
    cache_dir="cache/my_dataset",
    return_embeddings=True
)
```

## Testing

```bash
# Run tests (requires pytest and dependencies)
pytest RAG_supporters/dataset/test_torch_dataset.py -v
```

## Documentation

- **Complete Guide**: See `docs/CLUSTER_STEERING_DATASET.md`
- **API Reference**: See docstrings in `torch_dataset.py`
- **Clustering**: See `docs/CLUSTERING_AND_ASSIGNMENT.md`

## Key Features

✅ **5 Steering Modes**: Zero, Suggestion, Cluster Descriptor, LLM-Generated, Mixed  
✅ **Multi-Label**: Hard (one-hot) and Soft (probabilistic) targets  
✅ **Rich Metadata**: Full audit trail for each sample  
✅ **Cache Support**: Pre-compute and reuse embeddings  
✅ **Backward Compatible**: Standard mode unchanged  
✅ **Integration Ready**: Works with KeywordClusterer  

## Common Patterns

### Pattern 1: Ablation Study
```python
# Compare with/without steering
dataset_baseline = BaseDomainAssignDataset(..., steering_mode=SteeringMode.ZERO)
dataset_guided = BaseDomainAssignDataset(..., steering_mode=SteeringMode.CLUSTER_DESCRIPTOR)
```

### Pattern 2: Multi-Topic Questions
```python
# Use soft labels for ambiguous questions
cluster_labels = {0: [0.5, 0.5]}  # Equal membership in 2 topics
```

### Pattern 3: Hybrid Steering
```python
# Combine multiple steering signals
dataset = BaseDomainAssignDataset(
    ...,
    steering_mode=SteeringMode.MIXED,
    steering_weights={"suggestion": 0.6, "cluster_descriptor": 0.4}
)
```

## Troubleshooting

**Issue**: `ValueError: steering_mode must be specified`  
**Fix**: Set `steering_mode` when `return_triplets=True`

**Issue**: Missing steering embeddings  
**Fix**: Provide `cluster_descriptors` or `llm_steering_texts` for respective modes

**Issue**: Cluster label mismatch  
**Fix**: Ensure `cluster_labels` keys match sample indices (0 to len(df)-1)

## Contributing

When adding new steering modes or features:
1. Update `SteeringMode` enum
2. Implement in `_generate_steering_embedding()`
3. Add tests in `test_torch_dataset.py`
4. Update documentation

## License

Same as parent project.
