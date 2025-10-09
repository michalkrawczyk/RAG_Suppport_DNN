# Sample Generation Quick Reference

Quick reference guide for generating training samples in RAG datasets.

## Quick Start

```python
from RAG_supporters.dataset.templates.rag_mini_bioasq import RagMiniBioASQBase

# Initialize dataset
dataset = RagMiniBioASQBase(
    dataset_dir="./data/bioasq",
    embed_function=your_embedding_function
)

# Generate samples
samples = dataset.generate_samples("pairs_relevant")
```

## Sample Types at a Glance

### Pair Samples

| Type | Command | Use Case | Memory |
|------|---------|----------|--------|
| **Relevant** | `"pairs_relevant"` | Known relevant question-passage pairs | Low |
| **All Existing** | `"pairs_all_existing"` | All possible question-passage combinations | High* |
| **Embedding Similarity** | `"pairs_embedding_similarity"` | Top-k similar passages by embedding | Low |

*Optimized with batch processing

### Triplet Samples

| Type | Command | Use Case | Memory |
|------|---------|----------|--------|
| **Positive** | `"positive"` | Two relevant passages per question | Low |
| **Contrastive** | `"contrastive"` | One relevant + one non-relevant passage | Medium |
| **Similar** | `"similar"` | One relevant + one similar passage | Low |

## Common Parameters

### Pair Samples

```python
# Relevant pairs (no additional parameters)
dataset.generate_samples("pairs_relevant")

# All existing pairs
dataset.generate_samples("pairs_all_existing", 
    batch_size=100  # Batch size for memory optimization
)

# Embedding similarity pairs
dataset.generate_samples("pairs_embedding_similarity",
    top_k=3  # Number of similar passages to retrieve
)
```

### Triplet Samples

```python
# Positive triplets (no additional parameters)
dataset.generate_samples("positive")

# Contrastive triplets
dataset.generate_samples("contrastive",
    num_negative_samples=5,      # Number of negative samples
    keep_same_negatives=False,   # Reuse same negatives
    assume_relevant_best=True    # Label relevant as better
)

# Similar triplets
dataset.generate_samples("similar",
    score_threshold=0.3,         # Distance threshold
    top_k=3,                     # Number of similar passages
    assume_relevant_best=True    # Label relevant as better
)
```

## Memory Optimization

### Batch Size Guidelines

```python
# Low RAM (< 8 GB)
dataset.generate_samples("pairs_all_existing", batch_size=25)

# Medium RAM (8-16 GB) - Default
dataset.generate_samples("pairs_all_existing", batch_size=100)

# High RAM (16-32 GB)
dataset.generate_samples("pairs_all_existing", batch_size=200)

# Very High RAM (> 32 GB)
dataset.generate_samples("pairs_all_existing", batch_size=500)
```

## Output Formats

### Pair Samples
Returns: `pandas.DataFrame`

Columns:
- `question_id`: Question identifier
- `question_text`: Question text
- `source_id`: Source passage identifier
- `source_text`: Source passage text
- `answer`: Expected answer (optional)

### Triplet Samples
Returns: `List[SampleTripletRAGChroma]`

Attributes:
- `question_id`: Question identifier
- `source_id_1`: First source passage identifier
- `source_id_2`: Second source passage identifier
- `label`: Comparison label (-1, 0, 1, or 2)

## CSV Output

```python
# Save to CSV (default for pairs)
pairs = dataset.generate_samples("pairs_relevant", save_to_csv=True)
# Output: {dataset_dir}/pairs_pairs_relevant.csv

# Disable CSV saving
pairs = dataset.generate_samples("pairs_relevant", save_to_csv=False)

# Triplets are always saved to CSV
triplets = dataset.generate_samples("contrastive")
# Output: {dataset_dir}/triplets_contrastive.csv
```

## Label Interpretation

### Triplet Labels

- `-1`: Unlabeled / No preference
- `0`: Both passages are irrelevant
- `1`: First passage (source_id_1) is better
- `2`: Second passage (source_id_2) is better

### Default Labels by Sample Type

| Sample Type | Default Label | Meaning |
|-------------|---------------|---------|
| Positive | -1 | Both passages equally relevant |
| Contrastive (assume_relevant_best=True) | 1 | Relevant passage is better |
| Contrastive (assume_relevant_best=False) | -1 | Unlabeled |
| Similar (assume_relevant_best=True) | 1 | Relevant passage is better |
| Similar (assume_relevant_best=False) | -1 | Unlabeled |

## Example Workflows

### Training Data Preparation

```python
# 1. Generate relevant pairs for basic training
relevant_pairs = dataset.generate_samples("pairs_relevant")

# 2. Generate contrastive triplets for discrimination learning
contrastive = dataset.generate_samples("contrastive", num_negative_samples=10)

# 3. Generate similar triplets for fine-grained ranking
similar = dataset.generate_samples("similar", top_k=5, score_threshold=0.3)
```

### Comprehensive Evaluation Dataset

```python
# Generate all possible pairs (use with caution on large datasets)
all_pairs = dataset.generate_samples("pairs_all_existing", batch_size=50)

# Or use embedding similarity for a manageable subset
similar_pairs = dataset.generate_samples("pairs_embedding_similarity", top_k=20)
```

### Memory-Constrained Environment

```python
# Use smaller batches and fewer samples
dataset = RagMiniBioASQBase(
    dataset_dir="./data/bioasq",
    embed_function=your_embedding_function,
    loading_batch_size=50  # Set default batch size
)

pairs = dataset.generate_samples("pairs_all_existing", batch_size=25)
triplets = dataset.generate_samples("contrastive", num_negative_samples=3)
```

## Common Patterns

### Checking Generated Samples

```python
# Check pair samples
pairs = dataset.generate_samples("pairs_relevant")
print(f"Total pairs: {len(pairs)}")
print(f"Columns: {pairs.columns.tolist()}")
print(pairs.head())

# Check triplet samples
triplets = dataset.generate_samples("contrastive")
print(f"Total triplets: {len(triplets)}")
print(f"First triplet: {triplets[0]}")
```

### Combining Multiple Sample Types

```python
# Generate multiple types for comprehensive training
samples = {
    'relevant': dataset.generate_samples("pairs_relevant"),
    'contrastive': dataset.generate_samples("contrastive", num_negative_samples=5),
    'similar': dataset.generate_samples("similar", top_k=3)
}

print(f"Relevant pairs: {len(samples['relevant'])}")
print(f"Contrastive triplets: {len(samples['contrastive'])}")
print(f"Similar triplets: {len(samples['similar'])}")
```

### Loading Pre-Generated Samples

```python
import pandas as pd

# Load pair samples from CSV
pairs_df = pd.read_csv("./data/bioasq/pairs_pairs_relevant.csv")

# Load triplet samples from CSV
triplets_df = pd.read_csv("./data/bioasq/triplets_contrastive.csv")
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory | Reduce `batch_size` or use smaller `num_negative_samples` |
| Empty results | Check question metadata for relevant passages |
| Slow generation | Increase `batch_size` or reduce sample complexity |
| Large CSV files | Use `save_to_csv=False` and process in chunks |

## See Also

- [Complete Sample Generation Guide](SAMPLE_GENERATION_GUIDE.md) - Detailed documentation with examples
- `rag_dataset.py` - Base class API reference
- `rag_mini_bioasq.py` - BioASQ-specific implementation

---

For detailed explanations and advanced usage, see [SAMPLE_GENERATION_GUIDE.md](SAMPLE_GENERATION_GUIDE.md)
