# Sample Generation Guide

This guide provides comprehensive documentation for generating training samples (pairs and triplets) for RAG (Retrieval-Augmented Generation) models.

## Table of Contents

- [Overview](#overview)
- [Sample Types](#sample-types)
  - [Pair Samples](#pair-samples)
  - [Triplet Samples](#triplet-samples)
- [Usage Examples](#usage-examples)
- [Advanced Configuration](#advanced-configuration)
- [Memory Optimization](#memory-optimization)
- [Best Practices](#best-practices)

## Overview

The RAG dataset generator provides methods to create training samples that help models learn to:
- Rank passages by relevance to a question
- Distinguish between relevant and irrelevant content
- Compare similar passages and identify the most relevant one

### Sample Data Structures

**Pair Sample**: Question paired with a source passage
```python
{
    "question_id": str,      # Unique identifier for the question
    "question_text": str,    # The question text
    "source_id": str,        # Unique identifier for the source passage
    "source_text": str,      # The source passage text
    "answer": str            # Optional: Expected answer to the question
}
```

**Triplet Sample**: Question with two source passages for comparison
```python
SampleTripletRAGChroma(
    question_id: str,        # Unique identifier for the question
    source_id_1: str,        # First source passage ID
    source_id_2: str,        # Second source passage ID
    label: int               # -1=unlabeled, 0=both irrelevant, 1=first better, 2=second better
)
```

## Sample Types

### Pair Samples

Pair samples consist of questions paired with source passages. Three pairing strategies are available:

#### 1. Relevant Pairs (`pairs_relevant`)

Pairs questions with their known relevant passages from the dataset.

**Use Case**: Training models to identify relevant content when ground truth is available.

**Example**:
```python
# Generate relevant pairs
pairs_df = dataset.generate_samples("pairs_relevant")

# Result: DataFrame with question-source pairs where sources are known to be relevant
```

**Output**: Returns a pandas DataFrame with columns: `question_id`, `question_text`, `source_id`, `source_text`, `answer`

---

#### 2. All Existing Pairs (`pairs_all_existing`)

Pairs each question with ALL passages in the text corpus.

**Use Case**: Creating comprehensive datasets for exhaustive evaluation or when you need all possible combinations.

**Warning**: This generates `num_questions × num_passages` pairs, which can be very large!

**Example**:
```python
# Generate all possible pairs (memory optimized with batching)
pairs_df = dataset.generate_samples("pairs_all_existing")

# With custom batch size for memory-constrained environments
pairs_df = dataset.generate_samples("pairs_all_existing", batch_size=50)
```

**Memory Optimization**: Uses batch processing to avoid loading all passages into memory at once. See [Memory Optimization](#memory-optimization) section.

**Output**: Returns a pandas DataFrame with all question-passage combinations.

---

#### 3. Embedding Similarity Pairs (`pairs_embedding_similarity`)

Pairs questions with passages that are most similar in embedding space.

**Use Case**: Finding passages that are semantically related to questions based on vector similarity.

**Example**:
```python
# Generate pairs based on top-3 most similar passages
pairs_df = dataset.generate_samples("pairs_embedding_similarity", top_k=3)

# Increase to top-10 for more comprehensive coverage
pairs_df = dataset.generate_samples("pairs_embedding_similarity", top_k=10)
```

**Parameters**:
- `top_k` (int, default=3): Number of most similar passages to retrieve per question

**Output**: Returns a pandas DataFrame with question-passage pairs ranked by embedding similarity.

---

### Triplet Samples

Triplet samples consist of a question and two passages for comparison learning.

#### 1. Positive Triplets (`positive`)

Pairs a question with two relevant passages from the ground truth data.

**Use Case**: Training models to recognize that multiple passages can be equally relevant to the same question.

**Example**:
```python
# Generate positive triplets
triplets = dataset.generate_samples("positive")
```

**Characteristics**:
- Both passages are relevant to the question
- Label is set to `-1` (no preference between passages)
- Generates all combinations of relevant passage pairs for each question

**Output**: Returns a list of `SampleTripletRAGChroma` objects saved to CSV.

---

#### 2. Contrastive Triplets (`contrastive`)

Pairs a question with one relevant and one non-relevant passage.

**Use Case**: Training models to distinguish between relevant and irrelevant content.

**Example**:
```python
# Generate contrastive triplets with default settings
triplets = dataset.generate_samples("contrastive")

# Generate with custom negative samples
triplets = dataset.generate_samples(
    "contrastive",
    num_negative_samples=10,           # Number of negative passages per question
    keep_same_negatives=True,          # Reuse same negatives for all relevant passages
    assume_relevant_best=True          # Label relevant passage as better (label=1)
)
```

**Parameters**:
- `num_negative_samples` (int, default=5): Number of random negative passages to pair with each relevant passage
- `keep_same_negatives` (bool, default=False): If True, uses the same negative passages for all relevant passages of a question
- `assume_relevant_best` (bool, default=True): If True, sets label=1 (first passage better); if False, label=-1 (unlabeled)

**Output**: Returns a list of `SampleTripletRAGChroma` objects with contrastive examples.

---

#### 3. Similar Triplets (`similar`)

Pairs a question with one relevant passage and one passage that is similar but not marked as relevant.

**Use Case**: Creating challenging training examples where the model must distinguish between truly relevant and merely similar content.

**Example**:
```python
# Generate similar triplets
triplets = dataset.generate_samples("similar")

# Adjust similarity threshold
triplets = dataset.generate_samples(
    "similar",
    score_threshold=0.3,               # Distance threshold for "similar" passages
    top_k=5,                           # Number of similar passages to retrieve
    assume_relevant_best=True          # Label relevant passage as better
)
```

**Parameters**:
- `score_threshold` (float, default=0.3): Maximum distance to consider a passage as "similar"
- `top_k` (int, default=3): Number of similar passages to retrieve from embedding space
- `assume_relevant_best` (bool, default=True): If True, sets label=1; if False, label=-1

**Output**: Returns a list of `SampleTripletRAGChroma` objects with challenging examples.

---

## Usage Examples

### Basic Usage

```python
from RAG_supporters.dataset.templates.rag_mini_bioasq import RagMiniBioASQBase

# Initialize the dataset
dataset = RagMiniBioASQBase(
    dataset_dir="./data/bioasq",
    embed_function=your_embedding_function
)

# Generate different sample types
relevant_pairs = dataset.generate_samples("pairs_relevant")
contrastive_triplets = dataset.generate_samples("contrastive")
```

### Saving to CSV

All sample generation methods support automatic CSV saving:

```python
# Save pairs to CSV (default behavior)
pairs_df = dataset.generate_samples("pairs_relevant", save_to_csv=True)
# Output: ./data/bioasq/pairs_pairs_relevant.csv

# Disable CSV saving
pairs_df = dataset.generate_samples("pairs_relevant", save_to_csv=False)

# For triplets, CSV is automatically saved
triplets = dataset.generate_samples("contrastive")
# Output: ./data/bioasq/triplets_contrastive.csv
```

### Combining Multiple Sample Types

```python
# Generate multiple types for comprehensive training
relevant_pairs = dataset.generate_samples("pairs_relevant")
similar_triplets = dataset.generate_samples("similar", top_k=5)
contrastive_triplets = dataset.generate_samples("contrastive", num_negative_samples=10)

# Use the samples for training your model
# ...
```

## Advanced Configuration

### Customizing Batch Processing

For `pairs_all_existing`, you can control memory usage:

```python
# Small batch for very limited RAM (slower but uses less memory)
pairs = dataset.generate_samples("pairs_all_existing", batch_size=25)

# Large batch for faster processing (more memory required)
pairs = dataset.generate_samples("pairs_all_existing", batch_size=200)

# Default batch size (100) - good balance
pairs = dataset.generate_samples("pairs_all_existing")
```

### Loading Batch Size Configuration

Set the default batch size when initializing the dataset:

```python
dataset = RagMiniBioASQBase(
    dataset_dir="./data/bioasq",
    embed_function=your_embedding_function,
    loading_batch_size=150  # Default batch size for various operations
)
```

## Memory Optimization

### Understanding Memory Usage

Different sample types have different memory footprints:

| Sample Type | Memory Impact | Recommendation |
|-------------|---------------|----------------|
| `pairs_relevant` | Low | Safe for all dataset sizes |
| `pairs_embedding_similarity` | Low-Medium | Safe, limited by `top_k` |
| `pairs_all_existing` | High | Use batch processing |
| `positive` | Low | Limited by available relevant passages |
| `contrastive` | Medium | Controlled by `num_negative_samples` |
| `similar` | Low-Medium | Limited by `top_k` and `score_threshold` |

### Optimization for `pairs_all_existing`

The `pairs_all_existing` method is optimized for large datasets:

1. **Batch Processing**: Text corpus is loaded in configurable batches
2. **Generator Pattern**: Uses a generator to yield pairs incrementally
3. **Configurable Batch Size**: Adjust based on available RAM

**Example with different dataset sizes**:

```python
# Small dataset (< 1000 passages): Use larger batches
pairs = dataset.generate_samples("pairs_all_existing", batch_size=200)

# Medium dataset (1,000 - 10,000 passages): Use default
pairs = dataset.generate_samples("pairs_all_existing")  # batch_size=100

# Large dataset (> 10,000 passages): Use smaller batches
pairs = dataset.generate_samples("pairs_all_existing", batch_size=50)

# Very large dataset or limited RAM: Use minimal batches
pairs = dataset.generate_samples("pairs_all_existing", batch_size=25)
```

### Memory Estimation

For `pairs_all_existing`:
- **Number of pairs**: `num_questions × num_passages`
- **Memory per batch**: ~`batch_size × average_passage_size`
- **Peak memory**: Batch memory + DataFrame construction overhead

**Example calculation**:
- 100 questions × 10,000 passages = 1,000,000 pairs
- With batch_size=100: ~100 passages in memory at once (~0.06 MB)
- Without batching: All 10,000 passages in memory (~6.85 MB)

## Best Practices

### 1. Start Small

When working with new datasets, start with smaller sample types:

```python
# Test with relevant pairs first
relevant_pairs = dataset.generate_samples("pairs_relevant")
print(f"Generated {len(relevant_pairs)} relevant pairs")

# Then expand to other types
```

### 2. Choose Appropriate Sample Types

Select sample types based on your training objectives:

- **Learning relevance**: Use `pairs_relevant` or `contrastive`
- **Learning to rank**: Use `similar` triplets
- **Comprehensive evaluation**: Use `pairs_all_existing` (with caution)
- **Multi-aspect learning**: Combine multiple types

### 3. Monitor Memory Usage

For large datasets:

```python
import psutil
import os

process = psutil.Process(os.getpid())

# Check memory before
mem_before = process.memory_info().rss / 1024 / 1024  # MB
print(f"Memory before: {mem_before:.2f} MB")

# Generate samples
pairs = dataset.generate_samples("pairs_all_existing", batch_size=50)

# Check memory after
mem_after = process.memory_info().rss / 1024 / 1024  # MB
print(f"Memory after: {mem_after:.2f} MB")
print(f"Memory used: {mem_after - mem_before:.2f} MB")
```

### 4. Validate Generated Samples

Always validate the quality of generated samples:

```python
# Check pair samples
pairs_df = dataset.generate_samples("pairs_relevant")
print(f"Total pairs: {len(pairs_df)}")
print(f"Unique questions: {pairs_df['question_id'].nunique()}")
print(f"Unique sources: {pairs_df['source_id'].nunique()}")
print(pairs_df.head())

# Check triplet samples
triplets = dataset.generate_samples("contrastive", num_negative_samples=5)
print(f"Total triplets: {len(triplets)}")
print(f"Label distribution: {[t.label for t in triplets[:10]]}")
```

### 5. Save Intermediate Results

For long-running operations:

```python
# Enable CSV saving to preserve results
pairs_df = dataset.generate_samples("pairs_all_existing", save_to_csv=True)

# Load from CSV later if needed
import pandas as pd
pairs_df = pd.read_csv("./data/bioasq/pairs_pairs_all_existing.csv")
```

## Troubleshooting

### Out of Memory Errors

If you encounter memory errors with `pairs_all_existing`:

1. Reduce batch size: `batch_size=25`
2. Process in chunks: Generate samples for subsets of questions
3. Write directly to CSV without loading into memory (contact maintainers for this feature)

### Empty Results

If sample generation returns empty results:

- Check that questions have relevant passages in metadata
- Verify the dataset is properly loaded
- Review the pairing criteria (e.g., `score_threshold` for similar triplets)

### Slow Performance

To improve performance:

- Increase `batch_size` (if memory allows)
- Use `keep_same_negatives=True` for contrastive triplets
- Reduce `num_negative_samples` or `top_k` parameters
- Enable CSV saving to avoid re-generating samples

## API Reference

For detailed API documentation, refer to:

- `RAG_supporters/dataset/rag_dataset.py` - Base class definitions
- `RAG_supporters/dataset/templates/rag_mini_bioasq.py` - BioASQ implementation

### Main Methods

- `generate_samples(sample_type, save_to_csv=True, **kwargs)` - Main entry point for sample generation
- `_generate_pair_samples_df(question_db_ids, criterion, **kwargs)` - Internal method for pair generation
- `_generate_positive_triplet_samples(...)` - Generate positive triplet samples
- `_generate_contrastive_triplet_samples(...)` - Generate contrastive triplet samples
- `_generate_similar_triplet_samples(...)` - Generate similar triplet samples
- `_generate_all_existing_pairs(...)` - Helper generator for all-existing pairs

---

**Last Updated**: 2024
**Maintainer**: RAG_Suppport_DNN Team
