# Missing Module Documentation

This document provides concise documentation for modules identified as lacking documentation.

## Table of Contents

1. [Contrastive Learning](#contrastive-learning)
2. [Data Preparation](#data-preparation)  
3. [Data Validation](#data-validation)
4. [JASPER Builder](#jasper-builder)
5. [Embeddings Operations](#embeddings-operations)
6. [Clustering Operations](#clustering-operations)
7. [General Utilities](#general-utilities)

---

## Contrastive Learning

### NegativeMiner

**Location**: `RAG_supporters/contrastive/mine_negatives.py:38`

**Purpose**: Hard negative sampling with 4-tier difficulty-based strategy for contrastive learning.

**Tiers**:
1. In-cluster: Same cluster, excluding true source
2. Adjacent: Top-K nearest clusters  
3. High-similarity: Highest cosine similarity, wrong clusters
4. Random distant: Uniform random from far clusters

**Usage**:
```python
from RAG_supporters.contrastive import NegativeMiner

miner = NegativeMiner(
    source_embeddings, question_embeddings, centroid_embeddings,
    pair_indices, pair_cluster_ids, source_cluster_ids,
    n_neg=12, tier_proportions=[3, 4, 3, 2]
)
hard_negatives, tiers = miner.mine()  # Returns [n_pairs, n_neg] indices and tier labels
```

**Key Features**:
- True source never in own negative set
- Configurable tier proportions
- Graceful fallback for edge cases
- Batched similarity computation for efficiency

---

### SteeringBuilder

**Location**: `RAG_supporters/contrastive/build_steering.py:39`

**Purpose**: Generates steering signals for curriculum learning and subspace guidance.

**Steering Types**:
1. **Centroid**: Direct lookup of cluster centroid
2. **Keyword-weighted**: Weighted average of keyword embeddings
3. **Residual**: Normalized difference (centroid - question)

**Usage**:
```python
from RAG_supporters.contrastive import SteeringBuilder

builder = SteeringBuilder(
    question_embeddings, keyword_embeddings, centroid_embeddings,
    pair_indices, pair_cluster_ids, pair_keyword_ids
)
results = builder.build()  # Returns dict with steering tensors and distances
```

**Output**: Dictionary with keys:
- `steering_centroid`: [n_pairs, dim]
- `steering_keyword_weighted`: [n_pairs, dim]
- `steering_residual`: [n_pairs, dim]
- `centroid_distances`: [n_pairs] - for curriculum scheduling

**Validation**: Ensures unit norm for residuals, weights sum to 1, no NaN/Inf

---

## Data Preparation

### CSVMerger  

**Location**: `RAG_supporters/data_prep/merge_csv.py:22`

**Purpose**: Merges multiple CSV files with column normalization and deduplication.

**Features**:
- Normalizes column names via alias map
- Fills missing optional columns with null
- Groups by (question, source), applies merge rules:
  - Max scores
  - Union keywords
  - Longest answer
- Assigns unique IDs: `pair_idx`, `q_idx`, `s_idx`

**Usage**:
```python
from RAG_supporters.data_prep import CSVMerger

merger = CSVMerger(csv_paths=["data1.csv", "data2.csv"])
unified_df = merger.merge()
merger.save(output_dir="./output")  # Saves unified_pairs.parquet + JSON registries
```

**Output Files**:
- `unified_pairs.parquet` - Merged pairs with IDs
- `unique_questions.json` - Question ID → text mapping
- `unique_sources.json` - Source ID → text mapping

---

### DatasetSplitter (Simple)

**Location**: `RAG_supporters/data_prep/dataset_splitter.py:18`

**Purpose**: Simple train/val/test splitting with persistence.

**Usage**:
```python
from RAG_supporters.data_prep import DatasetSplitter

splitter = DatasetSplitter(random_state=42)
train_idx, val_idx = splitter.split(dataset_size=1000, val_ratio=0.2)
splitter.save("splits.json")

# Later, reload
loaded = DatasetSplitter.from_file("splits.json")
```

**Features**:
- Reproducible splits via random_state
- JSON persistence for exact split recovery
- Validation of split ratios

---

### DatasetSplitter (Question-Level Stratified)

**Location**: `RAG_supporters/data_prep/split.py:36`

**Purpose**: Question-level stratified splitting with cluster-based stratification.

**Features**:
- Groups pairs by question (no question leakage)
- Stratifies by dominant cluster per question
- Validates cluster coverage in all splits

**Usage**:
```python
from RAG_supporters.data_prep import stratified_split

train_idx, val_idx, test_idx = stratified_split(
    pair_indices, pair_cluster_ids,
    split_ratios=[0.8, 0.1, 0.1], random_state=42
)
```

**Guarantees**:
- No question in multiple splits
- All clusters represented in all splits
- Split ratios within ±1% of target

---

## Data Validation

### validation_utils.py

**Location**: `RAG_supporters/data_validation/validation_utils.py:1`

**Purpose**: Tensor validation with bounds checking and NaN detection.

**Functions**:
- `validate_tensor_2d(tensor, name)` - Validates 2D tensor shape
- `validate_tensor_1d(tensor, name)` - Validates 1D tensor shape
- `validate_embedding_dimensions(tensor1, tensor2)` - Checks dimension match
- `validate_pair_indices_bounds(indices, max_q, max_s)` - Validates index ranges
- `validate_cluster_ids_bounds(ids, n_clusters)` - Validates cluster IDs
- `validate_length_consistency(*tensors)` - Ensures same length
- `validate_no_nan_inf(tensor, name)` - Detects NaN/Inf values

**Usage**:
```python
from RAG_supporters.data_validation import (
    validate_tensor_2d, validate_no_nan_inf
)

validate_tensor_2d(embeddings, "question_embeddings")
validate_no_nan_inf(embeddings, "question_embeddings")
```

---

### tensor_utils.py

**Location**: `RAG_supporters/data_validation/tensor_utils.py:1`

**Purpose**: Tensor I/O operations with automatic validation.

**Functions**:
- `save_tensor(tensor, path, validate=True)` - Save with validation
- `load_tensor(path, validate=True)` - Load with validation
- `save_tensor_dict(tensors, output_dir)` - Batch save
- `load_tensor_dict(input_dir, keys)` - Batch load

**Features**:
- Automatic shape/type validation
- NaN/Inf detection
- Logging of save/load operations
- Handles both CPU and GPU tensors

---

### label_calculator.py

**Location**: `RAG_supporters/data_validation/label_calculator.py:1`

**Purpose**: Label normalization with softmax and L1 methods.

**Usage**:
```python
from RAG_supporters.data_validation import LabelCalculator

calc = LabelCalculator(method="softmax", temperature=1.0)
normalized = calc.normalize(scores)  # Returns [0, 1] normalized scores
```

**Methods**:
- `softmax`: Temperature-scaled softmax normalization
- `l1`: L1 normalization (sum to 1)
- `minmax`: Min-max scaling to [0, 1]

---

## JASPER Builder

### BuildConfig

**Purpose**: Configuration dataclass for dataset building parameters.

**Parameters**:
- `embedding_dim`: Embedding dimensionality
- `n_neg`: Number of hard negatives per pair
- `tier_proportions`: Negative tier distribution
- `split_ratios`: Train/val/test split ratios
- `steering_probabilities`: Steering variant probabilities
- `curriculum`: Curriculum learning settings
- `similarity_thresholds`: Thresholds for matching

**Usage**:
```python
from RAG_supporters.jasper import BuildConfig

config = BuildConfig(
    embedding_dim=384,
    n_neg=12,
    tier_proportions=[3, 4, 3, 2],
    split_ratios=[0.8, 0.1, 0.1]
)
config.save("config.json")
```

---

### DatasetFinalizer

**Purpose**: Cross-validation and integrity checks for built datasets.

**Validates**:
- Tensor shape consistency
- ID space completeness (no gaps)
- Referential integrity (all IDs resolve)
- Dimension consistency across embeddings
- Config matches actual data

**Usage**:
```python
from RAG_supporters.jasper import DatasetFinalizer

finalizer = DatasetFinalizer(dataset_dir="./output")
report = finalizer.validate()  # Returns validation report
finalizer.write_config()  # Writes validated config.json
```

---

### build_dataset

**Purpose**: Task orchestrator for complete dataset building pipeline.

**Usage**:
```python
from RAG_supporters.jasper import build_dataset

build_dataset(
    csv_paths=["data1.csv", "data2.csv"],
    cluster_json_path="clusters.json",
    embedding_model=model,
    output_dir="./dataset",
    config=BuildConfig(...)
)
```

**Pipeline Steps**:
1. Merge CSVs with deduplication
2. Parse cluster JSON
3. Link sources to clusters
4. Generate embeddings
5. Build steering signals
6. Mine hard negatives
7. Create train/val/test splits
8. Validate and finalize

**Logging**: Per-task timing and output summaries

---

## Embeddings Operations

### EmbeddingGenerator

**Purpose**: Batched embedding generation with caching.

**Features**:
- Batched processing for efficiency
- Caches embeddings to avoid recomputation
- Progress bars for long operations
- Supports sentence-transformers and LangChain models

---

### SteeringEmbeddingGenerator

**Purpose**: Generate steering embeddings with augmentations.

**Augmentation Strategies**:
- Centroid-based steering
- Keyword-weighted steering
- Residual steering
- Zero steering (baseline)

---

### SteeringConfig / SteeringMode

**SteeringMode Enum**:
- `ZERO`: No steering signal
- `CENTROID`: Cluster centroid steering
- `KEYWORD`: Keyword-weighted steering
- `RESIDUAL`: Residual (centroid - question) steering
- `MIXED`: Stochastic mixture

**SteeringConfig**: Configuration for steering generation with curriculum settings.

---

## Clustering Operations

### ClusterParser

**Location**: `RAG_supporters/clustering_ops/parse_clusters.py:27`

**Purpose**: Parses cluster JSON with keyword matching (exact + cosine fallback).

**Features**:
- Extracts keyword texts, embeddings, cluster IDs, centroids
- Exact match first, cosine fallback ≥ threshold
- Assigns `kw_idx` to unique keywords
- Generates cluster labels (nearest keyword to centroid)

**Usage**:
```python
from RAG_supporters.clustering_ops import ClusterParser

parser = ClusterParser("clusters.json", similarity_threshold=0.92)
cluster_data = parser.parse()  # Returns ClusterData with mappings
```

**Output**:
- `keyword_to_cluster.json` - Keyword → cluster mapping
- `cluster_to_keywords.json` - Cluster → keywords mapping
- `cluster_labels.json` - Human-readable cluster labels
- `unique_keywords.json` - Keyword ID → text mapping

---

### SourceClusterLinker

**Location**: `RAG_supporters/clustering_ops/link_sources.py:29`

**Purpose**: Links question-source pairs to clusters via keywords.

**Process**:
1. Collect keywords for each unique source
2. Map keywords → clusters
3. Resolve primary cluster per source (most keywords, tie-break by relevance)
4. Assign `pair_cluster_id` from source's primary cluster

**Usage**:
```python
from RAG_supporters.clustering_ops import SourceClusterLinker

linker = SourceClusterLinker(
    unified_pairs_df, keyword_mappings, cluster_data
)
pair_clusters, source_clusters = linker.link()
```

**Fallback**: Sources with no keywords assigned to nearest cluster by embedding similarity

---

## General Utilities

### text_utils.py

**Location**: `RAG_supporters/utils/text_utils.py:1`

**Functions**:
- `is_empty_text(text)` - Checks if text is None, empty, or whitespace-only
- `normalize_string(text)` - Lowercases, strips, normalizes whitespace
- `clean_text(text)` - Removes special chars, extra whitespace
- `truncate_text(text, max_len)` - Truncates with ellipsis

**Usage**:
```python
from RAG_supporters.utils.text_utils import is_empty_text, normalize_string

if not is_empty_text(source_text):
    normalized = normalize_string(source_text)
```

---

### suggestion_processing.py

**Purpose**: Processing utilities for LLM-generated suggestions.

**Functions**:
- `filter_by_field_value(suggestions, min_confidence)` - Filters by confidence
- `aggregate_unique_terms(suggestions, normalize)` - Deduplicates terms
- `parse_json_suggestions(text)` - Extracts JSON from LLM output

---

### text_splitters.py

**Purpose**: Text chunking and segmentation utilities.

**Functions**:
- `split_by_sentences(text, max_length)` - Sentence-aware splitting
- `split_by_tokens(text, max_tokens, tokenizer)` - Token-based splitting
- `split_with_overlap(text, chunk_size, overlap)` - Overlapping chunks

**Usage**:
```python
from RAG_supporters.utils.text_splitters import split_by_sentences

chunks = split_by_sentences(long_text, max_length=512)
```

---

## Related Documentation

- [JASPER Steering Dataset](./pytorch_datasets/JASPER_STEERING_DATASET.md)
- [JASPER Training Examples](./dataset/JASPER_TRAINING_EXAMPLE.md)
- [DataLoader Utilities](./pytorch_datasets/LOADER_UTILITIES.md)
- [Project Structure](../agents_notes/PROJECT_STRUCTURE.md)

---

## Documentation Status

✅ = Complete documentation exists  
⚠️ = Partial documentation exists  
❌ = No documentation (covered in this file)

This document provides concise reference for modules lacking comprehensive documentation. For detailed API references, see inline docstrings in source files.
