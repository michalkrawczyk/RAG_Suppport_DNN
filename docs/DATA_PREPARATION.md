# Data Preparation Modules

Documentation for data preparation components used in JASPER dataset building.

---

## CSVMerger  

**Location**: `RAG_supporters/data_prep/merge_csv.py:22`

**Purpose**: Merges multiple CSV files with column normalization and deduplication.

### Features

- Normalizes column names via alias map
- Fills missing optional columns with null
- Groups by (question, source), applies merge rules:
  - Max scores
  - Union keywords
  - Longest answer
- Assigns unique IDs: `pair_idx`, `q_idx`, `s_idx`

### Usage

```python
from RAG_supporters.data_prep import CSVMerger

merger = CSVMerger(csv_paths=["data1.csv", "data2.csv"])
unified_df = merger.merge()
merger.save(output_dir="./output")  # Saves unified_pairs.parquet + JSON registries
```

### Output Files

- `unified_pairs.parquet` - Merged pairs with IDs
- `unique_questions.json` - Question ID → text mapping
- `unique_sources.json` - Source ID → text mapping

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `csv_paths` | `List[str]` | List of CSV file paths to merge |
| `column_aliases` | `Dict[str, str]` | Custom column name aliases (optional) |
| `merge_strategy` | `Dict[str, str]` | Strategy per column: "max", "union", "longest" (optional) |

### Methods

#### `merge() -> pd.DataFrame`

Merges all CSVs and returns unified DataFrame.

**Returns**: DataFrame with columns: `pair_idx`, `q_idx`, `s_idx`, `question_text`, `source_text`, plus optional columns

#### `save(output_dir: Path)`

Saves merged data and registries.

**Saves**:
- `unified_pairs.parquet` - Main merged data
- `unique_questions.json` - Question registry
- `unique_sources.json` - Source registry

### Column Normalization

Default aliases:
- `question`, `question_text`, `q` → `question`
- `source`, `source_text`, `src` → `source`
- `answer`, `answer_text`, `ans` → `answer`
- `keywords`, `kw` → `keywords`
- `score`, `relevance`, `relevance_score` → `score`

### Merge Rules

**Score**: Takes maximum score across duplicates  
**Keywords**: Takes union of all keywords  
**Answer**: Takes longest answer text

---

## DatasetSplitter (Simple)

**Location**: `RAG_supporters/data_prep/dataset_splitter.py:18`

**Purpose**: Simple train/val/test splitting with persistence.

### Usage

```python
from RAG_supporters.data_prep import DatasetSplitter

splitter = DatasetSplitter(random_state=42)
train_idx, val_idx = splitter.split(dataset_size=1000, val_ratio=0.2)
splitter.save("splits.json")

# Later, reload
loaded = DatasetSplitter.from_file("splits.json")
```

### Features

- Reproducible splits via random_state
- JSON persistence for exact split recovery
- Validation of split ratios

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `random_state` | `int` | Random seed for reproducibility |

### Methods

#### `split(dataset_size: int, val_ratio: float, test_ratio: float = 0.0) -> Tuple`

Creates train/val/test splits.

**Parameters**:
- `dataset_size`: Total number of samples
- `val_ratio`: Validation set ratio (0.0-1.0)
- `test_ratio`: Test set ratio (0.0-1.0), optional

**Returns**: Tuple of `(train_indices, val_indices)` or `(train_indices, val_indices, test_indices)`

#### `save(filepath: str)`

Saves split indices to JSON.

#### `from_file(filepath: str) -> DatasetSplitter`

Loads splitter from JSON (class method).

---

## DatasetSplitter (Question-Level Stratified)

**Location**: `RAG_supporters/data_prep/split.py:36`

**Purpose**: Question-level stratified splitting with cluster-based stratification.

### Features

- Groups pairs by question (no question leakage)
- Stratifies by dominant cluster per question
- Validates cluster coverage in all splits

### Usage

```python
from RAG_supporters.data_prep import stratified_split

train_idx, val_idx, test_idx = stratified_split(
    pair_indices, pair_cluster_ids,
    split_ratios=[0.8, 0.1, 0.1], random_state=42
)
```

### Guarantees

- No question in multiple splits
- All clusters represented in all splits
- Split ratios within ±1% of target

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `pair_indices` | `torch.Tensor` | Pair indices `[n_pairs, 2]` with `(q_idx, s_idx)` |
| `pair_cluster_ids` | `torch.Tensor` | Cluster ID per pair `[n_pairs]` |
| `split_ratios` | `List[float]` | Ratios for train/val/test (must sum to 1.0) |
| `random_state` | `int` | Random seed for reproducibility |

### Returns

Tuple of `(train_indices, val_indices, test_indices)` where each is a `torch.Tensor` of pair indices.

### Algorithm

1. Groups pairs by `q_idx`
2. Assigns dominant cluster to each question group (most pairs in that cluster)
3. Performs stratified split on question groups by dominant cluster
4. Flattens question groups back to pair indices

### Validation

Validates:
- No question appears in multiple splits
- All clusters present in all splits
- Split sizes match target ratios (±1%)
- All pair indices accounted for

---

## Related Documentation

- [JASPER Builder Guide](../dataset/JASPER_BUILDER_GUIDE.md) - Full pipeline
- [Contrastive Learning](CONTRASTIVE_LEARNING.md) - NegativeMiner, SteeringBuilder
- [Data Validation](DATA_VALIDATION.md) - Validation utilities
- [Clustering Operations](CLUSTERING_OPERATIONS.md) - ClusterParser, SourceClusterLinker
