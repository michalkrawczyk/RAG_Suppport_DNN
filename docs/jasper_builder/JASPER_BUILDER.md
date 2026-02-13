# JASPER Builder Modules

Documentation for JASPER dataset builder components.

---

## BuildConfig

**Purpose**: Configuration dataclass for dataset building parameters.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embedding_dim` | `int` | Required | Embedding dimensionality |
| `n_neg` | `int` | 12 | Number of hard negatives per pair |
| `tier_proportions` | `List[int]` | `[3, 4, 3, 2]` | Negative tier distribution |
| `split_ratios` | `List[float]` | `[0.8, 0.1, 0.1]` | Train/val/test split ratios |
| `steering_probabilities` | `Dict[str, float]` | Equal | Steering variant probabilities |
| `curriculum` | `Dict` | None | Curriculum learning settings |
| `similarity_thresholds` | `Dict` | `{"keyword": 0.92}` | Thresholds for matching |
| `adjacent_k` | `int` | 3 | Adjacent clusters for Tier 2 negatives |
| `batch_size` | `int` | 256 | Embedding batch size |
| `random_seed` | `int` | 42 | Random seed for reproducibility |
| `show_progress` | `bool` | True | Show progress bars |

### Usage

```python
from RAG_supporters.jasper import BuildConfig

config = BuildConfig(
    embedding_dim=384,
    n_neg=12,
    tier_proportions=[3, 4, 3, 2],
    split_ratios=[0.8, 0.1, 0.1],
    steering_probabilities={
        "zero": 0.25,
        "centroid": 0.25,
        "keyword": 0.25,
        "residual": 0.25
    },
    curriculum={
        "zero_prob_start": 0.5,
        "zero_prob_end": 0.1,
        "epochs_total": 100
    }
)
config.save("config.json")
```

### Methods

#### `save(filepath: str)`

Serializes config to JSON.

#### `from_file(filepath: str) -> BuildConfig`

Loads config from JSON (class method).

#### `validate()`

Validates configuration parameters:
- `tier_proportions` sum to `n_neg`
- `split_ratios` sum to 1.0
- `steering_probabilities` sum to 1.0
- All values within valid ranges

---

## DatasetFinalizer

**Purpose**: Cross-validation and integrity checks for built datasets.

### Validates

- Tensor shape consistency across all embeddings
- ID space completeness (0..N-1, no gaps)
- Referential integrity (all IDs resolve to valid entities)
- Dimension consistency across all embeddings
- Config counts match actual tensor shapes
- Split indices cover all pairs exactly once

### Usage

```python
from RAG_supporters.jasper import DatasetFinalizer

finalizer = DatasetFinalizer(dataset_dir="./output")
report = finalizer.validate()  # Returns validation report

if report["valid"]:
    finalizer.write_config()  # Writes validated config.json
else:
    print(f"Validation errors: {report['errors']}")
```

### Methods

#### `__init__(dataset_dir: Path)`

Initializes finalizer with dataset directory.

#### `validate() -> Dict[str, Any]`

Performs comprehensive validation.

**Returns**: Dictionary with keys:
- `valid` (bool): Overall validation status
- `errors` (List[str]): List of validation errors
- `warnings` (List[str]): List of warnings
- `stats` (Dict): Dataset statistics

**Validation Checks**:
1. **Tensor Shapes**: All embeddings have dimension D
2. **ID Continuity**: All ID spaces (questions, sources, keywords, clusters) are 0..N-1
3. **Referential Integrity**: All indices in pair_index resolve
4. **Dimension Match**: All embedding tensors have same dimension
5. **Split Coverage**: Train/val/test cover all pairs exactly once
6. **Config Match**: Config counts match actual tensor shapes

#### `write_config()`

Writes validated configuration to `config.json`.

#### `generate_report() -> str`

Generates human-readable validation report.

---

## build_dataset

**Purpose**: Task orchestrator for complete dataset building pipeline.

### Usage

```python
from RAG_supporters.jasper import build_dataset, BuildConfig
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

build_dataset(
    csv_paths=["data1.csv", "data2.csv"],
    cluster_json_path="clusters.json",
    embedding_model=model,
    output_dir="./dataset",
    config=BuildConfig(embedding_dim=384, n_neg=12)
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `csv_paths` | `List[str]` | List of CSV file paths |
| `cluster_json_path` | `str` | Path to cluster JSON |
| `embedding_model` | Model | Embedding model (sentence-transformers or LangChain) |
| `output_dir` | `str` | Output directory for dataset |
| `config` | `BuildConfig` | Build configuration |

### Pipeline Steps

1. **CSV Merger** - Normalizes, merges, assigns IDs
2. **Cluster Parser** - Matches keywords, extracts centroids
3. **Source-Cluster Linker** - Links sources to clusters
4. **Embedding Generator** - Embeds questions, sources, keywords
5. **Steering Builder** - Creates steering signals
6. **Negative Miner** - Samples 4-tier hard negatives
7. **Dataset Splitter** - Stratified train/val/test splits
8. **Config Writer** - Validates and finalizes
9. **Assembly** - Logs summary

### Returns

Dictionary with keys:
- `success` (bool): Whether build succeeded
- `output_dir` (str): Path to built dataset
- `stats` (Dict): Build statistics
- `timing` (Dict): Per-task timing
- `warnings` (List[str]): Any warnings

### Logging

Logs per-task:
- Start time
- Completion time
- Output file sizes
- Statistics (e.g., "Generated 10,000 pairs")
- Warnings or fallbacks

### Error Handling

- Fails fast on critical errors
- Logs which task failed
- Leaves partial outputs for debugging
- Returns error information in result dict

---

## Related Documentation

- [JASPER Builder Guide](dataset/JASPER_BUILDER_GUIDE.md) - Complete user guide
- [Data Preparation](DATA_PREPARATION.md) - CSV merger and splitters
- [Contrastive Learning](CONTRASTIVE_LEARNING.md) - NegativeMiner, SteeringBuilder
- [Data Validation](DATA_VALIDATION.md) - Validation utilities
- [JASPER Dataset](pytorch_datasets/JASPER_STEERING_DATASET.md) - Runtime usage
