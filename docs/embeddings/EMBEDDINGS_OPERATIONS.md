# Embeddings Operations

Documentation for embedding generation and management components.

---

## EmbeddingGenerator

**Purpose**: Batched embedding generation with caching for efficient processing.

### Features

- Batched processing for efficiency (GPU/CPU optimized)
- Caches embeddings to avoid recomputation
- Progress bars for long operations
- Supports sentence-transformers and LangChain models
- Memory-efficient streaming for large datasets

### Usage

```python
from RAG_supporters.embeddings_ops import EmbeddingGenerator
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
generator = EmbeddingGenerator(model, batch_size=256, show_progress=True)

# Generate embeddings
texts = ["text 1", "text 2", ...]
embeddings = generator.embed(texts)  # Returns [n, dim] tensor
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | Model | Embedding model (sentence-transformers or LangChain) |
| `batch_size` | `int` | Batch size for processing (default: 256) |
| `device` | `str` | Device for computation - "cuda" or "cpu" (default: auto) |
| `show_progress` | `bool` | Show progress bars (default: True) |
| `cache_dir` | `Path` | Directory for caching (optional) |

### Methods

#### `embed(texts: List[str]) -> torch.Tensor`

Generates embeddings for text list.

**Parameters**:
- `texts`: List of text strings

**Returns**: Tensor of shape `[n, dim]`

#### `embed_batched(texts: List[str]) -> Iterator[torch.Tensor]`

Generates embeddings in batches (memory-efficient).

**Parameters**:
- `texts`: List of text strings

**Yields**: Batches of embeddings

#### `clear_cache()`

Clears embedding cache.

### Caching

When `cache_dir` is provided:
- Computes hash of text
- Checks cache before computing
- Saves new embeddings to cache
- Significantly speeds up repeated operations

---

## SteeringEmbeddingGenerator

**Purpose**: Generates steering embeddings with various augmentation strategies.

### Augmentation Strategies

1. **Centroid-based**: Uses cluster centroid as steering
2. **Keyword-weighted**: Weighted average of keyword embeddings
3. **Residual**: Difference between centroid and question
4. **Zero**: No steering (baseline)

### Usage

```python
from RAG_supporters.embeddings_ops import SteeringEmbeddingGenerator

generator = SteeringEmbeddingGenerator(
    question_embeddings, keyword_embeddings, centroid_embeddings,
    cluster_assignments
)

# Generate steering signals
steering = generator.generate(mode="centroid", normalize=True)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `question_embeddings` | `torch.Tensor` | Question embeddings `[n, dim]` |
| `keyword_embeddings` | `torch.Tensor` | Keyword embeddings `[k, dim]` |
| `centroid_embeddings` | `torch.Tensor` | Centroid embeddings `[c, dim]` |
| `cluster_assignments` | `torch.Tensor` | Cluster per question `[n]` |

### Methods

#### `generate(mode: str, normalize: bool = True) -> torch.Tensor`

Generates steering embeddings.

**Parameters**:
- `mode`: Steering mode - "zero", "centroid", "keyword", "residual"
- `normalize`: Whether to L2-normalize (default: True)

**Returns**: Steering embeddings `[n, dim]`

#### `generate_mixed(probabilities: Dict[str, float]) -> torch.Tensor`

Generates mixed steering (stochastic selection per sample).

**Parameters**:
- `probabilities`: Dict of mode → probability

**Returns**: Mixed steering embeddings `[n, dim]`

---

## SteeringConfig

**Purpose**: Configuration for steering generation.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | `SteeringMode` | `MIXED` | Steering mode |
| `probabilities` | `Dict[str, float]` | Equal | Mode probabilities for MIXED |
| `normalize` | `bool` | True | L2-normalize steering vectors |
| `temperature` | `float` | 1.0 | Temperature for keyword weighting |

### Usage

```python
from RAG_supporters.embeddings_ops import SteeringConfig, SteeringMode

config = SteeringConfig(
    mode=SteeringMode.MIXED,
    probabilities={
        "zero": 0.25,
        "centroid": 0.25,
        "keyword": 0.25,
        "residual": 0.25
    },
    normalize=True
)
```

---

## SteeringMode

**Purpose**: Enum for steering modes.

### Values

| Mode | Description |
|------|-------------|
| `ZERO` | No steering signal (zeros) |
| `CENTROID` | Cluster centroid as steering |
| `KEYWORD` | Keyword-weighted average |
| `RESIDUAL` | Residual (centroid - question) |
| `MIXED` | Stochastic mixture of modes |

### Usage

```python
from RAG_supporters.embeddings_ops import SteeringMode

# Use in configuration
mode = SteeringMode.CENTROID

# Check mode
if mode == SteeringMode.ZERO:
    # Handle zero steering
    pass
```

---

## Related Documentation

- [Contrastive Learning](CONTRASTIVE_LEARNING.md) - SteeringBuilder for full steering generation
- [JASPER Builder](JASPER_BUILDER.md) - Complete pipeline
- [Data Preparation](DATA_PREPARATION.md) - CSV merger and splitters
