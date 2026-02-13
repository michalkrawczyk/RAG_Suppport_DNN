# Data Validation Modules

Documentation for data validation utilities.

---

## validation_utils.py

**Location**: `RAG_supporters/data_validation/validation_utils.py:1`

**Purpose**: Tensor validation with bounds checking and NaN detection.

### Functions

#### `validate_tensor_2d(tensor: torch.Tensor, name: str)`

Validates that tensor is 2D with proper shape.

**Parameters**:
- `tensor`: Tensor to validate
- `name`: Name for error messages

**Raises**: `ValueError` if not 2D or has invalid shape

---

#### `validate_tensor_1d(tensor: torch.Tensor, name: str)`

Validates that tensor is 1D with proper shape.

**Parameters**:
- `tensor`: Tensor to validate
- `name`: Name for error messages

**Raises**: `ValueError` if not 1D or has invalid shape

---

#### `validate_embedding_dimensions(tensor1: torch.Tensor, tensor2: torch.Tensor)`

Checks that two tensors have matching embedding dimensions.

**Parameters**:
- `tensor1`: First tensor
- `tensor2`: Second tensor

**Raises**: `ValueError` if dimensions don't match

---

#### `validate_pair_indices_bounds(indices: torch.Tensor, max_q: int, max_s: int)`

Validates that pair indices are within valid bounds.

**Parameters**:
- `indices`: Pair indices `[n_pairs, 2]`
- `max_q`: Maximum valid question index
- `max_s`: Maximum valid source index

**Raises**: `ValueError` if any index is out of bounds

---

#### `validate_cluster_ids_bounds(ids: torch.Tensor, n_clusters: int)`

Validates that cluster IDs are within valid range.

**Parameters**:
- `ids`: Cluster IDs `[n]`
- `n_clusters`: Number of clusters

**Raises**: `ValueError` if any ID is invalid (< 0 or >= n_clusters)

---

#### `validate_length_consistency(*tensors)`

Ensures all tensors have the same length.

**Parameters**:
- `tensors`: Variable number of tensors to check

**Raises**: `ValueError` if lengths don't match

---

#### `validate_no_nan_inf(tensor: torch.Tensor, name: str)`

Detects NaN or Inf values in tensor.

**Parameters**:
- `tensor`: Tensor to check
- `name`: Name for error messages

**Raises**: `ValueError` if NaN or Inf detected

---

#### `validate_keyword_ids_list(keyword_ids: List[List[int]], n_keywords: int)`

Validates list of keyword ID lists.

**Parameters**:
- `keyword_ids`: List of lists of keyword IDs
- `n_keywords`: Total number of keywords

**Raises**: `ValueError` if any keyword ID is invalid

---

### Usage Example

```python
from RAG_supporters.data_validation import (
    validate_tensor_2d, 
    validate_no_nan_inf,
    validate_embedding_dimensions
)

# Validate embeddings
validate_tensor_2d(embeddings, "question_embeddings")
validate_no_nan_inf(embeddings, "question_embeddings")

# Validate dimension match
validate_embedding_dimensions(question_embs, source_embs)
```

---

## tensor_utils.py

**Location**: `RAG_supporters/data_validation/tensor_utils.py:1`

**Purpose**: Tensor I/O operations with automatic validation.

### Functions

#### `save_tensor(tensor: torch.Tensor, path: Path, validate: bool = True)`

Saves tensor with optional validation.

**Parameters**:
- `tensor`: Tensor to save
- `path`: Output file path
- `validate`: Whether to validate before saving (default: True)

**Validation**:
- Checks for NaN/Inf if enabled
- Logs tensor shape and dtype
- Creates parent directory if needed

---

#### `load_tensor(path: Path, validate: bool = True) -> torch.Tensor`

Loads tensor with optional validation.

**Parameters**:
- `path`: Input file path
- `validate`: Whether to validate after loading (default: True)

**Returns**: Loaded tensor

**Validation**:
- Checks for NaN/Inf if enabled
- Logs tensor shape and dtype

---

#### `save_tensor_dict(tensors: Dict[str, torch.Tensor], output_dir: Path)`

Batch saves multiple tensors.

**Parameters**:
- `tensors`: Dictionary of name → tensor
- `output_dir`: Output directory

**Saves**: Each tensor as `{name}.pt` in output directory

---

#### `load_tensor_dict(input_dir: Path, keys: List[str]) -> Dict[str, torch.Tensor]`

Batch loads multiple tensors.

**Parameters**:
- `input_dir`: Input directory
- `keys`: List of tensor names to load

**Returns**: Dictionary of name → tensor

---

### Usage Example

```python
from RAG_supporters.data_validation import save_tensor, load_tensor

# Save with validation
save_tensor(embeddings, "embeddings.pt", validate=True)

# Load with validation
embeddings = load_tensor("embeddings.pt", validate=True)

# Batch operations
tensors = {
    "question_embs": question_embs,
    "source_embs": source_embs
}
save_tensor_dict(tensors, "./output")
```

---

## label_calculator.py

**Location**: `RAG_supporters/data_validation/label_calculator.py:1`

**Purpose**: Label normalization with softmax and L1 methods.

### Class: LabelCalculator

Normalizes scores/labels for training.

#### Methods

##### `__init__(method: str = "softmax", temperature: float = 1.0)`

Initializes calculator.

**Parameters**:
- `method`: Normalization method - "softmax", "l1", or "minmax"
- `temperature`: Temperature for softmax (default: 1.0)

---

##### `normalize(scores: torch.Tensor) -> torch.Tensor`

Normalizes scores to [0, 1] range.

**Parameters**:
- `scores`: Raw scores tensor

**Returns**: Normalized scores

---

### Normalization Methods

**Softmax**: Temperature-scaled softmax normalization
```python
normalized = exp(scores / temperature) / sum(exp(scores / temperature))
```

**L1**: L1 normalization (sum to 1)
```python
normalized = scores / sum(abs(scores))
```

**MinMax**: Min-max scaling to [0, 1]
```python
normalized = (scores - min) / (max - min)
```

### Usage Example

```python
from RAG_supporters.data_validation import LabelCalculator

# Softmax normalization
calc = LabelCalculator(method="softmax", temperature=1.0)
normalized = calc.normalize(scores)

# L1 normalization
calc = LabelCalculator(method="l1")
normalized = calc.normalize(scores)
```

---

## Related Documentation

- [Data Preparation](DATA_PREPARATION.md) - CSV merger and splitters
- [Contrastive Learning](CONTRASTIVE_LEARNING.md) - NegativeMiner, SteeringBuilder
- [JASPER Builder Guide](dataset/JASPER_BUILDER_GUIDE.md) - Full pipeline
