````markdown
# JASPER Steering Dataset

## Overview

The **JASPER Steering Dataset** is a PyTorch Dataset that serves pre-computed embedding triplets `(question, steering, target_source)` with hard negatives and subspace labels. It enables training JASPER predictors (Joint Architecture for Subspace Prediction with Explainable Routing) to learn: *given a question embedding and an optional steering signal, predict where in latent space the correct source embedding lives*.

**Key Features:**
- **Zero I/O during training**: All embeddings preloaded at initialization
- **Curriculum learning**: Steering signal probabilities evolve across epochs
- **Hard negatives**: 4-tier negative sampling strategy (in-cluster, adjacent, high-similarity, random)
- **Hot-reloadable negatives**: Update negatives without restarting training
- **Distributed training ready**: Compatible with PyTorch DistributedSampler
- **Deterministic**: Same epoch + seed → identical samples

---

## Quick Start

### Basic Usage

```python
from RAG_supporters.dataset import create_loader, validate_first_batch

# Create DataLoader
loader = create_loader(
    dataset_dir="/path/to/dataset",
    split="train",
    batch_size=32,
    num_workers=4,
)

# Validate first batch
validate_first_batch(loader)

# Training loop
for epoch in range(100):
    loader.dataset_obj.set_epoch(epoch)  # Update curriculum
    
    for batch in loader:
        question_emb = batch["question_emb"]           # [B, D]
        target_emb = batch["target_source_emb"]        # [B, D]
        steering = batch["steering"]                   # [B, D]
        negatives = batch["negative_embs"]             # [B, N_neg, D]
        
        # Train model...
```

### Advanced: Force Steering for Validation

```python
# Create validation loader
val_loader = create_loader(dataset_dir, split="val", batch_size=32)

# Test with zero steering
val_loader.dataset_obj.force_steering("zero")
zero_metrics = evaluate(model, val_loader)

# Test with centroid steering
val_loader.dataset_obj.force_steering("centroid")
centroid_metrics = evaluate(model, val_loader)

# Restore stochastic steering
val_loader.dataset_obj.force_steering(None)
```

---

## Dataset Structure

The dataset is built from a single directory containing all required files:

```
dataset_dir/
├── config.json                        # Dataset configuration
├── question_embs.pt                   # [N_questions, D]
├── source_embs.pt                     # [N_sources, D]
├── keyword_embs.pt                    # [N_keywords, D]
├── centroid_embs.pt                   # [N_clusters, D]
├── pair_index.pt                      # [N_pairs, 2] (q_idx, s_idx)
├── pair_cluster_id.pt                 # [N_pairs]
├── pair_relevance.pt                  # [N_pairs] (normalized [0,1])
├── pair_keyword_ids.pt                # List[List[int]]
├── steering_centroid.pt               # [N_pairs, D]
├── steering_keyword_weighted.pt       # [N_pairs, D]
├── steering_residual.pt               # [N_pairs, D]
├── centroid_distances.pt              # [N_pairs]
├── hard_negatives.pt                  # [N_pairs, N_neg]
├── negative_tiers.pt                  # [N_pairs, N_neg]
├── train_idx.pt                       # Train split indices
├── val_idx.pt                         # Validation split indices
└── test_idx.pt                        # Test split indices
```

### Configuration Format

`config.json` contains:

```json
{
    "embedding_dim": 384,
    "n_neg": 12,
    "n_pairs": 10000,
    "n_questions": 5000,
    "n_sources": 8000,
    "n_keywords": 500,
    "n_clusters": 20,
    "steering_probabilities": {
        "zero": 0.25,
        "centroid": 0.25,
        "keyword": 0.25,
        "residual": 0.25
    },
    "curriculum": {
        "zero_prob_start": 0.5,
        "zero_prob_end": 0.1,
        "epochs_total": 100
    }
}
```

---

## Batch Schema

Each batch is a dictionary with the following keys:

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `question_emb` | `[B, D]` | `float32` | Question embeddings |
| `target_source_emb` | `[B, D]` | `float32` | Correct source embeddings |
| `steering` | `[B, D]` | `float32` | Steering signal (zero/centroid/keyword/residual) |
| `negative_embs` | `[B, N_neg, D]` | `float32` | Hard negative source embeddings |
| `cluster_id` | `[B]` | `long` | Cluster ID for steering (0 to N_clusters-1) |
| `relevance` | `[B]` | `float32` | Relevance score in [0, 1] |
| `centroid_distance` | `[B]` | `float32` | Cosine distance to cluster centroid in [0, 2] |
| `steering_variant` | `[B]` | `long` | Steering type: 0=zero, 1=centroid, 2=keyword, 3=residual |
| `negative_tiers` | `[B, N_neg]` | `long` | Tier for each negative: 1-4 |

---

## Steering Variants

The dataset supports 4 steering variants, selected stochastically per sample:

1. **Zero Steering** (`variant=0`):
   - `steering = zeros(D)`
   - Baseline: no guidance

2. **Centroid Steering** (`variant=1`):
   - `steering = centroid_embs[cluster_id]`
   - Direct cluster centroid as target subspace

3. **Keyword-Weighted Steering** (`variant=2`):
   - `steering = weighted_mean(keyword_embs, weights)`
   - Weights based on keyword relevance scores or cosine similarity to question

4. **Residual Steering** (`variant=3`):
   - `steering = normalize(centroid - question)`
   - Direction from question to centroid

### Curriculum Learning

Steering probabilities evolve across epochs:
- **Epoch 0**: High zero probability (e.g., 50%) → easier task
- **Epoch 100**: Low zero probability (e.g., 10%) → harder task with more steering

This gradual shift allows the model to first learn basic retrieval, then leverage steering signals.

---

## Hard Negatives

Negatives are sampled in 4 tiers (configurable proportions, e.g., `[3, 4, 3, 2]`):

1. **Tier 1: In-cluster negatives**
   - Random sources from the same cluster (excluding true source)
   - Tests within-cluster discrimination

2. **Tier 2: Adjacent cluster negatives**
   - Sources from top-K nearest clusters
   - Tests coarse-grained cluster boundaries

3. **Tier 3: High-similarity negatives**
   - Highest cosine similarity sources to question (wrong clusters)
   - Hardest negatives: semantically close but incorrect

4. **Tier 4: Random distant negatives**
   - Uniform random from far clusters
   - Easy negatives for stable training

---

## API Reference

### `JASPERSteeringDataset`

```python
class JASPERSteeringDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str | Path,
        split: Literal["train", "val", "test"],
        epoch: int = 0,
    )
```

**Parameters:**
- `dataset_dir`: Path to dataset folder
- `split`: Which split to load
- `epoch`: Initial epoch for curriculum learning

**Methods:**
- `__len__()`: Returns number of samples in split
- `__getitem__(idx)`: Returns batch dict (see schema above)
- `set_epoch(epoch)`: Update curriculum and reseed RNG
- `reload_negatives()`: Hot-reload negatives from disk
- `force_steering(variant)`: Force specific steering variant or restore stochastic

**Attributes:**
- `embedding_dim`: Dimensionality of embeddings
- `n_neg`: Number of hard negatives per sample
- `config`: Full dataset configuration

---

### `create_loader`

```python
def create_loader(
    dataset_dir: str | Path,
    split: Literal["train", "val", "test"],
    batch_size: int,
    num_workers: int = 0,
    distributed: bool = False,
    epoch: int = 0,
    drop_last: Optional[bool] = None,
    pin_memory: bool = True,
) -> DataLoader
```

**Parameters:**
- `dataset_dir`: Path to dataset folder
- `split`: Which split to load
- `batch_size`: Batch size
- `num_workers`: Number of worker processes
- `distributed`: Use DistributedSampler for multi-GPU
- `epoch`: Initial epoch
- `drop_last`: Drop last incomplete batch (default: True for train, False for val/test)
- `pin_memory`: Pin memory for faster GPU transfer

**Returns:** Configured `DataLoader`

---

### `set_epoch`

```python
def set_epoch(loader: DataLoader, epoch: int)
```

Set epoch for both dataset (curriculum) and sampler (distributed training).

---

### `validate_first_batch`

```python
def validate_first_batch(loader: DataLoader) -> bool
```

Validate that first batch has correct shapes and no NaN/Inf values.

**Checks:**
- All expected keys present
- Shapes match configuration
- No NaN/Inf in embeddings
- Cluster IDs, relevance, distances in valid ranges

---

## Distributed Training

```python
from torch.nn.parallel import DistributedDataParallel as DDP

# Create distributed loader
loader = create_loader(
    dataset_dir="/path/to/dataset",
    split="train",
    batch_size=32,
    num_workers=4,
    distributed=True,
    epoch=0,
)

# Training loop
for epoch in range(100):
    set_epoch(loader, epoch)  # Updates sampler + dataset
    
    for batch in loader:
        # Train DDP model...
```

---

## Hot-Reloading Negatives

Negatives can be refreshed periodically without restarting training:

```python
# Every N epochs, reload negatives
if epoch % 10 == 0:
    # External process updates hard_negatives.pt
    # (e.g., re-mine negatives based on current model)
    
    loader.dataset_obj.reload_negatives()
    print(f"Reloaded negatives at epoch {epoch}")
```

---

## Performance Tips

1. **Use multiple workers**: Set `num_workers=4` or higher for faster data loading
2. **Pin memory**: Keep `pin_memory=True` for GPU training
3. **Preload on SSD**: Store dataset on fast SSD for minimal load time
4. **Validate once**: Run `validate_first_batch()` once at start, not every epoch

---

## Building the Dataset

The dataset is built using the `dataset_builder` pipeline (see separate documentation):

```python
from RAG_supporters.dataset.dataset_builder import build_dataset

build_dataset(
    csv_paths=["data1.csv", "data2.csv"],
    cluster_json_path="clusters.json",
    embedding_model=model,
    output_dir="./dataset_output",
)
```

---

## Success Criteria

The dataset implementation satisfies:

- ✅ Zero I/O in `__getitem__`
- ✅ `__len__` returns split size
- ✅ Fixed schema with deterministic shapes
- ✅ Stochastic steering seeded by epoch
- ✅ Curriculum learning via `set_epoch`
- ✅ Hot-reload negatives
- ✅ Force steering for validation
- ✅ No data leakage between splits
- ✅ DistributedSampler compatible
- ✅ Default collation produces batch with no ragged tensors
- ✅ `num_workers > 0` functions without deadlock

---

## Testing

Comprehensive test suite covers:

- Initialization and splits
- `__getitem__` schema and shapes
- Steering variant distribution matches config
- `set_epoch` changes probabilities
- `force_steering` overrides correctly
- `reload_negatives` swaps tensor
- DataLoader batch shapes
- Determinism: same seed → same samples
- Validation utilities

Run tests:

```bash
pytest tests/test_jasper_steering_dataset.py -v
pytest tests/test_loader.py -v
```

---

## Troubleshooting

**Error: "Config file not found"**
- Ensure `config.json` exists in dataset directory
- Check path is correct and readable

**Error: "Split file not found"**
- Ensure `{split}_idx.pt` exists
- Valid splits: `train`, `val`, `test`

**Dimension mismatch errors**
- Verify all embedding tensors have same `D` (embedding_dim)
- Check config.json `embedding_dim` matches actual tensors

**NaN/Inf in batch**
- Check source embedding tensors for NaN/Inf
- Validate steering computations during build

**Slow data loading**
- Increase `num_workers`
- Move dataset to SSD
- Reduce batch size if memory-bound

---

## References

- **PyTorch Dataset**: https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
- **DistributedSampler**: https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
- **JEPA Paper**: https://arxiv.org/abs/2301.08243 (Joint-Embedding Predictive Architecture - inspiration for JASPER)

---

## See Also

- `dataset_builder/` - Dataset build pipeline documentation
- `tests/test_jasper_steering_dataset.py` - Test examples
- `docs/dataset/DATASET_SPLITTING.md` - Split strategy details
- `JEPA_STEERING_DATASET.md` - Legacy documentation (deprecated, use this document)

````
