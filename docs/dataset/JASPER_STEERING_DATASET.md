````markdown
# JASPER Steering Dataset

PyTorch Dataset serving pre-computed embedding triplets `(question, steering, target_source)` for training JASPER predictors to learn: *given a question embedding and optional steering, predict target source location in latent space*.

**Features:** Zero I/O training • GPU preloading • Curriculum learning • 4-tier hard negatives • Hot-reload • Context manager • Distributed ready • Deterministic • Referential integrity validation

---

## Quick Start

**Basic Training:**
```python
from RAG_supporters.dataset import create_loader, validate_first_batch

loader = create_loader("./dataset", split="train", batch_size=32, num_workers=4)
validate_first_batch(loader)  # Check data integrity

for epoch in range(100):
    loader.dataset_obj.set_epoch(epoch)  # Update curriculum
    for batch in loader:
        question_emb, target_emb, steering, negatives = (
            batch["question_emb"], batch["target_source_emb"],
            batch["steering"], batch["negative_embs"]  # [B,D], [B,D], [B,D], [B,N_neg,D]
        )
        # Train model...
```

**GPU Preloading (10-20% faster):**
```python
import torch
from RAG_supporters.pytorch_datasets import JASPERSteeringDataset

# All tensors loaded directly to GPU - zero transfer during training
with JASPERSteeringDataset(
    dataset_dir="./dataset", split="train", epoch=0,
    device=torch.device("cuda")  # Preload to GPU
) as dataset:
    print(f"Memory: {dataset.memory_usage_mb:.2f} MB on {dataset.device}")
    sample = dataset[0]  # Already on GPU!
```

**Load All Splits:**
```python
splits = JASPERSteeringDataset.create_combined_splits(
    "./dataset", epoch=0, device=torch.device("cuda")
)
train_loader = DataLoader(splits["train"], batch_size=32, shuffle=True)
val_loader = DataLoader(splits["val"], batch_size=32)
```

**Force Steering for Validation:**
```python
val_loader.dataset_obj.force_steering("zero")     # Test without steering
val_loader.dataset_obj.force_steering("centroid") # Test with centroid
val_loader.dataset_obj.force_steering(None)       # Restore stochastic
```

---

## Dataset Structure

```
dataset_dir/
├── config.json                 # Configuration (embedding_dim, n_neg, curriculum, etc.)
├── *_embs.pt                   # question[N_q,D], source[N_s,D], keyword[N_k,D], centroid[N_c,D]
├── pair_*.pt                   # index[N_p,2], cluster_id[N_p], relevance[N_p], keyword_ids[List]
├── steering_*.pt               # centroid[N_p,D], keyword_weighted[N_p,D], residual[N_p,D]
├── hard_negatives.pt           # [N_pairs, N_neg]
├── negative_tiers.pt           # [N_pairs, N_neg]
└── {split}_idx.pt              # train/val/test split indices
```

**config.json:** Defines embedding_dim, n_neg, steering_probabilities (per variant), and curriculum (zero_prob_start→end over epochs).

---

## Batch Schema

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `question_emb` | `[B, D]` | float32 | Question embeddings |
| `target_source_emb` | `[B, D]` | float32 | Correct source embeddings |
| `steering` | `[B, D]` | float32 | Steering signal (variant-dependent) |
| `negative_embs` | `[B, N_neg, D]` | float32 | Hard negative source embeddings |
| `cluster_id` | `[B]` | long | Cluster ID (0 to N_clusters-1) |
| `relevance` | `[B]` | float32 | Relevance score [0, 1] |
| `centroid_distance` | `[B]` | float32 | Cosine distance to centroid [0, 2] |
| `steering_variant` | `[B]` | long | 0=zero, 1=centroid, 2=keyword, 3=residual |
| `negative_tiers` | `[B, N_neg]` | long | Tier for each negative (1-4) |

---

## Steering Variants

4 steering types selected stochastically per sample (probabilities evolve via curriculum):

- **Zero** (variant=0): `steering = zeros(D)` - baseline, no guidance
- **Centroid** (variant=1): `steering = centroid_embs[cluster_id]` - direct cluster target
- **Keyword-Weighted** (variant=2): `steering = weighted_mean(keyword_embs)` - relevance-weighted keywords
- **Residual** (variant=3): `steering = normalize(centroid - question)` - direction to centroid

**Curriculum:** Zero probability decays from high (e.g., 50% @ epoch 0) to low (10% @ epoch 100), gradually increasing task difficulty.

---

## Hard Negatives

Negatives sampled in 4 tiers (e.g., proportions `[3, 4, 3, 2]`):

1. **In-cluster** - Same cluster, excludes true source (tests within-cluster discrimination)
2. **Adjacent cluster** - Top-K nearest clusters (tests coarse boundaries)
3. **High-similarity** - Highest cosine similarity to question, wrong clusters (hardest negatives)
4. **Random distant** - Uniform random from far clusters (easy negatives for stability)

**Hot-reload:** Update `hard_negatives.pt` during training without restart: `loader.dataset_obj.reload_negatives()`

---

## API Reference

**JASPERSteeringDataset(dataset_dir, split, epoch=0, device="cpu")**
- `dataset_dir`: Path to dataset folder
- `split`: "train", "val", or "test"
- `epoch`: Initial epoch for curriculum
- `device`: Device placement ("cpu", "cuda", or torch.device)

**Methods:**
- `__len__()`: Returns split size
- `__getitem__(idx)`: Returns batch dict (raises IndexError if out of bounds)
- `set_epoch(epoch)`: Update curriculum and reseed RNG
- `reload_negatives()`: Hot-reload hard_negatives.pt from disk
- `force_steering(variant)`: "zero", "centroid", "keyword", "residual", or None (restore stochastic)
- `close()`: Resource cleanup (auto-called via `__del__` or `__exit__`)
- `__enter__()` / `__exit__()`: Context manager support
- `create_combined_splits(dataset_dir, epoch=0, device="cpu")` [static]: Load all splits at once

**Attributes:** `embedding_dim`, `n_neg`, `config`, `device`, `memory_usage_mb`

**Helper Functions:**
- `create_loader(dataset_dir, split, batch_size, num_workers=0, distributed=False, epoch=0, drop_last=None, pin_memory=True)` → DataLoader
- `set_epoch(loader, epoch)`: Update both dataset curriculum and sampler
- `validate_first_batch(loader)`: Check shapes, NaN/Inf, value ranges

---

## Performance & Troubleshooting

**Performance Tips:**
- **GPU preloading** (`device="cuda"`): 10-20% faster for datasets <10GB (eliminates per-batch transfers)
- **Multiple workers** (`num_workers=4+`): Parallel data loading
- **Context manager** (`with` statement): Automatic cleanup and stats logging
- **SSD storage**: Minimizes init time

**Common Issues:**
- **Config/split not found**: Check `config.json` and `{split}_idx.pt` exist
- **IndexError**: Validate DataLoader sampler, check index bounds
- **Referential integrity violation**: Rebuild dataset (corrupted indices detected)
- **NaN/Inf in batch**: Check source embeddings for invalid values
- **CUDA OOM**: Use CPU dataset with `pin_memory=True` or reduce batch size
- **Slow loading**: Try GPU preloading, increase workers, use SSD

**Distributed Training:**
```python
loader = create_loader(dataset_dir, split="train", batch_size=32, distributed=True)
for epoch in range(100):
    set_epoch(loader, epoch)  # Updates sampler + curriculum
    # Train DDP model...
```

---

## Testing

```bash
pytest tests/test_jasper_steering_dataset.py -v
pytest tests/test_loader.py -v
```

Tests cover: initialization, `__getitem__` schema, steering distribution, epoch updates, force steering, hot-reload, DataLoader batching, determinism, and validation utilities.

---

## References

- [PyTorch Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) • [DistributedSampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler) • [JEPA Paper](https://arxiv.org/abs/2301.08243) (inspiration)
- [Dataset Build Pipeline](../dataset/README.md) • [Dataset Splitting](DATASET_SPLITTING.md)


