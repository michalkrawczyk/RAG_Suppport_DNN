# Contrastive Learning Modules

Documentation for contrastive learning components used in JASPER dataset building.

---

## NegativeMiner

**Location**: `RAG_supporters/contrastive/mine_negatives.py:38`

**Purpose**: Hard negative sampling with 4-tier difficulty-based strategy for contrastive learning.

### Tiers

1. **In-cluster**: Same cluster, excluding true source
2. **Adjacent**: Top-K nearest clusters  
3. **High-similarity**: Highest cosine similarity, wrong clusters
4. **Random distant**: Uniform random from far clusters

### Usage

```python
from RAG_supporters.contrastive import NegativeMiner

miner = NegativeMiner(
    source_embeddings, question_embeddings, centroid_embeddings,
    pair_indices, pair_cluster_ids, source_cluster_ids,
    n_neg=12, tier_proportions=[3, 4, 3, 2]
)
hard_negatives, tiers = miner.mine()  # Returns [n_pairs, n_neg] indices and tier labels
```

### Key Features

- True source never in own negative set
- Configurable tier proportions
- Graceful fallback for edge cases
- Batched similarity computation for efficiency

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `source_embeddings` | `torch.Tensor` | Source embeddings `[n_sources, dim]` |
| `question_embeddings` | `torch.Tensor` | Question embeddings `[n_questions, dim]` |
| `centroid_embeddings` | `torch.Tensor` | Cluster centroid embeddings `[n_clusters, dim]` |
| `pair_indices` | `torch.Tensor` | Pair indices `[n_pairs, 2]` mapping to `(q_idx, s_idx)` |
| `pair_cluster_ids` | `torch.Tensor` | Primary cluster ID per pair `[n_pairs]` |
| `source_cluster_ids` | `torch.Tensor` | Cluster assignment for each source `[n_sources]` |
| `n_neg` | `int` | Total number of negatives per pair |
| `tier_proportions` | `List[int]` | Negatives per tier (must sum to `n_neg`). Default: equal |
| `adjacent_k` | `int` | Number of adjacent clusters for Tier 2 (default: 3) |
| `random_seed` | `int` | Random seed for reproducibility (default: 42) |
| `show_progress` | `bool` | Show progress bars (default: True) |

### Methods

#### `mine() -> Tuple[torch.Tensor, torch.Tensor]`

Mines hard negatives for all pairs.

**Returns**:
- `hard_negatives` (`torch.Tensor`): Negative source indices `[n_pairs, n_neg]`
- `negative_tiers` (`torch.Tensor`): Tier labels `[n_pairs, n_neg]` with values 1-4

#### `save(output_dir: Path)`

Saves mined negatives to disk.

**Saves**:
- `hard_negatives.pt` - Negative indices
- `negative_tiers.pt` - Tier labels
- `mining_log.json` - Mining statistics and metadata

### Tier Strategy Details

#### Tier 1: In-Cluster Negatives
- **Selection**: Random sources from same cluster as positive
- **Exclusion**: True source always excluded
- **Purpose**: Tests fine-grained within-cluster discrimination
- **Fallback**: Spills to Tier 2 if cluster too small

#### Tier 2: Adjacent Cluster Negatives  
- **Selection**: Random sources from top-K nearest clusters
- **Purpose**: Tests coarse cluster boundaries
- **Fallback**: Spills to Tier 3 if insufficient adjacent sources

#### Tier 3: High-Similarity Negatives
- **Selection**: Highest cosine similarity to question (wrong clusters only)
- **Purpose**: Hardest negatives - semantically close but incorrect
- **Implementation**: Uses batched similarity computation for efficiency

#### Tier 4: Random Distant Negatives
- **Selection**: Uniform random from distant clusters  
- **Purpose**: Easy negatives for training stability
- **No fallback**: Always sufficient sources

### Edge Case Handling

- **Small clusters**: Graceful degradation to next tier
- **Insufficient negatives**: Logs warnings, fills remaining from later tiers
- **Single-cluster datasets**: Falls back to similarity-based sampling
- **Empty keyword sets**: Uses cluster-based strategies only

### Validation

The miner validates:
- True source never in own negative set
- All negative indices within valid source range
- Tier proportions match configuration (or documented fallbacks)
- All tensors have consistent shapes

---

## SteeringBuilder

**Location**: `RAG_supporters/contrastive/build_steering.py:39`

**Purpose**: Generates steering signals for curriculum learning and subspace guidance.

### Steering Types

1. **Centroid**: Direct lookup of cluster centroid
2. **Keyword-weighted**: Weighted average of keyword embeddings
3. **Residual**: Normalized difference (centroid - question)

### Usage

```python
from RAG_supporters.contrastive import SteeringBuilder

builder = SteeringBuilder(
    question_embeddings, keyword_embeddings, centroid_embeddings,
    pair_indices, pair_cluster_ids, pair_keyword_ids
)
results = builder.build()  # Returns dict with steering tensors and distances
```

### Output

Dictionary with keys:
- `steering_centroid`: `[n_pairs, dim]` - Direct cluster centroid lookup
- `steering_keyword_weighted`: `[n_pairs, dim]` - Weighted average of keywords
- `steering_residual`: `[n_pairs, dim]` - Normalized (centroid - question)
- `centroid_distances`: `[n_pairs]` - Cosine distance for curriculum scheduling

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `question_embeddings` | `torch.Tensor` | Question embeddings `[n_questions, dim]` |
| `keyword_embeddings` | `torch.Tensor` | Keyword embeddings `[n_keywords, dim]` |
| `centroid_embeddings` | `torch.Tensor` | Cluster centroid embeddings `[n_clusters, dim]` |
| `pair_indices` | `torch.Tensor` | Pair indices `[n_pairs, 2]` mapping to `(q_idx, s_idx)` |
| `pair_cluster_ids` | `torch.Tensor` | Primary cluster ID per pair `[n_pairs]` |
| `pair_keyword_ids` | `List[List[int]]` | Keyword IDs associated with each pair |
| `normalize_residual` | `bool` | Whether to normalize residual steering (default: False) |
| `fallback_strategy` | `str` | How to handle pairs with no keywords (default: "centroid") |
| `show_progress` | `bool` | Show progress bars (default: True) |

### Validation

Ensures:
- Unit norm for residuals
- Weights sum to 1
- No NaN/Inf values
- Centroid steering matches exact lookup

### Methods

#### `build() -> Dict[str, torch.Tensor]`

Builds all steering signals.

**Returns**: Dictionary with steering tensors and distances

#### `save(output_dir: Path)`

Saves steering signals to disk.

**Saves**:
- `steering_centroid.pt`
- `steering_keyword_weighted.pt`
- `steering_residual.pt`
- `centroid_distances.pt`

---

## Related Documentation

- [JASPER Builder Guide](../dataset/JASPER_BUILDER_GUIDE.md) - Full pipeline
- [JASPER Dataset](../pytorch_datasets/JASPER_STEERING_DATASET.md) - Runtime usage
- [Data Preparation](DATA_PREPARATION.md) - CSV merger and splitters
- [Data Validation](DATA_VALIDATION.md) - Validation utilities
