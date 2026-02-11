# Dataset Builder Pipeline

> **Status:** 🚧 Not yet implemented. Specifications below.

This directory will contain the complete pipeline for building JASPER Steering Datasets from CSV files and cluster JSON.

## Overview

The dataset builder processes:
- Multiple CSV files with question-source pairs
- Cluster JSON with keyword embeddings and centroids
- Embedding model for text encoding

And produces a single self-contained dataset directory ready for training.

## Pipeline Tasks

### Task 0: Scaffold ✅ Partial
- [ ] `__init__.py` - Package constants and types
- [ ] `config.py` - BuildConfig dataclass with serialization

### Task 1: CSV Merger
- [ ] `merge_csv.py` - Column normalization, deduplication, ID assignment
- [ ] Merge rules: max scores, union keywords, longest answer
- [ ] Outputs: `unified_pairs.parquet`, `unique_*.json` files

### Task 2: Cluster Parser
- [ ] `parse_clusters.py` - Parse cluster JSON format
- [ ] Keyword matching (exact + cosine fallback)
- [ ] Outputs: Cluster mappings and labels

### Task 3: Source-Cluster Linker
- [ ] `link_sources.py` - Link sources to clusters via keywords
- [ ] Primary cluster resolution
- [ ] Outputs: `pair_cluster_id.pt`, `pair_relevance.pt`

### Task 4: Embedding Generator
- [ ] `embed.py` - Batch embedding generation
- [ ] Sanity checks: centroid similarity, no NaN/Inf
- [ ] Outputs: `*_embs.pt` files

### Task 5: Steering Builder
- [ ] `build_steering.py` - Generate steering signals
- [ ] Variants: centroid, keyword-weighted, residual
- [ ] Outputs: `steering_*.pt`, `centroid_distances.pt`

### Task 6: Hard Negative Miner
- [ ] `mine_negatives.py` - 4-tier negative sampling
- [ ] Tiers: in-cluster, adjacent, high-similarity, random
- [ ] Outputs: `hard_negatives.pt`, `negative_tiers.pt`

### Task 7: Splitter
- [ ] `split.py` - Question-level stratified splitting
- [ ] No leakage: questions stay in one split
- [ ] Outputs: `train_idx.pt`, `val_idx.pt`, `test_idx.pt`

### Task 8: Config Writer & Validation
- [ ] `finalize.py` - Cross-validate all outputs
- [ ] Check referential integrity and dimensions
- [ ] Output: Final `config.json`

### Task 9: Build Orchestrator
- [ ] `build.py` - Main entry point
- [ ] Run Tasks 1-8 in sequence
- [ ] Per-task timing and logging

## Expected Input Format

### CSV Format

Columns (flexible names via alias map):
- `question` / `question_text` (required)
- `source` / `source_text` (required)
- `answer` / `answer_text` (optional)
- `keywords` (optional, JSON list or comma-separated)
- `relevance_score` / `score` (optional, float 0-1)

### Cluster JSON Format

Expected structure (from `keyword_clustering.py`):
```json
{
  "metadata": {
    "n_clusters": 20,
    "embedding_dim": 384
  },
  "cluster_assignments": { "keyword": cluster_id },
  "clusters": {
    "0": { "keywords": [...], "centroid": [...] }
  },
  "embeddings": { "keyword": [embedding] }
}
```

## Expected Output Structure

```
output_dir/
├── config.json
├── unified_pairs.parquet
├── unique_questions.json
├── unique_sources.json
├── unique_keywords.json
├── keyword_to_cluster.json
├── cluster_to_keywords.json
├── cluster_labels.json
├── source_to_keywords.json
├── source_to_clusters.json
├── question_embs.pt
├── source_embs.pt
├── keyword_embs.pt
├── centroid_embs.pt
├── pair_index.pt
├── pair_cluster_id.pt
├── pair_relevance.pt
├── pair_keyword_ids.pt
├── steering_centroid.pt
├── steering_keyword_weighted.pt
├── steering_residual.pt
├── centroid_distances.pt
├── hard_negatives.pt
├── negative_tiers.pt
├── train_idx.pt
├── val_idx.pt
└── test_idx.pt
```

## Usage (Planned)

```python
from RAG_supporters.dataset.dataset_builder import build_dataset

# Build dataset
build_dataset(
    csv_paths=["data1.csv", "data2.csv"],
    cluster_json_path="clusters.json",
    embedding_model=model,
    output_dir="./my_dataset",
    config={
        "n_neg": 12,
        "split_ratios": [0.8, 0.1, 0.1],
        "steering_probs": {"zero": 0.25, "centroid": 0.25, "keyword": 0.25, "residual": 0.25},
    }
)

# Use dataset
from RAG_supporters.dataset import create_loader
loader = create_loader("./my_dataset", split="train", batch_size=32)
```

## Success Criteria

All success criteria from problem statement apply:
- CSV merge handles duplicates and missing columns
- Cluster parsing matches keywords with fallback
- Every source has ≥1 cluster, one primary cluster
- Embeddings aligned by ID, same dimension
- Steering tensors validated (unit norm, weight sum to 1)
- Hard negatives: true source never in own negative set
- Splits: no question leakage, all clusters represented
- Config is single source of truth

## Implementation Notes

### Dependencies
- `pandas` for CSV handling
- `torch` for tensor operations
- `numpy` for numerical operations
- `sentence-transformers` or LangChain embeddings
- `scikit-learn` for stratified splitting
- `tqdm` for progress bars

### Design Principles
1. **Each task is independent**: Can be tested and debugged separately
2. **Fail fast**: Validate inputs at each step
3. **Graceful degradation**: Handle missing optional data
4. **Deterministic**: Same inputs + seed → identical outputs
5. **Memory efficient**: Process in batches where possible

### Testing Strategy
Create `tests/test_dataset_build.py` with:
- Unit tests for each task module
- Integration test for full pipeline
- Validate output against success criteria
- Test edge cases (empty clusters, missing keywords, etc.)

## Related Files

- **Runtime**: `RAG_supporters/dataset/jasper_steering_dataset.py` ✅ Implemented
- **Loader**: `RAG_supporters/dataset/loader.py` ✅ Implemented
- **Tests**: `tests/test_jasper_steering_dataset.py`, `tests/test_loader.py` ✅ Implemented
- **Docs**: `docs/dataset/JASPER_STEERING_DATASET.md` ✅ Implemented

## References

- Problem statement in issue description
- Existing utilities:
  - `RAG_supporters/clustering/clustering_data.py` for cluster JSON parsing
  - `RAG_supporters/embeddings/keyword_embedder.py` for embedding generation
  - `RAG_supporters/dataset/dataset_splitter.py` for splitting patterns
