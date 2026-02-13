# Dataset Builder Pipeline

> **Status:** ✅ PT pipeline implemented (Tasks 0-9). HDF5 output remains optional/future.

This directory will contain the complete pipeline for building JASPER Steering Datasets from CSV files and cluster JSON.

## Overview

The dataset builder processes:
- Multiple CSV files with question-source pairs (many-to-many relationships supported)
- Cluster JSON with keyword embeddings and centroids
- Embedding model for text encoding

And produces a single self-contained dataset directory ready for training.

**Key Features:**
- ✅ **Many-to-Many Support**: Questions can have multiple sources; sources can answer multiple questions
- ✅ **Smart Deduplication**: Only exact duplicate pairs are merged (preserves multi-source questions)
- ✅ **Flexible Column Names**: Automatic detection of question/source/answer columns via aliases
- ✅ **Keyword Clustering**: Links pairs to semantic clusters for curriculum learning
- ✅ **Hard Negatives**: Pre-computed 4-tier negative sampling for contrastive learning
- ✅ **Curriculum Ready**: Centroid distances and steering signals for progressive training

## Pipeline Tasks

### Task 0: Scaffold ✅ DONE
- [x] `__init__.py` - Package constants and types (updated with new exports)
- [x] `builder_config.py` - BuildConfig dataclass with serialization

### Task 1: CSV Merger ✅ DONE
- [x] `merge_csv.py` - Column normalization, deduplication, ID assignment
- [x] **Many-to-many support**: One question can have multiple sources, one source can answer multiple questions
- [x] **Deduplication**: Only exact duplicate pairs (same question + same source) are merged
- [x] Merge rules for duplicates: max scores, union keywords, longest answer
- [x] Outputs: `inspection.json` (optional human-readable metadata)
- [x] Tests: `tests/test_merge_csv.py` with comprehensive coverage including many-to-many validation

### Task 2: Cluster Parser ✅ DONE
- [x] `parse_clusters.py` - Parse KeywordClusterer JSON format
- [x] Load cluster metadata: assignments, centroids, topic_descriptors, embeddings
- [x] Keyword matching (exact + cosine fallback)
- [x] Outputs: Reference to KeywordClusterer JSON stored in config.json
- [x] Tests: `tests/test_parse_clusters.py` with comprehensive coverage

### Task 3: Source-Cluster Linker ✅ DONE
- [x] `link_sources.py` - Link sources to clusters via keywords from KeywordClusterer JSON
- [x] Primary cluster resolution using cluster memberships
- [x] Outputs: Pair-level cluster assignments (stored in dataset tensors)
- [x] Tests: `tests/test_link_sources.py` with comprehensive coverage

### Task 4: Embedding Generator ✅ DONE
- [x] `embed.py` - Batch embedding generation
- [x] Sanity checks: centroid similarity, no NaN/Inf
- [x] Outputs: `*_embs.pt` files
- [x] Tests: `tests/test_embed.py` with comprehensive coverage

### Task 5: Steering Builder ✅ DONE
- [x] `build_steering.py` - Generate steering signals
- [x] Variants: centroid, keyword-weighted, residual
- [x] Outputs: `steering_*.pt`, `centroid_distances.pt`
- [x] Tests: `tests/test_build_steering.py` with comprehensive coverage

### Task 6: Hard Negative Miner ✅ DONE
- [x] `mine_negatives.py` - 4-tier negative sampling
- [x] Tiers: in-cluster, adjacent, high-similarity, random
- [x] Outputs: `hard_negatives.pt`, `negative_tiers.pt`
- [x] Tests: `tests/test_mine_negatives.py` with comprehensive coverage

### Task 7: Splitter ✅ DONE
- [x] `split.py` - Question-level stratified splitting
- [x] No leakage: questions stay in one split
- [x] Outputs: `train_idx.pt`, `val_idx.pt`, `test_idx.pt`
- [x] Tests: `tests/test_split.py` with comprehensive coverage

### Task 8: Config Writer & Validation ✅ DONE
- [x] `finalize.py` - Cross-validate all outputs
- [x] Check referential integrity and dimensions
- [x] Output: Final `config.json`
- [x] Tests: `tests/test_finalize.py` with comprehensive coverage

### Task 9: Build Orchestrator ✅ DONE
- [x] `build.py` - Main entry point
- [x] Run Tasks 1-8 in sequence
- [x] Per-task timing and logging
- [x] Tests: `tests/test_dataset_build.py` with comprehensive coverage

### Supporting Utilities ✅ DONE
- [x] `validation_utils.py` - Shared validation functions
  - Tensor type and shape validation (2D, 1D)
  - Embedding dimension consistency checking
  - Index bounds validation for pairs and clusters
  - Length consistency validation across tensors/lists
  - Split ratio validation
  - Keyword ID list structure validation
  - **Purpose**: Eliminates ~200+ lines of duplicated validation code across builder classes
- [x] `tensor_utils.py` - Tensor I/O utilities
  - Standardized tensor loading with shape validation
  - Batch tensor loading for multiple files
  - Tensor saving with NaN/Inf checks
  - **Purpose**: Consolidates ~40 redundant torch.load operations with consistent error handling

## Expected Input Format

### CSV Format

Columns (flexible names via alias map):
- `question` / `question_text` (required)
- `source` / `source_text` (required)
- `answer` / `answer_text` (optional)
- `keywords` (optional, JSON list or comma-separated)
- `relevance_score` / `score` (optional, float 0-1)

**Many-to-Many Relationships:**
- ✅ Same question with different sources → Multiple pairs (preserved)
- ✅ Different questions with same source → Multiple pairs (preserved)
- ❌ Same question with same source → Duplicate pair (merged with max score)

**Example:**
```csv
question,source,score
"What is Python?","Python is a programming language",0.9
"What is Python?","Python is used for AI and ML",0.8  # Different source - kept
"What is Java?","Python is a programming language",0.7  # Different question - kept
"What is Python?","Python is a programming language",0.85  # Exact duplicate - merged (score=0.9 kept)
```
→ Results in 3 unique pairs

### Cluster JSON Format

Expected structure from `KeywordClusterer.save_results()` in `keyword_clustering.py`:
```json
{
  "metadata": {
    "algorithm": "kmeans",
    "n_clusters": 20,
    "n_keywords": 100,
    "embedding_dim": 384,
    "assignment_config": {...}
  },
  "cluster_assignments": { "keyword": cluster_id },
  "clusters": { "0": ["keyword1", "keyword2", ...] },
  "cluster_stats": {
    "0": {
      "size": 20,
      "topic_descriptors": ["top_keyword1", "top_keyword2", ...]
    }
  },
  "centroids": [[...], [...], ...],
  "embeddings": { "keyword": [embedding] }  // Optional
}
```

**Note:** The builder references this file rather than duplicating cluster metadata.
All keyword-to-cluster mappings, topic descriptors, and centroids come from this source.

## Expected Output Structure

### Standard Format (PyTorch Tensors)

Streamlined structure with redundant files removed:

```
output_dir/
├── config.json                      # Dataset configuration and metadata (see below)
├── inspection.json                  # [OPTIONAL] Human-readable inspection metadata
├── question_embs.pt                 # Question embeddings [N_questions, D]
├── source_embs.pt                   # Source embeddings [N_sources, D]
├── keyword_embs.pt                  # Keyword embeddings [N_keywords, D]
├── centroid_embs.pt                 # Cluster centroid embeddings [N_clusters, D]
├── pair_index.pt                    # Pair indices [N_pairs, 2] mapping to (q_idx, s_idx)
├── pair_cluster_id.pt               # Primary cluster ID per pair [N_pairs]
├── pair_relevance.pt                # Relevance scores [N_pairs] in range [0, 1]
├── pair_keyword_ids.pt              # List[List[int]] - Keyword IDs associated with each pair
├── steering_centroid.pt             # Centroid-based steering vectors [N_pairs, D]
├── steering_keyword_weighted.pt     # Keyword-weighted steering vectors [N_pairs, D]
├── steering_residual.pt             # Residual steering vectors [N_pairs, D]
├── centroid_distances.pt            # Cosine distance to cluster centroid [N_pairs]
├── hard_negatives.pt                # Hard negative source indices [N_pairs, n_neg]
├── negative_tiers.pt                # Negative difficulty tiers [N_pairs, n_neg], values 1-4
├── train_idx.pt                     # Training split indices [N_train]
├── val_idx.pt                       # Validation split indices [N_val]
└── test_idx.pt                      # Test split indices [N_test]
```

#### File Descriptions

**Core Configuration:**
- **config.json**: Single source of truth for dataset metadata. References KeywordClusterer JSON path, stores dimensions, split ratios, steering probabilities, and curriculum settings. Required for dataset loading.

**Inspection File (Optional):**
- **inspection.json**: Human-readable metadata for debugging and analysis. Contains question/source texts, keyword strings, cluster labels, and pair statistics. NOT loaded during training - for inspection only.

**Embeddings (Precomputed):**
- **question_embs.pt**: Dense embeddings for all unique questions. Indexed by question ID.
- **source_embs.pt**: Dense embeddings for all unique sources. Indexed by source ID.
- **keyword_embs.pt**: Dense embeddings for all unique keywords. Indexed by keyword ID.
- **centroid_embs.pt**: Cluster centroid embeddings. Indexed by cluster ID.

**Pair-Level Data:**
- **pair_index.pt**: Maps each training pair to (question_id, source_id). Shape [N_pairs, 2].
- **pair_cluster_id.pt**: Primary cluster assignment for each pair. Used for in-cluster negative sampling.
- **pair_relevance.pt**: Relevance score for each pair. Used for training signal or filtering.
- **pair_keyword_ids.pt**: Variable-length list of keyword IDs per pair. Saved as Python list for flexibility.

**Steering Signals:**
- **steering_centroid.pt**: Steering vector pointing to cluster centroid. Normalized to unit length.
- **steering_keyword_weighted.pt**: Weighted average of keyword embeddings. Normalized to unit length.
- **steering_residual.pt**: Residual between question and cluster centroid. Captures off-center signals.
- **centroid_distances.pt**: Cosine distance from question to cluster centroid. Used for curriculum learning.

**Hard Negatives:**
- **hard_negatives.pt**: Source indices for hard negatives [N_pairs, n_neg]. Stratified by difficulty tier.
- **negative_tiers.pt**: Tier labels (1-4) for each negative: 1=in-cluster, 2=adjacent-cluster, 3=high-similarity, 4=random.

**Dataset Splits:**
- **train_idx.pt**, **val_idx.pt**, **test_idx.pt**: Indices into pair arrays. Question-level splitting ensures no leakage.

**Why This Structure:**
- All embeddings preloaded → Zero I/O during training
- Pair-level design → Efficient multi-source questions
- Separate steering variants → Curriculum learning without recomputation
- Hard negatives precomputed → Batch construction in <1ms
- PyTorch tensors → Direct GPU transfer, no serialization overhead

**Removed redundant files** (data available in KeywordClusterer JSON):
- ❌ `unique_questions.json`, `unique_sources.json`, `unique_keywords.json`
- ❌ `keyword_to_cluster.json` → Use `cluster_assignments` from KeywordClusterer JSON
- ❌ `cluster_to_keywords.json` → Use `clusters` from KeywordClusterer JSON
- ❌ `cluster_labels.json` → Use `cluster_stats.topic_descriptors` from KeywordClusterer JSON
- ❌ `source_to_keywords.json`, `source_to_clusters.json` → Redundant with pair-level data

**config.json structure:**
```json
{
  "embedding_dim": 384,
  "n_neg": 12,
  "n_pairs": 10000,
  "n_questions": 5000,
  "n_sources": 8000,
  "n_keywords": 200,
  "n_clusters": 20,
  "clustering_source": "/path/to/keyword_clusterer_results.json",
  "steering_probabilities": {...},
  "curriculum": {...},
  "split_ratios": [0.8, 0.1, 0.1]
}
```

**inspection.json structure (optional):**
```json
{
  "metadata": {
    "created_at": "2026-02-11T10:30:00Z",
    "n_pairs": 10000,
    "n_questions": 5000,
    "n_sources": 8000,
    "clustering_source": "keyword_clusterer_results.json"
  },
  "questions": [
    {"id": 0, "text": "What is CRISPR?"},
    {"id": 1, "text": "How does photosynthesis work?"}
  ],
  "sources": [
    {"id": 0, "text": "CRISPR-Cas9 is a genome editing tool..."},
    {"id": 1, "text": "Photosynthesis converts light energy..."}
  ],
  "keywords": [
    {"id": 0, "term": "gene editing", "cluster_id": 2},
    {"id": 1, "term": "photosynthesis", "cluster_id": 5}
  ],
  "clusters": [
    {"id": 0, "label": "genomics", "size": 45},
    {"id": 1, "label": "plant biology", "size": 32}
  ],
  "pair_samples": [
    {
      "pair_id": 0,
      "question_id": 0,
      "source_id": 0,
      "cluster_id": 2,
      "relevance": 0.95,
      "keywords": ["gene editing", "CRISPR"],
      "split": "train"
    }
  ]
}
```

### HDF5 Format (Recommended for Large Datasets)

For better organization and single-file distribution:

```
output_dir/
├── config.json                      # References KeywordClusterer JSON
├── dataset.h5                       # All tensors in structured groups
│   ├── /embeddings/
│   │   ├── questions                # [N_questions, D]
│   │   ├── sources                  # [N_sources, D]
│   │   ├── keywords                 # [N_keywords, D]
│   │   └── centroids                # [N_clusters, D]
│   ├── /pairs/
│   │   ├── index                    # [N_pairs, 2]
│   │   ├── cluster_id               # [N_pairs]
│   │   ├── relevance                # [N_pairs]
│   │   └── keyword_ids              # [N_pairs] - VLen int array
│   ├── /steering/
│   │   ├── centroid                 # [N_pairs, D]
│   │   ├── keyword_weighted         # [N_pairs, D]
│   │   ├── residual                 # [N_pairs, D]
│   │   └── distances                # [N_pairs]
│   ├── /negatives/
│   │   ├── hard                     # [N_pairs, n_neg]
│   │   └── tiers                    # [N_pairs, n_neg]
│   └── /splits/
│       ├── train                    # [N_train]
│       ├── val                      # [N_val]
│       └── test                     # [N_test]
└── inspection.json                  # [OPTIONAL] Human-readable metadata
```

**HDF5 Advantages:**
- Single file (easier to version, distribute, and manage)
- Compression support (10-50% size reduction)
- Lazy loading (load only needed tensors)
- Atomic writes (no partial dataset states)
- Native support for variable-length arrays (keyword_ids)

**HDF5 Usage Example:**
```python
import h5py
import torch

# Save
with h5py.File(output_dir / "dataset.h5", "w") as f:
    emb_group = f.create_group("embeddings")
    emb_group.create_dataset("questions", data=question_embs.numpy(), compression="gzip")
    # ... other datasets

# Load in JASPERSteeringDataset
with h5py.File(self.dataset_dir / "dataset.h5", "r") as f:
    self.question_embs = torch.from_numpy(f["embeddings/questions"][:])
```

## Usage (Planned)

```python
from RAG_supporters.dataset import build_dataset

# Build dataset
build_dataset(
  csv_paths=["data1.csv", "data2.csv"],
  cluster_json_path="clusters.json",  # From KeywordClusterer.save_results()
  embedding_model=model,
  output_dir="./my_dataset",
  storage_format="pt",
  include_inspection_file=True,
  config={
    "n_neg": 12,
    "split_ratios": [0.8, 0.1, 0.1],
    "steering_probabilities": {
      "zero": 0.25,
      "centroid": 0.25,
      "keyword": 0.25,
      "residual": 0.25,
    },
  },
)

# Use dataset
from RAG_supporters.dataset import create_loader
loader = create_loader("./my_dataset", split="train", batch_size=32)
```

## Success Criteria

All success criteria from problem statement apply:
- CSV merge handles duplicates and missing columns
- **Many-to-many relationships preserved**: questions can have multiple sources, sources can answer multiple questions
- **Deduplication**: only exact duplicate pairs (same question + same source) are merged with max score
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
- `h5py` (optional) for HDF5 format support
- `json` (standard library) for inspection file

### Design Principles
1. **Each task is independent**: Can be tested and debugged separately
2. **Fail fast**: Validate inputs at each step
3. **Graceful degradation**: Handle missing optional data
4. **Deterministic**: Same inputs + seed → identical outputs
5. **Memory efficient**: Process in batches where possible
6. **DRY (Don't Repeat Yourself)**: Shared utilities for validation and I/O eliminate code duplication

### Code Architecture

**Shared Utilities** (used by all builder classes):
- `validation_utils.py` - Common validation functions to ensure consistency and reduce duplication
  - Validates tensor types, shapes, and dimensions
  - Checks index bounds and length consistency
  - Verifies split ratios and keyword ID structures
  - Provides clear, standardized error messages
- `tensor_utils.py` - Tensor loading/saving with automatic validation
  - Loads tensors with optional shape validation
  - Batch loading for multiple tensors
  - Checks for NaN/Inf values before saving
  - Consistent error handling across all I/O operations

**Builder Classes** (leverage shared utilities):
- `SteeringBuilder`, `NegativeMiner`, `DatasetSplitter` - Use validation_utils for input checking
- `JASPERSteeringDataset`, `DatasetFinalizer` - Use tensor_utils for I/O operations
- Benefits: Single source of truth for validation logic, easier maintenance, consistent behavior

### Testing Strategy
Create `tests/test_dataset_build.py` with:
- Unit tests for each task module
- Integration test for full pipeline
- Validate output against success criteria
- Test edge cases (empty clusters, missing keywords, etc.)

## Related Files

### Core Components
- **Runtime**: `RAG_supporters/pytorch_datasets/jasper_steering_dataset.py` ✅ Implemented
- **Loader**: `RAG_supporters/pytorch_datasets/loader.py` ✅ Implemented (DataLoader factory for JASPERSteeringDataset)
- **Builder**: `RAG_supporters/jasper/build.py` ✅ Implemented (moved to jasper/ module)
- **Finalizer**: `RAG_supporters/jasper/finalize.py` ✅ Implemented (moved to jasper/ module)

### Shared Utilities (New)
- **Validation**: `RAG_supporters/dataset/validation_utils.py` ✅ Implemented
- **Tensor I/O**: `RAG_supporters/dataset/tensor_utils.py` ✅ Implemented

### Tests
- **Dataset Tests**: `tests/test_jasper_steering_dataset.py` ✅ Implemented
- **Loader Tests**: `tests/test_loader.py` ✅ Implemented
- **Builder Tests**: `tests/test_dataset_build.py`, `tests/test_build_steering.py`, `tests/test_mine_negatives.py`, `tests/test_split.py`, `tests/test_finalize.py` ✅ Implemented

### Documentation
- **Usage Guide**: `docs/dataset/JASPER_STEERING_DATASET.md` ✅ Implemented
- **Training Example**: `docs/dataset/JASPER_TRAINING_EXAMPLE.md` ✅ Implemented
- **Project Structure**: `agents_notes/PROJECT_STRUCTURE.md` ✅ Updated with utilities

## References

- Problem statement in issue description
- Existing utilities:
  - `RAG_supporters/clustering/clustering_data.py` for cluster JSON parsing
  - `RAG_supporters/embeddings/keyword_embedder.py` for embedding generation
  - `RAG_supporters/dataset/dataset_splitter.py` for splitting patterns
