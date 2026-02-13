# Clustering Operations

Documentation for clustering and keyword matching components.

---

## ClusterParser

**Location**: `RAG_supporters/clustering_ops/parse_clusters.py:27`

**Purpose**: Parses cluster JSON with keyword matching (exact + cosine fallback).

### Features

- Extracts keyword texts, embeddings, cluster IDs, centroids
- Exact match first, cosine fallback ≥ threshold
- Assigns `kw_idx` to unique keywords
- Generates cluster labels (nearest keyword to centroid)

### Usage

```python
from RAG_supporters.clustering_ops import ClusterParser

parser = ClusterParser("clusters.json", similarity_threshold=0.92)
cluster_data = parser.parse()  # Returns ClusterData object
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `cluster_json_path` | `str` | Path to cluster JSON file |
| `similarity_threshold` | `float` | Cosine threshold for fallback matching (default: 0.92) |
| `csv_keywords` | `List[str]` | Additional keywords from CSV (optional) |

### Input Format

Expected JSON structure:
```json
{
  "metadata": {
    "n_clusters": 20,
    "embedding_dim": 384
  },
  "cluster_assignments": {
    "keyword1": 0,
    "keyword2": 1
  },
  "clusters": {
    "0": {
      "keywords": ["keyword1", "keyword3"],
      "centroid": [0.1, 0.2, ...]
    }
  },
  "embeddings": {
    "keyword1": [0.1, 0.2, ...],
    "keyword2": [0.3, 0.4, ...]
  }
}
```

### Output Files

- `unique_keywords.json` - Keyword ID → text mapping
- `keyword_to_cluster.json` - Keyword → cluster mapping
- `cluster_to_keywords.json` - Cluster → keywords mapping
- `cluster_labels.json` - Human-readable cluster labels

### Methods

#### `parse() -> ClusterData`

Parses cluster JSON and returns structured data.

**Returns**: `ClusterData` object with:
- `n_clusters`: Number of clusters
- `embedding_dim`: Embedding dimension
- `keyword_to_cluster`: Dict mapping keywords to clusters
- `cluster_to_keywords`: Dict mapping clusters to keyword lists
- `cluster_labels`: Dict mapping clusters to human-readable labels
- `keyword_embeddings`: Tensor of keyword embeddings
- `centroid_embeddings`: Tensor of centroid embeddings

#### `save(output_dir: Path)`

Saves parsed data to directory.

### Keyword Matching

1. **Exact Match**: Direct string match (case-insensitive)
2. **Cosine Fallback**: If no exact match, finds closest embedding with cosine ≥ threshold
3. **New Keywords**: If neither works, adds as new keyword with new ID

### Cluster Labeling

For each cluster:
1. Compute centroid embedding
2. Find keyword with highest cosine similarity to centroid
3. Use that keyword as cluster label

---

## SourceClusterLinker

**Location**: `RAG_supporters/clustering_ops/link_sources.py:29`

**Purpose**: Links question-source pairs to clusters via keywords.

### Process

1. Collects keywords for each unique source
2. Maps keywords → clusters
3. Resolves primary cluster per source (most keywords, tie-break by relevance)
4. Assigns `pair_cluster_id` from source's primary cluster

### Usage

```python
from RAG_supporters.clustering_ops import SourceClusterLinker

linker = SourceClusterLinker(
    unified_pairs_df, keyword_mappings, cluster_data
)
pair_clusters, source_clusters = linker.link()
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `unified_pairs` | `pd.DataFrame` | Merged pairs with questions, sources, keywords |
| `keyword_mappings` | `Dict` | Keyword → cluster mapping from ClusterParser |
| `cluster_data` | `ClusterData` | Parsed cluster data |
| `fallback_strategy` | `str` | How to handle sources with no keywords (default: "nearest") |

### Methods

#### `link() -> Tuple[torch.Tensor, Dict]`

Links sources to clusters.

**Returns**:
- `pair_cluster_ids`: Tensor `[n_pairs]` with cluster ID per pair
- `source_clusters`: Dict mapping `s_idx` → cluster ID

#### `save(output_dir: Path)`

Saves link data to directory.

**Saves**:
- `source_to_keywords.json` - Source → keywords mapping
- `source_to_clusters.json` - Source → clusters mapping
- `pair_cluster_id.pt` - Cluster ID per pair

### Primary Cluster Resolution

For each source:
1. Collect all keywords from pairs referencing it
2. Map keywords to clusters
3. Count keywords per cluster
4. Select cluster with most keywords
5. If tie, select based on average relevance score
6. If still tied, select lowest cluster ID (deterministic)

### Fallback Strategy

For sources with no keywords:

**Nearest** (default): Compute embedding similarity to all centroids, assign to nearest

**Random**: Randomly assign to a cluster

**None**: Leave unassigned (will error if used in training)

---

## Related Documentation

- [Data Preparation](DATA_PREPARATION.md) - CSV merger for getting pairs
- [JASPER Builder](JASPER_BUILDER.md) - Complete pipeline
- [Contrastive Learning](CONTRASTIVE_LEARNING.md) - Uses cluster assignments for negatives
- [JASPER Builder Guide](dataset/JASPER_BUILDER_GUIDE.md) - User guide
