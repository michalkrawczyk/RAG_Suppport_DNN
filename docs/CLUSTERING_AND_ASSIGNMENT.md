# Clustering Foundation and Source Assignment

## Overview

This module implements **Phase 1 & 2** of the subspace/cluster steering roadmap:

- **Phase 1: Clustering Foundation** - Discover topics/subspaces by clustering suggestion embeddings
- **Phase 2: Source Assignment** - Assign sources/questions to discovered clusters/subspaces

## Phase 1: Clustering Foundation

Phase 1 focuses on discovering topics or subspaces from suggestion embeddings using clustering algorithms.

### Key Features

- **K-means and Bisecting K-means** clustering algorithms
- **Topic descriptor extraction** - identify the n closest suggestions to each cluster centroid
- **Modular and extensible** design
- **Results persistence** - save and load clustering results with topic information

### Basic Usage

#### Cluster Suggestions from Embeddings

```python
from RAG_supporters.embeddings import KeywordEmbedder
from RAG_supporters.clustering import cluster_suggestions_from_embeddings

# Step 1: Create embeddings for suggestions
embedder = KeywordEmbedder()
suggestions = [
    "machine learning algorithms",
    "deep neural networks",
    "supervised learning",
    "natural language processing",
    "computer vision",
    # ... more suggestions
]
suggestion_embeddings = embedder.create_embeddings(suggestions)

# Step 2: Cluster suggestions to discover topics
clusterer, topics = cluster_suggestions_from_embeddings(
    suggestion_embeddings,
    n_clusters=5,              # Number of topics to discover
    algorithm="kmeans",         # or "bisecting_kmeans"
    n_descriptors=10,          # Top 10 suggestions per topic
    output_path="results/suggestion_clusters.json",
    random_state=42,
)

# Step 3: Explore discovered topics
for topic_id, descriptors in topics.items():
    print(f"\nTopic {topic_id}:")
    for desc in descriptors[:5]:
        print(f"  - {desc}")
```

#### Load Suggestions from CSV

If you have suggestions in a CSV file (from LLM output):

```python
from RAG_supporters.embeddings import KeywordEmbedder
from RAG_supporters.clustering import cluster_suggestions_from_embeddings

# Process CSV to create embeddings
embedder = KeywordEmbedder()
suggestion_embeddings = embedder.process_csv_to_embeddings(
    csv_path="data/llm_suggestions.csv",
    output_path="results/suggestion_embeddings.json",
    min_confidence=0.7,           # Filter by confidence
    suggestion_column="suggestions",  # Column name in CSV
)

# Cluster the suggestions
clusterer, topics = cluster_suggestions_from_embeddings(
    suggestion_embeddings,
    n_clusters=8,
    n_descriptors=15,
    output_path="results/suggestion_clusters.json",
)
```

### Using the SuggestionClusterer Class

For more control, use the `SuggestionClusterer` class directly:

```python
from RAG_supporters.clustering import SuggestionClusterer

# Initialize clusterer
clusterer = SuggestionClusterer(
    algorithm="bisecting_kmeans",
    n_clusters=10,
    random_state=42,
)

# Fit on suggestion embeddings
clusterer.fit(suggestion_embeddings)

# Extract topic descriptors
topics = clusterer.extract_topic_descriptors(
    n_descriptors=15,
    metric="cosine",  # or "euclidean"
)

# Get cluster assignments
assignments = clusterer.get_cluster_assignments()
print(f"Suggestion 'machine learning' assigned to cluster: {assignments['machine learning']}")

# Get all suggestions in a cluster
clusters = clusterer.get_clusters()
print(f"Cluster 0 contains {len(clusters[0])} suggestions")

# Save results
clusterer.save_results(
    "results/suggestion_clusters.json",
    include_topics=True,
    include_embeddings=False,  # Set to True to include embeddings
)
```

### Loading Saved Results

```python
from RAG_supporters.clustering import SuggestionClusterer

# Load from saved results
clusterer = SuggestionClusterer.from_results("results/suggestion_clusters.json")

# Use the loaded clusterer
topics = clusterer.topics
print(f"Loaded {len(topics)} topics")
```

### Output Format

The clustering results JSON includes:

```json
{
  "metadata": {
    "algorithm": "kmeans",
    "n_clusters": 5,
    "n_suggestions": 100,
    "random_state": 42
  },
  "cluster_assignments": {
    "machine learning": 0,
    "deep learning": 0,
    "NLP": 1,
    ...
  },
  "clusters": {
    "0": ["machine learning", "deep learning", ...],
    "1": ["NLP", "text processing", ...],
    ...
  },
  "cluster_stats": {
    "0": {
      "size": 25,
      "suggestions_sample": ["...", "..."],
      "topic_descriptors": ["...", "...", ...]
    },
    ...
  },
  "centroids": [[...], [...], ...]
}
```

## Phase 2: Source Assignment

Phase 2 assigns sources (or questions) to the discovered clusters/subspaces based on their embeddings.

### Key Features

- **Hard (one-hot) assignment** - assign each source to a single cluster
- **Soft (probabilistic) assignment** - assign sources to multiple clusters with probabilities
- **Multi-subspace membership** - sources can belong to multiple topics
- **Threshold filtering** - control assignment sensitivity
- **Temperature-scaled softmax** - adjust probability distribution sharpness

### Basic Usage

#### Assign Sources to Clusters

```python
from RAG_supporters.embeddings import KeywordEmbedder
from RAG_supporters.clustering import (
    SuggestionClusterer,
    assign_sources_to_clusters,
)

# Step 1: Load clustering results from Phase 1
clusterer = SuggestionClusterer.from_results("results/suggestion_clusters.json")
centroids = clusterer.clusterer.get_centroids()

# Step 2: Create embeddings for your sources/questions
embedder = KeywordEmbedder()
sources = {
    "source_1": "A detailed article about machine learning algorithms and their applications",
    "source_2": "Research paper on natural language processing techniques",
    "source_3": "Tutorial on computer vision and image recognition",
    # ... more sources
}

# Embed source texts
source_texts = list(sources.values())
source_text_embeddings = embedder.create_embeddings(source_texts)

# Map back to source IDs
source_embeddings = {
    source_id: source_text_embeddings[text]
    for source_id, text in sources.items()
}

# Step 3: Assign sources to clusters
assignments = assign_sources_to_clusters(
    source_embeddings,
    centroids,
    assignment_mode="soft",    # "hard" or "soft"
    threshold=0.1,             # Include clusters with prob > 0.1
    temperature=1.0,           # Softmax temperature
    metric="cosine",           # or "euclidean"
    output_path="results/source_assignments.json",
)

# Step 4: Explore assignments
for source_id, assignment in assignments.items():
    print(f"\n{source_id}:")
    print(f"  Primary cluster: {assignment['primary_cluster']}")
    print(f"  All clusters: {assignment['clusters']}")
    if 'probabilities' in assignment:
        top_clusters = sorted(
            assignment['probabilities'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        print(f"  Top 3 probabilities: {top_clusters}")
```

### Assignment Modes

#### Hard Assignment (One-Hot)

Assigns each source to exactly one cluster:

```python
from RAG_supporters.clustering import assign_sources_to_clusters

assignments = assign_sources_to_clusters(
    source_embeddings,
    centroids,
    assignment_mode="hard",
    threshold=0.3,  # Optional: minimum probability required
    output_path="results/hard_assignments.json",
)

# Each source assigned to single cluster
for source_id, assignment in assignments.items():
    clusters = assignment['clusters']
    # clusters will be either [] (if below threshold) or [cluster_id]
    print(f"{source_id}: {clusters}")
```

#### Soft Assignment (Multi-Subspace)

Assigns sources to multiple clusters with probabilities:

```python
assignments = assign_sources_to_clusters(
    source_embeddings,
    centroids,
    assignment_mode="soft",
    threshold=0.1,       # Include clusters with probability > 0.1
    temperature=1.0,     # Adjust distribution sharpness
    output_path="results/soft_assignments.json",
)

# Sources can belong to multiple clusters
for source_id, assignment in assignments.items():
    print(f"{source_id}:")
    for cluster_id in assignment['clusters']:
        prob = assignment['probabilities'][cluster_id]
        print(f"  Cluster {cluster_id}: {prob:.3f}")
```

### Using the SourceAssigner Class

For more control, use the `SourceAssigner` class directly:

```python
from RAG_supporters.clustering import SourceAssigner

# Initialize assigner
assigner = SourceAssigner(
    cluster_centroids=centroids,
    assignment_mode="soft",
    threshold=0.15,
    temperature=0.8,  # Lower = more peaked distribution
    metric="cosine",
)

# Assign a single source
source_embedding = embedder.create_embeddings(["example source text"])["example source text"]
assignment = assigner.assign_source(
    source_embedding,
    return_probabilities=True,
)

print(f"Primary cluster: {assignment['primary_cluster']}")
print(f"All assigned clusters: {assignment['clusters']}")
print(f"Probabilities: {assignment['probabilities']}")

# Assign multiple sources
assignments = assigner.assign_sources_batch(source_embeddings)

# Save results with custom metadata
metadata = {
    "dataset": "my_dataset",
    "date": "2025-12-06",
}
assigner.save_assignments(
    assignments,
    "results/source_assignments.json",
    metadata=metadata,
)
```

### Temperature Parameter

The temperature parameter controls the sharpness of the probability distribution:

```python
# High temperature (e.g., 2.0) -> more uniform distribution
# Sources assigned to more clusters
assignments_uniform = assign_sources_to_clusters(
    source_embeddings,
    centroids,
    assignment_mode="soft",
    temperature=2.0,
)

# Low temperature (e.g., 0.5) -> peaked distribution
# Sources concentrated on fewer clusters
assignments_peaked = assign_sources_to_clusters(
    source_embeddings,
    centroids,
    assignment_mode="soft",
    temperature=0.5,
)
```

### Output Format

The assignment results JSON includes:

```json
{
  "metadata": {
    "assignment_mode": "soft",
    "threshold": 0.1,
    "temperature": 1.0,
    "metric": "cosine",
    "n_clusters": 5
  },
  "statistics": {
    "total_sources": 50,
    "unassigned_sources": 2,
    "multi_cluster_sources": 35,
    "cluster_counts": {
      "0": 25,
      "1": 18,
      ...
    }
  },
  "assignments": {
    "source_1": {
      "mode": "soft",
      "primary_cluster": 0,
      "clusters": [0, 2, 3],
      "probabilities": {
        "0": 0.45,
        "1": 0.05,
        "2": 0.25,
        "3": 0.15,
        "4": 0.10
      }
    },
    ...
  }
}
```

## Complete Workflow Example

Here's a complete end-to-end example combining both phases:

```python
from RAG_supporters.embeddings import KeywordEmbedder
from RAG_supporters.clustering import (
    cluster_suggestions_from_embeddings,
    assign_sources_to_clusters,
    SuggestionClusterer,
)

# ========================================
# Phase 1: Cluster Suggestions
# ========================================

# Load and embed suggestions from CSV
embedder = KeywordEmbedder()
suggestion_embeddings = embedder.process_csv_to_embeddings(
    csv_path="data/llm_suggestions.csv",
    output_path="results/suggestion_embeddings.json",
    min_confidence=0.7,
)

# Cluster suggestions to discover topics
clusterer, topics = cluster_suggestions_from_embeddings(
    suggestion_embeddings,
    n_clusters=10,
    algorithm="bisecting_kmeans",
    n_descriptors=15,
    output_path="results/suggestion_clusters.json",
)

# Explore discovered topics
print("\n=== Discovered Topics ===")
for topic_id, descriptors in topics.items():
    print(f"\nTopic {topic_id}:")
    for desc in descriptors[:5]:
        print(f"  - {desc}")

# ========================================
# Phase 2: Assign Sources to Clusters
# ========================================

# Load clustering results
clusterer = SuggestionClusterer.from_results("results/suggestion_clusters.json")
centroids = clusterer.clusterer.get_centroids()

# Load and embed sources
import pandas as pd
sources_df = pd.read_csv("data/sources.csv")
source_embeddings = {}

for idx, row in sources_df.iterrows():
    source_id = row['id']
    source_text = row['text']
    
    # Embed source text
    emb = embedder.create_embeddings([source_text])[source_text]
    source_embeddings[source_id] = emb

# Assign sources to topics with soft assignment
assignments = assign_sources_to_clusters(
    source_embeddings,
    centroids,
    assignment_mode="soft",
    threshold=0.15,
    temperature=1.0,
    output_path="results/source_assignments.json",
)

# Analyze assignments
print("\n=== Source Assignments ===")
for source_id, assignment in list(assignments.items())[:5]:
    print(f"\n{source_id}:")
    print(f"  Primary topic: {assignment['primary_cluster']}")
    print(f"  Member of topics: {assignment['clusters']}")
    
    # Show top probabilities
    probs = assignment['probabilities']
    top_3 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"  Top 3: {[(t, f'{p:.3f}') for t, p in top_3]}")

print("\n=== Complete! ===")
print("Results saved to:")
print("  - results/suggestion_embeddings.json")
print("  - results/suggestion_clusters.json")
print("  - results/source_assignments.json")
```

## API Reference

### SuggestionClusterer

```python
class SuggestionClusterer(algorithm='kmeans', n_clusters=8, random_state=42, **kwargs)
```

**Methods:**
- `fit(suggestion_embeddings)` - Fit clustering model
- `extract_topic_descriptors(n_descriptors=10, metric='euclidean')` - Extract topic descriptors
- `get_cluster_assignments()` - Get suggestion-to-cluster mapping
- `get_clusters()` - Get clusters with suggestions
- `save_results(output_path, include_embeddings=False, include_topics=True)` - Save results
- `load_results(input_path)` - Load results (static method)
- `from_results(clustering_results_path)` - Create from saved results (class method)

### SourceAssigner

```python
class SourceAssigner(cluster_centroids, assignment_mode='soft', threshold=None, 
                     temperature=1.0, metric='cosine')
```

**Methods:**
- `assign_source(embedding, return_probabilities=True)` - Assign single source
- `assign_sources_batch(source_embeddings, return_probabilities=True)` - Assign multiple sources
- `compute_probabilities(embedding)` - Get probability distribution
- `compute_distances(embedding)` - Get distances to centroids
- `save_assignments(assignments, output_path, metadata=None)` - Save results
- `load_assignments(input_path)` - Load results (static method)

### Convenience Functions

```python
cluster_suggestions_from_embeddings(
    suggestion_embeddings,
    n_clusters=8,
    algorithm='kmeans',
    n_descriptors=10,
    output_path=None,
    random_state=42,
    **kwargs
)
```

```python
assign_sources_to_clusters(
    source_embeddings,
    cluster_centroids,
    assignment_mode='soft',
    threshold=None,
    temperature=1.0,
    metric='cosine',
    output_path=None,
    metadata=None
)
```

## Best Practices

1. **Choose appropriate number of clusters**: Start with a reasonable estimate (e.g., 5-15) and adjust based on topic coherence

2. **Filter suggestions by confidence**: Use `min_confidence` when processing CSV to ensure quality

3. **Use cosine similarity for text**: Cosine metric works better for text embeddings than Euclidean distance

4. **Adjust temperature for assignment**: 
   - Lower (0.5-0.8) for more focused assignments
   - Higher (1.2-2.0) for broader multi-topic membership

5. **Set appropriate thresholds**:
   - Hard mode: 0.3-0.5 ensures confident assignments
   - Soft mode: 0.1-0.2 includes relevant secondary clusters

6. **Save intermediate results**: Save embeddings, clusters, and assignments separately for flexibility

## Troubleshooting

**Issue: Too many/few clusters**
- Adjust `n_clusters` parameter
- Use `bisecting_kmeans` for hierarchical structure

**Issue: Topics not coherent**
- Increase `min_confidence` when filtering suggestions
- Try different clustering algorithms
- Adjust number of clusters

**Issue: All sources assigned to one cluster**
- Increase `temperature` to spread assignments
- Lower `threshold` to include more clusters
- Check if source embeddings are diverse

**Issue: No sources assigned (empty clusters)**
- Lower `threshold` parameter
- Check embedding quality
- Verify centroids are loaded correctly

## See Also

- **KeywordClusterer**: Base clustering functionality
- **KeywordEmbedder**: Embedding creation and CSV processing
- **suggestion_processing**: Suggestion filtering and aggregation utilities
