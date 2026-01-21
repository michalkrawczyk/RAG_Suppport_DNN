# Topic Distance Calculator Utility

## Overview

The `topic_distance_calculator` utility provides functionality to calculate embedding distances between text and topic keywords from KeywordClusterer results. Unlike the `TOPIC_RELEVANCE_PROB` agent method which uses LLM-based probabilistic assessment, this utility performs direct embedding distance calculations without LLM usage.

## Key Features

- **Direct Embedding Distance Calculation**: Computes distances using cosine or euclidean metrics
- **CSV Processing**: Batch process CSV files with question_text and source_text fields
- **Flexible Data Sources**: 
  - Embed text directly using an embedder
  - Fetch embeddings from database by ID (source_id, question_id)
- **KeywordClusterer Integration**: Uses topic descriptors and centroids from KeywordClusterer JSON
- **Separate Result Columns**: Generates different result columns for questions and sources

## Installation

The utility requires the following dependencies:

```bash
pip install numpy pandas scikit-learn tqdm
```

## Usage

### Basic Usage with Text Embedding

```python
from RAG_supporters.embeddings import KeywordEmbedder
from RAG_supporters.clustering.topic_distance_calculator import calculate_topic_distances_from_csv

# Initialize embedder
embedder = KeywordEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Process CSV file
result_df = calculate_topic_distances_from_csv(
    csv_path="data.csv",
    keyword_clusterer_json="clusters.json",
    embedder=embedder,
    question_col="question_text",
    source_col="source_text",
    metric="cosine",
    output_path="results.csv"
)
```

### Using Database IDs

If your CSV contains IDs for questions and sources, you can fetch embeddings directly from the database:

```python
from RAG_supporters.clustering.topic_distance_calculator import calculate_topic_distances_from_csv

# Process CSV with database lookup
result_df = calculate_topic_distances_from_csv(
    csv_path="data.csv",
    keyword_clusterer_json="clusters.json",
    question_id_col="question_id",
    source_id_col="source_id",
    database=my_database,  # Database object with get_embedding method
    metric="cosine",
    output_path="results.csv"
)
```

### Using the TopicDistanceCalculator Class

For more control, use the `TopicDistanceCalculator` class directly:

```python
from RAG_supporters.clustering.topic_distance_calculator import TopicDistanceCalculator
from RAG_supporters.embeddings import KeywordEmbedder

# Initialize calculator
calculator = TopicDistanceCalculator(
    keyword_clusterer_json="clusters.json",
    embedder=KeywordEmbedder(),
    metric="cosine"
)

# Process CSV
result_df = calculator.calculate_distances_for_csv(
    csv_path="data.csv",
    question_col="question_text",
    source_col="source_text",
    output_path="results.csv",
    show_progress=True
)
```

## Input Requirements

### CSV File Format

The input CSV must contain at least one of the following column combinations:

1. **Text columns** (for direct embedding):
   - `question_text`: Text of the question
   - `source_text`: Text of the source/answer

2. **ID columns** (for database lookup):
   - `question_id`: ID to fetch question embedding from database
   - `source_id`: ID to fetch source embedding from database

Example CSV:

```csv
question_text,source_text
"What is machine learning?","Machine learning is a subset of AI that enables computers to learn from data."
"How do databases work?","Databases store and manage structured data using tables and relationships."
```

### KeywordClusterer JSON Format

The KeywordClusterer JSON must contain:

- `centroids`: Array of cluster centroid vectors
- `cluster_stats`: Dictionary mapping cluster IDs to statistics including `topic_descriptors`

Example structure:

```json
{
  "metadata": {
    "n_clusters": 3,
    "embedding_dim": 384
  },
  "centroids": [
    [0.1, 0.2, ...],
    [0.3, 0.4, ...],
    [0.5, 0.6, ...]
  ],
  "cluster_stats": {
    "0": {
      "topic_descriptors": ["machine learning", "AI", "neural networks"],
      "size": 50
    },
    "1": {
      "topic_descriptors": ["database", "SQL", "storage"],
      "size": 30
    },
    "2": {
      "topic_descriptors": ["web development", "frontend", "javascript"],
      "size": 40
    }
  }
}
```

## Output Format

The utility adds the following columns to the input DataFrame:

### Question Distance Columns

- `question_term_distance_scores`: **JSON mapping {topic_descriptor: distance}** - Similar to TOPIC_RELEVANCE_PROB's format

### Source Distance Columns

- `source_term_distance_scores`: **JSON mapping {topic_descriptor: distance}** - Similar to TOPIC_RELEVANCE_PROB's format

Example output row (simplified for readability):

| Column | Example Value |
|--------|---------------|
| question_text | "What is ML?" |
| source_text | "ML is AI subset" |
| **question_term_distance_scores** | **{"machine learning": 0.12, "AI": 0.12, "databases": 0.45}** |
| **source_term_distance_scores** | **{"machine learning": 0.15, "AI": 0.15, "databases": 0.52}** |

## Distance Metrics

The utility supports two distance metrics:

### Cosine Distance (Recommended)

- Measures angular distance between vectors
- Range: [0, 2], where 0 is identical and 2 is opposite
- Best for text embeddings as it's invariant to magnitude
- Formula: `distance = 1 - cosine_similarity`

### Euclidean Distance

- Measures straight-line distance in embedding space
- Range: [0, ∞)
- Sensitive to magnitude differences
- Formula: `distance = sqrt(sum((a - b)^2))`

## Comparison with TOPIC_RELEVANCE_PROB

| Feature | TopicDistanceCalculator | TOPIC_RELEVANCE_PROB Agent |
|---------|------------------------|---------------------------|
| **Method** | Direct embedding distance | LLM-based probabilistic assessment |
| **Speed** | Fast (no LLM calls) | Slower (requires LLM) |
| **Cost** | Low (only embedding) | Higher (LLM API costs) |
| **Output Format** | Distance values (0-2 for cosine) + **JSON mapping** | Probability scores (0-1) + **JSON mapping** |
| **JSON Mapping** | `{topic: distance}` (both questions & sources) | `{topic: probability}` (questions only) |
| **Interpretability** | Numeric distances | Natural language explanations + probability |
| **Reasoning** | No reasoning provided | Optional reasoning per topic |
| **Scope** | Questions **and** sources | Questions only |
| **Use Case** | Large-scale batch processing | In-depth semantic analysis |

## Database Interface Requirements

If using database ID lookup, your database object must implement one of:

1. **Custom interface**: `get_embedding(item_id, collection)` method
2. **ChromaDB interface**: `get(ids=[...], include=["embeddings"])` method

Example database implementation:

```python
class MyDatabase:
    def get_embedding(self, item_id, collection="questions"):
        """Fetch embedding by ID from specified collection."""
        # Your implementation here
        return embedding_vector  # numpy array
```

## Performance Considerations

- **Batch Processing**: The utility processes rows sequentially. For very large CSV files, consider splitting into chunks.
- **Memory Usage**: Embeddings are computed/fetched one at a time to minimize memory footprint.
- **Progress Tracking**: Enable `show_progress=True` to monitor processing of large files.

## Error Handling

The utility handles various error conditions:

- Missing columns: Raises `ValueError` with descriptive message
- Invalid metric: Raises `ValueError` 
- Missing centroids: Raises `ValueError` during initialization
- Database errors: Logs warnings and continues with `None` values for failed lookups

## Examples

### Complete Example with KeywordEmbedder

```python
from RAG_supporters.embeddings import KeywordEmbedder
from RAG_supporters.clustering.topic_distance_calculator import calculate_topic_distances_from_csv
import pandas as pd

# Step 1: Prepare your data
data = {
    "question_text": [
        "What is deep learning?",
        "How do I optimize a database query?",
        "What is React used for?"
    ],
    "source_text": [
        "Deep learning uses neural networks with multiple layers.",
        "Database query optimization involves indexing and query planning.",
        "React is a JavaScript library for building user interfaces."
    ]
}
df = pd.DataFrame(data)
df.to_csv("input.csv", index=False)

# Step 2: Create embedder
embedder = KeywordEmbedder(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Step 3: Calculate distances
result_df = calculate_topic_distances_from_csv(
    csv_path="input.csv",
    keyword_clusterer_json="keyword_clusters.json",
    embedder=embedder,
    metric="cosine",
    output_path="output.csv",
    show_progress=True
)

# Step 4: Analyze results
import json

print(f"Processed {len(result_df)} rows")

# Show distance scores for first row
if pd.notna(result_df.at[0, 'question_term_distance_scores']):
    q_scores = json.loads(result_df.at[0, 'question_term_distance_scores'])
    print(f"Question distance scores: {q_scores}")
    print(f"Closest topic: {min(q_scores, key=q_scores.get)}")
```

### Integration with Domain Assessment Pipeline

```python
from RAG_supporters.clustering import cluster_keywords_from_embeddings
from RAG_supporters.embeddings import KeywordEmbedder
from RAG_supporters.clustering.topic_distance_calculator import calculate_topic_distances_from_csv

# Step 1: Create keyword clusters
embedder = KeywordEmbedder()
keywords = ["machine learning", "AI", "database", "SQL", "React", "frontend"]
keyword_embeddings = embedder.create_embeddings(keywords)

clusterer, topics = cluster_keywords_from_embeddings(
    keyword_embeddings,
    n_clusters=3,
    n_descriptors=5,
    output_path="clusters.json"
)

# Step 2: Calculate distances for your dataset
result_df = calculate_topic_distances_from_csv(
    csv_path="questions_and_sources.csv",
    keyword_clusterer_json="clusters.json",
    embedder=embedder,
    output_path="distances.csv"
)

# Step 3: Filter by topic relevance using JSON mapping
import json

# Get questions with low distance to "machine learning" topic
ml_questions = []
for idx, row in result_df.iterrows():
    if pd.notna(row['question_term_distance_scores']):
        scores = json.loads(row['question_term_distance_scores'])
        if 'machine learning' in scores and scores['machine learning'] < 0.3:
            ml_questions.append(idx)

print(f"Found {len(ml_questions)} questions highly relevant to machine learning")
```

## See Also

- [Clustering and Assignment Guide](CLUSTERING_AND_ASSIGNMENT.md)
- [Domain Assessment Examples](DOMAIN_ASSESSMENT_EXAMPLES.md)
- [KeywordClusterer API Documentation](../RAG_supporters/clustering/keyword_clustering.py)
