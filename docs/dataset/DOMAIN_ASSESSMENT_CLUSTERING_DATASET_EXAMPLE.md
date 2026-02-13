# Domain Assessment with Clustering - Complete Minimal Example

## Overview

This document provides a complete minimal example showing the workflow from:
1. Creating suggestions with the domain assessment agent
2. Clustering the suggestions to obtain cluster descriptors
3. Executing domain assessment based on cluster descriptors
4. Creating a PyTorch Dataset from the resulting CSV file

## Complete Workflow Example

### Step 1: Extract Domain Suggestions from Sources

First, we extract domain suggestions from source texts using the domain assessment agent.

```python
import pandas as pd
from langchain_openai import ChatOpenAI
from RAG_supporters.agents.domain_assesment import DomainAnalysisAgent, OperationMode

# Initialize agent
llm = ChatOpenAI(model="gpt-4", temperature=0.3)
agent = DomainAnalysisAgent(llm=llm)

# Sample source texts
sources = [
    {
        "source_text": "Machine learning is a subset of artificial intelligence that enables "
                      "systems to learn from data. Deep learning uses neural networks to "
                      "process complex patterns.",
        "chroma_source_id": "src_001"
    },
    {
        "source_text": "Natural language processing involves computational linguistics and "
                      "text analysis. Modern NLP systems use transformers and attention mechanisms.",
        "chroma_source_id": "src_002"
    },
    {
        "source_text": "Computer vision algorithms process and analyze digital images. "
                      "Convolutional neural networks are commonly used for image recognition tasks.",
        "chroma_source_id": "src_003"
    }
]

# Create DataFrame
df_sources = pd.DataFrame(sources)

# Extract domains from sources
print("Extracting domains from sources...")
df_sources_with_domains = agent.process_dataframe(
    df_sources,
    mode=OperationMode.EXTRACT,
    text_source_col="source_text",
    save_path="sources_with_domains.csv",
    use_batch_processing=True
)

print(f"Extracted domains from {len(df_sources_with_domains)} sources")
print(df_sources_with_domains[['chroma_source_id', 'primary_theme', 'total_suggestions']].head())
```

**Output CSV Structure (`sources_with_domains.csv`):**
```csv
source_text,chroma_source_id,suggestions,total_suggestions,primary_theme
"Machine learning is...","src_001","[{""term"": ""Machine Learning"", ""type"": ""domain"", ""confidence"": 0.95, ""reason"": ""Core topic""}, {""term"": ""Deep Learning"", ""type"": ""subdomain"", ""confidence"": 0.90, ""reason"": ""Specialized technique""}]",2,"Artificial Intelligence"
```

### Step 2: Cluster Suggestions to Obtain Descriptors

Next, we cluster the extracted suggestions to discover topics and obtain cluster descriptors.

```python
import json
from RAG_supporters.embeddings import KeywordEmbedder
from RAG_supporters.clustering import cluster_keywords_from_embeddings

# Step 2.1: Extract all unique suggestions from the CSV
df = pd.read_csv("sources_with_domains.csv")

all_suggestions = []
for suggestions_str in df['suggestions']:
    try:
        suggestions = json.loads(suggestions_str) if isinstance(suggestions_str, str) else suggestions_str
        for suggestion in suggestions:
            all_suggestions.append(suggestion['term'])
    except Exception as e:
        print(f"Error parsing suggestions: {e}")

# Remove duplicates
unique_suggestions = list(set(all_suggestions))
print(f"Found {len(unique_suggestions)} unique suggestions")
print(f"Sample suggestions: {unique_suggestions[:5]}")

# Step 2.2: Create embeddings for suggestions
embedder = KeywordEmbedder()
suggestion_embeddings = embedder.create_embeddings(unique_suggestions)

# Step 2.3: Cluster suggestions to discover topics
clusterer, topics = cluster_keywords_from_embeddings(
    suggestion_embeddings,
    n_clusters=3,              # Number of topics to discover
    algorithm="kmeans",        # Clustering algorithm
    n_descriptors=5,          # Top 5 suggestions per cluster as descriptors
    output_path="suggestion_clusters.json",
    random_state=42
)

# Step 2.4: Display discovered topics (cluster descriptors)
print("\n=== Discovered Topics (Cluster Descriptors) ===")
for cluster_id, descriptors in topics.items():
    print(f"\nCluster {cluster_id}:")
    for descriptor in descriptors:
        print(f"  - {descriptor}")
```

**Output JSON Structure (`suggestion_clusters.json`):**
```json
{
  "metadata": {
    "algorithm": "kmeans",
    "n_clusters": 3,
    "n_suggestions": 15,
    "random_state": 42
  },
  "cluster_assignments": {
    "Machine Learning": 0,
    "Deep Learning": 0,
    "Neural Networks": 0,
    "Natural Language Processing": 1,
    "Text Analysis": 1,
    "Transformers": 1,
    "Computer Vision": 2,
    "Image Recognition": 2,
    "Convolutional Neural Networks": 2
  },
  "clusters": {
    "0": ["Machine Learning", "Deep Learning", "Neural Networks", "Artificial Intelligence", "Supervised Learning"],
    "1": ["Natural Language Processing", "Text Analysis", "Transformers", "Computational Linguistics", "Attention Mechanisms"],
    "2": ["Computer Vision", "Image Recognition", "Convolutional Neural Networks", "Image Processing", "Pattern Recognition"]
  },
  "cluster_stats": {
    "0": {
      "size": 5,
      "topic_descriptors": ["Machine Learning", "Deep Learning", "Neural Networks", "Artificial Intelligence", "Supervised Learning"]
    },
    "1": {
      "size": 5,
      "topic_descriptors": ["Natural Language Processing", "Text Analysis", "Transformers", "Computational Linguistics", "Attention Mechanisms"]
    },
    "2": {
      "size": 5,
      "topic_descriptors": ["Computer Vision", "Image Recognition", "Convolutional Neural Networks", "Image Processing", "Pattern Recognition"]
    }
  },
  "centroids": [[...], [...], [...]]
}
```

### Step 3: Domain Assessment with Cluster Descriptors

Now we execute domain assessment on questions using the cluster descriptors as available terms.

```python
from RAG_supporters.clustering import KeywordClusterer

# Step 3.1: Load clustering results
clusterer = KeywordClusterer.from_results("suggestion_clusters.json")
topics = clusterer.topics

# Step 3.2: Prepare available terms (cluster descriptors)
# We'll use the top descriptors from each cluster
available_terms = []
for cluster_id, descriptors in topics.items():
    available_terms.extend(descriptors)  # Use all descriptors from all clusters

print(f"Available terms for assessment: {len(available_terms)}")
print(f"Terms: {available_terms}")

# Step 3.3: Create questions paired with sources
questions = [
    {
        "question_text": "What algorithms are used in neural network training?",
        "source_text": "Machine learning is a subset of artificial intelligence...",
        "chroma_question_id": "q_001",
        "chroma_source_id": "src_001"
    },
    {
        "question_text": "How do transformers process text sequences?",
        "source_text": "Natural language processing involves computational linguistics...",
        "chroma_question_id": "q_002",
        "chroma_source_id": "src_002"
    },
    {
        "question_text": "What techniques are used for image classification?",
        "source_text": "Computer vision algorithms process and analyze digital images...",
        "chroma_question_id": "q_003",
        "chroma_source_id": "src_003"
    }
]

df_questions = pd.DataFrame(questions)

# Step 3.4: Assess questions against cluster descriptors
print("\nAssessing questions against cluster descriptors...")
df_assessed = agent.process_dataframe(
    df_questions,
    mode=OperationMode.ASSESS,
    question_col="question_text",
    available_terms=available_terms,
    save_path="domain_assessment_with_clusters.csv",
    use_batch_processing=True
)

print(f"Assessed {len(df_assessed)} questions")
print(df_assessed[['question_text', 'primary_topics', 'total_selected']].head())
```

**Minimal Example CSV Output (`domain_assessment_with_clusters.csv`):**
```csv
source_text,question_text,chroma_source_id,chroma_question_id,selected_terms,total_selected,question_intent,primary_topics
"Machine learning is a subset of artificial intelligence...","What algorithms are used in neural network training?","src_001","q_001","[{""term"": ""Machine Learning"", ""type"": ""domain"", ""relevance_score"": 0.95, ""reason"": ""Directly addresses ML training algorithms""}, {""term"": ""Deep Learning"", ""type"": ""subdomain"", ""relevance_score"": 0.90, ""reason"": ""Neural networks are core to deep learning""}, {""term"": ""Neural Networks"", ""type"": ""keyword"", ""relevance_score"": 0.98, ""reason"": ""Question explicitly mentions neural networks""}]",3,"Understanding neural network training algorithms","[""Machine Learning"", ""Neural Networks"", ""Deep Learning""]"
"Natural language processing involves...","How do transformers process text sequences?","src_002","q_002","[{""term"": ""Natural Language Processing"", ""type"": ""domain"", ""relevance_score"": 0.92, ""reason"": ""Transformers are NLP models""}, {""term"": ""Transformers"", ""type"": ""keyword"", ""relevance_score"": 0.99, ""reason"": ""Question explicitly about transformers""}, {""term"": ""Text Analysis"", ""type"": ""keyword"", ""relevance_score"": 0.85, ""reason"": ""Processing sequences relates to text analysis""}]",3,"Understanding transformer architecture for text processing","[""Natural Language Processing"", ""Transformers""]"
```

### Step 4: Add Cluster Probabilities (Optional)

We can also add soft cluster assignments (probabilities) to the CSV for enhanced dataset creation.

```python
# Step 4.1: Embed questions and sources
question_texts = df_assessed['question_text'].tolist()
source_texts = df_assessed['source_text'].tolist()

question_embeddings = embedder.create_embeddings(question_texts)
source_embeddings = embedder.create_embeddings(source_texts)

# Step 4.2: Configure soft assignment
clusterer.configure_assignment(
    assignment_mode="soft",
    threshold=0.05,        # Include clusters with prob > 5%
    temperature=1.0,       # Standard softmax temperature
    metric="cosine"
)

# Step 4.3: Assign questions to clusters
cluster_probabilities = []
for i, question_text in enumerate(question_texts):
    embedding = question_embeddings[question_text]
    assignment = clusterer.assign(embedding)
    # Get probability vector for all clusters
    probs = assignment.get('probabilities', {})
    prob_vector = [probs.get(str(c), 0.0) for c in range(clusterer.n_clusters)]
    cluster_probabilities.append(json.dumps(prob_vector))

# Step 4.4: Add to DataFrame
df_assessed['cluster_probabilities'] = cluster_probabilities

# Save final CSV with cluster probabilities
df_assessed.to_csv("domain_assessment_with_clusters.csv", index=False)

print("\n=== Sample cluster probabilities ===")
for i, (q, probs) in enumerate(zip(question_texts[:3], cluster_probabilities[:3])):
    print(f"\nQuestion: {q[:60]}...")
    probs_list = json.loads(probs)
    for cluster_id, prob in enumerate(probs_list):
        if prob > 0.05:
            print(f"  Cluster {cluster_id}: {prob:.3f}")
```

**Final CSV with Cluster Probabilities:**
```csv
source_text,question_text,chroma_source_id,chroma_question_id,selected_terms,total_selected,question_intent,primary_topics,cluster_probabilities
"Machine learning is...","What algorithms are used in neural network training?","src_001","q_001","[...]",3,"Understanding neural network training algorithms","[""Machine Learning"", ""Neural Networks""]","[0.78, 0.12, 0.10]"
"Natural language processing...","How do transformers process text sequences?","src_002","q_002","[...]",3,"Understanding transformer architecture","[""Natural Language Processing"", ""Transformers""]","[0.05, 0.85, 0.10]"
"Computer vision algorithms...","What techniques are used for image classification?","src_003","q_003","[...]",2,"Image classification techniques","[""Computer Vision"", ""Image Recognition""]","[0.08, 0.07, 0.85]"
```

### Step 5: Create PyTorch Dataset from CSV

Finally, we create a PyTorch Dataset from the CSV file with clustering information.

**Important:** The dataset builder is aware that sources and questions may have different suggestion columns:
- **Sources** (from `EXTRACT` mode): Use column specified by `source_suggestions_col` (default: `'suggestions'`)
- **Questions** (from `ASSESS` mode): Use column specified by `question_suggestions_col` (default: `'selected_terms'`)

The builder automatically selects the correct column based on whether the row is a source or question.

```python
from pathlib import Path
from RAG_supporters.dataset import DomainAssessmentDatasetBuilder, ClusterLabeledDataset
from RAG_supporters.embeddings_ops import SteeringConfig, SteeringMode

# Step 5.1: Configure steering (optional augmentation)
steering_config = SteeringConfig(
    mode=[(SteeringMode.ZERO, 0.8), (SteeringMode.CLUSTER_DESCRIPTOR, 0.2)]
)

# Step 5.2: Build dataset from CSV + clustering JSON
# Option 1: Use model name (simplest)
builder = DomainAssessmentDatasetBuilder(
    csv_paths="domain_assessment_with_clusters.csv",
    clustering_json_path="suggestion_clusters.json",
    output_dir="dataset_output",
    embedding_model='all-MiniLM-L6-v2',  # Just pass model name!
    steering_config=steering_config,
    label_normalizer="softmax",
    label_temp=1.0,
    combined_label_weight=0.5,
    augment_noise_prob=0.1,
    augment_zero_prob=0.1,
    augment_noise_level=0.01
)

# Option 2: Use sentence-transformers model
from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
builder = DomainAssessmentDatasetBuilder(
    csv_paths="domain_assessment_with_clusters.csv",
    clustering_json_path="suggestion_clusters.json",
    output_dir="dataset_output",
    embedding_model=embedding_model,
    steering_config=steering_config
)

# Option 3: Use LangChain model (supports OpenAI, Cohere, etc.)
from langchain_openai import OpenAIEmbeddings
langchain_model = OpenAIEmbeddings(model="text-embedding-3-small")
builder = DomainAssessmentDatasetBuilder(
    csv_paths="domain_assessment_with_clusters.csv",
    clustering_json_path="suggestion_clusters.json",
    output_dir="dataset_output",
    embedding_model=langchain_model,
    steering_config=steering_config
)

# Option 4: Custom column names
# Use this if your CSV has different column names for suggestions
builder = DomainAssessmentDatasetBuilder(
    csv_paths="domain_assessment_with_clusters.csv",
    clustering_json_path="suggestion_clusters.json",
    output_dir="dataset_output",
    embedding_model='all-MiniLM-L6-v2',
    steering_config=steering_config,
    source_suggestions_col='source_domains',      # Custom column for sources
    question_suggestions_col='question_keywords'  # Custom column for questions
)

# Build the dataset
builder.build()

print("\n=== Dataset Build Complete ===")
print(f"Output directory: {builder.output_dir}")
print(f"Clusters: {builder.clustering_data.n_clusters}")
```

### Step 6: Load and Use the Dataset

Now we can load the created dataset and use it with PyTorch DataLoader.

```python
import torch
from torch.utils.data import DataLoader

# Step 6.1: Load dataset
dataset = ClusterLabeledDataset(
    dataset_dir="dataset_output",
    label_type="combined",      # 'source', 'steering', or 'combined'
    return_metadata=False,      # Set True to get metadata
    mmap_mode="r",             # Read-only memory mapping
    cache_size=1000            # LRU cache size
)

print(f"\n=== Dataset Loaded ===")
print(f"Total samples: {len(dataset)}")
print(f"Number of clusters: {dataset.n_clusters}")
print(f"Embedding dimension: {dataset.embedding_dim}")

# Step 6.2: Create DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Step 6.3: Iterate over batches
print("\n=== Sample Batch ===")
for batch_idx, (base_embeddings, steering_embeddings, labels) in enumerate(dataloader):
    print(f"Batch {batch_idx}:")
    print(f"  Base embeddings shape: {base_embeddings.shape}")
    print(f"  Steering embeddings shape: {steering_embeddings.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Sample label distribution: {labels[0]}")
    
    if batch_idx == 0:  # Just show first batch
        break

# Step 6.4: Access individual samples
print("\n=== Individual Sample ===")
base_emb, steering_emb, label = dataset[0]
print(f"Base embedding shape: {base_emb.shape}")
print(f"Steering embedding shape: {steering_emb.shape}")
print(f"Label shape: {label.shape}")
print(f"Label (cluster probabilities): {label}")
print(f"Primary cluster: {torch.argmax(label).item()}")

# Step 6.5: Access dataset metadata
print("\n=== Dataset Metadata ===")
print(f"Total samples: {len(dataset)}")
print(f"Number of clusters: {dataset.n_clusters}")
print(f"Embedding dimension: {dataset.embedding_dim}")
```

## Complete Code Summary

Here's a complete minimal script combining all steps:

```python
#!/usr/bin/env python3
"""Complete minimal example: Domain Assessment → Clustering → Dataset"""

import json
import pandas as pd
from pathlib import Path
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader

from RAG_supporters.agents.domain_assesment import DomainAnalysisAgent, OperationMode
from RAG_supporters.embeddings import KeywordEmbedder
from RAG_supporters.clustering import cluster_keywords_from_embeddings, KeywordClusterer
from RAG_supporters.dataset import DomainAssessmentDatasetBuilder, ClusterLabeledDataset
from RAG_supporters.embeddings_ops import SteeringConfig, SteeringMode

# ============================================================================
# STEP 1: Extract domain suggestions from sources
# ============================================================================
print("STEP 1: Extracting domain suggestions...")

llm = ChatOpenAI(model="gpt-4", temperature=0.3)
agent = DomainAnalysisAgent(llm=llm)

sources = [
    {"source_text": "Machine learning is a subset of AI that enables systems to learn from data...", 
     "chroma_source_id": "src_001"},
    {"source_text": "Natural language processing involves computational linguistics...", 
     "chroma_source_id": "src_002"},
    {"source_text": "Computer vision algorithms process and analyze digital images...", 
     "chroma_source_id": "src_003"}
]

df_sources = pd.DataFrame(sources)
df_sources_with_domains = agent.process_dataframe(
    df_sources, mode=OperationMode.EXTRACT, 
    text_source_col="source_text",
    save_path="sources_with_domains.csv"
)

# ============================================================================
# STEP 2: Cluster suggestions to obtain descriptors
# ============================================================================
print("\nSTEP 2: Clustering suggestions...")

# Extract unique suggestions
all_suggestions = []
for sugg_str in df_sources_with_domains['suggestions']:
    try:
        suggestions = json.loads(sugg_str)
        all_suggestions.extend([s['term'] for s in suggestions])
    except: pass

unique_suggestions = list(set(all_suggestions))

# Cluster suggestions
embedder = KeywordEmbedder()
suggestion_embeddings = embedder.create_embeddings(unique_suggestions)
clusterer, topics = cluster_keywords_from_embeddings(
    suggestion_embeddings, n_clusters=3, n_descriptors=5,
    output_path="suggestion_clusters.json"
)

# Get available terms (cluster descriptors)
available_terms = []
for descriptors in topics.values():
    available_terms.extend(descriptors)

# ============================================================================
# STEP 3: Domain assessment with cluster descriptors
# ============================================================================
print("\nSTEP 3: Assessing questions against cluster descriptors...")

questions = [
    {"question_text": "What algorithms are used in neural network training?",
     "source_text": sources[0]["source_text"],
     "chroma_question_id": "q_001", "chroma_source_id": "src_001"},
    {"question_text": "How do transformers process text sequences?",
     "source_text": sources[1]["source_text"],
     "chroma_question_id": "q_002", "chroma_source_id": "src_002"},
    {"question_text": "What techniques are used for image classification?",
     "source_text": sources[2]["source_text"],
     "chroma_question_id": "q_003", "chroma_source_id": "src_003"}
]

df_questions = pd.DataFrame(questions)
df_assessed = agent.process_dataframe(
    df_questions, mode=OperationMode.ASSESS,
    question_col="question_text", available_terms=available_terms,
    save_path="domain_assessment_with_clusters.csv"
)

# ============================================================================
# STEP 4: Add cluster probabilities
# ============================================================================
print("\nSTEP 4: Adding cluster probabilities...")

question_texts = df_assessed['question_text'].tolist()
question_embeddings = embedder.create_embeddings(question_texts)

clusterer.configure_assignment(assignment_mode="soft", threshold=0.05)
cluster_probabilities = []
for question_text in question_texts:
    embedding = question_embeddings[question_text]
    assignment = clusterer.assign(embedding)
    probs = assignment.get('probabilities', {})
    prob_vector = [probs.get(str(c), 0.0) for c in range(clusterer.n_clusters)]
    cluster_probabilities.append(json.dumps(prob_vector))

df_assessed['cluster_probabilities'] = cluster_probabilities
df_assessed.to_csv("domain_assessment_with_clusters.csv", index=False)

# ============================================================================
# STEP 5: Create PyTorch Dataset
# ============================================================================
print("\nSTEP 5: Building PyTorch Dataset...")

steering_config = SteeringConfig(mode=[(SteeringMode.ZERO, 0.8), (SteeringMode.CLUSTER_DESCRIPTOR, 0.2)])

builder = DomainAssessmentDatasetBuilder(
    csv_paths="domain_assessment_with_clusters.csv",
    clustering_json_path="suggestion_clusters.json",
    output_dir="dataset_output",
    embedding_model='all-MiniLM-L6-v2',  # Use string model name (simpler!)
    steering_config=steering_config,
    combined_label_weight=0.5
)

builder.build()
print(f"Dataset built in: {builder.output_dir}")

# ============================================================================
# STEP 6: Load and use Dataset
# ============================================================================
print("\nSTEP 6: Loading and using Dataset...")

dataset = ClusterLabeledDataset(
    dataset_dir="dataset_output",
    label_type="combined",
    return_metadata=False
)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch_idx, (base_emb, steering_emb, labels) in enumerate(dataloader):
    print(f"Batch {batch_idx}: base={base_emb.shape}, steering={steering_emb.shape}, labels={labels.shape}")
    break

print("\n✓ Complete workflow finished successfully!")
```

## Key Points

1. **CSV Structure**: The domain assessment CSV must contain:
   - `source_text`, `question_text`
   - `chroma_source_id`, `chroma_question_id`
   - Suggestion columns (configurable):
     - `suggestions` (default for sources): JSON list from EXTRACT mode
     - `selected_terms` (default for questions): JSON list from ASSESS mode
   - `cluster_probabilities` (optional, JSON array of probabilities)
   
   **Important:** The parser automatically distinguishes between source and question suggestions based on the configured column names (`source_suggestions_col` and `question_suggestions_col`).

2. **Clustering JSON**: Must contain:
   - `cluster_assignments`: Map from suggestion → cluster ID
   - `clusters`: Map from cluster ID → list of suggestions
   - `cluster_stats`: Statistics including `topic_descriptors`
   - `centroids`: Cluster centroids for assignment

3. **Dataset Output**: Creates:
   - `dataset.db`: SQLite database with metadata and labels
   - `base_embeddings.npy`: Memory-mapped base embeddings
   - `steering_embeddings.npy`: Memory-mapped steering embeddings

4. **Label Types**:
   - `source_label`: Label for base embedding (question/source)
   - `steering_label`: Label for steering embedding
   - `combined_label`: Weighted average for augmentation

## See Also

- [Domain Analysis Agent Documentation](../agents/DOMAIN_ANALYSIS_AGENT.md)
- [Clustering and Assignment Documentation](../clustering/CLUSTERING_AND_ASSIGNMENT.md)
- [Dataset Check Agent](../agents/DATASET_CHECK_AGENT.md)
