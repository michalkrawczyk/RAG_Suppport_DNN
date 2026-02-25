# LLM-Driven Cluster Steering and Ambiguity Resolution Guide

## Overview

This guide covers advanced LLM features for RAG subspace steering, including:
- Cluster activation via LLM-generated steering texts
- Question rephrasing for multi-membership scenarios
- Ambiguity resolution across multiple subspaces
- Best practices for LLM output validation and review

## Components

### 1. ClusterSteeringAgent

Located in `RAG_supporters/agents/cluster_steering.py`, this agent provides:

#### Generate Steering Text
```python
from RAG_supporters.agents.cluster_steering import ClusterSteeringAgent
from langchain_openai import ChatOpenAI

# Initialize agent
llm = ChatOpenAI(model="gpt-4")
agent = ClusterSteeringAgent(llm=llm)

# Generate steering text for a cluster
result = agent.generate_steering_text(
    question="What are the latest advances in neural networks?",
    cluster_id=2,
    cluster_descriptors=["deep learning", "neural networks", "AI", "machine learning"]
)

print(result["steering_text"])
# Output: "Recent developments in deep learning and neural network architectures..."
```

#### Rephrase Questions
```python
# Rephrase question to emphasize a specific cluster
result = agent.rephrase_question(
    question="How do transformers work?",
    target_cluster_id=1,
    cluster_descriptors=["NLP", "language models", "transformers", "attention"],
    alternate_clusters=[2, 5]
)

print(result["rephrased_question"])
# Shifted to emphasize NLP domain
```

#### Generate Multi-Cluster Variations
```python
# Generate question variations for multiple clusters
result = agent.generate_multi_cluster_variations(
    question="What is the impact of AI on healthcare?",
    cluster_info={
        0: ["healthcare", "medical", "diagnosis"],
        1: ["machine learning", "AI", "algorithms"],
        2: ["ethics", "policy", "regulation"]
    }
)

for variation in result["variations"]:
    print(f"Cluster {variation['target_cluster_id']}: {variation['rephrased_question']}")
```

#### Resolve Ambiguity
```python
# Analyze ambiguity in cluster assignments
result = agent.resolve_ambiguity(
    question="Explain quantum machine learning",
    cluster_assignments={
        0: 0.45,  # Quantum computing cluster
        1: 0.38,  # Machine learning cluster
        2: 0.12,  # Physics cluster
        3: 0.05   # Other cluster
    }
)

print(f"Is ambiguous: {result['is_ambiguous']}")
print(f"Recommendation: {result['recommendation']}")
print(f"Primary cluster: {result['primary_cluster']['cluster_id']}")
```

### 2. Steering Modes

The `SteeringConfig` class supports multiple steering modes:

```python
from RAG_supporters.dataset.steering.steering_config import SteeringConfig, SteeringMode

# Configure steering with multiple modes
config = SteeringConfig(
    mode=[
        (SteeringMode.SUGGESTION, 0.4),      # 40% use suggestions
        (SteeringMode.LLM_GENERATED, 0.3),   # 30% use LLM steering texts
        (SteeringMode.CLUSTER_DESCRIPTOR, 0.2),  # 20% use descriptors
        (SteeringMode.ZERO, 0.1)             # 10% no steering (baseline)
    ],
    multi_label_mode="soft",  # Allow multi-cluster assignment
    random_seed=42
)
```

**Steering Modes Explained:**

- **SUGGESTION**: Uses keyword suggestions extracted from source/question
- **LLM_GENERATED**: Uses LLM-generated steering texts (via ClusterSteeringAgent)
- **CLUSTER_DESCRIPTOR**: Uses cluster topic descriptors
- **ZERO**: No steering (baseline for ablation studies)

### 3. Integration with Dataset Builder

```python
from RAG_supporters.dataset.domain_assessment_dataset_builder import DomainAssessmentDatasetBuilder
from RAG_supporters.dataset.steering.steering_config import SteeringConfig, SteeringMode

# Create dataset with LLM steering
builder = DomainAssessmentDatasetBuilder(
    csv_paths="domain_assessments.csv",
    clustering_json_path="clusters.json",
    output_dir="dataset_output",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    steering_config=SteeringConfig(
        mode=[(SteeringMode.LLM_GENERATED, 0.5), (SteeringMode.SUGGESTION, 0.5)]
    )
)

# Build dataset (will generate steering embeddings)
builder.build()
```

## Best Practices

### 1. Ambiguity Handling

**When to Use Multi-Domain Assignment:**
- Question spans multiple clear topics (e.g., "AI in healthcare ethics")
- Multiple clusters have similar high probabilities (within 0.1 of each other)
- LLM ambiguity resolution recommends "multi-domain"

**When to Use Single-Domain Assignment:**
- Clear primary cluster (probability > 0.5)
- Secondary clusters significantly weaker (< 0.2)
- LLM ambiguity resolution recommends "single-domain"

### 2. Question Rephrasing Guidelines

**Good Rephrasing:**
- Preserves core information need
- Naturally incorporates target domain vocabulary
- Maintains grammatical correctness
- Shifts emphasis without changing intent

**Bad Rephrasing:**
- Forces unnatural keyword insertion
- Changes the fundamental question
- Creates ambiguous or confusing phrasing

### 3. Steering Text Generation

**Effective Steering Texts:**
- 1-3 sentences
- Incorporate 2-4 cluster descriptors naturally
- Guide retrieval without overfitting
- Maintain semantic coherence with question

**Example:**
```
Question: "How do neural networks learn?"
Cluster: Deep Learning (descriptors: backpropagation, gradients, optimization)

Good: "Neural network learning involves gradient-based optimization through 
backpropagation, adjusting weights to minimize error."

Bad: "backpropagation gradients optimization neural networks"  # Keyword stuffing
```

### 4. Validation Workflow

1. **Generate LLM outputs** using ClusterSteeringAgent
2. **Validate schema** using `llm_output_validation.py`
3. **Review samples** manually for quality
4. **Compute statistics** (confidence distributions, descriptor usage)
5. **Run ablation studies** comparing different steering modes

## Validation

### Automated Validation

```python
from RAG_supporters.utils.llm_output_validation import (
    validate_steering_text,
    validate_question_rephrase,
    validate_ambiguity_resolution,
    validate_batch_results
)

# Validate single result
is_valid, errors = validate_steering_text(steering_result)
if not is_valid:
    print(f"Validation errors: {errors}")

# Validate entire directory
report = validate_batch_results(
    results_dir=Path("llm_outputs"),
    output_report_path=Path("validation_report.json")
)
print(f"Overall validation rate: {report['overall_validation_rate']:.2%}")
```

### Statistical Analysis

```python
from RAG_supporters.utils.llm_output_validation import (
    compute_confidence_statistics,
    analyze_descriptor_usage
)

# Analyze confidence scores
stats = compute_confidence_statistics(steering_results)
print(f"Mean confidence: {stats['mean']:.3f}")
print(f"Std deviation: {stats['std']:.3f}")

# Analyze descriptor usage
usage = analyze_descriptor_usage(
    steering_results,
    available_descriptors={"deep learning", "AI", "neural networks", ...}
)
print(f"Avg descriptors per result: {usage['avg_descriptors_per_result']:.2f}")
print(f"Invalid descriptors: {usage.get('invalid_descriptors', [])}")
```

## Ablation Studies

Compare performance across steering modes:

```python
# Experiment 1: No steering (baseline)
config_baseline = SteeringConfig(mode=[(SteeringMode.ZERO, 1.0)])

# Experiment 2: Suggestions only
config_suggestions = SteeringConfig(mode=[(SteeringMode.SUGGESTION, 1.0)])

# Experiment 3: LLM steering only
config_llm = SteeringConfig(mode=[(SteeringMode.LLM_GENERATED, 1.0)])

# Experiment 4: Mixed (best practice)
config_mixed = SteeringConfig(
    mode=[
        (SteeringMode.LLM_GENERATED, 0.4),
        (SteeringMode.SUGGESTION, 0.3),
        (SteeringMode.CLUSTER_DESCRIPTOR, 0.2),
        (SteeringMode.ZERO, 0.1)
    ]
)

# Build datasets with each config and compare:
# - Retrieval accuracy
# - Cluster assignment quality
# - Diversity of results
```

## Multi-Membership Scenarios

### Example: Ambiguous Question

```python
question = "How do quantum computers impact cryptography?"

# This spans multiple domains:
# - Quantum computing
# - Cryptography
# - Computer science

# Generate variations for each aspect
variations_result = agent.generate_multi_cluster_variations(
    question=question,
    cluster_info={
        0: ["quantum computing", "qubits", "quantum gates"],
        1: ["cryptography", "encryption", "security"],
        2: ["algorithms", "complexity", "computation"]
    }
)

# Each variation emphasizes a different cluster
for v in variations_result["variations"]:
    print(f"[Cluster {v['target_cluster_id']}] {v['rephrased_question']}")

# Output:
# [Cluster 0] How do quantum computing principles affect cryptographic systems?
# [Cluster 1] What cryptographic implications arise from quantum computer capabilities?
# [Cluster 2] How does quantum computational complexity impact encryption algorithms?
```

### Handling Multi-Cluster Assignments

```python
# Soft assignment with threshold
from RAG_supporters.clustering import KeywordClusterer

clusterer = KeywordClusterer.from_results("clusters.json")
clusterer.configure_assignment(
    assignment_mode="soft",
    threshold=0.15,  # Include clusters with >= 15% probability
    metric="cosine"
)

# Assign question
result = clusterer.assign(question_embedding)
assigned_clusters = result["assigned_clusters"]  # e.g., [0, 1, 2]

# For each assigned cluster, generate steering text
for cluster_id in assigned_clusters:
    steering = agent.generate_steering_text(
        question=question,
        cluster_id=cluster_id,
        cluster_descriptors=clusterer.topics[cluster_id]
    )
    # Use steering text for retrieval in this subspace
```

## Troubleshooting

### Low Confidence Scores

**Causes:**
- Question too vague or broad
- Cluster descriptors not well-defined
- LLM uncertain about domain mapping

**Solutions:**
- Refine cluster descriptors
- Add more context to questions
- Use multi-cluster variations

### Poor Quality Rephrasing

**Causes:**
- Inappropriate descriptor selection
- Conflicting cluster themes
- Overly aggressive steering

**Solutions:**
- Review and curate cluster descriptors
- Use softer rephrasing (preserve more of original)
- Manual review and editing

### Ambiguity Resolution Failures

**Causes:**
- Too many similar probability scores
- Unclear question intent
- Overlapping cluster boundaries

**Solutions:**
- Improve clustering (adjust n_clusters)
- Use hierarchical clustering for sub-topics
- Request user clarification for edge cases

## Future Extensions

- **RL-based steering**: Use LLM outputs as feedback for reinforcement learning
- **Latent space steering**: Direct manipulation of embedding spaces
- **Adaptive thresholds**: Learn optimal assignment thresholds per cluster
- **Context-aware rephrasing**: Consider conversation history
- **Confidence calibration**: Improve LLM confidence score accuracy

## References

- `RAG_supporters/agents/cluster_steering.py` - Main agent implementation
- `RAG_supporters/prompts_templates/cluster_steering.py` - LLM prompts
- `RAG_supporters/dataset/steering_embedding_generator.py` - Steering generation
- `RAG_supporters/utils/llm_output_validation.py` - Validation utilities
- `docs/CLUSTERING_AND_ASSIGNMENT.md` - Clustering fundamentals
