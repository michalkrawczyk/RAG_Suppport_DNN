# DomainAnalysisAgent Documentation

## Overview

The `DomainAnalysisAgent` is a unified LangGraph-based agent for comprehensive domain analysis tasks. It provides five distinct operation modes for extracting domains from text, guessing required domains for questions, assessing question relevance to available domain terms, calculating topic relevance probabilities, and assessing relevance to grouped topic clusters. This agent is essential for domain classification, knowledge base organization, and question-domain matching in RAG applications.

### What It Does

The agent provides five powerful operation modes:
- **EXTRACT Mode**: Extract domains, subdomains, and keywords from source text
- **GUESS Mode**: Identify domains needed to answer a given question
- **ASSESS Mode**: Evaluate which available domain terms are most relevant to a question
- **TOPIC_RELEVANCE_PROB Mode**: Assess relevance probabilities between a question and individual topic descriptors
- **GROUP_TOPIC_RELEVANCE_PROB Mode**: Assess relevance probabilities between a question and grouped topic clusters

### Key Features

- **Five Operation Modes**: Extract, Guess, Assess, Topic Relevance Probability, and Grouped Topic Relevance Probability assessment
- **Structured Output**: Returns validated Pydantic models with scores and reasoning
- **Batch Processing**: Efficiently process multiple texts or questions
- **LangGraph Architecture**: Robust workflow with automatic retry logic
- **Flexible Input**: Works with text, DataFrames, and CSV files
- **OpenAI Batch Support**: Optimized batch processing for OpenAI models
- **Configurable Reasoning**: Optional reasoning explanations (via `include_reason` parameter, applies to TOPIC_RELEVANCE_PROB and GROUP_TOPIC_RELEVANCE_PROB modes)

## Installation

The agent requires the following dependencies:

```bash
pip install langchain langgraph pydantic pandas tqdm
```

For specific LLM providers:
```bash
# For OpenAI
pip install langchain-openai

# Install all agent dependencies
pip install -e .[openai]  # or .[nvidia] for NVIDIA
```

## Operation Modes

### EXTRACT Mode - Domain Extraction from Text

Extract domains, subdomains, and keywords from source text.

**Use Cases:**
- Categorizing documents
- Building domain taxonomies
- Extracting topics from articles

**Example:**
```python
from langchain_openai import ChatOpenAI
from RAG_supporters.agents.domain_assesment import DomainAnalysisAgent

llm = ChatOpenAI(model="gpt-4", temperature=0.3)
agent = DomainAnalysisAgent(llm=llm)

text = """
Machine learning is a subset of artificial intelligence that enables systems 
to learn and improve from experience. Deep learning, a specialized branch of 
machine learning, uses neural networks with multiple layers to process complex patterns.
"""

result = agent.extract_domains(text)
print(f"Primary theme: {result['primary_theme']}")
print(f"Suggestions: {result['suggestions']}")
```

**Output:**
```python
{
    'suggestions': [
        {'term': 'Artificial Intelligence', 'type': 'domain', 'confidence': 0.9, 'reason': '...'},
        {'term': 'Machine Learning', 'type': 'subdomain', 'confidence': 0.95, 'reason': '...'},
        {'term': 'Deep Learning', 'type': 'subdomain', 'confidence': 0.85, 'reason': '...'}
    ],
    'total_suggestions': 3,
    'primary_theme': 'Artificial Intelligence and Machine Learning'
}
```

### GUESS Mode - Domain Guessing for Questions

Identify which domains are needed to answer a question.

**Use Cases:**
- Routing questions to appropriate experts
- Selecting relevant knowledge bases
- Understanding question scope

**Example:**
```python
question = "How does gradient descent optimize neural network weights?"

result = agent.guess_domains(question)
print(f"Question category: {result['question_category']}")
print(f"Required domains: {[s['term'] for s in result['suggestions']]}")
```

**Output:**
```python
{
    'suggestions': [
        {'term': 'Machine Learning', 'type': 'domain', 'confidence': 0.9, 'reason': '...'},
        {'term': 'Neural Networks', 'type': 'subdomain', 'confidence': 0.95, 'reason': '...'},
        {'term': 'Optimization', 'type': 'keyword', 'confidence': 0.8, 'reason': '...'}
    ],
    'total_suggestions': 3,
    'question_category': 'Technical/Machine Learning'
}
```

### ASSESS Mode - Domain Relevance Assessment

Evaluate which available domain terms are most relevant to a question.

**Use Cases:**
- Matching questions to knowledge bases
- Filtering relevant content
- Domain-based routing

**Example:**
```python
question = "What is photosynthesis?"
available_terms = ["biology", "chemistry", "physics", "mathematics"]

result = agent.assess_domains(question, available_terms)
print(f"Question intent: {result['question_intent']}")
print(f"Primary topics: {result['primary_topics']}")
print(f"Selected terms: {[t['term'] for t in result['selected_terms']]}")
```

**Output:**
```python
{
    'selected_terms': [
        {'term': 'biology', 'type': 'domain', 'relevance_score': 0.95, 'reason': '...'},
        {'term': 'chemistry', 'type': 'domain', 'relevance_score': 0.7, 'reason': '...'}
    ],
    'total_selected': 2,
    'question_intent': 'Understanding the biological process of photosynthesis',
    'primary_topics': ['biology', 'chemistry']
}
```

### TOPIC_RELEVANCE_PROB Mode - Topic Relevance Probability Assessment

Assess relevance probabilities between a question and topic descriptors. This mode is designed to work with cluster descriptors from KeywordClusterer.

**Use Cases:**
- Matching questions to topic clusters
- Calculating semantic similarity probabilities
- Topic-based content filtering
- Question routing with confidence scores

**Example:**
```python
question = "How does gradient descent optimize neural networks?"
topic_descriptors = [
    "machine learning algorithms",
    "database systems", 
    "web development",
    "neural networks"
]

result = agent.assess_topic_relevance_prob(question, topic_descriptors)
print(f"Topic scores: {result['topic_scores']}")
print(f"Total topics: {result['total_topics']}")
```

**Output:**
```python
{
    'topic_scores': [
        {'topic_descriptor': 'machine learning algorithms', 'probability': 0.95},
        {'topic_descriptor': 'neural networks', 'probability': 0.98},
        {'topic_descriptor': 'database systems', 'probability': 0.1},
        {'topic_descriptor': 'web development', 'probability': 0.05}
    ],
    'total_topics': 4,
    'question_summary': 'Question about optimization in neural networks'
}
```

**With Reasoning (using `include_reason=True`):**
```python
# Initialize agent with reasoning enabled
agent = DomainAnalysisAgent(llm=llm, include_reason=True)

result = agent.assess_topic_relevance_prob(question, topic_descriptors)
```

**Output with reasoning:**
```python
{
    'topic_scores': [
        {
            'topic_descriptor': 'machine learning algorithms', 
            'probability': 0.95,
            'reason': 'Gradient descent is a fundamental machine learning optimization algorithm'
        },
        {
            'topic_descriptor': 'neural networks', 
            'probability': 0.98,
            'reason': 'Gradient descent is the primary method for training neural networks'
        },
        # ...
    ],
    'total_topics': 4,
    'question_summary': 'Question about optimization in neural networks'
}
```

### GROUP_TOPIC_RELEVANCE_PROB Mode - Grouped Topic Relevance

Assess relevance probabilities for groups/clusters of topic descriptors. Unlike TOPIC_RELEVANCE_PROB which assigns probabilities to individual descriptors, this mode groups descriptors by their cluster assignments and assigns a probability to each cluster.

**Use Cases:**
- Matching questions to topic clusters (not individual topics)
- Cluster-based question routing
- Understanding which topic group a question belongs to
- Simplified topic classification with grouped descriptors

**Key Difference from TOPIC_RELEVANCE_PROB:**
- **TOPIC_RELEVANCE_PROB**: Assigns probability to each individual topic descriptor
- **GROUP_TOPIC_RELEVANCE_PROB**: Groups descriptors by cluster and assigns probability to each cluster/group

**Example:**
```python
question = "How does gradient descent optimize neural networks?"

# Cluster data with grouped descriptors
cluster_data = {
    "cluster_stats": {
        "0": {"topic_descriptors": ["machine learning", "algorithms", "optimization"], "size": 50},
        "1": {"topic_descriptors": ["database", "SQL", "storage"], "size": 30},
        "2": {"topic_descriptors": ["web development", "frontend"], "size": 25}
    }
}

result = agent.assess_group_topic_relevance_prob(question, cluster_data)
print(f"Question: {result['question_text']}")
for group in result['group_probs']:
    print(f"Cluster {group['cluster_id']}: {group['probability']} - {group['descriptors']}")
```

**Output:**
```python
{
    'question_text': 'How does gradient descent optimize neural networks?',
    'group_probs': [
        {
            'cluster_id': 0,
            'descriptors': ['machine learning', 'algorithms', 'optimization'],
            'probability': 0.95
        },
        {
            'cluster_id': 1,
            'descriptors': ['database', 'SQL', 'storage'],
            'probability': 0.05
        },
        {
            'cluster_id': 2,
            'descriptors': ['web development', 'frontend'],
            'probability': 0.10
        }
    ],
    'total_groups': 3,
    'question_summary': 'Question about optimization in neural networks'
}
```

**With Reasoning (using `include_reason=True`):**
```python
# Initialize agent with reasoning enabled
agent = DomainAnalysisAgent(llm=llm, include_reason=True)

result = agent.assess_group_topic_relevance_prob(question, cluster_data)
```

**Output with reasoning:**
```python
{
    'question_text': 'How does gradient descent optimize neural networks?',
    'group_probs': [
        {
            'cluster_id': 0,
            'descriptors': ['machine learning', 'algorithms', 'optimization'],
            'probability': 0.95,
            'reason': 'This cluster covers ML optimization algorithms, highly relevant to gradient descent'
        },
        {
            'cluster_id': 1,
            'descriptors': ['database', 'SQL', 'storage'],
            'probability': 0.05,
            'reason': 'Database concepts are not directly related to neural network optimization'
        },
        # ...
    ],
    'total_groups': 3,
    'question_summary': 'Question about optimization in neural networks'
}
```

**Loading from File:**
```python
# Load cluster data from KeywordClusterer JSON file
result = agent.assess_group_topic_relevance_prob(
    question,
    "path/to/keyword_clusters.json"
)
```

## Basic Usage

### Initialize the Agent

```python
from langchain_openai import ChatOpenAI
from RAG_supporters.agents.domain_assesment import DomainAnalysisAgent

# Initialize with an LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.3)
agent = DomainAnalysisAgent(llm=llm, max_retries=3, batch_size=10)

# Initialize with reasoning enabled for topic relevance assessments
agent_with_reasoning = DomainAnalysisAgent(
    llm=llm, 
    max_retries=3, 
    batch_size=10,
    include_reason=True  # Enable reasoning explanations in topic relevance modes
)
```

### Single Text/Question Operations

```python
# Extract domains from text
result = agent.extract_domains("Your text here...")

# Guess domains for question
result = agent.guess_domains("Your question here?")

# Assess question against available terms
result = agent.assess_domains("Your question?", ["term1", "term2", "term3"])

# Assess topic relevance probabilities (individual descriptors)
topic_descriptors = ["ml", "databases", "web dev"]
result = agent.assess_topic_relevance_prob("Your question?", topic_descriptors)

# Assess grouped topic relevance probabilities (cluster-based)
cluster_data = {
    "cluster_stats": {
        "0": {"topic_descriptors": ["ml", "ai"], "size": 50},
        "1": {"topic_descriptors": ["db", "sql"], "size": 30}
    }
}
result = agent.assess_group_topic_relevance_prob("Your question?", cluster_data)
```

### Batch Processing

```python
# Batch extract domains
texts = ["Text 1...", "Text 2...", "Text 3..."]
results = agent.extract_domains_batch(texts)

# Batch guess domains
questions = ["Question 1?", "Question 2?", "Question 3?"]
results = agent.guess_domains_batch(questions)

# Batch assess domains
questions = ["Q1?", "Q2?"]
available_terms = ["physics", "chemistry", "biology"]
results = agent.assess_domains_batch(questions, available_terms)

# Batch assess topic relevance (individual descriptors)
questions = ["Q1?", "Q2?", "Q3?"]
topic_descriptors = ["ml", "databases", "web dev"]
results = agent.assess_topic_relevance_prob_batch(questions, topic_descriptors)

# Batch assess grouped topic relevance (clusters)
questions = ["Q1?", "Q2?", "Q3?"]
cluster_data = {
    "cluster_stats": {
        "0": {"topic_descriptors": ["ml", "ai"], "size": 50},
        "1": {"topic_descriptors": ["db", "sql"], "size": 30}
    }
}
results = agent.assess_group_topic_relevance_prob_batch(questions, cluster_data)
```

## DataFrame/CSV Processing

### Process DataFrame - Extract Mode

```python
import pandas as pd
from RAG_supporters.agents.domain_assesment import DomainAnalysisAgent, OperationMode

# Your DataFrame with source texts
df = pd.read_csv("articles.csv")

# Extract domains from each article
result_df = agent.process_dataframe(
    df,
    mode=OperationMode.EXTRACT,
    text_source_col="content",
    save_path="articles_with_domains.csv",
    use_batch_processing=True
)

# Check results
print(result_df[['content', 'primary_theme', 'total_suggestions']].head())
```

### Process DataFrame - Guess Mode

```python
# Your DataFrame with questions
df = pd.read_csv("questions.csv")

# Guess domains needed for each question
result_df = agent.process_dataframe(
    df,
    mode=OperationMode.GUESS,
    question_col="question_text",
    save_path="questions_with_domains.csv"
)

print(result_df[['question_text', 'question_category', 'total_suggestions']].head())
```

### Process DataFrame - Assess Mode

```python
# Your DataFrame with questions
df = pd.read_csv("questions.csv")

# Available domain terms
available_terms = ["physics", "chemistry", "biology", "mathematics"]

# Assess each question against available terms
result_df = agent.process_dataframe(
    df,
    mode=OperationMode.ASSESS,
    question_col="question_text",
    available_terms=available_terms,
    save_path="questions_assessed.csv"
)

print(result_df[['question_text', 'primary_topics', 'total_selected']].head())
```

### Process DataFrame - GROUP_TOPIC_RELEVANCE_PROB Mode

```python
# Your DataFrame with questions
df = pd.read_csv("questions.csv")

# Cluster data with grouped descriptors
cluster_data = {
    "cluster_stats": {
        "0": {"topic_descriptors": ["machine learning", "neural networks", "AI"], "size": 50},
        "1": {"topic_descriptors": ["database", "SQL", "storage"], "size": 30},
        "2": {"topic_descriptors": ["web development", "frontend", "JavaScript"], "size": 25}
    }
}

# Assess each question's relevance to topic clusters
result_df = agent.process_dataframe(
    df,
    mode=OperationMode.GROUP_TOPIC_RELEVANCE_PROB,
    question_col="question_text",
    cluster_data=cluster_data,
    save_path="questions_with_cluster_probs.csv"
)

# Access simplified cluster probability mapping
print(result_df[['question_text', 'question_cluster_probs']].head())
# question_cluster_probs column contains: {"0": 0.95, "1": 0.05, "2": 0.10}

# Access full group details with descriptors
print(result_df[['question_text', 'group_topic_relevance_prob_group_probs']].head())
```

## Advanced Usage

### Checkpoint Processing for Large Datasets

```python
# Process with checkpoints every 100 rows
result_df = agent.process_dataframe(
    df,
    mode=OperationMode.EXTRACT,
    text_source_col="content",
    save_path="articles.csv",
    checkpoint_batch_size=100,  # Save every 100 rows
    skip_existing=True,  # Skip rows with existing results
    use_batch_processing=True,
    batch_size=20
)
```

### Custom Retry Logic

```python
# Initialize with custom retry settings
agent = DomainAnalysisAgent(
    llm=llm,
    max_retries=5  # More retries for unreliable connections
)
```

### Progress Bar Control

```python
# Disable progress bar
result_df = agent.process_dataframe(
    df,
    mode=OperationMode.EXTRACT,
    text_source_col="content",
    progress_bar=False  # No progress bar
)
```

### Working with Complex Domain Terms

```python
# Available terms as structured objects
available_terms = [
    {"name": "machine_learning", "description": "AI subset focused on learning from data"},
    {"name": "deep_learning", "description": "Neural networks with multiple layers"},
    {"name": "nlp", "description": "Natural language processing"}
]

# The agent converts to JSON automatically
result = agent.assess_domains(question, available_terms)
```

## Complete Examples

### Example 1: Article Categorization System

```python
import pandas as pd
from langchain_openai import ChatOpenAI
from RAG_supporters.agents.domain_assesment import DomainAnalysisAgent, OperationMode

# Initialize agent
llm = ChatOpenAI(model="gpt-4", temperature=0.3)
agent = DomainAnalysisAgent(llm=llm)

# Sample articles
articles = [
    {
        "id": 1,
        "title": "Introduction to Neural Networks",
        "content": "Neural networks are computing systems inspired by biological neural networks..."
    },
    {
        "id": 2,
        "title": "Quantum Computing Basics",
        "content": "Quantum computers use quantum bits or qubits to perform computations..."
    },
    {
        "id": 3,
        "title": "Photosynthesis in Plants",
        "content": "Photosynthesis is the process by which plants convert light energy..."
    }
]

df = pd.DataFrame(articles)

# Extract domains
result_df = agent.process_dataframe(
    df,
    mode=OperationMode.EXTRACT,
    text_source_col="content",
    progress_bar=True
)

# Display categorization results
for idx, row in result_df.iterrows():
    print(f"\nArticle: {row['title']}")
    print(f"Primary Theme: {row['primary_theme']}")
    print(f"Suggestions: {row['suggestions']}")
```

### Example 2: Question Routing System

```python
import pandas as pd
from langchain_openai import ChatOpenAI
from RAG_supporters.agents.domain_assesment import DomainAnalysisAgent, OperationMode

# Initialize agent
llm = ChatOpenAI(model="gpt-4", temperature=0.3)
agent = DomainAnalysisAgent(llm=llm)

# Available expert domains
expert_domains = [
    "machine_learning",
    "quantum_physics",
    "molecular_biology",
    "organic_chemistry",
    "astrophysics"
]

# Incoming questions
questions = pd.DataFrame({
    "id": [1, 2, 3],
    "question_text": [
        "How do convolutional neural networks recognize images?",
        "What is quantum entanglement?",
        "How does DNA replication work?"
    ]
})

# Assess each question against expert domains
result_df = agent.process_dataframe(
    questions,
    mode=OperationMode.ASSESS,
    question_col="question_text",
    available_terms=expert_domains
)

# Route questions to experts
for idx, row in result_df.iterrows():
    selected = eval(row['selected_terms']) if isinstance(row['selected_terms'], str) else row['selected_terms']
    primary_expert = selected[0]['term'] if selected else "general"
    print(f"\nQuestion: {row['question_text']}")
    print(f"Route to: {primary_expert}")
    print(f"Confidence: {selected[0]['relevance_score'] if selected else 0}")
```

### Example 3: Knowledge Base Organization

```python
import pandas as pd
from langchain_openai import ChatOpenAI
from RAG_supporters.agents.domain_assesment import DomainAnalysisAgent, OperationMode
import json

# Initialize
llm = ChatOpenAI(model="gpt-4", temperature=0.3)
agent = DomainAnalysisAgent(llm=llm)

# Load knowledge base articles
kb_df = pd.read_csv("knowledge_base.csv")

# Extract domains from all articles
print("Extracting domains from knowledge base...")
result_df = agent.process_dataframe(
    kb_df,
    mode=OperationMode.EXTRACT,
    text_source_col="content",
    save_path="kb_with_domains.csv",
    use_batch_processing=True,
    batch_size=20
)

# Build domain taxonomy
domain_taxonomy = {}
for idx, row in result_df.iterrows():
    try:
        suggestions = eval(row['suggestions']) if isinstance(row['suggestions'], str) else row['suggestions']
        for suggestion in suggestions:
            term = suggestion['term']
            term_type = suggestion['type']
            if term_type not in domain_taxonomy:
                domain_taxonomy[term_type] = set()
            domain_taxonomy[term_type].add(term)
    except:
        pass

# Convert sets to lists for JSON serialization
domain_taxonomy = {k: list(v) for k, v in domain_taxonomy.items()}

# Save taxonomy
with open("domain_taxonomy.json", "w") as f:
    json.dump(domain_taxonomy, f, indent=2)

print(f"\nDomain Taxonomy:")
print(f"Domains: {len(domain_taxonomy.get('domain', []))}")
print(f"Subdomains: {len(domain_taxonomy.get('subdomain', []))}")
print(f"Keywords: {len(domain_taxonomy.get('keyword', []))}")
```

### Example 4: Topic Relevance Assessment with Cluster Descriptors

```python
import pandas as pd
from langchain_openai import ChatOpenAI
from RAG_supporters.agents.domain_assesment import DomainAnalysisAgent

# Initialize agent with reasoning enabled
llm = ChatOpenAI(model="gpt-4", temperature=0.3)
agent = DomainAnalysisAgent(llm=llm, include_reason=True)

# Topic descriptors from keyword clustering
topic_descriptors = [
    "machine learning and neural networks",
    "web development and frameworks",
    "database systems and SQL",
    "cloud computing and DevOps",
    "mobile app development"
]

# Questions to route
questions_df = pd.DataFrame({
    "id": [1, 2, 3, 4],
    "question": [
        "How does backpropagation work in deep learning?",
        "What is the difference between REST and GraphQL?",
        "How do I optimize PostgreSQL queries?",
        "What are the best practices for Kubernetes deployments?"
    ]
})

# Assess topic relevance for each question
results = []
for idx, row in questions_df.iterrows():
    result = agent.assess_topic_relevance_prob(
        row['question'],
        topic_descriptors
    )
    
    if result:
        # Find top topic
        top_topic = max(result['topic_scores'], key=lambda x: x['probability'])
        results.append({
            'question': row['question'],
            'top_topic': top_topic['topic_descriptor'],
            'probability': top_topic['probability'],
            'reason': top_topic.get('reason', 'N/A')
        })

results_df = pd.DataFrame(results)

# Display routing results
for idx, row in results_df.iterrows():
    print(f"\nQuestion: {row['question']}")
    print(f"Best Match: {row['top_topic']}")
    print(f"Probability: {row['probability']:.2f}")
    print(f"Reason: {row['reason']}")

# Batch processing for efficiency
print("\n--- Using Batch Processing ---")
batch_results = agent.assess_topic_relevance_prob_batch(
    questions_df['question'].tolist(),
    topic_descriptors,
    show_progress=True
)

# Process batch results
for question, result in zip(questions_df['question'], batch_results):
    if result:
        top_topic = max(result['topic_scores'], key=lambda x: x['probability'])
        print(f"\n{question}")
        print(f"  → {top_topic['topic_descriptor']} ({top_topic['probability']:.2f})")
```

## Expected Data Formats

### For EXTRACT Mode

Input DataFrame columns:
- `text_source_col`: Column containing text to analyze (e.g., "content", "article_text")

Output columns added:
- `suggestions`: List of domain suggestions (as string)
- `total_suggestions`: Number of suggestions
- `primary_theme`: Main identified theme
- `domain_analysis_error`: Error message if any

### For GUESS Mode

Input DataFrame columns:
- `question_col`: Column containing questions (e.g., "question_text")

Output columns added:
- `suggestions`: List of domain suggestions (as string)
- `total_suggestions`: Number of suggestions
- `question_category`: Identified question type/category
- `domain_analysis_error`: Error message if any

### For ASSESS Mode

Input DataFrame columns:
- `question_col`: Column containing questions
- `available_terms`: List of available domain terms (same for all rows)

Output columns added:
- `selected_terms`: List of selected relevant terms (as string)
- `total_selected`: Number of selected terms
- `question_intent`: Brief description of question intent
- `primary_topics`: Primary topics identified (as string)
- `domain_analysis_error`: Error message if any

## Workflow Architecture

The agent uses LangGraph for robust processing:

1. **Analyze**: Sends prompt to LLM based on operation mode
2. **Validate**: Validates response against Pydantic models
3. **Retry**: Automatically retries on parsing errors (up to max_retries)

### State Management

The agent maintains state through `AgentState` containing:
- `mode`: Operation mode (EXTRACT, GUESS, or ASSESS)
- `text_source`: Source text (for EXTRACT)
- `question`: Question text (for GUESS and ASSESS)
- `available_terms`: Available terms JSON (for ASSESS)
- `result`: Parsed result
- `error`: Error message if any
- `retry_count`: Current retry count
- `max_retries`: Maximum retry attempts

## API Reference

### DomainAnalysisAgent

```python
class DomainAnalysisAgent:
    def __init__(
        self,
        llm: BaseChatModel,
        max_retries: int = 3,
        batch_size: int = 10,
        include_reason: bool = False
    )
```

**Parameters:**
- `llm`: Language model for analysis
- `max_retries`: Maximum retries for parsing errors (default: 3)
- `batch_size`: Default batch size for batch processing (default: 10)
- `include_reason`: If True, include reasoning explanations in topic relevance assessments. Only applies to `assess_topic_relevance_prob` operations (TOPIC_RELEVANCE_PROB mode). Default is False. Setting to True increases token usage but provides explanations for probability scores.

#### Single Operation Methods

##### extract_domains

```python
extract_domains(text_source: str) -> Optional[Dict[str, Any]]
```

Extract domains, subdomains, and keywords from text.

**Returns:** Dictionary with `suggestions`, `total_suggestions`, and `primary_theme`

##### guess_domains

```python
guess_domains(question: str) -> Optional[Dict[str, Any]]
```

Guess domains needed to answer a question.

**Returns:** Dictionary with `suggestions`, `total_suggestions`, and `question_category`

##### assess_domains

```python
assess_domains(
    question: str,
    available_terms: Union[List[str], List[Dict], str]
) -> Optional[Dict[str, Any]]
```

Assess which available terms are most relevant to a question.

**Returns:** Dictionary with `selected_terms`, `total_selected`, `question_intent`, and `primary_topics`

##### assess_topic_relevance_prob

```python
assess_topic_relevance_prob(
    question: str,
    topic_descriptors: Union[List[str], List[Dict], str, Dict]
) -> Optional[Dict[str, Any]]
```

Assess relevance probabilities between a question and topic descriptors.

This method is designed to work with cluster descriptors from KeywordClusterer. It automatically handles:
- List of strings: `["topic1", "topic2"]`
- JSON string: `'["topic1", "topic2"]'`
- File path to JSON: `"path/to/clusters.json"`
- KeywordClusterer dict format with 'cluster_stats'

**Parameters:**
- `question`: The question to analyze
- `topic_descriptors`: Topic descriptors in any supported format

**Returns:** Dictionary with `topic_scores` (containing topic_descriptor, probability, and optionally reason if `include_reason=True`), `total_topics`, and optional `question_summary`

#### Batch Processing Methods

##### extract_domains_batch

```python
extract_domains_batch(
    text_sources: List[str]
) -> List[Optional[Dict[str, Any]]]
```

Extract domains from multiple texts in batch.

##### guess_domains_batch

```python
guess_domains_batch(
    questions: List[str]
) -> List[Optional[Dict[str, Any]]]
```

Guess domains for multiple questions in batch.

##### assess_domains_batch

```python
assess_domains_batch(
    questions: List[str],
    available_terms: Union[List[str], List[Dict], str]
) -> List[Optional[Dict[str, Any]]]
```

Assess multiple questions against available terms in batch.

##### assess_topic_relevance_prob_batch

```python
assess_topic_relevance_prob_batch(
    questions: List[str],
    topic_descriptors: Union[List[str], List[Dict], str, Dict],
    batch_size: Optional[int] = None,
    show_progress: bool = True
) -> List[Optional[Dict[str, Any]]]
```

Assess relevance probabilities for multiple questions against topic descriptors in batch.

**Parameters:**
- `questions`: List of questions to analyze
- `topic_descriptors`: Topic descriptors in any supported format (shared across all questions)
- `batch_size`: Batch size for processing (default: uses agent's batch_size)
- `show_progress`: Show progress bar (default: True)

**Returns:** List of dictionaries, each with `topic_scores`, `total_topics`, and optional `question_summary`

#### DataFrame Processing

##### process_dataframe

```python
process_dataframe(
    df: pd.DataFrame,
    mode: OperationMode,
    text_source_col: Optional[str] = None,
    question_col: Optional[str] = None,
    available_terms: Optional[Union[List[str], List[Dict], str]] = None,
    progress_bar: bool = True,
    save_path: Optional[str] = None,
    skip_existing: bool = True,
    checkpoint_batch_size: Optional[int] = None,
    use_batch_processing: bool = True,
    batch_size: Optional[int] = None
) -> pd.DataFrame
```

Process entire DataFrame based on operation mode.

**Parameters:**
- `df`: DataFrame to process
- `mode`: OperationMode (EXTRACT, GUESS, or ASSESS)
- `text_source_col`: Column name for text sources (EXTRACT mode)
- `question_col`: Column name for questions (GUESS and ASSESS modes)
- `available_terms`: Available terms (ASSESS mode)
- `progress_bar`: Show progress bar (default: True)
- `save_path`: Path to save results
- `skip_existing`: Skip rows with existing results (default: True)
- `checkpoint_batch_size`: Save checkpoint every N rows
- `use_batch_processing`: Use batch API if available (default: True)
- `batch_size`: Batch size for processing

**Returns:** DataFrame with added result columns

## Best Practices

1. **Choose Appropriate Mode**: 
   - Use EXTRACT for categorizing content
   - Use GUESS for question routing
   - Use ASSESS for matching questions to knowledge bases
   - Use TOPIC_RELEVANCE_PROB for calculating semantic similarity probabilities with cluster descriptors

2. **Set Reasonable Temperature**: Use 0.3-0.5 for consistent results

3. **Enable Batch Processing**: For OpenAI models, batch processing is much faster

4. **Use Checkpoints**: For large datasets, enable checkpoint_batch_size

5. **Skip Existing Results**: Use skip_existing=True to avoid reprocessing

6. **Monitor Costs**: Each operation requires 1 LLM call (+ retries if parsing fails)

7. **Handle Structured Output**: Results contain stringified Python objects; use `eval()` or `json.loads()` carefully

8. **Control Reasoning Explanations**: Set `include_reason=True` when initializing the agent if you need explanations for topic relevance probabilities. Note that this increases token usage but provides valuable insights into the model's decision-making process. Only applies to TOPIC_RELEVANCE_PROB mode.

## Performance Considerations

- **Batch Processing**: 5-10x faster for OpenAI models
- **Single Operation**: ~1-2 seconds per item
- **Batch Operation**: ~0.2-0.5 seconds per item (OpenAI)
- **Memory Usage**: Loads entire DataFrame into memory
- **API Costs**: Each item requires 1-2 LLM calls

## Troubleshooting

### Common Issues

**Issue: All results return None**
- Check LLM API credentials
- Verify network connectivity
- Ensure input text/questions are not empty
- Check max_retries setting

**Issue: Parsing errors**
- The agent has automatic retry with OutputFixingParser
- Increase max_retries if needed
- Check LLM model capabilities (use GPT-4 for better structured output)

**Issue: Slow processing**
- Enable batch processing: `use_batch_processing=True`
- Increase batch_size for OpenAI models
- Use faster model (gpt-3.5-turbo vs gpt-4)

**Issue: DataFrame columns not found**
- Verify column names match parameters
- Check for typos in column names
- Ensure required columns exist for the mode

## Logging

Enable logging to track progress:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## See Also

- **SourceEvaluationAgent**: For evaluating source quality with detailed scores
- **DatasetCheckAgent**: For comparing and selecting best sources
- **QuestionAugmentationAgent**: For generating and rephrasing questions
