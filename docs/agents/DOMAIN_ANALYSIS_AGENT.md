# DomainAnalysisAgent Documentation

## Overview

The `DomainAnalysisAgent` is a unified LangGraph-based agent for comprehensive domain analysis tasks. It provides three distinct operation modes for extracting domains from text, guessing required domains for questions, and assessing question relevance to available domain terms. This agent is essential for domain classification, knowledge base organization, and question-domain matching in RAG applications.

### What It Does

The agent provides three powerful operation modes:
- **EXTRACT Mode**: Extract domains, subdomains, and keywords from source text
- **GUESS Mode**: Identify domains needed to answer a given question
- **ASSESS Mode**: Evaluate which available domain terms are most relevant to a question

### Key Features

- **Three Operation Modes**: Extract, Guess, and Assess domain information
- **Structured Output**: Returns validated Pydantic models with scores and reasoning
- **Batch Processing**: Efficiently process multiple texts or questions
- **LangGraph Architecture**: Robust workflow with automatic retry logic
- **Flexible Input**: Works with text, DataFrames, and CSV files
- **OpenAI Batch Support**: Optimized batch processing for OpenAI models

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
pip install -r RAG_supporters/requirements_agents.txt
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

## Basic Usage

### Initialize the Agent

```python
from langchain_openai import ChatOpenAI
from RAG_supporters.agents.domain_assesment import DomainAnalysisAgent

# Initialize with an LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.3)
agent = DomainAnalysisAgent(llm=llm, max_retries=3, batch_size=10)
```

### Single Text/Question Operations

```python
# Extract domains from text
result = agent.extract_domains("Your text here...")

# Guess domains for question
result = agent.guess_domains("Your question here?")

# Assess question against available terms
result = agent.assess_domains("Your question?", ["term1", "term2", "term3"])
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
        batch_size: int = 10
    )
```

**Parameters:**
- `llm`: Language model for analysis
- `max_retries`: Maximum retries for parsing errors (default: 3)
- `batch_size`: Default batch size for batch processing (default: 10)

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

2. **Set Reasonable Temperature**: Use 0.3-0.5 for consistent results

3. **Enable Batch Processing**: For OpenAI models, batch processing is much faster

4. **Use Checkpoints**: For large datasets, enable checkpoint_batch_size

5. **Skip Existing Results**: Use skip_existing=True to avoid reprocessing

6. **Monitor Costs**: Each operation requires 1 LLM call (+ retries if parsing fails)

7. **Handle Structured Output**: Results contain stringified Python objects; use `eval()` or `json.loads()` carefully

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
