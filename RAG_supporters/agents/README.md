# RAG Supporters Agents

This directory contains specialized LLM-powered agents for RAG (Retrieval-Augmented Generation) dataset creation, curation, and enhancement.

## Available Agents

### 1. QuestionAugmentationAgent
Question generation and rephrasing in CSV/DataFrame contexts.

```python
from RAG_supporters.agents import QuestionAugmentationAgent
```

**Key Features:**
- Rephrase questions with source or domain context
- Generate alternative questions from sources
- Batch CSV/DataFrame processing

### 2. TextAugmentationAgent
Text augmentation while preserving meaning.

```python
from RAG_supporters.agents import TextAugmentationAgent
```

**Key Features:**
- Full text and sentence-level rephrasing
- Dataset augmentation
- Configurable modes (full, sentence, random)

### 3. DatasetCheckAgent
Source comparison and quality control.

```python
from RAG_supporters.agents import DatasetCheckAgent
```

**Key Features:**
- Compare two sources for a question
- Duplicate detection
- LangGraph-based workflow

### 4. DomainAnalysisAgent
Domain extraction, guessing, and assessment.

```python
from RAG_supporters.agents import DomainAnalysisAgent, OperationMode
```

**Key Features:**
- Extract domains from text (EXTRACT mode)
- Guess domains for questions (GUESS mode)
- Assess question-domain relevance (ASSESS mode)

### 5. SourceEvaluationAgent
Multi-dimensional source quality scoring.

```python
from RAG_supporters.agents import SourceEvaluationAgent
```

**Key Features:**
- 6-dimensional evaluation scores
- Batch processing support
- Source ranking capabilities

## Documentation

**📚 Complete documentation is available in the [docs/agents](../../docs/agents/) folder:**

- **[AGENTS_OVERVIEW.md](../../docs/agents/AGENTS_OVERVIEW.md)** - Overview with workflows and patterns
- **[CSV_QUESTION_AGENT.md](../../docs/agents/CSV_QUESTION_AGENT.md)** - QuestionAugmentationAgent
- **[TEXT_AUGMENTATION.md](../../docs/agents/TEXT_AUGMENTATION.md)** - TextAugmentationAgent
- **[DATASET_CHECK_AGENT.md](../../docs/agents/DATASET_CHECK_AGENT.md)** - DatasetCheckAgent
- **[DOMAIN_ANALYSIS_AGENT.md](../../docs/agents/DOMAIN_ANALYSIS_AGENT.md)** - DomainAnalysisAgent
- **[SOURCE_EVALUATION_AGENT.md](../../docs/agents/SOURCE_EVALUATION_AGENT.md)** - SourceEvaluationAgent

## Quick Start

```python
from langchain_openai import ChatOpenAI
from RAG_supporters.agents import QuestionAugmentationAgent

# Initialize with an LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.3)
agent = QuestionAugmentationAgent(llm=llm)

# Rephrase a question with source context
rephrased = agent.rephrase_question_with_source(
    question="What does it do?",
    source="Mitochondria produce ATP through cellular respiration..."
)
print(rephrased)
# Output: "What is the primary function of mitochondria?"
```

## Installation

Install agent dependencies:

```bash
pip install -r requirements_agents.txt
```

Or install individually:

```bash
pip install langchain langgraph langchain-core pydantic pandas tqdm
```

For specific LLM providers:

```bash
# OpenAI
pip install langchain-openai

# Other providers
pip install langchain-<provider>
```

## Agent Comparison

| Agent | Primary Task | Input | Output | Batch |
|-------|-------------|-------|--------|-------|
| QuestionAugmentationAgent | Question operations | Question + Source/Domain | Rephrased/Generated questions | ✅ |
| TextAugmentationAgent | Text augmentation | Text | Rephrased text | ✅ |
| DatasetCheckAgent | Source comparison | Question + 2 Sources | Label (0/1/2/-1) | ✅ |
| DomainAnalysisAgent | Domain analysis | Text/Question/Terms | Domain suggestions | ✅ |
| SourceEvaluationAgent | Quality scoring | Question + Source | 6 dimension scores | ✅ |

## Common Patterns

### DataFrame Processing

All agents support DataFrame processing:

```python
import pandas as pd

df = pd.read_csv("data.csv")
result_df = agent.process_dataframe(
    df,
    # Agent-specific parameters
    save_path="results.csv",
    use_batch_processing=True
)
```

### Batch Processing

For OpenAI models, batch processing is significantly faster:

```python
# Single operation
result = agent.method(input)

# Batch operation (OpenAI only)
results = agent.method_batch(inputs)
```

## Example Workflow

```python
from langchain_openai import ChatOpenAI
from RAG_supporters.agents import (
    QuestionAugmentationAgent,
    SourceEvaluationAgent,
    DomainAnalysisAgent
)

llm = ChatOpenAI(model="gpt-4")

# 1. Generate questions from sources
qa_agent = QuestionAugmentationAgent(llm=llm)
qa_pairs = qa_agent.process_csv_generation(
    "sources.csv",
    "qa_pairs.csv",
    n_questions=5
)

# 2. Evaluate source quality
eval_agent = SourceEvaluationAgent(llm=llm)
evaluated = eval_agent.process_dataframe(
    qa_pairs,
    question_col="question_text",
    source_col="source_text"
)

# 3. Categorize by domain
domain_agent = DomainAnalysisAgent(llm=llm)
categorized = domain_agent.process_dataframe(
    evaluated,
    mode=OperationMode.EXTRACT,
    text_source_col="source_text"
)

# 4. Filter high-quality pairs
high_quality = categorized[categorized['avg_score'] >= 7.0]
high_quality.to_csv("final_dataset.csv", index=False)
```

## Best Practices

1. **Model Selection**
   - Use GPT-4 for evaluation tasks
   - Use GPT-3.5-turbo for generation tasks
   - Set temperature 0.2-0.3 for consistency

2. **Batch Processing**
   - Enable for OpenAI models: `use_batch_processing=True`
   - Adjust batch_size (10-50) based on rate limits

3. **Large Datasets**
   - Use checkpoints: `checkpoint_batch_size=100`
   - Enable `skip_existing=True` for resumable processing
   - Save intermediate results

4. **Error Handling**
   - Set appropriate `max_retries` (3-5)
   - Check for None returns
   - Monitor error columns in DataFrames

## Contributing

When adding new agents:
1. Follow existing patterns (initialization, batch processing, DataFrame support)
2. Add comprehensive documentation in [docs/](../../docs/)
3. Include usage examples
4. Update this README
5. Update [AGENTS_OVERVIEW.md](../../docs/AGENTS_OVERVIEW.md)

## Requirements

See [requirements_agents.txt](../requirements_agents.txt) for dependencies.

## License

See repository root for license information.
