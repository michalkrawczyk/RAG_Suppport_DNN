# RAG Supporters Agents

> **📚 For comprehensive documentation, workflows, best practices, and detailed examples, see [AGENTS_OVERVIEW.md](../../docs/agents/AGENTS_OVERVIEW.md)**

This directory contains specialized LLM-powered agents for RAG (Retrieval-Augmented Generation) dataset creation, curation, and enhancement.

## Available Agents

| Agent | Purpose | Documentation |
|-------|---------|---------------|
| **QuestionAugmentationAgent** | Question generation and rephrasing | [CSV_QUESTION_AGENT.md](../../docs/agents/CSV_QUESTION_AGENT.md) |
| **TextAugmentationAgent** | Text augmentation while preserving meaning | [TEXT_AUGMENTATION.md](../../docs/agents/TEXT_AUGMENTATION.md) |
| **DatasetCheckAgent** | Source comparison and quality control | [DATASET_CHECK_AGENT.md](../../docs/agents/DATASET_CHECK_AGENT.md) |
| **DomainAnalysisAgent** | Domain extraction, guessing, and assessment | [DOMAIN_ANALYSIS_AGENT.md](../../docs/agents/DOMAIN_ANALYSIS_AGENT.md) |
| **SourceEvaluationAgent** | Multi-dimensional source quality scoring | [SOURCE_EVALUATION_AGENT.md](../../docs/agents/SOURCE_EVALUATION_AGENT.md) |

## Quick Start

```python
from langchain_openai import ChatOpenAI
from RAG_supporters.agents import QuestionAugmentationAgent

# Initialize with an LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.3)
agent = QuestionAugmentationAgent(llm=llm)

# Use the agent
rephrased = agent.rephrase_question_with_source(
    question="What does it do?",
    source="Mitochondria produce ATP through cellular respiration..."
)
# Output: "What is the primary function of mitochondria?"
```

**→ See [AGENTS_OVERVIEW.md](../../docs/agents/AGENTS_OVERVIEW.md) for complete workflows, best practices, and advanced usage**

## Installation

```bash
# Install with LLM provider (choose one)
pip install -e .[openai]   # For OpenAI support
pip install -e .[nvidia]   # For NVIDIA support
pip install -e .           # Base agents only
```

**→ See [AGENTS_OVERVIEW.md#installation](../../docs/agents/AGENTS_OVERVIEW.md#installation) for detailed setup instructions**

## Testing

All agents have comprehensive unit tests in `../../tests/`:

```bash
# Run all tests
pytest tests/

# Run specific agent tests
pytest tests/test_question_augmentation_agent.py -v
```

**→ See [AGENTS_OVERVIEW.md#testing](../../docs/agents/AGENTS_OVERVIEW.md#testing) for complete testing guidelines and best practices**

## Contributing

**All new agents MUST include:**
1. Comprehensive unit tests (see existing test files for examples)
2. Documentation in `../../docs/agents/`
3. Updates to this README and AGENTS_OVERVIEW.md

**→ See [AGENTS_OVERVIEW.md#contributing](../../docs/agents/AGENTS_OVERVIEW.md#contributing) and [AGENTS.md](../../AGENTS.md) for detailed guidelines**

## Additional Resources

- **[AGENTS_OVERVIEW.md](../../docs/agents/AGENTS_OVERVIEW.md)** - Complete documentation, workflows, examples, and best practices
- **[AGENTS.md](../../AGENTS.md)** - Technical guidelines, architecture, and standards
- **[pyproject.toml](../../pyproject.toml)** - Package configuration with optional dependencies
