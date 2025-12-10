# Documentation Index

This directory contains comprehensive documentation for all RAG Supporters components.

## Agent Documentation

The RAG Supporters library provides specialized LLM-powered agents for RAG dataset creation and curation.

### Overview
- **[AGENTS_OVERVIEW.md](AGENTS_OVERVIEW.md)** - Complete overview of all agents with comparison matrix, workflows, and best practices

### Individual Agent Documentation

1. **[QuestionAugmentationAgent](CSV_QUESTION_AGENT.md)** - Question generation and rephrasing
   - Rephrase questions to align with sources or domains
   - Generate alternative questions from sources
   - Batch CSV/DataFrame processing

2. **[TextAugmentationAgent](TEXT_AUGMENTATION.md)** - Text augmentation while preserving meaning
   - Full text and sentence-level rephrasing
   - Dataset augmentation for training data
   - Configurable augmentation modes

3. **[DatasetCheckAgent](DATASET_CHECK_AGENT.md)** - Source comparison and selection
   - Compare two sources for a question
   - Duplicate detection
   - Quality control workflows

4. **[DomainAnalysisAgent](DOMAIN_ANALYSIS_AGENT.md)** - Domain extraction and assessment
   - Extract domains from text (EXTRACT mode)
   - Guess domains needed for questions (GUESS mode)
   - Assess question relevance to domains (ASSESS mode)

5. **[SourceEvaluationAgent](SOURCE_EVALUATION_AGENT.md)** - Multi-dimensional source quality scoring
   - 6-dimensional evaluation (relevance, expertise, depth, clarity, objectivity, completeness)
   - Source ranking and quality control
   - Batch processing support

## Quick Start

```python
from langchain_openai import ChatOpenAI
from RAG_supporters.agents import (
    QuestionAugmentationAgent,
    TextAugmentationAgent,
    DatasetCheckAgent,
    DomainAnalysisAgent,
    SourceEvaluationAgent
)

# Initialize any agent
llm = ChatOpenAI(model="gpt-4")
agent = QuestionAugmentationAgent(llm=llm)
```

## Installation

```bash
# Install agent dependencies
pip install -r RAG_supporters/requirements_agents.txt

# Or install individually
pip install langchain langgraph langchain-core pydantic pandas tqdm
```

## Documentation Structure

Each agent documentation includes:
- **Overview** - Purpose and key features
- **Installation** - Dependencies and setup
- **Basic Usage** - Getting started examples
- **Advanced Usage** - Complex scenarios and configurations
- **DataFrame/CSV Processing** - Batch processing examples
- **Complete Examples** - Real-world use cases
- **API Reference** - Detailed method documentation
- **Best Practices** - Recommendations and tips
- **Troubleshooting** - Common issues and solutions

## Other Documentation

- **[SAMPLE_GENERATION_GUIDE.md](../RAG_supporters/dataset/SAMPLE_GENERATION_GUIDE.md)** - Guide for generating RAG dataset samples
- **[QUICK_REFERENCE.md](../RAG_supporters/dataset/QUICK_REFERENCE.md)** - Quick reference for dataset utilities

## Contributing

When adding new agents or features:
1. Follow the existing documentation structure
2. Include practical examples
3. Document all public methods
4. Add cross-references to related agents
5. Update this README and AGENTS_OVERVIEW.md

## Support

For questions or issues:
- Check the specific agent documentation
- Review the troubleshooting sections
- Consult AGENTS_OVERVIEW.md for workflows and patterns
