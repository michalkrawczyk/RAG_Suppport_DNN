# Agent Documentation

This directory contains comprehensive documentation for all RAG Supporters agents.

## Overview

- **[AGENTS_OVERVIEW.md](AGENTS_OVERVIEW.md)** - Complete overview of all agents with comparison matrix, workflows, and best practices

## Individual Agent Documentation

### 1. QuestionAugmentationAgent
**[CSV_QUESTION_AGENT.md](CSV_QUESTION_AGENT.md)**

Question generation and rephrasing in CSV/DataFrame contexts.

**Key Features:**
- Rephrase questions to align with sources or domains
- Generate alternative questions from sources
- Batch CSV/DataFrame processing

---

### 2. TextAugmentationAgent
**[TEXT_AUGMENTATION.md](TEXT_AUGMENTATION.md)**

Text augmentation while preserving meaning.

**Key Features:**
- Full text and sentence-level rephrasing
- Dataset augmentation for training data
- Configurable augmentation modes

---

### 3. DatasetCheckAgent
**[DATASET_CHECK_AGENT.md](DATASET_CHECK_AGENT.md)**

Source comparison and quality control.

**Key Features:**
- Compare two sources for a question
- Duplicate detection
- LangGraph-based workflow

---

### 4. DomainAnalysisAgent
**[DOMAIN_ANALYSIS_AGENT.md](DOMAIN_ANALYSIS_AGENT.md)**

Domain extraction, guessing, and assessment.

**Key Features:**
- Extract domains from text (EXTRACT mode)
- Guess domains for questions (GUESS mode)
- Assess question-domain relevance (ASSESS mode)

---

### 5. SourceEvaluationAgent
**[SOURCE_EVALUATION_AGENT.md](SOURCE_EVALUATION_AGENT.md)**

Multi-dimensional source quality scoring.

**Key Features:**
- 6-dimensional evaluation (relevance, expertise, depth, clarity, objectivity, completeness)
- Source ranking and quality control
- Batch processing support

---

## Quick Reference

| Agent | Primary Task | Documentation |
|-------|-------------|---------------|
| QuestionAugmentationAgent | Question operations | [CSV_QUESTION_AGENT.md](CSV_QUESTION_AGENT.md) |
| TextAugmentationAgent | Text augmentation | [TEXT_AUGMENTATION.md](TEXT_AUGMENTATION.md) |
| DatasetCheckAgent | Source comparison | [DATASET_CHECK_AGENT.md](DATASET_CHECK_AGENT.md) |
| DomainAnalysisAgent | Domain analysis | [DOMAIN_ANALYSIS_AGENT.md](DOMAIN_ANALYSIS_AGENT.md) |
| SourceEvaluationAgent | Quality scoring | [SOURCE_EVALUATION_AGENT.md](SOURCE_EVALUATION_AGENT.md) |

## Documentation Structure

Each agent documentation includes:
- **Overview** - Purpose and key features
- **Installation** - Dependencies and setup
- **Basic Usage** - Getting started examples
- **Advanced Usage** - Complex scenarios and configurations
- **DataFrame/CSV Processing** - Batch processing examples
- **Complete Examples** - Real-world use cases (3-4 per agent)
- **API Reference** - Detailed method documentation
- **Best Practices** - Recommendations and tips
- **Troubleshooting** - Common issues and solutions
- **Cross-references** - Links to related agents

## Getting Started

1. Start with [AGENTS_OVERVIEW.md](AGENTS_OVERVIEW.md) for a complete overview
2. Choose the agent that fits your use case
3. Follow the installation instructions in the specific agent documentation
4. Review the examples and adapt them to your needs

## Installation

```bash
# Install with LLM provider (choose one)
pip install -e .[openai]   # For OpenAI support
pip install -e .[nvidia]   # For NVIDIA support
pip install -e .           # Base agents only

# Or install individually (if not using pyproject.toml)
pip install langgraph pydantic pandas tqdm
pip install langchain-openai  # For OpenAI
```

## Support

For questions or issues:
- Check the specific agent documentation
- Review the troubleshooting sections
- Consult [AGENTS_OVERVIEW.md](AGENTS_OVERVIEW.md) for workflows and patterns
