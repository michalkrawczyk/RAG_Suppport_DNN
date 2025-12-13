# GitHub Copilot Instructions for RAG Support DNN

## Project Overview

This repository contains **RAG Support DNN** - a Python library for supporting Retrieval-Augmented Generation (RAG) systems with deep neural networks. The project provides tools for dataset generation, text augmentation, clustering, embeddings, and LangChain-based agents for RAG workflows.

### Key Components

- **Agents** (`RAG_supporters/agents/`) - LangChain-based agents for text/question augmentation, domain assessment, and dataset validation
- **Augmentations** (`RAG_supporters/augmentations/`) - Text and embedding augmentation utilities
- **Clustering** (`RAG_supporters/clustering/`) - Keyword/topic clustering with K-means and bisecting K-means
- **Dataset** (`RAG_supporters/dataset/`) - RAG dataset templates, sample generation, and PyTorch dataset wrappers
- **Embeddings** (`RAG_supporters/embeddings/`) - Keyword embedder and embedding I/O utilities
- **Neural Networks** (`RAG_supporters/nn/`) - Model builders and neural network components
- **Prompts** (`RAG_supporters/prompts_templates/`) - LangChain prompt templates for various agents
- **Utils** (`RAG_supporters/utils/`) - Text splitters, suggestion processing, and utility functions

## Code Style and Standards

### Python Style

- **Formatter**: Use `black` for code formatting (enforced in CI)
- **Linter**: Use `pylint` for code quality checks (runs on changed files)
- **Docstrings**: All public functions, classes, and modules must have docstrings following `pydocstyle` conventions
- **Python Version**: Target Python 3.11+

### Code Quality Requirements

1. **All code must be formatted with Black** - Run `black <file>` before committing
2. **All public APIs must have docstrings** - Use Google-style or NumPy-style docstrings
3. **Pass pylint checks** - Address pylint warnings on changed files
4. **Avoid TODOs in production code** - TODOs are flagged in CI (allowed but warned)

### Example Code Style

```python
"""Module docstring describing the module purpose."""

import typing
from typing import Optional, List, Dict

import pandas as pd
from langchain_core.language_models import BaseChatModel


class TextAugmentationAgent:
    """Agent for augmenting text with LLM-based rephrasing.
    
    This agent provides methods to rephrase entire texts or individual
    sentences while preserving semantic meaning.
    
    Args:
        llm: LangChain chat model for text generation
        verify_meaning: Whether to verify meaning preservation
        max_retries: Maximum retry attempts for LLM calls
    """
    
    def __init__(
        self,
        llm: BaseChatModel,
        verify_meaning: bool = False,
        max_retries: int = 3,
    ):
        """Initialize the text augmentation agent."""
        self.llm = llm
        self.verify_meaning = verify_meaning
        self.max_retries = max_retries
```

## Dependencies and Environment

### Core Dependencies

- **PyTorch** (`torch>=2.3.0`) - Neural network framework
- **Scikit-learn** (`scikit-learn>=1.3.0`) - Clustering and ML utilities
- **Pandas** (`pandas>=2.2.3`) - Data manipulation
- **NumPy** (`numpy>=2.0.2`) - Numerical operations
- **Datasets** (`datasets>=2.19.2`) - HuggingFace datasets
- **PyYAML** (`pyyaml>=6.0`) - Configuration parsing

### Optional Agent Dependencies

- **LangChain** (`langchain>=0.3.20`, `langchain-core==0.3.52`) - LLM agent framework
- **LangChain Community** (`langchain-community>=0.3.19`) - Additional integrations
- **LangChain Chroma** (`langchain-chroma>=0.2.3`) - Vector store support
- **LangChain OpenAI** (`langchain-openai==0.3.7`) - OpenAI integration
- **LangGraph** (`langgraph==0.3.8`) - Agent workflow graphs
- **Pydantic** (`pydantic==2.10.3`) - Data validation

### Installing Dependencies

```bash
# Core dependencies (mandatory)
pip install -r RAG_supporters/requirements.txt

# Agent dependencies (optional)
pip install -r RAG_supporters/requirements_agents.txt

# Development dependencies (for testing and linting)
pip install -r RAG_supporters/requirements-dev.txt
```

### Special Note: PyTorch CPU Version

For CI/CD and testing, use PyTorch CPU version to avoid slow CUDA installation:

```bash
pip install 'torch>=2.3.0' --index-url https://download.pytorch.org/whl/cpu
```

This is automatically handled in the dependency check workflow.

## Architecture Patterns

### LangChain Agents

When creating new agents in `RAG_supporters/agents/`:

1. **Inherit from or use LangChain base classes** - Use `BaseChatModel` for LLM integration
2. **Define prompts in separate templates** - Store prompt templates in `RAG_supporters/prompts_templates/`
3. **Implement retry logic** - Use `max_retries` parameter for LLM call resilience
4. **Support CSV/DataFrame processing** - Provide batch processing methods for datasets
5. **Use typing annotations** - All methods should have proper type hints

Example agent structure:

```python
from typing import Optional, List
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate


class MyAgent:
    """Agent description."""
    
    def __init__(self, llm: BaseChatModel, max_retries: int = 3):
        self.llm = llm
        self.max_retries = max_retries
    
    def process_single(self, text: str) -> Optional[str]:
        """Process a single text item."""
        pass
    
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process entire DataFrame with progress bar."""
        pass
    
    def process_csv(self, input_path: str, output_path: str):
        """Convenience method for CSV processing."""
        pass
```

### Dataset Patterns

When working with datasets in `RAG_supporters/dataset/`:

1. **Use template base classes** - Extend existing dataset templates
2. **Support multiple sample types** - Implement pair, triplet, and custom sample generation
3. **Optimize memory usage** - Use batch processing for large datasets
4. **Provide clear documentation** - Document sample types and parameters
5. **Return pandas DataFrames** - Standard output format for samples

### Clustering Patterns

When implementing clustering features:

1. **Support multiple algorithms** - K-means, bisecting K-means, etc.
2. **Provide topic descriptors** - Extract meaningful cluster representatives
3. **Support both hard and soft assignment** - One-hot and probabilistic clustering
4. **Save/load results** - Implement JSON serialization for results
5. **Use cosine similarity for text** - Prefer cosine over Euclidean for embeddings

## Testing

### Test Structure

- Tests are located alongside the modules they test (e.g., `test_keyword_clustering.py`)
- Use descriptive test names that explain what is being tested
- Tests should be independent and not rely on external resources

### Running Tests

```bash
# Run specific test file
python -m pytest RAG_supporters/clustering/test_keyword_clustering.py

# Run all tests
python -m pytest RAG_supporters/
```

### Writing Tests

```python
import pytest
from RAG_supporters.clustering import KeywordClusterer


def test_keyword_clusterer_initialization():
    """Test that KeywordClusterer initializes with correct defaults."""
    clusterer = KeywordClusterer(n_clusters=5)
    assert clusterer.n_clusters == 5
    assert clusterer.algorithm == "kmeans"


def test_keyword_clustering_fit():
    """Test that clustering can fit on sample embeddings."""
    # Test implementation
    pass
```

## Common Patterns and Best Practices

### 1. CSV Processing with Progress Bars

Always use `tqdm` for long-running operations:

```python
from tqdm import tqdm
import pandas as pd


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Process DataFrame with progress indication."""
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        # Process row
        result = process_row(row)
        results.append(result)
    return pd.DataFrame(results)
```

### 2. Column Mapping for Flexibility

Support custom column names in CSV processing:

```python
def process_csv(
    df: pd.DataFrame,
    columns_mapping: Optional[dict] = None
) -> pd.DataFrame:
    """Process CSV with flexible column mapping."""
    default_mapping = {
        "question_text": "question_text",
        "source_text": "source_text"
    }
    mapping = columns_mapping or default_mapping
    
    question_col = mapping["question_text"]
    source_col = mapping["source_text"]
    
    # Use mapped columns
    questions = df[question_col]
    sources = df[source_col]
```

### 3. Error Handling for LLM Calls

Implement retry logic with graceful degradation:

```python
def call_llm_with_retry(self, prompt: str) -> Optional[str]:
    """Call LLM with retry logic."""
    for attempt in range(self.max_retries):
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            if attempt == self.max_retries - 1:
                print(f"Failed after {self.max_retries} attempts: {e}")
                return None
            print(f"Attempt {attempt + 1} failed, retrying...")
```

### 4. Save and Load Results

Implement JSON serialization for persistence:

```python
import json


def save_results(self, output_path: str):
    """Save results to JSON file."""
    results = {
        "metadata": {...},
        "data": {...}
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


@classmethod
def from_results(cls, results_path: str):
    """Load results from JSON file."""
    with open(results_path, "r") as f:
        results = json.load(f)
    # Reconstruct object from results
    return cls(...)
```

## CI/CD Workflows

### Python Checks Workflow

Runs on all PRs and pushes to `main`:

- **Black formatting** - Enforces code style on changed files
- **Pylint** - Checks code quality on changed files
- **Pydocstyle** - Validates docstrings on changed files
- **TODO detection** - Flags TODO comments (warning only)

### Dependency Check Workflow

Runs when dependency files change:

- Installs core dependencies from `requirements.txt`
- Installs PyTorch CPU version for faster CI
- Validates dependency compatibility with `pip check`
- Optionally installs and checks agent dependencies
- Generates dependency tree with `pipdeptree`

## Documentation

### Adding Documentation

- **Module documentation** goes in `docs/` as Markdown files
- **API documentation** should be in docstrings
- **Quick references** for complex features (see `RAG_supporters/dataset/QUICK_REFERENCE.md`)
- **Comprehensive guides** for major features (see `docs/CSV_QUESTION_AGENT.md`)

### Documentation Structure

1. **Overview** - High-level description and use cases
2. **Installation** - Required dependencies
3. **Basic Usage** - Simple examples to get started
4. **Advanced Usage** - Complex scenarios and configurations
5. **API Reference** - Detailed method signatures and parameters
6. **Best Practices** - Recommendations and patterns
7. **Troubleshooting** - Common issues and solutions

## Key Files to Know

- `RAG_supporters/requirements.txt` - Core dependencies
- `RAG_supporters/requirements_agents.txt` - Optional LangChain dependencies
- `RAG_supporters/requirements-dev.txt` - Development dependencies
- `.github/workflows/python-checks.yml` - CI for code quality
- `.github/workflows/dependency-check.yml` - CI for dependencies
- `docs/` - User-facing documentation
- `RAG_supporters/dataset/templates/` - Dataset templates

## Special Considerations

### 1. Memory Management

- Large datasets should use batch processing
- Provide `batch_size` parameters for memory-intensive operations
- Document memory requirements in docstrings

### 2. Embedding Operations

- Use consistent embedding formats (dict mapping text to vectors)
- Support both numpy arrays and lists for embeddings
- Document embedding dimensions and formats

### 3. LangChain Integration

- Keep LangChain dependencies optional when possible
- Gracefully handle missing LangChain installations
- Use type hints from `langchain-core` for compatibility

### 4. Dataset Generation

- Support multiple sample types (pairs, triplets)
- Provide clear parameter documentation
- Use pandas DataFrames as standard output format

## When Adding New Features

1. **Follow existing patterns** - Look at similar modules for guidance
2. **Add comprehensive docstrings** - Document all public APIs
3. **Format with Black** - Run `black` before committing
4. **Add tests** - Write unit tests for new functionality
5. **Update documentation** - Add or update relevant docs
6. **Check dependencies** - Minimize new dependencies
7. **Consider memory usage** - Optimize for large datasets
8. **Provide examples** - Include usage examples in docstrings or docs

## Issue #43 Considerations

When working on features or changes, consider:

- **Dataset augmentation workflows** - Support for CSV-based question/answer generation
- **LangChain agent integration** - Consistent patterns across all agents
- **Batch processing efficiency** - Memory-optimized operations for large datasets
- **Flexible configuration** - Support for custom column mappings and parameters
- **Reproducibility** - Save/load functionality for intermediate results

## Need Help?

- Review existing agent implementations in `RAG_supporters/agents/`
- Check documentation in `docs/` for usage patterns
- Look at test files for examples of expected behavior
- Refer to CI workflows for quality standards
