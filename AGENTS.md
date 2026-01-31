# AGENTS.md - RAG Support DNN Project

## Project Context

This is a Python library for experiments on creating non-LLM solutions (specifically lightweight neural networks) for improving RAG (Retrieval-Augmented Generation) or creating new variants of it. The LLM-powered agents are used to support learning and prepare datasets for training these alternative models. It includes specialized agents for domain analysis, text augmentation, source evaluation, and dataset quality control.

## Core Technologies

- **Python 3.10+** required
- **PyTorch** for neural network components
- **LangChain/LangGraph** for agent orchestration
- **Pandas** for dataset manipulation
- **Scikit-learn** for clustering and embeddings
- **Pydantic** for data validation

## Commands

### Installation

```bash
# Install core dependencies only (no agents)
pip install -e .

# Install with agent support (choose one):
pip install -e .[openai]   # Agents with OpenAI support
pip install -e .[nvidia]   # Agents with NVIDIA support
pip install -e .[base]     # Agents only (no LLM provider)

# Install development tools (linting, testing, formatting)
pip install -e .[dev]
```

### Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_dataset_splitter.py

# Run with verbose output
pytest -v tests/

# Run with coverage
pytest --cov=RAG_supporters tests/
```

### Code Quality

```bash
# Format code with black
black RAG_supporters/ tests/

# Sort imports
isort RAG_supporters/ tests/

# Run linting
pylint RAG_supporters/
flake8 RAG_supporters/

# Type checking
mypy RAG_supporters/

# Check docstring style
pydocstyle RAG_supporters/
```

## Code Style & Conventions

### Import Organization

- **Standard library** imports first
- **Third-party** imports second (pandas, langchain, torch, etc.)
- **Local/relative** imports last
- Use `isort` to maintain consistent ordering

### Module Structure

Each module follows this pattern:
```python
"""Module docstring describing purpose."""

import standard_library
from typing import Any, Dict, List

import third_party
from third_party import SpecificClass

from local_module import LocalClass

LOGGER = logging.getLogger(__name__)

# Constants in UPPER_CASE
# Classes in PascalCase
# Functions/methods in snake_case
# Private members prefixed with _
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `DatasetCheckAgent`, `DomainAnalysisAgent`)
- **Functions/Methods**: `snake_case` (e.g., `rephrase_question_with_source`)
- **Variables**: `snake_case` (e.g., `train_indices`, `val_ratio`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `LOGGER`, `MAX_RETRIES`)
- **Private members**: Prefix with `_` (e.g., `_build_graph`, `_validate_input`)

### Type Hints

- Use type hints for all function signatures
- Import from `typing` module: `List`, `Dict`, `Optional`, `Union`, `Tuple`
- Example:
  ```python
  def process_dataframe(
      self,
      df: pd.DataFrame,
      question_col: str,
      source_col: str,
      skip_labeled: bool = True
  ) -> pd.DataFrame:
  ```

### Docstrings

- Use triple-quoted strings for all docstrings
- Follow **NumPy docstring style** for parameters, returns, and examples
- Include type information in docstrings
- Example:
  ```python
  def compare_text_sources(self, question: str, source1: str, source2: str) -> Dict[str, Any]:
      """Compare two text sources for a given question.
      
      Parameters
      ----------
      question : str
          The question to evaluate sources against
      source1 : str
          First source text
      source2 : str
          Second source text
          
      Returns
      -------
      Dict[str, Any]
          Dictionary containing comparison results with keys:
          - 'selected_source': 1 or 2
          - 'reason': Explanation of selection
          - 'label': Numerical label
      """
  ```

### Error Handling

- Use try/except blocks for external dependencies (LangChain, LLM calls)
- Provide helpful fallback messages when optional dependencies are missing
- Log errors using the module-level `LOGGER`
- Example pattern:
  ```python
  try:
      import optional_module
      HAS_OPTIONAL = True
  except ImportError as e:
      HAS_OPTIONAL = False
      _IMPORT_ERROR = str(e)
  ```

### Testing Conventions

- **Always include assertion messages** - Every `assert` statement must have a descriptive message
- Use clear, informative messages that explain what is being tested
- Example:
  ```python
  # ❌ WRONG - No message
  assert result is not None
  assert len(items) == 5
  
  # ✅ CORRECT - Clear messages
  assert result is not None, "Agent should return result for valid input"
  assert len(items) == 5, "Should return exactly 5 items from batch processing"
  ```
- Messages help diagnose failures quickly without reading test code
- Especially important for complex conditions or edge cases

### Code Formatting

- Use **Black** for code formatting (line length: 88)
- Use **isort** for import sorting
- Follow **PEP 8** style guide
- No manual formatting needed - tools handle it

## Architecture & Key Files

### Directory Structure

```
RAG_supporters/
├── agents/              # LLM-powered agents for dataset operations
├── augmentations/       # Embedding generation utilities
├── clustering/          # Keyword and data clustering
├── dataset/            # Dataset handling and generation
│   ├── steering/       # Steering/control configurations
│   ├── templates/      # Dataset templates (e.g., BioASQ)
│   └── utils/          # Dataset utility functions
├── embeddings/         # Embedding I/O and keyword embedding
├── nn/                 # Neural network models
├── prompts_templates/  # LLM prompt templates
└── utils/              # General utilities (text processing, etc.)

docs/                   # Comprehensive documentation
├── agents/            # Agent-specific documentation
└── *.md              # Guides (clustering, dataset splitting, etc.)

tests/                 # Unit tests
```

### Critical Agent Files

**DO NOT MODIFY** these files without understanding their LangGraph structure:
- [RAG_supporters/agents/dataset_check.py](RAG_supporters/agents/dataset_check.py) - Uses StateGraph with complex workflow
- [RAG_supporters/agents/domain_assesment.py](RAG_supporters/agents/domain_assesment.py) - Multi-mode agent with EXTRACT/GUESS/ASSESS

### Key Modules

#### Agents (`RAG_supporters/agents/`)
- **QuestionAugmentationAgent**: Question rephrasing and generation
- **TextAugmentationAgent**: Text augmentation while preserving meaning
- **DatasetCheckAgent**: Source comparison using LangGraph workflow
- **DomainAnalysisAgent**: Domain extraction, guessing, and assessment (3 operation modes)
- **SourceEvaluationAgent**: Multi-dimensional source quality scoring

#### Dataset (`RAG_supporters/dataset/`)
- **dataset_splitter.py**: Train/val/test splitting with persistence
- **rag_dataset.py**: RAG dataset handling and manipulation
- **cluster_labeled_dataset.py**: Cluster-aware dataset management
- **domain_assessment_dataset_builder.py**: Domain assessment dataset generation

#### Prompts (`RAG_supporters/prompts_templates/`)
- **domain_extraction.py**: Domain extraction, guessing, and assessment prompts
- **rag_generators.py**: RAG question/answer generation prompts
- **rag_verifiers.py**: RAG quality verification prompts
- **text_augmentation.py**: Text augmentation prompt templates

### Configuration Files

- **requirements.txt**: Core dependencies (torch, datasets, sklearn)
- **pyproject.toml**: Package configuration with optional dependencies:
  - `[openai]`: OpenAI LLM provider support
  - `[nvidia]`: NVIDIA LLM provider support
  - `[dev]`: Development tools (black, isort, pytest, pylint)

## Project-Specific Gotchas

### Import Paths

- **Relative imports** are used within the `RAG_supporters/` package
- When importing agents: `from RAG_supporters.agents import DatasetCheckAgent`
- When importing utilities from within package: `from prompts_templates.domain_extraction import ...`
- Agents expect to be imported from the package root, not run as scripts

### LangChain/LLM Dependencies

- Agent modules have **lazy imports** for LangChain dependencies
- Import errors are caught and stored in `_IMPORT_ERROR` variable
- Check for imports before using agents:
  ```python
  try:
      from RAG_supporters.agents import DatasetCheckAgent
  except ImportError as e:
      print(f"Install agent dependencies: pip install -e .[openai] or pip install -e .[nvidia]")
  ```

### State Management in Agents

- `DatasetCheckAgent` uses **LangGraph StateGraph** with typed state
- State is a `TypedDict` with specific keys - don't modify state structure
- Agent graphs are built in `_build_graph()` method - handle with care

### Pydantic Models

- Agents use **Pydantic v2** models for structured outputs
- Models include custom validators (e.g., `@model_validator`, `@field_validator`)
- LLM outputs are parsed using `PydanticOutputParser` with error recovery via `OutputFixingParser`

### Operation Modes

- `DomainAnalysisAgent` has three modes: **EXTRACT**, **GUESS**, **ASSESS**
- Mode is specified via `OperationMode` enum, not strings
- Each mode uses different prompts and returns different Pydantic models

### DataFrame Processing

- Agents support batch processing via `process_dataframe()` and `process_csv()` methods
- Use `skip_labeled=True` to avoid reprocessing rows with existing labels
- Progress bars are shown via `tqdm` for long operations

### Dataset Splitting Persistence

- `DatasetSplitter` saves split indices to JSON for reproducibility
- Always use the same `random_state` for consistent splits
- Load existing splits with `DatasetSplitter.from_file()`

### Text Validation

- Use `utils.text_utils.is_empty_text()` to check for empty/None strings
- Agents validate input text before processing
- Empty inputs may be skipped or raise errors depending on context

## Developer Environment Setup

### Python Version

- **Python 3.10 or higher** is required
- Recommended: Use `pyenv` for Python version management
  ```bash
  pyenv install 3.10
  pyenv local 3.10
  ```

### Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install all dependencies
pip install -r RAG_supporters/requirements.txt
pip install -e .[openai]   # or .[nvidia] for NVIDIA support
pip install -e .[dev]
```

### Environment Variables

For OpenAI-based agents, set your API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Running in Dev Container

This workspace is configured for dev containers (Ubuntu 24.04.3 LTS):
- All system dependencies are pre-installed
- Python environment is set up automatically
- Use `git`, `docker`, `kubectl` commands directly

### IDE Configuration

#### VS Code Settings (Recommended)
```json
{
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "python.sortImports.path": "isort",
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  }
}
```

## Repository Etiquette

### Branch Naming

- Feature branches: `feature/<description>` or `copilot/<description>`
- Bug fixes: `fix/<description>`
- Documentation: `docs/<description>`

### Commit Messages

- Use clear, descriptive commit messages
- Start with verb in imperative mood: "Add feature", "Fix bug", "Update docs"
- Reference issue numbers when applicable

### Pull Requests

- Ensure all tests pass before creating PR
- Run code quality checks (black, isort, pylint)
- **Update documentation and project structure** when:
  - Adding new features or functionality
  - Modifying existing features or APIs
  - Changing agent behavior or workflows
  - Update relevant files in `docs/` directory
  - Update `AGENTS.md` if adding/modifying agents
  - Update README.md if changing project structure
- Add tests for new functionality

### Code Review

- Keep changes focused and atomic
- Document why, not just what
- Update relevant documentation in `docs/` folder

## Documentation

### Main Documentation

- **[docs/README.md](docs/README.md)** - Documentation index
- **[docs/agents/AGENTS_OVERVIEW.md](docs/agents/AGENTS_OVERVIEW.md)** - Agent workflows and comparisons

### Agent Documentation

Each agent has detailed documentation in `docs/agents/`:
- **CSV_QUESTION_AGENT.md** - QuestionAugmentationAgent
- **TEXT_AUGMENTATION.md** - TextAugmentationAgent
- **DATASET_CHECK_AGENT.md** - DatasetCheckAgent
- **DOMAIN_ANALYSIS_AGENT.md** - DomainAnalysisAgent
- **SOURCE_EVALUATION_AGENT.md** - SourceEvaluationAgent

### Other Guides

- **[docs/DATASET_SPLITTING.md](docs/DATASET_SPLITTING.md)** - Dataset splitting with persistence
- **[docs/CLUSTERING_AND_ASSIGNMENT.md](docs/CLUSTERING_AND_ASSIGNMENT.md)** - Clustering workflows

## Quick Reference

### Initialize an Agent

```python
from langchain_openai import ChatOpenAI
from RAG_supporters.agents import QuestionAugmentationAgent

llm = ChatOpenAI(model="gpt-4", temperature=0.3)
agent = QuestionAugmentationAgent(llm=llm)
```

### Process a DataFrame

```python
import pandas as pd
from RAG_supporters.agents import DatasetCheckAgent

agent = DatasetCheckAgent(llm=llm)
df = pd.read_csv("dataset.csv")
result_df = agent.process_dataframe(
    df,
    question_col="question",
    source_col="source",
    skip_labeled=True
)
```

### Domain Analysis Modes

```python
from RAG_supporters.agents import DomainAnalysisAgent, OperationMode

agent = DomainAnalysisAgent(llm=llm, operation_mode=OperationMode.EXTRACT)
result = agent.extract_domains_from_source(source_text)

agent.operation_mode = OperationMode.GUESS
result = agent.guess_question_domains(question)

agent.operation_mode = OperationMode.ASSESS
result = agent.assess_question_against_terms(question, available_terms)
```

### Dataset Splitting

```python
from RAG_supporters.dataset import DatasetSplitter

splitter = DatasetSplitter(random_state=42)
train_idx, val_idx = splitter.split(dataset_size=1000, val_ratio=0.2)
splitter.save("splits.json")

# Later, load the same split
loaded = DatasetSplitter.from_file("splits.json")
```

## Common Patterns

### Error Handling for Agent Operations

```python
import logging

LOGGER = logging.getLogger(__name__)

try:
    result = agent.process_text(text)
except Exception as e:
    LOGGER.error(f"Agent processing failed: {e}")
    result = None  # or provide default
```

### Batch Processing with Progress

```python
from tqdm import tqdm

results = []
for item in tqdm(items, desc="Processing"):
    result = agent.process(item)
    results.append(result)
```

### Validating LLM Outputs

```python
from langchain_classic.output_parsers import OutputFixingParser
from langchain_core.output_parsers import PydanticOutputParser

parser = PydanticOutputParser(pydantic_object=YourModel)
fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)

try:
    parsed = parser.parse(llm_output)
except Exception:
    parsed = fixing_parser.parse(llm_output)  # Attempt to fix
```

## Notes for AI Coding Agents

### When Adding New Agents

1. Create agent class in `RAG_supporters/agents/`
2. Add documentation in `docs/agents/`
3. Update `RAG_supporters/agents/__init__.py` exports
4. Add examples to agent documentation
5. Update `docs/agents/AGENTS_OVERVIEW.md`
6. Write unit tests in `tests/`

### When Modifying Existing Agents

1. Check if agent uses LangGraph (StateGraph) - be cautious with state changes
2. Update corresponding documentation in `docs/agents/`
3. Run existing tests to ensure no regression
4. Update examples if API changes

### When Working with Prompts

1. Prompts are in `RAG_supporters/prompts_templates/`
2. Keep prompts concise and focused
3. Use Pydantic models for structured outputs
4. Test prompt changes with real LLM calls

### When Adding Dependencies

1. Add to appropriate location:
   - Core: `requirements.txt`
   - Agent extras: `pyproject.toml` under `[project.optional-dependencies]`
   - Dev tools: `pyproject.toml` under `[project.optional-dependencies]` dev section
2. Use lazy imports for optional dependencies
3. Provide helpful error messages if imports fail

---

**Last Updated**: January 19, 2026  
**Maintained By**: Project contributors  
**Questions**: See `docs/` for detailed documentation
