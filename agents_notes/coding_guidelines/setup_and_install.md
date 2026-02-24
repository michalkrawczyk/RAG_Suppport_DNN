# Setup, Installation & Core Technologies

> **When to read**: When setting up the project, adding a new dependency, checking which tech stack to use, or running tests/quality tools.

## Core Technologies

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.10+ | Required runtime |
| PyTorch | latest | Neural network components |
| LangChain / LangGraph | latest | Agent orchestration |
| Pandas | latest | Dataset manipulation |
| Scikit-learn | latest | Clustering and embeddings |
| Pydantic | v2 | Data validation |

## Installation

```bash
# Base install (no LLM provider)
pip install -e .

# With OpenAI provider
pip install -e .[openai]

# With NVIDIA provider
pip install -e .[nvidia]

# Dev tools (black, isort, pytest, pylint)
pip install -e .[dev]
```

Optional dependencies are declared in `pyproject.toml` under `[project.optional-dependencies]`.  
For the dependency files per use-case see `dependencies/`.

## Running Tests

```bash
# Stop on first failure
pytest tests/ -x

# With coverage report
pytest tests/ --cov=RAG_supporters
```

## Code Quality Commands

```bash
# Format
black RAG_supporters/

# Lint
ruff check RAG_supporters/

# Type check (agents only)
mypy RAG_supporters/agents/
```

## Adding Dependencies

| Dependency type | Where to add |
|-----------------|--------------|
| Core runtime | `requirements.txt` |
| Agent extras (OpenAI, NVIDIA…) | `pyproject.toml` `[project.optional-dependencies]` |
| Dev tools | `pyproject.toml` `[project.optional-dependencies]` dev section |

- Use lazy imports + `HAS_X` / `_IMPORT_ERROR` pattern for optional dependencies.
- Provide a helpful install hint in the raised error message.
