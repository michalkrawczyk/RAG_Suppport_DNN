# GitHub Copilot Instructions for RAG_Support_DNN

## Project Overview

This repository contains RAG (Retrieval-Augmented Generation) support tools and agents for dataset creation, curation, and enhancement. The project emphasizes quality, testability, and maintainability through strict development standards.

## Critical Rules

### 1. Testing is Non-Negotiable

**Every agent and significant feature MUST include comprehensive unit tests.**

- Test file pattern: `tests/test_<agent_name>.py`
- All tests must pass before code can be merged
- Mock all LLM calls using `unittest.mock` - tests must run without API keys
- Target 100% coverage for public methods

### 2. Architecture Boundaries Must Be Maintained

**Agents are standalone tools with strict import restrictions:**

✅ Agents MAY import from:
- `prompts_templates/` - Prompt definitions
- `langchain_core` - LLM abstractions
- Standard libraries (pandas, logging, etc.)

❌ Agents MUST NOT import from:
- `dataset/` - Dataset storage and management
- `nn/` - Neural network training
- `clustering/` - Clustering algorithms
- `embeddings/` - Embedding generation

**Rationale**: Agents should be reusable tools, not tightly coupled to specific pipelines.

### 3. LangChain Abstractions Required

**Never bypass LangChain for LLM interaction:**

- ✅ Use `BaseChatModel` from `langchain_core.language_models`
- ✅ Structured outputs via Pydantic models
- ❌ Direct API calls to OpenAI/Anthropic/etc.

**Rationale**: Maintains provider flexibility and enables testing through mocking.

### 4. Graceful Error Handling

**Agents must never crash on LLM failures:**

- Single operations: Return `None` on error
- DataFrame operations: Add `{agent_name}_error` column
- Log errors with appropriate severity
- Implement retry logic for transient failures

## Required Development Workflow

### Before Writing Code

1. Review [AGENTS.md](../docs/AGENTS.md) - Technical guidelines and architecture
2. Check [AGENTS_OVERVIEW.md](../docs/agents/AGENTS_OVERVIEW.md) - Existing agent patterns
3. Identify similar agents to follow as templates

### While Writing Code

1. Extract prompts to `prompts_templates/` - Never hardcode in agents
2. Add type hints to all function parameters and returns
3. Validate inputs in `__init__` methods
4. Implement both single-item and DataFrame batch processing
5. Add logging at INFO level for operations, ERROR for failures

### After Writing Code

1. Write comprehensive tests in `tests/` directory (see test template below)
2. Run test suite: `pytest tests/ -x`
3. Add agent documentation to `AGENTS_OVERVIEW.md`
4. Update this file if introducing new patterns

## Test Structure Template

All agent tests should follow this structure:

**File: `tests/test_{agent_name}.py`**

```python
"""Tests for AgentName."""
import pytest
from unittest.mock import Mock
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
import pandas as pd

pytest.importorskip("langchain")

def test_agent_import():
    """Test that agent can be imported."""
    from RAG_supporters.agents.{agent_name} import AgentName
    assert AgentName is not None

class TestAgentInit:
    """Test initialization scenarios."""
    # Test valid initialization
    # Test invalid LLM parameter
    # Test parameter validation

class TestAgentMethods:
    """Test core functionality."""
    # Test typical inputs
    # Test edge cases (empty, None, malformed)
    # Test error handling (LLM failures, parsing errors)

class TestAgentDataFrame:
    """Test batch processing."""
    # Test DataFrame processing
    # Test checkpoint support
    # Test error column creation

@pytest.mark.skip(reason="Requires API key")
class TestAgentIntegration:
    """Optional real LLM tests."""
    # Test with real LLM for validation
```

**Key Testing Principles:**
- Mock LLM with: `Mock(spec=BaseChatModel)`
- Mock responses: `AIMessage(content="Expected output")`
- Test errors: `Mock(side_effect=Exception("Error"))`
- **Always include assert messages**: Every `assert` must have a descriptive message explaining what is tested
  - ❌ Wrong: `assert result is not None`
  - ✅ Correct: `assert result is not None, "Agent should return result for valid input"`
- See existing tests for complete examples

## Quick Command Reference

```bash
# Setup - Choose ONE LangChain provider:
# For OpenAI:
pip install -e .[openai]
# For NVIDIA:
pip install -e .[nvidia]
# For base agents only (no LLM provider):
pip install -e .

# Development dependencies:
pip install -e .[dev]

# Run Tests
pytest tests/                           # All tests
pytest tests/test_{agent}.py -v        # Specific agent with details
pytest tests/ --cov=RAG_supporters     # With coverage
pytest tests/ -x                       # Stop at first failure

# Quality Checks (if configured)
black RAG_supporters/                  # Format code
ruff check RAG_supporters/             # Lint code
mypy RAG_supporters/agents/            # Type check
```

## Common Mistakes Reference

| ❌ **WRONG** | ✅ **CORRECT** | **Why** |
|-------------|---------------|---------|
| Hardcoded prompts in agent | Import from `prompts_templates/` | Centralized prompt management |
| `from openai import OpenAI` | `from langchain_core.language_models import BaseChatModel` | Provider abstraction and testability |
| Raise exception on LLM error | Return `None` or add error column | Graceful degradation |
| No tests for new agent | Comprehensive test file in `tests/` | Quality assurance requirement |
| Tests require API keys | Mock all LLM calls | Fast, free, reproducible |
| Agent imports from `dataset/` | Keep agents standalone | Architectural boundary |
| Process DataFrame row-by-row | Use batch processing methods | 10-50x performance gain |
| Return raw LLM strings | Pydantic validated models | Type safety and validation |
| `assert result is not None` | `assert result is not None, "Descriptive message"` | Assert messages enable quick failure diagnosis |

## Pull Request Checklist

**Before submitting, verify:**

- [ ] All tests pass: `pytest tests/`
- [ ] New agent has test file: `tests/test_{agent_name}.py`
- [ ] Test coverage maintained/improved
- [ ] Docstrings on all public methods
- [ ] Type hints on all parameters/returns
- [ ] Error handling tested
- [ ] No hardcoded API keys or secrets
- [ ] Prompts in `prompts_templates/`
- [ ] Agent added to `AGENTS_OVERVIEW.md`
- [ ] Architecture boundaries respected (no forbidden imports)

## Resources

### Technical Guidelines
- [AGENTS.md](../docs/AGENTS.md) - Architecture, tech stack, patterns, and standards

### Agent Documentation
- [AGENTS_OVERVIEW.md](../docs/agents/AGENTS_OVERVIEW.md) - Agent features and usage examples
- Individual agent docs in `docs/agents/`

### External Resources
- [pytest documentation](https://docs.pytest.org/)
- [unittest.mock guide](https://docs.python.org/3/library/unittest.mock.html)
- [LangChain documentation](https://python.langchain.com/)

---

**For AI Coding Agents**: Always start by reading AGENTS.md and AGENTS_OVERVIEW.md. When in doubt, follow existing agent patterns. Tests are mandatory, not optional.
