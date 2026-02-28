# Testing Standards

## Requirements

- **Every agent and significant feature must have a test file** in `tests/test_<agent_name>.py`
- All tests must pass before merging — run `pytest tests/ -x`
- Mock **all** LLM calls; tests must run without API keys
- Target coverage: **≥90%** Dataset classes, **≥80%** model classes, **≥70%** training utilities; 100% for all public agent methods

## Test File Structure

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
    from RAG_supporters.agents.agent_name import AgentName
    assert AgentName is not None, "AgentName class must be importable"


class TestAgentInit:
    """Test initialization scenarios."""
    # Test valid initialization
    # Test invalid LLM parameter (non-BaseChatModel)
    # Test parameter validation (e.g. empty model_name)


class TestAgentMethods:
    """Test core functionality."""
    # Test typical inputs
    # Test edge cases: empty string, None, malformed input
    # Test error handling: LLM raises Exception, parsing fails


class TestAgentDataFrame:
    """Test batch processing."""
    # Test process_dataframe() returns correct columns
    # Test checkpoint / skip_labeled behaviour
    # Test that error column is added on failure


@pytest.mark.skip(reason="Requires API key")
class TestAgentIntegration:
    """Optional real-LLM integration tests."""
    pass
```

## Mocking LLM

```python
mock_llm = Mock(spec=BaseChatModel)
mock_llm.invoke.return_value = AIMessage(content='{"field": "value"}')

# Simulate failure
mock_llm.invoke.side_effect = Exception("LLM unavailable")
```

## Assert Messages — Mandatory

Every `assert` must include a descriptive failure message:

```python
# ❌ Wrong
assert result is not None

# ✅ Correct
assert result is not None, "Agent should return a result for valid non-empty input"
assert "domain" in result, "Result must include the 'domain' key"
assert len(df) == original_len, "process_dataframe must not drop rows"
```

**Rationale**: messages make failures immediately diagnosable without reading source.

## Running Tests

```bash
pytest tests/                          # All tests
pytest tests/test_<agent>.py -v       # Specific agent, verbose
pytest tests/ --cov=RAG_supporters    # With coverage report
pytest tests/ -x                      # Stop on first failure
```

## Quality Checks

```bash
black RAG_supporters/       # Format
ruff check RAG_supporters/  # Lint
mypy RAG_supporters/agents/ # Type check
```
