# RAG Supporters Agents - Technical Guidelines

## Project Mission

RAG Supporters Agents are specialized LLM-powered components for RAG dataset creation, curation, and enhancement. Primary concerns:
- **Quality**: Ensure high-quality QA pairs through multi-dimensional evaluation
- **Consistency**: Maintain uniform patterns across all agents for maintainability
- **Testability**: All agents must be testable without external API dependencies
- **Performance**: Support batch processing for efficient large-scale operations

## Explicit Tech Stack

### Required Dependencies
```
langchain-core >= 0.3.0      # Core abstractions for LLM interaction
langgraph >= 0.3.0           # State machine workflows for complex agents
pydantic >= 2.0              # Structured output validation
pandas >= 1.5                # DataFrame batch processing
```

### Rationale
- **LangChain**: Industry standard for LLM integration with provider-agnostic abstractions
- **LangGraph**: Enables retry logic, checkpointing, and workflow visualization for complex multi-step agents
- **Pydantic**: Guarantees structured LLM outputs with automatic validation and type safety
- **Pandas**: Efficient batch processing and CSV I/O for dataset operations

### Provider Support
- **Primary**: OpenAI (GPT-3.5-turbo, GPT-4) with batch processing
- **Compatible**: Any LangChain-supported LLM provider (Anthropic, Cohere, local models)
- **Not Supported**: Direct API calls (must use LangChain abstractions)

## Architecture Boundaries

### Agent Layer (`RAG_supporters/agents/`)
**Responsibilities:**
- LLM interaction through LangChain abstractions
- Single-item processing methods
- DataFrame batch processing methods
- Pydantic model definitions for structured outputs

**Restrictions:**
- ❌ No direct dataset storage logic (use `dataset/` module)
- ❌ No embedding generation (use `augmentations/` or `embeddings/`)
- ❌ No neural network training (use `nn/` module)
- ❌ No clustering logic (use `clustering/` module)

### Prompt Templates (`prompts_templates/`)
**Responsibilities:**
- System prompts and few-shot examples
- Prompt engineering configurations

**Restrictions:**
- ❌ No agent instantiation
- ❌ No LLM calls

### Cross-Layer Rules
- Agents MAY import from `prompts_templates/`
- Agents MUST NOT import from `dataset/`, `nn/`, or `clustering/`
- Other modules MAY import and use agents as tools

## Security Guidelines

### API Key Management
- ✅ Use environment variables for LLM API keys
- ✅ Never log API keys or tokens
- ❌ Never hardcode credentials in agent code

### Input Validation
- ✅ Validate all user inputs before LLM calls (length, type, format)
- ✅ Sanitize DataFrame columns to prevent injection in prompts
- ✅ Use Pydantic models for structured output validation
- ❌ Never directly interpolate unsanitized user input into prompts

### LLM Output Handling
- ✅ Always validate LLM responses with Pydantic
- ✅ Implement fallback values for parsing failures
- ✅ Set max token limits to prevent excessive costs
- ❌ Never execute LLM-generated code without sandboxing

## Testing Requirements

### Coverage Expectations
- **Agents**: 100% of public methods
- **Critical Paths**: Error handling, DataFrame processing, batch operations
- **Edge Cases**: Empty inputs, malformed data, LLM failures

### Required Test Scenarios
1. **Initialization**: Valid/invalid LLM parameters
2. **Single Operations**: Typical inputs, edge cases, errors
3. **Batch Processing**: DataFrame operations, checkpointing
4. **Error Handling**: LLM failures, parsing errors, network issues
5. **Integration** (optional, skipped): Real LLM calls for validation

### Mocking Strategy
```python
from unittest.mock import Mock
from langchain_core.messages import AIMessage

# Standard mock pattern
mock_llm = Mock(spec=BaseChatModel)
mock_llm.invoke = Mock(return_value=AIMessage(content="Expected output"))
```

## Common Mistakes

| ❌ Wrong | ✅ Correct | Reason |
|---------|-----------|--------|
| `agent.process()` without error handling | Try-except with fallback return | Agents should never crash on LLM failures |
| Hardcoded prompt strings in agent code | Import from `prompts_templates/` | Centralized prompt management |
| Direct API calls to OpenAI | Use `BaseChatModel` abstraction | Provider flexibility and testability |
| Returning raw LLM strings | Pydantic validated models | Type safety and validation |
| Processing DataFrames row-by-row | Use batch processing methods | 10-50x performance improvement |
| Tests requiring API keys | Mock all LLM calls | Fast, free, reproducible tests |
| Agent importing from `dataset/` | Agent as standalone tool | Maintain layer separation |
| Single massive agent | Multiple focused agents | Single responsibility principle |

## Quick Commands Reference

### Development
```bash
# Install dependencies
pip install -r RAG_supporters/requirements_agents.txt
pip install -r RAG_supporters/requirements-dev.txt

# Run agent in Python
python -c "from RAG_supporters.agents.{agent_name} import Agent; agent = Agent(llm); agent.process(...)"
```

### Testing
```bash
# All tests
pytest tests/

# Specific agent
pytest tests/test_{agent_name}.py -v

# Coverage report
pytest tests/ --cov=RAG_supporters.agents --cov-report=html

# Stop at first failure
pytest tests/ -x

# Integration tests (requires API keys)
pytest tests/ --skip-integration=false
```

### Quality Checks
```bash
# Type checking (if using mypy)
mypy RAG_supporters/agents/

# Linting (if using ruff/flake8)
ruff check RAG_supporters/agents/
flake8 RAG_supporters/agents/

# Format (if using black)
black RAG_supporters/agents/
```

## Integration Requirements

### LLM Provider Compatibility
- Must accept `BaseChatModel` from `langchain_core.language_models`
- Must support `.invoke()` method for single calls
- Should support `.batch()` method for batch processing (OpenAI only)

### Input/Output Contracts
- **Input**: Accept Python primitives (str, dict) or pandas DataFrames
- **Output**: Return Python primitives, Pydantic models, or DataFrames
- **Errors**: Return None or error objects, never raise uncaught exceptions

### DataFrame Schema
When processing DataFrames:
- Preserve original columns
- Add new columns for agent outputs
- Include error columns (e.g., `{agent_name}_error`)
- Support checkpointing with `save_path` parameter

## Code Quality Tools

### Required Tools
- **pytest**: Testing framework (`pytest tests/`)
- **unittest.mock**: LLM mocking in tests
- **pytest-cov**: Coverage reporting

### Recommended Tools
- **black**: Code formatting (`black RAG_supporters/`)
- **ruff**: Fast linting (`ruff check RAG_supporters/`)
- **mypy**: Static type checking (`mypy RAG_supporters/`)

### Pre-commit Hook
Consider adding:
```bash
pytest tests/ -x && black RAG_supporters/ && ruff check RAG_supporters/
```

## Stack Decisions

### ✅ Use Always
- `BaseChatModel` from `langchain_core` for LLM abstraction
- Pydantic models for structured outputs
- LangGraph for multi-step agents with retry logic
- `unittest.mock` for LLM mocking in tests
- Pandas DataFrame for batch processing

### ❌ Never Use
- Direct OpenAI/Anthropic API clients (breaks abstraction)
- String-based LLM responses without validation
- Global state or singleton agents
- Synchronous loops for batch processing (use batch methods)
- Tests that require real API keys

### ⚠️ Use With Caution
- High temperature (>0.7) for evaluation tasks - causes inconsistency
- Very long prompts (>8k tokens) - increases cost and latency
- Nested agent calls - can compound errors

## Error Handling Standards

### Exception Patterns
```python
# Agent method signature
def process(self, input: str) -> Optional[Result]:
    try:
        # LLM call
        response = self._llm.invoke(...)
        # Validation
        return self._parse_response(response)
    except Exception as e:
        LOGGER.error(f"Agent failed: {e}")
        return None  # Graceful degradation
```

### Error Response Format
- Single operations: Return `None` on failure
- DataFrame operations: Add error column with error message
- Structured outputs: Include `error: Optional[str]` field in Pydantic model

### Severity Levels
- **CRITICAL**: Agent initialization failures (invalid LLM)
- **ERROR**: LLM call failures, parsing errors
- **WARNING**: Validation warnings, fallback usage
- **INFO**: Normal operation logs

## Logging Guidelines

### Logging Levels
```python
import logging
LOGGER = logging.getLogger(__name__)

# CRITICAL: Agent cannot initialize
LOGGER.critical("Failed to initialize agent: invalid LLM type")

# ERROR: Operation failed
LOGGER.error(f"LLM call failed: {error}")

# WARNING: Unexpected but handled
LOGGER.warning("Using fallback parsing due to invalid JSON")

# INFO: Normal operation tracking
LOGGER.info(f"Processing batch of {len(df)} items")

# DEBUG: Detailed debugging (not in production)
LOGGER.debug(f"LLM response: {response[:100]}...")
```

### What to Log
- ✅ Agent initialization (INFO)
- ✅ Batch processing progress (INFO)
- ✅ LLM call failures (ERROR)
- ✅ Parsing/validation failures (ERROR)
- ✅ Fallback usage (WARNING)

### What NOT to Log
- ❌ API keys or credentials
- ❌ Full user inputs (may contain PII)
- ❌ Complete LLM responses (verbose, potential PII)
- ❌ Every single operation in batch (use progress bars)

## Code Review Checklist

### Before Submitting PR
- [ ] All tests pass (`pytest tests/`)
- [ ] Test coverage maintained or improved
- [ ] New agent has corresponding test file in `tests/`
- [ ] All public methods have docstrings
- [ ] Type hints for all function parameters and returns
- [ ] Error handling tested for all public methods
- [ ] DataFrame processing includes checkpoint support
- [ ] No hardcoded API keys or credentials
- [ ] Prompts extracted to `prompts_templates/`
- [ ] Agent added to `AGENTS_OVERVIEW.md`
- [ ] Integration tests marked with `@pytest.mark.skip`

### Code Quality
- [ ] Follows PEP 8 style guidelines
- [ ] Functions follow single responsibility principle
- [ ] No cross-layer imports (agents → dataset/nn/clustering)
- [ ] LLM abstraction used (BaseChatModel, not direct API)
- [ ] Pydantic models for structured outputs

### Documentation
- [ ] Agent purpose clearly documented
- [ ] Usage examples provided
- [ ] Parameter descriptions complete
- [ ] Return value format specified
- [ ] Error cases documented

### Performance
- [ ] Batch processing implemented for DataFrame operations
- [ ] Checkpointing support for large datasets
- [ ] No synchronous loops for batch operations
- [ ] Appropriate max_retries set (3-5)

---

**Note**: This document focuses on technical implementation guidelines. For agent features, usage examples, and API details, see [AGENTS_OVERVIEW.md](AGENTS_OVERVIEW.md).

**Last Updated**: January 2026  
**Version**: 1.0
