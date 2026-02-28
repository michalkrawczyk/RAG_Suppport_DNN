# Architecture & Boundaries

## Agent Import Restrictions

Agents are **standalone tools**. Forbidden and allowed imports:

| ✅ MAY import from | ❌ MUST NOT import from |
|--------------------|-------------------------|
| `prompts_templates/` | `dataset/` |
| `langchain_core` | `nn/` |
| Standard libraries (pandas, logging, etc.) | `clustering/` |
| | `embeddings/` |

**Rationale**: Agents must be reusable and decoupled from specific pipelines.

## LangChain Abstractions

Never bypass LangChain for LLM interaction:

- ✅ Use `BaseChatModel` from `langchain_core.language_models`
- ✅ Structured outputs via Pydantic models with `PydanticOutputParser`
- ✅ Use `OutputFixingParser` for error recovery on malformed LLM output
- ❌ Direct API calls to OpenAI / Anthropic / NVIDIA / etc.

**Rationale**: Maintains provider flexibility and enables test mocking.

## Lazy Imports for Optional Dependencies

Use the `HAS_X` / `_IMPORT_ERROR` pattern for optional dependencies:

```python
try:
    from langchain_core.language_models import BaseChatModel
    HAS_LANGCHAIN = True
    _IMPORT_ERROR = None
except ImportError as e:
    HAS_LANGCHAIN = False
    _IMPORT_ERROR = str(e)
```

Raise a helpful error at call-time if the dependency is missing.

## Pydantic Models

- Use **Pydantic v2** models for all structured LLM outputs
- Add custom validators with `@model_validator` and `@field_validator` where needed
- Return typed Pydantic instances, never raw LLM strings

## LangGraph Agents (DatasetCheckAgent)

- `DatasetCheckAgent` uses `StateGraph` with a `TypedDict` state — **do not modify state structure**
- Graphs are built in `_build_graph()` — treat this method with care
- Prefer understanding the full graph topology before any changes

## Operation Modes (DomainAnalysisAgent)

- Modes: **EXTRACT**, **GUESS**, **ASSESS** — use `OperationMode` enum, not raw strings
- Each mode has its own prompt template and Pydantic output model

## DataFrame Batch Processing

- Implement both `process_item()` (single) and `process_dataframe()` / `process_csv()` (batch)
- Use `skip_labeled=True` to avoid reprocessing rows that already have labels
- Show progress with `tqdm` for long-running operations
- Batch processing is 10–50× faster than row-by-row loops
- Preserve the original DataFrame index in all operations

## Import Paths within Package

- Use **relative imports** inside `RAG_supporters/`
- From outside: `from RAG_supporters.agents import DatasetCheckAgent`
- From inside agents: `from prompts_templates.domain_extraction import ...`
- Agents must be imported from the package root, not run as scripts

## Critical Files — Handle with Care

| File | Risk |
|------|------|
| `RAG_supporters/agents/dataset_check.py` | LangGraph StateGraph with complex workflow |
| `RAG_supporters/agents/domain_assesment.py` | Multi-mode agent with 3 operation modes |
