# Error Handling

## LLM Failure Policy

Agents **must never crash** on LLM failures:

| Context | Behaviour |
|---------|-----------|
| Single-item operation | Return `None` |
| DataFrame operation | Add `{agent_name}_error` column with the error message |
| Transient network errors | Implement retry logic before giving up |

Log all failures with `LOGGER.error(...)` before returning.

## Input Validation

- Validate all inputs in `__init__` — raise `ValueError` with a clear message for bad construction arguments
- Use `utils.text_utils.is_empty_text()` to check for empty/None strings before processing
- Empty inputs should be skipped gracefully (do not pass them to the LLM)

## Lazy Import Error Propagation

```python
def some_method(self):
    if _IMPORT_ERROR:
        raise ImportError(
            f"Required dependency not installed: {_IMPORT_ERROR}. "
            "Run: pip install -e .[openai]"
        )
```

## Retry Logic Pattern

Apply retries only for transient failures (network timeouts, rate limits):

```python
for attempt in range(max_retries):
    try:
        result = self._llm.invoke(prompt)
        break
    except Exception as e:
        LOGGER.warning("Attempt %d failed: %s", attempt + 1, e)
        if attempt == max_retries - 1:
            LOGGER.error("All retries exhausted for input: %s", text[:80])
            return None
```

## DataFrame Error Column Convention

When a row fails during batch processing:

```python
df.at[idx, f"{self.agent_name}_error"] = str(exception)
df.at[idx, result_column] = None
```

Always add the error column even if only some rows fail.
