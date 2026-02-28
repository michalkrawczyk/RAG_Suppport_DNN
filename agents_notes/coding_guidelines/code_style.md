# Code Style & Conventions

## Formatting

- **Black** (line length 88) for code formatting
- **isort** for import ordering: standard library → third-party → local
- Run: `black RAG_supporters/` and `ruff check RAG_supporters/`

## Naming Conventions

| Construct | Style | Example |
|-----------|-------|---------|
| Classes | `PascalCase` | `DomainAnalysisAgent` |
| Functions / Variables | `snake_case` | `process_dataframe` |
| Constants | `UPPER_SNAKE_CASE` | `MAX_RETRIES` |
| Private members | `_` prefix | `_build_graph` |

## Type Hints

- Required on **all** function parameters and return values
- Use `typing` module: `List`, `Dict`, `Optional`, `Tuple`, etc.
- Example:
  ```python
  def process(self, text: str, max_tokens: Optional[int] = None) -> Optional[str]:
  ```

## Docstrings

- **NumPy style**, triple-quoted
- Include parameter types and return types
- Example:
  ```python
  def assess(self, text: str) -> Optional[AssessmentResult]:
      """
      Assess the domain relevance of the text.

      Parameters
      ----------
      text : str
          Input text to assess.

      Returns
      -------
      Optional[AssessmentResult]
          Parsed assessment, or None on failure.
      """
  ```

## Logging

- Use module-level logger: `LOGGER = logging.getLogger(__name__)`
- INFO level for normal operations
- ERROR level for failures
- Never use `print()` for operational output
