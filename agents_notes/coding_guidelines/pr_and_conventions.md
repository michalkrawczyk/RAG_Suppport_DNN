# PR Checklist & Project Conventions

## Pull Request Checklist

Before submitting a PR, verify all items:

**Tests**
- [ ] All tests pass: `pytest tests/`
- [ ] New agent has a test file: `tests/test_<agent_name>.py`
- [ ] Test coverage maintained or improved
- [ ] Error paths are tested

**Code Quality**
- [ ] Docstrings on all public methods
- [ ] Type hints on all parameters and return values
- [ ] No hardcoded API keys or secrets
- [ ] Prompts stored in `prompts_templates/`, not inline

**Architecture**
- [ ] Architecture boundaries respected (see [architecture.md](architecture.md))
- [ ] Agent added to `RAG_supporters/agents/__init__.py` exports

**Documentation**
- [ ] Agent added to `docs/agents/AGENTS_OVERVIEW.md`
- [ ] `agents_notes/PROJECT_STRUCTURE.md` updated if files were added/deleted/significantly modified

**Conventions**
- [ ] `#@agent` flags processed (extracted, documented, removed from source)

## #@agent Flag Processing

The `#@agent` flag marks instructions embedded in code or prompts that should become permanent rules.

### Steps (required in every PR)

1. **Search** changed files for `#@agent` markers
2. **Extract** the instruction text that follows the flag
3. **Remove** the `#@agent` comment from source
4. **Document** the rule in both `AGENTS.md` and `copilot-instructions.md` under the appropriate section
5. **Format** as a precise, concise rule matching existing style

**Example:**
```python
# #@agent: Always validate required columns before processing DataFrames
```
→ Added to Architecture section: "Always validate required columns exist before processing DataFrames"

## Repository Etiquette

- **Branch naming**: `feature/`, `fix/`, `docs/`, `copilot/` prefixes
- **Commits**: imperative mood, reference issue numbers when applicable
- **PRs**: all tests pass; update `docs/`, `AGENTS.md`, `README.md`, and `agents_notes/PROJECT_STRUCTURE.md` for any file additions, deletions, or API changes
- Document **why**, not just what

## Common Mistakes Reference

| ❌ Wrong | ✅ Correct | Why |
|---------|-----------|-----|
| Hardcoded prompts in agent | Import from `prompts_templates/` | Centralised prompt management |
| `from openai import OpenAI` | `from langchain_core.language_models import BaseChatModel` | Provider abstraction |
| Raise exception on LLM error | Return `None` or add error column | Graceful degradation |
| No tests for new agent | Comprehensive test file in `tests/` | Quality assurance |
| Tests require API keys | Mock all LLM calls | Fast, free, reproducible |
| Agent imports from `dataset/` | Keep agents standalone | Architectural boundary |
| Process DataFrame row-by-row | Use batch processing methods | 10–50× performance gain |
| Return raw LLM strings | Pydantic validated models | Type safety |
| `assert result is not None` | `assert result is not None, "message"` | Diagnosable failures |
