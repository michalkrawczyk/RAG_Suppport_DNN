# Agent Development Workflow

## Before Writing Code

1. Read [AGENTS.md](../../AGENTS.md) — architecture and tech stack
2. Read [docs/agents/AGENTS_OVERVIEW.md](../../docs/agents/AGENTS_OVERVIEW.md) — existing agent patterns
3. Find a similar existing agent to use as a template
4. Use `search_module_map.py` to look up class/method signatures without opening large files:
   ```bash
   python agents_notes/search_module_map.py DatasetCheckAgent --type class
   python agents_notes/search_module_map.py process_dataframe --type method
   ```
   See [MODULE_MAP_USAGE.md](../MODULE_MAP_USAGE.md) for full reference.
   Regenerate map if missing: `python agents_notes/generate_module_map.py`

## While Writing Code

1. **Prompts** → always extract to `RAG_supporters/prompts_templates/`; never hardcode in agents
2. **Type hints** → add to all function parameters and return values
3. **Validation** → validate constructor arguments in `__init__`
4. **Both interfaces** → implement single-item (`process_item`) and batch (`process_dataframe` / `process_csv`)
5. **Logging** → INFO for normal operations, ERROR for failures

- **Never bypass LangChain** for LLM interaction — always use `BaseChatModel` abstractions.
## After Writing Code

1. Write comprehensive tests in `tests/test_<agent_name>.py` (see [testing.md](testing.md))
2. Run: `pytest tests/ -x`
3. Add agent documentation to `docs/agents/`
4. Update `docs/agents/AGENTS_OVERVIEW.md`
5. Update `agents_notes/PROJECT_STRUCTURE.md`
6. Export the new class in `RAG_supporters/agents/__init__.py`

## Adding New Agents — Checklist

- [ ] Agent class in `RAG_supporters/agents/<agent_name>.py`
- [ ] Prompts in `RAG_supporters/prompts_templates/<agent_name>.py`
- [ ] Export in `RAG_supporters/agents/__init__.py`
- [ ] Documentation in `docs/agents/<AGENT_NAME>.md`
- [ ] Entry in `docs/agents/AGENTS_OVERVIEW.md`
- [ ] Tests in `tests/test_<agent_name>.py`
- [ ] `agents_notes/PROJECT_STRUCTURE.md` updated

## Modifying Existing Agents

1. Check if the agent uses LangGraph (`StateGraph`) — be cautious with state changes.
2. Read the corresponding `docs/agents/<AGENT>.md` for workflow diagrams and state descriptions.
3. Run `pytest tests/test_<agent_name>.py` before **and** after changes to catch regressions.
4. Update `docs/agents/<AGENT>.md` if any public behaviour changes.

### Files requiring extra caution

**`RAG_supporters/agents/dataset_check.py`**
- Uses `StateGraph` with a multi-node workflow; state transitions and node functions are tightly coupled.
- Read [docs/agents/DATASET_CHECK_AGENT.md](../../docs/agents/DATASET_CHECK_AGENT.md) before touching this file.

**`RAG_supporters/agents/domain_assesment.py`**
- Implements three operation modes: **EXTRACT**, **GUESS**, **ASSESS**; routing logic between modes is non-obvious.
- Read [docs/agents/DOMAIN_ANALYSIS_AGENT.md](../../docs/agents/DOMAIN_ANALYSIS_AGENT.md) before touching this file.

## Working with Prompts

- Store all prompts in `RAG_supporters/prompts_templates/`
- Keep prompts concise and focused on a single task
- Always use Pydantic models for structured outputs
- Test prompt changes with real LLM calls in a separate integration test

## Adding Dependencies

See **[setup_and_install.md](setup_and_install.md)** for the dependency table, install commands, and the lazy-import pattern for optional dependencies.

## Dataset Splitting

- `DatasetSplitter` saves indices to JSON for reproducibility — always pass the same `random_state`
- Load existing splits: `DatasetSplitter.from_file(path)`

## Non-Negotiable Rules

- **Never bypass LangChain** for LLM interaction — always use `BaseChatModel` abstractions.
- **Every agent requires a test file** with mocked LLM calls in `tests/test_<agent_name>.py`.
- **Agents must never crash** on LLM failures — graceful error handling is mandatory.