# Documentation Guidelines

> **When to read**: When adding a new agent, modifying an existing agent's public API, adding a new module, or performing a PR that adds/removes/renames files.

## Where Documentation Lives

| Location | Contents | When to Update |
|----------|----------|----------------|
| `docs/agents/AGENTS_OVERVIEW.md` | High-level comparison table and workflow summary for all agents | Every time an agent is added or its public interface changes |
| `docs/agents/<AGENT_NAME>.md` | Detailed workflow, state diagram, parameters, examples for one agent | When modifying that agent's behaviour or API |
| `docs/DATASET_SPLITTING.md` | Dataset splitting with persistence, `DatasetSplitter` usage | When changing `dataset_splitter.py` |
| `docs/CLUSTERING_AND_ASSIGNMENT.md` | Clustering workflows | When changing clustering modules |
| `docs/README.md` | Documentation index — entry point for navigating `docs/` | When a new `docs/` file is added or removed |
| `agents_notes/PROJECT_STRUCTURE.md` | File-by-file listing with purpose descriptions for the whole repo | **Every PR** that adds, deletes, or significantly modifies files |

## Agent Documentation Files (`docs/agents/`)

Each agent must have a corresponding documentation file:

| File | Agent |
|------|-------|
| `CSV_QUESTION_AGENT.md` | `QuestionAugmentationAgent` |
| `TEXT_AUGMENTATION.md` | `TextAugmentationAgent` |
| `DATASET_CHECK_AGENT.md` | `DatasetCheckAgent` |
| `DOMAIN_ANALYSIS_AGENT.md` | `DomainAnalysisAgent` |
| `SOURCE_EVALUATION_AGENT.md` | `SourceEvaluationAgent` |

### Minimum required sections in an agent doc

1. **Purpose** — one paragraph describing what the agent does
2. **Inputs / Outputs** — parameter table
3. **Workflow** — step-by-step or state diagram
4. **Usage Example** — minimal working code snippet
5. **Notes / Caveats** — LangGraph structure warnings, mode routing, etc.

## agents_notes/ Tooling Documentation

| File | Contents | When to Read |
|------|----------|--------------|
| `agents_notes/MODULE_MAP_USAGE.md` | How to generate and search the module map index | Before browsing source for class/method signatures |
| `agents_notes/PROJECT_STRUCTURE.md` | File-by-file listing with purpose descriptions | When navigating the repo or performing a PR |

## agent_ignore/ Directory

Holds **generated artifacts intentionally hidden from LLM agents** (excluded via `.vscode/settings.json` and `.gitignore`). Contents are git-ignored; only `.gitkeep` is tracked. Store files here that should not appear in agent context windows (e.g. `module_map.json`, large caches).
