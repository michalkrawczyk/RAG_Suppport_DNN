# AGENTS.md - RAG Support DNN Project

## Project Context

This is a Python library for experiments on creating non-LLM solutions (specifically lightweight neural networks) for improving RAG (Retrieval-Augmented Generation) or creating new variants of it. The LLM-powered agents are used to support learning and prepare datasets for training these alternative models. It includes specialized agents for domain analysis, text augmentation, source evaluation, and dataset quality control.

## Core Technologies

- **Python 3.10+** required
- **PyTorch** for neural network components
- **LangChain/LangGraph** for agent orchestration
- **Pandas** for dataset manipulation
- **Scikit-learn** for clustering and embeddings
- **Pydantic** for data validation

## Code Style & Conventions

- **Formatting**: Black (line length 88) + isort; standard library → third-party → local import order
- **Naming**: `PascalCase` classes, `snake_case` functions/variables, `UPPER_SNAKE_CASE` constants, `_` prefix for private members
- **Type hints**: required on all function signatures; use `typing` module (`List`, `Dict`, `Optional`, etc.)
- **Docstrings**: NumPy style, triple-quoted, include parameter types
- **Error handling**: lazy-import optional dependencies with `HAS_X`/`_IMPORT_ERROR` pattern; log with module-level `LOGGER`; return `None` rather than raising on LLM failures
- **Tests**: every `assert` must have a descriptive message (e.g., `assert x is not None, "reason"`)

## Architecture & Key Files

### Directory Structure

```
RAG_supporters/
├── agents/              # LLM-powered agents for dataset operations
├── augmentations/       # Embedding generation utilities
├── clustering/          # Keyword and data clustering
├── dataset/            # Dataset handling and generation
│   ├── steering/       # Steering/control configurations
│   ├── templates/      # Dataset templates (e.g., BioASQ)
│   └── utils/          # Dataset utility functions
├── embeddings/         # Embedding I/O and keyword embedding
├── nn/                 # Neural network models
├── prompts_templates/  # LLM prompt templates
└── utils/              # General utilities (text processing, etc.)

docs/                   # Comprehensive documentation
├── agents/            # Agent-specific documentation
└── *.md              # Guides (clustering, dataset splitting, etc.)

tests/                 # Unit tests
```

### Critical Agent Files

**DO NOT MODIFY** these files without understanding their LangGraph structure:
- [RAG_supporters/agents/dataset_check.py](RAG_supporters/agents/dataset_check.py) - Uses StateGraph with complex workflow
- [RAG_supporters/agents/domain_assesment.py](RAG_supporters/agents/domain_assesment.py) - Multi-mode agent with EXTRACT/GUESS/ASSESS

### Key Modules

#### Agents (`RAG_supporters/agents/`)
- **QuestionAugmentationAgent**: Question rephrasing and generation
- **TextAugmentationAgent**: Text augmentation while preserving meaning
- **DatasetCheckAgent**: Source comparison using LangGraph workflow
- **DomainAnalysisAgent**: Domain extraction, guessing, and assessment (3 operation modes)
- **SourceEvaluationAgent**: Multi-dimensional source quality scoring

#### Dataset (`RAG_supporters/dataset/`)
- **dataset_splitter.py**: Train/val/test splitting with persistence
- **rag_dataset.py**: RAG dataset handling and manipulation
- **cluster_labeled_dataset.py**: Cluster-aware dataset management
- **domain_assessment_dataset_builder.py**: Domain assessment dataset generation

#### Prompts (`RAG_supporters/prompts_templates/`)
- **domain_extraction.py**: Domain extraction, guessing, and assessment prompts
- **rag_generators.py**: RAG question/answer generation prompts
- **rag_verifiers.py**: RAG quality verification prompts
- **text_augmentation.py**: Text augmentation prompt templates

### Configuration Files

- **requirements.txt**: Core dependencies (torch, datasets, sklearn)
- **pyproject.toml**: Package configuration with optional dependencies:
  - `[openai]`: OpenAI LLM provider support
  - `[nvidia]`: NVIDIA LLM provider support
  - `[dev]`: Development tools (black, isort, pytest, pylint)

## Project-Specific Gotchas

### Import Paths

- **Relative imports** are used within the `RAG_supporters/` package
- When importing agents: `from RAG_supporters.agents import DatasetCheckAgent`
- When importing utilities from within package: `from prompts_templates.domain_extraction import ...`
- Agents expect to be imported from the package root, not run as scripts

### LangChain/LLM Dependencies

- Agent modules have **lazy imports** for LangChain dependencies
- Import errors are caught and stored in `_IMPORT_ERROR` variable
- Wrap agent imports in `try/except ImportError` and remind the user to run `pip install -e .[openai]` or `pip install -e .[nvidia]`

### State Management in Agents

- `DatasetCheckAgent` uses **LangGraph StateGraph** with typed state
- State is a `TypedDict` with specific keys - don't modify state structure
- Agent graphs are built in `_build_graph()` method - handle with care

### Pydantic Models

- Agents use **Pydantic v2** models for structured outputs
- Models include custom validators (e.g., `@model_validator`, `@field_validator`)
- LLM outputs are parsed using `PydanticOutputParser` with error recovery via `OutputFixingParser`

### Operation Modes

- `DomainAnalysisAgent` has three modes: **EXTRACT**, **GUESS**, **ASSESS**
- Mode is specified via `OperationMode` enum, not strings
- Each mode uses different prompts and returns different Pydantic models

### DataFrame Processing

- Agents support batch processing via `process_dataframe()` and `process_csv()` methods
- Use `skip_labeled=True` to avoid reprocessing rows with existing labels
- Progress bars are shown via `tqdm` for long operations

### Dataset Splitting Persistence

- `DatasetSplitter` saves split indices to JSON for reproducibility
- Always use the same `random_state` for consistent splits
- Load existing splits with `DatasetSplitter.from_file()`

### Text Validation

- Use `utils.text_utils.is_empty_text()` to check for empty/None strings
- Agents validate input text before processing
- Empty inputs may be skipped or raise errors depending on context



## Repository Etiquette

- Branch naming: `feature/`, `fix/`, `docs/`, `copilot/` prefixes
- Commits: imperative mood, reference issue numbers when applicable
- PRs: all tests pass; update `docs/`, `AGENTS.md`, `README.md`, and `agents_notes/PROJECT_STRUCTURE.md` for any file additions/deletions/API changes
- Document why, not just what

## Documentation

### Main Documentation

- **[docs/README.md](docs/README.md)** - Documentation index
- **[docs/agents/AGENTS_OVERVIEW.md](docs/agents/AGENTS_OVERVIEW.md)** - Agent workflows and comparisons

### Agent Documentation

Each agent has detailed documentation in `docs/agents/`:
- **CSV_QUESTION_AGENT.md** - QuestionAugmentationAgent
- **TEXT_AUGMENTATION.md** - TextAugmentationAgent
- **DATASET_CHECK_AGENT.md** - DatasetCheckAgent
- **DOMAIN_ANALYSIS_AGENT.md** - DomainAnalysisAgent
- **SOURCE_EVALUATION_AGENT.md** - SourceEvaluationAgent

### Other Guides

- **[docs/DATASET_SPLITTING.md](docs/DATASET_SPLITTING.md)** - Dataset splitting with persistence
- **[docs/CLUSTERING_AND_ASSIGNMENT.md](docs/CLUSTERING_AND_ASSIGNMENT.md)** - Clustering workflows

### Project Structure Documentation

- **[agents_notes/PROJECT_STRUCTURE.md](agents_notes/PROJECT_STRUCTURE.md)** - File-by-file listing with purpose descriptions; **must be updated in every PR that adds, deletes, or significantly modifies files**

## #@agent Flag Convention

The `#@agent` flag marks instructions in code/prompts that should be extracted to documentation:

1. **Search** for `#@agent` markers in changed files
2. **Extract** the instruction text following the flag
3. **Remove** the `#@agent` comment from source
4. **Document** in AGENTS.md and copilot-instructions.md
5. **Format** as precise, concise rules matching existing style

**Example:**
```python
# #@agent: All DataFrame operations must preserve original index
```
→ Extract to AGENTS.md as: "Preserve original DataFrame index in all operations"





## Notes for AI Coding Agents

### Module Map Search Tool

Before reading source files to find class signatures or method parameters, use the
module map search tool — it is faster and avoids loading large files:

```bash
# Look up a class
python agents_notes/search_module_map.py DatasetCheckAgent --type class

# Look up a method signature
python agents_notes/search_module_map.py process_dataframe --type method

# Find all call sites of a function
python agents_notes/search_module_map.py scan_directories --type usage
```

See **[agents_notes/MODULE_MAP_USAGE.md](agents_notes/MODULE_MAP_USAGE.md)** for full reference.

If `agent_ignore/module_map.json` is missing, regenerate it first:
```bash
python agents_notes/generate_module_map.py
```

### When Adding New Agents

1. Create agent class in `RAG_supporters/agents/`
2. Add documentation in `docs/agents/`
3. Update `RAG_supporters/agents/__init__.py` exports
4. Add examples to agent documentation
5. Update `docs/agents/AGENTS_OVERVIEW.md`
6. Write unit tests in `tests/`

### When Modifying Existing Agents

1. Check if agent uses LangGraph (StateGraph) - be cautious with state changes
2. Update corresponding documentation in `docs/agents/`
3. Run existing tests to ensure no regression
4. Update examples if API changes

### When Working with Prompts

1. Prompts are in `RAG_supporters/prompts_templates/`
2. Keep prompts concise and focused
3. Use Pydantic models for structured outputs
4. Test prompt changes with real LLM calls

### When Adding Dependencies

1. Add to appropriate location:
   - Core: `requirements.txt`
   - Agent extras: `pyproject.toml` under `[project.optional-dependencies]`
   - Dev tools: `pyproject.toml` under `[project.optional-dependencies]` dev section
2. Use lazy imports for optional dependencies
3. Provide helpful error messages if imports fail

---

**Last Updated**: January 19, 2026  
**Maintained By**: Project contributors  
**Questions**: See `docs/` for detailed documentation
