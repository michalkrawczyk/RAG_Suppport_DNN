# AGENTS.md - RAG Support DNN Project

Python library exploring lightweight neural network alternatives to LLM-based RAG. LLM agents are used solely to prepare and curate training datasets; the actual retrieval at inference time is done by a trained PyTorch model (JASPER), not an LLM.

## Two-Phase System

**Phase 1 — Dataset Curation (LLM agents)**
Five LangChain agents (`RAG_supporters/agents/`) process raw CSV data into curated triplets. Agents call LLMs but produce only training data; they are not used at inference time.

**Phase 2 — Neural Training & Inference (PyTorch)**
The JASPER model (`RAG_supporters/nn/`) trains on those triplets and performs retrieval without any LLM at inference time.

## Directory Structure

```
RAG_supporters/
├── agents/              # 5 LLM agents for dataset operations
├── prompts_templates/   # All LLM prompts (never inline)
├── nn/                  # JASPER models, losses, trainer, XAI
├── pytorch_datasets/    # JASPERSteeringDataset + loaders
├── jasper/              # End-to-end dataset builder (Tasks 1–9)
├── dataset/             # Domain assessment datasets (legacy)
├── clustering/          # Keyword and topic clustering
├── clustering_ops/      # Cluster operations
├── contrastive/         # Hard negative mining, SteeringBuilder
├── data_prep/           # CSV merging, dataset splitting
├── data_validation/     # Tensor utilities
├── embeddings_ops/      # Embedding generation & steering
├── embeddings/          # Embedding I/O
├── augmentations/       # Text augmentation utilities
└── utils/               # Text processing, suggestions

tests/                   # 27+ unit test files (one per agent/module)
docs/                    # Per-module documentation
examples/                # Training scripts
agents_notes/            # Project tooling, coding guidelines, module map
agent_ignore/            # Git-ignored generated artifacts (module_map.json)
dependencies/            # Modular requirements files
```

## Agents (`RAG_supporters/agents/`)

| Agent | Purpose |
|---|---|
| `QuestionAugmentationAgent` | Question generation/rephrasing with source context |
| `TextAugmentationAgent` | Text augmentation preserving semantic meaning |
| `DatasetCheckAgent` | Source comparison via LangGraph `StateGraph` |
| `DomainAnalysisAgent` | Domain extraction/guessing/assessment (3 modes) |
| `SourceEvaluationAgent` | 6-dimensional source quality scoring |

**Agent rules (non-negotiable):**
- Agents may only import from `prompts_templates/` and stdlib — never from `dataset/`, `nn/`, or `clustering/`.
- Always use `BaseChatModel` abstractions; never bypass LangChain for LLM interaction.
- Every agent needs both a single-item method and a batch `process_dataframe` method.
- Prompts live in `prompts_templates/` only, never hardcoded.
- Use Pydantic v2 models for LLM outputs with `OutputFixingParser`.
- Agents must never crash on LLM failures — graceful error handling is mandatory.

## Neural Network (`RAG_supporters/nn/`)

**Core model:** `JASPERPredictor` — JEPA-style predictor: `(question_emb + steering_emb) → predicted_source_emb`
**EMA:** `EMAEncoder` wraps the encoder with cosine tau schedule (0.996→0.999)
**Phase 2:** `SubspaceRouter` (Gumbel-Softmax), `DecomposedJASPERPredictor` (coarse+fine decomposition), `XAIInterface`

Losses: `JASPERLoss`, `ContrastiveLoss` (InfoNCE), `CentroidLoss`, `VICRegLoss`, `RoutingLoss`, `EntropyRegularization`, `ResidualPenalty`, `DisentanglementLoss`

**Training:** `JASPERTrainer` handles curriculum learning, EMA updates, and checkpointing. `TrainingMonitor` handles metrics and optional W&B integration.

## PyTorch Datasets (`RAG_supporters/pytorch_datasets/`)

`JASPERSteeringDataset` — primary dataset:
- Zero I/O training (embeddings in memory or HDF5)
- Curriculum learning: call `dataset.set_epoch(epoch)` each epoch
- 4-tier hard negative sampling: in-cluster → adjacent → high-similarity → random
- Supports HDF5 and memmap storage formats

## JASPER Dataset Builder (`RAG_supporters/jasper/`)

Pipeline Tasks 1–9: CSV merge → cluster parsing → embedding generation → steering computation → negative mining → train/val/test split → persistence → validation → finalization. Configured by `BuildConfig` (validates and serializes to JSON). `SQLiteStorageManager` is the storage backend.

## Contrastive Tools (`RAG_supporters/contrastive/`)

`NegativeMiner`: 4-tier hard negative sampling with configurable proportions.
`SteeringBuilder`: Generates steering signals (centroid, keyword-weighted, residual).

## Guidelines — Read Only When Needed

| File | Read when |
|------|-----------|
| [coding_guidelines/setup_and_install.md](agents_notes/coding_guidelines/setup_and_install.md) | Installing, adding dependencies, running tests/quality tools |
| [coding_guidelines/architecture.md](agents_notes/coding_guidelines/architecture.md) | Writing or reviewing agent imports or LangChain abstractions |
| [coding_guidelines/agent_workflow.md](agents_notes/coding_guidelines/agent_workflow.md) | Adding or modifying an agent (includes cautions for complex agents) |
| [coding_guidelines/error_handling.md](agents_notes/coding_guidelines/error_handling.md) | Adding error paths or LLM failure handling |
| [coding_guidelines/testing.md](agents_notes/coding_guidelines/testing.md) | Writing agent tests or unsure about mocking patterns |
| [coding_guidelines/code_style.md](agents_notes/coding_guidelines/code_style.md) | Unsure about formatting, naming, type hints, or docstrings |
| [coding_guidelines/pr_and_conventions.md](agents_notes/coding_guidelines/pr_and_conventions.md) | Preparing a PR, naming a branch/commit, processing `#@agent` flags |
| [documentation_guidelines/documentation_guidelines.md](agents_notes/documentation_guidelines/documentation_guidelines.md) | Adding/modifying agents or modules, or PRs that change file structure |

## Testing Conventions

- All LLM calls must be mocked; no real API calls in unit tests.
- Target 100% coverage for agent code.
- One test file per agent: `tests/test_<agent_name>.py`.
- After any changes, regenerate the module map: `python agents_notes/generate_module_map.py`

## Module Map (fast signature lookup)

Before opening source files to find class or method signatures, use:

```bash
python agents_notes/search_module_map.py DatasetCheckAgent --type class
python agents_notes/search_module_map.py process_dataframe --type method
```

See [agents_notes/MODULE_MAP_USAGE.md](agents_notes/MODULE_MAP_USAGE.md). Regenerate if missing: `python agents_notes/generate_module_map.py`

---

**Last Updated**: February 24, 2026
