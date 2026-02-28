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
agent_ignore/            # (Never use that) Ignored files and artifacts 
dependencies/            # Modular requirements files
```

## Agents (`RAG_supporters/agents/`)

See [agents_notes/module_notes/agents_overview.md](agents_notes/module_notes/agents_overview.md) — read only when working with agents (agent list, purposes, and agent rules).

## Guidelines — Read Only When Needed

| File | Read when |
|------|-----------|
| [coding_guidelines/code_style.md](agents_notes/coding_guidelines/code_style.md) | Unsure about formatting, naming, type hints, or docstrings |
| [coding_guidelines/pr_and_conventions.md](agents_notes/coding_guidelines/pr_and_conventions.md) | Preparing a PR, naming a branch/commit, processing `#@agent` flags |
| [documentation_guidelines/documentation_guidelines.md](agents_notes/documentation_guidelines/documentation_guidelines.md) | Adding/modifying agents or modules, or PRs that change file structure |
| [module_reference.md](agents_notes/module_reference.md) | Working on `nn/`, `pytorch_datasets/`, `jasper/`, or `contrastive/` code |
| [module_notes/agents_overview.md](agents_notes/module_notes/agents_overview.md) | Looking up agent names, purposes, or agent rules |

## Testing

See [testing.md](agents_notes/coding_guidelines/testing.md). After any changes: `python agents_notes/generate_module_map.py`

## Module Map (fast signature lookup)

Before opening source files to find class or method signatures, use:

```bash
python agents_notes/search_module_map.py DatasetCheckAgent --type class
python agents_notes/search_module_map.py process_dataframe --type method
```

See [agents_notes/MODULE_MAP_USAGE.md](agents_notes/MODULE_MAP_USAGE.md). Regenerate if missing: `python agents_notes/generate_module_map.py`

# **Backward compatibility code is not added by default** — ask the user before implementing it.
# Try to be precise and concise in responses
# Avoid code snippets when planning