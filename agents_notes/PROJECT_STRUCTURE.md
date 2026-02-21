# RAG_Support_DNN Project Structure

> **Maintenance Note**: This file should be updated with every PR that adds, deletes, or significantly modifies files.

## Root Directory

### Configuration Files
- **pyproject.toml** - Python package configuration with dependencies, optional extras (openai, nvidia, dev), and project metadata
- **.gitignore** - Git ignore patterns for Python artifacts, IDE files, and build outputs
- **README.md** - Project overview and quick start guide

### Documentation Files
- **AGENTS.md** - Comprehensive technical documentation for developers: architecture, commands, conventions, patterns, gotchas
- **.github/copilot-instructions.md** - GitHub Copilot coding standards and project-specific rules

### Directories
- **RAG_supporters/** - Main Python package with all source code
- **tests/** - Unit tests for agents and core functionality
- **docs/** - Comprehensive documentation organized by topic
- **dependencies/** - External dependencies or requirements files
- **agents_notes/** - Project structure documentation, tooling scripts, and module-map usage guide
- **agent_ignore/** - Generated artifacts hidden from Copilot (module_map.json); contents git-ignored

---

## RAG_supporters/ (Main Package)

### Package Root
- **__init__.py** - Package initialization, exports main classes and utilities

### agents/ - LLM-Powered Agents
Agent modules for dataset operations and analysis. All agents use LangChain abstractions and support batch processing.

- **__init__.py** - Exports agent classes: DatasetCheckAgent, DomainAnalysisAgent, etc.
- **dataset_check.py** - DatasetCheckAgent: Uses LangGraph StateGraph to compare and select better source for questions
- **domain_assesment.py** - DomainAnalysisAgent: Three operation modes (EXTRACT/GUESS/ASSESS) for domain analysis
- **question_augmentation_agent.py** - QuestionAugmentationAgent: Question rephrasing and generation with source context
- **source_assesment.py** - SourceEvaluationAgent: Multi-dimensional source quality scoring (relevance, completeness, etc.)
- **text_augmentation.py** - TextAugmentationAgent: Text augmentation while preserving semantic meaning
- **README.md** - Agent module overview and usage patterns

### augmentations/ - Embedding Generation
- **__init__.py** - Module initialization
- **embedding.py** - Embedding generation utilities for text augmentation

### clustering/ - Data Clustering
Keyword-based and topic-based clustering for dataset organization.

- **__init__.py** - Exports clustering classes
- **clustering_data.py** - ClusteringData: Data structures for cluster management and persistence
- **keyword_clustering.py** - KeywordClustering: Clusters items by keyword similarity using embeddings
- **topic_distance_calculator.py** - TopicDistanceCalculator: Calculates semantic distances between topics

### clustering_ops/ - Clustering Operations
Keyword-based clustering and cluster assignment operations. Reusable for any keyword/topic clustering project.

- **__init__.py** - Exports ClusterParser, SourceClusterLinker, and utility functions
- **parse_clusters.py** - ClusterParser: Parses KeywordClusterer JSON format with keyword matching (exact + cosine fallback)
- **link_sources.py** - SourceClusterLinker: Links question-source pairs to clusters via keyword intersection with majority voting

### contrastive/ - Contrastive Learning Tools
Hard negative mining and steering signals for contrastive learning. Highly reusable for metric learning, siamese networks, triplet loss training.

- **__init__.py** - Exports NegativeMiner, SteeringBuilder, and utility functions
- **mine_negatives.py** - NegativeMiner: 4-tier hard negative sampling (in-cluster, adjacent, high-similarity, random) for contrastive learning with configurable tier proportions and validation
- **build_steering.py** - SteeringBuilder: Generates steering signals (centroid, keyword-weighted, residual) and centroid distances for curriculum learning

### data_prep/ - Data Preprocessing
Generic data preprocessing utilities for CSV merging, deduplication, and dataset splitting. Works with any tabular data format.

- **__init__.py** - Exports CSVMerger and DatasetSplitter with utility functions
- **merge_csv.py** - CSVMerger: Merges multiple CSV files with column normalization, deduplication, and ID assignment
- **dataset_splitter.py** - Simple train/val/test splitting with persistence to JSON (for specific test compatibility)

### data_validation/ - Data Validation & Tensor Utilities
Highly reusable PyTorch validation utilities for any project. Zero project-specific dependencies.

- **__init__.py** - Exports validation functions, tensor I/O utilities, and label calculators
- **validation_utils.py** - Tensor shape/type/bounds validation, dimension consistency checks, NaN/Inf detection - eliminates code duplication
- **tensor_utils.py** - Tensor loading and storage utilities with shape validation and error handling - standardizes torch.load operations
- **label_calculator.py** - Label normalization utilities (softmax, L1) for training

### dataset/ - Domain Assessment Datasets (Legacy)
Domain assessment dataset builders and steering configuration. Most functionality moved to specialized modules.

- **__init__.py** - Exports DomainAssessmentDatasetBuilder and DomainAssessmentParser (SteeringConfig and SteeringMode are now in embeddings_ops)
- **dataset_builder_README.md** - README and specifications for JASPER dataset builder pipeline (Tasks 0-9)
- **domain_assessment_dataset_builder.py** - DomainAssessmentDatasetBuilder: Builds datasets for domain assessment tasks
- **domain_assessment_parser.py** - Parsers for domain assessment data formats

#### dataset/templates/ - Dataset Templates
- **__init__.py** - Module initialization
- **rag_mini_bioasq.py** - BioASQ dataset template and loader for medical domain RAG

#### dataset/utils/ - Dataset Utilities
- **__init__.py** - Module initialization
- **dataset_loader.py** - Generic dataset loading utilities

### embeddings_ops/ - Embedding Operations
Embedding generation and manipulation utilities. Extends existing embeddings/ module with batch processing and validation.

- **__init__.py** - Exports EmbeddingGenerator, SteeringEmbeddingGenerator, SteeringConfig, SteeringMode, and utility functions
- **embed.py** - EmbeddingGenerator: Batch embedding generation with validation (NaN/Inf checks, centroid similarity sanity checks)
- **steering_config.py** - Configuration classes for model steering/control mechanisms (SteeringConfig, SteeringMode) - moved here to avoid circular imports
- **steering_embedding_generator.py** - SteeringEmbeddingGenerator: Generates steering embeddings for model control with augmentations

### jasper/ - JASPER Dataset Builder
JASPER-specific dataset builder orchestration. Project-specific but demonstrates patterns for future dataset builders.

- **__init__.py** - Exports build_dataset, BuildConfig, DatasetFinalizer, finalize_dataset, SQLiteStorageManager
- **build.py** - build_dataset: Task 9 orchestrator that runs Tasks 1-8 sequentially with per-task timing/logging and final config validation
- **builder_config.py** - BuildConfig: Configuration dataclass for JASPER dataset building with validation and JSON serialization
- **finalize.py** - DatasetFinalizer: Cross-validates all builder outputs, checks referential integrity/dimensions, and writes final config.json
- **sqlite_storage.py** - SQLiteStorageManager: SQLite + numpy memmap storage backend for dataset persistence

### pytorch_datasets/ - PyTorch Dataset Implementations
PyTorch Dataset classes for training. Specific implementations but patterns are highly reusable.

- **__init__.py** - Exports JASPERSteeringDataset, RAG datasets, ClusterLabeledDataset, DataLoader utilities
- **jasper_steering_dataset.py** - JASPERSteeringDataset: PyTorch dataset for JASPER - pre-computed embedding triplets with hard negatives, curriculum learning, device placement, context manager, referential integrity validation, HDF5 storage format support, memory-mapped loading for large datasets (>10GB)
- **rag_dataset.py** - RAGDataset: Core PyTorch dataset class for RAG question-answer-source triples
- **cluster_labeled_dataset.py** - ClusterLabeledDataset: PyTorch dataset with cluster assignments and labels
- **loader.py** - DataLoader factory and validation utilities for JASPER Steering Dataset (create_loader, validate_first_batch, set_epoch)

### embeddings/ - Embedding I/O
Embedding generation, storage, and retrieval.

- **__init__.py** - Exports embedding classes
- **io.py** - Embedding input/output operations and serialization
- **keyword_embedder.py** - KeywordEmbedder: Generates embeddings for keywords and terms

### nn/ - Neural Network Models
Lightweight neural network models for RAG improvement. Implements JASPER (Joint Architecture for Subspace Prediction with Explainable Routing) — JEPA-style predictor with EMA, multi-objective losses, subspace routing, and XAI.

- **__init__.py** - Exports all models, losses, training, and inference components from Phase 1 and Phase 2

#### nn/models/ - Model Implementations
- **__init__.py** - Exports JASPERPredictor, EMAEncoder, SubspaceRouter, DecomposedJASPERPredictor, and ConfigurableModel
- **model_builder.py** - Factory functions for building and configuring neural network models
- **jasper_predictor.py** - JASPERPredictor: JEPA-style predictor (question_emb + steering_emb → predicted_source_emb) with YAML config support
- **ema_encoder.py** - EMAEncoder: Exponential Moving Average wrapper for target encoder with cosine tau schedule (tau_min=0.996 → tau_max=0.999)
- **subspace_router.py** - SubspaceRouter: Concept bottleneck routing (question+steering → routing_weights [K]) with Gumbel-Softmax, XAI explain() method
- **decomposed_predictor.py** - DecomposedJASPERPredictor: Coarse+fine prediction (prediction = centroid_anchor + residual), returns full explanation_dict with atypicality

#### nn/losses/ - Loss Functions
- **__init__.py** - Exports all loss classes
- **jasper_losses.py** - JASPERLoss, ContrastiveLoss (InfoNCE), CentroidLoss, VICRegLoss, JASPERMultiObjectiveLoss — combined loss with per-component logging
- **routing_losses.py** - RoutingLoss, EntropyRegularization (diversity→confidence schedule), ResidualPenalty, DisentanglementLoss — routing-specific objectives

#### nn/training/ - Training Utilities
- **__init__.py** - Exports JASPERTrainer, JASPERTrainerConfig, TrainingMonitor
- **jasper_trainer.py** - JASPERTrainer: Training loop with EMA updates, curriculum learning, checkpoint save/load, and full metric logging
- **monitoring.py** - TrainingMonitor: Metric collection, loss curve plotting (per-component subplots), steering distribution evolution, CSV/JSON export

#### nn/inference/ - Inference & XAI
- **__init__.py** - Exports XAIInterface
- **xai_interface.py** - XAIInterface: Inference-time explainability — explain_prediction() returns primary_subspace, routing_distribution, steering_influence (KL/L2), atypicality, similar_known_pairs, actionable_signal

### prompts_templates/ - LLM Prompt Templates
Centralized prompt definitions for all agents. Never hardcode prompts in agent code.

- **__init__.py** - Module initialization
- **domain_extraction.py** - Prompts for domain extraction, guessing, and assessment operations
- **rag_generators.py** - Prompts for RAG question and answer generation
- **rag_verifiers.py** - Prompts for RAG quality verification and validation
- **text_augmentation.py** - Prompts for text augmentation while preserving meaning

### utils/ - General Utilities
Cross-cutting utilities for text processing and data manipulation.

- **__init__.py** - Module initialization
- **suggestion_processing.py** - Processing utilities for LLM-generated suggestions
- **text_splitters.py** - Text splitting utilities for chunking and segmentation
- **text_utils.py** - Text validation, cleaning, and transformation utilities (e.g., is_empty_text)

---

## tests/ - Unit Tests

Test modules follow pattern `test_<module_name>.py`. All tests mock LLM calls for reproducibility.

- **conftest.py** - Pytest configuration and shared fixtures
- **test_builder_config.py** - Tests for BuildConfig (initialization, validation, serialization)
- **test_dataset_check_agent.py** - Tests for DatasetCheckAgent (LangGraph workflow, mocking)
- **test_dataset_splitter.py** - Tests for DatasetSplitter (splitting logic, persistence)
- **test_domain_assesment.py** - Tests for DomainAnalysisAgent (all three operation modes)
- **test_jasper_steering_dataset.py** - Tests for JASPERSteeringDataset (initialization, getitem, steering, curriculum, storage formats [PT/HDF5], memory-mapping, HDF5 conversion)
- **test_keyword_clustering.py** - Tests for KeywordClustering (clustering algorithms)
- **test_loader.py** - Tests for JASPER Steering DataLoader (batch shapes, iteration, validation)
- **test_merge_csv.py** - Tests for CSVMerger (normalization, deduplication, ID assignment)
- **test_parse_clusters.py** - Tests for ClusterParser (keyword matching, fuzzy matching, cluster metadata)
- **test_link_sources.py** - Tests for SourceClusterLinker (pair-to-cluster linking, fallback strategies, validation)
- **test_embed.py** - Tests for EmbeddingGenerator (batch generation, validation, sanity checks)
- **test_build_steering.py** - Tests for SteeringBuilder (centroid, keyword-weighted, residual steering variants, distances, validations)
- **test_dataset_build.py** - Tests for Task 9 build orchestrator (end-to-end pipeline execution, artifact generation, storage format validation)
- **test_mine_negatives.py** - Tests for NegativeMiner (4-tier negative sampling, validation, edge cases with small clusters)
- **test_split.py** - Tests for DatasetSplitter (question-level splitting, stratification, no leakage validation, determinism)
- **test_finalize.py** - Tests for DatasetFinalizer (cross-validation, referential integrity checks, config writing)
- **test_question_augmentation_agent.py** - Tests for QuestionAugmentationAgent
- **test_source_evaluation_agent.py** - Tests for SourceEvaluationAgent
- **test_text_augmentation_agent.py** - Tests for TextAugmentationAgent
- **test_topic_distance_calculator.py** - Tests for TopicDistanceCalculator
- **test_validation_utils.py** - Tests for validation_utils shared utilities (tensor validation, bounds checking, etc.)
- **test_tensor_utils.py** - Tests for tensor_utils I/O functions (loading, saving, batch operations)
- **test_monitoring.py** - Tests for TrainingMonitor (metric logging, W&B payload bool filtering, plot_losses/plot_steering_distribution guards, CSV+JSON export, empty-history regression)
- **test_jasper_predictor.py** - Tests for JASPERPredictor (initialization, forward pass shapes, gradient flow, YAML config loading)
- **test_ema_encoder.py** - Tests for EMAEncoder (EMA update correctness, tau cosine schedule, state dict save/load, no-grad target encoding)
- **test_jasper_losses.py** - Tests for individual and combined JASPER losses (JASPERLoss, ContrastiveLoss, CentroidLoss, VICRegLoss, JASPERMultiObjectiveLoss collapse prevention)
- **test_jasper_trainer.py** - Tests for JASPERTrainer (single training step, EMA update, checkpoint save/load, curriculum set_epoch)
- **test_subspace_router.py** - Tests for SubspaceRouter (routing_weights validity, Gumbel-Softmax differentiability, XAI output structure)
- **test_decomposed_predictor.py** - Tests for DecomposedJASPERPredictor (coarse+fine decomposition, atypicality = ||fine||, centroid alignment)
- **test_routing_losses.py** - Tests for RoutingLoss, EntropyRegularization, ResidualPenalty, DisentanglementLoss
- **test_xai_interface.py** - Tests for XAIInterface (full XAI output structure, steering influence KL/L2, batch processing)

---

## docs/ - Documentation

### Root Documentation
- **README.md** - Documentation index with links to all guides

### docs/agents/ - Agent Documentation
Detailed usage guides for each agent with examples.

- **README.md** - Agent documentation index
- **AGENTS_OVERVIEW.md** - Comprehensive comparison of all agents with workflows
- **CSV_QUESTION_AGENT.md** - QuestionAugmentationAgent usage guide
- **DATASET_CHECK_AGENT.md** - DatasetCheckAgent usage guide (LangGraph workflow)
- **DOMAIN_ANALYSIS_AGENT.md** - DomainAnalysisAgent usage guide (three modes)
- **SOURCE_EVALUATION_AGENT.md** - SourceEvaluationAgent usage guide
- **TEXT_AUGMENTATION.md** - TextAugmentationAgent usage guide

### docs/nn/ - Neural Network Documentation
Documentation for JASPER predictor, training, subspace routing, and XAI components.

- **JASPER_PREDICTOR.md** - JASPERPredictor and EMAEncoder architecture, YAML config reference, EMA schedule, usage examples
- **TRAINING_JASPER.md** - JASPERTrainer training script guide, config format, tuning tips, checkpoint management, troubleshooting
- **SUBSPACE_JASPER.md** - DecomposedJASPERPredictor routing architecture, coarse+fine decomposition, training with routing losses
- **XAI_INTERFACE.md** - XAIInterface output format, field interpretation guide, use cases, steering influence analysis

### docs/clustering/ - Clustering Documentation
- **README.md** - Clustering documentation index
- **CLUSTERING_AND_ASSIGNMENT.md** - Clustering workflows and patterns
- **TOPIC_DISTANCE_CALCULATOR.md** - TopicDistanceCalculator usage guide

### docs/dataset/ - Dataset Documentation
- **README.md** - Dataset documentation index
- **DATASET_SPLITTING.md** - DatasetSplitter guide with persistence examples
- **DOMAIN_ASSESSMENT_CLUSTERING_DATASET_EXAMPLE.md** - Domain assessment with clustering examples
- **DOMAIN_ASSESSMENT_EXAMPLES.md** - Domain assessment usage examples
- **JASPER_STEERING_DATASET.md** - JASPER Steering Dataset guide: PyTorch dataset for pre-computed embedding triplets with curriculum learning and hard negatives
- **JASPER_TRAINING_EXAMPLE.md** - Training examples for JASPER Steering Dataset with curriculum learning and hard negatives

### docs/pytorch_datasets/ - PyTorch Dataset Documentation
- **README.md** - PyTorch datasets overview, quick start, feature comparison table
- **JASPER_STEERING_DATASET.md** - JASPERSteeringDataset: curriculum learning, hard negatives, zero I/O training, HDF5/PT storage formats, memory-mapping
- **CLUSTER_LABELED_DATASET.md** - ClusterLabeledDataset: domain classification with memmap storage and LRU cache
- **RAG_DATASET.md** - BaseRAGDatasetGenerator: abstract base for RAG triplet generation and ChromaDB sampling
- **LOADER_UTILITIES.md** - DataLoader factory functions, batch validation, distributed training support
- **STORAGE_FORMATS.md** - Comprehensive guide to PT, HDF5, and memory-mapped loading strategies for large datasets

---

### Root-Level Files (examples/, configs/)

#### examples/ - Training Scripts
- **train_jasper_predictor.py** - CLI training script for JASPERPredictor with EMA; supports `--config` and `--resume` flags, runs end-to-end JASPER Phase 1 training
- **train_subspace_jasper.py** - CLI training script for DecomposedJASPERPredictor with subspace routing, routing accuracy logging, and XAI validation on val set

#### configs/ - Training Configurations
- **jasper_base.yaml** - Base YAML config for JASPERPredictor training (model, ema, loss, training, dataset sections)
- **subspace_jasper.yaml** - YAML config for DecomposedJASPERPredictor training with routing loss weights and subspace parameters

---

## agents_notes/ - Project Tooling & Structure Documentation

- **PROJECT_STRUCTURE.md** - Comprehensive file-by-file listing of all project files with concise purpose descriptions
- **MODULE_MAP_USAGE.md** - Guide for coding agents: how to generate and search the module map
- **generate_module_map.py** - CLI script that AST-parses Python files and emits `agent_ignore/module_map.json` (classes, methods, functions, docstrings)
- **search_module_map.py** - CLI search tool for `agent_ignore/module_map.json`; supports filtering by symbol type (class/method/function/module), package, parent_module; outputs human-readable or JSON

---

## agent_ignore/ - Generated Artifacts (Hidden from Copilot)

> This directory is excluded from VS Code file explorer and Copilot search context via `.vscode/settings.json`.
> Its contents are also git-ignored (`agent_ignore/*` in `.gitignore`); only `.gitkeep` is tracked.
---

## File Count Summary

- **Python Source Files**: 62 files in RAG_supporters/ (added 9 JASPER Phase 1 & 2 source files)
- **Test Files**: 27 files in tests/ (added 8 JASPER test files)
- **Documentation Files**: 25 markdown files in docs/ (added 4 docs/nn/ files)
- **Configuration**: 4 files (pyproject.toml, .gitignore, jasper_base.yaml, subspace_jasper.yaml)
- **Root Documentation**: 2 files (AGENTS.md, README.md)
- **Tooling Scripts**: 3 files in agents_notes/ (generate_module_map.py, search_module_map.py, MODULE_MAP_USAGE.md)
- **Training Scripts**: 2 files in examples/ (train_jasper_predictor.py, train_subspace_jasper.py)

---

## Update Guidelines

When submitting a PR that modifies the file structure:

1. **New Files**: Add entry under appropriate section with concise description
2. **Deleted Files**: Remove entry from this document
3. **Renamed Files**: Update both old and new locations
4. **Significant Changes**: Update file description if purpose changes
5. **New Directories**: Add new section with directory overview

Keep descriptions **precise and concise** - one line per file explaining its primary purpose.
