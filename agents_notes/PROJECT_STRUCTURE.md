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
- **agents_notes/** - Project structure documentation and agent-specific notes

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

### dataset/ - Dataset Management
Dataset creation, manipulation, splitting, and storage.

- **__init__.py** - Exports dataset classes
- **builder_config.py** - BuildConfig: Configuration dataclass for JASPER dataset building with validation and JSON serialization
- **cluster_labeled_dataset.py** - ClusterLabeledDataset: Dataset with cluster assignments and labels
- **dataset_builder_README.md** - README and specifications for JASPER dataset builder pipeline (Tasks 0-9)
- **dataset_splitter.py** - DatasetSplitter: Train/val/test splitting with persistence to JSON
- **domain_assessment_dataset_builder.py** - DomainAssessmentDatasetBuilder: Builds datasets for domain assessment tasks
- **domain_assessment_parser.py** - Parsers for domain assessment data formats
- **jasper_steering_dataset.py** - JASPERSteeringDataset: PyTorch dataset for JASPER (Joint Architecture for Subspace Prediction with Explainable Routing) - pre-computed embedding triplets with hard negatives, curriculum learning, device placement support, context manager, and referential integrity validation
- **jepa_steering_dataset.py** - REMOVED: JEPA renamed to JASPER, file raises ImportError with migration instructions
- **label_calculator.py** - Label calculation utilities for dataset annotations
- **loader.py** - DataLoader factory and utilities for JASPER Steering Dataset (create_loader, validate_first_batch)
- **merge_csv.py** - CSVMerger: Merges multiple CSV files with column normalization, deduplication, and ID assignment
- **parse_clusters.py** - ClusterParser: Parses KeywordClusterer JSON format with keyword matching (exact + cosine fallback)
- **link_sources.py** - SourceClusterLinker: Links question-source pairs to clusters via keyword intersection with majority voting
- **embed.py** - EmbeddingGenerator: Batch embedding generation with validation (NaN/Inf checks, centroid similarity)
- **build_steering.py** - SteeringBuilder: Generates steering signals (centroid, keyword-weighted, residual) and centroid distances for curriculum learning
- **build.py** - build_dataset Task 9 orchestrator: runs Tasks 1-8 sequentially with per-task timing/logging and final config validation
- **mine_negatives.py** - NegativeMiner: 4-tier hard negative sampling (in-cluster, adjacent, high-similarity, random) for contrastive learning
- **split.py** - DatasetSplitter: Question-level stratified train/val/test splitting with no leakage, saves to PyTorch tensors
- **finalize.py** - DatasetFinalizer: Cross-validates all builder outputs, checks referential integrity/dimensions, and writes final config.json
- **validation_utils.py** - Shared validation utilities for tensor type/shape checking, dimension consistency, and bounds validation - eliminates code duplication across builder classes
- **tensor_utils.py** - Tensor loading and storage utilities with shape validation and error handling - standardizes torch.load operations across dataset classes
- **rag_dataset.py** - RAGDataset: Core dataset class for RAG question-answer-source triples
- **sqlite_storage.py** - SQLite-based storage backend for dataset persistence
- **steering_embedding_generator.py** - Generates steering embeddings for model control

#### dataset/steering/ - Steering Configuration
- **__init__.py** - Module initialization
- **steering_config.py** - Configuration classes for model steering/control mechanisms

#### dataset/templates/ - Dataset Templates
- **__init__.py** - Module initialization
- **rag_mini_bioasq.py** - BioASQ dataset template and loader for medical domain RAG

#### dataset/utils/ - Dataset Utilities
- **__init__.py** - Module initialization
- **dataset_loader.py** - Generic dataset loading utilities

### embeddings/ - Embedding I/O
Embedding generation, storage, and retrieval.

- **__init__.py** - Exports embedding classes
- **io.py** - Embedding input/output operations and serialization
- **keyword_embedder.py** - KeywordEmbedder: Generates embeddings for keywords and terms

### nn/ - Neural Network Models
Lightweight neural network models for RAG improvement.

- **__init__.py** - Module initialization

#### nn/models/ - Model Implementations
- **__init__.py** - Exports model classes
- **model_builder.py** - Factory functions for building and configuring neural network models

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
- **test_jasper_steering_dataset.py** - Tests for JASPERSteeringDataset (initialization, getitem, steering, curriculum)
- **test_jepa_steering_dataset.py** - Legacy filename: Tests JASPERSteeringDataset (JEPA renamed to JASPER)
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

---

## docs/ - Documentation

### Root Documentation
- **README.md** - Documentation index with links to all guides
- **AGENTS.md** - Duplicate of root AGENTS.md (technical reference)

### docs/agents/ - Agent Documentation
Detailed usage guides for each agent with examples.

- **README.md** - Agent documentation index
- **AGENTS_OVERVIEW.md** - Comprehensive comparison of all agents with workflows
- **CSV_QUESTION_AGENT.md** - QuestionAugmentationAgent usage guide
- **DATASET_CHECK_AGENT.md** - DatasetCheckAgent usage guide (LangGraph workflow)
- **DOMAIN_ANALYSIS_AGENT.md** - DomainAnalysisAgent usage guide (three modes)
- **SOURCE_EVALUATION_AGENT.md** - SourceEvaluationAgent usage guide
- **TEXT_AUGMENTATION.md** - TextAugmentationAgent usage guide

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
- **JEPA_STEERING_DATASET.md** - Deprecated: Use JASPER_STEERING_DATASET.md (JEPA renamed to JASPER)
- **JEPA_TRAINING_EXAMPLE.md** - Deprecated: Use JASPER_TRAINING_EXAMPLE.md (JEPA renamed to JASPER)

---

## File Count Summary

- **Python Source Files**: 53 files in RAG_supporters/ (added build.py for Task 9)
- **Test Files**: 18 files in tests/ (added test_dataset_build.py)
- **Documentation Files**: 16 markdown files in docs/
- **Configuration**: 2 files (pyproject.toml, .gitignore)
- **Root Documentation**: 2 files (AGENTS.md, README.md)

---

## Update Guidelines

When submitting a PR that modifies the file structure:

1. **New Files**: Add entry under appropriate section with concise description
2. **Deleted Files**: Remove entry from this document
3. **Renamed Files**: Update both old and new locations
4. **Significant Changes**: Update file description if purpose changes
5. **New Directories**: Add new section with directory overview

Keep descriptions **precise and concise** - one line per file explaining its primary purpose.
