# Documentation Index

This directory contains comprehensive documentation for all RAG Supporters components.

## Developer Guidelines

- **[AGENTS.md](../AGENTS.md)** - Technical guidelines, architecture, commands, code style, and development patterns for the entire project

## Agent Documentation

The RAG Supporters library provides specialized LLM-powered agents for RAG dataset creation and curation.

### Overview
- **[AGENTS_OVERVIEW.md](agents/AGENTS_OVERVIEW.md)** - Complete overview of all agents with comparison matrix, workflows, and best practices

### Individual Agent Documentation

1. **[QuestionAugmentationAgent](agents/CSV_QUESTION_AGENT.md)** - Question generation and rephrasing
   - Rephrase questions to align with sources or domains
   - Generate alternative questions from sources
   - Batch CSV/DataFrame processing

2. **[TextAugmentationAgent](agents/TEXT_AUGMENTATION.md)** - Text augmentation while preserving meaning
   - Full text and sentence-level rephrasing
   - Dataset augmentation for training data
   - Configurable augmentation modes

3. **[DatasetCheckAgent](agents/DATASET_CHECK_AGENT.md)** - Source comparison and selection
   - Compare two sources for a question
   - Duplicate detection
   - Quality control workflows

4. **[DomainAnalysisAgent](agents/DOMAIN_ANALYSIS_AGENT.md)** - Domain extraction and assessment
   - Extract domains from text (EXTRACT mode)
   - Guess domains needed for questions (GUESS mode)
   - Assess question relevance to domains (ASSESS mode)

5. **[SourceEvaluationAgent](agents/SOURCE_EVALUATION_AGENT.md)** - Multi-dimensional source quality scoring
   - 6-dimensional evaluation (relevance, expertise, depth, clarity, objectivity, completeness)
   - Source ranking and quality control
   - Batch processing support

## Quick Start

```python
from langchain_openai import ChatOpenAI
from RAG_supporters.agents import (
    QuestionAugmentationAgent,
    TextAugmentationAgent,
    DatasetCheckAgent,
    DomainAnalysisAgent,
    SourceEvaluationAgent
)

# Initialize any agent
llm = ChatOpenAI(model="gpt-4")
agent = QuestionAugmentationAgent(llm=llm)
```

## Installation

```bash
# Install with LLM provider (choose one)
pip install -e .[openai]   # For OpenAI support
pip install -e .[nvidia]   # For NVIDIA support
pip install -e .           # Base agents only

# Or install individually (if not using pyproject.toml)
pip install langgraph pydantic pandas tqdm
pip install langchain-openai  # For OpenAI
```

## Documentation Structure

Each agent documentation includes:
- **Overview** - Purpose and key features
- **Installation** - Dependencies and setup
- **Basic Usage** - Getting started examples
- **Advanced Usage** - Complex scenarios and configurations
- **DataFrame/CSV Processing** - Batch processing examples
- **Complete Examples** - Real-world use cases
- **API Reference** - Detailed method documentation
- **Best Practices** - Recommendations and tips
- **Troubleshooting** - Common issues and solutions

## Other Documentation

### Module Documentation

Comprehensive reference for all modules organized by category:

- **[Contrastive Learning](CONTRASTIVE_LEARNING.md)** - NegativeMiner (4-tier hard negative sampling), SteeringBuilder (steering signal generation)
- **[Data Preparation](DATA_PREPARATION.md)** - CSVMerger, DatasetSplitter (simple and stratified versions)
- **[Data Validation](DATA_VALIDATION.md)** - validation_utils, tensor_utils, label_calculator
- **[JASPER Builder](JASPER_BUILDER.md)** - BuildConfig, DatasetFinalizer, build_dataset orchestrator
- **[Embeddings Operations](EMBEDDINGS_OPERATIONS.md)** - EmbeddingGenerator, SteeringEmbeddingGenerator, SteeringConfig/Mode
- **[Clustering Operations](CLUSTERING_OPERATIONS.md)** - ClusterParser, SourceClusterLinker
- **[General Utilities](GENERAL_UTILITIES.md)** - text_utils, suggestion_processing, text_splitters

### Utilities

- **[TOPIC_DISTANCE_CALCULATOR.md](./clustering/TOPIC_DISTANCE_CALCULATOR.md)** - Calculate embedding distances to topic keywords without LLM
- **[CLUSTERING_AND_ASSIGNMENT.md](./clustering/CLUSTERING_AND_ASSIGNMENT.md)** - Guide for keyword clustering and cluster assignment
- **[DATASET_SPLITTING.md](./dataset/DATASET_SPLITTING.md)** - Guide for splitting datasets with persistent sample tracking

### Datasets

- **[JASPER Builder Guide](./dataset/JASPER_BUILDER_GUIDE.md)** - **Complete user guide for building JASPER datasets** from CSV and cluster JSON
- **[JASPER_STEERING_DATASET.md](./dataset/JASPER_STEERING_DATASET.md)** - PyTorch dataset for pre-computed embedding triplets with hard negatives (JASPER: Joint Architecture for Subspace Prediction with Explainable Routing). Features: zero-I/O training, curriculum learning, GPU preloading, context manager support, referential integrity validation
- **[JASPER_TRAINING_EXAMPLE.md](./dataset/JASPER_TRAINING_EXAMPLE.md)** - Training examples for JASPER steering dataset
- **[DATASET_SPLITTING.md](./dataset/DATASET_SPLITTING.md)** - Guide for splitting datasets with persistent sample tracking
- **[SAMPLE_GENERATION_GUIDE.md](../RAG_supporters/dataset/SAMPLE_GENERATION_GUIDE.md)** - Guide for generating RAG dataset samples
- **[QUICK_REFERENCE.md](../RAG_supporters/dataset/QUICK_REFERENCE.md)** - Quick reference for dataset utilities
- **[DOMAIN_ASSESSMENT_CLUSTERING_DATASET_EXAMPLE.md](./dataset/DOMAIN_ASSESSMENT_CLUSTERING_DATASET_EXAMPLE.md)** - Domain assessment with clustering examples

### PyTorch Datasets

- **[PyTorch Datasets Overview](./pytorch_datasets/README.md)** - Complete overview of all PyTorch Dataset implementations with quick start and comparison
- **[JASPERSteeringDataset](./pytorch_datasets/JASPER_STEERING_DATASET.md)** - Pre-computed embeddings with curriculum learning and hard negatives for JASPER training
- **[ClusterLabeledDataset](./pytorch_datasets/CLUSTER_LABELED_DATASET.md)** - Domain classification dataset with memmap storage and efficient LRU caching
- **[RAG Dataset Generator](./pytorch_datasets/RAG_DATASET.md)** - Abstract base for RAG triplet generation with ChromaDB storage
- **[DataLoader Utilities](./pytorch_datasets/LOADER_UTILITIES.md)** - Factory functions and validation utilities for creating DataLoaders

## Contributing

When adding new agents or features:
1. Follow the existing documentation structure
2. Include practical examples
3. Document all public methods
4. Add cross-references to related agents
5. Update this README and `agents/AGENTS_OVERVIEW.md`

## Support

For questions or issues:
- Check the specific agent documentation in `agents/`
- Review the troubleshooting sections
- Consult `agents/AGENTS_OVERVIEW.md` for workflows and patterns
