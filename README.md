This is a Python library for experiments on creating non-LLM solutions (specifically lightweight neural networks) for improving RAG (Retrieval-Augmented Generation) or creating new variants of it. The LLM-powered agents are used to support learning and prepare datasets for training these alternative models. It includes specialized agents for domain analysis, text augmentation, source evaluation, and dataset quality control.

## Features

- **LLM-Powered Agents**: Dataset preparation, quality control, and augmentation
- **PyTorch Datasets**: Pre-computed embedding datasets for efficient training
  - **JASPER Steering Dataset**: Zero I/O dataset with curriculum learning and hard negatives (Joint Architecture for Subspace Prediction with Explainable Routing)
  - **Cluster Labeled Dataset**: Domain assessment with cluster steering
- **Dataset Splitting**: Reproducible train/val/test splits with persistence
- **Clustering**: Keyword-based and topic-based clustering utilities
- **Embedding Generation**: Batch embedding with sentence-transformers and LangChain

## Quick Start: JASPER Steering Dataset

```python
from RAG_supporters.dataset import create_loader, validate_first_batch

# Create DataLoader
loader = create_loader(
    dataset_dir="/path/to/dataset",
    split="train",
    batch_size=32,
    num_workers=4,
)

# Validate first batch
validate_first_batch(loader)

# Training loop
for epoch in range(100):
    loader.dataset_obj.set_epoch(epoch)  # Update curriculum
    
    for batch in loader:
        question_emb = batch["question_emb"]     # [B, D]
        target_emb = batch["target_source_emb"]  # [B, D]
        steering = batch["steering"]             # [B, D]
        negatives = batch["negative_embs"]       # [B, N_neg, D]
        
        # Train model...
```

See [docs/dataset/JASPER_STEERING_DATASET.md](docs/dataset/JASPER_STEERING_DATASET.md) for complete documentation.

For more info read about functionalities read the docs.