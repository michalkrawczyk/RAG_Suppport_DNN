"""
RAG Supporters - Tools and agents for RAG dataset creation, curation, and enhancement.

This package provides specialized LLM-powered agents and utilities for working with
Retrieval-Augmented Generation (RAG) datasets, including:

- Agents: Question augmentation, text augmentation, dataset checking, domain analysis,
  and source evaluation
- Clustering: Keyword clustering and topic distance calculation
- Dataset utilities: Splitting, labeling, and storage
- Embeddings: Embedding generation and I/O
"""

__version__ = "1.0.0"

from RAG_supporters.DEFAULT_CONSTS import (  # noqa: E402
    ColKeys,
    EmbKeys,
    PairArtifactKeys,
    DEFAULT_COL_KEYS,
    DEFAULT_EMB_KEYS,
    DEFAULT_PA_KEYS,
)

__all__ = [
    "ColKeys",
    "EmbKeys",
    "PairArtifactKeys",
    "DEFAULT_COL_KEYS",
    "DEFAULT_EMB_KEYS",
    "DEFAULT_PA_KEYS",
]
