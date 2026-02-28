"""
Embedding utilities for text embedding creation and processing.

This package provides modular components for working with embeddings:

- text_embedder: TextEmbedder class for creating and managing embeddings
- io: CSV loading utilities for suggestions

These tools support the RAG and clustering workflows by enabling efficient
text embedding generation and similarity-based operations.

Recommended usage:
    from RAG_supporters.embeddings import TextEmbedder
    from RAG_supporters.embeddings.io import load_suggestions_from_csv
"""

from .text_embedder import TextEmbedder

__all__ = [
    "TextEmbedder",
]
