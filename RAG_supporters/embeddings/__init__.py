"""
Embedding utilities for keyword extraction, embedding creation, and processing.

This package provides modular components for working with embeddings:

- keyword_embedder: KeywordEmbedder class for creating and managing embeddings
- io: CSV loading utilities for suggestions

These tools support the RAG and clustering workflows by enabling efficient
keyword extraction, embedding generation, and similarity-based operations.

Recommended usage:
    from RAG_supporters.embeddings import KeywordEmbedder
    from RAG_supporters.embeddings.io import load_suggestions_from_csv
"""

from .keyword_embedder import KeywordEmbedder

__all__ = [
    "KeywordEmbedder",
]
