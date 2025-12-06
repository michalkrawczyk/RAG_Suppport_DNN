"""
Embedding utilities for keyword extraction, embedding creation, and processing.

This package provides modular components for working with embeddings:

- keyword_embedder: KeywordEmbedder class for creating and managing embeddings
- io: CSV loading utilities for suggestions
- core: DEPRECATED - Suggestion processing moved to utils.suggestion_processing

These tools support the RAG and clustering workflows by enabling efficient
keyword extraction, embedding generation, and similarity-based operations.

Recommended usage:
    from RAG_supporters.embeddings import KeywordEmbedder
    from RAG_supporters.utils.suggestion_processing import filter_by_field_value
"""

from .keyword_embedder import KeywordEmbedder

# Re-export from core for backward compatibility (now redirects to utils)
from .core import aggregate_unique_terms, filter_by_field_value

# Re-export from io
from .io import load_suggestions_from_csv

__all__ = [
    "KeywordEmbedder",
    # Backward compatibility - these are now in utils.suggestion_processing
    "filter_by_field_value",
    "aggregate_unique_terms",
    # I/O
    "load_suggestions_from_csv",
]
