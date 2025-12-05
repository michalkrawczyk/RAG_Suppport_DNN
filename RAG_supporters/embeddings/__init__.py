"""
Embedding utilities for keyword extraction, embedding creation, and processing.

This package provides modular components for working with embeddings:

- core: Core filtering, aggregation, and embedding creation functions
- io: CSV/JSON I/O utilities for embeddings
- wrapper: High-level KeywordEmbedder class for complete pipelines

These tools support the RAG and clustering workflows by enabling efficient
keyword extraction, embedding generation, and similarity-based operations.
"""

from .core import (
    aggregate_unique_terms,
    create_embeddings_for_strings,
    filter_by_field_value,
)
from .io import (
    load_embeddings_from_json,
    load_suggestions_from_csv,
    save_embeddings_to_json,
)
from .wrapper import KeywordEmbedder

__all__ = [
    # Core functions
    "filter_by_field_value",
    "aggregate_unique_terms",
    "create_embeddings_for_strings",
    # I/O functions
    "load_suggestions_from_csv",
    "save_embeddings_to_json",
    "load_embeddings_from_json",
    # Classes
    "KeywordEmbedder",
]
