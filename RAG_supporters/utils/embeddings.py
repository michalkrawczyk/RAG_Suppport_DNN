"""
Embedding utilities for keyword extraction, embedding, and clustering.

DEPRECATED: This module is maintained for backward compatibility only.

The embedding functionality has been moved to the RAG_supporters.embeddings package:
- embeddings.core: Core filtering, aggregation, and embedding creation
- embeddings.io: CSV/JSON I/O utilities
- embeddings.wrapper: High-level KeywordEmbedder class
- clustering.keyword_clustering: KeywordClusterer for KMeans-based clustering
- clustering.centroid_comparison: CentroidComparator for similarity search

For new code, import directly from RAG_supporters.embeddings or RAG_supporters.clustering:

    from RAG_supporters.embeddings import KeywordEmbedder, filter_by_field_value
    from RAG_supporters.clustering import KeywordClusterer, CentroidComparator

This file maintains backward compatibility by re-exporting all public APIs.
"""

# Re-export from the new embeddings package
from ..embeddings import (
    KeywordEmbedder,
    aggregate_unique_terms,
    create_embeddings_for_strings,
    filter_by_field_value,
    load_embeddings_from_json,
    load_suggestions_from_csv,
    save_embeddings_to_json,
)

# Re-export clustering classes
from ..clustering import CentroidComparator, KeywordClusterer

# Make all original exports available for backward compatibility
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
    "KeywordClusterer",
    "CentroidComparator",
]
