"""
Embedding utilities for keyword extraction, embedding, and clustering.

DEPRECATED: This module is maintained for backward compatibility only.

The embedding functionality has been reorganized:
- RAG_supporters.embeddings.KeywordEmbedder: Main class for embedding operations
- RAG_supporters.utils.suggestion_processing: LLM suggestion filtering/aggregation
- RAG_supporters.clustering.KeywordClusterer: KMeans-based clustering
- RAG_supporters.clustering.CentroidComparator: Similarity search

For new code, import directly:

    from RAG_supporters.embeddings import KeywordEmbedder
    from RAG_supporters.utils.suggestion_processing import filter_by_field_value
    from RAG_supporters.clustering import KeywordClusterer, CentroidComparator

This file maintains backward compatibility by re-exporting all public APIs.
"""

# Re-export from the new embeddings package
from ..embeddings import KeywordEmbedder, load_suggestions_from_csv

# Re-export from utils.suggestion_processing
from .suggestion_processing import aggregate_unique_terms, filter_by_field_value

# Re-export clustering classes
from ..clustering import CentroidComparator, KeywordClusterer

# For backward compatibility with old function-based API
create_embeddings_for_strings = KeywordEmbedder.create_embeddings
save_embeddings_to_json = KeywordEmbedder.save_embeddings
load_embeddings_from_json = KeywordEmbedder.load_embeddings

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
