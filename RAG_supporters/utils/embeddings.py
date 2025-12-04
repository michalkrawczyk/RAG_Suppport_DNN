"""
Embedding utilities for keyword extraction, embedding, and clustering.

This module has been refactored for better modularity. The original functionality
is now split across multiple focused modules:

- embedding_core: Core filtering, aggregation, and embedding creation
- embedding_io: CSV/JSON I/O utilities
- embedding_wrapper: High-level KeywordEmbedder class
- clustering.keyword_clustering: KeywordClusterer for KMeans-based clustering
- clustering.centroid_comparison: CentroidComparator for similarity search

This file maintains backward compatibility by re-exporting all public APIs.
For new code, prefer importing directly from the specialized modules.

TODO: In future iterations, consider further separations:
- Similarity/distance metrics into a dedicated module
- Batch processing utilities for large-scale operations
- Integration with neural pipeline steering mechanisms
- Enhanced clustering with auto-optimization of cluster numbers
"""

# Re-export core functions
from .embedding_core import (
    aggregate_unique_terms,
    create_embeddings_for_strings,
    filter_by_field_value,
)

# Re-export I/O functions
from .embedding_io import (
    load_embeddings_from_json,
    load_suggestions_from_csv,
    save_embeddings_to_json,
)

# Re-export wrapper class
from .embedding_wrapper import KeywordEmbedder

# Re-export clustering classes from new clustering module
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
