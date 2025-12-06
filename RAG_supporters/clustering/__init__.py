"""
Clustering utilities for keyword and suggestion embeddings.

This module provides clustering tools for working with embeddings:

- KeywordClusterer: Cluster embeddings using KMeans or Bisecting KMeans,
  with integrated centroid comparison methods for similarity search
- SuggestionClusterer: Cluster suggestions to discover topics/subspaces (Phase 1)
- SourceAssigner: Assign sources to clusters/subspaces (Phase 2)
- Convenience functions for complete clustering and assignment pipelines

These tools support the subspace/cluster steering roadmap by enabling
organized grouping of embeddings and efficient similarity-based retrieval.
"""

from .keyword_clustering import KeywordClusterer
from .suggestion_clustering import (
    SuggestionClusterer,
    cluster_suggestions_from_embeddings,
)
from .source_assignment import (
    SourceAssigner,
    assign_sources_to_clusters,
)

__all__ = [
    "KeywordClusterer",
    "SuggestionClusterer",
    "cluster_suggestions_from_embeddings",
    "SourceAssigner",
    "assign_sources_to_clusters",
]
