"""
Clustering utilities for keyword and suggestion embeddings.

This module provides clustering tools for working with embeddings:

- KeywordClusterer: Cluster embeddings using KMeans or Bisecting KMeans,
  with integrated centroid comparison methods, similarity search, and
  topic descriptor extraction for subspace/cluster discovery (Phase 1)
- SourceAssigner: Assign sources to clusters/subspaces (Phase 2)
- Convenience functions for complete clustering and assignment pipelines

These tools support the subspace/cluster steering roadmap by enabling
organized grouping of embeddings and efficient similarity-based retrieval.
"""

from .keyword_clustering import (KeywordClusterer,
                                 cluster_keywords_from_embeddings)
from .source_assignment import SourceAssigner, assign_sources_to_clusters

__all__ = [
    "KeywordClusterer",
    "cluster_keywords_from_embeddings",
    "SourceAssigner",
    "assign_sources_to_clusters",
]
