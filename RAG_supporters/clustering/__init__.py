"""
Clustering utilities for keyword embeddings.

This module provides clustering tools for working with keyword embeddings:

- KeywordClusterer: Cluster embeddings using KMeans or Bisecting KMeans,
  with integrated centroid comparison methods for similarity search

These tools support the subspace/cluster steering roadmap by enabling
organized grouping of embeddings and efficient similarity-based retrieval.
"""

from .keyword_clustering import KeywordClusterer

__all__ = [
    "KeywordClusterer",
]
