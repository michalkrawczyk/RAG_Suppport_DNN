"""
Clustering utilities for keyword embeddings.

This module provides clustering and comparison tools for working with
keyword embeddings:

- KeywordClusterer: Cluster embeddings using KMeans or Bisecting KMeans
- CentroidComparator: Compare new embeddings against cluster centroids

These tools support the subspace/cluster steering roadmap by enabling
organized grouping of embeddings and efficient similarity-based retrieval.
"""

from .centroid_comparison import CentroidComparator
from .keyword_clustering import KeywordClusterer

__all__ = [
    "KeywordClusterer",
    "CentroidComparator",
]
