"""
Clustering utilities for keyword and suggestion embeddings.

This module provides clustering tools for working with embeddings:

- ClusteringData: Data container for cluster information with JSON loader
- KeywordClusterer: Unified clustering class with KMeans or Bisecting KMeans,
  integrated centroid comparison methods, similarity search, topic descriptor
  extraction (Phase 1), and source-to-cluster assignment (Phase 2)
- Convenience functions for complete clustering workflows

These tools support the subspace/cluster steering roadmap by enabling
organized grouping of embeddings and efficient similarity-based retrieval.

The KeywordClusterer class combines clustering and assignment functionality,
supporting both hard (one-hot) and soft (multi-subspace) assignment modes
with configurable temperature scaling and thresholds.
"""

from .clustering_data import ClusteringData
from .keyword_clustering import KeywordClusterer, cluster_keywords_from_embeddings
from .topic_distance_calculator import (
    TopicDistanceCalculator,
    calculate_topic_distances_from_csv,
)

__all__ = [
    "ClusteringData",
    "KeywordClusterer",
    "cluster_keywords_from_embeddings",
    "TopicDistanceCalculator",
    "calculate_topic_distances_from_csv",
]
