"""
Centroid comparison utilities for cluster-based similarity search.

This module provides tools for comparing embeddings against cluster centroids:
- Distance computation (Euclidean and Cosine)
- Nearest cluster finding
- Keyword and text comparison against centroids

Useful for assigning new data to existing clusters or performing cluster-based
retrieval and steering operations.

TODO: Future extensions could include:
- Additional distance metrics (Manhattan, Mahalanobis)
- Batch comparison utilities for efficiency
- Integration with steering mechanisms for neural pipeline control
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

LOGGER = logging.getLogger(__name__)


class CentroidComparator:
    """Compare keywords or new text against cluster centroids."""

    def __init__(
        self,
        centroids: np.ndarray,
        cluster_labels: Optional[List[int]] = None,
        cluster_info: Optional[Dict[int, Dict[str, Any]]] = None,
    ):
        """
        Initialize the comparator.

        Parameters
        ----------
        centroids : np.ndarray
            Cluster centroids, shape (n_clusters, embedding_dim)
        cluster_labels : Optional[List[int]]
            Optional labels for clusters
        cluster_info : Optional[Dict[int, Dict[str, Any]]]
            Optional additional info about each cluster
        """
        self.centroids = centroids
        self.cluster_labels = cluster_labels or list(range(len(centroids)))
        self.cluster_info = cluster_info or {}

    @classmethod
    def from_clustering_results(
        cls,
        clustering_results_path: str,
    ) -> "CentroidComparator":
        """
        Create comparator from saved clustering results.

        Parameters
        ----------
        clustering_results_path : str
            Path to clustering results JSON

        Returns
        -------
        CentroidComparator
            Initialized comparator
        """
        with open(clustering_results_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        centroids = np.array(data["centroids"])
        cluster_labels = list(range(len(centroids)))

        # Get cluster info if available
        cluster_info = {}
        if "cluster_stats" in data:
            for label_str, stats in data["cluster_stats"].items():
                cluster_info[int(label_str)] = stats

        return cls(centroids, cluster_labels, cluster_info)

    def compute_distances(
        self,
        embedding: np.ndarray,
        metric: str = "euclidean",
    ) -> np.ndarray:
        """
        Compute distances from embedding to all centroids.

        Parameters
        ----------
        embedding : np.ndarray
            Query embedding vector
        metric : str
            Distance metric: 'euclidean' or 'cosine'

        Returns
        -------
        np.ndarray
            Array of distances to each centroid
        """
        if metric == "euclidean":
            distances = np.linalg.norm(self.centroids - embedding, axis=1)
        elif metric == "cosine":
            # Cosine distance = 1 - cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity

            similarities = cosine_similarity([embedding], self.centroids)[0]
            distances = 1 - similarities
        else:
            raise ValueError(
                f"Unknown metric: {metric}. Choose 'euclidean' or 'cosine'"
            )

        return distances

    def find_nearest_cluster(
        self,
        embedding: np.ndarray,
        metric: str = "euclidean",
        top_k: int = 1,
    ) -> Union[Tuple[int, float], List[Tuple[int, float]]]:
        """
        Find the nearest cluster(s) for an embedding.

        Parameters
        ----------
        embedding : np.ndarray
            Query embedding vector
        metric : str
            Distance metric
        top_k : int
            Number of nearest clusters to return

        Returns
        -------
        Union[Tuple[int, float], List[Tuple[int, float]]]
            If top_k=1: (cluster_label, distance)
            If top_k>1: List of (cluster_label, distance) tuples
        """
        distances = self.compute_distances(embedding, metric)

        if top_k == 1:
            nearest_idx = np.argmin(distances)
            return self.cluster_labels[nearest_idx], float(distances[nearest_idx])
        else:
            nearest_indices = np.argsort(distances)[:top_k]
            return [
                (self.cluster_labels[idx], float(distances[idx]))
                for idx in nearest_indices
            ]

    def get_all_distances(
        self,
        embedding: np.ndarray,
        metric: str = "euclidean",
        sorted: bool = True,
    ) -> Dict[int, float]:
        """
        Get distances to all clusters.

        Parameters
        ----------
        embedding : np.ndarray
            Query embedding vector
        metric : str
            Distance metric
        sorted : bool
            Whether to sort by distance (ascending)

        Returns
        -------
        Dict[int, float]
            Dictionary mapping cluster labels to distances
        """
        distances = self.compute_distances(embedding, metric)

        distance_dict = {
            label: float(dist) for label, dist in zip(self.cluster_labels, distances)
        }

        if sorted:
            distance_dict = dict(sorted(distance_dict.items(), key=lambda x: x[1]))

        return distance_dict

    def compare_keyword(
        self,
        keyword: str,
        keyword_embeddings: Dict[str, np.ndarray],
        metric: str = "euclidean",
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Compare a keyword against centroids.

        Parameters
        ----------
        keyword : str
            Keyword to compare
        keyword_embeddings : Dict[str, np.ndarray]
            Dictionary of keyword embeddings
        metric : str
            Distance metric
        top_k : int
            Number of nearest clusters to return

        Returns
        -------
        Dict[str, Any]
            Comparison results with nearest clusters and all distances
        """
        if keyword not in keyword_embeddings:
            raise ValueError(f"Keyword '{keyword}' not found in embeddings")

        embedding = keyword_embeddings[keyword]

        top_clusters = self.find_nearest_cluster(embedding, metric, top_k)
        all_distances = self.get_all_distances(embedding, metric, sorted=True)

        return {
            "keyword": keyword,
            "top_clusters": top_clusters,
            "all_distances": all_distances,
            "metric": metric,
        }

    def compare_text(
        self,
        text: str,
        embedding_model: Any,
        metric: str = "euclidean",
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Compare arbitrary text against centroids.

        Parameters
        ----------
        text : str
            Text to compare
        embedding_model : Any
            Model to generate text embedding
        metric : str
            Distance metric
        top_k : int
            Number of nearest clusters to return

        Returns
        -------
        Dict[str, Any]
            Comparison results
        """
        # Generate embedding for text
        embedding = embedding_model.encode([text], convert_to_numpy=True)[0]

        top_clusters = self.find_nearest_cluster(embedding, metric, top_k)
        all_distances = self.get_all_distances(embedding, metric, sorted=True)

        return {
            "text": text,
            "top_clusters": top_clusters,
            "all_distances": all_distances,
            "metric": metric,
        }
