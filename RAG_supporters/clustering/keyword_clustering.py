"""
Keyword clustering using KMeans and Bisecting KMeans algorithms.

This module provides clustering capabilities for keyword embeddings:
- KMeans and Bisecting KMeans clustering algorithms
- Cluster assignment and grouping utilities
- Results saving and loading

TODO: Future extensions could include:
- Additional clustering algorithms (DBSCAN, hierarchical)
- Automatic optimal cluster number detection (elbow method, silhouette score)
- Cluster quality metrics and visualization
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

LOGGER = logging.getLogger(__name__)


class KeywordClusterer:
    """Cluster keyword embeddings using KMeans or Bisecting KMeans."""

    def __init__(
        self,
        algorithm: str = "kmeans",
        n_clusters: int = 8,
        random_state: int = 42,
        **kwargs,
    ):
        """
        Initialize the clusterer.

        Parameters
        ----------
        algorithm : str
            Clustering algorithm: 'kmeans' or 'bisecting_kmeans'
        n_clusters : int
            Number of clusters
        random_state : int
            Random state for reproducibility
        **kwargs
            Additional arguments for the clustering algorithm
        """
        self.algorithm = algorithm
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kwargs = kwargs

        self.model = self._create_model()
        self.keywords = None
        self.embeddings_matrix = None
        self.cluster_labels = None

    def _create_model(self):
        """Create the clustering model."""
        try:
            from sklearn.cluster import BisectingKMeans, KMeans
        except ImportError:
            raise ImportError(
                "scikit-learn is required for clustering. "
                "Install with: pip install scikit-learn"
            )

        if self.algorithm == "kmeans":
            return KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                **self.kwargs,
            )
        elif self.algorithm == "bisecting_kmeans":
            return BisectingKMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                **self.kwargs,
            )
        else:
            raise ValueError(
                f"Unknown algorithm: {self.algorithm}. "
                "Choose 'kmeans' or 'bisecting_kmeans'"
            )

    def fit(
        self,
        keyword_embeddings: Dict[str, np.ndarray],
    ) -> "KeywordClusterer":
        """
        Fit the clustering model.

        Parameters
        ----------
        keyword_embeddings : Dict[str, np.ndarray]
            Dictionary mapping keywords to embeddings

        Returns
        -------
        KeywordClusterer
            Self for chaining
        """
        # Convert to matrix
        self.keywords = list(keyword_embeddings.keys())
        self.embeddings_matrix = np.array(
            [keyword_embeddings[kw] for kw in self.keywords]
        )

        LOGGER.info(
            f"Fitting {self.algorithm} with {self.n_clusters} clusters "
            f"on {len(self.keywords)} keywords"
        )

        # Fit model
        self.model.fit(self.embeddings_matrix)
        self.cluster_labels = self.model.labels_

        LOGGER.info("Clustering complete")

        return self

    def get_cluster_assignments(self) -> Dict[str, int]:
        """
        Get cluster assignments for each keyword.

        Returns
        -------
        Dict[str, int]
            Dictionary mapping keywords to cluster labels
        """
        if self.cluster_labels is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return {
            keyword: int(label)
            for keyword, label in zip(self.keywords, self.cluster_labels)
        }

    def get_clusters(self) -> Dict[int, List[str]]:
        """
        Get keywords grouped by cluster.

        Returns
        -------
        Dict[int, List[str]]
            Dictionary mapping cluster labels to lists of keywords
        """
        if self.cluster_labels is None:
            raise ValueError("Model not fitted. Call fit() first.")

        clusters = {}
        for keyword, label in zip(self.keywords, self.cluster_labels):
            label = int(label)
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(keyword)

        return clusters

    def get_centroids(self) -> np.ndarray:
        """
        Get cluster centroids.

        Returns
        -------
        np.ndarray
            Array of cluster centroids, shape (n_clusters, embedding_dim)
        """
        if not hasattr(self.model, "cluster_centers_"):
            raise ValueError("Model not fitted or doesn't have centroids")

        return self.model.cluster_centers_

    def save_results(
        self,
        output_path: str,
        include_embeddings: bool = False,
    ):
        """
        Save clustering results to JSON.

        Parameters
        ----------
        output_path : str
            Path to save results
        include_embeddings : bool
            Whether to include embeddings in output
        """
        # Get cluster assignments and grouped clusters
        assignments = self.get_cluster_assignments()
        clusters = self.get_clusters()

        # Calculate cluster statistics
        cluster_stats = {
            str(label): {
                "size": len(keywords),
                "keywords_sample": keywords[:10],  # First 10 for preview
            }
            for label, keywords in clusters.items()
        }

        # Prepare output
        output_data = {
            "metadata": {
                "algorithm": self.algorithm,
                "n_clusters": self.n_clusters,
                "n_keywords": len(self.keywords),
                "random_state": self.random_state,
            },
            "cluster_assignments": assignments,
            "clusters": {str(k): v for k, v in clusters.items()},
            "cluster_stats": cluster_stats,
            "centroids": self.get_centroids().tolist(),
        }

        if include_embeddings:
            output_data["embeddings"] = {
                kw: emb.tolist()
                for kw, emb in zip(self.keywords, self.embeddings_matrix)
            }

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        LOGGER.info(f"Saved clustering results to {output_path}")

    @staticmethod
    def load_results(input_path: str) -> Dict[str, Any]:
        """
        Load clustering results from JSON.

        Parameters
        ----------
        input_path : str
            Path to results file

        Returns
        -------
        Dict[str, Any]
            Clustering results
        """
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data
