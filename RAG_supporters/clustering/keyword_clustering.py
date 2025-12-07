"""
Keyword clustering using KMeans and Bisecting KMeans algorithms.

This module provides clustering capabilities for keyword embeddings:
- KMeans and Bisecting KMeans clustering algorithms
- Cluster assignment and grouping utilities
- Centroid-based similarity search and comparison
- Results saving and loading

TODO: Future extensions could include:
- Additional clustering algorithms (DBSCAN, hierarchical)
- Automatic optimal cluster number detection (elbow method, silhouette score)
- Cluster quality metrics and visualization
- Additional distance metrics (Manhattan, Mahalanobis)
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

LOGGER = logging.getLogger(__name__)


class KeywordClusterer:
    """
    Cluster keyword embeddings and perform centroid-based comparisons.

    This class combines clustering functionality with centroid comparison,
    allowing both cluster creation and similarity search operations.
    """

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
        self.cluster_info = {}
        self.topics = {}

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

    def extract_topic_descriptors(
        self,
        n_descriptors: int = 10,
        metric: str = "euclidean",
    ) -> Dict[int, List[str]]:
        """
        Extract topic descriptors (n closest keywords) for each cluster.

        This method identifies the most representative keywords for each cluster
        by finding the n keywords closest to each cluster centroid.

        Parameters
        ----------
        n_descriptors : int
            Number of closest keywords to use as descriptors per topic
        metric : str
            Distance metric: 'euclidean' or 'cosine'

        Returns
        -------
        Dict[int, List[str]]
            Dictionary mapping cluster/topic IDs to lists of descriptor keywords

        Examples
        --------
        >>> clusterer = KeywordClusterer(n_clusters=3)
        >>> clusterer.fit(keyword_embeddings)
        >>> topics = clusterer.extract_topic_descriptors(n_descriptors=5)
        >>> print(topics[0])  # Top 5 keywords for cluster 0
        ['machine learning', 'deep learning', 'neural networks', ...]
        """
        if self.cluster_labels is None:
            raise ValueError("Model not fitted. Call fit() first.")

        centroids = self.get_centroids()
        topics = {}

        for cluster_id in range(self.n_clusters):
            centroid = centroids[cluster_id]

            # Calculate distances from all keywords to this centroid
            if metric == "euclidean":
                distances = np.linalg.norm(self.embeddings_matrix - centroid, axis=1)
            elif metric == "cosine":
                similarities = cosine_similarity([centroid], self.embeddings_matrix)[0]
                distances = 1 - similarities
            else:
                raise ValueError(
                    f"Unknown metric: {metric}. Choose 'euclidean' or 'cosine'"
                )

            # Get indices of n closest keywords
            closest_indices = np.argsort(distances)[:n_descriptors]

            # Get the corresponding keywords
            descriptors = [self.keywords[idx] for idx in closest_indices]
            topics[cluster_id] = descriptors

        self.topics = topics
        LOGGER.info(
            f"Extracted {n_descriptors} descriptors for each of {len(topics)} topics"
        )

        return topics

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
        centroids = self.get_centroids()

        if metric == "euclidean":
            distances = np.linalg.norm(centroids - embedding, axis=1)
        elif metric == "cosine":
            # Cosine distance = 1 - cosine similarity
            similarities = cosine_similarity([embedding], centroids)[0]
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
        cluster_labels_list = list(range(self.n_clusters))

        if top_k == 1:
            nearest_idx = np.argmin(distances)
            return cluster_labels_list[nearest_idx], float(distances[nearest_idx])
        else:
            nearest_indices = np.argsort(distances)[:top_k]
            return [
                (cluster_labels_list[idx], float(distances[idx]))
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
        cluster_labels_list = list(range(self.n_clusters))

        distance_dict = {
            label: float(dist) for label, dist in zip(cluster_labels_list, distances)
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

    def save_results(
        self,
        output_path: str,
        include_embeddings: bool = False,
        include_topics: bool = True,
    ):
        """
        Save clustering results to JSON.

        Parameters
        ----------
        output_path : str
            Path to save results
        include_embeddings : bool
            Whether to include embeddings in output
        include_topics : bool
            Whether to include topic descriptors in output
        """
        # Get cluster assignments and grouped clusters
        assignments = self.get_cluster_assignments()
        clusters = self.get_clusters()

        # Calculate cluster statistics
        cluster_stats = {}
        for label, keywords in clusters.items():
            stats = {
                "size": len(keywords),
                "keywords_sample": keywords[:10],  # First 10 for preview
            }

            # Add topic descriptors if available and requested
            if include_topics and self.topics and label in self.topics:
                stats["topic_descriptors"] = self.topics[label]

            cluster_stats[str(label)] = stats

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

    @classmethod
    def from_results(
        cls,
        clustering_results_path: str,
    ) -> "KeywordClusterer":
        """
        Create clusterer from saved clustering results.

        Note: This creates a clusterer with centroids loaded for comparison
        operations. The underlying model is not fully fitted, so clustering
        operations (like fit()) should not be used on the returned instance.

        Parameters
        ----------
        clustering_results_path : str
            Path to clustering results JSON

        Returns
        -------
        KeywordClusterer
            Initialized clusterer with loaded centroids for comparison
        """
        with open(clustering_results_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        metadata = data.get("metadata", {})
        clusterer = cls(
            algorithm=metadata.get("algorithm", "kmeans"),
            n_clusters=metadata.get("n_clusters", 8),
            random_state=metadata.get("random_state", 42),
        )

        # Load centroids
        centroids = np.array(data["centroids"])

        # Create a minimal fitted model state for comparison operations
        # We create dummy data to fit the model, then replace centroids
        dummy_data = np.zeros((clusterer.n_clusters, centroids.shape[1]))
        clusterer.model.fit(dummy_data)
        clusterer.model.cluster_centers_ = centroids

        # Get cluster info if available
        if "cluster_stats" in data:
            for label_str, stats in data["cluster_stats"].items():
                label = int(label_str)
                clusterer.cluster_info[label] = stats
                # Load topic descriptors if available
                if "topic_descriptors" in stats:
                    clusterer.topics[label] = stats["topic_descriptors"]

        LOGGER.info(
            f"Loaded clustering results with {len(centroids)} centroids from {clustering_results_path}"
        )

        return clusterer


def cluster_keywords_from_embeddings(
    keyword_embeddings: Dict[str, np.ndarray],
    n_clusters: int = 8,
    algorithm: str = "kmeans",
    n_descriptors: int = 10,
    output_path: Optional[str] = None,
    random_state: int = 42,
    **kwargs,
) -> Tuple[KeywordClusterer, Dict[int, List[str]]]:
    """
    Complete pipeline to cluster keywords and extract topics.

    This is a convenience function that performs the full clustering workflow:
    1. Cluster keyword embeddings
    2. Extract topic descriptors
    3. Optionally save results

    Parameters
    ----------
    keyword_embeddings : Dict[str, np.ndarray]
        Dictionary mapping keywords to embeddings
    n_clusters : int
        Number of clusters/topics to discover
    algorithm : str
        Clustering algorithm: 'kmeans' or 'bisecting_kmeans'
    n_descriptors : int
        Number of descriptors per topic
    output_path : Optional[str]
        Path to save results (if None, results are not saved)
    random_state : int
        Random state for reproducibility
    **kwargs
        Additional arguments for the clustering algorithm

    Returns
    -------
    Tuple[KeywordClusterer, Dict[int, List[str]]]
        Tuple of (fitted clusterer, topic descriptors)

    Examples
    --------
    >>> from RAG_supporters.embeddings import KeywordEmbedder
    >>> embedder = KeywordEmbedder()
    >>> keywords = ["machine learning", "deep learning", "neural networks"]
    >>> embeddings = embedder.create_embeddings(keywords)
    >>> clusterer, topics = cluster_keywords_from_embeddings(
    ...     embeddings,
    ...     n_clusters=2,
    ...     n_descriptors=5,
    ...     output_path="results/keyword_clusters.json"
    ... )
    >>> print(f"Found {len(topics)} topics")
    Found 2 topics
    """
    # Create and fit clusterer
    clusterer = KeywordClusterer(
        algorithm=algorithm,
        n_clusters=n_clusters,
        random_state=random_state,
        **kwargs,
    )
    clusterer.fit(keyword_embeddings)

    # Extract topic descriptors
    topics = clusterer.extract_topic_descriptors(n_descriptors=n_descriptors)

    # Save if output path provided
    if output_path:
        clusterer.save_results(output_path, include_topics=True)

    return clusterer, topics
