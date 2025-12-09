"""
Keyword clustering using KMeans and Bisecting KMeans algorithms.

This module provides a unified clustering solution for keyword/suggestion embeddings
with integrated Phase 1 (clustering) and Phase 2 (assignment) functionality for the
subspace/cluster steering roadmap.

Main Components
---------------
KeywordClusterer : class
    Unified clustering class that handles:
    - Clustering keyword/suggestion embeddings (Phase 1)
    - Extracting topic descriptors from clusters
    - Assigning new items to clusters (Phase 2)
    - Saving and loading clustering configurations

cluster_keywords_from_embeddings : function
    Convenience function for the complete clustering workflow

Features
--------
Phase 1: Clustering Foundation
    - KMeans and Bisecting KMeans algorithms
    - Topic descriptor extraction (n closest keywords to centroids)
    - Cluster quality metrics and statistics
    - Persistent storage of results with metadata

Phase 2: Source Assignment
    - Hard (one-hot) and soft (multi-subspace) assignment modes
    - Temperature-scaled softmax probability distributions
    - Configurable threshold filtering
    - Batch processing support
    - State management and fitted model validation

Distance Metrics
    - Euclidean distance
    - Cosine similarity/distance

Examples
--------
Basic clustering workflow:

>>> from RAG_supporters.embeddings import KeywordEmbedder
>>> from RAG_supporters.clustering import cluster_keywords_from_embeddings
>>>
>>> # Create embeddings
>>> embedder = KeywordEmbedder()
>>> keywords = ["machine learning", "deep learning", "AI"]
>>> embeddings = embedder.create_embeddings(keywords)
>>>
>>> # Cluster and extract topics
>>> clusterer, topics = cluster_keywords_from_embeddings(
...     embeddings,
...     n_clusters=2,
...     n_descriptors=5,
...     output_path="clusters.json"
... )

Assignment workflow:

>>> # Load saved clustering
>>> from RAG_supporters.clustering import KeywordClusterer
>>> clusterer = KeywordClusterer.from_results("clusters.json")
>>>
>>> # Configure assignment
>>> clusterer.configure_assignment(
...     assignment_mode="soft",
...     threshold=0.15,
...     temperature=1.0,
...     metric="cosine"
... )
>>>
>>> # Assign new items
>>> new_embedding = embedder.create_embeddings(["neural networks"])
>>> result = clusterer.assign(new_embedding["neural networks"])
>>> print(result['primary_cluster'])
0

Notes
-----
- Models loaded via `from_results()` are in a partially initialized state
  suitable for assignment operations but should not be re-fitted
- Assignment configuration (mode, threshold, temperature, metric) is
  persisted in save/load operations
- All assignment methods validate fitted state before operating

See Also
--------
RAG_supporters.embeddings.KeywordEmbedder : For creating embeddings
docs/CLUSTERING_AND_ASSIGNMENT.md : Complete usage guide

TODO: Future extensions
-----------------------
- Additional clustering algorithms (DBSCAN, hierarchical, Agglomerative)
- Automatic optimal cluster number detection (elbow method, silhouette score)
- Cluster quality metrics and visualization tools
- Additional distance metrics (Manhattan, Mahalanobis)
- Online/incremental clustering support
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

        # Store centroids directly (independent of sklearn model)
        self._centroids = None
        self._is_fitted = False

        # Assignment configuration defaults
        self._default_assignment_mode = "soft"
        self._default_threshold = 0.1
        self._default_metric = "cosine"

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

        # Store centroids directly (independent of sklearn model)
        self._centroids = self.model.cluster_centers_.copy()
        self._is_fitted = True

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
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        return self._centroids

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

    def configure_assignment(
        self,
        assignment_mode: str = "soft",
        threshold: float = 0.1,
        metric: str = "cosine",
    ) -> "KeywordClusterer":
        """
        Configure default assignment parameters.

        Parameters
        ----------
        assignment_mode : str
            Assignment mode:
            - 'hard': Assigns to single best cluster only
            - 'soft': Assigns to all clusters above threshold probability
        threshold : float
            Minimum probability for cluster assignment (0.0 to 1.0)
            - For hard mode: minimum confidence to assign (else returns no clusters)
            - For soft mode: minimum probability to include cluster in assignment
            Recommended ranges:
            - Soft mode: 0.05-0.2 (include clusters with modest relevance)
            - Hard mode: 0.3-0.5 (require strong confidence)
        metric : str
            Distance metric: 'euclidean' or 'cosine'
            Recommended: 'cosine' for text embeddings

        Returns
        -------
        KeywordClusterer
            Self for chaining

        Examples
        --------
        >>> clusterer.configure_assignment(
        ...     mode="soft",
        ...     threshold=0.1,
        ...     metric="cosine"
        ... )
        """
        if assignment_mode not in ["hard", "soft"]:
            raise ValueError(
                f"Invalid assignment_mode: {assignment_mode}. Choose 'hard' or 'soft'"
            )

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")

        if metric not in ["euclidean", "cosine"]:
            raise ValueError(
                f"Invalid metric: {metric}. Choose 'euclidean' or 'cosine'"
            )

        self._default_assignment_mode = assignment_mode
        self._default_threshold = threshold
        self._default_metric = metric

        LOGGER.info(
            f"Configured assignment: mode={assignment_mode}, "
            f"threshold={threshold}, metric={metric}"
        )

        return self

    def _check_fitted(self):
        """Check if the model has been fitted."""
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() or load from results first.")

    def _compute_probabilities(
        self,
        embedding: np.ndarray,
        metric: str,
    ) -> np.ndarray:
        """
        Compute probability distribution over clusters.

        Converts distances to probabilities using similarity normalization:
        - For cosine: similarity = 1 - distance, normalized
        - For euclidean: similarity = 1 / (1 + distance), normalized

        Parameters
        ----------
        embedding : np.ndarray
            Query embedding vector
        metric : str
            Distance metric

        Returns
        -------
        np.ndarray
            Probability distribution over clusters (sums to 1.0)
        """
        # Validate embedding dimension
        if embedding.shape[0] != self._centroids.shape[1]:
            raise ValueError(
                f"Embedding dimension {embedding.shape[0]} doesn't match "
                f"centroid dimension {self._centroids.shape[1]}"
            )

        distances = self.compute_distances(embedding, metric)

        # Convert distances to similarities
        if metric == "cosine":
            # Cosine distance is in [0, 2], closer to 0 is more similar
            # Convert to similarity: 1 - distance gives us similarity in [0, 1]
            similarities = 1 - distances
            # Clip to handle numerical issues
            similarities = np.clip(similarities, 0, 1)
        else:  # euclidean
            # Euclidean distances are unbounded, use inverse relationship
            # Add small epsilon to avoid division by zero
            similarities = 1.0 / (1.0 + distances)

        # Normalize to probabilities (sum to 1)
        probabilities = similarities / similarities.sum()

        return probabilities

    def _hard_assignment(
        self,
        probabilities: np.ndarray,
        threshold: float,
    ) -> Tuple[List[int], Optional[int]]:
        """
        Perform hard (single-cluster) assignment.

        Parameters
        ----------
        probabilities : np.ndarray
            Probability distribution over clusters
        threshold : float
            Minimum probability required for assignment

        Returns
        -------
        Tuple[List[int], Optional[int]]
            (assigned_clusters, primary_cluster)
            If max probability < threshold: ([], None)
            Otherwise: ([best_cluster], best_cluster)
        """
        primary_cluster = int(np.argmax(probabilities))

        if probabilities[primary_cluster] < threshold:
            LOGGER.debug(
                f"Hard assignment: max probability {probabilities[primary_cluster]:.3f} "
                f"below threshold {threshold}, no assignment made"
            )
            return [], None

        return [primary_cluster], primary_cluster

    def _soft_assignment(
        self,
        probabilities: np.ndarray,
        threshold: float,
    ) -> Tuple[List[int], int]:
        """
        Perform soft (multi-cluster) assignment.

        Parameters
        ----------
        probabilities : np.ndarray
            Probability distribution over clusters
        threshold : float
            Minimum probability required for cluster inclusion

        Returns
        -------
        Tuple[List[int], int]
            (assigned_clusters, primary_cluster)
            assigned_clusters: All clusters with probability >= threshold
            primary_cluster: Cluster with highest probability
        """
        primary_cluster = int(np.argmax(probabilities))

        # Filter clusters by threshold
        assigned_clusters = [
            i for i, prob in enumerate(probabilities) if prob >= threshold
        ]

        if not assigned_clusters:
            LOGGER.warning(
                f"Soft assignment: no clusters above threshold {threshold}, "
                f"assigning to primary cluster {primary_cluster} only"
            )
            assigned_clusters = [primary_cluster]

        LOGGER.debug(
            f"Soft assignment: {len(assigned_clusters)} clusters above "
            f"threshold {threshold}"
        )

        return assigned_clusters, primary_cluster

    def assign(
        self,
        embedding: np.ndarray,
        mode: Optional[str] = None,
        threshold: Optional[float] = None,
        metric: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Assign an embedding to cluster(s).

        This method computes the probability distribution over clusters and
        assigns the embedding based on the specified mode and threshold.

        Parameters
        ----------
        embedding : np.ndarray
            Query embedding vector
        mode : Optional[str]
            Assignment mode ('hard' or 'soft'). Uses configured default if None.
            - 'hard': Single cluster assignment (if above threshold)
            - 'soft': Multi-cluster assignment (all clusters above threshold)
        threshold : Optional[float]
            Minimum probability for assignment. Uses configured default if None.
            Must be between 0.0 and 1.0.
        metric : Optional[str]
            Distance metric ('euclidean' or 'cosine'). Uses configured default if None.

        Returns
        -------
        Dict[str, Any]
            Assignment results with keys:
            - mode: Assignment mode used
            - probabilities: Dict mapping cluster IDs to probabilities (always included)
            - assigned_clusters: List of assigned cluster IDs (may be empty in hard mode)
            - primary_cluster: ID of most likely cluster (None if no assignment in hard mode)
            - threshold_used: Threshold value that was applied
            - metric: Distance metric used

        Examples
        --------
        >>> # Hard assignment (single cluster)
        >>> result = clusterer.assign(embedding, mode="hard", threshold=0.3)
        >>> print(result['assigned_clusters'])  # [2] or [] if below threshold

        >>> # Soft assignment (multi-cluster)
        >>> result = clusterer.assign(embedding, mode="soft", threshold=0.1)
        >>> print(result['assigned_clusters'])  # [0, 2, 5] - all clusters > 10%
        >>> print(result['probabilities'])      # {0: 0.45, 1: 0.05, 2: 0.25, ...}
        """
        self._check_fitted()

        # Use defaults if not specified
        mode = mode if mode is not None else self._default_assignment_mode
        threshold = threshold if threshold is not None else self._default_threshold
        metric = metric if metric is not None else self._default_metric

        # Validate parameters
        if mode not in ["hard", "soft"]:
            raise ValueError(f"Invalid mode: {mode}. Choose 'hard' or 'soft'")

        if threshold is None:
            raise ValueError(
                "Threshold is required. Set via configure_assignment() or pass explicitly."
            )

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")

        # Compute probabilities
        probabilities = self._compute_probabilities(embedding, metric)

        # Perform assignment based on mode
        if mode == "hard":
            assigned_clusters, primary_cluster = self._hard_assignment(
                probabilities, threshold
            )
        else:  # soft
            assigned_clusters, primary_cluster = self._soft_assignment(
                probabilities, threshold
            )

        return {
            "mode": mode,
            "probabilities": {i: float(prob) for i, prob in enumerate(probabilities)},
            "assigned_clusters": assigned_clusters,
            "primary_cluster": primary_cluster,
            "threshold_used": threshold,
            "metric": metric,
        }

    def assign_batch(
        self,
        embeddings: Dict[str, np.ndarray],
        mode: Optional[str] = None,
        threshold: Optional[float] = None,
        metric: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Assign multiple embeddings to clusters.

        Parameters
        ----------
        embeddings : Dict[str, np.ndarray]
            Dictionary mapping IDs to embedding vectors
        mode : Optional[str]
            Assignment mode. Uses default if None.
        threshold : Optional[float]
            Probability threshold. Uses default if None.
        metric : Optional[str]
            Distance metric. Uses default if None.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary mapping IDs to assignment results

        Examples
        --------
        >>> clusterer = KeywordClusterer(n_clusters=3)
        >>> clusterer.fit(keyword_embeddings)
        >>> source_embeddings = {"src1": emb1, "src2": emb2}
        >>> results = clusterer.assign_batch(source_embeddings, mode="soft")
        """
        self._check_fitted()

        assignments = {}
        for item_id, embedding in embeddings.items():
            assignments[item_id] = self.assign(
                embedding,
                mode=mode,
                threshold=threshold,
                metric=metric,
            )

        LOGGER.info(f"Assigned {len(assignments)} items to clusters")

        return assignments

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
        self._check_fitted()

        # Validate embedding dimension
        if embedding.shape[0] != self._centroids.shape[1]:
            raise ValueError(
                f"Embedding dimension {embedding.shape[0]} doesn't match "
                f"centroid dimension {self._centroids.shape[1]}"
            )

        if metric == "euclidean":
            distances = np.linalg.norm(self._centroids - embedding, axis=1)
        elif metric == "cosine":
            # Cosine distance = 1 - cosine similarity
            similarities = cosine_similarity([embedding], self._centroids)[0]
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
            Model to generate text embedding (must have encode() method)
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
        assignments = self.get_cluster_assignments() if self.keywords else {}
        clusters = self.get_clusters() if self.keywords else {}

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
                "n_keywords": len(self.keywords) if self.keywords else 0,
                "random_state": self.random_state,
                "embedding_dim": self._centroids.shape[1],
                "assignment_config": {
                    "default_mode": self._default_assignment_mode,
                    "default_threshold": self._default_threshold,
                    "default_metric": self._default_metric,
                },
            },
            "cluster_assignments": assignments,
            "clusters": {str(k): v for k, v in clusters.items()},
            "cluster_stats": cluster_stats,
            "centroids": self._centroids.tolist(),
        }

        if include_embeddings and self.keywords:
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

        This creates a clusterer with centroids loaded for assignment and
        comparison operations. The underlying sklearn clustering model is
        not restored, but all necessary components for assignment operations
        are available.

        Parameters
        ----------
        clustering_results_path : str
            Path to clustering results JSON

        Returns
        -------
        KeywordClusterer
            Initialized clusterer with loaded centroids and config

        Examples
        --------
        >>> # Save results
        >>> clusterer.fit(embeddings)
        >>> clusterer.save_results("clusters.json")

        >>> # Load and use for assignment
        >>> loaded = KeywordClusterer.from_results("clusters.json")
        >>> result = loaded.assign(new_embedding)
        """
        with open(clustering_results_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        metadata = data.get("metadata", {})
        clusterer = cls(
            algorithm=metadata.get("algorithm", "kmeans"),
            n_clusters=metadata.get("n_clusters", 8),
            random_state=metadata.get("random_state", 42),
        )

        # Restore assignment configuration if available
        assignment_config = metadata.get("assignment_config", {})
        if assignment_config:
            clusterer._default_assignment_mode = assignment_config.get(
                "default_mode", "soft"
            )
            clusterer._default_threshold = assignment_config.get(
                "default_threshold", 0.1
            )
            clusterer._default_metric = assignment_config.get(
                "default_metric", "cosine"
            )

        # Load centroids directly (no sklearn model needed for assignment)
        clusterer._centroids = np.array(data["centroids"])
        clusterer._is_fitted = True

        # Get cluster info if available
        if "cluster_stats" in data:
            for label_str, stats in data["cluster_stats"].items():
                label = int(label_str)
                clusterer.cluster_info[label] = stats
                # Load topic descriptors if available
                if "topic_descriptors" in stats:
                    clusterer.topics[label] = stats["topic_descriptors"]

        LOGGER.info(
            f"Loaded clustering results with {len(clusterer._centroids)} centroids "
            f"from {clustering_results_path}"
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
