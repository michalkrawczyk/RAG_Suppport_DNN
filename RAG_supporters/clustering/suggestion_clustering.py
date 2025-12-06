"""
Clustering for suggestion embeddings to discover topics/subspaces.

This module provides Phase 1 functionality for topic discovery through clustering:
- Clustering suggestion embeddings using k-means or bisecting k-means
- Extracting topic descriptors (closest suggestions per cluster)
- Saving and loading clustering results with topic information

Part of the subspace/cluster steering roadmap.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .keyword_clustering import KeywordClusterer

LOGGER = logging.getLogger(__name__)


class SuggestionClusterer:
    """
    Cluster suggestion embeddings to discover topics/subspaces.
    
    This class extends KeywordClusterer functionality specifically for
    suggestions, providing topic extraction and descriptor methods.
    """

    def __init__(
        self,
        algorithm: str = "kmeans",
        n_clusters: int = 8,
        random_state: int = 42,
        **kwargs,
    ):
        """
        Initialize the suggestion clusterer.

        Parameters
        ----------
        algorithm : str
            Clustering algorithm: 'kmeans' or 'bisecting_kmeans'
        n_clusters : int
            Number of clusters/topics to discover
        random_state : int
            Random state for reproducibility
        **kwargs
            Additional arguments for the clustering algorithm
        """
        self.clusterer = KeywordClusterer(
            algorithm=algorithm,
            n_clusters=n_clusters,
            random_state=random_state,
            **kwargs,
        )
        self.suggestions = None
        self.suggestion_embeddings = None
        self.topics = {}

    def fit(
        self,
        suggestion_embeddings: Dict[str, np.ndarray],
    ) -> "SuggestionClusterer":
        """
        Fit the clustering model on suggestion embeddings.

        Parameters
        ----------
        suggestion_embeddings : Dict[str, np.ndarray]
            Dictionary mapping suggestions to embeddings

        Returns
        -------
        SuggestionClusterer
            Self for chaining
        """
        self.suggestion_embeddings = suggestion_embeddings
        self.suggestions = list(suggestion_embeddings.keys())
        
        # Fit the underlying clusterer
        self.clusterer.fit(suggestion_embeddings)
        
        LOGGER.info(f"Clustered {len(self.suggestions)} suggestions into {self.clusterer.n_clusters} topics")
        
        return self

    def extract_topic_descriptors(
        self,
        n_descriptors: int = 10,
        metric: str = "euclidean",
    ) -> Dict[int, List[str]]:
        """
        Extract topic descriptors (n closest suggestions) for each cluster.

        Parameters
        ----------
        n_descriptors : int
            Number of closest suggestions to use as descriptors per topic
        metric : str
            Distance metric: 'euclidean' or 'cosine'

        Returns
        -------
        Dict[int, List[str]]
            Dictionary mapping cluster/topic IDs to lists of descriptor suggestions
        """
        if self.clusterer.cluster_labels is None:
            raise ValueError("Model not fitted. Call fit() first.")

        centroids = self.clusterer.get_centroids()
        topics = {}

        for cluster_id in range(self.clusterer.n_clusters):
            centroid = centroids[cluster_id]
            
            # Calculate distances from all suggestions to this centroid
            if metric == "euclidean":
                distances = np.linalg.norm(
                    self.clusterer.embeddings_matrix - centroid, axis=1
                )
            elif metric == "cosine":
                from sklearn.metrics.pairwise import cosine_similarity
                similarities = cosine_similarity(
                    [centroid], self.clusterer.embeddings_matrix
                )[0]
                distances = 1 - similarities
            else:
                raise ValueError(
                    f"Unknown metric: {metric}. Choose 'euclidean' or 'cosine'"
                )

            # Get indices of n closest suggestions
            closest_indices = np.argsort(distances)[:n_descriptors]
            
            # Get the corresponding suggestions
            descriptors = [self.clusterer.keywords[idx] for idx in closest_indices]
            topics[cluster_id] = descriptors

        self.topics = topics
        LOGGER.info(f"Extracted {n_descriptors} descriptors for each of {len(topics)} topics")
        
        return topics

    def get_cluster_assignments(self) -> Dict[str, int]:
        """
        Get cluster assignments for each suggestion.

        Returns
        -------
        Dict[str, int]
            Dictionary mapping suggestions to cluster labels
        """
        return self.clusterer.get_cluster_assignments()

    def get_clusters(self) -> Dict[int, List[str]]:
        """
        Get suggestions grouped by cluster.

        Returns
        -------
        Dict[int, List[str]]
            Dictionary mapping cluster labels to lists of suggestions
        """
        return self.clusterer.get_clusters()

    def save_results(
        self,
        output_path: str,
        include_embeddings: bool = False,
        include_topics: bool = True,
    ):
        """
        Save clustering results with topic information to JSON.

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
        for label, suggestions in clusters.items():
            stats = {
                "size": len(suggestions),
                "suggestions_sample": suggestions[:10],  # First 10 for preview
            }
            
            # Add topic descriptors if available and requested
            if include_topics and self.topics and label in self.topics:
                stats["topic_descriptors"] = self.topics[label]
            
            cluster_stats[str(label)] = stats

        # Prepare output
        output_data = {
            "metadata": {
                "algorithm": self.clusterer.algorithm,
                "n_clusters": self.clusterer.n_clusters,
                "n_suggestions": len(self.suggestions),
                "random_state": self.clusterer.random_state,
            },
            "cluster_assignments": assignments,
            "clusters": {str(k): v for k, v in clusters.items()},
            "cluster_stats": cluster_stats,
            "centroids": self.clusterer.get_centroids().tolist(),
        }

        if include_embeddings:
            output_data["embeddings"] = {
                sugg: emb.tolist()
                for sugg, emb in self.suggestion_embeddings.items()
            }

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        LOGGER.info(f"Saved clustering results with topics to {output_path}")

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
            Clustering results including topics
        """
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data

    @classmethod
    def from_results(
        cls,
        clustering_results_path: str,
    ) -> "SuggestionClusterer":
        """
        Create clusterer from saved clustering results.

        Note: This creates a clusterer with centroids loaded for comparison
        operations. The underlying model is not fully fitted.

        Parameters
        ----------
        clustering_results_path : str
            Path to clustering results JSON

        Returns
        -------
        SuggestionClusterer
            Initialized clusterer with loaded centroids and topics
        """
        with open(clustering_results_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        metadata = data.get("metadata", {})
        clusterer = cls(
            algorithm=metadata.get("algorithm", "kmeans"),
            n_clusters=metadata.get("n_clusters", 8),
            random_state=metadata.get("random_state", 42),
        )

        # Load centroids into the underlying KeywordClusterer
        centroids = np.array(data["centroids"])
        dummy_data = np.zeros((clusterer.clusterer.n_clusters, centroids.shape[1]))
        clusterer.clusterer.model.fit(dummy_data)
        clusterer.clusterer.model.cluster_centers_ = centroids

        # Load topics if available
        if "cluster_stats" in data:
            for label_str, stats in data["cluster_stats"].items():
                label = int(label_str)
                if "topic_descriptors" in stats:
                    clusterer.topics[label] = stats["topic_descriptors"]
                clusterer.clusterer.cluster_info[label] = stats

        LOGGER.info(
            f"Loaded clustering results with {len(centroids)} topics from {clustering_results_path}"
        )

        return clusterer


def cluster_suggestions_from_embeddings(
    suggestion_embeddings: Dict[str, np.ndarray],
    n_clusters: int = 8,
    algorithm: str = "kmeans",
    n_descriptors: int = 10,
    output_path: Optional[str] = None,
    random_state: int = 42,
    **kwargs,
) -> Tuple[SuggestionClusterer, Dict[int, List[str]]]:
    """
    Complete pipeline to cluster suggestions and extract topics.

    This is a convenience function that performs the full Phase 1 workflow:
    1. Cluster suggestion embeddings
    2. Extract topic descriptors
    3. Optionally save results

    Parameters
    ----------
    suggestion_embeddings : Dict[str, np.ndarray]
        Dictionary mapping suggestions to embeddings
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
    Tuple[SuggestionClusterer, Dict[int, List[str]]]
        Tuple of (fitted clusterer, topic descriptors)

    Examples
    --------
    >>> from RAG_supporters.embeddings import KeywordEmbedder
    >>> embedder = KeywordEmbedder()
    >>> suggestions = ["machine learning", "deep learning", "neural networks"]
    >>> embeddings = embedder.create_embeddings(suggestions)
    >>> clusterer, topics = cluster_suggestions_from_embeddings(
    ...     embeddings,
    ...     n_clusters=2,
    ...     n_descriptors=5,
    ...     output_path="results/suggestion_clusters.json"
    ... )
    >>> print(f"Found {len(topics)} topics")
    Found 2 topics
    """
    # Create and fit clusterer
    clusterer = SuggestionClusterer(
        algorithm=algorithm,
        n_clusters=n_clusters,
        random_state=random_state,
        **kwargs,
    )
    clusterer.fit(suggestion_embeddings)

    # Extract topic descriptors
    topics = clusterer.extract_topic_descriptors(n_descriptors=n_descriptors)

    # Save if output path provided
    if output_path:
        clusterer.save_results(output_path, include_topics=True)

    return clusterer, topics
