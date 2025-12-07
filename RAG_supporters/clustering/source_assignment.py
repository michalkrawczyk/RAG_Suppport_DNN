"""
Source assignment to clusters/subspaces based on suggestions.

This module provides Phase 2 functionality for assigning sources to clusters:
- Assign sources/questions to clusters using their suggestion embeddings
- Calculate probability distributions over clusters
- Support for one-hot (hard) and soft (multi-subspace) assignments
- Threshold and softmax-based assignment logic

Part of the subspace/cluster steering roadmap.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

LOGGER = logging.getLogger(__name__)


class SourceAssigner:
    """
    Assign sources to clusters/subspaces based on embeddings.

    Supports both hard (one-hot) and soft (probabilistic) assignment modes,
    enabling single or multi-subspace membership.
    """

    def __init__(
        self,
        cluster_centroids: np.ndarray,
        assignment_mode: str = "soft",
        threshold: Optional[float] = None,
        temperature: float = 1.0,
        metric: str = "cosine",
    ):
        """
        Initialize the source assigner.

        Parameters
        ----------
        cluster_centroids : np.ndarray
            Array of cluster centroids, shape (n_clusters, embedding_dim)
        assignment_mode : str
            Assignment mode: 'hard' (one-hot) or 'soft' (probabilistic)
        threshold : Optional[float]
            Threshold for assignment (used in hard mode or for filtering in soft mode)
            For hard mode: assign to cluster if probability > threshold
            For soft mode: include clusters with probability > threshold
        temperature : float
            Temperature parameter for softmax (used in soft mode)
            Higher values = more uniform distribution
            Lower values = more peaked distribution
        metric : str
            Distance metric: 'euclidean' or 'cosine'
        """
        self.cluster_centroids = cluster_centroids
        self.n_clusters = cluster_centroids.shape[0]
        self.assignment_mode = assignment_mode
        self.threshold = threshold
        self.temperature = temperature
        self.metric = metric

        # Validate parameters
        if assignment_mode not in ["hard", "soft"]:
            raise ValueError(
                f"Invalid assignment_mode: {assignment_mode}. Choose 'hard' or 'soft'"
            )

        if metric not in ["euclidean", "cosine"]:
            raise ValueError(
                f"Invalid metric: {metric}. Choose 'euclidean' or 'cosine'"
            )

        LOGGER.info(
            f"Initialized SourceAssigner with {self.n_clusters} clusters, "
            f"mode={assignment_mode}, metric={metric}"
        )

    def compute_distances(
        self,
        embedding: np.ndarray,
    ) -> np.ndarray:
        """
        Compute distances from embedding to all centroids.

        Parameters
        ----------
        embedding : np.ndarray
            Query embedding vector

        Returns
        -------
        np.ndarray
            Array of distances to each centroid
        """
        if self.metric == "euclidean":
            distances = np.linalg.norm(self.cluster_centroids - embedding, axis=1)
        elif self.metric == "cosine":
            similarities = cosine_similarity([embedding], self.cluster_centroids)[0]
            distances = 1 - similarities

        return distances

    def compute_probabilities(
        self,
        embedding: np.ndarray,
    ) -> np.ndarray:
        """
        Compute probability distribution over clusters using softmax.

        Parameters
        ----------
        embedding : np.ndarray
            Query embedding vector

        Returns
        -------
        np.ndarray
            Array of probabilities for each cluster (sums to 1.0)
        """
        distances = self.compute_distances(embedding)

        # Convert distances to similarities (negative distances)
        # Apply temperature scaling
        logits = -distances / self.temperature

        # Apply softmax
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
        probabilities = exp_logits / np.sum(exp_logits)

        return probabilities

    def assign_source(
        self,
        embedding: np.ndarray,
        return_probabilities: bool = True,
    ) -> Dict[str, Any]:
        """
        Assign a source to cluster(s).

        Parameters
        ----------
        embedding : np.ndarray
            Source embedding vector
        return_probabilities : bool
            Whether to include full probability distribution in results

        Returns
        -------
        Dict[str, Any]
            Assignment results with the following keys:
            - mode: Assignment mode used
            - clusters: List of assigned cluster IDs (or single ID for hard mode)
            - probabilities: Dict mapping cluster IDs to probabilities (if requested)
            - primary_cluster: ID of most likely cluster
        """
        probabilities = self.compute_probabilities(embedding)

        result = {
            "mode": self.assignment_mode,
        }

        if self.assignment_mode == "hard":
            # Hard assignment: assign to single cluster
            primary_cluster = int(np.argmax(probabilities))

            # Apply threshold if specified
            if self.threshold is not None:
                if probabilities[primary_cluster] < self.threshold:
                    # No assignment if below threshold
                    result["clusters"] = []
                    result["primary_cluster"] = None
                else:
                    result["clusters"] = [primary_cluster]
                    result["primary_cluster"] = primary_cluster
            else:
                result["clusters"] = [primary_cluster]
                result["primary_cluster"] = primary_cluster

        elif self.assignment_mode == "soft":
            # Soft assignment: assign to multiple clusters
            primary_cluster = int(np.argmax(probabilities))
            result["primary_cluster"] = primary_cluster

            # Apply threshold if specified
            if self.threshold is not None:
                # Only include clusters above threshold
                assigned_clusters = [
                    i for i, prob in enumerate(probabilities) if prob >= self.threshold
                ]
            else:
                # Include all clusters with their probabilities
                assigned_clusters = list(range(self.n_clusters))

            result["clusters"] = assigned_clusters

        # Include probabilities if requested
        if return_probabilities:
            result["probabilities"] = {
                i: float(prob) for i, prob in enumerate(probabilities)
            }

        return result

    def assign_sources_batch(
        self,
        source_embeddings: Dict[str, np.ndarray],
        return_probabilities: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Assign multiple sources to clusters.

        Parameters
        ----------
        source_embeddings : Dict[str, np.ndarray]
            Dictionary mapping source IDs to embeddings
        return_probabilities : bool
            Whether to include full probability distributions

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary mapping source IDs to assignment results
        """
        assignments = {}

        for source_id, embedding in source_embeddings.items():
            assignments[source_id] = self.assign_source(
                embedding,
                return_probabilities=return_probabilities,
            )

        LOGGER.info(f"Assigned {len(assignments)} sources to clusters")

        return assignments

    def save_assignments(
        self,
        assignments: Dict[str, Dict[str, Any]],
        output_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Save assignment results to JSON.

        Parameters
        ----------
        assignments : Dict[str, Dict[str, Any]]
            Assignment results from assign_sources_batch
        output_path : str
            Path to save results
        metadata : Optional[Dict[str, Any]]
            Additional metadata to include
        """
        # Calculate assignment statistics
        cluster_counts = {i: 0 for i in range(self.n_clusters)}
        multi_cluster_count = 0
        unassigned_count = 0

        for source_id, assignment in assignments.items():
            assigned_clusters = assignment.get("clusters", [])

            if len(assigned_clusters) == 0:
                unassigned_count += 1
            elif len(assigned_clusters) > 1:
                multi_cluster_count += 1

            for cluster_id in assigned_clusters:
                cluster_counts[cluster_id] += 1

        stats = {
            "total_sources": len(assignments),
            "unassigned_sources": unassigned_count,
            "multi_cluster_sources": multi_cluster_count,
            "cluster_counts": cluster_counts,
        }

        # Prepare output
        output_data = {
            "metadata": {
                "assignment_mode": self.assignment_mode,
                "threshold": self.threshold,
                "temperature": self.temperature,
                "metric": self.metric,
                "n_clusters": self.n_clusters,
            },
            "statistics": stats,
            "assignments": assignments,
        }

        # Add custom metadata if provided
        if metadata:
            output_data["metadata"].update(metadata)

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        LOGGER.info(f"Saved assignments to {output_path}")
        LOGGER.info(
            f"Stats: {stats['total_sources']} total, "
            f"{stats['unassigned_sources']} unassigned, "
            f"{stats['multi_cluster_sources']} multi-cluster"
        )

    @staticmethod
    def load_assignments(input_path: str) -> Dict[str, Any]:
        """
        Load assignment results from JSON.

        Parameters
        ----------
        input_path : str
            Path to assignments file

        Returns
        -------
        Dict[str, Any]
            Assignment results with metadata and statistics
        """
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        LOGGER.info(f"Loaded assignments from {input_path}")

        return data


def assign_sources_to_clusters(
    source_embeddings: Dict[str, np.ndarray],
    cluster_centroids: np.ndarray,
    assignment_mode: str = "soft",
    threshold: Optional[float] = None,
    temperature: float = 1.0,
    metric: str = "cosine",
    output_path: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Complete pipeline to assign sources to clusters.

    This is a convenience function that performs the full Phase 2 workflow:
    1. Initialize source assigner with cluster centroids
    2. Assign all sources to clusters
    3. Optionally save results

    Parameters
    ----------
    source_embeddings : Dict[str, np.ndarray]
        Dictionary mapping source IDs to embeddings
    cluster_centroids : np.ndarray
        Array of cluster centroids from Phase 1
    assignment_mode : str
        'hard' for one-hot assignment, 'soft' for probabilistic
    threshold : Optional[float]
        Threshold for assignment filtering
    temperature : float
        Softmax temperature (higher = more uniform)
    metric : str
        Distance metric: 'euclidean' or 'cosine'
    output_path : Optional[str]
        Path to save results (if None, results are not saved)
    metadata : Optional[Dict[str, Any]]
        Additional metadata to include in saved results

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Assignment results for all sources

    Examples
    --------
    >>> # After Phase 1 clustering
    >>> from RAG_supporters.clustering import SuggestionClusterer
    >>> clusterer = SuggestionClusterer.from_results("suggestion_clusters.json")
    >>> centroids = clusterer.clusterer.get_centroids()
    >>>
    >>> # Assign sources using their embeddings
    >>> from RAG_supporters.embeddings import KeywordEmbedder
    >>> embedder = KeywordEmbedder()
    >>> source_texts = {"source1": "text about ML", "source2": "text about DL"}
    >>> source_embeddings = embedder.create_embeddings(list(source_texts.values()))
    >>> source_embeddings = {k: source_embeddings[v] for k, v in source_texts.items()}
    >>>
    >>> assignments = assign_sources_to_clusters(
    ...     source_embeddings,
    ...     centroids,
    ...     assignment_mode="soft",
    ...     threshold=0.1,
    ...     output_path="results/source_assignments.json"
    ... )
    """
    # Create assigner
    assigner = SourceAssigner(
        cluster_centroids=cluster_centroids,
        assignment_mode=assignment_mode,
        threshold=threshold,
        temperature=temperature,
        metric=metric,
    )

    # Assign sources
    assignments = assigner.assign_sources_batch(source_embeddings)

    # Save if output path provided
    if output_path:
        assigner.save_assignments(assignments, output_path, metadata=metadata)

    return assignments
