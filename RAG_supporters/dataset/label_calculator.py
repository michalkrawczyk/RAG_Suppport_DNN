"""Label calculation system for domain assessment dataset.

Calculates 3 types of labels:
1. Source/question labels: Based on CSV cluster_probabilities or centroid distances
2. Steering embedding labels: Based on centroid distances
3. Combined labels: Average of source and steering labels (for masking augmentation)
"""

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist

from clustering.clustering_data import ClusteringData
from utils.text_utils import normalize_string

logger = logging.getLogger(__name__)


class LabelNormalizationMethod(str, Enum):
    """Supported label normalization methods."""

    SOFTMAX = "softmax"
    L1 = "l1"


class LabelNormalizer(ABC):
    """Base class for label normalization strategies."""

    @abstractmethod
    def normalize(self, distances: np.ndarray) -> np.ndarray:
        """
        Normalize distances to probability distribution.

        Args:
            distances: Array of distances to cluster centroids

        Returns:
            Normalized probability vector
        """
        pass


class SoftmaxNormalizer(LabelNormalizer):
    """Softmax normalization: exp(-d) / sum(exp(-d))."""

    def __init__(self, temperature: float = 1.0):
        """
        Initialize normalizer.

        Args:
            temperature: Temperature parameter for softmax (default 1.0)
        """
        self.temperature = temperature

    def normalize(self, distances: np.ndarray) -> np.ndarray:
        """Softmax normalization with temperature scaling."""
        # Use negative distances (closer = higher probability)
        logits = -distances / self.temperature

        # Numerical stability: subtract max
        logits_max = np.max(logits)
        exp_logits = np.exp(logits - logits_max)

        # Handle underflow: if all exp_logits are 0, return uniform distribution
        exp_sum = np.sum(exp_logits)
        if exp_sum == 0 or np.isnan(exp_sum) or np.isinf(exp_sum):
            logger.warning(
                "Underflow in softmax normalization (all exp_logits are 0). "
                "Returning uniform distribution."
            )
            return np.ones_like(distances) / len(distances)

        # Normalize
        return exp_logits / exp_sum


class L1Normalizer(LabelNormalizer):
    """L1 normalization: inverse distances normalized to sum to 1."""

    def normalize(self, distances: np.ndarray) -> np.ndarray:
        """L1 normalization."""
        # Add small epsilon to avoid division by zero
        eps = 1e-10
        inverse_distances = 1.0 / (distances + eps)
        return inverse_distances / np.sum(inverse_distances)


def get_normalizer(method: LabelNormalizationMethod, **kwargs) -> LabelNormalizer:
    """
    Get label normalizer instance for the specified method.

    Args:
        method: Normalization method
        **kwargs: Additional arguments (e.g., temperature for softmax)

    Returns:
        LabelNormalizer instance
    """
    if method == LabelNormalizationMethod.SOFTMAX:
        temperature = kwargs.get("temperature", 1.0)
        return SoftmaxNormalizer(temperature=temperature)
    elif method == LabelNormalizationMethod.L1:
        return L1Normalizer()
    else:
        raise ValueError(f"Unknown normalization method: {method}")


class LabelCalculator:
    """
    Calculates cluster membership labels for dataset samples.

    Supports 3 label types:
    1. Source/question labels (from CSV or computed)
    2. Steering embedding labels (computed from centroids)
    3. Combined labels (for masking augmentation)
    """

    def __init__(
        self,
        clustering_data: ClusteringData,
        normalization_method: LabelNormalizationMethod = LabelNormalizationMethod.SOFTMAX,
        temperature: float = 1.0,
    ):
        """
        Initialize label calculator.

        Args:
            clustering_data: ClusteringData with centroids
            normalization_method: Method to normalize distances to probabilities
            temperature: Temperature for softmax normalization
        """
        self.clustering_data = clustering_data
        self.normalizer = get_normalizer(normalization_method, temperature=temperature)

        if clustering_data.centroids is None:
            raise ValueError("ClusteringData must have centroids for label calculation")

        self.n_clusters = clustering_data.n_clusters
        logger.info(f"Initialized LabelCalculator with {self.n_clusters} clusters")
        logger.info(f"Normalization: {normalization_method.value}")

    def calculate_source_labels_from_suggestions(
        self, suggestions: List[Dict], suggestion_embeddings: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Calculate source labels from suggestion texts and their embeddings.

        Computes average distance of all suggestion embeddings to each centroid,
        then normalizes to probability distribution.

        Args:
            suggestions: List of suggestion dicts with 'term' field
            suggestion_embeddings: Dict mapping suggestion terms to embeddings

        Returns:
            Probability vector of shape (n_clusters,)
        """
        if not suggestions:
            logger.warning("No suggestions provided, returning uniform distribution")
            return np.ones(self.n_clusters) / self.n_clusters

        # Collect embeddings for suggestions that we have
        valid_embeddings = []
        missing_terms = []
        for sugg in suggestions:
            term = sugg.get("term", "")
            # Normalize term to match the keys in suggestion_embeddings
            # (KeywordEmbedder normalizes all keys during embedding creation)
            normalized_term = normalize_string(term)
            if normalized_term in suggestion_embeddings:
                valid_embeddings.append(suggestion_embeddings[normalized_term])
            else:
                missing_terms.append(f"{term} (normalized: {normalized_term})")
                logger.debug(f"Missing embedding for suggestion: {term} (normalized: {normalized_term})")

        if missing_terms:
            logger.info(f"Missing embeddings for {len(missing_terms)}/{len(suggestions)} suggestions: {missing_terms[:3]}")
        
        if not valid_embeddings:
            logger.warning(
                f"No valid suggestion embeddings found for {len(suggestions)} suggestions, returning uniform distribution"
            )
            return np.ones(self.n_clusters) / self.n_clusters

        # Stack embeddings
        embeddings_matrix = np.vstack(valid_embeddings)

        # Compute distances to all centroids for each suggestion
        distances = cdist(
            embeddings_matrix, self.clustering_data.centroids, metric="cosine"
        )

        # Average distances across all suggestions
        avg_distances = np.mean(distances, axis=0)

        # Normalize to probabilities
        return self.normalizer.normalize(avg_distances)

    def calculate_question_labels_from_csv(
        self, cluster_probabilities: Optional[List[float]]
    ) -> np.ndarray:
        """
        Use pre-computed cluster probabilities from CSV.

        If probabilities not available, returns uniform distribution.

        Args:
            cluster_probabilities: Probability vector from CSV

        Returns:
            Probability vector of shape (n_clusters,)
        """
        if cluster_probabilities is None:
            logger.warning(
                "No cluster_probabilities in CSV, returning uniform distribution"
            )
            return np.ones(self.n_clusters) / self.n_clusters

        # Validate length
        if len(cluster_probabilities) != self.n_clusters:
            logger.error(
                f"cluster_probabilities length {len(cluster_probabilities)} "
                f"!= n_clusters {self.n_clusters}"
            )
            return np.ones(self.n_clusters) / self.n_clusters

        # Validate it's a probability distribution
        probs = np.array(cluster_probabilities, dtype=np.float32)
        prob_sum = np.sum(probs)

        if not np.isclose(prob_sum, 1.0, atol=1e-3):
            logger.warning(f"cluster_probabilities sum to {prob_sum}, normalizing")
            if prob_sum > 0:
                probs = probs / prob_sum
            else:
                return np.ones(self.n_clusters) / self.n_clusters

        return probs

    def calculate_question_labels_from_embedding(
        self, question_embedding: np.ndarray
    ) -> np.ndarray:
        """
        Calculate question labels from embedding (distance to centroids).

        Args:
            question_embedding: Question embedding vector

        Returns:
            Probability vector of shape (n_clusters,)
        """
        # Compute distances to all centroids
        distances = cdist(
            question_embedding.reshape(1, -1),
            self.clustering_data.centroids,
            metric="cosine",
        )[0]

        # Normalize to probabilities
        return self.normalizer.normalize(distances)

    def calculate_steering_labels(self, steering_embedding: np.ndarray) -> np.ndarray:
        """
        Calculate steering embedding labels from centroid distances.

        Args:
            steering_embedding: Steering embedding vector

        Returns:
            Probability vector of shape (n_clusters,)
        """
        # Check if zero embedding (no steering)
        if np.allclose(steering_embedding, 0.0):
            logger.debug("Zero steering embedding, returning uniform distribution")
            return np.ones(self.n_clusters) / self.n_clusters

        # Compute distances to all centroids
        distances = cdist(
            steering_embedding.reshape(1, -1),
            self.clustering_data.centroids,
            metric="cosine",
        )[0]

        # Normalize to probabilities
        return self.normalizer.normalize(distances)

    def calculate_combined_labels(
        self,
        source_or_question_labels: np.ndarray,
        steering_labels: np.ndarray,
        steering_weight: float = 0.5,
    ) -> np.ndarray:
        """
        Calculate combined labels for masking augmentation.

        Weighted average of source/question and steering labels.

        Args:
            source_or_question_labels: Source or question label vector
            steering_labels: Steering label vector
            steering_weight: Weight for steering labels (0-1)

        Returns:
            Combined probability vector
        """
        if not 0.0 <= steering_weight <= 1.0:
            raise ValueError(
                f"steering_weight must be in [0, 1], got {steering_weight}"
            )

        # Weighted average
        combined = (
            1 - steering_weight
        ) * source_or_question_labels + steering_weight * steering_labels

        # Ensure it's normalized (should already be, but just in case)
        combined = combined / np.sum(combined)

        return combined

    def calculate_all_labels(
        self,
        source_or_question_embedding: np.ndarray,
        steering_embedding: np.ndarray,
        csv_cluster_probabilities: Optional[List[float]] = None,
        suggestions: Optional[List[Dict]] = None,
        suggestion_embeddings: Optional[Dict[str, np.ndarray]] = None,
        use_csv_for_question: bool = True,
        steering_weight: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate all 3 label types for a sample.

        Args:
            source_or_question_embedding: Base embedding
            steering_embedding: Steering embedding
            csv_cluster_probabilities: Pre-computed probabilities from CSV (for questions)
            suggestions: List of suggestions (for sources)
            suggestion_embeddings: Embeddings for suggestions (for sources)
            use_csv_for_question: If True and csv_cluster_probabilities available,
                                  use them for question labels
            steering_weight: Weight for combined labels

        Returns:
            Tuple of (source_question_labels, steering_labels, combined_labels)
        """
        # Calculate source/question labels
        if suggestions and suggestion_embeddings:
            # Source: use suggestions
            source_question_labels = self.calculate_source_labels_from_suggestions(
                suggestions, suggestion_embeddings
            )
        elif use_csv_for_question and csv_cluster_probabilities:
            # Question: use CSV probabilities if available
            source_question_labels = self.calculate_question_labels_from_csv(
                csv_cluster_probabilities
            )
        else:
            # Fallback: compute from embedding
            source_question_labels = self.calculate_question_labels_from_embedding(
                source_or_question_embedding
            )

        # Calculate steering labels
        steering_labels = self.calculate_steering_labels(steering_embedding)

        # Calculate combined labels
        combined_labels = self.calculate_combined_labels(
            source_question_labels, steering_labels, steering_weight
        )

        return source_question_labels, steering_labels, combined_labels
