"""
Steering Signal Builder for JASPER Steering Dataset.

This module generates steering signals for curriculum learning:
- Centroid steering: Direction from question to cluster centroid
- Keyword-weighted steering: Weighted average of pair keywords
- Residual steering: Residual from question to centroid
- Centroid distances: Cosine distance for curriculum scheduling

Key Features:
- Three steering variants for diverse training strategies
- Normalized steering vectors (unit length)
- Centroid distance computation for curriculum learning
- Validation of steering vector properties
- Support for pairs with no keywords (fallback strategies)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

from .validation_utils import (
    validate_tensor_2d,
    validate_tensor_1d,
    validate_embedding_dimensions,
    validate_pair_indices_bounds,
    validate_cluster_ids_bounds,
    validate_length_consistency,
    validate_keyword_ids_list
)

LOGGER = logging.getLogger(__name__)


class SteeringBuilder:
    """Build steering signals for curriculum learning.
    
    This class generates three types of steering signals:
    1. Centroid steering: Normalized direction from question to cluster centroid
    2. Keyword-weighted: Normalized weighted average of keyword embeddings
    3. Residual: Difference between question and centroid (off-center signal)
    
    Additionally computes centroid distances for curriculum scheduling.
    
    Parameters
    ----------
    question_embeddings : torch.Tensor
        Question embeddings [n_questions, dim]
    keyword_embeddings : torch.Tensor
        Keyword embeddings [n_keywords, dim]
    centroid_embeddings : torch.Tensor
        Cluster centroid embeddings [n_clusters, dim]
    pair_indices : torch.Tensor
        Pair indices [n_pairs, 2] mapping to (question_idx, source_idx)
    pair_cluster_ids : torch.Tensor
        Primary cluster ID per pair [n_pairs]
    pair_keyword_ids : List[List[int]]
        Keyword IDs associated with each pair
    normalize_residual : bool, optional
        Whether to normalize residual steering vectors (default: False)
    fallback_strategy : str, optional
        How to handle pairs with no keywords:
        - "centroid": Use centroid steering only
        - "zero": Use zero vector
        - "random": Use random unit vector
        Default: "centroid"
    show_progress : bool, optional
        Show progress bars (default: True)
    
    Attributes
    ----------
    question_embeddings : torch.Tensor
        Question embeddings
    keyword_embeddings : torch.Tensor
        Keyword embeddings
    centroid_embeddings : torch.Tensor
        Centroid embeddings
    pair_indices : torch.Tensor
        Pair index mapping
    pair_cluster_ids : torch.Tensor
        Cluster assignments
    pair_keyword_ids : List[List[int]]
        Keyword associations
    n_pairs : int
        Total number of pairs
    embedding_dim : int
        Embedding dimension
    normalize_residual : bool
        Whether to normalize residuals
    fallback_strategy : str
        Keyword fallback strategy
    show_progress : bool
        Progress bar visibility
    
    Examples
    --------
    >>> import torch
    >>> from RAG_supporters.dataset import SteeringBuilder
    >>> 
    >>> # Prepare embeddings
    >>> question_embs = torch.randn(100, 384)
    >>> keyword_embs = torch.randn(50, 384)
    >>> centroid_embs = torch.randn(5, 384)
    >>> pair_indices = torch.randint(0, 100, (500, 2))
    >>> pair_cluster_ids = torch.randint(0, 5, (500,))
    >>> pair_keyword_ids = [[0, 1, 2], [3, 4], ...]  # Variable length
    >>> 
    >>> # Build steering signals
    >>> builder = SteeringBuilder(
    ...     question_embeddings=question_embs,
    ...     keyword_embeddings=keyword_embs,
    ...     centroid_embeddings=centroid_embs,
    ...     pair_indices=pair_indices,
    ...     pair_cluster_ids=pair_cluster_ids,
    ...     pair_keyword_ids=pair_keyword_ids
    ... )
    >>> 
    >>> # Generate all steering signals
    >>> results = builder.build_all_steering()
    >>> print(results.keys())
    dict_keys(['centroid', 'keyword_weighted', 'residual', 'distances'])
    """
    
    def __init__(
        self,
        question_embeddings: torch.Tensor,
        keyword_embeddings: torch.Tensor,
        centroid_embeddings: torch.Tensor,
        pair_indices: torch.Tensor,
        pair_cluster_ids: torch.Tensor,
        pair_keyword_ids: List[List[int]],
        normalize_residual: bool = False,
        fallback_strategy: str = "centroid",
        show_progress: bool = True
    ):
        """Initialize steering builder."""
        # Validate inputs
        self._validate_inputs(
            question_embeddings,
            keyword_embeddings,
            centroid_embeddings,
            pair_indices,
            pair_cluster_ids,
            pair_keyword_ids
        )
        
        self.question_embeddings = question_embeddings
        self.keyword_embeddings = keyword_embeddings
        self.centroid_embeddings = centroid_embeddings
        self.pair_indices = pair_indices
        self.pair_cluster_ids = pair_cluster_ids
        self.pair_keyword_ids = pair_keyword_ids
        
        self.n_pairs = len(pair_indices)
        self.embedding_dim = question_embeddings.shape[1]
        self.normalize_residual = normalize_residual
        self.fallback_strategy = fallback_strategy
        self.show_progress = show_progress
        
        # Validate fallback strategy
        valid_strategies = ["centroid", "zero", "random"]
        if fallback_strategy not in valid_strategies:
            raise ValueError(
                f"Invalid fallback_strategy: {fallback_strategy}. "
                f"Must be one of {valid_strategies}"
            )
        
        LOGGER.info(
            f"Initialized SteeringBuilder: {self.n_pairs} pairs, "
            f"dim={self.embedding_dim}, fallback={fallback_strategy}"
        )
    
    def _validate_inputs(
        self,
        question_embeddings: torch.Tensor,
        keyword_embeddings: torch.Tensor,
        centroid_embeddings: torch.Tensor,
        pair_indices: torch.Tensor,
        pair_cluster_ids: torch.Tensor,
        pair_keyword_ids: List[List[int]]
    ):
        """Validate input tensors and shapes."""
        # Validate embedding dimensions consistency
        validate_embedding_dimensions(
            (question_embeddings, "question_embeddings"),
            (keyword_embeddings, "keyword_embeddings"),
            (centroid_embeddings, "centroid_embeddings")
        )
        
        # Validate pair structures
        validate_tensor_2d(pair_indices, "pair_indices", expected_cols=2)
        validate_tensor_1d(pair_cluster_ids, "pair_cluster_ids")
        
        # Validate length consistency
        n_pairs = pair_indices.shape[0]
        validate_length_consistency(
            (pair_cluster_ids, "pair_cluster_ids", n_pairs),
            (pair_keyword_ids, "pair_keyword_ids", n_pairs)
        )
        
        # Validate index bounds
        validate_pair_indices_bounds(
            pair_indices,
            n_questions=question_embeddings.shape[0],
            n_sources=question_embeddings.shape[0]  # Not used for source validation here
        )
        
        validate_cluster_ids_bounds(
            pair_cluster_ids,
            n_clusters=centroid_embeddings.shape[0],
            name="pair_cluster_ids"
        )
        
        # Validate keyword IDs structure
        validate_keyword_ids_list(
            pair_keyword_ids,
            n_pairs=n_pairs,
            n_keywords=keyword_embeddings.shape[0],
            name="pair_keyword_ids"
        )
    
    def build_centroid_steering(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build centroid-based steering vectors and distances.
        
        For each pair, computes:
        - Steering vector: Normalized direction from question to cluster centroid
        - Distance: Cosine distance to centroid (1 - cosine_similarity)
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (steering_vectors, centroid_distances)
            - steering_vectors: [n_pairs, dim], normalized to unit length
            - centroid_distances: [n_pairs], cosine distance in [0, 2]
        
        Examples
        --------
        >>> steering, distances = builder.build_centroid_steering()
        >>> print(steering.shape, distances.shape)
        torch.Size([500, 384]) torch.Size([500])
        >>> print(f"Min distance: {distances.min():.3f}")
        >>> print(f"Max distance: {distances.max():.3f}")
        """
        LOGGER.info("Building centroid steering vectors and distances")
        
        steering_vectors = torch.zeros(
            self.n_pairs, self.embedding_dim, dtype=torch.float32
        )
        centroid_distances = torch.zeros(self.n_pairs, dtype=torch.float32)
        
        iterator = range(self.n_pairs)
        if self.show_progress:
            iterator = tqdm(iterator, desc="Centroid steering", unit="pair")
        
        for pair_idx in iterator:
            question_idx = self.pair_indices[pair_idx, 0].item()
            cluster_id = self.pair_cluster_ids[pair_idx].item()
            
            question_emb = self.question_embeddings[question_idx]
            centroid_emb = self.centroid_embeddings[cluster_id]
            
            # Compute steering direction (question -> centroid)
            steering = centroid_emb - question_emb
            
            # Normalize to unit length
            norm = torch.norm(steering)
            if norm > 1e-8:
                steering = steering / norm
            else:
                # Question is exactly at centroid, use zero vector
                LOGGER.debug(f"Pair {pair_idx}: Question at centroid, using zero steering")
                steering = torch.zeros_like(steering)
            
            steering_vectors[pair_idx] = steering
            
            # Compute cosine distance
            # Distance = 1 - cosine_similarity
            # Handle case where embeddings might not be normalized
            q_norm = torch.norm(question_emb)
            c_norm = torch.norm(centroid_emb)
            
            if q_norm > 1e-8 and c_norm > 1e-8:
                cosine_sim = torch.dot(question_emb, centroid_emb) / (q_norm * c_norm)
                # Clamp to [-1, 1] to handle numerical errors
                cosine_sim = torch.clamp(cosine_sim, -1.0, 1.0)
                distance = 1.0 - cosine_sim
            else:
                # One or both embeddings are zero
                LOGGER.warning(f"Pair {pair_idx}: Zero embedding detected, setting distance=1.0")
                distance = torch.tensor(1.0)
            
            centroid_distances[pair_idx] = distance
        
        # Validate outputs
        self._validate_steering_vectors(steering_vectors, "centroid")
        self._validate_distances(centroid_distances)
        
        LOGGER.info(
            f"Built centroid steering: "
            f"mean_distance={centroid_distances.mean():.3f}, "
            f"std={centroid_distances.std():.3f}"
        )
        
        return steering_vectors, centroid_distances
    
    def build_keyword_weighted_steering(self) -> torch.Tensor:
        """Build keyword-weighted steering vectors.
        
        For each pair, computes weighted average of keyword embeddings
        associated with the pair, normalized to unit length.
        
        For pairs with no keywords, uses fallback strategy:
        - "centroid": Use centroid steering
        - "zero": Use zero vector
        - "random": Use random unit vector
        
        Returns
        -------
        torch.Tensor
            Keyword-weighted steering vectors [n_pairs, dim]
        
        Examples
        --------
        >>> steering = builder.build_keyword_weighted_steering()
        >>> print(steering.shape)
        torch.Size([500, 384])
        >>> # Check normalization
        >>> norms = torch.norm(steering, dim=1)
        >>> print(f"All unit vectors: {torch.allclose(norms, torch.ones_like(norms), atol=1e-6)}")
        """
        LOGGER.info("Building keyword-weighted steering vectors")
        
        steering_vectors = torch.zeros(
            self.n_pairs, self.embedding_dim, dtype=torch.float32
        )
        
        n_fallback = 0
        
        iterator = range(self.n_pairs)
        if self.show_progress:
            iterator = tqdm(iterator, desc="Keyword steering", unit="pair")
        
        for pair_idx in iterator:
            keyword_ids = self.pair_keyword_ids[pair_idx]
            
            if len(keyword_ids) == 0:
                # No keywords, use fallback
                n_fallback += 1
                steering = self._get_fallback_steering(pair_idx)
            else:
                # Compute weighted average of keyword embeddings
                # Equal weights for simplicity
                keyword_embs = self.keyword_embeddings[keyword_ids]
                steering = keyword_embs.mean(dim=0)
                
                # Normalize to unit length
                norm = torch.norm(steering)
                if norm > 1e-8:
                    steering = steering / norm
                else:
                    # Keywords sum to zero, use fallback
                    LOGGER.debug(
                        f"Pair {pair_idx}: Keywords sum to zero, using fallback"
                    )
                    steering = self._get_fallback_steering(pair_idx)
            
            steering_vectors[pair_idx] = steering
        
        if n_fallback > 0:
            LOGGER.info(
                f"Used fallback steering for {n_fallback}/{self.n_pairs} pairs "
                f"({100 * n_fallback / self.n_pairs:.1f}%)"
            )
        
        # Validate outputs
        self._validate_steering_vectors(steering_vectors, "keyword_weighted")
        
        LOGGER.info("Built keyword-weighted steering vectors")
        
        return steering_vectors
    
    def build_residual_steering(self) -> torch.Tensor:
        """Build residual steering vectors.
        
        For each pair, computes residual between question embedding and
        cluster centroid. This captures off-center signals that can be
        useful for curriculum learning.
        
        Returns
        -------
        torch.Tensor
            Residual steering vectors [n_pairs, dim]
            Normalized to unit length if normalize_residual=True
        
        Examples
        --------
        >>> steering = builder.build_residual_steering()
        >>> print(steering.shape)
        torch.Size([500, 384])
        """
        LOGGER.info("Building residual steering vectors")
        
        steering_vectors = torch.zeros(
            self.n_pairs, self.embedding_dim, dtype=torch.float32
        )
        
        iterator = range(self.n_pairs)
        if self.show_progress:
            iterator = tqdm(iterator, desc="Residual steering", unit="pair")
        
        for pair_idx in iterator:
            question_idx = self.pair_indices[pair_idx, 0].item()
            cluster_id = self.pair_cluster_ids[pair_idx].item()
            
            question_emb = self.question_embeddings[question_idx]
            centroid_emb = self.centroid_embeddings[cluster_id]
            
            # Compute residual (question - centroid)
            residual = question_emb - centroid_emb
            
            if self.normalize_residual:
                # Normalize to unit length
                norm = torch.norm(residual)
                if norm > 1e-8:
                    residual = residual / norm
                else:
                    # Question is exactly at centroid
                    LOGGER.debug(
                        f"Pair {pair_idx}: Question at centroid, zero residual"
                    )
                    residual = torch.zeros_like(residual)
            
            steering_vectors[pair_idx] = residual
        
        # Validate outputs
        self._validate_steering_vectors(
            steering_vectors,
            "residual",
            allow_zero=not self.normalize_residual
        )
        
        LOGGER.info("Built residual steering vectors")
        
        return steering_vectors
    
    def build_all_steering(self) -> Dict[str, torch.Tensor]:
        """Build all steering variants and distances.
        
        Convenience method that generates all steering signals in one call.
        
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary with keys:
            - "centroid": Centroid steering [n_pairs, dim]
            - "keyword_weighted": Keyword steering [n_pairs, dim]
            - "residual": Residual steering [n_pairs, dim]
            - "distances": Centroid distances [n_pairs]
        
        Examples
        --------
        >>> results = builder.build_all_steering()
        >>> for key, tensor in results.items():
        ...     print(f"{key}: {tensor.shape}")
        centroid: torch.Size([500, 384])
        keyword_weighted: torch.Size([500, 384])
        residual: torch.Size([500, 384])
        distances: torch.Size([500])
        """
        LOGGER.info("Building all steering variants")
        
        # Build all variants
        centroid_steering, centroid_distances = self.build_centroid_steering()
        keyword_steering = self.build_keyword_weighted_steering()
        residual_steering = self.build_residual_steering()
        
        results = {
            "centroid": centroid_steering,
            "keyword_weighted": keyword_steering,
            "residual": residual_steering,
            "distances": centroid_distances
        }
        
        LOGGER.info("Built all steering variants successfully")
        
        return results
    
    def _get_fallback_steering(self, pair_idx: int) -> torch.Tensor:
        """Get fallback steering for pairs with no keywords.
        
        Parameters
        ----------
        pair_idx : int
            Index of the pair
        
        Returns
        -------
        torch.Tensor
            Fallback steering vector [dim]
        """
        if self.fallback_strategy == "centroid":
            # Use centroid steering
            question_idx = self.pair_indices[pair_idx, 0].item()
            cluster_id = self.pair_cluster_ids[pair_idx].item()
            
            question_emb = self.question_embeddings[question_idx]
            centroid_emb = self.centroid_embeddings[cluster_id]
            
            steering = centroid_emb - question_emb
            norm = torch.norm(steering)
            if norm > 1e-8:
                steering = steering / norm
            else:
                steering = torch.zeros_like(steering)
            
            return steering
        
        elif self.fallback_strategy == "zero":
            return torch.zeros(self.embedding_dim, dtype=torch.float32)
        
        elif self.fallback_strategy == "random":
            # Generate random unit vector
            random_vec = torch.randn(self.embedding_dim, dtype=torch.float32)
            return random_vec / torch.norm(random_vec)
        
        else:
            raise ValueError(f"Unknown fallback_strategy: {self.fallback_strategy}")
    
    def _validate_steering_vectors(
        self,
        steering: torch.Tensor,
        variant: str,
        allow_zero: bool = False
    ):
        """Validate steering vector properties.
        
        Parameters
        ----------
        steering : torch.Tensor
            Steering vectors to validate
        variant : str
            Variant name (for error messages)
        allow_zero : bool, optional
            Whether to allow zero vectors (default: False)
        """
        # Check for NaN or Inf
        if torch.isnan(steering).any():
            n_nan = torch.isnan(steering).sum().item()
            raise ValueError(
                f"{variant} steering contains {n_nan} NaN values"
            )
        
        if torch.isinf(steering).any():
            n_inf = torch.isinf(steering).sum().item()
            raise ValueError(
                f"{variant} steering contains {n_inf} Inf values"
            )
        
        # Check normalization (for non-residual or normalized residual)
        if not allow_zero or self.normalize_residual:
            norms = torch.norm(steering, dim=1)
            
            # Check for zero vectors
            zero_mask = norms < 1e-8
            if zero_mask.any() and not allow_zero:
                n_zero = zero_mask.sum().item()
                LOGGER.warning(
                    f"{variant} steering contains {n_zero} zero vectors "
                    f"({100 * n_zero / self.n_pairs:.2f}%)"
                )
            
            # Check unit norm for non-zero vectors
            non_zero_norms = norms[~zero_mask]
            if len(non_zero_norms) > 0:
                max_deviation = (non_zero_norms - 1.0).abs().max().item()
                if max_deviation > 1e-4:
                    LOGGER.warning(
                        f"{variant} steering vectors not unit normalized: "
                        f"max deviation={max_deviation:.6f}"
                    )
    
    def _validate_distances(self, distances: torch.Tensor):
        """Validate centroid distances.
        
        Parameters
        ----------
        distances : torch.Tensor
            Distances to validate
        """
        # Check for NaN or Inf
        if torch.isnan(distances).any():
            n_nan = torch.isnan(distances).sum().item()
            raise ValueError(f"Distances contain {n_nan} NaN values")
        
        if torch.isinf(distances).any():
            n_inf = torch.isinf(distances).sum().item()
            raise ValueError(f"Distances contain {n_inf} Inf values")
        
        # Check range [0, 2] for cosine distance
        min_dist = distances.min().item()
        max_dist = distances.max().item()
        
        if min_dist < -1e-6:
            LOGGER.warning(f"Negative distance detected: {min_dist:.6f}")
        
        if max_dist > 2.0 + 1e-6:
            LOGGER.warning(f"Distance > 2.0 detected: {max_dist:.6f}")
    
    def save(
        self,
        output_dir: Union[str, Path],
        steering_results: Optional[Dict[str, torch.Tensor]] = None
    ):
        """Save steering tensors to files.
        
        Parameters
        ----------
        output_dir : str or Path
            Output directory for tensor files
        steering_results : Dict[str, torch.Tensor], optional
            Pre-computed steering results (if None, will compute)
        
        Examples
        --------
        >>> builder.save("output_dir/")
        >>> # Or save pre-computed results
        >>> results = builder.build_all_steering()
        >>> builder.save("output_dir/", steering_results=results)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if steering_results is None:
            steering_results = self.build_all_steering()
        
        # Save steering variants
        torch.save(
            steering_results["centroid"],
            output_dir / "steering_centroid.pt"
        )
        torch.save(
            steering_results["keyword_weighted"],
            output_dir / "steering_keyword_weighted.pt"
        )
        torch.save(
            steering_results["residual"],
            output_dir / "steering_residual.pt"
        )
        
        # Save distances
        torch.save(
            steering_results["distances"],
            output_dir / "centroid_distances.pt"
        )
        
        LOGGER.info(f"Saved steering tensors to {output_dir}")


def build_steering(
    question_embeddings: torch.Tensor,
    keyword_embeddings: torch.Tensor,
    centroid_embeddings: torch.Tensor,
    pair_indices: torch.Tensor,
    pair_cluster_ids: torch.Tensor,
    pair_keyword_ids: List[List[int]],
    output_dir: Union[str, Path],
    normalize_residual: bool = False,
    fallback_strategy: str = "centroid",
    show_progress: bool = True
) -> Dict[str, torch.Tensor]:
    """Build and save steering signals (convenience function).
    
    Parameters
    ----------
    question_embeddings : torch.Tensor
        Question embeddings [n_questions, dim]
    keyword_embeddings : torch.Tensor
        Keyword embeddings [n_keywords, dim]
    centroid_embeddings : torch.Tensor
        Cluster centroid embeddings [n_clusters, dim]
    pair_indices : torch.Tensor
        Pair indices [n_pairs, 2]
    pair_cluster_ids : torch.Tensor
        Cluster IDs per pair [n_pairs]
    pair_keyword_ids : List[List[int]]
        Keyword IDs per pair
    output_dir : str or Path
        Output directory for saved tensors
    normalize_residual : bool, optional
        Normalize residual vectors (default: False)
    fallback_strategy : str, optional
        Keyword fallback strategy (default: "centroid")
    show_progress : bool, optional
        Show progress bars (default: True)
    
    Returns
    -------
    Dict[str, torch.Tensor]
        Dictionary with steering variants and distances
    
    Examples
    --------
    >>> from RAG_supporters.dataset import build_steering
    >>> results = build_steering(
    ...     question_embeddings=q_embs,
    ...     keyword_embeddings=k_embs,
    ...     centroid_embeddings=c_embs,
    ...     pair_indices=indices,
    ...     pair_cluster_ids=cluster_ids,
    ...     pair_keyword_ids=keyword_ids,
    ...     output_dir="dataset/"
    ... )
    """
    builder = SteeringBuilder(
        question_embeddings=question_embeddings,
        keyword_embeddings=keyword_embeddings,
        centroid_embeddings=centroid_embeddings,
        pair_indices=pair_indices,
        pair_cluster_ids=pair_cluster_ids,
        pair_keyword_ids=pair_keyword_ids,
        normalize_residual=normalize_residual,
        fallback_strategy=fallback_strategy,
        show_progress=show_progress
    )
    
    results = builder.build_all_steering()
    builder.save(output_dir, steering_results=results)
    
    return results
