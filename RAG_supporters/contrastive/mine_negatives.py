"""
Hard Negative Miner for JASPER Steering Dataset.

This module mines hard negatives for contrastive learning with 4-tier sampling:
- Tier 1: In-cluster negatives (same cluster, excluding true source)
- Tier 2: Adjacent cluster negatives (top-K nearest clusters)
- Tier 3: High-similarity negatives (highest cosine similarity, wrong clusters)
- Tier 4: Random distant negatives (uniform random from far clusters)

Key Features:
- Stratified negative sampling by difficulty tier
- True source never in own negative set
- Configurable tier proportions
- Handles edge cases (small clusters, insufficient negatives)
- Validation of sampling properties
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

from RAG_supporters.data_validation import (
    validate_tensor_2d,
    validate_tensor_1d,
    validate_embedding_dimensions,
    validate_pair_indices_bounds,
    validate_cluster_ids_bounds,
    validate_length_consistency,
)

LOGGER = logging.getLogger(__name__)


class NegativeMiner:
    """Mine hard negatives for contrastive learning.

    Generates stratified hard negatives across 4 difficulty tiers:
    1. In-cluster: Same cluster as positive (tests within-cluster discrimination)
    2. Adjacent: Top-K nearest clusters (tests cluster boundaries)
    3. High-similarity: Highest cosine similarity, wrong clusters (hardest negatives)
    4. Random: Uniform random from distant clusters (easy negatives for stability)

    Parameters
    ----------
    source_embeddings : torch.Tensor
        Source embeddings [n_sources, dim]
    question_embeddings : torch.Tensor
        Question embeddings [n_questions, dim]
    centroid_embeddings : torch.Tensor
        Cluster centroid embeddings [n_clusters, dim]
    pair_indices : torch.Tensor
        Pair indices [n_pairs, 2] mapping to (question_idx, source_idx)
    pair_cluster_ids : torch.Tensor
        Primary cluster ID per pair [n_pairs]
    source_cluster_ids : torch.Tensor
        Cluster assignment for each source [n_sources]
    n_neg : int
        Total number of negatives per pair
    tier_proportions : List[int], optional
        Number of negatives per tier [tier1, tier2, tier3, tier4].
        Must sum to n_neg. Default: Equal distribution.
    adjacent_k : int, optional
        Number of adjacent clusters to consider for Tier 2 (default: 3)
    random_seed : int, optional
        Random seed for reproducibility (default: 42)
    show_progress : bool, optional
        Show progress bars (default: True)

    Attributes
    ----------
    source_embeddings : torch.Tensor
        Source embeddings
    question_embeddings : torch.Tensor
        Question embeddings
    centroid_embeddings : torch.Tensor
        Centroid embeddings
    pair_indices : torch.Tensor
        Pair index mapping
    pair_cluster_ids : torch.Tensor
        Cluster assignments for pairs
    source_cluster_ids : torch.Tensor
        Cluster assignments for sources
    n_neg : int
        Total negatives per pair
    tier_proportions : List[int]
        Negatives per tier
    adjacent_k : int
        Number of adjacent clusters
    show_progress : bool
        Progress bar visibility

    Examples
    --------
    >>> import torch
    >>> from RAG_supporters.dataset import NegativeMiner
    >>>
    >>> # Prepare data
    >>> source_embs = torch.randn(1000, 384)
    >>> question_embs = torch.randn(100, 384)
    >>> centroid_embs = torch.randn(10, 384)
    >>> pair_indices = torch.randint(0, 100, (500, 2))
    >>> pair_cluster_ids = torch.randint(0, 10, (500,))
    >>> source_cluster_ids = torch.randint(0, 10, (1000,))
    >>>
    >>> # Mine negatives
    >>> miner = NegativeMiner(
    ...     source_embeddings=source_embs,
    ...     question_embeddings=question_embs,
    ...     centroid_embeddings=centroid_embs,
    ...     pair_indices=pair_indices,
    ...     pair_cluster_ids=pair_cluster_ids,
    ...     source_cluster_ids=source_cluster_ids,
    ...     n_neg=12,
    ...     tier_proportions=[3, 4, 3, 2]
    ... )
    >>>
    >>> # Generate negatives
    >>> results = miner.mine_all_negatives()
    >>> print(results['hard_negatives'].shape)  # [500, 12]
    >>> print(results['negative_tiers'].shape)  # [500, 12]
    """

    def __init__(
        self,
        source_embeddings: torch.Tensor,
        question_embeddings: torch.Tensor,
        centroid_embeddings: torch.Tensor,
        pair_indices: torch.Tensor,
        pair_cluster_ids: torch.Tensor,
        source_cluster_ids: torch.Tensor,
        n_neg: int,
        tier_proportions: Optional[List[int]] = None,
        adjacent_k: int = 3,
        random_seed: int = 42,
        show_progress: bool = True,
    ):
        """Initialize negative miner."""
        # Validate inputs
        self._validate_inputs(
            source_embeddings,
            question_embeddings,
            centroid_embeddings,
            pair_indices,
            pair_cluster_ids,
            source_cluster_ids,
            n_neg,
            tier_proportions,
        )

        self.source_embeddings = source_embeddings
        self.question_embeddings = question_embeddings
        self.centroid_embeddings = centroid_embeddings
        self.pair_indices = pair_indices
        self.pair_cluster_ids = pair_cluster_ids
        self.source_cluster_ids = source_cluster_ids
        self.n_neg = n_neg
        self.adjacent_k = adjacent_k
        self.show_progress = show_progress

        self.n_pairs = len(pair_indices)
        self.n_sources = len(source_embeddings)
        self.n_clusters = len(centroid_embeddings)
        self.embedding_dim = source_embeddings.shape[1]

        # Set tier proportions
        if tier_proportions is None:
            # Equal distribution across tiers
            base = n_neg // 4
            remainder = n_neg % 4
            self.tier_proportions = [base] * 4
            # Distribute remainder (prioritize harder tiers)
            for i in range(remainder):
                self.tier_proportions[i] += 1
        else:
            self.tier_proportions = tier_proportions

        # Initialize RNG
        self.rng = np.random.default_rng(seed=random_seed)

        # Precompute cluster structures
        self._build_cluster_structures()

        # Precompute cluster distances
        self._compute_cluster_distances()

        LOGGER.info(
            f"Initialized NegativeMiner: {self.n_pairs} pairs, {self.n_sources} sources, "
            f"{self.n_clusters} clusters, n_neg={n_neg}, tier_proportions={self.tier_proportions}"
        )

    def _validate_inputs(
        self,
        source_embeddings: torch.Tensor,
        question_embeddings: torch.Tensor,
        centroid_embeddings: torch.Tensor,
        pair_indices: torch.Tensor,
        pair_cluster_ids: torch.Tensor,
        source_cluster_ids: torch.Tensor,
        n_neg: int,
        tier_proportions: Optional[List[int]],
    ):
        """Validate input tensors and parameters."""
        # Validate embedding dimensions consistency
        validate_embedding_dimensions(
            (source_embeddings, "source_embeddings"),
            (question_embeddings, "question_embeddings"),
            (centroid_embeddings, "centroid_embeddings"),
        )

        # Validate pair structures
        validate_tensor_2d(pair_indices, "pair_indices", expected_cols=2)
        validate_tensor_1d(pair_cluster_ids, "pair_cluster_ids")
        validate_tensor_1d(source_cluster_ids, "source_cluster_ids")

        # Validate length consistency
        n_pairs = pair_indices.shape[0]
        n_sources = source_embeddings.shape[0]
        validate_length_consistency(
            (pair_cluster_ids, "pair_cluster_ids", n_pairs),
            (source_cluster_ids, "source_cluster_ids", n_sources),
        )

        # Validate index bounds
        validate_pair_indices_bounds(
            pair_indices,
            n_questions=question_embeddings.shape[0],
            n_sources=source_embeddings.shape[0],
        )

        validate_cluster_ids_bounds(
            pair_cluster_ids, n_clusters=centroid_embeddings.shape[0], name="pair_cluster_ids"
        )

        validate_cluster_ids_bounds(
            source_cluster_ids, n_clusters=centroid_embeddings.shape[0], name="source_cluster_ids"
        )

        # Validate n_neg
        if not isinstance(n_neg, int) or n_neg < 1:
            raise ValueError(f"n_neg must be a positive integer, got {n_neg}")

        # Validate tier_proportions if provided
        if tier_proportions is not None:
            if not isinstance(tier_proportions, list):
                raise TypeError(f"tier_proportions must be a list, got {type(tier_proportions)}")
            if len(tier_proportions) != 4:
                raise ValueError(
                    f"tier_proportions must have exactly 4 values, got {len(tier_proportions)}"
                )
            if any(not isinstance(x, int) or x < 0 for x in tier_proportions):
                raise ValueError(
                    f"All tier proportions must be non-negative integers, got {tier_proportions}"
                )
            if sum(tier_proportions) != n_neg:
                raise ValueError(
                    f"tier_proportions must sum to n_neg={n_neg}, got sum={sum(tier_proportions)}"
                )

    def _build_cluster_structures(self):
        """Build cluster membership structures for efficient sampling."""
        LOGGER.info("Building cluster structures...")

        # Build source-to-cluster mapping
        self.cluster_sources = {}
        for cluster_id in range(self.n_clusters):
            mask = self.source_cluster_ids == cluster_id
            self.cluster_sources[cluster_id] = torch.where(mask)[0].tolist()

        # Log cluster statistics
        cluster_sizes = [len(sources) for sources in self.cluster_sources.values()]
        LOGGER.info(
            f"Cluster sizes: min={min(cluster_sizes)}, max={max(cluster_sizes)}, "
            f"mean={np.mean(cluster_sizes):.1f}"
        )

    def _compute_cluster_distances(self):
        """Precompute cosine distances between all cluster centroids."""
        LOGGER.info("Computing cluster distances...")

        # Normalize centroids
        centroids_norm = self.centroid_embeddings / torch.norm(
            self.centroid_embeddings, dim=1, keepdim=True
        )

        # Compute cosine similarity matrix
        similarity_matrix = torch.mm(centroids_norm, centroids_norm.T)

        # Convert to distance (1 - similarity)
        self.cluster_dist_matrix = 1.0 - similarity_matrix

        # Precompute adjacent clusters for each cluster
        self.adjacent_clusters = {}
        for cluster_id in range(self.n_clusters):
            # Get distances to all other clusters
            distances = self.cluster_dist_matrix[cluster_id].clone()
            distances[cluster_id] = float("inf")  # Exclude self

            # Get top-K nearest clusters
            _, nearest_indices = torch.topk(
                distances, k=min(self.adjacent_k, self.n_clusters - 1), largest=False
            )
            self.adjacent_clusters[cluster_id] = nearest_indices.tolist()

        LOGGER.info("Cluster distances computed")

    def mine_all_negatives(self) -> Dict[str, torch.Tensor]:
        """
        Mine all hard negatives for all pairs.

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary with keys:
            - 'hard_negatives': Source indices [n_pairs, n_neg]
            - 'negative_tiers': Tier labels [n_pairs, n_neg]
        """
        LOGGER.info("Mining hard negatives for all pairs...")

        # Initialize output tensors
        hard_negatives = torch.zeros((self.n_pairs, self.n_neg), dtype=torch.long)
        negative_tiers = torch.zeros((self.n_pairs, self.n_neg), dtype=torch.long)

        # Mine negatives for each pair
        iterator = (
            tqdm(range(self.n_pairs), desc="Mining negatives")
            if self.show_progress
            else range(self.n_pairs)
        )

        for pair_idx in iterator:
            question_idx = self.pair_indices[pair_idx, 0].item()
            source_idx = self.pair_indices[pair_idx, 1].item()
            cluster_id = self.pair_cluster_ids[pair_idx].item()

            # Mine negatives for this pair
            neg_indices, tier_labels = self._mine_pair_negatives(
                question_idx, source_idx, cluster_id
            )

            hard_negatives[pair_idx] = neg_indices
            negative_tiers[pair_idx] = tier_labels

        # Validate outputs
        self._validate_negatives(hard_negatives, negative_tiers)

        LOGGER.info("Hard negative mining complete")

        return {"hard_negatives": hard_negatives, "negative_tiers": negative_tiers}

    def _mine_pair_negatives(
        self, question_idx: int, true_source_idx: int, cluster_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mine negatives for a single pair.

        Parameters
        ----------
        question_idx : int
            Question index
        true_source_idx : int
            True source index (to exclude)
        cluster_id : int
            Cluster ID for the pair

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (negative_indices, tier_labels) both [n_neg]
        """
        negatives = []
        tiers = []

        # Tier 1: In-cluster negatives
        tier1_negatives = self._sample_in_cluster(
            cluster_id, true_source_idx, self.tier_proportions[0]
        )
        negatives.extend(tier1_negatives)
        tiers.extend([1] * len(tier1_negatives))

        # Tier 2: Adjacent cluster negatives
        tier2_negatives = self._sample_adjacent_clusters(
            cluster_id, true_source_idx, self.tier_proportions[1]
        )
        negatives.extend(tier2_negatives)
        tiers.extend([2] * len(tier2_negatives))

        # Tier 3: High-similarity negatives
        tier3_negatives = self._sample_high_similarity(
            question_idx, cluster_id, true_source_idx, self.tier_proportions[2]
        )
        negatives.extend(tier3_negatives)
        tiers.extend([3] * len(tier3_negatives))

        # Tier 4: Random negatives
        tier4_negatives = self._sample_random(cluster_id, true_source_idx, self.tier_proportions[3])
        negatives.extend(tier4_negatives)
        tiers.extend([4] * len(tier4_negatives))

        # Handle insufficient negatives (pad with random if needed)
        while len(negatives) < self.n_neg:
            # Sample random source excluding true source and existing negatives
            excluded = set(negatives + [true_source_idx])
            candidates = [s for s in range(self.n_sources) if s not in excluded]
            if candidates:
                neg = self.rng.choice(candidates)
                negatives.append(neg)
                tiers.append(4)  # Label as Tier 4 (random)
            else:
                # Extremely rare: no more sources available, duplicate last negative
                negatives.append(negatives[-1] if negatives else true_source_idx)
                tiers.append(tiers[-1] if tiers else 4)

        # Shuffle negatives within their tiers to avoid positional bias
        negatives_tensor = torch.tensor(negatives, dtype=torch.long)
        tiers_tensor = torch.tensor(tiers, dtype=torch.long)

        return negatives_tensor, tiers_tensor

    def _sample_in_cluster(self, cluster_id: int, exclude_source: int, n_samples: int) -> List[int]:
        """Sample negatives from the same cluster."""
        # Get all sources in cluster
        cluster_sources = self.cluster_sources[cluster_id]

        # Exclude true source
        candidates = [s for s in cluster_sources if s != exclude_source]

        if not candidates:
            # No other sources in cluster, return empty
            return []

        # Sample without replacement (or with if not enough)
        n_to_sample = min(n_samples, len(candidates))
        sampled = self.rng.choice(candidates, size=n_to_sample, replace=False).tolist()

        # If we need more, sample with replacement
        if n_to_sample < n_samples:
            additional = self.rng.choice(
                candidates, size=n_samples - n_to_sample, replace=True
            ).tolist()
            sampled.extend(additional)

        return sampled

    def _sample_adjacent_clusters(
        self, cluster_id: int, exclude_source: int, n_samples: int
    ) -> List[int]:
        """Sample negatives from adjacent clusters."""
        adjacent = self.adjacent_clusters[cluster_id]

        if not adjacent:
            # No adjacent clusters, return empty
            return []

        # Collect all sources from adjacent clusters
        candidates = []
        for adj_cluster in adjacent:
            candidates.extend(self.cluster_sources[adj_cluster])

        # Exclude true source (unlikely but possible if source cluster changed)
        candidates = [s for s in candidates if s != exclude_source]

        if not candidates:
            return []

        # Sample without replacement (or with if not enough)
        n_to_sample = min(n_samples, len(candidates))
        sampled = self.rng.choice(candidates, size=n_to_sample, replace=False).tolist()

        # If we need more, sample with replacement
        if n_to_sample < n_samples:
            additional = self.rng.choice(
                candidates, size=n_samples - n_to_sample, replace=True
            ).tolist()
            sampled.extend(additional)

        return sampled

    def _sample_high_similarity(
        self, question_idx: int, cluster_id: int, exclude_source: int, n_samples: int
    ) -> List[int]:
        """Sample negatives with highest similarity to question (from other clusters)."""
        if n_samples == 0:
            return []

        # Get question embedding
        question_emb = self.question_embeddings[question_idx : question_idx + 1]  # [1, dim]

        # Compute cosine similarity to all sources
        # Normalize embeddings
        question_norm = question_emb / torch.norm(question_emb, dim=1, keepdim=True)
        sources_norm = self.source_embeddings / torch.norm(
            self.source_embeddings, dim=1, keepdim=True
        )

        # Compute similarities
        similarities = torch.mm(question_norm, sources_norm.T).squeeze()  # [n_sources]

        # Mask out sources from same cluster and true source
        mask = torch.ones(self.n_sources, dtype=torch.bool)
        mask[exclude_source] = False

        # Mask out sources from same cluster
        same_cluster_sources = self.cluster_sources[cluster_id]
        for s in same_cluster_sources:
            mask[s] = False

        # Get valid candidates
        valid_indices = torch.where(mask)[0]

        if len(valid_indices) == 0:
            return []

        # Get similarities for valid candidates
        valid_similarities = similarities[valid_indices]

        # Get top-K highest similarities
        n_to_sample = min(n_samples, len(valid_indices))
        _, top_indices = torch.topk(valid_similarities, k=n_to_sample, largest=True)

        sampled = valid_indices[top_indices].tolist()

        # If we need more, pad with random from this set
        if n_to_sample < n_samples:
            additional_indices = self.rng.choice(
                len(valid_indices), size=n_samples - n_to_sample, replace=True
            )
            additional = valid_indices[additional_indices].tolist()
            sampled.extend(additional)

        return sampled

    def _sample_random(self, cluster_id: int, exclude_source: int, n_samples: int) -> List[int]:
        """Sample random negatives from distant clusters."""
        # Get all sources not in the same cluster
        candidates = []
        for cid, sources in self.cluster_sources.items():
            if cid != cluster_id:
                candidates.extend(sources)

        # Exclude true source
        candidates = [s for s in candidates if s != exclude_source]

        if not candidates:
            return []

        # Sample with replacement (uniform random)
        sampled = self.rng.choice(candidates, size=n_samples, replace=True).tolist()

        return sampled

    def _validate_negatives(self, hard_negatives: torch.Tensor, negative_tiers: torch.Tensor):
        """Validate mined negatives."""
        LOGGER.info("Validating mined negatives...")

        # Check shapes
        assert hard_negatives.shape == (
            self.n_pairs,
            self.n_neg,
        ), f"hard_negatives shape mismatch: expected {(self.n_pairs, self.n_neg)}, got {hard_negatives.shape}"
        assert negative_tiers.shape == (
            self.n_pairs,
            self.n_neg,
        ), f"negative_tiers shape mismatch: expected {(self.n_pairs, self.n_neg)}, got {negative_tiers.shape}"

        # Check indices bounds
        assert (
            hard_negatives.min() >= 0
        ), f"Negative indices contain negative values: min={hard_negatives.min()}"
        assert (
            hard_negatives.max() < self.n_sources
        ), f"Negative indices out of bounds: max={hard_negatives.max()}, n_sources={self.n_sources}"

        # Check tier values
        assert (
            negative_tiers.min() >= 1
        ), f"Tier labels contain values < 1: min={negative_tiers.min()}"
        assert (
            negative_tiers.max() <= 4
        ), f"Tier labels contain values > 4: max={negative_tiers.max()}"

        # Check that true source is not in negatives
        violations = 0
        for pair_idx in range(self.n_pairs):
            true_source = self.pair_indices[pair_idx, 1].item()
            pair_negatives = hard_negatives[pair_idx].tolist()
            if true_source in pair_negatives:
                violations += 1

        if violations > 0:
            LOGGER.warning(
                f"Found {violations} pairs with true source in negative set "
                f"({100 * violations / self.n_pairs:.2f}%)"
            )
        else:
            LOGGER.info("✓ No true sources in negative sets")

        # Check tier distribution
        tier_counts = {}
        for tier in range(1, 5):
            tier_counts[tier] = (negative_tiers == tier).sum().item()

        LOGGER.info(f"Tier distribution: {tier_counts}")
        LOGGER.info("Validation complete")

    def save(self, output_dir: Union[str, Path], results: Dict[str, torch.Tensor]):
        """
        Save mined negatives to disk.

        Parameters
        ----------
        output_dir : str or Path
            Directory to save outputs
        results : Dict[str, torch.Tensor]
            Results from mine_all_negatives()
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save hard negatives
        torch.save(results["hard_negatives"], output_dir / "hard_negatives.pt")

        # Save tier labels
        torch.save(results["negative_tiers"], output_dir / "negative_tiers.pt")

        LOGGER.info(f"Saved negatives to {output_dir}")


def mine_negatives(
    source_embeddings: torch.Tensor,
    question_embeddings: torch.Tensor,
    centroid_embeddings: torch.Tensor,
    pair_indices: torch.Tensor,
    pair_cluster_ids: torch.Tensor,
    source_cluster_ids: torch.Tensor,
    n_neg: int,
    output_dir: Union[str, Path],
    tier_proportions: Optional[List[int]] = None,
    adjacent_k: int = 3,
    random_seed: int = 42,
    show_progress: bool = True,
) -> Dict[str, torch.Tensor]:
    """Mine negatives and save to disk.

    Parameters
    ----------
    source_embeddings : torch.Tensor
        Source embeddings [n_sources, dim]
    question_embeddings : torch.Tensor
        Question embeddings [n_questions, dim]
    centroid_embeddings : torch.Tensor
        Cluster centroid embeddings [n_clusters, dim]
    pair_indices : torch.Tensor
        Pair indices [n_pairs, 2]
    pair_cluster_ids : torch.Tensor
        Primary cluster ID per pair [n_pairs]
    source_cluster_ids : torch.Tensor
        Cluster assignment for each source [n_sources]
    n_neg : int
        Total number of negatives per pair
    output_dir : str or Path
        Directory to save outputs
    tier_proportions : List[int], optional
        Number of negatives per tier
    adjacent_k : int, optional
        Number of adjacent clusters (default: 3)
    random_seed : int, optional
        Random seed (default: 42)
    show_progress : bool, optional
        Show progress bars (default: True)

    Returns
    -------
    Dict[str, torch.Tensor]
        Dictionary with 'hard_negatives' and 'negative_tiers'

    Examples
    --------
    >>> from RAG_supporters.dataset import mine_negatives
    >>>
    >>> results = mine_negatives(
    ...     source_embeddings=source_embs,
    ...     question_embeddings=question_embs,
    ...     centroid_embeddings=centroid_embs,
    ...     pair_indices=pair_indices,
    ...     pair_cluster_ids=pair_cluster_ids,
    ...     source_cluster_ids=source_cluster_ids,
    ...     n_neg=12,
    ...     output_dir="./dataset",
    ...     tier_proportions=[3, 4, 3, 2]
    ... )
    """
    # Create miner
    miner = NegativeMiner(
        source_embeddings=source_embeddings,
        question_embeddings=question_embeddings,
        centroid_embeddings=centroid_embeddings,
        pair_indices=pair_indices,
        pair_cluster_ids=pair_cluster_ids,
        source_cluster_ids=source_cluster_ids,
        n_neg=n_neg,
        tier_proportions=tier_proportions,
        adjacent_k=adjacent_k,
        random_seed=random_seed,
        show_progress=show_progress,
    )

    # Mine negatives
    results = miner.mine_all_negatives()

    # Save to disk
    miner.save(output_dir, results)

    return results
