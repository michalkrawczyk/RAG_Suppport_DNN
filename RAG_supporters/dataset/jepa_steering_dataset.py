"""
PyTorch Dataset for JEPA Steering with pre-computed embeddings and hard negatives.

This module provides the JEPASteeringDataset class that serves pre-computed embedding
triplets (question, steering, target_source) with hard negatives and subspace labels.
All embeddings are preloaded for zero I/O during training.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

LOGGER = logging.getLogger(__name__)


class JEPASteeringDataset(Dataset):
    """
    PyTorch Dataset for JEPA steering with pre-computed embeddings.

    Serves (question, steering, target_source) triplets with hard negatives.
    All data is preloaded from disk during initialization for zero I/O in __getitem__.

    Parameters
    ----------
    dataset_dir : str or Path
        Directory containing all dataset files (config.json, *.pt tensors, etc.)
    split : Literal["train", "val", "test"]
        Which split to use
    epoch : int, optional
        Initial epoch number for curriculum learning, by default 0

    Attributes
    ----------
    config : Dict[str, Any]
        Dataset configuration loaded from config.json
    embedding_dim : int
        Dimensionality of all embeddings
    n_neg : int
        Number of hard negatives per sample
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        split: Literal["train", "val", "test"],
        epoch: int = 0,
    ):
        """Initialize dataset and preload all tensors."""
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.current_epoch = epoch

        if not self.dataset_dir.exists():
            raise ValueError(f"Dataset directory not found: {self.dataset_dir}")

        # Load configuration
        config_path = self.dataset_dir / "config.json"
        if not config_path.exists():
            raise ValueError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            self.config = json.load(f)

        LOGGER.info(f"Loading {split} split from {self.dataset_dir}")

        # Store key dimensions
        self.embedding_dim = self.config["embedding_dim"]
        self.n_neg = self.config["n_neg"]

        # Load split indices
        split_file = self.dataset_dir / f"{split}_idx.pt"
        if not split_file.exists():
            raise ValueError(f"Split file not found: {split_file}")
        self.split_indices = torch.load(split_file, weights_only=True)

        LOGGER.info(f"Split size: {len(self.split_indices)}")

        # Preload all embeddings (zero I/O during training)
        self._load_embeddings()

        # Preload pair-level data
        self._load_pair_data()

        # Preload hard negatives
        self._load_negatives()

        # Preload steering tensors
        self._load_steering()

        # Initialize steering curriculum
        self.steering_probs = self._compute_steering_probs(epoch)
        self.rng = np.random.default_rng(seed=42 + epoch)  # Deterministic per epoch

        # Steering override (for validation)
        self._forced_steering: Optional[str] = None

        LOGGER.info(
            f"Dataset initialized: {len(self)} samples, "
            f"embedding_dim={self.embedding_dim}, n_neg={self.n_neg}"
        )

    def _load_embeddings(self):
        """Load all embedding tensors."""
        LOGGER.info("Loading embeddings...")

        self.question_embs = torch.load(
            self.dataset_dir / "question_embs.pt", weights_only=True
        )
        self.source_embs = torch.load(self.dataset_dir / "source_embs.pt", weights_only=True)
        self.keyword_embs = torch.load(
            self.dataset_dir / "keyword_embs.pt", weights_only=True
        )
        self.centroid_embs = torch.load(
            self.dataset_dir / "centroid_embs.pt", weights_only=True
        )

        # Validate dimensions
        assert (
            self.question_embs.size(1) == self.embedding_dim
        ), "Question embedding dimension mismatch"
        assert (
            self.source_embs.size(1) == self.embedding_dim
        ), "Source embedding dimension mismatch"
        assert (
            self.keyword_embs.size(1) == self.embedding_dim
        ), "Keyword embedding dimension mismatch"
        assert (
            self.centroid_embs.size(1) == self.embedding_dim
        ), "Centroid embedding dimension mismatch"

        LOGGER.info(
            f"Loaded embeddings: {len(self.question_embs)} questions, "
            f"{len(self.source_embs)} sources, {len(self.keyword_embs)} keywords, "
            f"{len(self.centroid_embs)} centroids"
        )

    def _load_pair_data(self):
        """Load pair-level data."""
        LOGGER.info("Loading pair data...")

        self.pair_index = torch.load(self.dataset_dir / "pair_index.pt", weights_only=True)
        self.pair_cluster_id = torch.load(
            self.dataset_dir / "pair_cluster_id.pt", weights_only=True
        )
        self.pair_relevance = torch.load(
            self.dataset_dir / "pair_relevance.pt", weights_only=True
        )

        # Load pair_keyword_ids (stored as PT file with list-of-lists)
        pair_keyword_ids_path = self.dataset_dir / "pair_keyword_ids.pt"
        self.pair_keyword_ids = torch.load(pair_keyword_ids_path, weights_only=False)

        LOGGER.info(f"Loaded data for {len(self.pair_index)} pairs")

    def _load_negatives(self):
        """Load hard negative data."""
        LOGGER.info("Loading hard negatives...")

        self.hard_negatives = torch.load(
            self.dataset_dir / "hard_negatives.pt", weights_only=True
        )
        self.negative_tiers = torch.load(
            self.dataset_dir / "negative_tiers.pt", weights_only=True
        )

        assert self.hard_negatives.size(1) == self.n_neg, "Hard negatives count mismatch"

        LOGGER.info(f"Loaded {self.n_neg} hard negatives per pair")

    def _load_steering(self):
        """Load steering tensors."""
        LOGGER.info("Loading steering tensors...")

        self.steering_centroid = torch.load(
            self.dataset_dir / "steering_centroid.pt", weights_only=True
        )
        self.steering_keyword_weighted = torch.load(
            self.dataset_dir / "steering_keyword_weighted.pt", weights_only=True
        )
        self.steering_residual = torch.load(
            self.dataset_dir / "steering_residual.pt", weights_only=True
        )
        self.centroid_distances = torch.load(
            self.dataset_dir / "centroid_distances.pt", weights_only=True
        )

        LOGGER.info("Steering tensors loaded")

    def _compute_steering_probs(self, epoch: int) -> Dict[str, float]:
        """
        Compute steering variant probabilities based on curriculum.

        Parameters
        ----------
        epoch : int
            Current training epoch

        Returns
        -------
        Dict[str, float]
            Probability for each steering variant (zero, centroid, keyword, residual)
        """
        curriculum = self.config.get("curriculum", {})

        if not curriculum:
            # Default: use config probabilities
            return self.config["steering_probabilities"]

        # Curriculum learning: gradually shift from zero to complex steering
        zero_start = curriculum.get("zero_prob_start", 0.5)
        zero_end = curriculum.get("zero_prob_end", 0.1)
        epochs_total = curriculum.get("epochs_total", 100)

        # Linear decay for zero steering
        progress = min(epoch / max(epochs_total, 1), 1.0)
        zero_prob = zero_start + (zero_end - zero_start) * progress

        # Distribute remaining probability among other variants
        remaining = 1.0 - zero_prob
        base_probs = self.config["steering_probabilities"]
        total_non_zero = 1.0 - base_probs.get("zero", 0.0)

        if total_non_zero > 0:
            scale = remaining / total_non_zero
            probs = {
                "zero": zero_prob,
                "centroid": base_probs.get("centroid", 0.3) * scale,
                "keyword": base_probs.get("keyword", 0.3) * scale,
                "residual": base_probs.get("residual", 0.3) * scale,
            }
        else:
            probs = {"zero": zero_prob, "centroid": remaining, "keyword": 0.0, "residual": 0.0}

        return probs

    def __len__(self) -> int:
        """Return number of samples in current split."""
        return len(self.split_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Parameters
        ----------
        idx : int
            Index within current split

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary containing:
            - question_emb: [D] tensor
            - target_source_emb: [D] tensor
            - steering: [D] tensor (selected variant or zeros)
            - negative_embs: [N_neg, D] tensor
            - cluster_id: scalar tensor
            - relevance: scalar tensor
            - centroid_distance: scalar tensor
            - steering_variant: scalar tensor (0=zero, 1=centroid, 2=keyword, 3=residual)
            - negative_tiers: [N_neg] tensor
        """
        # Resolve pair index
        pair_idx = self.split_indices[idx].item()

        # Get question and source indices
        q_idx = self.pair_index[pair_idx, 0].item()
        s_idx = self.pair_index[pair_idx, 1].item()

        # Get embeddings
        question_emb = self.question_embs[q_idx]
        target_source_emb = self.source_embs[s_idx]

        # Select steering variant
        if self._forced_steering is not None:
            variant = self._forced_steering
        else:
            variant = self.rng.choice(
                list(self.steering_probs.keys()), p=list(self.steering_probs.values())
            )

        # Get steering tensor
        if variant == "zero":
            steering = torch.zeros(self.embedding_dim, dtype=question_emb.dtype)
            variant_id = 0
        elif variant == "centroid":
            steering = self.steering_centroid[pair_idx]
            variant_id = 1
        elif variant == "keyword":
            steering = self.steering_keyword_weighted[pair_idx]
            variant_id = 2
        elif variant == "residual":
            steering = self.steering_residual[pair_idx]
            variant_id = 3
        else:
            raise ValueError(f"Unknown steering variant: {variant}")

        # Get hard negatives
        neg_indices = self.hard_negatives[pair_idx]
        negative_embs = self.source_embs[neg_indices]

        # Get metadata
        cluster_id = self.pair_cluster_id[pair_idx]
        relevance = self.pair_relevance[pair_idx]
        centroid_distance = self.centroid_distances[pair_idx]
        neg_tiers = self.negative_tiers[pair_idx]

        return {
            "question_emb": question_emb,
            "target_source_emb": target_source_emb,
            "steering": steering,
            "negative_embs": negative_embs,
            "cluster_id": cluster_id,
            "relevance": relevance,
            "centroid_distance": centroid_distance,
            "steering_variant": torch.tensor(variant_id, dtype=torch.long),
            "negative_tiers": neg_tiers,
        }

    def set_epoch(self, epoch: int):
        """
        Update epoch for curriculum learning and reseed RNG.

        Parameters
        ----------
        epoch : int
            New epoch number
        """
        self.current_epoch = epoch
        self.steering_probs = self._compute_steering_probs(epoch)
        self.rng = np.random.default_rng(seed=42 + epoch)

        LOGGER.info(
            f"Set epoch {epoch}, steering probs: "
            f"{', '.join(f'{k}={v:.2f}' for k, v in self.steering_probs.items())}"
        )

    def reload_negatives(self):
        """
        Hot-reload hard negatives from disk.

        Useful for periodic negative refreshing during training without restarting.
        """
        LOGGER.info("Reloading hard negatives from disk...")

        self.hard_negatives = torch.load(
            self.dataset_dir / "hard_negatives.pt", weights_only=True
        )
        self.negative_tiers = torch.load(
            self.dataset_dir / "negative_tiers.pt", weights_only=True
        )

        # Verify shape consistency
        assert (
            self.hard_negatives.size(1) == self.n_neg
        ), "Reloaded negatives have inconsistent count"

        LOGGER.info("Hard negatives reloaded successfully")

    def force_steering(self, variant: Optional[Literal["zero", "centroid", "keyword", "residual"]]):
        """
        Force a specific steering variant (disables stochastic selection).

        Parameters
        ----------
        variant : Optional[Literal["zero", "centroid", "keyword", "residual"]]
            Steering variant to force, or None to restore stochastic selection
        """
        if variant is not None and variant not in ["zero", "centroid", "keyword", "residual"]:
            raise ValueError(
                f"Invalid steering variant: {variant}. "
                f"Must be one of: zero, centroid, keyword, residual"
            )

        self._forced_steering = variant

        if variant is None:
            LOGGER.info("Restored stochastic steering selection")
        else:
            LOGGER.info(f"Forced steering variant: {variant}")
