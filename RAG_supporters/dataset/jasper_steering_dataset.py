"""
PyTorch Dataset for JASPER (Joint Architecture for Subspace Prediction with Explainable Routing).

This module provides the JASPERSteeringDataset class that serves pre-computed embedding
triplets (question, steering, target_source) with hard negatives and subspace labels.
All embeddings are preloaded for zero I/O during training.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from RAG_supporters.data_validation.tensor_utils import load_tensor_artifact

LOGGER = logging.getLogger(__name__)


class JASPERSteeringDataset(Dataset):
    """
    PyTorch Dataset for JASPER steering with pre-computed embeddings.

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
    device : torch.device, optional
        Device to transfer tensors to during initialization (default: CPU)

    Attributes
    ----------
    config : Dict[str, Any]
        Dataset configuration loaded from config.json
    embedding_dim : int
        Dimensionality of all embeddings
    n_neg : int
        Number of hard negatives per sample
    device : torch.device
        Device where tensors are stored
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        split: Literal["train", "val", "test"],
        epoch: int = 0,
        device: Optional[torch.device] = None,
    ):
        """Initialize dataset and preload all tensors."""
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.current_epoch = epoch
        self.device = device if device is not None else torch.device("cpu")

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

        # Validate split file exists
        split_file = self.dataset_dir / f"{split}_idx.pt"
        if not split_file.exists():
            raise ValueError(
                f"Split file not found: {split_file}. Valid splits are: train, val, test"
            )

        # Load split indices
        self.split_indices = load_tensor_artifact(
            self.dataset_dir, f"{split}_idx.pt", expected_shape=(None,)
        )

        LOGGER.info(f"Split size: {len(self.split_indices)}")

        # Preload all embeddings (zero I/O during training)
        self._load_embeddings()

        # Preload pair-level data
        self._load_pair_data()

        # Preload hard negatives
        self._load_negatives()

        # Preload steering tensors
        self._load_steering()

        # Validate referential integrity
        self._validate_referential_integrity()

        # Move tensors to target device if requested
        if self.device != torch.device("cpu"):
            self._move_to_device()

        # Initialize steering curriculum
        self.steering_probs = self._compute_steering_probs(epoch)
        self.rng = np.random.default_rng(seed=42 + epoch)  # Deterministic per epoch

        # Steering override (for validation)
        self._forced_steering: Optional[str] = None

        # Log memory usage
        total_mb = self._compute_memory_usage()
        LOGGER.info(
            f"Dataset initialized: {len(self)} samples, "
            f"embedding_dim={self.embedding_dim}, n_neg={self.n_neg}, "
            f"device={self.device}, memory={total_mb:.1f} MB"
        )

    def _load_embeddings(self):
        """Load all embedding tensors."""
        LOGGER.info("Loading embeddings...")

        self.question_embs = load_tensor_artifact(
            self.dataset_dir, "question_embs.pt", expected_shape=(None, self.embedding_dim)
        )
        self.source_embs = load_tensor_artifact(
            self.dataset_dir, "source_embs.pt", expected_shape=(None, self.embedding_dim)
        )
        self.keyword_embs = load_tensor_artifact(
            self.dataset_dir, "keyword_embs.pt", expected_shape=(None, self.embedding_dim)
        )
        self.centroid_embs = load_tensor_artifact(
            self.dataset_dir, "centroid_embs.pt", expected_shape=(None, self.embedding_dim)
        )

        LOGGER.info(
            f"Loaded embeddings: {len(self.question_embs)} questions, "
            f"{len(self.source_embs)} sources, {len(self.keyword_embs)} keywords, "
            f"{len(self.centroid_embs)} centroids"
        )

    def _load_pair_data(self):
        """Load pair-level data."""
        LOGGER.info("Loading pair data...")

        self.pair_index = load_tensor_artifact(
            self.dataset_dir, "pair_index.pt", expected_shape=(None, 2)
        )
        self.pair_cluster_id = load_tensor_artifact(
            self.dataset_dir, "pair_cluster_id.pt", expected_shape=(None,)
        )
        self.pair_relevance = load_tensor_artifact(
            self.dataset_dir, "pair_relevance.pt", expected_shape=(None,)
        )

        # Load pair_keyword_ids (stored as PT file with list-of-lists)
        self.pair_keyword_ids = load_tensor_artifact(
            self.dataset_dir, "pair_keyword_ids.pt", weights_only=False
        )

        # Validate split indices reference valid pairs
        max_split_idx = self.split_indices.max().item()
        if max_split_idx >= len(self.pair_index):
            raise ValueError(
                f"Split indices reference pair {max_split_idx}, "
                f"but only {len(self.pair_index)} pairs exist"
            )

        LOGGER.info(f"Loaded data for {len(self.pair_index)} pairs")

    def _load_negatives(self):
        """Load hard negative data."""
        LOGGER.info("Loading hard negatives...")

        self.hard_negatives = load_tensor_artifact(
            self.dataset_dir, "hard_negatives.pt", expected_shape=(None, self.n_neg)
        )
        self.negative_tiers = load_tensor_artifact(
            self.dataset_dir, "negative_tiers.pt", expected_shape=(None, self.n_neg)
        )

        LOGGER.info(f"Loaded {self.n_neg} hard negatives per pair")

    def _load_steering(self):
        """Load steering tensors."""
        LOGGER.info("Loading steering tensors...")

        self.steering_centroid = load_tensor_artifact(
            self.dataset_dir, "steering_centroid.pt", expected_shape=(None, self.embedding_dim)
        )
        self.steering_keyword_weighted = load_tensor_artifact(
            self.dataset_dir,
            "steering_keyword_weighted.pt",
            expected_shape=(None, self.embedding_dim),
        )
        self.steering_residual = load_tensor_artifact(
            self.dataset_dir, "steering_residual.pt", expected_shape=(None, self.embedding_dim)
        )
        self.centroid_distances = load_tensor_artifact(
            self.dataset_dir, "centroid_distances.pt", expected_shape=(None,)
        )

        LOGGER.info("Steering tensors loaded")

    def _validate_referential_integrity(self):
        """Validate that all indices reference valid data."""
        LOGGER.debug("Validating referential integrity...")

        # Validate pair_index references valid questions and sources
        max_q_idx = self.pair_index[:, 0].max().item()
        if max_q_idx >= len(self.question_embs):
            raise ValueError(
                f"pair_index references question {max_q_idx}, "
                f"but only {len(self.question_embs)} questions exist"
            )

        max_s_idx = self.pair_index[:, 1].max().item()
        if max_s_idx >= len(self.source_embs):
            raise ValueError(
                f"pair_index references source {max_s_idx}, "
                f"but only {len(self.source_embs)} sources exist"
            )

        # Validate hard_negatives reference valid sources
        max_neg_idx = self.hard_negatives.max().item()
        if max_neg_idx >= len(self.source_embs):
            raise ValueError(
                f"hard_negatives references source {max_neg_idx}, "
                f"but only {len(self.source_embs)} sources exist"
            )

        # Validate steering tensors match number of pairs
        n_pairs = len(self.pair_index)
        if len(self.steering_centroid) != n_pairs:
            raise ValueError(
                f"steering_centroid has {len(self.steering_centroid)} entries, "
                f"expected {n_pairs}"
            )
        if len(self.steering_keyword_weighted) != n_pairs:
            raise ValueError(
                f"steering_keyword_weighted has {len(self.steering_keyword_weighted)} entries, "
                f"expected {n_pairs}"
            )
        if len(self.steering_residual) != n_pairs:
            raise ValueError(
                f"steering_residual has {len(self.steering_residual)} entries, "
                f"expected {n_pairs}"
            )

        LOGGER.debug("Referential integrity validated")

    def _move_to_device(self):
        """Move all tensors to target device."""
        LOGGER.info(f"Moving tensors to {self.device}...")

        self.question_embs = self.question_embs.to(self.device)
        self.source_embs = self.source_embs.to(self.device)
        self.keyword_embs = self.keyword_embs.to(self.device)
        self.centroid_embs = self.centroid_embs.to(self.device)

        self.pair_index = self.pair_index.to(self.device)
        self.pair_cluster_id = self.pair_cluster_id.to(self.device)
        self.pair_relevance = self.pair_relevance.to(self.device)

        self.hard_negatives = self.hard_negatives.to(self.device)
        self.negative_tiers = self.negative_tiers.to(self.device)

        self.steering_centroid = self.steering_centroid.to(self.device)
        self.steering_keyword_weighted = self.steering_keyword_weighted.to(self.device)
        self.steering_residual = self.steering_residual.to(self.device)
        self.centroid_distances = self.centroid_distances.to(self.device)

        self.split_indices = self.split_indices.to(self.device)

        LOGGER.info("Tensors moved to device")

    def _compute_memory_usage(self) -> float:
        """Compute total memory usage in MB."""
        tensors = [
            self.question_embs,
            self.source_embs,
            self.keyword_embs,
            self.centroid_embs,
            self.pair_index,
            self.pair_cluster_id,
            self.pair_relevance,
            self.hard_negatives,
            self.negative_tiers,
            self.steering_centroid,
            self.steering_keyword_weighted,
            self.steering_residual,
            self.centroid_distances,
            self.split_indices,
        ]
        total_bytes = sum(t.element_size() * t.nelement() for t in tensors)
        return total_bytes / 1024 / 1024

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

        Raises
        ------
        IndexError
            If idx is out of bounds
        """
        # Validate index bounds
        if not 0 <= idx < len(self.split_indices):
            raise IndexError(f"Index {idx} out of range [0, {len(self.split_indices)})")

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

        self.hard_negatives = load_tensor_artifact(
            self.dataset_dir, "hard_negatives.pt", expected_shape=(None, self.n_neg)
        )
        self.negative_tiers = load_tensor_artifact(
            self.dataset_dir, "negative_tiers.pt", expected_shape=(None, self.n_neg)
        )

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

    def close(self):
        """Release resources and log statistics."""
        if hasattr(self, "question_embs"):
            # Log final memory usage
            total_mb = self._compute_memory_usage()
            LOGGER.info(
                f"Dataset closed. Split: {self.split}, "
                f"Device: {self.device}, Memory: {total_mb:.1f} MB"
            )

            # Clear tensor references to help garbage collection
            del self.question_embs
            del self.source_embs
            del self.keyword_embs
            del self.centroid_embs
            del self.pair_index
            del self.pair_cluster_id
            del self.pair_relevance
            del self.hard_negatives
            del self.negative_tiers
            del self.steering_centroid
            del self.steering_keyword_weighted
            del self.steering_residual
            del self.centroid_distances
            del self.split_indices

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and release resources."""
        self.close()

    def __del__(self):
        """Destructor - safety fallback for cleanup."""
        try:
            self.close()
        except Exception as e:
            LOGGER.debug(f"Exception during JASPERSteeringDataset.__del__: {e}")

    @staticmethod
    def create_combined_splits(
        dataset_dir: Union[str, Path],
        epoch: int = 0,
        device: Optional[torch.device] = None,
    ) -> Dict[str, "JASPERSteeringDataset"]:
        """
        Load all splits (train/val/test) from one directory.

        Parameters
        ----------
        dataset_dir : str or Path
            Directory containing dataset files
        epoch : int, optional
            Initial epoch for curriculum learning (default: 0)
        device : torch.device, optional
            Device to load tensors to (default: CPU)

        Returns
        -------
        Dict[str, JASPERSteeringDataset]
            Dictionary with keys "train", "val", "test"

        Examples
        --------
        >>> splits = JASPERSteeringDataset.create_combined_splits(
        ...     "output/jasper_dataset",
        ...     device=torch.device("cuda"),
        ...     epoch=0
        ... )
        >>> train_dataset = splits["train"]
        >>> val_dataset = splits["val"]
        """
        return {
            "train": JASPERSteeringDataset(dataset_dir, split="train", epoch=epoch, device=device),
            "val": JASPERSteeringDataset(dataset_dir, split="val", epoch=epoch, device=device),
            "test": JASPERSteeringDataset(dataset_dir, split="test", epoch=epoch, device=device),
        }
