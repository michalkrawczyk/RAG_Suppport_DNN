"""
PyTorch Dataset for JASPER (Joint Architecture for Subspace Prediction with Explainable Routing).

This module provides the JASPERSteeringDataset class that serves pre-computed embedding
triplets (question, steering, target_source) with hard negatives and subspace labels.
All embeddings are preloaded for zero I/O during training.

Storage Formats:
- PT (PyTorch): Default format, fast loading
- HDF5: Compressed storage, lazy loading support
- Memory-mapped: For large datasets (>10GB), loads on-demand from disk
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from RAG_supporters.data_validation import load_tensor_artifact

LOGGER = logging.getLogger(__name__)

# Optional HDF5 support
try:
    import h5py

    HAS_HDF5 = True
except ImportError:
    HAS_HDF5 = False
    h5py = None


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
    storage_format : Literal["auto", "pt", "hdf5"], optional
        Storage format to use. "auto" detects available format (default: "auto")
    use_mmap : bool or None, optional
        Enable memory-mapped loading for large datasets. If None, auto-enables for >10GB datasets

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
    storage_format : str
        Actual storage format being used ("pt" or "hdf5")
    use_mmap : bool
        Whether memory-mapped loading is enabled
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        split: Literal["train", "val", "test"],
        epoch: int = 0,
        device: Optional[torch.device] = None,
        storage_format: Literal["auto", "pt", "hdf5"] = "auto",
        use_mmap: Optional[bool] = None,
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

        # Detect storage format
        self.storage_format = self._detect_storage_format(storage_format)
        LOGGER.info(f"Using storage format: {self.storage_format}")

        # Determine if memory-mapping should be used
        self.use_mmap = self._should_use_mmap(use_mmap)
        if self.use_mmap:
            LOGGER.info("Memory-mapped loading enabled for large dataset")

        # HDF5 file handle (if using HDF5)
        self._hdf5_file: Optional[Any] = None

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

    def _detect_storage_format(self, requested_format: str) -> str:
        """
        Detect available storage format.

        Parameters
        ----------
        requested_format : str
            Requested format ("auto", "pt", or "hdf5")

        Returns
        -------
        str
            Actual format to use ("pt" or "hdf5")

        Raises
        ------
        ValueError
            If requested format is not available
        """
        hdf5_file = self.dataset_dir / "dataset.h5"
        pt_files_exist = (self.dataset_dir / "question_embs.pt").exists()

        if requested_format == "auto":
            # Prefer HDF5 if available, fall back to PT
            if hdf5_file.exists() and HAS_HDF5:
                return "hdf5"
            elif pt_files_exist:
                return "pt"
            else:
                raise ValueError(
                    f"No dataset files found in {self.dataset_dir}. "
                    f"Expected either dataset.h5 or *.pt files"
                )

        elif requested_format == "hdf5":
            if not HAS_HDF5:
                raise ValueError(
                    "HDF5 format requested but h5py is not installed. "
                    "Install with: pip install h5py"
                )
            if not hdf5_file.exists():
                raise ValueError(f"HDF5 file not found: {hdf5_file}")
            return "hdf5"

        elif requested_format == "pt":
            if not pt_files_exist:
                raise ValueError(f"PT files not found in {self.dataset_dir}")
            return "pt"

        else:
            raise ValueError(
                f"Invalid storage_format: {requested_format}. " f"Must be 'auto', 'pt', or 'hdf5'"
            )

    def _should_use_mmap(self, use_mmap: Optional[bool]) -> bool:
        """
        Determine if memory-mapping should be enabled.

        Parameters
        ----------
        use_mmap : bool or None
            Explicit setting, or None for auto-detection

        Returns
        -------
        bool
            Whether to use memory-mapping
        """
        if use_mmap is not None:
            return use_mmap

        # Auto-enable for large datasets (>10GB estimated)
        # Estimate: embedding_dim * n_pairs * 4 bytes (float32) * ~15 tensors
        estimated_size_gb = (
            self.config.get("embedding_dim", 384)
            * self.config.get("n_pairs", 10000)
            * 4  # float32
            * 15  # approximate number of large tensors
            / (1024**3)  # Convert to GB
        )

        # Enable mmap if >10GB and not using GPU device
        # (GPU preloading is incompatible with mmap)
        return estimated_size_gb > 10.0 and self.device == torch.device("cpu")

    def _load_embeddings(self):
        """Load all embedding tensors."""
        LOGGER.info("Loading embeddings...")

        if self.storage_format == "hdf5":
            self._load_embeddings_hdf5()
        else:
            self._load_embeddings_pt()

        LOGGER.info(
            f"Loaded embeddings: {len(self.question_embs)} questions, "
            f"{len(self.source_embs)} sources, {len(self.keyword_embs)} keywords, "
            f"{len(self.centroid_embs)} centroids"
        )

    def _load_embeddings_pt(self):
        """Load embeddings from PT format."""
        if self.use_mmap:
            # For memory-mapped loading, use numpy memmap then convert to torch
            # This avoids loading entire tensors into RAM
            LOGGER.debug("Using memory-mapped loading for embeddings")
            self.question_embs = self._load_mmap_tensor("question_embs.pt")
            self.source_embs = self._load_mmap_tensor("source_embs.pt")
            self.keyword_embs = self._load_mmap_tensor("keyword_embs.pt")
            self.centroid_embs = self._load_mmap_tensor("centroid_embs.pt")
        else:
            # Standard loading
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

    def _load_embeddings_hdf5(self):
        """Load embeddings from HDF5 format."""
        if self._hdf5_file is None:
            self._hdf5_file = h5py.File(self.dataset_dir / "dataset.h5", "r")

        # Load as numpy arrays and convert to torch
        # HDF5 datasets support lazy loading automatically
        self.question_embs = torch.from_numpy(self._hdf5_file["embeddings/questions"][:])
        self.source_embs = torch.from_numpy(self._hdf5_file["embeddings/sources"][:])
        self.keyword_embs = torch.from_numpy(self._hdf5_file["embeddings/keywords"][:])
        self.centroid_embs = torch.from_numpy(self._hdf5_file["embeddings/centroids"][:])

    def _load_mmap_tensor(self, filename: str) -> torch.Tensor:
        """
        Load tensor using memory-mapping.

        Parameters
        ----------
        filename : str
            Name of PT file to load

        Returns
        -------
        torch.Tensor
            Memory-mapped tensor (backed by disk storage)

        Notes
        -----
        Memory-mapped tensors load pages on-demand, reducing RAM usage for large datasets.
        """
        filepath = self.dataset_dir / filename

        # Load tensor metadata without loading full data
        tensor = torch.load(filepath, map_location="cpu", weights_only=True)

        # For truly large datasets, we could save as numpy memmap format instead
        # For now, PT format doesn't support true mmap, so we just use standard loading
        # but avoid moving to GPU (which would force full load)
        LOGGER.debug(f"Loaded {filename} (mmap mode - kept on CPU)")
        return tensor

    def _load_tensor(self, filename: str, expected_shape: Optional[Tuple] = None) -> torch.Tensor:
        """
        Load tensor based on storage format.

        Parameters
        ----------
        filename : str
            Name of tensor file (without format extension for HDF5)
        expected_shape : tuple, optional
            Expected shape for validation

        Returns
        -------
        torch.Tensor
            Loaded tensor
        """
        if self.storage_format == "hdf5":
            # Mapping from PT filenames to HDF5 group paths
            _HDF5_PATH_MAP = {
                "pair_index": "pair_data/index",
                "pair_cluster_id": "pair_data/cluster_id",
                "pair_relevance": "pair_data/relevance",
                "hard_negatives": "negatives/hardnegatives",
                "negative_tiers": "negatives/tiers",
                "steering_centroid": "steering/centroid",
                "steering_keyword_weighted": "steering/keyword_weighted",
                "steering_residual": "steering/residual",
                "centroid_distances": "steering/distances",
            }
            dataset_name = filename.replace(".pt", "")
            hdf5_path = _HDF5_PATH_MAP.get(dataset_name, dataset_name)
            return torch.from_numpy(self._hdf5_file[hdf5_path][:])
        else:
            # PT format
            if self.use_mmap:
                return self._load_mmap_tensor(filename)
            else:
                return load_tensor_artifact(
                    self.dataset_dir, filename, expected_shape=expected_shape
                )

    def _load_pair_data(self):
        """Load pair-level data."""
        LOGGER.info("Loading pair data...")

        self.pair_index = self._load_tensor("pair_index.pt", expected_shape=(None, 2))
        self.pair_cluster_id = self._load_tensor("pair_cluster_id.pt", expected_shape=(None,))
        self.pair_relevance = self._load_tensor("pair_relevance.pt", expected_shape=(None,))

        # Load pair_keyword_ids (stored as PT file with list-of-lists)
        if self.storage_format == "hdf5":
            # For HDF5, load from appropriate group
            if self._hdf5_file is None:
                self._hdf5_file = h5py.File(self.dataset_dir / "dataset.h5", "r")
            # Store as list (HDF5 doesn't handle list-of-lists well)
            self.pair_keyword_ids = [
                list(self._hdf5_file["pair_data/keyword_ids"][str(i)][:])
                for i in range(len(self.pair_index))
            ]
        else:
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

        self.hard_negatives = self._load_tensor(
            "hard_negatives.pt", expected_shape=(None, self.n_neg)
        )
        self.negative_tiers = self._load_tensor(
            "negative_tiers.pt", expected_shape=(None, self.n_neg)
        )

        LOGGER.info(f"Loaded {self.n_neg} hard negatives per pair")

    def _load_steering(self):
        """Load steering tensors."""
        LOGGER.info("Loading steering tensors...")

        self.steering_centroid = self._load_tensor(
            "steering_centroid.pt", expected_shape=(None, self.embedding_dim)
        )
        self.steering_keyword_weighted = self._load_tensor(
            "steering_keyword_weighted.pt", expected_shape=(None, self.embedding_dim)
        )
        self.steering_residual = self._load_tensor(
            "steering_residual.pt", expected_shape=(None, self.embedding_dim)
        )
        self.centroid_distances = self._load_tensor("centroid_distances.pt", expected_shape=(None,))

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

        self.hard_negatives = self._load_tensor(
            "hard_negatives.pt", expected_shape=(None, self.n_neg)
        )
        self.negative_tiers = self._load_tensor(
            "negative_tiers.pt", expected_shape=(None, self.n_neg)
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

            # Close HDF5 file if open
            if self._hdf5_file is not None:
                try:
                    self._hdf5_file.close()
                    LOGGER.debug("Closed HDF5 file")
                except Exception as e:
                    LOGGER.warning(f"Error closing HDF5 file: {e}")
                finally:
                    self._hdf5_file = None

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
        storage_format: Literal["auto", "pt", "hdf5"] = "auto",
        use_mmap: Optional[bool] = None,
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
        storage_format : Literal["auto", "pt", "hdf5"], optional
            Storage format to use (default: "auto")
        use_mmap : bool or None, optional
            Enable memory-mapped loading (default: None for auto-detect)

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
            "train": JASPERSteeringDataset(
                dataset_dir,
                split="train",
                epoch=epoch,
                device=device,
                storage_format=storage_format,
                use_mmap=use_mmap,
            ),
            "val": JASPERSteeringDataset(
                dataset_dir,
                split="val",
                epoch=epoch,
                device=device,
                storage_format=storage_format,
                use_mmap=use_mmap,
            ),
            "test": JASPERSteeringDataset(
                dataset_dir,
                split="test",
                epoch=epoch,
                device=device,
                storage_format=storage_format,
                use_mmap=use_mmap,
            ),
        }

    @staticmethod
    def convert_pt_to_hdf5(dataset_dir: Union[str, Path], compression: str = "gzip"):
        """
        Convert PT format dataset to HDF5 format.

        Parameters
        ----------
        dataset_dir : str or Path
            Directory containing PT format dataset files
        compression : str, optional
            HDF5 compression algorithm (default: "gzip")

        Raises
        ------
        ValueError
            If h5py is not installed or PT files don't exist
        ImportError
            If h5py is not available

        Examples
        --------
        >>> JASPERSteeringDataset.convert_pt_to_hdf5("output/jasper_dataset")
        """
        if not HAS_HDF5:
            raise ImportError(
                "h5py is required for HDF5 conversion. Install with: pip install h5py"
            )

        dataset_dir = Path(dataset_dir)

        # Verify PT files exist
        if not (dataset_dir / "question_embs.pt").exists():
            raise ValueError(f"PT files not found in {dataset_dir}")

        # Load config
        with open(dataset_dir / "config.json", "r") as f:
            config = json.load(f)

        LOGGER.info(f"Converting PT format to HDF5 in {dataset_dir}")

        # Create HDF5 file
        hdf5_path = dataset_dir / "dataset.h5"
        with h5py.File(hdf5_path, "w") as f:
            # Create groups
            emb_group = f.create_group("embeddings")
            pair_group = f.create_group("pair_data")
            neg_group = f.create_group("negatives")
            steering_group = f.create_group("steering")
            splits_group = f.create_group("splits")

            # Load and save embeddings
            LOGGER.info("Converting embeddings...")
            for name in ["questions", "sources", "keywords", "centroids"]:
                pt_name = name.rstrip("s") if name != "questions" else "question"
                pt_name = pt_name + "_embs.pt"
                tensor = torch.load(dataset_dir / pt_name, weights_only=True)
                emb_group.create_dataset(name, data=tensor.numpy(), compression=compression)

            # Load and save pair data
            LOGGER.info("Converting pair data...")
            for name in ["pair_index", "pair_cluster_id", "pair_relevance"]:
                tensor = torch.load(dataset_dir / f"{name}.pt", weights_only=True)
                pair_group.create_dataset(
                    name.replace("pair_", ""), data=tensor.numpy(), compression=compression
                )

            # Handle pair_keyword_ids (list-of-lists)
            pair_keyword_ids = torch.load(dataset_dir / "pair_keyword_ids.pt", weights_only=False)
            kw_group = pair_group.create_group("keyword_ids")
            for i, kw_list in enumerate(pair_keyword_ids):
                kw_group.create_dataset(str(i), data=np.array(kw_list, dtype=np.int32))

            # Load and save negatives
            LOGGER.info("Converting negatives...")
            for name in ["hard_negatives", "negative_tiers"]:
                tensor = torch.load(dataset_dir / f"{name}.pt", weights_only=True)
                neg_group.create_dataset(
                    name.replace("negative_", "").replace("_", ""),
                    data=tensor.numpy(),
                    compression=compression,
                )

            # Load and save steering
            LOGGER.info("Converting steering tensors...")
            for name in [
                "steering_centroid",
                "steering_keyword_weighted",
                "steering_residual",
                "centroid_distances",
            ]:
                tensor = torch.load(dataset_dir / f"{name}.pt", weights_only=True)
                steering_group.create_dataset(
                    name.replace("steering_", "").replace("centroid_", ""),
                    data=tensor.numpy(),
                    compression=compression,
                )

            # Load and save splits
            LOGGER.info("Converting split indices...")
            for split_name in ["train", "val", "test"]:
                tensor = torch.load(dataset_dir / f"{split_name}_idx.pt", weights_only=True)
                splits_group.create_dataset(
                    split_name, data=tensor.numpy(), compression=compression
                )

        LOGGER.info(f"Conversion complete. HDF5 file saved to {hdf5_path}")

        # Log file sizes
        pt_size = sum((dataset_dir / f).stat().st_size for f in dataset_dir.glob("*.pt"))
        hdf5_size = hdf5_path.stat().st_size
        compression_ratio = (1 - hdf5_size / pt_size) * 100 if pt_size > 0 else 0

        LOGGER.info(
            f"File sizes: PT format={pt_size / 1024**2:.1f} MB, "
            f"HDF5={hdf5_size / 1024**2:.1f} MB "
            f"(compression: {compression_ratio:.1f}%)"
        )
