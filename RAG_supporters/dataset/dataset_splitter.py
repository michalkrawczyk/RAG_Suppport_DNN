"""Dataset splitting utilities with persistent sample tracking.

This module provides functionality to split datasets into training and validation sets
with support for saving and restoring split assignments. This ensures consistency
across runs and experiments.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

LOGGER = logging.getLogger(__name__)


class DatasetSplitter:
    """
    Utility class for splitting datasets with persistent sample tracking.

    This class provides methods to split dataset indices into training and validation
    sets, save the split configuration, and restore it for consistent experiments.

    Examples
    --------
    >>> # Create a split
    >>> splitter = DatasetSplitter(random_state=42)
    >>> train_indices, val_indices = splitter.split(dataset_size=1000, val_ratio=0.2)
    >>> splitter.save_split('split_config.json')
    >>>
    >>> # Later, restore the same split
    >>> splitter2 = DatasetSplitter.load_split('split_config.json')
    >>> train_indices2, val_indices2 = splitter2.get_split()
    """

    def __init__(self, random_state: Optional[int] = None):
        """
        Initialize DatasetSplitter.

        Parameters
        ----------
        random_state : Optional[int]
            Random seed for reproducibility. If None, splits will be non-deterministic.
        """
        self.random_state = random_state
        self.train_indices: Optional[List[int]] = None
        self.val_indices: Optional[List[int]] = None
        self.dataset_size: Optional[int] = None
        self.val_ratio: Optional[float] = None

    def split(
        self,
        dataset_size: int,
        val_ratio: float = 0.2,
        shuffle: bool = True,
    ) -> Tuple[List[int], List[int]]:
        """
        Split dataset indices into training and validation sets.

        Parameters
        ----------
        dataset_size : int
            Total number of samples in the dataset (must be >= 2)
        val_ratio : float, optional
            Ratio of validation samples (between 0 and 1). Default is 0.2.
        shuffle : bool, optional
            Whether to shuffle indices before splitting. Default is True.

        Returns
        -------
        Tuple[List[int], List[int]]
            Tuple of (train_indices, val_indices)

        Raises
        ------
        ValueError
            If val_ratio is not between 0 and 1, dataset_size is not positive,
            or if the split would result in empty train or validation sets.

        Notes
        -----
        Both train and validation sets must have at least one sample. For small
        datasets, ensure val_ratio is chosen such that both sets are non-empty:
        - Minimum val_ratio: 1/dataset_size
        - Maximum val_ratio: (dataset_size-1)/dataset_size
        """
        if not 0 < val_ratio < 1:
            raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")

        if dataset_size <= 0:
            raise ValueError(f"dataset_size must be positive, got {dataset_size}")

        # Calculate split point
        val_size = int(dataset_size * val_ratio)

        # Ensure both train and validation sets have at least one sample
        if val_size == 0:
            raise ValueError(
                f"Validation set would be empty with dataset_size={dataset_size} "
                f"and val_ratio={val_ratio}. Minimum validation set size is 1. "
                f"Consider using a larger dataset or increasing val_ratio to at least "
                f"{1.0/dataset_size:.4f}"
            )

        train_size = dataset_size - val_size
        if train_size == 0:
            raise ValueError(
                f"Training set would be empty with dataset_size={dataset_size} "
                f"and val_ratio={val_ratio}. Consider using a smaller val_ratio."
            )

        # Store parameters
        self.dataset_size = dataset_size
        self.val_ratio = val_ratio

        # Create indices
        indices = np.arange(dataset_size)

        # Shuffle if requested
        if shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(indices)

        # Split indices
        self.val_indices = indices[:val_size].tolist()
        self.train_indices = indices[val_size:].tolist()

        LOGGER.info(
            f"Split dataset of size {dataset_size} into "
            f"{len(self.train_indices)} train and {len(self.val_indices)} val samples"
        )

        return self.train_indices, self.val_indices

    def get_split(self) -> Tuple[List[int], List[int]]:
        """
        Get the current split.

        Returns
        -------
        Tuple[List[int], List[int]]
            Tuple of (train_indices, val_indices)

        Raises
        ------
        ValueError
            If no split has been created or loaded
        """
        if self.train_indices is None or self.val_indices is None:
            raise ValueError(
                "No split available. Call split() first or load from file."
            )

        return self.train_indices, self.val_indices

    def save_split(
        self,
        output_path: Union[str, Path],
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Save split configuration to a JSON file.

        Parameters
        ----------
        output_path : Union[str, Path]
            Path where the split configuration will be saved
        metadata : Optional[Dict]
            Additional metadata to save with the split

        Raises
        ------
        ValueError
            If no split has been created
        """
        if self.train_indices is None or self.val_indices is None:
            raise ValueError("No split to save. Call split() first.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        split_data = {
            "train_indices": self.train_indices,
            "val_indices": self.val_indices,
            "dataset_size": self.dataset_size,
            "val_ratio": self.val_ratio,
            "random_state": self.random_state,
            "metadata": metadata or {},
        }

        with open(output_path, "w") as f:
            json.dump(split_data, f, indent=2)

        LOGGER.info(f"Saved split configuration to {output_path}")

    @classmethod
    def load_split(cls, input_path: Union[str, Path]) -> "DatasetSplitter":
        """
        Load split configuration from a JSON file.

        Parameters
        ----------
        input_path : Union[str, Path]
            Path to the split configuration file

        Returns
        -------
        DatasetSplitter
            DatasetSplitter instance with loaded split

        Raises
        ------
        FileNotFoundError
            If the split configuration file does not exist
        ValueError
            If the file format is invalid
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Split configuration not found: {input_path}")

        with open(input_path, "r") as f:
            split_data = json.load(f)

        # Validate required fields
        required_fields = ["train_indices", "val_indices", "dataset_size"]
        missing_fields = [f for f in required_fields if f not in split_data]
        if missing_fields:
            raise ValueError(
                f"Invalid split file format. Missing fields: {missing_fields}"
            )

        # Create instance and restore state
        splitter = cls(random_state=split_data.get("random_state"))
        splitter.train_indices = split_data["train_indices"]
        splitter.val_indices = split_data["val_indices"]
        splitter.dataset_size = split_data["dataset_size"]
        splitter.val_ratio = split_data.get("val_ratio")

        LOGGER.info(
            f"Loaded split configuration from {input_path}: "
            f"{len(splitter.train_indices)} train, {len(splitter.val_indices)} val"
        )

        return splitter

    def validate_split(self, dataset_size: int) -> bool:
        """
        Validate that the current split is compatible with a dataset.

        Parameters
        ----------
        dataset_size : int
            Size of the dataset to validate against

        Returns
        -------
        bool
            True if split is valid for the dataset

        Raises
        ------
        ValueError
            If split is invalid or incompatible
        """
        if self.train_indices is None or self.val_indices is None:
            raise ValueError("No split to validate. Load or create a split first.")

        # Check for empty indices
        if not self.train_indices and not self.val_indices:
            raise ValueError("Split contains no indices")

        # Check that all indices are within bounds - more efficient for large datasets
        all_indices = set(self.train_indices) | set(self.val_indices)

        if all_indices:  # Only call max() if set is not empty
            max_idx = max(all_indices)

            if max_idx >= dataset_size:
                raise ValueError(
                    f"Split contains indices up to {max_idx}, "
                    f"but dataset size is only {dataset_size}"
                )

        # Check for duplicates
        if len(all_indices) != len(self.train_indices) + len(self.val_indices):
            raise ValueError("Split contains duplicate indices")

        # Check that stored dataset_size matches if available
        if self.dataset_size is not None and self.dataset_size != dataset_size:
            LOGGER.warning(
                f"Dataset size mismatch: split was created for size {self.dataset_size}, "
                f"but validating against size {dataset_size}"
            )

        return True


def create_train_val_split(
    dataset_size: int,
    val_ratio: float = 0.2,
    random_state: Optional[int] = None,
    shuffle: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    metadata: Optional[Dict] = None,
) -> Tuple[List[int], List[int]]:
    """
    Create a train/val split.

    Parameters
    ----------
    dataset_size : int
        Total number of samples in the dataset
    val_ratio : float, optional
        Ratio of validation samples. Default is 0.2.
    random_state : Optional[int], optional
        Random seed for reproducibility
    shuffle : bool, optional
        Whether to shuffle indices before splitting. Default is True.
    save_path : Optional[Union[str, Path]], optional
        If provided, save split configuration to this path
    metadata : Optional[Dict], optional
        Additional metadata to save with the split

    Returns
    -------
    Tuple[List[int], List[int]]
        Tuple of (train_indices, val_indices)

    Examples
    --------
    >>> train_idx, val_idx = create_train_val_split(
    ...     dataset_size=1000,
    ...     val_ratio=0.2,
    ...     random_state=42,
    ...     save_path='my_split.json'
    ... )
    """
    splitter = DatasetSplitter(random_state=random_state)
    train_indices, val_indices = splitter.split(
        dataset_size=dataset_size,
        val_ratio=val_ratio,
        shuffle=shuffle,
    )

    if save_path is not None:
        splitter.save_split(save_path, metadata=metadata)

    return train_indices, val_indices
