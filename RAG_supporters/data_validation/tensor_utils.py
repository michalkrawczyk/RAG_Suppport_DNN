"""
Tensor loading and storage utilities for dataset builder pipeline.

This module provides common tensor I/O operations to avoid code duplication
in dataset loading (JASPERSteeringDataset, DatasetFinalizer, etc.).

Key Features:
- Standardized tensor loading with error handling
- Optional shape validation
- Support for weights_only flag
- Clear error messages for missing files
"""

import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import torch

LOGGER = logging.getLogger(__name__)


def load_tensor_artifact(
    dataset_dir: Union[str, Path],
    filename: str,
    weights_only: bool = True,
    expected_shape: Optional[Tuple[Optional[int], ...]] = None,
    required: bool = True,
) -> Optional[torch.Tensor]:
    """Load tensor artifact with validation.

    Parameters
    ----------
    dataset_dir : str or Path
        Directory containing tensor files
    filename : str
        Tensor filename (e.g., "question_embs.pt")
    weights_only : bool, optional
        Whether to use weights_only flag in torch.load (default: True)
    expected_shape : Tuple[Optional[int], ...], optional
        Expected tensor shape. Use None for dimensions that can vary.
        Example: (None, 384) means any number of rows, 384 columns
    required : bool, optional
        Whether file is required (default: True). If False and file missing,
        returns None instead of raising error.

    Returns
    -------
    torch.Tensor or None
        Loaded tensor, or None if not required and missing

    Raises
    ------
    FileNotFoundError
        If required file is missing
    ValueError
        If tensor shape doesn't match expected_shape

    Examples
    --------
    >>> # Load embeddings with shape validation
    >>> question_embs = load_tensor_artifact(
    ...     "output/dataset",
    ...     "question_embs.pt",
    ...     expected_shape=(None, 384)  # Any rows, 384 cols
    ... )
    >>>
    >>> # Load optional file
    >>> metadata = load_tensor_artifact(
    ...     "output/dataset",
    ...     "metadata.pt",
    ...     required=False
    ... )
    """
    dataset_path = Path(dataset_dir)
    file_path = dataset_path / filename

    if not file_path.exists():
        if not required:
            LOGGER.debug(f"Optional file not found: {file_path}")
            return None
        raise FileNotFoundError(f"Required tensor file not found: {file_path}")

    # Load tensor
    tensor = torch.load(file_path, weights_only=weights_only)

    # Validate shape if specified
    if expected_shape is not None:
        if len(expected_shape) != tensor.ndim:
            raise ValueError(
                f"{filename} has {tensor.ndim} dimensions, "
                f"expected {len(expected_shape)} dimensions"
            )

        for dim_idx, (expected, actual) in enumerate(zip(expected_shape, tensor.shape)):
            if expected is not None and expected != actual:
                raise ValueError(
                    f"{filename} dimension {dim_idx} has size {actual}, " f"expected {expected}"
                )

    # Only log shape if it's a tensor (not for lists loaded with weights_only=False)
    if isinstance(tensor, torch.Tensor):
        LOGGER.debug(f"Loaded {filename}: shape={tuple(tensor.shape)}")
    else:
        LOGGER.debug(f"Loaded {filename}: type={type(tensor).__name__}")
    return tensor


def load_multiple_tensors(
    dataset_dir: Union[str, Path],
    file_specs: List[Tuple[str, str, bool, Optional[Tuple[Optional[int], ...]]]],
) -> dict:
    """Load multiple tensor artifacts efficiently.

    Parameters
    ----------
    dataset_dir : str or Path
        Directory containing tensor files
    file_specs : List[Tuple[str, str, bool, Optional[Tuple]]]
        List of (key, filename, weights_only, expected_shape) tuples

    Returns
    -------
    dict
        Dictionary mapping keys to loaded tensors

    Examples
    --------
    >>> specs = [
    ...     ("question_embs", "question_embs.pt", True, (None, 384)),
    ...     ("source_embs", "source_embs.pt", True, (None, 384)),
    ...     ("pair_index", "pair_index.pt", True, (None, 2)),
    ... ]
    >>> tensors = load_multiple_tensors("output/dataset", specs)
    >>> print(tensors.keys())
    dict_keys(['question_embs', 'source_embs', 'pair_index'])
    """
    dataset_path = Path(dataset_dir)
    result = {}

    for key, filename, weights_only, expected_shape in file_specs:
        result[key] = load_tensor_artifact(
            dataset_path, filename, weights_only=weights_only, expected_shape=expected_shape
        )

    return result


def save_tensor_artifact(
    tensor: torch.Tensor, output_dir: Union[str, Path], filename: str, validate: bool = True
) -> None:
    """Save tensor artifact with optional validation.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to save
    output_dir : str or Path
        Output directory
    filename : str
        Output filename (e.g., "embeddings.pt")
    validate : bool, optional
        Whether to check for NaN/Inf before saving (default: True)

    Raises
    ------
    ValueError
        If tensor contains NaN or Inf values (when validate=True)

    Examples
    --------
    >>> embeddings = torch.randn(100, 384)
    >>> save_tensor_artifact(embeddings, "output/dataset", "embeddings.pt")
    """
    if validate:
        if torch.isnan(tensor).any():
            raise ValueError(f"Cannot save {filename}: tensor contains NaN values")
        if torch.isinf(tensor).any():
            raise ValueError(f"Cannot save {filename}: tensor contains Inf values")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    file_path = output_path / filename
    torch.save(tensor, file_path)

    LOGGER.debug(f"Saved {filename}: shape={tuple(tensor.shape)}")
