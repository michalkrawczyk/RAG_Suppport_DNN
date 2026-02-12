"""
Shared validation utilities for dataset builder pipeline.

This module provides common validation functions to avoid code duplication
across builder classes (SteeringBuilder, NegativeMiner, DatasetSplitter, etc.).

Key Features:
- Tensor type and shape validation
- Dimension consistency checks
- Index bounds validation
- Ratio validation for splits
"""

import logging
from typing import Any, List, Optional, Tuple

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)


def validate_tensor_2d(
    tensor: Any,
    name: str,
    expected_cols: Optional[int] = None,
    min_rows: int = 1
) -> None:
    """Validate that tensor is 2D with optional column count check.
    
    Parameters
    ----------
    tensor : Any
        Object to validate
    name : str
        Name of tensor for error messages
    expected_cols : int, optional
        Expected number of columns, if None no check is performed
    min_rows : int, optional
        Minimum number of rows required (default: 1)
    
    Raises
    ------
    TypeError
        If tensor is not torch.Tensor
    ValueError
        If tensor is not 2D, has wrong column count, or too few rows
    
    Examples
    --------
    >>> validate_tensor_2d(torch.randn(10, 384), "embeddings", expected_cols=384)
    >>> validate_tensor_2d(torch.randn(5, 2), "pair_indices", expected_cols=2)
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be torch.Tensor, got {type(tensor)}")
    
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {tensor.shape}")
    
    if tensor.shape[0] < min_rows:
        raise ValueError(
            f"{name} must have at least {min_rows} rows, got {tensor.shape[0]}"
        )
    
    if expected_cols is not None and tensor.shape[1] != expected_cols:
        raise ValueError(
            f"{name} expected {expected_cols} columns, got {tensor.shape[1]}"
        )


def validate_tensor_1d(
    tensor: Any,
    name: str,
    expected_length: Optional[int] = None,
    min_length: int = 1
) -> None:
    """Validate that tensor is 1D with optional length check.
    
    Parameters
    ----------
    tensor : Any
        Object to validate
    name : str
        Name of tensor for error messages
    expected_length : int, optional
        Expected length, if None no check is performed
    min_length : int, optional
        Minimum length required (default: 1)
    
    Raises
    ------
    TypeError
        If tensor is not torch.Tensor
    ValueError
        If tensor is not 1D, has wrong length, or too short
    
    Examples
    --------
    >>> validate_tensor_1d(torch.tensor([1, 2, 3]), "cluster_ids", expected_length=3)
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be torch.Tensor, got {type(tensor)}")
    
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {tensor.shape}")
    
    if tensor.shape[0] < min_length:
        raise ValueError(
            f"{name} must have at least {min_length} elements, got {tensor.shape[0]}"
        )
    
    if expected_length is not None and tensor.shape[0] != expected_length:
        raise ValueError(
            f"{name} expected length {expected_length}, got {tensor.shape[0]}"
        )


def validate_embedding_dimensions(
    *tensors_and_names: Tuple[torch.Tensor, str]
) -> int:
    """Validate that all embedding tensors have consistent dimensions.
    
    Parameters
    ----------
    *tensors_and_names : Tuple[torch.Tensor, str]
        Pairs of (tensor, name) to validate
    
    Returns
    -------
    int
        The common embedding dimension
    
    Raises
    ------
    ValueError
        If tensors have inconsistent dimensions
    
    Examples
    --------
    >>> q_emb = torch.randn(100, 384)
    >>> s_emb = torch.randn(200, 384)
    >>> k_emb = torch.randn(50, 384)
    >>> dim = validate_embedding_dimensions(
    ...     (q_emb, "question_embs"),
    ...     (s_emb, "source_embs"),
    ...     (k_emb, "keyword_embs")
    ... )
    >>> print(dim)  # 384
    """
    if not tensors_and_names:
        raise ValueError("At least one tensor must be provided")
    
    dimensions = {}
    for tensor, name in tensors_and_names:
        validate_tensor_2d(tensor, name)
        dimensions[name] = tensor.shape[1]
    
    # Check all dimensions match
    unique_dims = set(dimensions.values())
    if len(unique_dims) > 1:
        dim_str = ", ".join(f"{name}={dim}" for name, dim in dimensions.items())
        raise ValueError(f"Embedding dimensions must match: {dim_str}")
    
    return list(unique_dims)[0]


def validate_pair_indices_bounds(
    pair_indices: torch.Tensor,
    n_questions: int,
    n_sources: int,
    name: str = "pair_indices"
) -> None:
    """Validate that pair indices are within bounds.
    
    Parameters
    ----------
    pair_indices : torch.Tensor
        Pair indices [n_pairs, 2] with (question_idx, source_idx)
    n_questions : int
        Number of available questions
    n_sources : int
        Number of available sources
    name : str, optional
        Name for error messages (default: "pair_indices")
    
    Raises
    ------
    ValueError
        If any indices are out of bounds
    
    Examples
    --------
    >>> pair_indices = torch.tensor([[0, 5], [1, 10], [2, 3]])
    >>> validate_pair_indices_bounds(pair_indices, n_questions=50, n_sources=20)
    """
    validate_tensor_2d(pair_indices, name, expected_cols=2)
    
    max_q_idx = pair_indices[:, 0].max().item()
    max_s_idx = pair_indices[:, 1].max().item()
    
    if max_q_idx >= n_questions:
        raise ValueError(
            f"{name} contains question index {max_q_idx} "
            f"but only {n_questions} questions exist"
        )
    
    if max_s_idx >= n_sources:
        raise ValueError(
            f"{name} contains source index {max_s_idx} "
            f"but only {n_sources} sources exist"
        )


def validate_cluster_ids_bounds(
    cluster_ids: torch.Tensor,
    n_clusters: int,
    name: str = "cluster_ids"
) -> None:
    """Validate that cluster IDs are within bounds.
    
    Parameters
    ----------
    cluster_ids : torch.Tensor
        Cluster IDs tensor [n_items]
    n_clusters : int
        Number of available clusters
    name : str, optional
        Name for error messages (default: "cluster_ids")
    
    Raises
    ------
    ValueError
        If any cluster IDs are out of bounds
    
    Examples
    --------
    >>> cluster_ids = torch.tensor([0, 1, 2, 1, 0])
    >>> validate_cluster_ids_bounds(cluster_ids, n_clusters=5)
    """
    validate_tensor_1d(cluster_ids, name)
    
    max_cluster_id = cluster_ids.max().item()
    if max_cluster_id >= n_clusters:
        raise ValueError(
            f"{name} contains cluster ID {max_cluster_id} "
            f"but only {n_clusters} clusters exist"
        )


def validate_length_consistency(
    *tensors_and_names: Tuple[Any, str, int]
) -> None:
    """Validate that tensors/lists have consistent lengths.
    
    Parameters
    ----------
    *tensors_and_names : Tuple[Any, str, int]
        Tuples of (tensor/list, name, expected_length)
    
    Raises
    ------
    ValueError
        If lengths don't match expectations
    
    Examples
    --------
    >>> pairs = torch.randn(100, 2)
    >>> clusters = torch.randint(0, 5, (100,))
    >>> keywords = [[1, 2], [3]] * 50  # 100 items
    >>> validate_length_consistency(
    ...     (pairs, "pair_indices", 100),
    ...     (clusters, "pair_cluster_ids", 100),
    ...     (keywords, "pair_keyword_ids", 100)
    ... )
    """
    for item, name, expected_length in tensors_and_names:
        if isinstance(item, torch.Tensor):
            actual_length = item.shape[0]
        elif isinstance(item, (list, tuple)):
            actual_length = len(item)
        else:
            raise TypeError(
                f"{name} must be torch.Tensor or list, got {type(item)}"
            )
        
        if actual_length != expected_length:
            raise ValueError(
                f"{name} length ({actual_length}) must equal {expected_length}"
            )


def validate_split_ratios(
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    tolerance: float = 1e-6
) -> None:
    """Validate train/val/test split ratios.
    
    Parameters
    ----------
    train_ratio : float
        Training set ratio
    val_ratio : float
        Validation set ratio
    test_ratio : float
        Test set ratio
    tolerance : float, optional
        Tolerance for sum check (default: 1e-6)
    
    Raises
    ------
    ValueError
        If ratios are invalid or don't sum to 1.0
    
    Examples
    --------
    >>> validate_split_ratios(0.7, 0.15, 0.15)
    >>> validate_split_ratios(0.8, 0.1, 0.1)
    """
    # Check individual ratios are in valid range
    if not (0 < train_ratio < 1):
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
    
    if not (0 < val_ratio < 1):
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
    
    if not (0 < test_ratio < 1):
        raise ValueError(f"test_ratio must be in (0, 1), got {test_ratio}")
    
    # Check sum
    ratio_sum = train_ratio + val_ratio + test_ratio
    if not np.isclose(ratio_sum, 1.0, atol=tolerance):
        raise ValueError(
            f"Ratios must sum to 1.0, got {ratio_sum:.6f} "
            f"(train={train_ratio}, val={val_ratio}, test={test_ratio})"
        )


def validate_keyword_ids_list(
    pair_keyword_ids: Any,
    n_pairs: int,
    n_keywords: int,
    name: str = "pair_keyword_ids"
) -> None:
    """Validate list-of-lists structure for pair keyword IDs.
    
    Parameters
    ----------
    pair_keyword_ids : Any
        Should be list of lists of keyword IDs
    n_pairs : int
        Expected number of pairs
    n_keywords : int
        Number of available keywords for bounds checking
    name : str, optional
        Name for error messages (default: "pair_keyword_ids")
    
    Raises
    ------
    TypeError
        If structure is invalid
    ValueError
        If lengths don't match or IDs are out of bounds
    
    Examples
    --------
    >>> pair_keyword_ids = [[0, 1, 2], [1, 3], [], [4, 5]]
    >>> validate_keyword_ids_list(pair_keyword_ids, n_pairs=4, n_keywords=10)
    """
    if not isinstance(pair_keyword_ids, list):
        raise TypeError(f"{name} must be list, got {type(pair_keyword_ids)}")
    
    if len(pair_keyword_ids) != n_pairs:
        raise ValueError(
            f"{name} length ({len(pair_keyword_ids)}) must equal n_pairs ({n_pairs})"
        )
    
    for pair_idx, keyword_ids in enumerate(pair_keyword_ids):
        if not isinstance(keyword_ids, list):
            raise TypeError(
                f"{name}[{pair_idx}] must be list, got {type(keyword_ids)}"
            )
        
        for kw_id in keyword_ids:
            if not isinstance(kw_id, int):
                raise TypeError(
                    f"{name}[{pair_idx}] contains non-int ID: {kw_id}"
                )
            
            if kw_id < 0 or kw_id >= n_keywords:
                raise ValueError(
                    f"{name}[{pair_idx}] contains out-of-range keyword ID {kw_id}, "
                    f"valid range is [0, {n_keywords - 1}]"
                )
