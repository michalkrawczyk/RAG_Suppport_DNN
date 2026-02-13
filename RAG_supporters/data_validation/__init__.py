"""Data validation and utilities for RAG_supporters.

This module provides tensor validation, tensor I/O, and label normalization
utilities. These are highly reusable across any PyTorch project.

Key Features:
- Tensor shape and type validation
- NaN/Inf detection
- Index bounds validation  
- Dimension consistency checks
- Tensor I/O with shape validation
- Label normalization (softmax, L1)

Examples
--------
>>> from RAG_supporters.data_validation import validate_tensor_2d, load_tensor_artifact
>>>
>>> # Validate tensor shape
>>> validate_tensor_2d(embeddings, "embeddings", expected_cols=384)
>>>
>>> # Load tensor with validation
>>> tensor = load_tensor_artifact("output/", "embeddings.pt", expected_shape=(None, 384))
"""

from .validation_utils import (
    validate_tensor_2d,
    validate_tensor_1d,
    validate_embedding_dimensions,
    validate_pair_indices_bounds,
    validate_cluster_ids_bounds,
    validate_length_consistency,
    validate_split_ratios,
    validate_keyword_ids_list,
    validate_no_nan_inf,
    validate_values_in_range,
)
from .tensor_utils import (
    load_tensor_artifact,
    load_multiple_tensors,
    save_tensor_artifact,
)
from .label_calculator import (
    LabelCalculator,
    LabelNormalizationMethod,
    get_normalizer,
    SoftmaxNormalizer,
    L1Normalizer,
)

__all__ = [
    # Validation functions
    "validate_tensor_2d",
    "validate_tensor_1d",
    "validate_embedding_dimensions",
    "validate_pair_indices_bounds",
    "validate_cluster_ids_bounds",
    "validate_length_consistency",
    "validate_split_ratios",
    "validate_keyword_ids_list",
    "validate_no_nan_inf",
    "validate_values_in_range",
    # Tensor I/O
    "load_tensor_artifact",
    "load_multiple_tensors",
    "save_tensor_artifact",
    # Label calculation
    "LabelCalculator",
    "LabelNormalizationMethod",
    "get_normalizer",
    "SoftmaxNormalizer",
    "L1Normalizer",
]
