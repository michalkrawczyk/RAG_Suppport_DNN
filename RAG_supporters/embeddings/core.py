"""
Core embedding functions for filtering, aggregation, and embedding creation.

This module contains the fundamental operations for working with embeddings:
- Filtering suggestions by field values
- Aggregating unique terms from suggestions
- Creating embeddings for strings using sentence transformers

These functions are independent and can be used separately or combined in pipelines.
Part of the RAG_supporters.embeddings package.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


def filter_by_field_value(
    suggestions: List[Dict[str, Any]],
    min_value: float = 0.7,
    field_name: str = "confidence",
) -> List[Dict[str, Any]]:
    """
    Filter suggestions by a numeric field threshold.

    Parameters
    ----------
    suggestions : List[Dict[str, Any]]
        List of suggestions, each containing numeric field to filter by
    min_value : float
        Minimum threshold value (0.0 to 1.0 for confidence-like fields)
    field_name : str
        Name of the numeric field to filter by (default: 'confidence')

    Returns
    -------
    List[Dict[str, Any]]
        Filtered suggestions that meet the threshold

    Examples
    --------
    >>> suggestions = [
    ...     {'term': 'machine learning', 'confidence': 0.9},
    ...     {'term': 'data science', 'confidence': 0.6},
    ...     {'term': 'AI', 'confidence': 0.85}
    ... ]
    >>> filtered = filter_by_field_value(suggestions, 0.7)
    >>> len(filtered)
    2

    >>> # Filter by custom field
    >>> suggestions = [
    ...     {'term': 'keyword1', 'score': 0.8},
    ...     {'term': 'keyword2', 'score': 0.5}
    ... ]
    >>> filtered = filter_by_field_value(suggestions, 0.6, field_name='score')
    >>> len(filtered)
    1
    """
    if not suggestions:
        LOGGER.warning("Empty suggestions list provided")
        return []

    filtered = []
    skipped_missing_field = 0

    for suggestion in suggestions:
        # Check if field exists
        if field_name not in suggestion:
            skipped_missing_field += 1
            continue

        field_value = suggestion.get(field_name, 0.0)

        # Try to convert to float if not already
        try:
            field_value = float(field_value)
        except (ValueError, TypeError):
            LOGGER.warning(
                f"Could not convert field '{field_name}' value '{field_value}' to float, skipping"
            )
            continue

        if field_value >= min_value:
            filtered.append(suggestion)

    if skipped_missing_field > 0:
        LOGGER.warning(
            f"Skipped {skipped_missing_field} suggestions missing field '{field_name}'"
        )

    LOGGER.info(
        f"Filtered {len(filtered)}/{len(suggestions)} suggestions "
        f"with {field_name} >= {min_value}"
    )

    return filtered


def aggregate_unique_terms(
    suggestions: List[Dict[str, Any]],
    term_key: str = "term",
    normalize: bool = True,
    return_counts: bool = False,
) -> Tuple[List[str], Optional[Dict[str, int]]]:
    """
    Aggregate terms into unique keywords.

    Parameters
    ----------
    suggestions : List[Dict[str, Any]]
        List of suggestions, each containing term information
    term_key : str
        Key name for the term in suggestion dictionaries
    normalize : bool
        Whether to normalize terms (lowercase, strip whitespace)
    return_counts : bool
        If True, also return occurrence counts for each unique term

    Returns
    -------
    Tuple[List[str], Optional[Dict[str, int]]]
        Tuple of (unique_keywords, counts_dict)
        - unique_keywords: List of unique keywords (order preserved)
        - counts_dict: Dict mapping keywords to counts if return_counts=True, else None

    Examples
    --------
    >>> suggestions = [
    ...     {'term': 'Machine Learning', 'confidence': 0.9},
    ...     {'term': 'machine learning', 'confidence': 0.85},
    ...     {'term': 'Deep Learning', 'confidence': 0.8},
    ...     {'term': 'Machine Learning', 'confidence': 0.75}
    ... ]
    >>> keywords, counts = aggregate_unique_terms(suggestions, normalize=True)
    >>> keywords
    ['machine learning', 'deep learning']
    >>> counts is None
    True

    >>> keywords, counts = aggregate_unique_terms(suggestions, normalize=True, return_counts=True)
    >>> keywords
    ['machine learning', 'deep learning']
    >>> counts
    {'machine learning': 3, 'deep learning': 1}
    """
    if not suggestions:
        LOGGER.warning("Empty suggestions list provided")
        return [], None

    keywords = []
    seen = set()
    keyword_counts = {} if return_counts else None

    for suggestion in suggestions:
        term = suggestion.get(term_key, "")
        if not term:
            continue

        # Normalize if requested
        if normalize:
            term = term.lower().strip()
        else:
            term = term.strip()

        # Track counts if requested
        if return_counts:
            keyword_counts[term] = keyword_counts.get(term, 0) + 1

        # Add to list if not seen (preserving order)
        if term not in seen:
            keywords.append(term)
            seen.add(term)

    log_msg = f"Aggregated {len(keywords)} unique keywords from {len(suggestions)} suggestions"
    if return_counts:
        log_msg += " (with counts)"
    LOGGER.info(log_msg)

    return keywords, keyword_counts


def create_embeddings_for_strings(
    str_list: List[str],
    embedding_model: Optional[Any] = None,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
    show_progress: bool = True,
    normalize_embeddings: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Create embeddings for each keyword separately.

    Parameters
    ----------
    str_list : List[str]
        List of words to embed
    embedding_model : Optional[Any]
        Pre-loaded embedding model. If None, will load model_name
    model_name : str
        Name of the embedding model to use if embedding_model is None
    batch_size : int
        Batch size for embedding generation
    show_progress : bool
        Whether to show progress bar during embedding generation
    normalize_embeddings : bool
        Whether to L2-normalize the embeddings

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping each word to its embedding vector

    Examples
    --------
    >>> str_list = ['machine learning', 'data science', 'artificial intelligence']
    >>> embeddings = create_embeddings_for_strings(str_list)
    >>> len(embeddings)
    3
    >>> embeddings['machine learning'].shape
    (384,)  # Depends on model

    Raises
    ------
    ImportError
        If sentence-transformers is not installed
    ValueError
        If keywords list is empty
    """
    if not str_list:
        raise ValueError("Keywords list cannot be empty")

    # Remove duplicates while preserving order
    unique_keywords = list(dict.fromkeys(str_list))

    if len(unique_keywords) < len(str_list):
        LOGGER.warning(
            f"Removed {len(str_list) - len(unique_keywords)} duplicate keywords"
        )

    # Load model if not provided
    if embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer

            LOGGER.info(f"Loading embedding model: {model_name}")
            embedding_model = SentenceTransformer(model_name)
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for embedding generation. "
                "Install with: pip install sentence-transformers"
            )

    LOGGER.info(f"Generating embeddings for {len(unique_keywords)} keywords")

    # Generate embeddings
    embeddings = embedding_model.encode(
        unique_keywords,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )

    # Create keyword -> embedding mapping
    keyword_embeddings = {
        keyword: embedding for keyword, embedding in zip(unique_keywords, embeddings)
    }

    embedding_dim = embeddings.shape[1]
    LOGGER.info(
        f"Successfully generated embeddings with dimension {embedding_dim} "
        f"for {len(keyword_embeddings)} keywords"
    )

    return keyword_embeddings
