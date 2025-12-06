"""
Utilities for processing LLM suggestions from CSV files.

This module provides functions for processing suggestion data returned from LLM
calls, typically stored in CSV files as JSON-formatted fields:
- Filtering suggestions by field values (e.g., confidence scores)
- Aggregating unique terms from suggestions

These utilities are independent of embedding operations and can be used for
general data processing tasks.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

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
