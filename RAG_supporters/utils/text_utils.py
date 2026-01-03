"""Text processing utilities."""

import ast
import json
import logging
import re
from typing import Any, Dict, List, Optional, Union

LOGGER = logging.getLogger(__name__)


def is_empty_text(text: str) -> bool:
    """Check if the text is empty or only whitespace."""
    if not text or text.strip() == "":
        return True
    if text.lower() == "nan":
        return True
    return False


def normalize_string(text: str) -> str:
    """
    Normalize a string by converting to lowercase and removing multiple spaces.

    Parameters
    ----------
    text : str
        String to normalize

    Returns
    -------
    str
        Normalized string

    Examples
    --------
    >>> normalize_string("  Machine Learning  ")
    'machine learning'
    >>> normalize_string("DATA   SCIENCE")
    'data science'
    """
    # Lowercase, strip, and remove multiple spaces
    normalized = text.lower().strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def parse_json_or_literal(
    data_str: str,
    expected_type: Optional[type] = None,
    debug: bool = False,
) -> Union[List, Dict, Any]:
    """
    Robustly parse JSON or Python literal format strings.

    Attempts JSON parsing first (fast and standard), then falls back to
    safe Python literal evaluation if needed.

    Parameters
    ----------
    data_str : str
        String containing data in JSON or Python literal format
    expected_type : Optional[type]
        Expected type for validation (e.g., list, dict)
    debug : bool, default=False
        Enable debug logging for parsing details

    Returns
    -------
    Union[List, Dict, Any]
        Parsed data structure

    Raises
    ------
    ValueError
        If parsing fails or type validation fails
    """
    data_str = data_str.strip()

    if not data_str:
        raise ValueError("Cannot parse empty string")

    # Try JSON first (preferred)
    try:
        result = json.loads(data_str)
        if debug:
            LOGGER.debug("Parsed as JSON")

    except json.JSONDecodeError as json_error:
        if debug:
            LOGGER.debug(f"JSON failed: {json_error}, trying literal eval")

        # Fallback to Python literal eval (safe)
        try:
            result = ast.literal_eval(data_str)
            if debug:
                LOGGER.warning(
                    "Used Python literal eval. Consider using JSON format (double quotes)."
                )

        except (ValueError, SyntaxError) as eval_error:
            raise ValueError(
                f"Failed to parse data.\n"
                f"JSON error: {json_error}\n"
                f"Literal error: {eval_error}\n"
                f"Preview: {data_str[:200]}"
            ) from eval_error

    # Validate type if specified
    if expected_type is not None and not isinstance(result, expected_type):
        raise ValueError(
            f"Expected {expected_type.__name__}, got {type(result).__name__}"
        )

    return result
