"""
I/O utilities for loading suggestions from CSV files.

This module handles file operations for loading LLM suggestions:
- Loading suggestions from CSV files with chunked processing

For embedding save/load operations, use KeywordEmbedder.save_embeddings() and
KeywordEmbedder.load_embeddings() static methods.

Supports chunked processing for large files and includes progress tracking.
Part of the RAG_supporters.embeddings package.
"""

import ast
import json
import logging
from typing import Any, Dict, List


import pandas as pd

LOGGER = logging.getLogger(__name__)


def load_suggestions_from_csv(
    csv_path: str,
    suggestion_column: str = "suggestions",
    chunksize: int = 1000,
    show_progress: bool = True,
) -> List[Dict[str, Any]]:
    """
    Load and parse suggestions from CSV file (supports large files).

    Parameters
    ----------
    csv_path : str
        Path to the CSV file
    suggestion_column : str
        Name of the column containing suggestions (as JSON)
    chunksize : int
        Number of rows to process at a time (for large files)
    show_progress : bool
        Whether to show progress bar for large files

    Returns
    -------
    List[Dict[str, Any]]
        List of parsed suggestions

    Examples
    --------
    >>> suggestions = load_suggestions_from_csv('results.csv', 'suggestions')
    >>> len(suggestions)
    150
    """
    LOGGER.info(f"Loading suggestions from {csv_path}")

    # First, get total rows for progress tracking
    total_rows = None
    if show_progress:
        try:
            # Quick count of lines
            with open(csv_path, "r", encoding="utf-8") as f:
                total_rows = sum(1 for _ in f) - 1  # Exclude header
        except Exception as e:
            # If counting rows fails, skip progress tracking (non-critical).
            LOGGER.debug(f"Failed to count rows for progress tracking: {e}")

    all_suggestions = []

    try:
        # Read CSV in chunks
        chunk_iterator = pd.read_csv(csv_path, chunksize=chunksize)

        # Wrap with progress bar if requested
        if show_progress and total_rows:
            try:
                from tqdm import tqdm

                chunk_iterator = tqdm(
                    chunk_iterator,
                    total=(total_rows // chunksize) + 1,
                    desc="Processing CSV chunks",
                )
            except ImportError:
                LOGGER.debug("tqdm not available, skipping progress bar")

        rows_processed = 0

        for chunk_df in chunk_iterator:
            if suggestion_column not in chunk_df.columns:
                raise ValueError(f"Column '{suggestion_column}' not found in CSV")

            for idx, row in chunk_df.iterrows():
                suggestion_str = row[suggestion_column]

                if pd.isna(suggestion_str):
                    continue

                try:
                    # Parse JSON string
                    if isinstance(suggestion_str, str):
                        suggestion_str = suggestion_str.strip()
                        suggestions = json.loads(suggestion_str)
                    else:
                        LOGGER.warning(
                            f"Unexpected type at row {idx}: {type(suggestion_str)}"
                        )
                        continue

                    # Handle list of dicts
                    if isinstance(suggestions, list):
                        for item in suggestions:
                            if isinstance(item, dict):
                                all_suggestions.append(item)
                            else:
                                LOGGER.warning(
                                    f"Unexpected suggestion item type at row {idx}: {type(item)}"
                                )
                    else:
                        LOGGER.warning(
                            f"Expected list of suggestions at row {idx}, got {type(suggestions)}"
                        )

                except json.JSONDecodeError as e:
                    LOGGER.error(f"JSON decode error at row {idx}: {e}")
                    continue
                except Exception as e:
                    LOGGER.error(f"Error parsing suggestions at row {idx}: {e}")
                    continue

            rows_processed += len(chunk_df)

        LOGGER.info(
            f"Loaded {len(all_suggestions)} total suggestions from {rows_processed} rows"
        )

    except Exception as e:
        LOGGER.error(f"Error reading CSV file: {e}")
        raise

    return all_suggestions


def parse_json_or_literal(
    data_str: str,
    expected_type: Optional[type] = None,
) -> Union[List, Dict, Any]:
    """
    Robustly parse JSON or Python literal format strings.
    
    This function provides resilient parsing that handles both standard JSON
    (with double quotes) and Python literal syntax (with single quotes).
    It first attempts JSON parsing (preferred), then falls back to safe
    Python literal evaluation.
    
    Parameters
    ----------
    data_str : str
        String containing data in JSON or Python literal format
    expected_type : Optional[type]
        Expected type of the result (e.g., list, dict) for validation.
        If None, any valid type is accepted.
        
    Returns
    -------
    Union[List, Dict, Any]
        Parsed data structure (list, dict, or other Python literal)
        
    Raises
    ------
    ValueError
        If the string cannot be parsed as either JSON or Python literal,
        or if the result doesn't match the expected_type
        
    Examples
    --------
    >>> # Handles standard JSON (double quotes)
    >>> json_str = '[{"term": "ML", "type": "domain"}]'
    >>> parse_json_or_literal(json_str)
    [{'term': 'ML', 'type': 'domain'}]
    
    >>> # Also handles Python literal format (single quotes)
    >>> py_str = "[{'term': 'ML', 'type': 'domain'}]"
    >>> parse_json_or_literal(py_str)
    [{'term': 'ML', 'type': 'domain'}]
    
    >>> # With type validation
    >>> parse_json_or_literal('[1, 2, 3]', expected_type=list)
    [1, 2, 3]
    
    >>> # Type mismatch raises error
    >>> parse_json_or_literal('{"key": "value"}', expected_type=list)
    ValueError: Expected result type list, got dict
    
    Notes
    -----
    - JSON parsing is attempted first (faster and more standard)
    - Python literal evaluation uses ast.literal_eval() which is safe
      (doesn't execute arbitrary code)
    - Useful for handling LLM outputs that may use Python syntax instead of JSON
    """
    # Remove any leading/trailing whitespace
    data_str = data_str.strip()
    
    # Handle empty string
    if not data_str:
        raise ValueError("Cannot parse empty string")
    
    try:
        # Try JSON first (standard, preferred, and faster)
        result = json.loads(data_str)
        LOGGER.debug("Successfully parsed as JSON")
        
    except json.JSONDecodeError as json_error:
        LOGGER.debug(
            f"JSON parsing failed: {json_error}, "
            f"attempting Python literal evaluation"
        )
        
        try:
            # Fallback to safe Python literal evaluation
            # Handles: strings, bytes, numbers, tuples, lists, dicts, sets,
            # booleans, None, and nested structures
            result = ast.literal_eval(data_str)
            
            LOGGER.warning(
                "Parsed data using Python literal evaluation. "
                "Consider using standard JSON format (double quotes) "
                "for better compatibility and performance."
            )
            
        except (ValueError, SyntaxError) as eval_error:
            raise ValueError(
                f"Failed to parse data string as JSON or Python literal.\n"
                f"JSON parsing error: {str(json_error)}\n"
                f"Python literal error: {str(eval_error)}\n"
                f"Input preview (first 200 chars): {data_str[:200]}"
            ) from eval_error
    
    # Validate type if specified
    if expected_type is not None and not isinstance(result, expected_type):
        raise ValueError(
            f"Expected result type {expected_type.__name__}, "
            f"got {type(result).__name__}"
        )
    
    return result