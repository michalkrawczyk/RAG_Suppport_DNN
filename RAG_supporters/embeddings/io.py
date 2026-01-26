"""
I/O utilities for loading suggestions from CSV files.

This module handles file operations for loading LLM suggestions:
- Loading suggestions from CSV files with chunked processing

For embedding save/load operations, use KeywordEmbedder.save_embeddings() and
KeywordEmbedder.load_embeddings() static methods.

Supports chunked processing for large files and includes progress tracking.
Part of the RAG_supporters.embeddings package.
"""

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
                        LOGGER.warning(f"Unexpected type at row {idx}: {type(suggestion_str)}")
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

        LOGGER.info(f"Loaded {len(all_suggestions)} total suggestions from {rows_processed} rows")

    except Exception as e:
        LOGGER.error(f"Error reading CSV file: {e}")
        raise

    return all_suggestions
