"""
I/O utilities for loading and saving embeddings and suggestions.

This module handles file operations for embeddings:
- Loading suggestions from CSV files
- Saving embeddings to JSON format
- Loading embeddings from JSON format

Supports chunked processing for large files and includes progress tracking.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
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
            with open(csv_path, 'r', encoding='utf-8') as f:
                total_rows = sum(1 for _ in f) - 1  # Exclude header
        except Exception:
            pass

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
                    desc="Processing CSV chunks"
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


def save_embeddings_to_json(
        keyword_embeddings: Dict[str, np.ndarray],
        output_path: str,
        model_name: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
):
    """
    Save keyword embeddings to JSON file.

    Parameters
    ----------
    keyword_embeddings : Dict[str, np.ndarray]
        Dictionary mapping keywords to embedding vectors
    output_path : str
        Path to save JSON file
    model_name : str
        Name of the embedding model used
    metadata : Optional[Dict[str, Any]]
        Additional metadata to include

    Examples
    --------
    >>> embeddings = {'keyword1': np.array([0.1, 0.2]), 'keyword2': np.array([0.3, 0.4])}
    >>> save_embeddings_to_json(embeddings, 'embeddings.json', 'my-model')
    """
    if not keyword_embeddings:
        LOGGER.warning("No embeddings to save")
        return

    # Convert embeddings to lists for JSON serialization
    embeddings_json = {
        keyword: embedding.tolist()
        for keyword, embedding in keyword_embeddings.items()
    }

    # Get embedding dimension
    first_embedding = next(iter(keyword_embeddings.values()))
    embedding_dim = len(first_embedding)

    # Prepare output data
    output_data = {
        "metadata": {
            "model_name": model_name,
            "embedding_dimension": embedding_dim,
            "num_keywords": len(keyword_embeddings),
            **(metadata or {}),
        },
        "embeddings": embeddings_json,
    }

    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    LOGGER.info(f"Saved {len(keyword_embeddings)} embeddings to {output_path}")


def load_embeddings_from_json(
        input_path: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Load keyword embeddings from JSON file.

    Parameters
    ----------
    input_path : str
        Path to JSON file

    Returns
    -------
    Tuple[Dict[str, np.ndarray], Dict[str, Any]]
        Tuple of (keyword_embeddings, metadata)

    Examples
    --------
    >>> embeddings, metadata = load_embeddings_from_json('embeddings.json')
    >>> print(metadata['model_name'])
    'sentence-transformers/all-MiniLM-L6-v2'
    """
    LOGGER.info(f"Loading embeddings from {input_path}")

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convert lists back to numpy arrays
    keyword_embeddings = {
        keyword: np.array(embedding)
        for keyword, embedding in data["embeddings"].items()
    }

    metadata = data.get("metadata", {})

    LOGGER.info(
        f"Loaded {len(keyword_embeddings)} embeddings "
        f"(model: {metadata.get('model_name')}, "
        f"dim: {metadata.get('embedding_dimension')})"
    )

    return keyword_embeddings, metadata
