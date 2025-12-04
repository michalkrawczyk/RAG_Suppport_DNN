"""
Convenience wrapper class for keyword embedding operations.

This module provides a high-level interface for the complete embedding pipeline:
- Loading suggestions from CSV
- Filtering and aggregating keywords
- Creating embeddings
- Saving results

The KeywordEmbedder class integrates the core functions from embedding_core
and embedding_io modules to provide a simple, one-call interface for common
workflows.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np

from .embedding_core import (
    aggregate_unique_terms,
    create_embeddings_for_strings,
    filter_by_field_value,
)
from .embedding_io import load_suggestions_from_csv, save_embeddings_to_json

LOGGER = logging.getLogger(__name__)


class KeywordEmbedder:
    """
    Wrapper class for keyword embedding operations.
    Uses the core functions internally.
    """

    def __init__(
            self,
            embedding_model: Optional[Any] = None,
            model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        Initialize the keyword embedder.

        Parameters
        ----------
        embedding_model : Optional[Any]
            Pre-loaded embedding model
        model_name : str
            Name of the embedding model to use if embedding_model is None
        """
        self.model_name = model_name
        self.model = embedding_model

        # Load model if not provided
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                LOGGER.info(f"Loading embedding model: {model_name}")
                self.model = SentenceTransformer(model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Install with: pip install sentence-transformers"
                )

    def process_csv_to_embeddings(
            self,
            csv_path: str,
            output_path: str,
            min_confidence: float = 0.7,
            suggestion_column: str = "suggestions",
            normalize_keywords: bool = True,
            batch_size: int = 32,
            show_progress: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Complete pipeline: load CSV, filter, aggregate, embed, and save.

        Parameters
        ----------
        csv_path : str
            Path to input CSV file
        output_path : str
            Path to save embeddings JSON
        min_confidence : float
            Minimum confidence threshold
        suggestion_column : str
            Name of the suggestion column in CSV
        normalize_keywords : bool
            Whether to normalize keywords (lowercase, strip)
        batch_size : int
            Batch size for embedding generation
        show_progress : bool
            Whether to show progress bar

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary mapping keywords to embeddings
        """
        # Step 1: Load suggestions from CSV
        suggestions = load_suggestions_from_csv(csv_path, suggestion_column)

        # Step 2: Filter by confidence
        filtered_suggestions = filter_by_field_value(
            suggestions, min_confidence
        )

        # Step 3: Aggregate unique keywords
        keywords = aggregate_unique_terms(
            filtered_suggestions,
            normalize=normalize_keywords,
        )[0]

        if not keywords:
            LOGGER.warning("No keywords found after filtering and aggregation")
            return {}

        # Step 4: Create embeddings for keywords
        keyword_embeddings = create_embeddings_for_strings(
            keywords,
            embedding_model=self.model,
            model_name=self.model_name,
            batch_size=batch_size,
            show_progress=show_progress,
        )

        # Step 5: Save embeddings
        metadata = {
            "source_csv": str(csv_path),
            "min_confidence": min_confidence,
            "normalized": normalize_keywords,
        }
        save_embeddings_to_json(
            keyword_embeddings,
            output_path,
            model_name=self.model_name,
            metadata=metadata,
        )

        return keyword_embeddings
