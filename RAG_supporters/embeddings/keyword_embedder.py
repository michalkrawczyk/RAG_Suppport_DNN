"""
KeywordEmbedder class for keyword embedding operations.

This module provides a high-level interface for the complete embedding pipeline:
- Creating embeddings for strings using sentence transformers
- Saving and loading embeddings to/from JSON
- Processing CSV files with LLM suggestions into embeddings

The KeywordEmbedder class encapsulates embedding model management and provides
both instance methods and static utility methods for embedding operations.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


class KeywordEmbedder:
    """
    Class for keyword embedding operations.

    Provides methods for creating, saving, and loading keyword embeddings using
    sentence transformer models.
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

    def create_embeddings(
        self,
        str_list: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
        normalize_embeddings: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Create embeddings for a list of strings.

        Parameters
        ----------
        str_list : List[str]
            List of strings to embed
        batch_size : int
            Batch size for embedding generation
        show_progress : bool
            Whether to show progress bar during embedding generation
        normalize_embeddings : bool
            Whether to L2-normalize the embeddings

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary mapping each string to its embedding vector

        Raises
        ------
        ValueError
            If string list is empty

        Examples
        --------
        >>> embedder = KeywordEmbedder()
        >>> str_list = ['machine learning', 'data science']
        >>> embeddings = embedder.create_embeddings(str_list)
        >>> len(embeddings)
        2
        """
        if not str_list:
            raise ValueError("String list cannot be empty")

        # Remove duplicates while preserving order
        unique_strs = list(dict.fromkeys(str_list))

        if len(unique_strs) < len(str_list):
            LOGGER.warning(
                f"Removed {len(str_list) - len(unique_strs)} duplicate strings"
            )

        LOGGER.info(f"Generating embeddings for {len(unique_strs)} strings")

        # Generate embeddings
        embeddings = self.model.encode(
            unique_strs,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings,
        )

        # Create string -> embedding mapping
        string_embeddings = {
            string: embedding for string, embedding in zip(unique_strs, embeddings)
        }

        embedding_dim = embeddings.shape[1]
        LOGGER.info(
            f"Successfully generated embeddings with dimension {embedding_dim} "
            f"for {len(string_embeddings)} strings"
        )

        return string_embeddings

    @staticmethod
    def save_embeddings(
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
        >>> embeddings = {'keyword1': np.array([0.1, 0.2])}
        >>> KeywordEmbedder.save_embeddings(embeddings, 'embeddings.json', 'my-model')
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

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        LOGGER.info(f"Saved {len(keyword_embeddings)} embeddings to {output_path}")

    @staticmethod
    def load_embeddings(
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
        >>> embeddings, metadata = KeywordEmbedder.load_embeddings('embeddings.json')
        >>> print(metadata['model_name'])
        'sentence-transformers/all-MiniLM-L6-v2'
        """
        LOGGER.info(f"Loading embeddings from {input_path}")

        with open(input_path, "r", encoding="utf-8") as f:
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
        from ..utils.suggestion_processing import (
            aggregate_unique_terms,
            filter_by_field_value,
        )
        from .io import load_suggestions_from_csv

        # Step 1: Load suggestions from CSV
        suggestions = load_suggestions_from_csv(csv_path, suggestion_column)

        # Step 2: Filter by confidence
        filtered_suggestions = filter_by_field_value(suggestions, min_confidence)

        # Step 3: Aggregate unique keywords
        keywords = aggregate_unique_terms(
            filtered_suggestions,
            normalize=normalize_keywords,
        )[0]

        if not keywords:
            LOGGER.warning("No keywords found after filtering and aggregation")
            return {}

        # Step 4: Create embeddings for keywords
        keyword_embeddings = self.create_embeddings(
            keywords,
            batch_size=batch_size,
            show_progress=show_progress,
        )

        # Step 5: Save embeddings
        metadata = {
            "source_csv": str(csv_path),
            "min_confidence": min_confidence,
            "normalized": normalize_keywords,
        }
        self.save_embeddings(
            keyword_embeddings,
            output_path,
            model_name=self.model_name,
            metadata=metadata,
        )

        return keyword_embeddings
