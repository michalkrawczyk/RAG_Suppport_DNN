"""
KeywordEmbedder class for keyword embedding operations.

This module provides a high-level interface for the complete embedding pipeline:
- Creating embeddings for strings using sentence transformers or LangChain models
- Saving and loading embeddings to/from JSON
- Processing CSV files with LLM suggestions into embeddings

The KeywordEmbedder class encapsulates embedding model management and provides
both instance methods and static utility methods for embedding operations.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from tqdm import tqdm
from RAG_supporters.utils.text_utils import normalize_string

LOGGER = logging.getLogger(__name__)


class KeywordEmbedder:
    """
    Class for keyword embedding operations.

    Supports both sentence-transformers and LangChain embedding models.
    Provides methods for creating, saving, and loading keyword embeddings.
    """

    def __init__(
        self,
        embedding_model: Optional[Any] = None,
        model_name: Optional[str] = None,
        model_type: Optional[Literal["sentence-transformers", "langchain"]] = None,
    ):
        """
        Initialize the keyword embedder.

        Parameters
        ----------
        embedding_model : Optional[Any]
            Pre-loaded embedding model (sentence-transformers or LangChain)
        model_name : Optional[str]
            Name of the embedding model to use. If None and embedding_model is provided,
            will attempt to extract from the model instance. If None and no model provided,
            defaults to "sentence-transformers/all-MiniLM-L6-v2"
        model_type : Optional[Literal["sentence-transformers", "langchain"]]
            Type of embedding model. If None, will be auto-detected.
        """
        self.model = embedding_model

        # Detect or set model type
        if self.model is not None:
            self.model_type = model_type or self._detect_model_type(self.model)
            # Extract model name from instance if not provided
            self.model_name = model_name or self._extract_model_name(self.model)
        else:
            self.model_type = model_type or "sentence-transformers"
            self.model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
            self.model = self._load_default_model()

    def _detect_model_type(self, model: Any) -> str:
        """
        Detect whether the model is from sentence-transformers or LangChain.

        Parameters
        ----------
        model : Any
            Embedding model instance

        Returns
        -------
        str
            Model type: "sentence-transformers" or "langchain"

        Raises
        ------
        ValueError
            If model type cannot be detected
        """
        # Check for LangChain Embeddings interface
        if hasattr(model, "embed_documents") and hasattr(model, "embed_query"):
            LOGGER.info("Detected LangChain embedding model")
            return "langchain"
        # Check for sentence-transformers
        elif hasattr(model, "encode"):
            LOGGER.info("Detected sentence-transformers model")
            return "sentence-transformers"
        else:
            raise ValueError(
                "Unable to detect model type. Model must be either a "
                "sentence-transformers SentenceTransformer or a LangChain Embeddings model"
            )

    def _extract_model_name(self, model: Any) -> str:
        """
        Extract model name from a model instance.

        Parameters
        ----------
        model : Any
            Embedding model instance

        Returns
        -------
        str
            Extracted model name or a default value
        """
        if self.model_type == "sentence-transformers":
            # SentenceTransformer models have various ways to get the name
            if hasattr(model, "model_name"):
                return model.model_name
            elif hasattr(model, "_model_name"):
                return model._model_name
            # Try to get from the config
            elif hasattr(model, "_modules") and "0" in model._modules:
                auto_model = model._modules["0"]
                if hasattr(auto_model, "auto_model"):
                    config = getattr(auto_model.auto_model, "config", None)
                    if config and hasattr(config, "_name_or_path"):
                        return config._name_or_path
            return "sentence-transformers/unknown"

        elif self.model_type == "langchain":
            # LangChain models have different attribute names for the model identifier
            # Try common attribute names in order of preference
            model_attrs = [
                "model",  # OpenAIEmbeddings, OllamaEmbeddings, CohereEmbeddings
                "model_name",  # HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
                "model_id",  # BedrockEmbeddings
                "deployment",  # AzureOpenAIEmbeddings
                "endpoint",  # some custom embeddings
                "repo_id",  # HuggingFaceInferenceAPIEmbeddings
            ]

            for attr in model_attrs:
                if hasattr(model, attr):
                    value = getattr(model, attr)
                    if value is not None and isinstance(value, str):
                        LOGGER.info(f"Extracted model name from attribute '{attr}': {value}")
                        return value

            # If no model name found, try to get from class name
            class_name = model.__class__.__name__
            LOGGER.warning(
                f"Could not extract model name from {class_name}, using class name as fallback"
            )
            return f"langchain/{class_name}"

        return "unknown"

    def _load_default_model(self) -> Any:
        """
        Load the default embedding model based on model_type.

        Returns
        -------
        Any
            Loaded embedding model

        Raises
        ------
        ImportError
            If required library is not installed
        ValueError
            If model_type is not supported
        """
        if self.model_type == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer

                LOGGER.info(f"Loading sentence-transformers model: {self.model_name}")
                return SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Install with: pip install sentence-transformers"
                )
        elif self.model_type == "langchain":
            raise ValueError(
                "Cannot load default LangChain model. "
                "Please provide a pre-initialized LangChain embedding model."
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def _generate_embeddings(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
        normalize_embeddings: bool = False,
    ) -> np.ndarray:
        """
        Generate embeddings using the appropriate model type.

        Parameters
        ----------
        texts : List[str]
            List of texts to embed
        batch_size : int
            Batch size for embedding generation
        show_progress : bool
            Whether to show progress bar
        normalize_embeddings : bool
            Whether to L2-normalize the embeddings

        Returns
        -------
        np.ndarray
            Array of embeddings with shape (len(texts), embedding_dim)

        Raises
        ------
        ValueError
            If model_type is not supported
        """
        if self.model_type == "sentence-transformers":
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=normalize_embeddings,
            )
            return embeddings

        elif self.model_type == "langchain":
            # LangChain uses embed_documents which returns List[List[float]]
            if show_progress:
                try:
                    from tqdm import tqdm

                    # Process in batches with progress bar
                    all_embeddings = []
                    for i in tqdm(
                        range(0, len(texts), batch_size),
                        desc="Embedding",
                        disable=not show_progress,
                    ):
                        batch = texts[i : i + batch_size]
                        batch_embeddings = self.model.embed_documents(batch)
                        all_embeddings.extend(batch_embeddings)
                    embeddings = np.array(all_embeddings)
                except ImportError:
                    LOGGER.warning("tqdm not available, progress bar disabled")
                    embeddings = np.array(self.model.embed_documents(texts))
            else:
                embeddings = np.array(self.model.embed_documents(texts))

            # Apply normalization if requested
            if normalize_embeddings:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                # Add small epsilon to avoid division by zero
                embeddings = embeddings / (norms + 1e-8)

            return embeddings

        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

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

        >>> # With LangChain
        >>> from langchain_openai import OpenAIEmbeddings
        >>> model = OpenAIEmbeddings(model="text-embedding-3-small")
        >>> embedder = KeywordEmbedder(embedding_model=model)
        >>> embeddings = embedder.create_embeddings(str_list)
        """
        if not str_list:
            raise ValueError("String list cannot be empty")

        # Normalize strings and remove duplicates while preserving order
        normalized_strs = [normalize_string(s) for s in str_list]
        unique_strs = list(dict.fromkeys(normalized_strs))

        if len(unique_strs) < len(normalized_strs):
            LOGGER.warning(
                f"Removed {len(normalized_strs) - len(unique_strs)} duplicate strings after normalization"
            )

        LOGGER.info(
            f"Generating embeddings for {len(unique_strs)} strings "
            f"using {self.model_type} model: {self.model_name}"
        )

        # Generate embeddings using the unified interface
        embeddings = self._generate_embeddings(
            unique_strs,
            batch_size=batch_size,
            show_progress=show_progress,
            normalize_embeddings=normalize_embeddings,
        )

        # Create string -> embedding mapping
        string_embeddings = {
            string: embedding for string, embedding in zip(unique_strs, embeddings)
        }

        embedding_dim = embeddings.shape[1]
        LOGGER.info(
            f"Successfully generated embeddings with dimension {embedding_dim} "
            f"for {len(string_embeddings)} strings using {self.model_type}"
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
            for keyword, embedding in tqdm(
                keyword_embeddings.items(),
                desc="Converting embeddings",
                unit="embedding",
            )
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
            LOGGER.info(f"Saving embeddings to {output_path}")
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
            keyword: np.array(embedding) for keyword, embedding in data["embeddings"].items()
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

        Examples
        --------
        >>> embedder = KeywordEmbedder()
        >>> embeddings = embedder.process_csv_to_embeddings(
        ...     'suggestions.csv',
        ...     'embeddings.json',
        ...     min_confidence=0.7
        ... )
        """
        from ..utils.suggestion_processing import (
            aggregate_unique_terms,
            filter_by_field_value,
        )
        from .io import load_suggestions_from_csv

        # Step 1: Load suggestions from CSV
        LOGGER.info(f"Loading suggestions from {csv_path}")
        suggestions = load_suggestions_from_csv(csv_path, suggestion_column)

        # Step 2: Filter by confidence
        LOGGER.info(f"Filtering suggestions with min_confidence={min_confidence}")
        filtered_suggestions = filter_by_field_value(suggestions, min_confidence)

        # Step 3: Aggregate unique keywords
        LOGGER.info("Aggregating unique keywords")
        keywords = aggregate_unique_terms(
            filtered_suggestions,
            normalize=normalize_keywords,
        )[0]

        if not keywords:
            LOGGER.warning("No keywords found after filtering and aggregation")
            return {}

        LOGGER.info(f"Found {len(keywords)} unique keywords")

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
            "model_type": self.model_type,
        }
        self.save_embeddings(
            keyword_embeddings,
            output_path,
            model_name=self.model_name,
            metadata=metadata,
        )

        return keyword_embeddings
