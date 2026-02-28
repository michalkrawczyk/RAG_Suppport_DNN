"""
Embedding Generator for JASPER Steering Dataset Builder.

This module handles batch embedding generation for questions, sources, keywords,
and cluster centroids. Includes sanity checks for NaN/Inf values and centroid
similarity validation.

Key Features:
- Batch embedding generation for all text types
- Automatic model type detection (sentence-transformers or LangChain)
- Sanity checks: NaN/Inf detection, centroid similarity validation
- Progress tracking with tqdm
- Memory-efficient batch processing
- Outputs: PyTorch tensors saved as *.pt files
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from RAG_supporters.embeddings import TextEmbedder
from RAG_supporters.clustering_ops import ClusterParser
from RAG_supporters.DEFAULT_CONSTS import DEFAULT_COL_KEYS

LOGGER = logging.getLogger(__name__)

# Module-level aliases so frozen-dataclass values can be used as function defaults
_DEFAULT_QUESTION_COL: str = DEFAULT_COL_KEYS.question
_DEFAULT_SOURCE_COL: str = DEFAULT_COL_KEYS.source
_DEFAULT_KEYWORDS_COL: str = DEFAULT_COL_KEYS.keywords


class EmbeddingGenerator:
    """Generate and validate embeddings for dataset components.

    This class handles embedding generation for all dataset components:
    questions, sources, keywords, and cluster centroids. Performs sanity
    checks to ensure embedding quality before saving.

    Parameters
    ----------
    embedding_model : Any
        Embedding model (sentence-transformers or LangChain)
    cluster_parser : ClusterParser, optional
        Parser for cluster metadata (required for centroid validation)
    batch_size : int, optional
        Batch size for embedding generation (default: 32)
    show_progress : bool, optional
        Show progress bars (default: True)
    normalize_embeddings : bool, optional
        L2-normalize embeddings (default: False)

    Attributes
    ----------
    embedder : TextEmbedder
        Unified embedding interface
    cluster_parser : ClusterParser, optional
        Cluster metadata parser
    batch_size : int
        Batch size for processing
    show_progress : bool
        Whether to show progress bars
    normalize_embeddings : bool
        Whether to L2-normalize embeddings

    Examples
    --------
    >>> from sentence_transformers import SentenceTransformer
    >>> from RAG_supporters.dataset import EmbeddingGenerator, ClusterParser
    >>>
    >>> model = SentenceTransformer("all-MiniLM-L6-v2")
    >>> parser = ClusterParser("clusters.json")
    >>> generator = EmbeddingGenerator(model, parser)
    >>>
    >>> # Generate question embeddings
    >>> questions = ["What is Python?", "What is Java?"]
    >>> question_embs = generator.generate_text_embeddings(questions)
    >>> print(question_embs.shape)
    torch.Size([2, 384])
    """

    def __init__(
        self,
        embedding_model: Any,
        cluster_parser: Optional[ClusterParser] = None,
        batch_size: int = 32,
        show_progress: bool = True,
        normalize_embeddings: bool = False,
    ):
        """Initialize embedding generator."""
        # Wrap model in TextEmbedder for unified interface
        self.embedder = TextEmbedder(embedding_model=embedding_model)
        self.cluster_parser = cluster_parser
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.normalize_embeddings = normalize_embeddings

        LOGGER.info(
            f"Initialized EmbeddingGenerator with {self.embedder.model_type} model: "
            f"{self.embedder.model_name}"
        )

    def _check_for_invalid_values(
        self, embeddings: np.ndarray, text_type: str
    ) -> Tuple[bool, List[str]]:
        """Check embeddings for NaN or Inf values.

        Parameters
        ----------
        embeddings : np.ndarray
            Embeddings array to check
        text_type : str
            Type of embeddings (for error messages)

        Returns
        -------
        Tuple[bool, List[str]]
            (is_valid, list_of_errors)
        """
        errors = []

        # Check for NaN
        if np.isnan(embeddings).any():
            n_nan = np.isnan(embeddings).sum()
            errors.append(f"Found {n_nan} NaN values in {text_type} embeddings")

        # Check for Inf
        if np.isinf(embeddings).any():
            n_inf = np.isinf(embeddings).sum()
            errors.append(f"Found {n_inf} Inf values in {text_type} embeddings")

        # Check for all-zero embeddings
        zero_mask = np.all(embeddings == 0, axis=1)
        if zero_mask.any():
            n_zero = zero_mask.sum()
            errors.append(f"Found {n_zero} all-zero embeddings in {text_type}")

        return len(errors) == 0, errors

    def _validate_centroid_similarity(
        self,
        keyword_embeddings: Dict[str, np.ndarray],
        centroid_embeddings: np.ndarray,
        min_similarity: float = 0.3,
    ) -> Tuple[bool, List[str]]:
        """Validate that centroids are similar to their cluster keywords.

        For each cluster, computes cosine similarity between centroid and
        keywords in that cluster. Checks that average similarity is above
        threshold.

        Parameters
        ----------
        keyword_embeddings : Dict[str, np.ndarray]
            Mapping from keyword strings to embeddings
        centroid_embeddings : np.ndarray
            Cluster centroid embeddings [n_clusters, dim]
        min_similarity : float, optional
            Minimum average similarity required (default: 0.3)

        Returns
        -------
        Tuple[bool, List[str]]
            (is_valid, list_of_warnings)
        """
        if self.cluster_parser is None:
            LOGGER.warning("No cluster parser provided, skipping centroid validation")
            return True, []

        warnings = []

        for cluster_id in range(self.cluster_parser.n_clusters):
            # Get keywords in this cluster
            cluster_keywords = self.cluster_parser.get_cluster_keywords(cluster_id)
            if not cluster_keywords:
                warnings.append(f"Cluster {cluster_id} has no keywords")
                continue

            # Get keyword embeddings for this cluster
            cluster_embs = []
            for keyword in cluster_keywords:
                if keyword in keyword_embeddings:
                    cluster_embs.append(keyword_embeddings[keyword])

            if not cluster_embs:
                warnings.append(
                    f"Cluster {cluster_id}: No keyword embeddings found "
                    f"for keywords {cluster_keywords[:3]}..."
                )
                continue

            cluster_embs = np.array(cluster_embs)
            centroid = centroid_embeddings[cluster_id]

            # Compute cosine similarity
            # Normalize vectors
            cluster_embs_norm = cluster_embs / (
                np.linalg.norm(cluster_embs, axis=1, keepdims=True) + 1e-8
            )
            centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-8)

            similarities = cluster_embs_norm @ centroid_norm
            avg_similarity = similarities.mean()

            if avg_similarity < min_similarity:
                warnings.append(
                    f"Cluster {cluster_id}: Low centroid similarity "
                    f"(avg={avg_similarity:.3f}, min={min_similarity})"
                )

        return len(warnings) == 0, warnings

    def generate_text_embeddings(
        self, texts: List[str], text_type: str = "text", validate: bool = True
    ) -> torch.Tensor:
        """Generate embeddings for a list of texts.

        Parameters
        ----------
        texts : List[str]
            List of texts to embed
        text_type : str, optional
            Type of text (for logging and validation)
        validate : bool, optional
            Perform sanity checks on embeddings (default: True)

        Returns
        -------
        torch.Tensor
            Embeddings tensor [n_texts, embedding_dim]

        Raises
        ------
        ValueError
            If embeddings contain NaN or Inf values

        Examples
        --------
        >>> generator = EmbeddingGenerator(model)
        >>> texts = ["Sample text 1", "Sample text 2"]
        >>> embeddings = generator.generate_text_embeddings(texts, text_type="source")
        >>> print(embeddings.shape)
        torch.Size([2, 384])
        """
        if not texts:
            raise ValueError(f"Cannot generate embeddings for empty {text_type} list")

        LOGGER.info(f"Generating embeddings for {len(texts)} {text_type}(s)")

        # Generate embeddings using TextEmbedder
        embeddings = self.embedder._generate_embeddings(
            texts=texts,
            batch_size=self.batch_size,
            show_progress=self.show_progress,
            normalize_embeddings=self.normalize_embeddings,
        )

        # Validate embeddings
        if validate:
            is_valid, errors = self._check_for_invalid_values(embeddings, text_type)
            if not is_valid:
                error_msg = f"Invalid {text_type} embeddings: " + "; ".join(errors)
                LOGGER.error(error_msg)
                raise ValueError(error_msg)

        # Convert to PyTorch tensor
        embeddings_tensor = torch.from_numpy(embeddings).float()

        LOGGER.info(
            f"Generated {text_type} embeddings: "
            f"shape={embeddings_tensor.shape}, "
            f"dtype={embeddings_tensor.dtype}"
        )

        return embeddings_tensor

    def _resolve_keyword_embeddings(
        self,
        df: pd.DataFrame,
        keywords_col: str,
        validate: bool,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, int], str]:
        """Resolve keyword embeddings from the best available source.

        Priority:
        1. Pre-computed embeddings in ``cluster_parser.keyword_embeddings``
           (from the ``"embeddings"`` key in the cluster JSON — only present
           when :meth:`KeywordClusterer.save_results` was called with
           ``include_embeddings=True``).  Re-used directly; no model call.
        2. ``cluster_parser.get_all_keywords()`` — curated vocabulary from the
           ``"cluster_assignments"`` key.  Embeddings are generated via the
           embedding model.
        3. Fallback when no ``cluster_parser`` is set — scan ``df[keywords_col]``
           as the vocabulary (original behaviour).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with a keywords column (used only for the CSV fallback).
        keywords_col : str
            Name of the keywords column in *df*.
        validate : bool
            Whether to run sanity checks on freshly generated embeddings.

        Returns
        -------
        Tuple[Optional[torch.Tensor], Dict[str, int], str]
            ``(keyword_embs, keyword_to_id, source_description)``
            - ``keyword_embs``: ``[n_keywords, dim]`` tensor, or ``None`` when
              no keywords are available.
            - ``keyword_to_id``: mapping from keyword string to row index.
            - ``source_description``: human-readable string for logging.
        """
        # --- Priority 1: pre-computed embeddings from cluster JSON ----------
        if self.cluster_parser is not None and self.cluster_parser.keyword_embeddings is not None:
            precomputed = self.cluster_parser.keyword_embeddings
            if not precomputed:
                return None, {}, "cluster JSON embeddings (empty)"
            sorted_kws = sorted(precomputed)
            keyword_to_id = {kw: i for i, kw in enumerate(sorted_kws)}
            stacked = np.stack([precomputed[kw] for kw in sorted_kws], axis=0).astype(np.float32)
            keyword_embs = torch.from_numpy(stacked)
            source = f"cluster JSON embeddings ({len(keyword_to_id)} keywords, no model call)"
            LOGGER.info(
                "Using %d pre-computed keyword embeddings from cluster JSON.", len(sorted_kws)
            )
            return keyword_embs, keyword_to_id, source

        # --- Priority 2: cluster JSON vocabulary (no pre-computed embeddings) -
        if self.cluster_parser is not None:
            unique_keywords = self.cluster_parser.get_all_keywords()
            source = f"cluster JSON assignments ({len(unique_keywords)} keywords)"
            LOGGER.info(
                "No pre-computed embeddings in cluster JSON; embedding %d keywords "
                "from cluster_assignments.",
                len(unique_keywords),
            )
        else:
            # --- Priority 3: scan the CSV column ----------------------------
            all_keywords: set = set()
            for kws in df[keywords_col]:
                if isinstance(kws, list):
                    all_keywords.update(kws)
                elif isinstance(kws, str):
                    all_keywords.update([kw.strip() for kw in kws.split(",")])
            unique_keywords = sorted(all_keywords)
            source = f"CSV column '{keywords_col}' ({len(unique_keywords)} keywords)"

        if not unique_keywords:
            return None, {}, source

        keyword_embs, keyword_to_id = self.generate_keyword_embeddings(
            unique_keywords, validate=validate
        )
        return keyword_embs, keyword_to_id, source

    def generate_keyword_embeddings(
        self, keywords: List[str], validate: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, int]]:
        """Generate embeddings for keywords.

        Parameters
        ----------
        keywords : List[str]
            List of unique keywords
        validate : bool, optional
            Perform sanity checks (default: True)

        Returns
        -------
        Tuple[torch.Tensor, Dict[str, int]]
            (embeddings_tensor, keyword_to_id_mapping)
            - embeddings_tensor: [n_keywords, embedding_dim]
            - keyword_to_id_mapping: Maps keyword string to index

        Examples
        --------
        >>> generator = EmbeddingGenerator(model)
        >>> keywords = ["python", "java", "programming"]
        >>> embs, kw_map = generator.generate_keyword_embeddings(keywords)
        >>> print(embs.shape, len(kw_map))
        torch.Size([3, 384]) 3
        """
        LOGGER.info(f"Generating embeddings for {len(keywords)} keywords")

        # Create keyword-to-ID mapping
        keyword_to_id = {kw: i for i, kw in enumerate(keywords)}

        # Generate embeddings
        embeddings_tensor = self.generate_text_embeddings(
            texts=keywords, text_type="keyword", validate=validate
        )

        return embeddings_tensor, keyword_to_id

    def generate_centroid_embeddings(
        self,
        keyword_embeddings_dict: Optional[Dict[str, np.ndarray]] = None,
        validate: bool = True,
        min_similarity: float = 0.3,
    ) -> torch.Tensor:
        """Generate or load cluster centroid embeddings.

        If cluster_parser is provided, loads precomputed centroids from
        KeywordClusterer JSON and validates them against keyword embeddings.

        Parameters
        ----------
        keyword_embeddings_dict : Dict[str, np.ndarray], optional
            Keyword embeddings for validation
        validate : bool, optional
            Perform sanity checks (default: True)
        min_similarity : float, optional
            Minimum average similarity for centroid validation (default: 0.3)

        Returns
        -------
        torch.Tensor
            Centroid embeddings [n_clusters, embedding_dim]

        Raises
        ------
        ValueError
            If cluster_parser is None or centroids are invalid

        Examples
        --------
        >>> generator = EmbeddingGenerator(model, cluster_parser)
        >>> centroids = generator.generate_centroid_embeddings()
        >>> print(centroids.shape)
        torch.Size([20, 384])
        """
        if self.cluster_parser is None:
            raise ValueError(
                "ClusterParser required for centroid embeddings. "
                "Provide cluster_parser during initialization."
            )

        LOGGER.info(f"Loading centroid embeddings for {self.cluster_parser.n_clusters} clusters")

        # Get centroids from cluster parser
        centroids = self.cluster_parser.get_all_centroids()

        if centroids is None or len(centroids) == 0:
            raise ValueError(
                "No centroids found in cluster JSON. "
                "Ensure KeywordClusterer saved centroid embeddings."
            )

        # Validate centroids
        if validate:
            is_valid, errors = self._check_for_invalid_values(centroids, "centroid")
            if not is_valid:
                error_msg = "Invalid centroid embeddings: " + "; ".join(errors)
                LOGGER.error(error_msg)
                raise ValueError(error_msg)

            # Validate similarity to cluster keywords
            if keyword_embeddings_dict is not None:
                is_similar, warnings = self._validate_centroid_similarity(
                    keyword_embeddings_dict, centroids, min_similarity
                )

                for warning in warnings:
                    LOGGER.warning(warning)

                if not is_similar:
                    LOGGER.warning(
                        "Some centroids have low similarity to their keywords. "
                        "This may indicate clustering quality issues."
                    )

        # Convert to PyTorch tensor
        centroids_tensor = torch.from_numpy(centroids).float()

        LOGGER.info(
            f"Loaded centroid embeddings: "
            f"shape={centroids_tensor.shape}, "
            f"dtype={centroids_tensor.dtype}"
        )

        return centroids_tensor

    def generate_all_embeddings(
        self,
        df: pd.DataFrame,
        question_col: str = _DEFAULT_QUESTION_COL,
        source_col: str = _DEFAULT_SOURCE_COL,
        keywords_col: str = _DEFAULT_KEYWORDS_COL,
        validate: bool = True,
        on_empty_keywords: str = "none",
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Generate embeddings for all dataset components.

        Generates embeddings for:
        - Unique questions
        - Unique sources
        - Unique keywords
        - Cluster centroids (if cluster_parser provided)

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with question-source pairs
        question_col : str, optional
            Column name for questions (default: "question")
        source_col : str, optional
            Column name for sources (default: "source")
        keywords_col : str, optional
            Column name for keywords (default: "keywords")
        validate : bool, optional
            Perform sanity checks (default: True)
        on_empty_keywords : str, optional
            Behaviour when the dataset contains no keywords.
            - ``"none"`` (default): set ``keyword_embs`` to ``None`` and
              ``keyword_to_id`` to ``{}``; log a warning.
            - ``"error"``: raise ``ValueError``.

        Returns
        -------
        Dict[str, Optional[torch.Tensor]]
            Dictionary with keys:
            - "question_embs": [n_questions, dim]
            - "source_embs": [n_sources, dim]
            - "keyword_embs": [n_keywords, dim], or ``None`` when absent
            - "centroid_embs": [n_clusters, dim] (if cluster_parser provided
              and keywords present)
            - "question_to_id": Dict[str, int]
            - "source_to_id": Dict[str, int]
            - "keyword_to_id": Dict[str, int]

        Examples
        --------
        >>> generator = EmbeddingGenerator(model, cluster_parser)
        >>> df = pd.read_csv("merged.csv")
        >>> embeddings = generator.generate_all_embeddings(df)
        >>> print(embeddings.keys())
        dict_keys(['question_embs', 'source_embs', 'keyword_embs', 'centroid_embs', ...])
        """
        LOGGER.info("Generating embeddings for all dataset components")

        # Extract unique texts
        unique_questions = df[question_col].unique().tolist()
        unique_sources = df[source_col].unique().tolist()

        # Resolve keyword embeddings (cluster JSON → cluster assignments → CSV fallback)
        keyword_embs, keyword_to_id, keyword_source = self._resolve_keyword_embeddings(
            df=df, keywords_col=keywords_col, validate=validate
        )

        LOGGER.info(
            "Dataset contains: %d questions, %d sources, keywords from %s",
            len(unique_questions),
            len(unique_sources),
            keyword_source,
        )

        # Generate embeddings
        result = {}

        # Questions
        question_embs = self.generate_text_embeddings(
            unique_questions, text_type="question", validate=validate
        )
        result["question_embs"] = question_embs
        result["question_to_id"] = {q: i for i, q in enumerate(unique_questions)}

        # Sources
        source_embs = self.generate_text_embeddings(
            unique_sources, text_type="source", validate=validate
        )
        result["source_embs"] = source_embs
        result["source_to_id"] = {s: i for i, s in enumerate(unique_sources)}

        # Keywords
        if keyword_embs is not None:
            result["keyword_embs"] = keyword_embs
            result["keyword_to_id"] = keyword_to_id
        else:
            valid_modes = {"none", "error"}
            if on_empty_keywords not in valid_modes:
                raise ValueError(
                    f"on_empty_keywords must be one of {valid_modes!r}, "
                    f"got {on_empty_keywords!r}"
                )
            if on_empty_keywords == "error":
                raise ValueError(
                    "Dataset contains no keywords. "
                    "Set on_empty_keywords='none' to proceed without keyword embeddings."
                )
            LOGGER.warning(
                "No keywords found in the dataset; keyword_embs will be None and "
                "keyword_to_id will be {}. Downstream keyword-weighted steering will "
                "use the fallback strategy for every row."
            )
            result["keyword_embs"] = None
            result["keyword_to_id"] = {}

        # Centroids (if cluster parser provided and keywords are present)
        if self.cluster_parser is not None and keyword_embs is not None:
            # Create keyword embeddings dict for validation
            keyword_embs_dict = {kw: keyword_embs[idx].numpy() for kw, idx in keyword_to_id.items()}

            centroid_embs = self.generate_centroid_embeddings(
                keyword_embeddings_dict=keyword_embs_dict, validate=validate
            )
            result["centroid_embs"] = centroid_embs

        LOGGER.info("Successfully generated all embeddings")

        return result

    def save_embeddings(
        self,
        embeddings_dict: Dict[str, torch.Tensor],
        output_dir: Union[str, Path],
        prefix: str = "",
    ) -> None:
        """Save embeddings to PyTorch tensor files.

        Parameters
        ----------
        embeddings_dict : Dict[str, torch.Tensor]
            Dictionary of embeddings to save
        output_dir : str or Path
            Output directory for tensor files
        prefix : str, optional
            Prefix for output files (default: "")

        Examples
        --------
        >>> generator = EmbeddingGenerator(model)
        >>> embeddings = generator.generate_all_embeddings(df)
        >>> generator.save_embeddings(embeddings, "output/")
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Define file names (skip non-tensor items like mappings)
        tensor_keys = ["question_embs", "source_embs", "keyword_embs", "centroid_embs"]

        # Infer embedding dimension from any present tensor for None fallbacks
        embed_dim: Optional[int] = None
        for _k in ("question_embs", "source_embs"):
            _t = embeddings_dict.get(_k)
            if isinstance(_t, torch.Tensor) and _t.ndim == 2:
                embed_dim = _t.shape[1]
                break

        for key in tensor_keys:
            if key not in embeddings_dict:
                continue
            tensor = embeddings_dict[key]
            if tensor is None:
                # Save a zero-row placeholder so loaders can always find the file
                if embed_dim is not None:
                    tensor = torch.zeros(0, embed_dim, dtype=torch.float32)
                    LOGGER.info(
                        f"Saving placeholder for absent {key} "
                        f"(shape={tensor.shape}) to {output_dir}"
                    )
                else:
                    LOGGER.warning(f"Cannot infer embedding dim; skipping {key}.pt")
                    continue
            file_name = f"{prefix}{key}.pt" if prefix else f"{key}.pt"
            file_path = output_dir / file_name

            torch.save(tensor, file_path)
            LOGGER.info(f"Saved {key}: shape={tensor.shape} to {file_path}")


def generate_embeddings(
    df: pd.DataFrame,
    embedding_model: Any,
    cluster_parser: Optional[ClusterParser] = None,
    output_dir: Optional[Union[str, Path]] = None,
    question_col: str = "question",
    source_col: str = "source",
    keywords_col: str = "keywords",
    batch_size: int = 32,
    normalize_embeddings: bool = False,
    validate: bool = True,
    on_empty_keywords: str = "none",
) -> Dict[str, Optional[torch.Tensor]]:
    """Convenience function to generate and optionally save embeddings.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with question-source pairs
    embedding_model : Any
        Embedding model (sentence-transformers or LangChain)
    cluster_parser : ClusterParser, optional
        Parser for cluster metadata
    output_dir : str or Path, optional
        If provided, save embeddings to this directory
    question_col : str, optional
        Column name for questions (default: "question")
    source_col : str, optional
        Column name for sources (default: "source")
    keywords_col : str, optional
        Column name for keywords (default: "keywords")
    batch_size : int, optional
        Batch size for embedding generation (default: 32)
    normalize_embeddings : bool, optional
        L2-normalize embeddings (default: False)
    validate : bool, optional
        Perform sanity checks (default: True)
    on_empty_keywords : str, optional
        Behaviour when the dataset contains no keywords.
        ``"none"`` (default) sets ``keyword_embs=None`` and warns;
        ``"error"`` raises ``ValueError``.

    Returns
    -------
    Dict[str, Optional[torch.Tensor]]
        Dictionary of embeddings and mappings. ``keyword_embs`` may be
        ``None`` when no keywords are present and ``on_empty_keywords='none'``.

    Examples
    --------
    >>> from sentence_transformers import SentenceTransformer
    >>> from RAG_supporters.dataset import generate_embeddings, ClusterParser
    >>>
    >>> model = SentenceTransformer("all-MiniLM-L6-v2")
    >>> parser = ClusterParser("clusters.json")
    >>> df = pd.read_csv("merged.csv")
    >>>
    >>> embeddings = generate_embeddings(
    ...     df=df,
    ...     embedding_model=model,
    ...     cluster_parser=parser,
    ...     output_dir="output/"
    ... )
    """
    generator = EmbeddingGenerator(
        embedding_model=embedding_model,
        cluster_parser=cluster_parser,
        batch_size=batch_size,
        show_progress=True,
        normalize_embeddings=normalize_embeddings,
    )

    embeddings = generator.generate_all_embeddings(
        df=df,
        question_col=question_col,
        source_col=source_col,
        keywords_col=keywords_col,
        validate=validate,
        on_empty_keywords=on_empty_keywords,
    )

    # Save if output directory provided
    if output_dir is not None:
        generator.save_embeddings(embeddings, output_dir)

    return embeddings
