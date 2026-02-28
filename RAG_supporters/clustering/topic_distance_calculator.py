"""
Utility function for calculating embedding distances to topic keywords.

This module provides functionality to process CSV files with question_text and source_text fields,
calculating embedding distances to topic keywords from KeywordClusterer result JSON.
Unlike TOPIC_RELEVANCE_PROB, this does not use LLM but directly computes embedding distances.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from tqdm import tqdm

from RAG_supporters.utils.text_utils import normalize_string

LOGGER = logging.getLogger(__name__)


class TopicDistanceCalculator:
    """
    Calculate embedding distances between text and topic keywords.

    This class processes CSV files containing question_text and source_text fields,
    computing distances to topic keywords from KeywordClusterer JSON files.
    Supports fetching embeddings from database by ID or using provided embeddings.
    """

    def __init__(
        self,
        keyword_clusterer_json: Union[str, Path, Dict],
        embedder: Optional[Any] = None,
        metric: str = "cosine",
        enable_cache: bool = True,
    ):
        """
        Initialize the topic distance calculator.

        Parameters
        ----------
        keyword_clusterer_json : Union[str, Path, Dict]
            Path to KeywordClusterer JSON file or loaded dict with centroids and topic_descriptors
        embedder : Optional[Any]
            TextEmbedder instance for encoding text.
            Required if text needs to be embedded (not fetching from database).
            If not a TextEmbedder, will be wrapped automatically.
        metric : str
            Distance metric: "cosine" or "euclidean". Default is "cosine".
        enable_cache : bool
            Whether to cache embeddings for repeated texts. Default is True.
            Disable for memory-constrained environments or unique texts.
        """
        # Wrap embedder in TextEmbedder if necessary
        if embedder is not None:
            from RAG_supporters.embeddings.text_embedder import TextEmbedder

            if not isinstance(embedder, TextEmbedder):
                LOGGER.info("Wrapping provided embedder in TextEmbedder")
                embedder = TextEmbedder(embedding_model=embedder)
        self.embedder = embedder
        self.metric = metric
        self._enable_cache = enable_cache
        self._embedding_cache = {}  # Cache for text embeddings to avoid re-embedding same texts

        if metric not in ["cosine", "euclidean"]:
            raise ValueError(f"Invalid metric: {metric}. Choose 'cosine' or 'euclidean'")

        # Load KeywordClusterer data
        if isinstance(keyword_clusterer_json, dict):
            self.clusterer_data = keyword_clusterer_json
        else:
            with open(keyword_clusterer_json, "r", encoding="utf-8") as f:
                self.clusterer_data = json.load(f)

        # Extract centroids and topic descriptors
        self._load_centroids_and_topics()

    def clear_cache(self) -> int:
        """
        Clear the embedding cache.

        Returns
        -------
        int
            Number of cached entries that were cleared
        """
        cache_size = len(self._embedding_cache)
        self._embedding_cache.clear()
        LOGGER.info(f"Cleared {cache_size} cached embeddings")
        return cache_size

    @property
    def cache_size(self) -> int:
        """
        Get the current size of the embedding cache.

        Returns
        -------
        int
            Number of cached embeddings
        """
        return len(self._embedding_cache)

    @property
    def cache_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the embedding cache.

        Returns
        -------
        Dict[str, Any]
            Dictionary with cache statistics including size and enabled status
        """
        cache_keys = list(self._embedding_cache.keys())
        if len(self._embedding_cache) <= 5:
            texts_preview = cache_keys[:5]
        else:
            texts_preview = cache_keys[:5] + ["..."]

        return {
            "enabled": self._enable_cache,
            "size": len(self._embedding_cache),
            "texts": texts_preview,
        }

    def _load_centroids_and_topics(self):
        """Load centroids and topic descriptors from KeywordClusterer data."""
        if "centroids" not in self.clusterer_data:
            raise ValueError(
                "KeywordClusterer JSON missing 'centroids' field. "
                "Ensure the JSON was saved with include_embeddings=True or contains centroid information."
            )

        self.centroids = np.array(self.clusterer_data["centroids"])
        LOGGER.info(
            f"Loaded {len(self.centroids)} centroids with dimension {self.centroids.shape[1]}"
        )

        # Extract topic descriptors from cluster_stats
        self.topic_descriptors = {}
        self.all_topic_descriptors = []  # Flat list of all topic descriptors
        if "cluster_stats" in self.clusterer_data:
            for cluster_id, stats in self.clusterer_data["cluster_stats"].items():
                if "topic_descriptors" in stats:
                    self.topic_descriptors[int(cluster_id)] = stats["topic_descriptors"]
                    self.all_topic_descriptors.extend(stats["topic_descriptors"])
            LOGGER.info(f"Loaded topic descriptors for {len(self.topic_descriptors)} clusters")
            LOGGER.info(f"Total unique topic descriptors: {len(set(self.all_topic_descriptors))}")
        else:
            LOGGER.warning("No cluster_stats found in KeywordClusterer JSON")

    def _create_distance_json_mapping(self, distances: np.ndarray) -> str:
        """
        Create JSON mapping of topic descriptors to their distances.

        For each cluster, assigns the cluster's distance to all its topic descriptors.
        This creates a {topic_descriptor: distance} mapping similar to the format
        used by DomainAssessmentAgent's question_term_relevance_scores.

        Note: If a topic descriptor appears in multiple clusters, the minimum distance
        is retained (closest cluster wins).

        Parameters
        ----------
        distances : np.ndarray
            Array of distances to each cluster centroid, shape (n_clusters,)

        Returns
        -------
        str
            JSON string mapping topic_descriptor to distance value
        """
        # Validate dimensions
        if len(distances) != len(self.centroids):
            raise ValueError(
                f"Distance array length ({len(distances)}) does not match "
                f"number of centroids ({len(self.centroids)})"
            )

        distance_mapping = {}

        # Iterate through all clusters and their topic descriptors
        for cluster_id, distance in enumerate(distances):
            if cluster_id in self.topic_descriptors:
                # Assign this cluster's distance to all its topic descriptors
                for topic_descriptor in self.topic_descriptors[cluster_id]:
                    # If descriptor appears in multiple clusters, keep minimum distance
                    if topic_descriptor in distance_mapping:
                        distance_mapping[topic_descriptor] = min(
                            distance_mapping[topic_descriptor], float(distance)
                        )
                    else:
                        distance_mapping[topic_descriptor] = float(distance)

        return json.dumps(distance_mapping)

    def _compute_distances_to_centroids(self, embedding: np.ndarray) -> np.ndarray:
        """
        Compute distances from embedding to all centroids.

        Parameters
        ----------
        embedding : np.ndarray
            Input embedding vector, shape (embedding_dim,)

        Returns
        -------
        np.ndarray
            Distances to each centroid, shape (n_clusters,)
        """
        # Ensure embedding is 2D for sklearn functions
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        if self.metric == "cosine":
            # Cosine similarity returns values in [-1, 1], higher is more similar
            # Convert to distance: distance = 1 - similarity
            similarities = cosine_similarity(embedding, self.centroids)[0]
            distances = 1 - similarities
        else:  # euclidean
            distances = euclidean_distances(embedding, self.centroids)[0]

        return distances

    def _get_embedding_from_text(self, text: str) -> np.ndarray:
        """
        Get embedding for text using TextEmbedder.

        Uses caching to avoid re-embedding the same text multiple times.

        Parameters
        ----------
        text : str
            Text to embed

        Returns
        -------
        np.ndarray
            Embedding vector
        """
        if self.embedder is None:
            raise ValueError(
                "Embedder is required to embed text. "
                "Provide a TextEmbedder during initialization or use database IDs."
            )

        # Check cache first (if enabled)
        if self._enable_cache and text in self._embedding_cache:
            return self._embedding_cache[text]

        # Use TextEmbedder interface
        embeddings_dict = self.embedder.create_embeddings([text])
        normalized_text = normalize_string(text)
        embedding = embeddings_dict[normalized_text]

        # Cache the result (if enabled)
        if self._enable_cache:
            self._embedding_cache[text] = embedding

        return embedding

    def _get_embeddings_batch(self, texts: list) -> Dict[str, np.ndarray]:
        """
        Get embeddings for multiple texts in batch using TextEmbedder.

        Uses caching and batch processing for efficiency.

        Parameters
        ----------
        texts : list
            List of texts to embed

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary mapping original text to embedding vector
        """
        if self.embedder is None:
            raise ValueError(
                "Embedder is required to embed text. "
                "Provide a TextEmbedder during initialization or use database IDs."
            )

        result = {}
        texts_to_embed = []
        text_indices = []  # Track which texts need embedding

        # Check cache for each text
        for i, text in enumerate(texts):
            if self._enable_cache and text in self._embedding_cache:
                result[text] = self._embedding_cache[text]
            else:
                texts_to_embed.append(text)
                text_indices.append(i)

        # Batch embed uncached texts
        if texts_to_embed:
            embeddings_dict = self.embedder.create_embeddings(texts_to_embed)

            # Map back to original texts and cache
            for i, text in enumerate(texts_to_embed):
                normalized_text = normalize_string(text)
                embedding = embeddings_dict[normalized_text]
                result[text] = embedding

                # Cache the result (if enabled)
                if self._enable_cache:
                    self._embedding_cache[text] = embedding

        return result

    def _get_embedding_from_database(
        self, item_id: str, database: Any, collection_name: str = "questions"
    ) -> Optional[np.ndarray]:
        """
        Fetch embedding from database by ID.

        Parameters
        ----------
        item_id : str
            ID of the item in the database
        database : Any
            Database object with get_embedding method or similar interface
        collection_name : str
            Collection name ("questions" or "sources")

        Returns
        -------
        Optional[np.ndarray]
            Embedding vector or None if not found
        """
        # Try different database interfaces
        embedding = None
        if hasattr(database, "get_embedding"):
            embedding = database.get_embedding(item_id, collection=collection_name)
        elif hasattr(database, "get"):
            # ChromaDB interface
            result = database.get(ids=[item_id], include=["embeddings"])
            if result and "embeddings" in result and result["embeddings"]:
                embedding = np.array(result["embeddings"][0])
        else:
            raise ValueError(
                "Database must have 'get_embedding' or 'get' method with embeddings support"
            )

        if embedding is None:
            LOGGER.warning(
                f"Embedding not found in database for ID '{item_id}' in collection '{collection_name}'"
            )

        return embedding

    def calculate_distances_for_csv(
        self,
        csv_path: Union[str, Path],
        question_col: str = "question_text",
        source_col: str = "source_text",
        question_id_col: Optional[str] = None,
        source_id_col: Optional[str] = None,
        database: Optional[Any] = None,
        output_path: Optional[Union[str, Path]] = None,
        show_progress: bool = True,
        save_on_interrupt: bool = True,
    ) -> pd.DataFrame:
        """
        Calculate embedding distances to topic keywords for CSV file.

        Parameters
        ----------
        csv_path : Union[str, Path]
            Path to input CSV file with question_text and source_text columns
        question_col : str
            Column name for question text. Default is "question_text".
        source_col : str
            Column name for source text. Default is "source_text".
        question_id_col : Optional[str]
            Column name for question IDs (for database lookup). If provided, fetches embeddings from database.
        source_id_col : Optional[str]
            Column name for source IDs (for database lookup). If provided, fetches embeddings from database.
        database : Optional[Any]
            Database object for fetching embeddings by ID. Required if using ID columns.
        output_path : Optional[Union[str, Path]]
            Path to save output CSV. If None, doesn't save.
        show_progress : bool
            Show progress bar. Default is True.
        save_on_interrupt : bool
            If True, saves partial results when interrupted with KeyboardInterrupt.
            The whole original DataFrame is saved with distance scores filled only for
            processed rows (unprocessed rows have None/NaN).
            Results are saved to output_path (overwrites if exists).
            The function returns the partial DataFrame instead of raising the interrupt.
            Default is True.

        Returns
        -------
        pd.DataFrame
            DataFrame with added columns for distances to each topic cluster:
            - question_term_distance_scores: JSON mapping {topic_descriptor: distance} for question
            - source_term_distance_scores: JSON mapping {topic_descriptor: distance} for source
        """
        # Load CSV
        df = pd.read_csv(csv_path)
        LOGGER.info(f"Loaded CSV with {len(df)} rows")

        # Validate columns
        required_cols = []
        if question_id_col:
            required_cols.append(question_id_col)
        else:
            required_cols.append(question_col)
        if source_id_col:
            required_cols.append(source_id_col)
        else:
            required_cols.append(source_col)

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Initialize result columns
        df["question_term_distance_scores"] = None
        df["source_term_distance_scores"] = None

        # Initialize result lists for better performance
        question_distance_scores = []
        source_distance_scores = []

        # Track processing statistics
        successful_questions = 0
        successful_sources = 0
        skipped_questions = 0
        skipped_sources = 0

        # Batch process: collect all texts first, then generate embeddings in batch
        question_texts = []
        source_texts = []
        question_indices = []  # Track which rows have questions
        source_indices = []  # Track which rows have sources

        # Step 1: Collect all texts that need embedding
        LOGGER.info("Collecting texts for batch embedding...")
        for idx in range(len(df)):
            row = df.iloc[idx]

            # Collect question texts (if not using database IDs)
            if (
                not question_id_col
                or question_id_col not in df.columns
                or pd.isna(row.get(question_id_col))
            ):
                if question_col in df.columns and pd.notna(row[question_col]):
                    question_text = str(row[question_col])
                    # Only add if not already in cache
                    if not self._enable_cache or question_text not in self._embedding_cache:
                        if question_text not in question_texts:  # Avoid duplicates
                            question_texts.append(question_text)
                    question_indices.append((idx, question_text))

            # Collect source texts (if not using database IDs)
            if (
                not source_id_col
                or source_id_col not in df.columns
                or pd.isna(row.get(source_id_col))
            ):
                if source_col in df.columns and pd.notna(row[source_col]):
                    source_text = str(row[source_col])
                    # Only add if not already in cache
                    if not self._enable_cache or source_text not in self._embedding_cache:
                        if source_text not in source_texts:  # Avoid duplicates
                            source_texts.append(source_text)
                    source_indices.append((idx, source_text))

        # Step 2: Generate embeddings in batch
        question_embeddings_map = {}
        source_embeddings_map = {}

        if question_texts:
            LOGGER.info(f"Generating embeddings for {len(question_texts)} unique questions...")
            question_embeddings_map = self._get_embeddings_batch(question_texts)

        if source_texts:
            LOGGER.info(f"Generating embeddings for {len(source_texts)} unique sources...")
            source_embeddings_map = self._get_embeddings_batch(source_texts)

        # Step 3: Process each row with pre-computed embeddings
        iterator = range(len(df))
        if show_progress:
            iterator = tqdm(iterator, desc="Calculating topic distances", unit="row")

        interrupted = False
        last_processed_idx = -1

        try:
            for idx in iterator:
                row = df.iloc[idx]

                # Process question
                question_score = None
                if (
                    question_id_col
                    and question_id_col in df.columns
                    and pd.notna(row[question_id_col])
                ):
                    # Fetch from database
                    if database is None:
                        raise ValueError("Database required when using question_id_col")
                    question_embedding = self._get_embedding_from_database(
                        str(row[question_id_col]), database, "questions"
                    )
                elif question_col in df.columns and pd.notna(row[question_col]):
                    # Get from batch-generated embeddings or cache
                    question_text = str(row[question_col])
                    question_embedding = question_embeddings_map.get(question_text)
                    if question_embedding is None and self._enable_cache:
                        question_embedding = self._embedding_cache.get(question_text)
                else:
                    LOGGER.warning(f"Row {idx}: No valid question data")
                    question_embedding = None

                if question_embedding is not None:
                    question_distances = self._compute_distances_to_centroids(question_embedding)
                    # Create JSON mapping {topic: distance}
                    question_score = self._create_distance_json_mapping(question_distances)
                    successful_questions += 1
                else:
                    skipped_questions += 1

                question_distance_scores.append(question_score)

                # Process source
                source_score = None
                if source_id_col and source_id_col in df.columns and pd.notna(row[source_id_col]):
                    # Fetch from database
                    if database is None:
                        raise ValueError("Database required when using source_id_col")
                    source_embedding = self._get_embedding_from_database(
                        str(row[source_id_col]), database, "sources"
                    )
                elif source_col in df.columns and pd.notna(row[source_col]):
                    # Get from batch-generated embeddings or cache
                    source_text = str(row[source_col])
                    source_embedding = source_embeddings_map.get(source_text)
                    if source_embedding is None and self._enable_cache:
                        source_embedding = self._embedding_cache.get(source_text)
                else:
                    LOGGER.warning(f"Row {idx}: No valid source data")
                    source_embedding = None

                if source_embedding is not None:
                    source_distances = self._compute_distances_to_centroids(source_embedding)
                    # Create JSON mapping {topic: distance}
                    source_score = self._create_distance_json_mapping(source_distances)
                    successful_sources += 1
                else:
                    skipped_sources += 1

                source_distance_scores.append(source_score)
                last_processed_idx = idx

        except KeyboardInterrupt:
            interrupted = True
            LOGGER.warning(
                f"\nProcessing interrupted by user at row {last_processed_idx + 1}/{len(df)}"
            )

            # Keep the whole original DataFrame but only fill processed rows
            df_partial = df.copy()

            # Initialize columns with None for all rows
            df_partial["question_term_distance_scores"] = None
            df_partial["source_term_distance_scores"] = None

            # Fill in only the processed rows
            for i in range(last_processed_idx + 1):
                df_partial.at[df_partial.index[i], "question_term_distance_scores"] = (
                    question_distance_scores[i]
                )
                df_partial.at[df_partial.index[i], "source_term_distance_scores"] = (
                    source_distance_scores[i]
                )

            if save_on_interrupt and output_path:
                # Save to output path directly (overwrite)
                df_partial.to_csv(output_path, index=False)
                LOGGER.info(
                    f"Saved partial results ({last_processed_idx + 1}/{len(df)} rows processed) to {output_path}"
                )

            LOGGER.info(
                f"Partial processing summary - Questions: {successful_questions} processed, "
                f"{skipped_questions} skipped. Sources: {successful_sources} processed, "
                f"{skipped_sources} skipped."
            )

            # Return partial DataFrame instead of raising
            return df_partial

        # Assign results to dataframe columns
        df["question_term_distance_scores"] = question_distance_scores
        df["source_term_distance_scores"] = source_distance_scores

        # Log processing summary
        LOGGER.info(
            f"Processing complete: {len(df)} total rows. "
            f"Questions: {successful_questions} processed, {skipped_questions} skipped. "
            f"Sources: {successful_sources} processed, {skipped_sources} skipped."
        )

        # Save if output path provided
        if output_path:
            df.to_csv(output_path, index=False)
            LOGGER.info(f"Saved results to {output_path}")

        return df


def calculate_topic_distances_from_csv(
    csv_path: Union[str, Path],
    keyword_clusterer_json: Union[str, Path, Dict],
    embedder: Optional[Any] = None,
    question_col: str = "question_text",
    source_col: str = "source_text",
    question_id_col: Optional[str] = None,
    source_id_col: Optional[str] = None,
    database: Optional[Any] = None,
    metric: str = "cosine",
    enable_cache: bool = True,
    output_path: Optional[Union[str, Path]] = None,
    show_progress: bool = True,
    save_on_interrupt: bool = True,
) -> pd.DataFrame:
    """
    Calculate topic distances from CSV file.

    This function processes a CSV file with question_text and source_text fields,
    calculating embedding distances to topic keywords from KeywordClusterer JSON.
    Unlike TOPIC_RELEVANCE_PROB agent method, this uses direct embedding distance
    calculation without LLM.

    Parameters
    ----------
    csv_path : Union[str, Path]
        Path to input CSV file
    keyword_clusterer_json : Union[str, Path, Dict]
        Path to KeywordClusterer JSON file or loaded dict
    embedder : Optional[Any]
        TextEmbedder instance for encoding text. Required if not using database IDs.
    question_col : str
        Column name for question text. Default is "question_text".
    source_col : str
        Column name for source text. Default is "source_text".
    question_id_col : Optional[str]
        Column name for question IDs (for database lookup)
    source_id_col : Optional[str]
        Column name for source IDs (for database lookup)
    database : Optional[Any]
        Database object for fetching embeddings by ID
    metric : str
        Distance metric: "cosine" or "euclidean". Default is "cosine".
    enable_cache : bool
        Whether to cache embeddings for repeated texts. Default is True.
        Disable for memory-constrained environments or unique texts.
    output_path : Optional[Union[str, Path]]
        Path to save output CSV
    show_progress : bool
        Show progress bar. Default is True.
    save_on_interrupt : bool
        If True, saves partial results when interrupted with KeyboardInterrupt.
        The whole original DataFrame is saved with distance scores filled only for
        processed rows (unprocessed rows have None/NaN).
        Results are saved to output_path (overwrites if exists).
        The function returns the partial DataFrame instead of raising the interrupt.
        Default is True.

    Returns
    -------
    pd.DataFrame
        DataFrame with added distance columns:
        - question_term_distance_scores: JSON mapping {topic_descriptor: distance} for question
        - source_term_distance_scores: JSON mapping {topic_descriptor: distance} for source

    Examples
    --------
    >>> from RAG_supporters.embeddings import TextEmbedder
    >>> from RAG_supporters.clustering.topic_distance_calculator import (
    ...     calculate_topic_distances_from_csv,
    ... )
    >>>
    >>> # Using text embedding
    >>> embedder = TextEmbedder()
    >>> result_df = calculate_topic_distances_from_csv(
    ...     csv_path="data.csv",
    ...     keyword_clusterer_json="clusters.json",
    ...     embedder=embedder,
    ...     output_path="results.csv"
    ... )
    >>>
    >>> # Using database IDs
    >>> result_df = calculate_topic_distances_from_csv(
    ...     csv_path="data.csv",
    ...     keyword_clusterer_json="clusters.json",
    ...     question_id_col="question_id",
    ...     source_id_col="source_id",
    ...     database=my_database,
    ...     output_path="results.csv"
    ... )
    """
    calculator = TopicDistanceCalculator(
        keyword_clusterer_json=keyword_clusterer_json,
        embedder=embedder,
        metric=metric,
        enable_cache=enable_cache,
    )

    return calculator.calculate_distances_for_csv(
        csv_path=csv_path,
        question_col=question_col,
        source_col=source_col,
        question_id_col=question_id_col,
        source_id_col=source_id_col,
        database=database,
        output_path=output_path,
        show_progress=show_progress,
        save_on_interrupt=save_on_interrupt,
    )
