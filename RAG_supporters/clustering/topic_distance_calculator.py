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
    ):
        """
        Initialize the topic distance calculator.

        Parameters
        ----------
        keyword_clusterer_json : Union[str, Path, Dict]
            Path to KeywordClusterer JSON file or loaded dict with centroids and topic_descriptors
        embedder : Optional[Any]
            Embedding model for encoding text (e.g., KeywordEmbedder instance).
            Required if text needs to be embedded (not fetching from database).
        metric : str
            Distance metric: "cosine" or "euclidean". Default is "cosine".
        """
        self.embedder = embedder
        self.metric = metric

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

    def _load_centroids_and_topics(self):
        """Load centroids and topic descriptors from KeywordClusterer data."""
        if "centroids" not in self.clusterer_data:
            raise ValueError(
                "KeywordClusterer JSON missing 'centroids' field. "
                "Ensure the JSON was saved with include_embeddings=True or contains centroid information."
            )

        self.centroids = np.array(self.clusterer_data["centroids"])
        LOGGER.info(f"Loaded {len(self.centroids)} centroids with dimension {self.centroids.shape[1]}")

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

    def _create_distance_json_mapping(
        self, distances: np.ndarray
    ) -> str:
        """
        Create JSON mapping of topic descriptors to their distances.

        For each cluster, assigns the cluster's distance to all its topic descriptors.
        This creates a {topic_descriptor: distance} mapping similar to the format
        used by DomainAssessmentAgent's question_term_relevance_scores.

        Parameters
        ----------
        distances : np.ndarray
            Array of distances to each cluster centroid, shape (n_clusters,)

        Returns
        -------
        str
            JSON string mapping topic_descriptor to distance value
        """
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

    def _compute_distances_to_centroids(
        self, embedding: np.ndarray
    ) -> np.ndarray:
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
        Get embedding for text using the embedder.

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
                "Provide an embedder during initialization or use database IDs."
            )

        # Check embedder type and use appropriate method
        if hasattr(self.embedder, "create_embeddings"):
            # KeywordEmbedder interface
            embeddings_dict = self.embedder.create_embeddings([text])
            return embeddings_dict[text]
        elif hasattr(self.embedder, "embed_query"):
            # LangChain interface
            return np.array(self.embedder.embed_query(text))
        elif hasattr(self.embedder, "encode"):
            # sentence-transformers interface
            return self.embedder.encode(text)
        else:
            raise ValueError(
                "Embedder must have 'create_embeddings', 'embed_query', or 'encode' method"
            )

    def _get_embedding_from_database(
        self,
        item_id: str,
        database: Any,
        collection_name: str = "questions"
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
        if hasattr(database, "get_embedding"):
            return database.get_embedding(item_id, collection=collection_name)
        elif hasattr(database, "get"):
            # ChromaDB interface
            result = database.get(ids=[item_id], include=["embeddings"])
            if result and "embeddings" in result and result["embeddings"]:
                return np.array(result["embeddings"][0])
        else:
            raise ValueError(
                "Database must have 'get_embedding' or 'get' method with embeddings support"
            )
        return None

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

        Returns
        -------
        pd.DataFrame
            DataFrame with added columns for distances to each topic cluster:
            - question_closest_topic_keywords: topic descriptors for closest cluster
            - source_closest_topic_keywords: topic descriptors for closest cluster
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
        df["question_closest_topic_keywords"] = None
        df["source_closest_topic_keywords"] = None
        df["question_term_distance_scores"] = None  # JSON mapping {topic: distance}
        df["source_term_distance_scores"] = None  # JSON mapping {topic: distance}

        # Process each row
        iterator = range(len(df))
        if show_progress:
            iterator = tqdm(iterator, desc="Calculating topic distances", unit="row")

        for idx in iterator:
            row = df.iloc[idx]

            # Process question
            if question_id_col and question_id_col in df.columns and pd.notna(row[question_id_col]):
                # Fetch from database
                if database is None:
                    raise ValueError("Database required when using question_id_col")
                question_embedding = self._get_embedding_from_database(
                    str(row[question_id_col]), database, "questions"
                )
            elif question_col in df.columns and pd.notna(row[question_col]):
                # Embed text
                question_embedding = self._get_embedding_from_text(str(row[question_col]))
            else:
                LOGGER.warning(f"Row {idx}: No valid question data")
                question_embedding = None

            if question_embedding is not None:
                question_distances = self._compute_distances_to_centroids(question_embedding)
                closest_topic = int(np.argmin(question_distances))
                if closest_topic in self.topic_descriptors:
                    df.at[idx, "question_closest_topic_keywords"] = json.dumps(
                        self.topic_descriptors[closest_topic]
                    )
                # Create JSON mapping {topic: distance}
                df.at[idx, "question_term_distance_scores"] = self._create_distance_json_mapping(
                    question_distances
                )

            # Process source
            if source_id_col and source_id_col in df.columns and pd.notna(row[source_id_col]):
                # Fetch from database
                if database is None:
                    raise ValueError("Database required when using source_id_col")
                source_embedding = self._get_embedding_from_database(
                    str(row[source_id_col]), database, "sources"
                )
            elif source_col in df.columns and pd.notna(row[source_col]):
                # Embed text
                source_embedding = self._get_embedding_from_text(str(row[source_col]))
            else:
                LOGGER.warning(f"Row {idx}: No valid source data")
                source_embedding = None

            if source_embedding is not None:
                source_distances = self._compute_distances_to_centroids(source_embedding)
                closest_topic = int(np.argmin(source_distances))
                if closest_topic in self.topic_descriptors:
                    df.at[idx, "source_closest_topic_keywords"] = json.dumps(
                        self.topic_descriptors[closest_topic]
                    )
                # Create JSON mapping {topic: distance}
                df.at[idx, "source_term_distance_scores"] = self._create_distance_json_mapping(
                    source_distances
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
    output_path: Optional[Union[str, Path]] = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Convenience function to calculate topic distances from CSV.

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
        Embedding model for encoding text. Required if not using database IDs.
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
    output_path : Optional[Union[str, Path]]
        Path to save output CSV
    show_progress : bool
        Show progress bar. Default is True.

    Returns
    -------
    pd.DataFrame
        DataFrame with added distance columns:
        - question_closest_topic_keywords: topic descriptors for closest cluster
        - source_closest_topic_keywords: topic descriptors for closest cluster
        - question_term_distance_scores: JSON mapping {topic_descriptor: distance} for question
        - source_term_distance_scores: JSON mapping {topic_descriptor: distance} for source

    Examples
    --------
    >>> from RAG_supporters.embeddings import KeywordEmbedder
    >>> from RAG_supporters.clustering.topic_distance_calculator import calculate_topic_distances_from_csv
    >>>
    >>> # Using text embedding
    >>> embedder = KeywordEmbedder()
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
    )
