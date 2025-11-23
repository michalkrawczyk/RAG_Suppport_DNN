# TODO Docstring

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)
# TODO: SPlit it to separate files

# ============================================================================
# CORE FUNCTIONS - Independent implementations
# ============================================================================

def filter_by_field_value(
        suggestions: List[Dict[str, Any]],
        min_value: float = 0.7,
        field_name: str = 'confidence',
) -> List[Dict[str, Any]]:
    """
    Filter suggestions by a numeric field threshold.

    Parameters
    ----------
    suggestions : List[Dict[str, Any]]
        List of suggestions, each containing numeric field to filter by
    min_value : float
        Minimum threshold value (0.0 to 1.0 for confidence-like fields)
    field_name : str
        Name of the numeric field to filter by (default: 'confidence')

    Returns
    -------
    List[Dict[str, Any]]
        Filtered suggestions that meet the threshold

    Examples
    --------
    >>> suggestions = [
    ...     {'term': 'machine learning', 'confidence': 0.9},
    ...     {'term': 'data science', 'confidence': 0.6},
    ...     {'term': 'AI', 'confidence': 0.85}
    ... ]
    >>> filtered = filter_by_field_value(suggestions, 0.7)
    >>> len(filtered)
    2

    >>> # Filter by custom field
    >>> suggestions = [
    ...     {'term': 'keyword1', 'score': 0.8},
    ...     {'term': 'keyword2', 'score': 0.5}
    ... ]
    >>> filtered = filter_by_field_value(suggestions, 0.6, field_name='score')
    >>> len(filtered)
    1
    """
    if not suggestions:
        LOGGER.warning("Empty suggestions list provided")
        return []

    filtered = []
    skipped_missing_field = 0

    for suggestion in suggestions:
        # Check if field exists
        if field_name not in suggestion:
            skipped_missing_field += 1
            continue

        field_value = suggestion.get(field_name, 0.0)

        # Try to convert to float if not already
        try:
            field_value = float(field_value)
        except (ValueError, TypeError):
            LOGGER.warning(
                f"Could not convert field '{field_name}' value '{field_value}' to float, skipping"
            )
            continue

        if field_value >= min_value:
            filtered.append(suggestion)

    if skipped_missing_field > 0:
        LOGGER.warning(
            f"Skipped {skipped_missing_field} suggestions missing field '{field_name}'"
        )

    LOGGER.info(
        f"Filtered {len(filtered)}/{len(suggestions)} suggestions "
        f"with {field_name} >= {min_value}"
    )

    return filtered


def aggregate_unique_terms(
        suggestions: List[Dict[str, Any]],
        term_key: str = 'term',
        normalize: bool = True,
        return_counts: bool = False,
) -> Tuple[List[str], Optional[Dict[str, int]]]:
    """
    Aggregate terms into unique keywords.

    Parameters
    ----------
    suggestions : List[Dict[str, Any]]
        List of suggestions, each containing term information
    term_key : str
        Key name for the term in suggestion dictionaries
    normalize : bool
        Whether to normalize terms (lowercase, strip whitespace)
    return_counts : bool
        If True, also return occurrence counts for each unique term

    Returns
    -------
    Tuple[List[str], Optional[Dict[str, int]]]
        Tuple of (unique_keywords, counts_dict)
        - unique_keywords: List of unique keywords (order preserved)
        - counts_dict: Dict mapping keywords to counts if return_counts=True, else None

    Examples
    --------
    >>> suggestions = [
    ...     {'term': 'Machine Learning', 'confidence': 0.9},
    ...     {'term': 'machine learning', 'confidence': 0.85},
    ...     {'term': 'Deep Learning', 'confidence': 0.8},
    ...     {'term': 'Machine Learning', 'confidence': 0.75}
    ... ]
    >>> keywords, counts = aggregate_unique_terms(suggestions, normalize=True)
    >>> keywords
    ['machine learning', 'deep learning']
    >>> counts is None
    True

    >>> keywords, counts = aggregate_unique_terms(suggestions, normalize=True, return_counts=True)
    >>> keywords
    ['machine learning', 'deep learning']
    >>> counts
    {'machine learning': 3, 'deep learning': 1}
    """
    if not suggestions:
        LOGGER.warning("Empty suggestions list provided")
        return [], None

    keywords = []
    seen = set()
    keyword_counts = {} if return_counts else None

    for suggestion in suggestions:
        term = suggestion.get(term_key, '')
        if not term:
            continue

        # Normalize if requested
        if normalize:
            term = term.lower().strip()
        else:
            term = term.strip()

        # Track counts if requested
        if return_counts:
            keyword_counts[term] = keyword_counts.get(term, 0) + 1

        # Add to list if not seen (preserving order)
        if term not in seen:
            keywords.append(term)
            seen.add(term)

    log_msg = f"Aggregated {len(keywords)} unique keywords from {len(suggestions)} suggestions"
    if return_counts:
        log_msg += " (with counts)"
    LOGGER.info(log_msg)

    return keywords, keyword_counts


def create_embeddings_for_strings(
        str_list: List[str],
        embedding_model: Optional[Any] = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        show_progress: bool = True,
        normalize_embeddings: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Create embeddings for each keyword separately.

    Parameters
    ----------
    str_list : List[str]
        List of words to embed
    embedding_model : Optional[Any]
        Pre-loaded embedding model. If None, will load model_name
    model_name : str
        Name of the embedding model to use if embedding_model is None
    batch_size : int
        Batch size for embedding generation
    show_progress : bool
        Whether to show progress bar during embedding generation
    normalize_embeddings : bool
        Whether to L2-normalize the embeddings

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping each word to its embedding vector

    Examples
    --------
    >>> keywords = ['machine learning', 'data science', 'artificial intelligence']
    >>> embeddings = create_embeddings_for_strings(str_list)
    >>> len(embeddings)
    3
    >>> embeddings['machine learning'].shape
    (384,)  # Depends on model

    Raises
    ------
    ImportError
        If sentence-transformers is not installed
    ValueError
        If keywords list is empty
    """
    if not str_list:
        raise ValueError("Keywords list cannot be empty")

    # Remove duplicates while preserving order
    unique_keywords = list(dict.fromkeys(str_list))

    if len(unique_keywords) < len(str_list):
        LOGGER.warning(
            f"Removed {len(str_list) - len(unique_keywords)} duplicate keywords"
        )

    # Load model if not provided
    if embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer

            LOGGER.info(f"Loading embedding model: {model_name}")
            embedding_model = SentenceTransformer(model_name)
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for embedding generation. "
                "Install with: pip install sentence-transformers"
            )

    LOGGER.info(f"Generating embeddings for {len(unique_keywords)} keywords")

    # Generate embeddings
    embeddings = embedding_model.encode(
        unique_keywords,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )

    # Create keyword -> embedding mapping
    keyword_embeddings = {
        keyword: embedding
        for keyword, embedding in zip(unique_keywords, embeddings)
    }

    embedding_dim = embeddings.shape[1]
    LOGGER.info(
        f"Successfully generated embeddings with dimension {embedding_dim} "
        f"for {len(keyword_embeddings)} keywords"
    )

    return keyword_embeddings


# ============================================================================
# UTILITY FUNCTIONS - CSV/JSON I/O
# ============================================================================

def load_suggestions_from_csv(
        csv_path: str,
        suggestion_column: str = "suggestions",
        chunksize: int = 1000,
        show_progress: bool = True,
) -> List[Dict[str, Any]]:
    # TODO: Refactor
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


# ============================================================================
# CONVENIENCE WRAPPER CLASS
# ============================================================================

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


# ============================================================================
# CLUSTERING FUNCTIONS
# ============================================================================

class KeywordClusterer:
    """
    Cluster keyword embeddings using KMeans or Bisecting KMeans.
    """

    def __init__(
            self,
            algorithm: str = "kmeans",
            n_clusters: int = 8,
            random_state: int = 42,
            **kwargs,
    ):
        """
        Initialize the clusterer.

        Parameters
        ----------
        algorithm : str
            Clustering algorithm: 'kmeans' or 'bisecting_kmeans'
        n_clusters : int
            Number of clusters
        random_state : int
            Random state for reproducibility
        **kwargs
            Additional arguments for the clustering algorithm
        """
        self.algorithm = algorithm
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kwargs = kwargs

        self.model = self._create_model()
        self.keywords = None
        self.embeddings_matrix = None
        self.cluster_labels = None

    def _create_model(self):
        """Create the clustering model"""
        try:
            from sklearn.cluster import BisectingKMeans, KMeans
        except ImportError:
            raise ImportError(
                "scikit-learn is required for clustering. "
                "Install with: pip install scikit-learn"
            )

        if self.algorithm == "kmeans":
            return KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                **self.kwargs,
            )
        elif self.algorithm == "bisecting_kmeans":
            return BisectingKMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                **self.kwargs,
            )
        else:
            raise ValueError(
                f"Unknown algorithm: {self.algorithm}. "
                "Choose 'kmeans' or 'bisecting_kmeans'"
            )

    def fit(
            self,
            keyword_embeddings: Dict[str, np.ndarray],
    ) -> 'KeywordClusterer':
        """
        Fit the clustering model.

        Parameters
        ----------
        keyword_embeddings : Dict[str, np.ndarray]
            Dictionary mapping keywords to embeddings

        Returns
        -------
        KeywordClusterer
            Self for chaining
        """
        # Convert to matrix
        self.keywords = list(keyword_embeddings.keys())
        self.embeddings_matrix = np.array([
            keyword_embeddings[kw] for kw in self.keywords
        ])

        LOGGER.info(
            f"Fitting {self.algorithm} with {self.n_clusters} clusters "
            f"on {len(self.keywords)} keywords"
        )

        # Fit model
        self.model.fit(self.embeddings_matrix)
        self.cluster_labels = self.model.labels_

        LOGGER.info("Clustering complete")

        return self

    def get_cluster_assignments(self) -> Dict[str, int]:
        """
        Get cluster assignments for each keyword.

        Returns
        -------
        Dict[str, int]
            Dictionary mapping keywords to cluster labels
        """
        if self.cluster_labels is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return {
            keyword: int(label)
            for keyword, label in zip(self.keywords, self.cluster_labels)
        }

    def get_clusters(self) -> Dict[int, List[str]]:
        """
        Get keywords grouped by cluster.

        Returns
        -------
        Dict[int, List[str]]
            Dictionary mapping cluster labels to lists of keywords
        """
        if self.cluster_labels is None:
            raise ValueError("Model not fitted. Call fit() first.")

        clusters = {}
        for keyword, label in zip(self.keywords, self.cluster_labels):
            label = int(label)
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(keyword)

        return clusters

    def get_centroids(self) -> np.ndarray:
        """
        Get cluster centroids.

        Returns
        -------
        np.ndarray
            Array of cluster centroids, shape (n_clusters, embedding_dim)
        """
        if not hasattr(self.model, 'cluster_centers_'):
            raise ValueError("Model not fitted or doesn't have centroids")

        return self.model.cluster_centers_

    def save_results(
            self,
            output_path: str,
            include_embeddings: bool = False,
    ):
        """
        Save clustering results to JSON.

        Parameters
        ----------
        output_path : str
            Path to save results
        include_embeddings : bool
            Whether to include embeddings in output
        """
        # Get cluster assignments and grouped clusters
        assignments = self.get_cluster_assignments()
        clusters = self.get_clusters()

        # Calculate cluster statistics
        cluster_stats = {
            str(label): {
                "size": len(keywords),
                "keywords_sample": keywords[:10],  # First 10 for preview
            }
            for label, keywords in clusters.items()
        }

        # Prepare output
        output_data = {
            "metadata": {
                "algorithm": self.algorithm,
                "n_clusters": self.n_clusters,
                "n_keywords": len(self.keywords),
                "random_state": self.random_state,
            },
            "cluster_assignments": assignments,
            "clusters": {str(k): v for k, v in clusters.items()},
            "cluster_stats": cluster_stats,
            "centroids": self.get_centroids().tolist(),
        }

        if include_embeddings:
            output_data["embeddings"] = {
                kw: emb.tolist()
                for kw, emb in zip(self.keywords, self.embeddings_matrix)
            }

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        LOGGER.info(f"Saved clustering results to {output_path}")

    @staticmethod
    def load_results(input_path: str) -> Dict[str, Any]:
        """
        Load clustering results from JSON.

        Parameters
        ----------
        input_path : str
            Path to results file

        Returns
        -------
        Dict[str, Any]
            Clustering results
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data


# ============================================================================
# CENTROID COMPARISON FUNCTIONS
# ============================================================================

class CentroidComparator:
    """
    Compare keywords or new text against cluster centroids.
    """

    def __init__(
            self,
            centroids: np.ndarray,
            cluster_labels: Optional[List[int]] = None,
            cluster_info: Optional[Dict[int, Dict[str, Any]]] = None,
    ):
        """
        Initialize the comparator.

        Parameters
        ----------
        centroids : np.ndarray
            Cluster centroids, shape (n_clusters, embedding_dim)
        cluster_labels : Optional[List[int]]
            Optional labels for clusters
        cluster_info : Optional[Dict[int, Dict[str, Any]]]
            Optional additional info about each cluster
        """
        self.centroids = centroids
        self.cluster_labels = cluster_labels or list(range(len(centroids)))
        self.cluster_info = cluster_info or {}

    @classmethod
    def from_clustering_results(
            cls,
            clustering_results_path: str,
    ) -> 'CentroidComparator':
        """
        Create comparator from saved clustering results.

        Parameters
        ----------
        clustering_results_path : str
            Path to clustering results JSON

        Returns
        -------
        CentroidComparator
            Initialized comparator
        """
        with open(clustering_results_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        centroids = np.array(data['centroids'])
        cluster_labels = list(range(len(centroids)))

        # Get cluster info if available
        cluster_info = {}
        if 'cluster_stats' in data:
            for label_str, stats in data['cluster_stats'].items():
                cluster_info[int(label_str)] = stats

        return cls(centroids, cluster_labels, cluster_info)

    def compute_distances(
            self,
            embedding: np.ndarray,
            metric: str = "euclidean",
    ) -> np.ndarray:
        """
        Compute distances from embedding to all centroids.

        Parameters
        ----------
        embedding : np.ndarray
            Query embedding vector
        metric : str
            Distance metric: 'euclidean' or 'cosine'

        Returns
        -------
        np.ndarray
            Array of distances to each centroid
        """
        if metric == "euclidean":
            distances = np.linalg.norm(self.centroids - embedding, axis=1)
        elif metric == "cosine":
            # Cosine distance = 1 - cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity([embedding], self.centroids)[0]
            distances = 1 - similarities
        else:
            raise ValueError(
                f"Unknown metric: {metric}. Choose 'euclidean' or 'cosine'"
            )

        return distances

    def find_nearest_cluster(
            self,
            embedding: np.ndarray,
            metric: str = "euclidean",
            top_k: int = 1,
    ) -> Union[Tuple[int, float], List[Tuple[int, float]]]:
        """
        Find the nearest cluster(s) for an embedding.

        Parameters
        ----------
        embedding : np.ndarray
            Query embedding vector
        metric : str
            Distance metric
        top_k : int
            Number of nearest clusters to return

        Returns
        -------
        Union[Tuple[int, float], List[Tuple[int, float]]]
            If top_k=1: (cluster_label, distance)
            If top_k>1: List of (cluster_label, distance) tuples
        """
        distances = self.compute_distances(embedding, metric)

        if top_k == 1:
            nearest_idx = np.argmin(distances)
            return self.cluster_labels[nearest_idx], float(distances[nearest_idx])
        else:
            nearest_indices = np.argsort(distances)[:top_k]
            return [
                (self.cluster_labels[idx], float(distances[idx]))
                for idx in nearest_indices
            ]

    def get_all_distances(
            self,
            embedding: np.ndarray,
            metric: str = "euclidean",
            sorted: bool = True,
    ) -> Dict[int, float]:
        """
        Get distances to all clusters.

        Parameters
        ----------
        embedding : np.ndarray
            Query embedding vector
        metric : str
            Distance metric
        sorted : bool
            Whether to sort by distance (ascending)

        Returns
        -------
        Dict[int, float]
            Dictionary mapping cluster labels to distances
        """
        distances = self.compute_distances(embedding, metric)

        distance_dict = {
            label: float(dist)
            for label, dist in zip(self.cluster_labels, distances)
        }

        if sorted:
            distance_dict = dict(
                sorted(distance_dict.items(), key=lambda x: x[1])
            )

        return distance_dict

    def compare_keyword(
            self,
            keyword: str,
            keyword_embeddings: Dict[str, np.ndarray],
            metric: str = "euclidean",
            top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Compare a keyword against centroids.

        Parameters
        ----------
        keyword : str
            Keyword to compare
        keyword_embeddings : Dict[str, np.ndarray]
            Dictionary of keyword embeddings
        metric : str
            Distance metric
        top_k : int
            Number of nearest clusters to return

        Returns
        -------
        Dict[str, Any]
            Comparison results with nearest clusters and all distances
        """
        if keyword not in keyword_embeddings:
            raise ValueError(f"Keyword '{keyword}' not found in embeddings")

        embedding = keyword_embeddings[keyword]

        top_clusters = self.find_nearest_cluster(embedding, metric, top_k)
        all_distances = self.get_all_distances(embedding, metric, sorted=True)

        return {
            "keyword": keyword,
            "top_clusters": top_clusters,
            "all_distances": all_distances,
            "metric": metric,
        }

    def compare_text(
            self,
            text: str,
            embedding_model: Any,
            metric: str = "euclidean",
            top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Compare arbitrary text against centroids.

        Parameters
        ----------
        text : str
            Text to compare
        embedding_model : Any
            Model to generate text embedding
        metric : str
            Distance metric
        top_k : int
            Number of nearest clusters to return

        Returns
        -------
        Dict[str, Any]
            Comparison results
        """
        # Generate embedding for text
        embedding = embedding_model.encode([text], convert_to_numpy=True)[0]

        top_clusters = self.find_nearest_cluster(embedding, metric, top_k)
        all_distances = self.get_all_distances(embedding, metric, sorted=True)

        return {
            "text": text,
            "top_clusters": top_clusters,
            "all_distances": all_distances,
            "metric": metric,
        }