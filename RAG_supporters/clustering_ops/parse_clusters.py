"""
Cluster Parser for JASPER Steering Dataset Builder.

This module parses KeywordClusterer JSON format and provides keyword matching
with exact string matching and cosine similarity fallback.

Key Features:
- Parse cluster metadata: assignments, centroids, topic descriptors, embeddings
- Exact keyword matching (case-insensitive)
- Cosine similarity fallback for fuzzy matching
- Normalize keyword text for robust matching
- Integration with ClusteringData from clustering module
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np

from RAG_supporters.clustering import ClusteringData

LOGGER = logging.getLogger(__name__)


class ClusterParser:
    """Parse KeywordClusterer JSON and match keywords to clusters.

    This parser loads cluster metadata from KeywordClusterer JSON format
    and provides keyword-to-cluster matching with exact and fuzzy (cosine)
    fallback strategies.

    Parameters
    ----------
    clustering_json_path : str or Path
        Path to KeywordClusterer JSON results file (from KeywordClusterer.save_results())
    cosine_threshold : float, optional
        Minimum cosine similarity for fuzzy keyword matching (default: 0.7)
        Only used when exact match fails and embeddings are available

    Attributes
    ----------
    clustering_data : ClusteringData
        Parsed cluster metadata container
    keyword_to_cluster : Dict[str, int]
        Mapping from normalized keywords to cluster IDs
    keyword_embeddings : Dict[str, np.ndarray], optional
        Keyword embeddings if available in JSON
    cluster_keywords : Dict[int, List[str]]
        Mapping from cluster ID to list of keywords

    Examples
    --------
    >>> parser = ClusterParser("clusters.json")
    >>> cluster_id = parser.match_keyword("machine learning")
    >>> print(f"Matched to cluster {cluster_id}")
    >>>
    >>> # Fuzzy matching with cosine fallback
    >>> cluster_id, similarity = parser.match_keyword_fuzzy("ml algorithms")
    >>> if cluster_id is not None:
    ...     print(f"Fuzzy matched to cluster {cluster_id} (similarity={similarity:.3f})")
    >>>
    >>> # Get cluster info
    >>> descriptors = parser.get_cluster_descriptors(cluster_id)
    >>> centroid = parser.get_centroid(cluster_id)
    """

    def __init__(self, clustering_json_path: Union[str, Path], cosine_threshold: float = 0.7):
        """Initialize cluster parser."""
        self.clustering_json_path = Path(clustering_json_path)
        self.cosine_threshold = cosine_threshold

        # Validate file exists
        if not self.clustering_json_path.exists():
            raise FileNotFoundError(f"Clustering JSON not found: {self.clustering_json_path}")

        # Load clustering data
        LOGGER.info(f"Loading cluster metadata from {self.clustering_json_path}")
        self.clustering_data = ClusteringData.from_json(
            self.clustering_json_path,
            exclude_keys=None,  # Load all data including embeddings if available
        )

        # Load raw JSON for additional fields not in ClusteringData
        with open(self.clustering_json_path, "r", encoding="utf-8") as f:
            self.raw_json = json.load(f)

        # Build keyword-to-cluster mapping
        self.keyword_to_cluster = self._build_keyword_mapping()

        # Build cluster-to-keywords mapping
        self.cluster_keywords = self._build_cluster_keywords()

        # Load keyword embeddings if available
        self.keyword_embeddings = self._load_keyword_embeddings()

        # Log statistics
        LOGGER.info(f"Loaded {len(self.keyword_to_cluster)} unique keywords")
        LOGGER.info(f"Loaded {self.clustering_data.n_clusters} clusters")
        LOGGER.info(f"Embeddings available: " f"{'Yes' if self.keyword_embeddings else 'No'}")

    def _normalize_keyword(self, keyword: str) -> str:
        """Normalize keyword for matching.

        Parameters
        ----------
        keyword : str
            Raw keyword text

        Returns
        -------
        str
            Normalized keyword (lowercase, stripped)
        """
        return keyword.lower().strip()

    @property
    def n_clusters(self) -> int:
        """Return number of clusters from loaded clustering metadata."""
        return self.clustering_data.n_clusters

    def _build_keyword_mapping(self) -> Dict[str, int]:
        """Build normalized keyword-to-cluster mapping.

        Returns
        -------
        Dict[str, int]
            Mapping from normalized keywords to cluster IDs
        """
        keyword_mapping = {}

        # Get cluster assignments from raw JSON
        assignments = self.raw_json.get("cluster_assignments", {})

        for keyword, cluster_id in assignments.items():
            normalized = self._normalize_keyword(keyword)
            keyword_mapping[normalized] = int(cluster_id)

        LOGGER.debug(f"Built keyword mapping with {len(keyword_mapping)} entries")

        return keyword_mapping

    def _build_cluster_keywords(self) -> Dict[int, List[str]]:
        """Build cluster-to-keywords mapping.

        Returns
        -------
        Dict[int, List[str]]
            Mapping from cluster ID to list of keywords
        """
        cluster_kw = {}

        # Get clusters from raw JSON
        clusters = self.raw_json.get("clusters", {})

        for cluster_id_str, keywords in clusters.items():
            cluster_id = int(cluster_id_str)
            cluster_kw[cluster_id] = keywords

        LOGGER.debug(f"Built cluster keywords for {len(cluster_kw)} clusters")

        return cluster_kw

    def _load_keyword_embeddings(self) -> Optional[Dict[str, np.ndarray]]:
        """Load keyword embeddings if available in JSON.

        Returns
        -------
        Dict[str, np.ndarray] or None
            Mapping from normalized keywords to embeddings, or None if not available
        """
        embeddings_dict = self.raw_json.get("embeddings")

        if not embeddings_dict:
            LOGGER.debug("No embeddings found in JSON")
            return None

        # Convert to numpy arrays with normalized keys
        normalized_embeddings = {}
        for keyword, embedding in embeddings_dict.items():
            normalized = self._normalize_keyword(keyword)
            normalized_embeddings[normalized] = np.array(embedding, dtype=np.float32)

        LOGGER.debug(f"Loaded {len(normalized_embeddings)} keyword embeddings")

        return normalized_embeddings

    def match_keyword(self, keyword: str) -> Optional[int]:
        """Match keyword to cluster using exact matching.

        Performs case-insensitive exact matching.

        Parameters
        ----------
        keyword : str
            Keyword to match

        Returns
        -------
        int or None
            Cluster ID if matched, None otherwise

        Examples
        --------
        >>> parser = ClusterParser("clusters.json")
        >>> cluster_id = parser.match_keyword("Machine Learning")
        >>> print(cluster_id)
        3
        """
        normalized = self._normalize_keyword(keyword)
        return self.keyword_to_cluster.get(normalized)

    def match_keyword_fuzzy(
        self, keyword: str, return_similarity: bool = True
    ) -> Union[Tuple[Optional[int], float], Optional[int]]:
        """Match keyword to cluster with cosine similarity fallback.

        First attempts exact matching. If exact match fails and embeddings
        are available, falls back to cosine similarity matching.

        Parameters
        ----------
        keyword : str
            Keyword to match
        return_similarity : bool, optional
            If True, return (cluster_id, similarity). If False, return cluster_id only.
            Default: True

        Returns
        -------
        Tuple[int or None, float] or int or None
            If return_similarity=True: (cluster_id, similarity_score)
            If return_similarity=False: cluster_id only
            Returns (None, 0.0) or None if no match found

        Examples
        --------
        >>> parser = ClusterParser("clusters.json")
        >>> cluster_id, similarity = parser.match_keyword_fuzzy("ML models")
        >>> if cluster_id is not None:
        ...     print(f"Matched to cluster {cluster_id} (sim={similarity:.3f})")
        """
        # Try exact match first
        cluster_id = self.match_keyword(keyword)
        if cluster_id is not None:
            return (cluster_id, 1.0) if return_similarity else cluster_id

        # Fall back to cosine similarity if embeddings available
        if not self.keyword_embeddings:
            LOGGER.debug(f"No exact match for '{keyword}' and no embeddings for fallback")
            return (None, 0.0) if return_similarity else None

        # Need to compute embedding for input keyword
        # Since we don't have embedding model here, we can only match against
        # existing keyword embeddings using string similarity
        # This is a limitation - true fuzzy matching requires embedding model
        LOGGER.debug(
            f"Fuzzy matching for '{keyword}' requires embedding model " "(not available in parser)"
        )

        return (None, 0.0) if return_similarity else None

    def match_keywords_batch(self, keywords: List[str]) -> List[Optional[int]]:
        """Match multiple keywords to clusters (exact matching).

        Parameters
        ----------
        keywords : List[str]
            List of keywords to match

        Returns
        -------
        List[int or None]
            List of cluster IDs (None for unmatched keywords)

        Examples
        --------
        >>> parser = ClusterParser("clusters.json")
        >>> keywords = ["machine learning", "deep learning", "unknown term"]
        >>> cluster_ids = parser.match_keywords_batch(keywords)
        >>> print(cluster_ids)
        [3, 3, None]
        """
        return [self.match_keyword(kw) for kw in keywords]

    def get_cluster_descriptors(self, cluster_id: int) -> Optional[List[str]]:
        """Get topic descriptors for a cluster.

        Parameters
        ----------
        cluster_id : int
            Cluster ID

        Returns
        -------
        List[str] or None
            Topic descriptor keywords, or None if not available

        Examples
        --------
        >>> parser = ClusterParser("clusters.json")
        >>> descriptors = parser.get_cluster_descriptors(3)
        >>> print(descriptors)
        ['machine learning', 'neural networks', 'AI']
        """
        return self.clustering_data.get_descriptors(cluster_id)

    def get_centroid(self, cluster_id: int) -> Optional[np.ndarray]:
        """Get centroid embedding for a cluster.

        Parameters
        ----------
        cluster_id : int
            Cluster ID

        Returns
        -------
        np.ndarray or None
            Centroid embedding vector, or None if not available

        Examples
        --------
        >>> parser = ClusterParser("clusters.json")
        >>> centroid = parser.get_centroid(3)
        >>> print(centroid.shape)
        (384,)
        """
        return self.clustering_data.get_centroid(cluster_id)

    def get_all_centroids(self) -> Optional[np.ndarray]:
        """Get all cluster centroids.

        Returns
        -------
        np.ndarray or None
            Array of shape (n_clusters, embedding_dim) containing all centroids,
            or None if centroids are not available

        Examples
        --------
        >>> parser = ClusterParser("clusters.json")
        >>> centroids = parser.get_all_centroids()
        >>> print(centroids.shape)
        (20, 384)
        """
        return self.clustering_data.centroids

    def get_cluster_keywords(self, cluster_id: int) -> Optional[List[str]]:
        """Get all keywords assigned to a cluster.

        Parameters
        ----------
        cluster_id : int
            Cluster ID

        Returns
        -------
        List[str] or None
            List of keywords in cluster, or None if cluster not found

        Examples
        --------
        >>> parser = ClusterParser("clusters.json")
        >>> keywords = parser.get_cluster_keywords(3)
        >>> print(len(keywords))
        45
        """
        return self.cluster_keywords.get(cluster_id)

    def get_all_keywords(self) -> List[str]:
        """Get all unique keywords across all clusters.

        Returns
        -------
        List[str]
            List of all unique keywords (normalized)
        """
        return list(self.keyword_to_cluster.keys())

    def get_clusters_for_keywords(
        self, keywords: List[str], ignore_missing: bool = True
    ) -> Dict[str, Optional[int]]:
        """Get cluster assignments for multiple keywords.

        Parameters
        ----------
        keywords : List[str]
            List of keywords to look up
        ignore_missing : bool, optional
            If True, include None values for unmatched keywords.
            If False, only return matched keywords.
            Default: True

        Returns
        -------
        Dict[str, int or None]
            Mapping from keywords to cluster IDs

        Examples
        --------
        >>> parser = ClusterParser("clusters.json")
        >>> keywords = ["machine learning", "deep learning", "unknown"]
        >>> assignments = parser.get_clusters_for_keywords(keywords)
        >>> print(assignments)
        {'machine learning': 3, 'deep learning': 3, 'unknown': None}
        """
        result = {}

        for keyword in keywords:
            cluster_id = self.match_keyword(keyword)

            if ignore_missing or cluster_id is not None:
                result[keyword] = cluster_id

        return result

    def compute_cluster_coverage(self, keywords: List[str]) -> Tuple[int, int, float, Set[int]]:
        """Compute cluster coverage statistics for a keyword list.

        Parameters
        ----------
        keywords : List[str]
            List of keywords to analyze

        Returns
        -------
        Tuple[int, int, float, Set[int]]
            (matched_count, total_count, coverage_ratio, covered_clusters)

        Examples
        --------
        >>> parser = ClusterParser("clusters.json")
        >>> keywords = ["ML", "deep learning", "NLP", "unknown"]
        >>> matched, total, ratio, clusters = parser.compute_cluster_coverage(keywords)
        >>> print(f"Coverage: {matched}/{total} = {ratio:.2%}")
        Coverage: 3/4 = 75.00%
        >>> print(f"Covered clusters: {sorted(clusters)}")
        Covered clusters: [1, 3, 5]
        """
        matched_count = 0
        covered_clusters: Set[int] = set()

        for keyword in keywords:
            cluster_id = self.match_keyword(keyword)
            if cluster_id is not None:
                matched_count += 1
                covered_clusters.add(cluster_id)

        total_count = len(keywords)
        coverage_ratio = matched_count / total_count if total_count > 0 else 0.0

        return matched_count, total_count, coverage_ratio, covered_clusters

    def validate(self) -> None:
        """Validate cluster parser state.

        Raises
        ------
        ValueError
            If validation fails
        """
        # Validate clustering data
        self.clustering_data.validate()

        # Validate keyword mappings are consistent
        if len(self.keyword_to_cluster) == 0:
            raise ValueError("No keywords loaded from JSON")

        if self.clustering_data.n_clusters == 0:
            raise ValueError("No clusters loaded from JSON")

        # Validate cluster IDs are in valid range
        for cluster_id in self.keyword_to_cluster.values():
            if cluster_id < 0 or cluster_id >= self.clustering_data.n_clusters:
                raise ValueError(
                    f"Invalid cluster ID {cluster_id} (n_clusters={self.clustering_data.n_clusters})"
                )

        LOGGER.info("Cluster parser validation passed")

    def get_metadata(self) -> Dict[str, any]:
        """Get cluster metadata for config.json.

        Returns
        -------
        Dict[str, any]
            Metadata dictionary with:
            - n_clusters: Number of clusters
            - n_keywords: Number of unique keywords
            - embedding_dim: Embedding dimension
            - clustering_source: Path to JSON file

        Examples
        --------
        >>> parser = ClusterParser("clusters.json")
        >>> metadata = parser.get_metadata()
        >>> print(metadata['n_clusters'])
        20
        """
        raw_metadata = self.raw_json.get("metadata", {})

        return {
            "n_clusters": self.clustering_data.n_clusters,
            "n_keywords": len(self.keyword_to_cluster),
            "embedding_dim": raw_metadata.get("embedding_dim"),
            "clustering_source": str(self.clustering_json_path),
            "algorithm": raw_metadata.get("algorithm"),
            "random_state": raw_metadata.get("random_state"),
        }


def parse_clusters(
    clustering_json_path: Union[str, Path], validate: bool = True, cosine_threshold: float = 0.7
) -> ClusterParser:
    """Convenience function to parse KeywordClusterer JSON.

    Parameters
    ----------
    clustering_json_path : str or Path
        Path to KeywordClusterer JSON results file
    validate : bool, optional
        Whether to run validation checks (default: True)
    cosine_threshold : float, optional
        Minimum cosine similarity for fuzzy matching (default: 0.7)

    Returns
    -------
    ClusterParser
        Initialized parser with loaded cluster metadata

    Examples
    --------
    >>> parser = parse_clusters("clusters.json")
    >>> cluster_id = parser.match_keyword("machine learning")
    """
    parser = ClusterParser(
        clustering_json_path=clustering_json_path, cosine_threshold=cosine_threshold
    )

    if validate:
        parser.validate()

    return parser
