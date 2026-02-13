"""
Source-Cluster Linker for JASPER Steering Dataset Builder.

This module links sources to clusters via keywords from KeywordClusterer JSON.
Resolves primary cluster assignments for each question-source pair based on
keyword overlap and cluster memberships.

Key Features:
- Link pairs to clusters via keyword intersection
- Resolve primary cluster using majority voting
- Handle pairs with no keyword matches (fallback to closest cluster)
- Validate cluster assignments for dataset integrity
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from RAG_supporters.clustering_ops import ClusterParser

LOGGER = logging.getLogger(__name__)


class SourceClusterLinker:
    """Link question-source pairs to clusters via keywords.
    
    This class assigns each pair to a primary cluster based on keyword
    overlap with cluster assignments from KeywordClusterer. Uses majority
    voting when multiple clusters are present, with fallback strategies
    for pairs without keyword matches.
    
    Parameters
    ----------
    cluster_parser : ClusterParser
        Parser for cluster metadata and keyword-cluster mappings
    fallback_strategy : str, optional
        Strategy for pairs without keyword matches:
        - "random": Assign to random cluster
        - "largest": Assign to largest cluster
        - "uniform": Distribute uniformly across clusters
        Default: "largest"
    
    Attributes
    ----------
    cluster_parser : ClusterParser
        Cluster metadata parser
    n_clusters : int
        Number of clusters from KeywordClusterer
    fallback_strategy : str
        Strategy for handling unmatched pairs
    
    Examples
    --------
    >>> from RAG_supporters.dataset import ClusterParser, SourceClusterLinker
    >>> parser = ClusterParser("clusters.json")
    >>> linker = SourceClusterLinker(parser)
    >>> 
    >>> # Link single pair
    >>> keywords = ["machine learning", "neural networks"]
    >>> cluster_id = linker.link_pair(keywords)
    >>> print(f"Primary cluster: {cluster_id}")
    >>>
    >>> # Link DataFrame of pairs
    >>> df = pd.read_csv("merged.csv")
    >>> cluster_assignments = linker.link_dataframe(df, keywords_col="keywords")
    >>> print(f"Assigned {len(cluster_assignments)} pairs to clusters")
    """
    
    def __init__(
        self,
        cluster_parser: ClusterParser,
        fallback_strategy: str = "largest"
    ):
        """Initialize source-cluster linker."""
        self.cluster_parser = cluster_parser
        self.n_clusters = cluster_parser.clustering_data.n_clusters
        self.fallback_strategy = fallback_strategy
        
        # Validate fallback strategy
        valid_strategies = ["random", "largest", "uniform"]
        if fallback_strategy not in valid_strategies:
            raise ValueError(
                f"Invalid fallback_strategy: {fallback_strategy}. "
                f"Must be one of {valid_strategies}"
            )
        
        # Precompute cluster sizes for fallback
        self.cluster_sizes = self._compute_cluster_sizes()
        
        LOGGER.info(
            f"Initialized SourceClusterLinker with {self.n_clusters} clusters, "
            f"fallback={fallback_strategy}"
        )
    
    def _compute_cluster_sizes(self) -> Dict[int, int]:
        """Compute number of keywords per cluster.
        
        Returns
        -------
        Dict[int, int]
            Mapping from cluster ID to number of keywords
        """
        cluster_sizes = {}
        for cluster_id in range(self.n_clusters):
            keywords = self.cluster_parser.get_cluster_keywords(cluster_id)
            cluster_sizes[cluster_id] = len(keywords) if keywords else 0
        
        return cluster_sizes
    
    def _get_fallback_cluster(self, pair_id: Optional[int] = None) -> int:
        """Get fallback cluster ID for pairs without keyword matches.
        
        Parameters
        ----------
        pair_id : int, optional
            Pair ID for reproducible uniform assignment
        
        Returns
        -------
        int
            Fallback cluster ID
        """
        if self.fallback_strategy == "random":
            return np.random.randint(0, self.n_clusters)
        
        elif self.fallback_strategy == "largest":
            # Assign to cluster with most keywords
            return max(self.cluster_sizes, key=self.cluster_sizes.get)
        
        elif self.fallback_strategy == "uniform":
            # Distribute uniformly based on pair_id
            if pair_id is None:
                return np.random.randint(0, self.n_clusters)
            return pair_id % self.n_clusters
        
        else:
            # Should never reach here due to validation in __init__
            raise ValueError(f"Unknown fallback strategy: {self.fallback_strategy}")
    
    def link_pair(
        self,
        keywords: List[str],
        pair_id: Optional[int] = None
    ) -> int:
        """Link a single pair to its primary cluster.
        
        Uses majority voting: cluster with most keyword matches wins.
        If no keywords match, uses fallback strategy.
        
        Parameters
        ----------
        keywords : List[str]
            List of keywords associated with the pair
        pair_id : int, optional
            Pair ID for reproducible fallback assignment
        
        Returns
        -------
        int
            Primary cluster ID (0 to n_clusters-1)
        
        Examples
        --------
        >>> linker = SourceClusterLinker(parser)
        >>> cluster_id = linker.link_pair(["python", "programming", "data science"])
        >>> print(cluster_id)
        1
        """
        if not keywords:
            LOGGER.debug(f"Pair {pair_id} has no keywords, using fallback")
            return self._get_fallback_cluster(pair_id)
        
        # Match keywords to clusters
        cluster_ids = self.cluster_parser.match_keywords_batch(keywords)
        
        # Filter out None values (unmatched keywords)
        matched_clusters = [cid for cid in cluster_ids if cid is not None]
        
        if not matched_clusters:
            LOGGER.debug(
                f"Pair {pair_id} keywords {keywords} did not match any cluster, "
                f"using fallback"
            )
            return self._get_fallback_cluster(pair_id)
        
        # Majority voting: most frequent cluster
        cluster_counts = {}
        for cluster_id in matched_clusters:
            cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
        
        primary_cluster = max(cluster_counts, key=cluster_counts.get)
        
        LOGGER.debug(
            f"Pair {pair_id} linked to cluster {primary_cluster} "
            f"({cluster_counts[primary_cluster]}/{len(keywords)} keywords matched)"
        )
        
        return primary_cluster
    
    def link_batch(
        self,
        keywords_list: List[List[str]],
        pair_ids: Optional[List[int]] = None
    ) -> List[int]:
        """Link multiple pairs to clusters in batch.
        
        Parameters
        ----------
        keywords_list : List[List[str]]
            List of keyword lists, one per pair
        pair_ids : List[int], optional
            List of pair IDs for reproducible fallback assignment
        
        Returns
        -------
        List[int]
            List of primary cluster IDs
        
        Examples
        --------
        >>> linker = SourceClusterLinker(parser)
        >>> keywords_list = [
        ...     ["python", "programming"],
        ...     ["database", "sql"],
        ...     ["machine learning"]
        ... ]
        >>> cluster_ids = linker.link_batch(keywords_list)
        >>> print(cluster_ids)
        [1, 2, 0]
        """
        if pair_ids is None:
            pair_ids = [None] * len(keywords_list)
        
        cluster_assignments = []
        for keywords, pair_id in zip(keywords_list, pair_ids):
            cluster_id = self.link_pair(keywords, pair_id)
            cluster_assignments.append(cluster_id)
        
        return cluster_assignments
    
    def link_dataframe(
        self,
        df: pd.DataFrame,
        keywords_col: str = "keywords",
        pair_id_col: str = "pair_id",
        output_col: str = "cluster_id",
        show_progress: bool = True
    ) -> pd.DataFrame:
        """Link all pairs in DataFrame to clusters.
        
        Adds a new column with primary cluster assignments.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with pairs and keywords
        keywords_col : str, optional
            Column name containing keyword lists (default: "keywords")
        pair_id_col : str, optional
            Column name containing pair IDs (default: "pair_id")
        output_col : str, optional
            Output column name for cluster assignments (default: "cluster_id")
        show_progress : bool, optional
            Show progress bar (default: True)
        
        Returns
        -------
        pd.DataFrame
            DataFrame with cluster_id column added
        
        Examples
        --------
        >>> linker = SourceClusterLinker(parser)
        >>> df = pd.read_csv("merged.csv")
        >>> df_linked = linker.link_dataframe(df)
        >>> print(df_linked[["pair_id", "keywords", "cluster_id"]].head())
        """
        # Validate columns exist
        if keywords_col not in df.columns:
            raise ValueError(f"Keywords column '{keywords_col}' not found in DataFrame")
        
        # Check if pair_id exists, otherwise use index
        if pair_id_col not in df.columns:
            LOGGER.warning(
                f"Pair ID column '{pair_id_col}' not found, using DataFrame index"
            )
            pair_ids = df.index.tolist()
        else:
            pair_ids = df[pair_id_col].tolist()
        
        # Extract keywords
        keywords_list = df[keywords_col].tolist()
        
        # Link pairs to clusters
        cluster_assignments = []
        iterator = zip(keywords_list, pair_ids)
        if show_progress:
            iterator = tqdm(
                iterator,
                total=len(df),
                desc="Linking pairs to clusters"
            )
        
        for keywords, pair_id in iterator:
            # Ensure keywords is a list
            if isinstance(keywords, str):
                keywords = [kw.strip() for kw in keywords.split(",")]
            elif not isinstance(keywords, list):
                keywords = []
            
            cluster_id = self.link_pair(keywords, pair_id)
            cluster_assignments.append(cluster_id)
        
        # Add cluster assignments to DataFrame
        df_result = df.copy()
        df_result[output_col] = cluster_assignments
        
        # Log statistics
        cluster_distribution = pd.Series(cluster_assignments).value_counts().sort_index()
        LOGGER.info(
            f"Linked {len(df_result)} pairs to {self.n_clusters} clusters. "
            f"Distribution: {cluster_distribution.to_dict()}"
        )
        
        return df_result
    
    def validate_assignments(
        self,
        cluster_assignments: Union[List[int], np.ndarray, torch.Tensor]
    ) -> Dict[str, any]:
        """Validate cluster assignments for dataset integrity.
        
        Checks:
        - All cluster IDs are in valid range [0, n_clusters-1]
        - All clusters have at least one assignment
        - Distribution is not too imbalanced
        
        Parameters
        ----------
        cluster_assignments : List[int] or np.ndarray or torch.Tensor
            Array of cluster assignments
        
        Returns
        -------
        Dict[str, any]
            Validation results with keys:
            - "valid": bool - Whether assignments are valid
            - "errors": List[str] - List of validation errors
            - "warnings": List[str] - List of validation warnings
            - "statistics": Dict - Assignment statistics
        
        Examples
        --------
        >>> linker = SourceClusterLinker(parser)
        >>> cluster_ids = [0, 0, 1, 1, 2, 2]
        >>> validation = linker.validate_assignments(cluster_ids)
        >>> if validation["valid"]:
        ...     print("Assignments are valid!")
        >>> else:
        ...     print("Errors:", validation["errors"])
        """
        # Convert to numpy array
        if isinstance(cluster_assignments, torch.Tensor):
            cluster_assignments = cluster_assignments.cpu().numpy()
        elif isinstance(cluster_assignments, list):
            cluster_assignments = np.array(cluster_assignments)
        
        errors = []
        warnings = []
        
        # Check valid range
        min_id = cluster_assignments.min()
        max_id = cluster_assignments.max()
        
        if min_id < 0:
            errors.append(f"Invalid cluster ID found: {min_id} (must be >= 0)")
        if max_id >= self.n_clusters:
            errors.append(
                f"Invalid cluster ID found: {max_id} "
                f"(must be < {self.n_clusters})"
            )
        
        # Check all clusters represented
        unique_clusters = set(cluster_assignments.tolist())
        all_clusters = set(range(self.n_clusters))
        missing_clusters = all_clusters - unique_clusters
        
        if missing_clusters:
            warnings.append(
                f"Clusters {sorted(missing_clusters)} have no assignments. "
                f"This may cause issues during training."
            )
        
        # Check distribution balance
        cluster_counts = pd.Series(cluster_assignments).value_counts()
        mean_count = cluster_counts.mean()
        std_count = cluster_counts.std()
        
        if std_count / mean_count > 2.0:  # Coefficient of variation > 2
            warnings.append(
                f"Highly imbalanced cluster distribution detected. "
                f"Mean={mean_count:.1f}, Std={std_count:.1f}. "
                f"Consider rebalancing or using stratified sampling."
            )
        
        statistics = {
            "n_pairs": len(cluster_assignments),
            "n_clusters_used": len(unique_clusters),
            "n_clusters_total": self.n_clusters,
            "min_pairs_per_cluster": int(cluster_counts.min()),
            "max_pairs_per_cluster": int(cluster_counts.max()),
            "mean_pairs_per_cluster": float(mean_count),
            "std_pairs_per_cluster": float(std_count),
            "distribution": cluster_counts.to_dict()
        }
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "statistics": statistics
        }


def link_sources(
    df: pd.DataFrame,
    cluster_parser: ClusterParser,
    keywords_col: str = "keywords",
    pair_id_col: str = "pair_id",
    output_col: str = "cluster_id",
    fallback_strategy: str = "largest",
    show_progress: bool = True
) -> pd.DataFrame:
    """Convenience function to link pairs to clusters.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with question-source pairs and keywords
    cluster_parser : ClusterParser
        Parser for cluster metadata
    keywords_col : str, optional
        Column name containing keyword lists (default: "keywords")
    pair_id_col : str, optional
        Column name containing pair IDs (default: "pair_id")
    output_col : str, optional
        Output column name for cluster assignments (default: "cluster_id")
    fallback_strategy : str, optional
        Strategy for pairs without keyword matches (default: "largest")
    show_progress : bool, optional
        Show progress bar (default: True)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with cluster_id column added
    
    Examples
    --------
    >>> from RAG_supporters.dataset import ClusterParser, link_sources
    >>> import pandas as pd
    >>> 
    >>> parser = ClusterParser("clusters.json")
    >>> df = pd.read_csv("merged.csv")
    >>> df_linked = link_sources(df, parser)
    >>> print(df_linked[["pair_id", "cluster_id"]].head())
    """
    linker = SourceClusterLinker(
        cluster_parser=cluster_parser,
        fallback_strategy=fallback_strategy
    )
    
    return linker.link_dataframe(
        df=df,
        keywords_col=keywords_col,
        pair_id_col=pair_id_col,
        output_col=output_col,
        show_progress=show_progress
    )
