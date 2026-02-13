"""Clustering operations for keyword-based cluster assignment.

This module provides tools for parsing cluster metadata from KeywordClusterer
and linking data items to clusters via keyword matching.

Key Features:
- Parse KeywordClusterer JSON format
- Keyword-to-cluster matching (exact and fuzzy)
- Source-cluster linking via majority voting
- Fallback strategies for unmatched items
- Cluster coverage statistics

Examples
--------
>>> from RAG_supporters.clustering_ops import ClusterParser, SourceClusterLinker
>>>
>>> # Parse cluster metadata
>>> parser = ClusterParser("clusters.json")
>>> cluster_id = parser.match_keyword("machine learning")
>>>
>>> # Link pairs to clusters
>>> linker = SourceClusterLinker(parser)
>>> df_linked = linker.link_dataframe(df, keywords_col="keywords")
>>> print(df_linked[["pair_id", "cluster_id"]].head())
"""

from .parse_clusters import ClusterParser, parse_clusters
from .link_sources import SourceClusterLinker, link_sources

__all__ = [
    "ClusterParser",
    "parse_clusters",
    "SourceClusterLinker",
    "link_sources",
]
