"""
Backward compatibility layer for BaseDomainAssignDataset.

This module provides the legacy BaseDomainAssignDataset interface while delegating
to the new modular architecture in the steering/ package.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from langchain_core.embeddings.embeddings import Embeddings

from .steering import (
    CacheManager,
    ClusteringData,
    DatasetBuilder,
    SteeringConfig,
    SteeringDataset,
    SteeringMode,
)

# Re-export SteeringMode for backward compatibility
__all__ = ["SteeringMode", "BaseDomainAssignDataset"]


class BaseDomainAssignDataset:
    """
    Backward compatibility wrapper for the legacy BaseDomainAssignDataset interface.
    
    This class delegates to the new modular architecture (SteeringDataset + DatasetBuilder)
    while maintaining the original API for existing code.
    
    Usage (Legacy API - still supported):
        dataset = BaseDomainAssignDataset(
            df=df,
            embedding_model=embeddings,
            steering_mode=SteeringMode.CLUSTER_DESCRIPTOR,
            cluster_labels=cluster_labels,
            cluster_descriptors=cluster_descriptors,
            multi_label_mode="soft"
        ).build()
        
        sample = dataset[0]
    
    New users should prefer the modular API:
        from RAG_supporters.dataset.steering import (
            ClusteringData, SteeringConfig, DatasetBuilder, SteeringDataset
        )
        
        clustering_data = ClusteringData.from_json("clusters.json")
        config = SteeringConfig.from_single_mode(SteeringMode.CLUSTER_DESCRIPTOR)
        builder = DatasetBuilder(df, model, clustering_data, config)
        builder.build()
        dataset = SteeringDataset.from_builder(builder)
    """

    def __init__(
        self,
        df: Union[pd.DataFrame, str, Path],
        embedding_model: Optional[Embeddings] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        source_col: str = "source",
        question_col: str = "question",
        suggestions_col: str = "suggestions",
        min_confidence: float = 0.0,
        suggestion_types: Optional[List[str]] = None,
        return_embeddings: bool = True,
        chunksize: int = 10000,
        embedding_model_name: Optional[str] = None,
        # Cluster steering parameters
        steering_mode: Optional[Union[SteeringMode, str, List[tuple]]] = None,
        clustering_results_path: Optional[Union[str, Path]] = None,
        cluster_labels: Optional[Union[Dict[int, int], Dict[int, List[float]]]] = None,
        cluster_descriptors: Optional[Dict[int, List[str]]] = None,
        llm_steering_texts: Optional[Union[Dict[int, str], str, Path]] = None,
        return_triplets: bool = True,
        multi_label_mode: str = "hard",
        steering_weights: Optional[Dict[str, float]] = None,
        # Sample weighting parameters
        sample_weights: Optional[Dict[int, float]] = None,
        balance_clusters: bool = False,
        read_csv_chunksize: Optional[int] = None,
    ):
        """
        Initialize the dataset builder with legacy API.
        
        This constructor translates legacy parameters to the new modular architecture.
        """
        self.logger = logging.getLogger(__name__)
        
        # Store parameters for build()
        self.df = df
        self.embedding_model = embedding_model
        self.cache_dir = cache_dir
        self.source_col = source_col
        self.question_col = question_col
        self.suggestions_col = suggestions_col
        self.min_confidence = min_confidence
        self.suggestion_types = suggestion_types
        self.return_embeddings = return_embeddings
        self.chunksize = chunksize
        self.embedding_model_name = embedding_model_name
        self.return_triplets = return_triplets
        self.sample_weights = sample_weights
        self.balance_clusters = balance_clusters
        self.read_csv_chunksize = read_csv_chunksize
        
        # Build ClusteringData
        self.clustering_data = None
        if clustering_results_path:
            self.clustering_data = ClusteringData.from_json(clustering_results_path)
            # Override with provided values if given
            if cluster_labels is not None:
                self.clustering_data.set_labels(cluster_labels)
            if cluster_descriptors is not None:
                self.clustering_data.descriptors = cluster_descriptors
        elif cluster_labels is not None or cluster_descriptors is not None:
            # Create ClusteringData from provided parameters
            self.clustering_data = ClusteringData(
                n_clusters=0,  # Will be inferred
                labels=cluster_labels or {},
                descriptors=cluster_descriptors or {},
                centroids=None,
                metadata={}
            )
        
        # Build SteeringConfig
        if steering_mode:
            self.steering_config = self._create_steering_config(
                steering_mode, multi_label_mode, steering_weights
            )
        else:
            self.steering_config = None
        
        # Store LLM steering texts
        self.llm_steering_texts = llm_steering_texts
        
        # Will be set by build()
        self._dataset = None
        self._builder = None
    
    def _create_steering_config(
        self,
        steering_mode: Union[SteeringMode, str, List[tuple]],
        multi_label_mode: str,
        steering_weights: Optional[Dict[str, float]]
    ) -> SteeringConfig:
        """Create SteeringConfig from legacy parameters."""
        # Handle single mode
        if isinstance(steering_mode, (SteeringMode, str)):
            if isinstance(steering_mode, str):
                steering_mode = SteeringMode(steering_mode)
            return SteeringConfig.from_single_mode(
                steering_mode,
                multi_label_mode=multi_label_mode,
                mixed_weights=steering_weights
            )
        
        # Handle multi-mode with probabilities
        if isinstance(steering_mode, list):
            # Convert string modes to enum
            mode_list = []
            for mode, prob in steering_mode:
                if isinstance(mode, str):
                    mode = SteeringMode(mode)
                mode_list.append((mode, prob))
            
            return SteeringConfig(
                mode_list=mode_list,
                multi_label_mode=multi_label_mode,
                mixed_weights=steering_weights
            )
        
        raise ValueError(f"Invalid steering_mode type: {type(steering_mode)}")
    
    def build(self) -> "BaseDomainAssignDataset":
        """
        Build the dataset by computing embeddings and caching if needed.
        
        Returns:
            self for chaining
        """
        # Create DatasetBuilder
        self._builder = DatasetBuilder(
            df=self.df,
            embedding_model=self.embedding_model,
            clustering_data=self.clustering_data,
            steering_config=self.steering_config,
            cache_dir=self.cache_dir,
            source_col=self.source_col,
            question_col=self.question_col,
            suggestions_col=self.suggestions_col,
            min_confidence=self.min_confidence,
            suggestion_types=self.suggestion_types,
            chunksize=self.chunksize,
            embedding_model_name=self.embedding_model_name,
            llm_steering_texts=self.llm_steering_texts,
            sample_weights=self.sample_weights,
            balance_clusters=self.balance_clusters,
            read_csv_chunksize=self.read_csv_chunksize
        )
        
        # Build the dataset
        self._builder.build()
        
        # Create SteeringDataset
        self._dataset = SteeringDataset.from_builder(self._builder)
        
        return self
    
    def __len__(self):
        """Return number of samples in the dataset."""
        if self._dataset is None:
            raise RuntimeError("Dataset not built. Call build() first.")
        return len(self._dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample from the dataset."""
        if self._dataset is None:
            raise RuntimeError("Dataset not built. Call build() first.")
        return self._dataset[idx]
    
    def report_statistics(self):
        """Report dataset statistics including missing suggestions."""
        if self._dataset:
            self._dataset.report_statistics()
    
    def get_missing_suggestions_count(self) -> int:
        """Get count of samples with missing suggestions."""
        if self._dataset:
            return self._dataset.get_missing_suggestions_count()
        return 0


# Backward compatibility: CachedDomainAssignDataset removed
# Users should use SteeringDataset.from_cache() instead
def build_and_load_dataset(cache_dir: Union[str, Path]):
    """
    DEPRECATED: Load a cached dataset.
    
    Use SteeringDataset.from_cache(cache_dir) instead.
    
    Args:
        cache_dir: Directory containing cached dataset
        
    Returns:
        SteeringDataset instance
    """
    logging.warning(
        "build_and_load_dataset() is deprecated. "
        "Use SteeringDataset.from_cache(cache_dir) instead."
    )
    return SteeringDataset.from_cache(cache_dir)
