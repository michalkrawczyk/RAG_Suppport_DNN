"""Clustering data handling component."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


@dataclass
class ClusteringData:
    """
    Container for cluster-related data.
    
    Attributes:
        n_clusters: Number of clusters
        labels: Mapping of sample indices to cluster labels (hard or soft)
        descriptors: Mapping of cluster IDs to topic descriptors
        centroids: Cluster centroids as numpy array
        metadata: Additional clustering metadata
    """
    
    n_clusters: int
    labels: Optional[Dict[int, Union[int, List[float]]]] = None
    descriptors: Optional[Dict[int, List[str]]] = None
    centroids: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "ClusteringData":
        """
        Load clustering data from JSON file.
        
        Args:
            path: Path to clustering results JSON file
            
        Returns:
            ClusteringData instance
            
        Raises:
            ValueError: If file not found or invalid format
        """
        path = Path(path)
        if not path.exists():
            raise ValueError(f"Clustering results file not found: {path}")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Extract metadata
        metadata = data.get('metadata', {})
        n_clusters = metadata.get('n_clusters')
        if n_clusters is None:
            raise ValueError("Clustering JSON must contain 'metadata.n_clusters'")
        
        # Extract cluster descriptors
        descriptors = cls._extract_descriptors(data)
        
        # Extract centroids if available
        centroids = None
        if 'centroids' in data:
            centroids = np.array(data['centroids'], dtype=np.float32)
        
        logging.info(f"Loaded clustering results from {path}")
        logging.info(f"  - n_clusters: {n_clusters}")
        logging.info(f"  - n_keywords: {metadata.get('n_keywords', 'N/A')}")
        logging.info(f"  - descriptors: {len(descriptors)} clusters")
        
        return cls(
            n_clusters=n_clusters,
            labels=None,  # Labels typically come from assign_batch results
            descriptors=descriptors,
            centroids=centroids,
            metadata=metadata
        )
    
    @staticmethod
    def _extract_descriptors(data: Dict[str, Any]) -> Dict[int, List[str]]:
        """Extract cluster descriptors from clustering JSON."""
        cluster_stats = data.get('cluster_stats', {})
        descriptors = {}
        
        for cluster_id_str, stats in cluster_stats.items():
            cluster_id = int(cluster_id_str)
            topic_descriptors = stats.get('topic_descriptors', [])
            if topic_descriptors:
                descriptors[cluster_id] = topic_descriptors
        
        return descriptors
    
    def get_label(self, idx: int) -> Optional[Union[int, List[float]]]:
        """
        Get cluster label for a sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Cluster label (int for hard, List[float] for soft) or None
        """
        if self.labels is None:
            return None
        return self.labels.get(idx)
    
    def get_primary_cluster(self, idx: int) -> Optional[int]:
        """
        Get primary cluster ID for a sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Primary cluster ID or None
        """
        if self.labels is None or idx not in self.labels:
            return None
        
        assignment = self.labels[idx]
        if isinstance(assignment, int):
            return assignment
        elif isinstance(assignment, list) and assignment:
            # For soft labels, return index with highest probability
            return int(np.argmax(assignment))
        return None
    
    def get_descriptors(self, cluster_id: int) -> Optional[List[str]]:
        """
        Get topic descriptors for a cluster.
        
        Args:
            cluster_id: Cluster ID
            
        Returns:
            List of descriptor strings or None
        """
        if self.descriptors is None:
            return None
        return self.descriptors.get(cluster_id)
    
    def get_centroid(self, cluster_id: int) -> Optional[np.ndarray]:
        """
        Get centroid for a cluster.
        
        Args:
            cluster_id: Cluster ID
            
        Returns:
            Centroid vector or None
        """
        if self.centroids is None or cluster_id >= len(self.centroids):
            return None
        return self.centroids[cluster_id]
    
    def set_labels(self, labels: Dict[int, Union[int, List[float]]]):
        """
        Set cluster labels.
        
        Args:
            labels: Mapping of sample indices to cluster labels
        """
        self.labels = labels
        logging.info(f"Set cluster labels for {len(labels)} samples")
    
    def infer_n_clusters_from_labels(self) -> int:
        """
        Infer number of clusters from labels.
        
        Returns:
            Number of clusters
            
        Raises:
            ValueError: If labels are not set or invalid
        """
        if not self.labels:
            raise ValueError("No cluster labels available")
        
        # Get first assignment to check format
        first_assignment = next(iter(self.labels.values()))
        
        if isinstance(first_assignment, int):
            # For int labels, find max + 1
            return max(self.labels.values()) + 1
        elif isinstance(first_assignment, list):
            # For probability distributions, use length
            return len(first_assignment)
        else:
            raise ValueError(f"Unexpected cluster label type: {type(first_assignment)}")
    
    def validate(self):
        """
        Validate clustering data consistency.
        
        Raises:
            ValueError: If data is inconsistent
        """
        if self.n_clusters <= 0:
            raise ValueError(f"Invalid n_clusters: {self.n_clusters}")
        
        if self.descriptors:
            max_cluster_id = max(self.descriptors.keys())
            if max_cluster_id >= self.n_clusters:
                raise ValueError(
                    f"Descriptor cluster ID {max_cluster_id} >= n_clusters {self.n_clusters}"
                )
        
        if self.centroids is not None:
            if len(self.centroids) != self.n_clusters:
                raise ValueError(
                    f"Centroids length {len(self.centroids)} != n_clusters {self.n_clusters}"
                )
        
        logging.info("Clustering data validation passed")
