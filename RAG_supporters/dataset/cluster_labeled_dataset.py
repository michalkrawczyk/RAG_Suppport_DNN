"""PyTorch Dataset for cluster-labeled domain assessment data."""

import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from RAG_supporters.dataset.sqlite_storage import SQLiteStorageManager


class ClusterLabeledDataset(Dataset):
    """
    PyTorch Dataset for domain assessment with 3-type labels.

    Returns triplets: (base_embedding, steering_embedding, label)

    Label types:
    - source_label: Label for base embedding (source/question)
    - steering_label: Label for steering embedding
    - combined_label: Weighted average for augmentation masking

    Storage: SQLite (metadata) + numpy memmap (embeddings).
    """

    def __init__(
        self,
        dataset_dir: Union[str, Path],
        label_type: str = "combined",
        return_metadata: bool = False,
        mmap_mode: str = "r",
        cache_size: int = 1000,
    ):
        """
        Initialize cluster-labeled dataset.

        Args:
            dataset_dir: Directory containing dataset.db and embedding files
            label_type: Which label to return ('source', 'steering', 'combined')
            return_metadata: Whether to return metadata dict
            mmap_mode: Mode for numpy memmap ('r', 'r+', 'w+', 'c')
            cache_size: Maximum number of samples to keep in LRU cache
        """
        self.dataset_dir = Path(dataset_dir)
        self.label_type = label_type
        self.return_metadata = return_metadata
        self.mmap_mode = mmap_mode
        self._cache_size = cache_size

        if label_type not in ["source", "steering", "combined"]:
            raise ValueError(
                f"label_type must be 'source', 'steering', or 'combined', got '{label_type}'"
            )

        # Open storage
        db_path = self.dataset_dir / "dataset.db"
        if not db_path.exists():
            raise FileNotFoundError(f"Dataset database not found: {db_path}")

        self.storage = SQLiteStorageManager(db_path)

        # Load embedding files
        self.base_embeddings = self._load_embeddings("base")
        self.steering_embeddings = self._load_embeddings("steering")

        # Validate embedding shapes match
        if self.base_embeddings.shape != self.steering_embeddings.shape:
            raise ValueError(
                f"Base and steering embeddings shape mismatch: "
                f"{self.base_embeddings.shape} vs {self.steering_embeddings.shape}"
            )

        # Cache for dataset size (avoids repeated DB queries)
        self._dataset_size = self.storage.get_dataset_size()
        
        # Validate dataset size matches embedding size
        if self._dataset_size > len(self.base_embeddings):
            raise ValueError(
                f"Dataset size ({self._dataset_size}) exceeds embedding array size "
                f"({len(self.base_embeddings)})"
            )
        
        # LRU cache for frequently accessed samples (thread-safe)
        self._sample_cache: OrderedDict[int, Dict[str, Any]] = OrderedDict()
        self._cache_lock = threading.Lock()  # Thread-safe for multi-worker DataLoader
        
        # Mapping from sample_id to index for efficient cache invalidation
        # Built incrementally as samples are accessed
        self._sample_id_to_idx: Dict[int, int] = {}
        
        # Cache statistics for monitoring
        self._cache_hits = 0
        self._cache_misses = 0

        # Load dataset metadata
        self.n_clusters = self.storage.get_dataset_info("n_clusters")

        logging.info(f"Loaded ClusterLabeledDataset from {dataset_dir}")
        logging.info(f"  Samples: {self._dataset_size}")
        logging.info(f"  Clusters: {self.n_clusters}")
        logging.info(f"  Label type: {label_type}")
        logging.info(f"  Cache size: {cache_size}")
        logging.info(f"  Base embeddings shape: {self.base_embeddings.shape}")
        logging.info(f"  Steering embeddings shape: {self.steering_embeddings.shape}")

    def __len__(self) -> int:
        """Return dataset size."""
        return self._dataset_size

    def __getitem__(self, idx: int) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]],
    ]:
        """
        Get sample at index.

        Args:
            idx: Sample index

        Returns:
            If return_metadata=False:
                (base_embedding, steering_embedding, label)
            If return_metadata=True:
                (base_embedding, steering_embedding, label, metadata)
        """
        # Validate index bounds
        if not 0 <= idx < self._dataset_size:
            raise IndexError(f"Index {idx} out of range [0, {self._dataset_size})")
        
        # Thread-safe cache access
        with self._cache_lock:
            if idx in self._sample_cache:
                # Cache hit - move to end (mark as recently used)
                self._cache_hits += 1
                self._sample_cache.move_to_end(idx)
                sample = self._sample_cache[idx]
            else:
                # Cache miss - query from storage
                self._cache_misses += 1
                sample = self.storage.get_sample_by_index(idx)
                if sample is None:
                    raise IndexError(f"Sample index {idx} not found in storage")
                
                # Build sample_id to index mapping incrementally
                self._sample_id_to_idx[sample["sample_id"]] = idx
                
                # Add to cache with LRU eviction
                if len(self._sample_cache) >= self._cache_size:
                    # Remove least recently used (first item)
                    removed_idx, removed_sample = self._sample_cache.popitem(last=False)
                    # Also remove from id mapping
                    self._sample_id_to_idx.pop(removed_sample["sample_id"], None)
                
                self._sample_cache[idx] = sample
        
        embedding_idx = sample["embedding_idx"]
        
        # Validate embedding index bounds
        if not 0 <= embedding_idx < len(self.base_embeddings):
            raise IndexError(
                f"Embedding index {embedding_idx} out of bounds "
                f"[0, {len(self.base_embeddings)})"
            )

        # Get embeddings (no unnecessary copy - torch.from_numpy handles it)
        base_emb = torch.from_numpy(self.base_embeddings[embedding_idx])
        steering_emb = torch.from_numpy(self.steering_embeddings[embedding_idx])

        # Get label based on type (no unnecessary copy)
        if self.label_type == "source":
            label = torch.from_numpy(sample["source_label"])
        elif self.label_type == "steering":
            label = torch.from_numpy(sample["steering_label"])
        else:  # combined
            label = torch.from_numpy(sample["combined_label"])

        if self.return_metadata:
            metadata = {
                "sample_id": sample["sample_id"],
                "sample_type": sample["sample_type"],
                "text": sample["text"],
                "chroma_id": sample["chroma_id"],
                "suggestions": sample["suggestions"],
                "steering_mode": sample["steering_mode"],
                "source_label": torch.from_numpy(sample["source_label"]),
                "steering_label": torch.from_numpy(sample["steering_label"]),
                "combined_label": torch.from_numpy(sample["combined_label"]),
            }
            return base_emb, steering_emb, label, metadata

        return base_emb, steering_emb, label

    def get_sample_by_id(self, sample_id: int) -> Optional[Dict[str, Any]]:
        """
        Get sample by database ID.

        Args:
            sample_id: Sample ID from database

        Returns:
            Sample dictionary or None
        """
        return self.storage.get_sample(sample_id)

    def update_labels(
        self,
        sample_id: int,
        source_label: Optional[np.ndarray] = None,
        steering_label: Optional[np.ndarray] = None,
        combined_label: Optional[np.ndarray] = None,
    ):
        """
        Update labels for a sample (for manual correction).

        Args:
            sample_id: Sample ID from database
            source_label: New source label
            steering_label: New steering label
            combined_label: New combined label
        """
        with self._cache_lock:
            # Invalidate cache FIRST (before DB update) to prevent race condition
            if sample_id in self._sample_id_to_idx:
                # We have the mapping - efficient invalidation
                idx = self._sample_id_to_idx[sample_id]
                if idx in self._sample_cache:
                    del self._sample_cache[idx]
                del self._sample_id_to_idx[sample_id]
                logging.debug(
                    f"Invalidated cache for sample {sample_id} (index {idx})"
                )
            else:
                # Sample not in mapping - search cache to be safe
                idx_to_remove = None
                for idx, cached_sample in self._sample_cache.items():
                    if cached_sample["sample_id"] == sample_id:
                        idx_to_remove = idx
                        break
                
                if idx_to_remove is not None:
                    del self._sample_cache[idx_to_remove]
                    logging.debug(
                        f"Invalidated cache for sample {sample_id} "
                        f"(index {idx_to_remove}, found by search)"
                    )
            
            # Now update DB (cache already invalidated)
            self.storage.update_labels(
                sample_id, source_label, steering_label, combined_label
            )
            
        logging.info(f"Updated labels for sample {sample_id}")

    def get_cache_stats(self) -> Dict[str, Union[int, float]]:
        """
        Get cache performance statistics.

        Returns:
            Dictionary with cache hits, misses, and hit rate
        """
        total_accesses = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_accesses if total_accesses > 0 else 0.0
        
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "total_accesses": total_accesses,
            "hit_rate": hit_rate,
            "cache_size": len(self._sample_cache),
            "max_cache_size": self._cache_size,
        }

    def _load_embeddings(self, embedding_type: str) -> np.ndarray:
        """
        Load embeddings as memory-mapped array.

        Args:
            embedding_type: 'base' or 'steering'

        Returns:
            Memory-mapped numpy array
        """
        info = self.storage.get_embedding_file_info(embedding_type)
        if info is None:
            raise ValueError(
                f"No embedding file registered for type '{embedding_type}'"
            )

        file_path = Path(info["file_path"])
        if not file_path.exists():
            raise FileNotFoundError(f"Embedding file not found: {file_path}")

        # Load with memory mapping
        embeddings = np.load(str(file_path), mmap_mode=self.mmap_mode)

        logging.info(
            f"Loaded {embedding_type} embeddings: {file_path} (shape={embeddings.shape})"
        )

        return embeddings

    def close(self):
        """Close storage connections and cleanup resources."""
        # Close database connection
        if hasattr(self, 'storage'):
            self.storage.close()
        
        # Clear caches
        if hasattr(self, '_cache_lock'):
            with self._cache_lock:
                if hasattr(self, '_sample_cache'):
                    self._sample_cache.clear()
                if hasattr(self, '_sample_id_to_idx'):
                    self._sample_id_to_idx.clear()
        
        # Log final cache statistics
        if hasattr(self, '_cache_hits'):
            stats = self.get_cache_stats()
            logging.info(f"Dataset closed. Final cache stats: {stats}")
        
        # Explicitly delete memmap references to help garbage collection
        if hasattr(self, 'base_embeddings'):
            del self.base_embeddings
        if hasattr(self, 'steering_embeddings'):
            del self.steering_embeddings

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and close storage."""
        self.close()

    def __del__(self):
        """Destructor - safety fallback for cleanup."""
        try:
            self.close()
        except:
            pass  # Silently fail in destructor

    @staticmethod
    def create_from_csvs(
        csv_paths: Union[str, Path, list],
        clustering_json_path: Union[str, Path],
        output_dir: Union[str, Path],
        embedding_model: Any,
        **builder_kwargs,
    ) -> "ClusterLabeledDataset":
        """
        Build and load dataset from CSV files.

        Args:
            csv_paths: Path(s) to CSV files
            clustering_json_path: Path to clustering JSON
            output_dir: Output directory
            embedding_model: Embedding model
            **builder_kwargs: Additional arguments for DomainAssessmentDatasetBuilder

        Returns:
            Loaded ClusterLabeledDataset
        """
        from RAG_supporters.dataset.domain_assessment_dataset_builder import (
            DomainAssessmentDatasetBuilder,
        )

        builder = DomainAssessmentDatasetBuilder(
            csv_paths=csv_paths,
            clustering_json_path=clustering_json_path,
            output_dir=output_dir,
            embedding_model=embedding_model,
            **builder_kwargs,
        )
        
        try:
            builder.build()
        finally:
            # Ensure builder is closed even if build() fails
            builder.close()

        return ClusterLabeledDataset(output_dir)