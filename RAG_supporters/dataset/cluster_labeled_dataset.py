"""PyTorch Dataset for cluster-labeled domain assessment data."""

import logging
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
        mmap_mode: str = 'r'
    ):
        """
        Initialize cluster-labeled dataset.
        
        Args:
            dataset_dir: Directory containing dataset.db and embedding files
            label_type: Which label to return ('source', 'steering', 'combined')
            return_metadata: Whether to return metadata dict
            mmap_mode: Mode for numpy memmap ('r', 'r+', 'w+', 'c')
        """
        self.dataset_dir = Path(dataset_dir)
        self.label_type = label_type
        self.return_metadata = return_metadata
        self.mmap_mode = mmap_mode
        
        if label_type not in ['source', 'steering', 'combined']:
            raise ValueError(
                f"label_type must be 'source', 'steering', or 'combined', got '{label_type}'"
            )
        
        # Open storage
        db_path = self.dataset_dir / "dataset.db"
        if not db_path.exists():
            raise FileNotFoundError(f"Dataset database not found: {db_path}")
        
        self.storage = SQLiteStorageManager(db_path)
        
        # Load embedding files
        self.base_embeddings = self._load_embeddings('base')
        self.steering_embeddings = self._load_embeddings('steering')
        
        # Cache samples
        self.samples = self.storage.get_all_samples()
        
        # Load dataset metadata
        self.n_clusters = self.storage.get_dataset_info('n_clusters')
        
        logging.info(f"Loaded ClusterLabeledDataset from {dataset_dir}")
        logging.info(f"  Samples: {len(self.samples)}")
        logging.info(f"  Clusters: {self.n_clusters}")
        logging.info(f"  Label type: {label_type}")
        logging.info(f"  Base embeddings shape: {self.base_embeddings.shape}")
        logging.info(f"  Steering embeddings shape: {self.steering_embeddings.shape}")
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]]:
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
        sample = self.samples[idx]
        embedding_idx = sample['embedding_idx']
        
        # Get embeddings
        base_emb = torch.from_numpy(self.base_embeddings[embedding_idx].copy())
        steering_emb = torch.from_numpy(self.steering_embeddings[embedding_idx].copy())
        
        # Get label based on type
        if self.label_type == 'source':
            label = torch.from_numpy(sample['source_label'].copy())
        elif self.label_type == 'steering':
            label = torch.from_numpy(sample['steering_label'].copy())
        else:  # combined
            label = torch.from_numpy(sample['combined_label'].copy())
        
        if self.return_metadata:
            metadata = {
                'sample_id': sample['sample_id'],
                'sample_type': sample['sample_type'],
                'text': sample['text'],
                'chroma_id': sample['chroma_id'],
                'suggestions': sample['suggestions'],
                'steering_mode': sample['steering_mode'],
                'source_label': torch.from_numpy(sample['source_label'].copy()),
                'steering_label': torch.from_numpy(sample['steering_label'].copy()),
                'combined_label': torch.from_numpy(sample['combined_label'].copy())
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
        combined_label: Optional[np.ndarray] = None
    ):
        """
        Update labels for a sample (for manual correction).
        
        Args:
            sample_id: Sample ID from database
            source_label: New source label
            steering_label: New steering label
            combined_label: New combined label
        """
        self.storage.update_labels(
            sample_id, source_label, steering_label, combined_label
        )
        # Refresh samples cache
        self.samples = self.storage.get_all_samples()
        logging.info(f"Updated labels for sample {sample_id}")
    
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
            raise ValueError(f"No embedding file registered for type '{embedding_type}'")
        
        file_path = Path(info['file_path'])
        if not file_path.exists():
            raise FileNotFoundError(f"Embedding file not found: {file_path}")
        
        # Load with memory mapping
        embeddings = np.load(str(file_path), mmap_mode=self.mmap_mode)
        
        logging.info(f"Loaded {embedding_type} embeddings: {file_path} (shape={embeddings.shape})")
        
        return embeddings
    
    def close(self):
        """Close storage connections."""
        self.storage.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    @staticmethod
    def create_from_csvs(
        csv_paths: Union[str, Path, list],
        clustering_json_path: Union[str, Path],
        output_dir: Union[str, Path],
        embedding_model: Any,
        **builder_kwargs
    ) -> "ClusterLabeledDataset":
        """
        Factory method to build and load dataset from CSVs.
        
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
            DomainAssessmentDatasetBuilder
        )
        
        builder = DomainAssessmentDatasetBuilder(
            csv_paths=csv_paths,
            clustering_json_path=clustering_json_path,
            output_dir=output_dir,
            embedding_model=embedding_model,
            **builder_kwargs
        )
        builder.build()
        builder.close()
        
        return ClusterLabeledDataset(output_dir)
