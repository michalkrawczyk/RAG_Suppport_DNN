"""Simplified steering dataset for serving data."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from langchain_core.embeddings.embeddings import Embeddings
from torch.utils.data import Dataset

from ...clustering.clustering_data import ClusteringData
from ..utils.dataset_loader import filter_suggestions, parse_suggestions_safe
from .cache_manager import CacheManager
from .dataset_builder import DatasetBuilder
from .steering_config import SteeringConfig, SteeringMode
from .steering_generator import SteeringGenerator


class SteeringDataset(Dataset):
    """
    PyTorch Dataset for serving steering data.
    
    Simplified to only handle data serving - all building/computation
    is delegated to DatasetBuilder.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        parsed_suggestions: List[List[str]],
        unique_suggestions: List[str],
        text_embeddings: Optional[List[Dict[str, np.ndarray]]] = None,
        suggestion_embeddings: Optional[Dict[str, np.ndarray]] = None,
        clustering_data: Optional[ClusteringData] = None,
        steering_config: Optional[SteeringConfig] = None,
        embedding_model: Optional[Embeddings] = None,
        source_col: str = "source",
        question_col: str = "question",
        suggestions_col: str = "suggestions",
        min_confidence: float = 0.0,
        suggestion_types: Optional[List[str]] = None,
        llm_steering_texts: Optional[Dict[int, str]] = None,
        cluster_descriptor_embeddings: Optional[Dict[int, np.ndarray]] = None,
        llm_steering_embeddings: Optional[Dict[int, np.ndarray]] = None,
        steering_embeddings: Optional[Dict[int, np.ndarray]] = None,
        sample_weights: Optional[Dict[int, float]] = None,
        balance_clusters: bool = False,
        return_embeddings: bool = True,
    ):
        """
        Initialize steering dataset.
        
        Args:
            df: DataFrame with source data
            parsed_suggestions: Pre-parsed suggestions for each sample
            unique_suggestions: List of unique suggestions
            text_embeddings: Pre-computed text embeddings
            suggestion_embeddings: Pre-computed suggestion embeddings
            clustering_data: Cluster data (labels, descriptors)
            steering_config: Steering configuration
            embedding_model: Model for on-the-fly embedding computation
            source_col: Column name for source texts
            question_col: Column name for question texts
            suggestions_col: Column name for suggestions
            min_confidence: Minimum confidence for filtering suggestions
            suggestion_types: Allowed suggestion types
            llm_steering_texts: LLM-generated steering texts
            cluster_descriptor_embeddings: Pre-computed cluster descriptor embeddings
            llm_steering_embeddings: Pre-computed LLM steering embeddings
            steering_embeddings: Pre-computed steering embeddings
            sample_weights: Manual sample weights
            balance_clusters: Auto-compute weights for cluster balancing
            return_embeddings: Whether to return embeddings (vs raw text)
        """
        self.df = df
        self.parsed_suggestions = parsed_suggestions
        self.unique_suggestions = unique_suggestions
        self.text_embeddings = text_embeddings
        self.suggestion_embeddings = suggestion_embeddings
        self.clustering_data = clustering_data
        self.steering_config = steering_config or SteeringConfig()
        self.embedding_model = embedding_model
        
        self.source_col = source_col
        self.question_col = question_col
        self.suggestions_col = suggestions_col
        self.min_confidence = min_confidence
        self.suggestion_types = suggestion_types
        
        self.llm_steering_texts = llm_steering_texts
        self.sample_weights = sample_weights or {}
        self.balance_clusters = balance_clusters
        self.return_embeddings = return_embeddings
        
        # Initialize steering generator
        embedding_dim = None
        if text_embeddings and len(text_embeddings) > 0:
            embedding_dim = text_embeddings[0]["question"].shape[0]
        
        self.steering_generator = SteeringGenerator(
            config=self.steering_config,
            clustering_data=self.clustering_data,
            embedding_model=self.embedding_model,
            embedding_dim=embedding_dim,
            text_embeddings_cache=self.text_embeddings,
            suggestion_embeddings_cache=self.suggestion_embeddings,
            steering_embeddings_cache=steering_embeddings,
            cluster_descriptor_embeddings_cache=cluster_descriptor_embeddings,
            llm_steering_embeddings_cache=llm_steering_embeddings,
            llm_steering_texts=self.llm_steering_texts,
        )
        
        # Cluster weight computation cache
        self._cluster_weight_cache = {}
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with sample data
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")
        
        row = self.df.iloc[idx]
        suggestions = self.parsed_suggestions[idx]
        
        # Determine if we should return triplets
        return_triplets = (
            len(self.steering_config.mode) > 0 and
            self.return_embeddings
        )
        
        if return_triplets:
            return self._get_triplet(idx, row, suggestions)
        elif self.return_embeddings:
            return self._get_embeddings(idx, row, suggestions)
        else:
            return self._get_text(idx, row, suggestions)
    
    def _get_triplet(self, idx: int, row: Any, suggestions: List[str]) -> Dict[str, Any]:
        """Get sample in triplet mode."""
        # Get base embedding (question)
        if self.text_embeddings is not None:
            text_emb = self.text_embeddings[idx]
            question_emb = text_emb["question"]
            source_emb = text_emb["source"]
        else:
            question_text = str(row[self.question_col])
            source_text = str(row[self.source_col])
            question_emb = np.array(
                self.embedding_model.embed_query(question_text), dtype=np.float32
            )
            source_emb = np.array(
                self.embedding_model.embed_query(source_text), dtype=np.float32
            )
        
        # Get steering embedding
        steering_emb = self.steering_generator.generate(idx, suggestions)
        
        # Get target
        target = self.steering_generator.generate_target(idx)
        
        # Build metadata
        actual_steering_mode = self.steering_generator.select_mode(idx)
        metadata = {
            "steering_mode": actual_steering_mode.value if actual_steering_mode else None,
            "suggestion_texts": suggestions,
            "source_text": str(row[self.source_col]),
            "question_text": str(row[self.question_col]),
            "sample_index": idx,
        }
        
        # Add cluster info if available
        if self.clustering_data and self.clustering_data.labels:
            assignment = self.clustering_data.get_label(idx)
            if assignment is not None:
                metadata["cluster_assignment"] = assignment
                cluster_id = self.clustering_data.get_primary_cluster(idx)
                if cluster_id is not None:
                    descriptors = self.clustering_data.get_descriptors(cluster_id)
                    if descriptors:
                        metadata["cluster_descriptors"] = descriptors
        
        # Add LLM steering text if available
        if self.llm_steering_texts and idx in self.llm_steering_texts:
            metadata["llm_steering_text"] = self.llm_steering_texts[idx]
        
        result = {
            "base_embedding": torch.from_numpy(question_emb),
            "source_embedding": torch.from_numpy(source_emb),
            "idx": idx,
            "metadata": metadata,
        }
        
        if steering_emb is not None:
            result["steering_embedding"] = torch.from_numpy(steering_emb)
        
        if target is not None:
            if isinstance(target, np.ndarray):
                result["target"] = torch.from_numpy(target)
            else:
                result["target"] = torch.tensor(target, dtype=torch.long)
        
        # Add sample weight if available
        if idx in self.sample_weights:
            result["weight"] = torch.tensor(self.sample_weights[idx], dtype=torch.float32)
        elif self.balance_clusters and target is not None:
            result["weight"] = self._compute_balanced_weight(idx)
        
        return result
    
    def _get_embeddings(self, idx: int, row: Any, suggestions: List[str]) -> Dict[str, Any]:
        """Get sample in embedding mode."""
        # Get text embeddings
        if self.text_embeddings is not None:
            text_emb = self.text_embeddings[idx]
            source_emb = text_emb["source"]
            question_emb = text_emb["question"]
        else:
            source_text = str(row[self.source_col])
            question_text = str(row[self.question_col])
            source_emb = np.array(
                self.embedding_model.embed_query(source_text), dtype=np.float32
            )
            question_emb = np.array(
                self.embedding_model.embed_query(question_text), dtype=np.float32
            )
        
        # Get suggestion embeddings
        suggestion_embeds = []
        for term in suggestions:
            if self.suggestion_embeddings and term in self.suggestion_embeddings:
                emb = self.suggestion_embeddings[term]
            else:
                emb = np.array(
                    self.embedding_model.embed_query(term), dtype=np.float32
                )
        
            suggestion_embeds.append(emb)
        
        return {
            "source": torch.from_numpy(source_emb),
            "question": torch.from_numpy(question_emb),
            "suggestions": [torch.from_numpy(emb) for emb in suggestion_embeds],
            "suggestion_texts": suggestions,
            "idx": idx,
        }
    
    def _get_text(self, idx: int, row: Any, suggestions: List[str]) -> Dict[str, Any]:
        """Get sample in text mode."""
        return {
            "source": str(row[self.source_col]),
            "question": str(row[self.question_col]),
            "suggestions": suggestions,
            "suggestion_texts": suggestions,
            "idx": idx,
        }
    
    def _compute_balanced_weight(self, idx: int) -> torch.Tensor:
        """Compute balanced weight for cluster balancing."""
        if not self.clustering_data or not self.clustering_data.labels:
            return torch.tensor(1.0, dtype=torch.float32)
        
        cluster_id = self.clustering_data.get_primary_cluster(idx)
        if cluster_id is None:
            return torch.tensor(1.0, dtype=torch.float32)
        
        # Check cache
        if cluster_id in self._cluster_weight_cache:
            return self._cluster_weight_cache[cluster_id]
        
        # Compute weight: inverse frequency
        cluster_counts = {}
        for label in self.clustering_data.labels.values():
            if isinstance(label, int):
                cid = label
            elif isinstance(label, list):
                cid = int(np.argmax(label))
            else:
                continue
            cluster_counts[cid] = cluster_counts.get(cid, 0) + 1
        
        total_samples = len(self.clustering_data.labels)
        cluster_freq = cluster_counts.get(cluster_id, 1) / total_samples
        weight = 1.0 / (cluster_freq + 1e-6)  # Inverse frequency
        
        # Normalize weights to have mean=1.0
        all_weights = [1.0 / (cluster_counts.get(cid, 1) / total_samples + 1e-6) 
                       for cid in cluster_counts.keys()]
        mean_weight = sum(all_weights) / len(all_weights)
        weight = weight / mean_weight
        
        weight_tensor = torch.tensor(weight, dtype=torch.float32)
        self._cluster_weight_cache[cluster_id] = weight_tensor
        return weight_tensor
    
    @classmethod
    def from_cache(
        cls,
        cache_dir: Union[str, Path],
        return_embeddings: bool = True,
    ) -> "SteeringDataset":
        """
        Load dataset from cache.
        
        Args:
            cache_dir: Cache directory path
            return_embeddings: Whether to load embeddings
            
        Returns:
            SteeringDataset instance
        """
        cache_manager = CacheManager(cache_dir)
        
        if not cache_manager.exists():
            raise ValueError(f"Cache directory does not exist: {cache_dir}")
        
        # Load all data
        data = cache_manager.load_all(return_embeddings=return_embeddings)
        
        # Extract metadata
        metadata = data["metadata"]
        
        # Reconstruct steering config
        steering_modes = metadata.get("steering_modes")
        steering_config = None
        if steering_modes:
            mode_list = [(SteeringMode(mode), prob) for mode, prob in steering_modes]
            steering_config = SteeringConfig(
                mode=mode_list,
                multi_label_mode=metadata.get("multi_label_mode", "hard"),
            )
        
        # Reconstruct clustering data
        clustering_data = None
        cluster_labels = data.get("cluster_labels")
        cluster_descriptors = data.get("cluster_descriptors")
        if cluster_labels or cluster_descriptors:
            # Infer n_clusters
            n_clusters = None
            if cluster_labels:
                first_label = next(iter(cluster_labels.values()))
                if isinstance(first_label, int):
                    n_clusters = max(cluster_labels.values()) + 1
                elif isinstance(first_label, list):
                    n_clusters = len(first_label)
            
            if n_clusters:
                clustering_data = ClusteringData(
                    n_clusters=n_clusters,
                    labels=cluster_labels,
                    descriptors=cluster_descriptors,
                )
        
        # Note: DataFrame not saved in cache, need to reconstruct or pass separately
        # For now, create a minimal DataFrame
        df = pd.DataFrame({"idx": list(range(metadata["length"]))})
        
        return cls(
            df=df,
            parsed_suggestions=data["parsed_suggestions"],
            unique_suggestions=data["unique_suggestions"],
            text_embeddings=data.get("text_embeddings"),
            suggestion_embeddings=data.get("suggestion_embeddings"),
            clustering_data=clustering_data,
            steering_config=steering_config,
            cluster_descriptor_embeddings=data.get("cluster_descriptor_embeddings"),
            llm_steering_embeddings=data.get("llm_steering_embeddings"),
            steering_embeddings=data.get("steering_embeddings"),
            llm_steering_texts=data.get("llm_steering_texts"),
            return_embeddings=return_embeddings,
        )
    
    @classmethod
    def from_builder(cls, builder: DatasetBuilder) -> "SteeringDataset":
        """
        Create dataset from a DatasetBuilder.
        
        Args:
            builder: DatasetBuilder instance with built data
            
        Returns:
            SteeringDataset instance
        """
        return cls(
            df=builder.df,
            parsed_suggestions=builder.parsed_suggestions,
            unique_suggestions=builder.unique_suggestions,
            text_embeddings=builder.text_embeddings,
            suggestion_embeddings=builder.suggestion_embeddings,
            clustering_data=builder.clustering_data,
            steering_config=builder.steering_config,
            embedding_model=builder.embedding_model,
            source_col=builder.source_col,
            question_col=builder.question_col,
            suggestions_col=builder.suggestions_col,
            min_confidence=builder.min_confidence,
            suggestion_types=builder.suggestion_types,
            llm_steering_texts=builder.llm_steering_texts,
            cluster_descriptor_embeddings=builder.cluster_descriptor_embeddings,
            llm_steering_embeddings=builder.llm_steering_embeddings,
            steering_embeddings=builder.steering_embeddings,
            return_embeddings=builder.return_embeddings,
        )
    
    def get_missing_suggestions_count(self) -> int:
        """Get count of missing suggestions."""
        return self.steering_generator.get_missing_suggestions_count()
    
    def report_statistics(self):
        """Report dataset statistics."""
        logging.info("=" * 60)
        logging.info("Dataset Statistics")
        logging.info("=" * 60)
        logging.info(f"Total samples: {len(self)}")
        logging.info(f"Unique suggestions: {len(self.unique_suggestions)}")
        
        if self.steering_config and len(self.steering_config.mode) > 0:
            logging.info(f"Steering modes: {self.steering_config.mode}")
            logging.info(f"Multi-label mode: {self.steering_config.multi_label_mode}")
        
        missing_count = self.get_missing_suggestions_count()
        if missing_count > 0:
            pct = (missing_count / len(self)) * 100
            logging.warning(
                f"Missing suggestions: {missing_count} ({pct:.1f}% of samples)"
            )
        
        if self.clustering_data and self.clustering_data.labels:
            logging.info(f"Cluster assignments: {len(self.clustering_data.labels)}")
            logging.info(f"Number of clusters: {self.clustering_data.n_clusters}")
        
        logging.info("=" * 60)
