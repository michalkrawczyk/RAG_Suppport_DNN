"""Steering embedding generator with mode support and augmentations."""

import logging
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from RAG_supporters.augmentations.embedding import (
    random_noise_embedding,
    random_zero_embedding
)
from RAG_supporters.clustering.clustering_data import ClusteringData
from RAG_supporters.dataset.steering.steering_config import SteeringConfig, SteeringMode


class SteeringEmbeddingGenerator:
    """
    Generates steering embeddings based on configured modes.
    
    Supports:
    - SUGGESTION: Uses suggestion embeddings
    - CLUSTER_DESCRIPTOR: Uses cluster descriptor embeddings
    - LLM_GENERATED: Uses LLM-generated steering text embeddings
    - ZERO: Returns zero vector (no steering)
    - MIXED: Weighted combination of modes
    
    Integrates augmentations from embedding.py for noise injection.
    """
    
    def __init__(
        self,
        config: SteeringConfig,
        clustering_data: ClusteringData,
        embedding_model: Any,  # Model with encode() method
        suggestion_embeddings: Optional[Dict[str, np.ndarray]] = None,
        llm_steering_texts: Optional[Dict[str, str]] = None,
        augment_noise_prob: float = 0.0,
        augment_zero_prob: float = 0.0,
        augment_noise_level: float = 0.01
    ):
        """
        Initialize steering embedding generator.
        
        Args:
            config: Steering configuration with modes
            clustering_data: Cluster data with centroids and descriptors
            embedding_model: Model for encoding text to embeddings
            suggestion_embeddings: Pre-computed suggestion embeddings {suggestion: embedding}
            llm_steering_texts: LLM-generated steering texts {key: text}
            augment_noise_prob: Probability of applying noise augmentation
            augment_zero_prob: Probability of zeroing steering embedding
            augment_noise_level: Std deviation for noise augmentation
        """
        self.config = config
        self.clustering_data = clustering_data
        self.embedding_model = embedding_model
        self.suggestion_embeddings = suggestion_embeddings or {}
        self.llm_steering_texts = llm_steering_texts or {}
        self.augment_noise_prob = augment_noise_prob
        self.augment_zero_prob = augment_zero_prob
        self.augment_noise_level = augment_noise_level
        self.rng = random.Random(config.random_seed)
    
    def generate(
        self,
        sample_id: int,
        suggestions: Optional[List[str]] = None,
        cluster_id: Optional[int] = None
    ) -> Tuple[np.ndarray, SteeringMode]:
        """
        Generate steering embedding for a sample.
        
        Args:
            sample_id: Sample identifier
            suggestions: List of suggestion strings
            cluster_id: Primary cluster ID for descriptor mode
            
        Returns:
            Tuple of (steering_embedding, mode_used)
        """
        # Select mode based on configured probabilities
        mode = self._select_mode()
        
        # Generate embedding based on mode
        if mode == SteeringMode.ZERO:
            embedding = self._generate_zero_embedding()
        elif mode == SteeringMode.SUGGESTION:
            embedding = self._generate_suggestion_embedding(suggestions)
        elif mode == SteeringMode.CLUSTER_DESCRIPTOR:
            embedding = self._generate_descriptor_embedding(cluster_id)
        elif mode == SteeringMode.LLM_GENERATED:
            embedding = self._generate_llm_embedding(sample_id)
        elif mode == SteeringMode.MIXED:
            embedding = self._generate_mixed_embedding(
                sample_id, suggestions, cluster_id
            )
        else:
            raise ValueError(f"Unsupported steering mode: {mode}")
        
        # Apply augmentations
        embedding = self._apply_augmentations(embedding)
        
        return embedding, mode
    
    def _select_mode(self) -> SteeringMode:
        """Select steering mode based on configured probabilities."""
        if not self.config.mode:
            return SteeringMode.ZERO
        
        modes, probabilities = zip(*self.config.mode)
        return self.rng.choices(modes, weights=probabilities, k=1)[0]
    
    def _generate_zero_embedding(self) -> np.ndarray:
        """Generate zero embedding (no steering)."""
        embedding_dim = self.clustering_data.metadata.get('embedding_dim', 384)
        return np.zeros(embedding_dim, dtype=np.float32)
    
    def _generate_suggestion_embedding(
        self,
        suggestions: Optional[List[str]]
    ) -> np.ndarray:
        """
        Generate embedding from suggestions.
        
        Args:
            suggestions: List of suggestion strings
            
        Returns:
            Suggestion embedding (averaged if multiple)
        """
        if not suggestions:
            logging.warning("No suggestions provided, returning zero embedding")
            return self._generate_zero_embedding()
        
        # Try to use pre-computed embeddings first
        embeddings = []
        for suggestion in suggestions:
            if suggestion in self.suggestion_embeddings:
                embeddings.append(self.suggestion_embeddings[suggestion])
            else:
                # Fallback: encode on-the-fly
                try:
                    emb = self.embedding_model.encode([suggestion])[0]
                    embeddings.append(emb)
                except Exception as e:
                    logging.warning(f"Failed to encode suggestion '{suggestion}': {e}")
        
        if not embeddings:
            return self._generate_zero_embedding()
        
        # Average if multiple suggestions
        return np.mean(embeddings, axis=0).astype(np.float32)
    
    def _generate_descriptor_embedding(
        self,
        cluster_id: Optional[int]
    ) -> np.ndarray:
        """
        Generate embedding from cluster descriptors.
        
        Args:
            cluster_id: Cluster ID
            
        Returns:
            Descriptor embedding (averaged)
        """
        if cluster_id is None:
            logging.warning("No cluster_id provided, returning zero embedding")
            return self._generate_zero_embedding()
        
        descriptors = self.clustering_data.get_descriptors(cluster_id)
        if not descriptors:
            logging.warning(f"No descriptors for cluster {cluster_id}")
            return self._generate_zero_embedding()
        
        # Encode descriptors
        try:
            embeddings = self.embedding_model.encode(descriptors)
            return np.mean(embeddings, axis=0).astype(np.float32)
        except Exception as e:
            logging.error(f"Failed to encode descriptors for cluster {cluster_id}: {e}")
            return self._generate_zero_embedding()
    
    def _generate_llm_embedding(self, sample_id: int) -> np.ndarray:
        """
        Generate embedding from LLM-generated steering text.
        
        Args:
            sample_id: Sample identifier
            
        Returns:
            LLM steering text embedding
        """
        key = str(sample_id)
        if key not in self.llm_steering_texts:
            logging.warning(f"No LLM steering text for sample {sample_id}")
            return self._generate_zero_embedding()
        
        try:
            text = self.llm_steering_texts[key]
            embedding = self.embedding_model.encode([text])[0]
            return embedding.astype(np.float32)
        except Exception as e:
            logging.error(f"Failed to encode LLM text for sample {sample_id}: {e}")
            return self._generate_zero_embedding()
    
    def _generate_mixed_embedding(
        self,
        sample_id: int,
        suggestions: Optional[List[str]],
        cluster_id: Optional[int]
    ) -> np.ndarray:
        """
        Generate weighted combination of multiple steering modes.
        
        Args:
            sample_id: Sample identifier
            suggestions: List of suggestion strings
            cluster_id: Primary cluster ID
            
        Returns:
            Mixed steering embedding
        """
        weights = self.config.mixed_weights
        if not weights:
            logging.warning("No mixed weights configured, using equal weights")
            weights = {
                'suggestion': 0.5,
                'cluster_descriptor': 0.5
            }
        
        embeddings = []
        weight_values = []
        
        for mode_name, weight in weights.items():
            if weight == 0:
                continue
            
            if mode_name == 'suggestion':
                emb = self._generate_suggestion_embedding(suggestions)
            elif mode_name == 'cluster_descriptor':
                emb = self._generate_descriptor_embedding(cluster_id)
            elif mode_name == 'llm_generated':
                emb = self._generate_llm_embedding(sample_id)
            else:
                logging.warning(f"Unknown mode in mixed_weights: {mode_name}")
                continue
            
            embeddings.append(emb)
            weight_values.append(weight)
        
        if not embeddings:
            return self._generate_zero_embedding()
        
        # Weighted average
        embeddings = np.array(embeddings)
        weights_normalized = np.array(weight_values) / np.sum(weight_values)
        return np.average(embeddings, axis=0, weights=weights_normalized).astype(np.float32)
    
    def _apply_augmentations(self, embedding: np.ndarray) -> np.ndarray:
        """
        Apply augmentations from embedding.py.
        
        Args:
            embedding: Original embedding
            
        Returns:
            Augmented embedding
        """
        # Apply zero augmentation (higher priority)
        if self.augment_zero_prob > 0:
            embedding = random_zero_embedding(embedding, self.augment_zero_prob)
            # If zeroed, skip noise augmentation
            if np.allclose(embedding, 0):
                return embedding
        
        # Apply noise augmentation
        if self.augment_noise_prob > 0:
            embedding = random_noise_embedding(
                embedding,
                noise_level=self.augment_noise_level,
                probability=self.augment_noise_prob
            )
        
        return embedding
