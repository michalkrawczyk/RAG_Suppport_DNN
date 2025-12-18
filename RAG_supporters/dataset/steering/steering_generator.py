"""Steering embedding generation component."""

import logging
import random
from typing import Dict, List, Optional, Union

import numpy as np
from langchain_core.embeddings.embeddings import Embeddings

from ...clustering.clustering_data import ClusteringData
from .steering_config import SteeringConfig, SteeringMode


class SteeringGenerator:
    """
    Generates steering embeddings based on configured mode.
    
    Handles:
    - Mode selection for multi-mode configs
    - Steering embedding generation per mode
    - Target generation (hard/soft labels)
    - Missing suggestions tracking
    """
    
    def __init__(
        self,
        config: SteeringConfig,
        clustering_data: Optional[ClusteringData],
        embedding_model: Optional[Embeddings] = None,
        embedding_dim: Optional[int] = None,
        text_embeddings_cache: Optional[List[Dict[str, np.ndarray]]] = None,
        suggestion_embeddings_cache: Optional[Dict[str, np.ndarray]] = None,
        steering_embeddings_cache: Optional[Dict[int, np.ndarray]] = None,
        cluster_descriptor_embeddings_cache: Optional[Dict[int, np.ndarray]] = None,
        llm_steering_embeddings_cache: Optional[Dict[int, np.ndarray]] = None,
        llm_steering_texts: Optional[Dict[int, str]] = None,
    ):
        """
        Initialize steering generator.
        
        Args:
            config: Steering configuration
            clustering_data: Cluster data (labels, descriptors, etc.)
            embedding_model: Model for computing embeddings on-the-fly
            embedding_dim: Embedding dimension
            text_embeddings_cache: Cached text embeddings
            suggestion_embeddings_cache: Cached suggestion embeddings
            steering_embeddings_cache: Pre-computed steering embeddings
            cluster_descriptor_embeddings_cache: Cached cluster descriptor embeddings
            llm_steering_embeddings_cache: Cached LLM steering embeddings
            llm_steering_texts: LLM-generated steering texts
        """
        self.config = config
        self.clustering_data = clustering_data
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        
        # Caches
        self._text_embeddings_cache = text_embeddings_cache
        self._suggestion_embeddings_cache = suggestion_embeddings_cache
        self._steering_embeddings_cache = steering_embeddings_cache
        self._cluster_descriptor_embeddings_cache = cluster_descriptor_embeddings_cache
        self._llm_steering_embeddings_cache = llm_steering_embeddings_cache
        self.llm_steering_texts = llm_steering_texts
        
        # Statistics
        self._missing_suggestions_count = 0
    
    def select_mode(self, idx: int) -> SteeringMode:
        """
        Select steering mode for a sample based on probabilities.
        
        Args:
            idx: Sample index
            
        Returns:
            Selected steering mode
        """
        if not self.config.mode or len(self.config.mode) == 0:
            raise ValueError("No steering mode configured")
        
        if len(self.config.mode) == 1:
            return self.config.mode[0][0]
        
        # Sample based on probabilities using local Random instance
        modes, probs = zip(*self.config.mode)
        # Use sample index as seed for reproducibility
        rng = random.Random(idx + self.config.random_seed)
        selected_mode = rng.choices(modes, weights=probs, k=1)[0]
        return selected_mode
    
    def generate(
        self, idx: int, suggestions: List[str]
    ) -> Optional[np.ndarray]:
        """
        Generate steering embedding for a sample.
        
        Args:
            idx: Sample index
            suggestions: List of suggestion texts
            
        Returns:
            Steering embedding as numpy array or None
        """
        # Select steering mode for this sample
        steering_mode = self.select_mode(idx)
        
        # Determine embedding dimension
        embedding_dim = self.embedding_dim
        if embedding_dim is None and self._text_embeddings_cache:
            embedding_dim = self._text_embeddings_cache[0]["question"].shape[0]
        
        if steering_mode == SteeringMode.ZERO:
            return self._generate_zero(embedding_dim)
        
        elif steering_mode == SteeringMode.SUGGESTION:
            return self._generate_suggestion(idx, suggestions, embedding_dim)
        
        elif steering_mode == SteeringMode.CLUSTER_DESCRIPTOR:
            return self._generate_cluster_descriptor(idx, embedding_dim)
        
        elif steering_mode == SteeringMode.LLM_GENERATED:
            return self._generate_llm(idx, embedding_dim)
        
        elif steering_mode == SteeringMode.MIXED:
            return self._generate_mixed(idx, suggestions, embedding_dim)
        
        return None
    
    def _generate_zero(self, embedding_dim: Optional[int]) -> Optional[np.ndarray]:
        """Generate zero baseline steering."""
        if embedding_dim is None:
            raise ValueError("Cannot determine embedding dimension for zero vector")
        return np.zeros(embedding_dim, dtype=np.float32)
    
    def _generate_suggestion(
        self, idx: int, suggestions: List[str], embedding_dim: Optional[int]
    ) -> Optional[np.ndarray]:
        """Generate suggestion-based steering."""
        if not suggestions:
            if embedding_dim:
                self._missing_suggestions_count += 1
                return np.zeros(embedding_dim, dtype=np.float32)
            return None
        
        # Check for pre-computed steering embedding
        if self._steering_embeddings_cache is not None and idx in self._steering_embeddings_cache:
            return self._steering_embeddings_cache[idx]
        
        # Pick a random suggestion from available ones (reproducible per sample)
        rng = random.Random(idx + self.config.random_seed)
        selected_suggestion = rng.choice(suggestions)
        
        # Check suggestion embedding cache
        if self._suggestion_embeddings_cache and selected_suggestion in self._suggestion_embeddings_cache:
            return self._suggestion_embeddings_cache[selected_suggestion]
        
        # Track if suggestion not found in cache
        if self._suggestion_embeddings_cache and selected_suggestion not in self._suggestion_embeddings_cache:
            self._missing_suggestions_count += 1
            logging.debug(f"Suggestion '{selected_suggestion}' not found in cache for sample {idx}")
        
        # Compute on-the-fly
        if self.embedding_model:
            emb = np.array(
                self.embedding_model.embed_query(selected_suggestion), dtype=np.float32
            )
            return emb
        
        return None
    
    def _generate_cluster_descriptor(
        self, idx: int, embedding_dim: Optional[int]
    ) -> Optional[np.ndarray]:
        """Generate cluster descriptor-based steering."""
        if not self.clustering_data or not self.clustering_data.labels or not self.clustering_data.descriptors:
            if embedding_dim:
                return np.zeros(embedding_dim, dtype=np.float32)
            return None
        
        # Get cluster assignment for this sample
        cluster_id = self.clustering_data.get_primary_cluster(idx)
        if cluster_id is None:
            if embedding_dim:
                return np.zeros(embedding_dim, dtype=np.float32)
            return None
        
        # Get cached descriptor embedding
        if (
            self._cluster_descriptor_embeddings_cache
            and cluster_id in self._cluster_descriptor_embeddings_cache
        ):
            return self._cluster_descriptor_embeddings_cache[cluster_id]
        
        # Compute on-the-fly
        descriptors = self.clustering_data.get_descriptors(cluster_id)
        if descriptors and self.embedding_model:
            # Use first descriptor
            emb = np.array(
                self.embedding_model.embed_query(descriptors[0]),
                dtype=np.float32,
            )
            return emb
        
        if embedding_dim:
            return np.zeros(embedding_dim, dtype=np.float32)
        return None
    
    def _generate_llm(
        self, idx: int, embedding_dim: Optional[int]
    ) -> Optional[np.ndarray]:
        """Generate LLM-based steering."""
        if not self.llm_steering_texts:
            if embedding_dim:
                return np.zeros(embedding_dim, dtype=np.float32)
            return None
        
        # Get cached LLM steering embedding
        if (
            self._llm_steering_embeddings_cache is not None
            and idx in self._llm_steering_embeddings_cache
        ):
            return self._llm_steering_embeddings_cache[idx]
        
        # Compute on-the-fly
        if idx in self.llm_steering_texts and self.embedding_model:
            steering_text = self.llm_steering_texts[idx]
            emb = np.array(
                self.embedding_model.embed_query(steering_text), dtype=np.float32
            )
            return emb
        
        if embedding_dim:
            return np.zeros(embedding_dim, dtype=np.float32)
        return None
    
    def _generate_mixed(
        self, idx: int, suggestions: List[str], embedding_dim: Optional[int]
    ) -> Optional[np.ndarray]:
        """Generate mixed (weighted combination) steering."""
        embeddings = []
        weights = []
        
        # Collect embeddings based on weights
        if self.config.mixed_weights.get("suggestion", 0) > 0 and suggestions:
            if self._suggestion_embeddings_cache and suggestions[0] in self._suggestion_embeddings_cache:
                embeddings.append(self._suggestion_embeddings_cache[suggestions[0]])
                weights.append(self.config.mixed_weights["suggestion"])
        
        if self.config.mixed_weights.get("cluster_descriptor", 0) > 0 and self.clustering_data:
            cluster_id = self.clustering_data.get_primary_cluster(idx)
            if (
                cluster_id is not None
                and self._cluster_descriptor_embeddings_cache
                and cluster_id in self._cluster_descriptor_embeddings_cache
            ):
                embeddings.append(
                    self._cluster_descriptor_embeddings_cache[cluster_id]
                )
                weights.append(self.config.mixed_weights["cluster_descriptor"])
        
        if not embeddings:
            if embedding_dim:
                return np.zeros(embedding_dim, dtype=np.float32)
            return None
        
        # Weighted average
        weights = np.array(weights, dtype=np.float32)
        weights = weights / weights.sum()  # Normalize
        mixed_emb = sum(w * emb for w, emb in zip(weights, embeddings))
        return mixed_emb.astype(np.float32)
    
    def generate_target(self, idx: int) -> Optional[Union[int, np.ndarray]]:
        """
        Generate target (cluster/subspace assignment) for a sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Target as int (hard) or numpy array (soft) or None
        """
        if not self.clustering_data or not self.clustering_data.labels:
            return None
        
        assignment = self.clustering_data.get_label(idx)
        if assignment is None:
            return None
        
        if self.config.multi_label_mode == "hard":
            # Return cluster index
            if isinstance(assignment, int):
                # Validate assignment is within bounds
                n_clusters = self.clustering_data.n_clusters
                if assignment < 0 or assignment >= n_clusters:
                    logging.warning(
                        f"Assignment {assignment} out of bounds [0, {n_clusters}), "
                        f"clipping to valid range"
                    )
                    assignment = max(0, min(assignment, n_clusters - 1))
                return assignment
            elif isinstance(assignment, list):
                # Check for empty list
                if not assignment:
                    logging.warning(f"Empty assignment list for sample {idx}")
                    return 0  # Default to first cluster
                # Return primary cluster (argmax)
                return int(np.argmax(assignment))
        
        elif self.config.multi_label_mode == "soft":
            # Return probability distribution
            if isinstance(assignment, list):
                if not assignment:
                    logging.warning(f"Empty assignment list for sample {idx}")
                    # Return uniform distribution as fallback
                    n_clusters = self.clustering_data.n_clusters
                    return np.ones(n_clusters, dtype=np.float32) / n_clusters
                return np.array(assignment, dtype=np.float32)
            elif isinstance(assignment, int):
                # Convert int to one-hot distribution
                n_clusters = self.clustering_data.n_clusters
                # Validate bounds
                if assignment < 0 or assignment >= n_clusters:
                    logging.warning(
                        f"Assignment {assignment} out of bounds [0, {n_clusters}), "
                        f"clipping to valid range"
                    )
                    assignment = max(0, min(assignment, n_clusters - 1))
                one_hot = np.zeros(n_clusters, dtype=np.float32)
                one_hot[assignment] = 1.0
                return one_hot
        
        return None
    
    def get_missing_suggestions_count(self) -> int:
        """Get count of missing suggestions."""
        return self._missing_suggestions_count
    
    def precompute(self, parsed_suggestions: List[List[str]], n_samples: int):
        """
        Pre-compute steering embeddings for all samples.
        
        Args:
            parsed_suggestions: Parsed suggestions for each sample
            n_samples: Total number of samples
        """
        # Only pre-compute for SUGGESTION mode with single mode config
        if len(self.config.mode) != 1 or self.config.mode[0][0] != SteeringMode.SUGGESTION:
            logging.info("Skipping steering embeddings pre-computation (not single SUGGESTION mode)")
            return
        
        if not self._suggestion_embeddings_cache:
            logging.info("No suggestion embeddings cache, skipping steering pre-computation")
            return
        
        logging.info("Pre-computing steering embeddings...")
        self._steering_embeddings_cache = {}
        
        for idx in range(n_samples):
            suggestions = parsed_suggestions[idx]
            if suggestions:
                # Pick random suggestion (reproducible)
                rng = random.Random(idx + self.config.random_seed)
                selected_suggestion = rng.choice(suggestions)
                
                if selected_suggestion in self._suggestion_embeddings_cache:
                    self._steering_embeddings_cache[idx] = self._suggestion_embeddings_cache[selected_suggestion]
        
        logging.info(f"  ✓ Pre-computed {len(self._steering_embeddings_cache)} steering embeddings")
