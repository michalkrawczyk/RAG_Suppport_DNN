"""Steering dataset builder component for computing embeddings and building cache."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from langchain_core.embeddings.embeddings import Embeddings
from tqdm import tqdm

from ...clustering.clustering_data import ClusteringData
from ..utils.dataset_loader import filter_suggestions, parse_suggestions_safe
from .cache_manager import CacheManager
from .steering_config import SteeringConfig, SteeringMode
from .steering_generator import SteeringGenerator


class SteeringDatasetBuilder:
    """
    Builds steering datasets by computing embeddings and preparing caches.
    
    This builder is specifically for creating datasets with steering embeddings
    for subspace/cluster-based steering in neural networks.
    
    Handles:
    - Parsing suggestions
    - Computing text embeddings
    - Computing suggestion embeddings  
    - Computing steering embeddings
    - Validation
    - Cache persistence with clustering JSON reference
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        embedding_model: Embeddings,
        clustering_data: Optional[ClusteringData] = None,
        steering_config: Optional[SteeringConfig] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        clustering_json_path: Optional[Union[str, Path]] = None,
        source_col: str = "source",
        question_col: str = "question",
        suggestions_col: str = "suggestions",
        min_confidence: float = 0.0,
        suggestion_types: Optional[List[str]] = None,
        chunksize: int = 32,
        llm_steering_texts: Optional[Dict[int, str]] = None,
        return_embeddings: bool = True,
        version: Optional[str] = None,
    ):
        """
        Initialize steering dataset builder.
        
        Args:
            df: DataFrame with source data
            embedding_model: Model for computing embeddings
            clustering_data: Cluster data (labels, descriptors)
            steering_config: Steering configuration
            cache_dir: Directory for caching results
            clustering_json_path: Path to clustering results JSON (from KeywordClusterer.save_results)
            source_col: Column name for source texts
            question_col: Column name for question texts
            suggestions_col: Column name for suggestions
            min_confidence: Minimum confidence for filtering suggestions
            suggestion_types: Allowed suggestion types
            chunksize: Batch size for embedding computation
            llm_steering_texts: LLM-generated steering texts
            return_embeddings: Whether to compute embeddings
            version: Cache version hash
        """
        self.df = df
        self.embedding_model = embedding_model
        self.clustering_data = clustering_data
        self.steering_config = steering_config or SteeringConfig()
        self.cache_manager = CacheManager(cache_dir) if cache_dir else None
        self.clustering_json_path = str(clustering_json_path) if clustering_json_path else None
        
        self.source_col = source_col
        self.question_col = question_col
        self.suggestions_col = suggestions_col
        self.min_confidence = min_confidence
        self.suggestion_types = suggestion_types
        self.chunksize = chunksize
        self.llm_steering_texts = llm_steering_texts
        self.return_embeddings = return_embeddings
        self.version = version
        
        # Built data
        self.parsed_suggestions = None
        self.unique_suggestions = None
        self.text_embeddings = None
        self.suggestion_embeddings = None
        self.cluster_descriptor_embeddings = None
        self.llm_steering_embeddings = None
        self.steering_embeddings = None
    
    def build(self, save_to_cache: bool = True) -> Dict[str, any]:
        """
        Build the complete dataset by precomputing all embeddings.
        
        Args:
            save_to_cache: If True and cache_dir is set, save to disk
            
        Returns:
            Dictionary with all built data
        """
        has_steering = self.steering_config and len(self.steering_config.mode) > 0
        total_steps = 6 if (self.return_embeddings and has_steering) else 4
        
        logging.info("=" * 60)
        logging.info("Building Domain Assign Dataset")
        if self.cache_manager:
            logging.info(f"Cache directory: {self.cache_manager.cache_dir}")
        logging.info(f"Version: {self.version}")
        if has_steering:
            logging.info(f"Steering modes: {self.steering_config.mode}")
        logging.info("=" * 60)
        
        # Step 1: Parse and filter suggestions
        logging.info(f"\n[1/{total_steps}] Parsing suggestions...")
        self.parsed_suggestions = self.parse_suggestions()
        
        # Step 2: Extract unique suggestions
        logging.info(f"\n[2/{total_steps}] Extracting unique suggestions...")
        self.unique_suggestions = self.extract_unique_suggestions()
        
        # Step 3 & 4: Compute embeddings
        if self.return_embeddings:
            logging.info(f"\n[3/{total_steps}] Computing text embeddings...")
            self.text_embeddings = self.compute_text_embeddings()
            
            logging.info(f"\n[4/{total_steps}] Computing suggestion embeddings...")
            self.suggestion_embeddings = self.compute_suggestion_embeddings()
            
            # Step 5 & 6: Compute steering embeddings if needed
            if has_steering:
                logging.info(f"\n[5/{total_steps}] Computing steering embeddings...")
                self.compute_steering_embeddings()
                
                logging.info(f"\n[6/{total_steps}] Validating cluster assignments...")
                self.validate_cluster_assignments()
        else:
            logging.info(f"\n[3/{total_steps}] Skipping embeddings (return_embeddings=False)")
            logging.info(f"[4/{total_steps}] Skipping embeddings")
        
        # Save to disk if requested
        if save_to_cache and self.cache_manager:
            logging.info("\nSaving to cache...")
            self.save_cache()
        
        logging.info("\n" + "=" * 60)
        logging.info("✓ Build complete!")
        logging.info(f"  - Total samples: {len(self.df)}")
        logging.info(f"  - Unique suggestions: {len(self.unique_suggestions)}")
        if has_steering:
            logging.info(f"  - Steering modes: {self.steering_config.mode}")
        if self.cache_manager and save_to_cache:
            logging.info(f"  - Cache location: {self.cache_manager.cache_dir}")
        logging.info("=" * 60)
        
        return self.get_built_data()
    
    def parse_suggestions(self) -> List[List[str]]:
        """Parse and filter all suggestions."""
        all_parsed = []
        
        for idx in tqdm(range(len(self.df)), desc="Parsing"):
            raw_suggestions = self.df.iloc[idx][self.suggestions_col]
            suggestions = parse_suggestions_safe(raw_suggestions)
            filtered = filter_suggestions(
                suggestions, self.min_confidence, self.suggestion_types
            )
            all_parsed.append(filtered)
        
        return all_parsed
    
    def extract_unique_suggestions(self) -> List[str]:
        """Extract all unique suggestion terms."""
        if self.parsed_suggestions is None:
            self.parsed_suggestions = self.parse_suggestions()
        
        unique_terms = set()
        for suggestions in self.parsed_suggestions:
            unique_terms.update(suggestions)
        
        unique_list = sorted(list(unique_terms))
        logging.info(f"Found {len(unique_list)} unique suggestions")
        return unique_list
    
    def compute_text_embeddings(self) -> List[Dict[str, np.ndarray]]:
        """Compute embeddings for all source texts and questions."""
        all_embeddings = []
        
        total_rows = len(self.df)
        for start_idx in tqdm(
            range(0, total_rows, self.chunksize), desc="Embedding texts"
        ):
            end_idx = min(start_idx + self.chunksize, total_rows)
            
            # Get batch
            batch_df = self.df.iloc[start_idx:end_idx]
            sources = batch_df[self.source_col].astype(str).tolist()
            questions = batch_df[self.question_col].astype(str).tolist()
            
            # Embed batch
            source_embeds = self.embedding_model.embed_documents(sources)
            question_embeds = self.embedding_model.embed_documents(questions)
            
            # Store
            for s_emb, q_emb in zip(source_embeds, question_embeds):
                all_embeddings.append(
                    {
                        "source": np.array(s_emb, dtype=np.float32),
                        "question": np.array(q_emb, dtype=np.float32),
                    }
                )
        
        return all_embeddings
    
    def compute_suggestion_embeddings(self) -> Dict[str, np.ndarray]:
        """Compute embeddings for all unique suggestions."""
        if self.unique_suggestions is None:
            self.unique_suggestions = self.extract_unique_suggestions()
        
        embeddings_dict = {}
        
        total_suggestions = len(self.unique_suggestions)
        for start_idx in tqdm(
            range(0, total_suggestions, self.chunksize), desc="Embedding suggestions"
        ):
            end_idx = min(start_idx + self.chunksize, total_suggestions)
            
            # Get batch
            batch = self.unique_suggestions[start_idx:end_idx]
            
            # Embed batch
            embeds = self.embedding_model.embed_documents(batch)
            
            # Store
            for term, emb in zip(batch, embeds):
                embeddings_dict[term] = np.array(emb, dtype=np.float32)
        
        return embeddings_dict
    
    def compute_steering_embeddings(self):
        """Compute steering embeddings based on steering mode."""
        if not self.steering_config or len(self.steering_config.mode) == 0:
            return
        
        if not self.return_embeddings:
            return
        
        # Get single mode if only one configured
        single_mode = self.steering_config.get_single_mode()
        
        if single_mode == SteeringMode.ZERO:
            # No computation needed for zero mode
            return
        
        elif single_mode == SteeringMode.CLUSTER_DESCRIPTOR:
            self._compute_cluster_descriptor_embeddings()
        
        elif single_mode == SteeringMode.LLM_GENERATED:
            self._compute_llm_steering_embeddings()
        
        elif single_mode == SteeringMode.SUGGESTION:
            self._compute_suggestion_steering_embeddings()
        
        elif single_mode == SteeringMode.MIXED:
            self._compute_mixed_steering_embeddings()
        
        elif self.steering_config.is_multi_mode():
            # For multi-mode, compute all necessary embeddings
            for mode, _ in self.steering_config.mode:
                if mode == SteeringMode.CLUSTER_DESCRIPTOR:
                    self._compute_cluster_descriptor_embeddings()
                elif mode == SteeringMode.LLM_GENERATED:
                    self._compute_llm_steering_embeddings()
                elif mode == SteeringMode.SUGGESTION:
                    self._compute_suggestion_steering_embeddings()
    
    def _compute_cluster_descriptor_embeddings(self):
        """Compute embeddings for cluster descriptors."""
        if not self.clustering_data or not self.clustering_data.descriptors:
            logging.warning("No cluster descriptors provided for CLUSTER_DESCRIPTOR mode")
            return
        
        self.cluster_descriptor_embeddings = {}
        
        all_descriptors = []
        descriptor_to_cluster = {}
        
        for cluster_id, descriptors in self.clustering_data.descriptors.items():
            if descriptors:
                # Use first descriptor as representative
                descriptor = descriptors[0]
                all_descriptors.append(descriptor)
                descriptor_to_cluster[descriptor] = cluster_id
        
        if all_descriptors:
            # Batch embed all descriptors
            for start_idx in tqdm(
                range(0, len(all_descriptors), self.chunksize),
                desc="Embedding cluster descriptors",
            ):
                end_idx = min(start_idx + self.chunksize, len(all_descriptors))
                batch = all_descriptors[start_idx:end_idx]
                embeds = self.embedding_model.embed_documents(batch)
                
                for descriptor, emb in zip(batch, embeds):
                    cluster_id = descriptor_to_cluster[descriptor]
                    self.cluster_descriptor_embeddings[cluster_id] = np.array(
                        emb, dtype=np.float32
                    )
    
    def _compute_llm_steering_embeddings(self):
        """Compute embeddings for LLM-generated texts."""
        if not self.llm_steering_texts:
            logging.warning("No LLM steering texts provided for LLM_GENERATED mode")
            return
        
        self.llm_steering_embeddings = {}
        
        all_texts = list(self.llm_steering_texts.items())
        
        for start_idx in tqdm(
            range(0, len(all_texts), self.chunksize),
            desc="Embedding LLM steering texts",
        ):
            end_idx = min(start_idx + self.chunksize, len(all_texts))
            batch_items = all_texts[start_idx:end_idx]
            batch_texts = [text for _, text in batch_items]
            
            embeds = self.embedding_model.embed_documents(batch_texts)
            
            for (idx, _), emb in zip(batch_items, embeds):
                self.llm_steering_embeddings[idx] = np.array(
                    emb, dtype=np.float32
                )
    
    def _compute_suggestion_steering_embeddings(self):
        """Pre-compute steering embeddings for each sample."""
        self.steering_embeddings = {}
        
        # Create generator to handle mode selection
        generator = SteeringGenerator(
            config=self.steering_config,
            clustering_data=self.clustering_data,
            embedding_model=None,
            text_embeddings_cache=self.text_embeddings,
            suggestion_embeddings_cache=self.suggestion_embeddings,
        )
        
        # Pre-compute steering embeddings
        generator.precompute(self.parsed_suggestions, len(self.df))
        
        if generator._steering_embeddings_cache:
            self.steering_embeddings = generator._steering_embeddings_cache
    
    def _compute_mixed_steering_embeddings(self):
        """Compute all component embeddings for mixed mode."""
        # Compute cluster descriptors if needed
        if (
            self.clustering_data and
            self.clustering_data.descriptors and
            self.steering_config.mixed_weights.get("cluster_descriptor", 0) > 0
        ):
            logging.info("  Computing cluster descriptor embeddings for mixed mode...")
            self._compute_cluster_descriptor_embeddings()
    
    def validate_cluster_assignments(self):
        """Validate cluster assignments are consistent with data."""
        if not self.clustering_data or not self.clustering_data.labels:
            return
        
        n_samples = len(self.df)
        n_assigned = len(self.clustering_data.labels)
        
        if n_assigned != n_samples:
            logging.warning(
                f"Cluster labels count ({n_assigned}) doesn't match dataset size ({n_samples})"
            )
        
        # Check for missing indices
        missing = set(range(n_samples)) - set(self.clustering_data.labels.keys())
        if missing:
            logging.warning(f"Missing cluster assignments for {len(missing)} samples")
        
        logging.info(f"Validated {n_assigned} cluster assignments")
    
    def save_cache(self):
        """
        Save all processed data to cache.
        
        Embeddings are saved as memory-mapped numpy files (.npy) for efficient loading.
        Clustering info is stored as a JSON path reference instead of duplicating data.
        """
        if not self.cache_manager:
            raise ValueError("No cache manager configured")
        
        from .cache_manager import (
            TEXT_EMBEDDINGS_FILE,
            SUGGESTION_EMBEDDINGS_FILE,
            STEERING_EMBEDDINGS_FILE,
            CLUSTER_DESCRIPTOR_EMBEDDINGS_FILE,
            LLM_STEERING_EMBEDDINGS_FILE,
            CLUSTERING_JSON_PATH_FILE,
            LLM_STEERING_TEXTS_FILE,
            SUGGESTIONS_FILE,
        )
        
        # Prepare metadata
        metadata = {
            "version": self.version,
            "length": len(self.df),
            "source_col": self.source_col,
            "question_col": self.question_col,
            "suggestions_col": self.suggestions_col,
            "min_confidence": self.min_confidence,
            "suggestion_types": self.suggestion_types,
            "embedding_model_name": getattr(self.embedding_model, "model_name", "unknown"),
            "num_unique_suggestions": len(self.unique_suggestions) if self.unique_suggestions else 0,
            "has_embeddings": self.text_embeddings is not None,
            # Steering-related metadata
            "steering_modes": [(mode.value, prob) for mode, prob in self.steering_config.mode] if self.steering_config.mode else None,
            "multi_label_mode": self.steering_config.multi_label_mode,
            "has_cluster_labels": self.clustering_data and self.clustering_data.labels is not None,
            "has_cluster_descriptors": self.clustering_data and self.clustering_data.descriptors is not None,
            "has_llm_steering": self.llm_steering_texts is not None,
            "has_clustering_json": self.clustering_json_path is not None,
        }
        
        # Prepare data files
        data = {
            "parsed_suggestions.pkl": self.parsed_suggestions,
            SUGGESTIONS_FILE: self.unique_suggestions,
        }
        
        # Save embeddings as memory-mapped numpy arrays (.npy)
        if self.text_embeddings is not None:
            data[TEXT_EMBEDDINGS_FILE] = self.text_embeddings
        
        if self.suggestion_embeddings is not None:
            data[SUGGESTION_EMBEDDINGS_FILE] = self.suggestion_embeddings
        
        if self.cluster_descriptor_embeddings is not None:
            data[CLUSTER_DESCRIPTOR_EMBEDDINGS_FILE] = self.cluster_descriptor_embeddings
        
        if self.llm_steering_embeddings is not None:
            data[LLM_STEERING_EMBEDDINGS_FILE] = self.llm_steering_embeddings
        
        if self.steering_embeddings is not None:
            data[STEERING_EMBEDDINGS_FILE] = self.steering_embeddings
        
        # Store reference to clustering JSON instead of duplicating data
        if self.clustering_json_path:
            data[CLUSTERING_JSON_PATH_FILE] = self.clustering_json_path
        
        # Store LLM steering texts if provided
        if self.llm_steering_texts is not None:
            data[LLM_STEERING_TEXTS_FILE] = self.llm_steering_texts
        
        # Save
        self.cache_manager.save(data, metadata)
    
    def get_built_data(self) -> Dict[str, any]:
        """
        Get all built data as a dictionary.
        
        Returns:
            Dictionary with all built components
        """
        return {
            "df": self.df,
            "parsed_suggestions": self.parsed_suggestions,
            "unique_suggestions": self.unique_suggestions,
            "text_embeddings": self.text_embeddings,
            "suggestion_embeddings": self.suggestion_embeddings,
            "cluster_descriptor_embeddings": self.cluster_descriptor_embeddings,
            "llm_steering_embeddings": self.llm_steering_embeddings,
            "steering_embeddings": self.steering_embeddings,
            "clustering_data": self.clustering_data,
            "steering_config": self.steering_config,
            "llm_steering_texts": self.llm_steering_texts,
            "source_col": self.source_col,
            "question_col": self.question_col,
            "suggestions_col": self.suggestions_col,
        }
