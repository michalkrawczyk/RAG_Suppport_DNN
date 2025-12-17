"""Cache management for steering datasets."""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ..utils.dataset_loader import compute_cache_version


class CacheManager:
    """
    Manages caching and loading of dataset components.
    
    Handles saving and loading of:
    - Text embeddings
    - Suggestion embeddings
    - Steering embeddings
    - Cluster data
    - LLM steering texts
    - Metadata
    """
    
    def __init__(self, cache_dir: Union[str, Path]):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory for cache files
        """
        self.cache_dir = Path(cache_dir)
    
    def exists(self) -> bool:
        """
        Check if cache directory exists.
        
        Returns:
            True if cache directory exists
        """
        return self.cache_dir.exists() and (self.cache_dir / "metadata.json").exists()
    
    def compute_version(self, **kwargs) -> str:
        """
        Compute cache version hash.
        
        Args:
            **kwargs: Parameters to include in version hash
            
        Returns:
            Version hash string
        """
        return compute_cache_version(**kwargs)
    
    def validate(self, expected_version: str) -> bool:
        """
        Validate cache version matches expected version.
        
        Args:
            expected_version: Expected version hash
            
        Returns:
            True if version matches
        """
        if not self.exists():
            return False
        
        metadata = self.load_metadata()
        return metadata.get("version") == expected_version
    
    def save(self, data: Dict[str, Any], metadata: Dict[str, Any]):
        """
        Save all data to cache.
        
        Args:
            data: Dictionary of data to save (filename -> data)
            metadata: Metadata dictionary
        """
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata_path = self.cache_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logging.info(f"  ✓ Saved metadata.json")
        
        # Save data files
        for filename, content in data.items():
            if content is None:
                continue
                
            filepath = self.cache_dir / filename
            
            if filename.endswith(".json"):
                with open(filepath, "w") as f:
                    json.dump(content, f, indent=2)
            else:
                with open(filepath, "wb") as f:
                    pickle.dump(content, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logging.info(f"  ✓ Saved {filename}")
    
    def load_metadata(self) -> Dict[str, Any]:
        """
        Load metadata from cache.
        
        Returns:
            Metadata dictionary
            
        Raises:
            ValueError: If metadata file not found
        """
        metadata_path = self.cache_dir / "metadata.json"
        
        if not metadata_path.exists():
            raise ValueError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, "r") as f:
            return json.load(f)
    
    def load_pickle(self, filename: str) -> Any:
        """
        Load pickled data from cache.
        
        Args:
            filename: Name of pickle file
            
        Returns:
            Loaded data
            
        Raises:
            ValueError: If file not found
        """
        filepath = self.cache_dir / filename
        
        if not filepath.exists():
            raise ValueError(f"Cache file not found: {filepath}")
        
        with open(filepath, "rb") as f:
            return pickle.load(f)
    
    def load_pickle_optional(self, filename: str) -> Optional[Any]:
        """
        Load pickled data if it exists, otherwise return None.
        
        Args:
            filename: Name of pickle file
            
        Returns:
            Loaded data or None
        """
        filepath = self.cache_dir / filename
        
        if not filepath.exists():
            return None
        
        with open(filepath, "rb") as f:
            return pickle.load(f)
    
    def load_all(self, return_embeddings: bool = True) -> Dict[str, Any]:
        """
        Load all cached data.
        
        Args:
            return_embeddings: Whether to load embedding data
            
        Returns:
            Dictionary with all loaded data
        """
        if not self.exists():
            raise ValueError(f"Cache directory does not exist: {self.cache_dir}")
        
        logging.info(f"Loading dataset from cache: {self.cache_dir}")
        metadata = self.load_metadata()
        
        logging.info(f"  Version: {metadata['version']}")
        logging.info(f"  Samples: {metadata['length']}")
        logging.info(f"  Unique suggestions: {metadata.get('num_unique_suggestions', 0)}")
        
        # Load required data
        data = {
            "metadata": metadata,
            "parsed_suggestions": self.load_pickle("parsed_suggestions.pkl"),
            "unique_suggestions": self.load_pickle("unique_suggestions.pkl"),
        }
        
        # Load embeddings if requested
        if return_embeddings:
            has_embeddings = metadata.get("has_embeddings", True)
            if not has_embeddings:
                raise ValueError(
                    "Cache was built without embeddings. "
                    "Set return_embeddings=False or rebuild cache with embeddings."
                )
            data["text_embeddings"] = self.load_pickle("text_embeddings.pkl")
            data["suggestion_embeddings"] = self.load_pickle("suggestion_embeddings.pkl")
        else:
            data["text_embeddings"] = None
            data["suggestion_embeddings"] = None
        
        # Load steering-related caches if available
        data["cluster_labels"] = self.load_pickle_optional("cluster_labels.pkl")
        data["cluster_descriptors"] = self.load_pickle_optional("cluster_descriptors.pkl")
        data["llm_steering_texts"] = self.load_pickle_optional("llm_steering_texts.pkl")
        data["cluster_descriptor_embeddings"] = self.load_pickle_optional(
            "cluster_descriptor_embeddings.pkl"
        )
        data["llm_steering_embeddings"] = self.load_pickle_optional(
            "llm_steering_embeddings.pkl"
        )
        data["steering_embeddings"] = self.load_pickle_optional("steering_embeddings.pkl")
        
        logging.info("✓ Dataset loaded successfully")
        return data
