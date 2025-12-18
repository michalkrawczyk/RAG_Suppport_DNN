"""Cache management for steering datasets with integrity checks."""

import hashlib
import json
import logging
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from ..utils.dataset_loader import compute_cache_version


# Cache file constants
METADATA_FILE = "metadata.json"
TEXT_EMBEDDINGS_FILE = "text_embeddings.npy"  # Memory-mapped numpy
SUGGESTION_EMBEDDINGS_FILE = "suggestion_embeddings.npy"  # Memory-mapped numpy
STEERING_EMBEDDINGS_FILE = "steering_embeddings.npy"  # Memory-mapped numpy
CLUSTER_DESCRIPTOR_EMBEDDINGS_FILE = "cluster_descriptor_embeddings.npy"  # Memory-mapped numpy
LLM_STEERING_EMBEDDINGS_FILE = "llm_steering_embeddings.npy"  # Memory-mapped numpy
CLUSTERING_JSON_PATH_FILE = "clustering_json_path.txt"  # Reference to clustering JSON
LLM_STEERING_TEXTS_FILE = "llm_steering_texts.pkl"
SUGGESTIONS_FILE = "suggestions.pkl"
SAMPLE_WEIGHTS_FILE = "sample_weights.pkl"
CHECKSUMS_FILE = "checksums.json"

# Metadata keys
KEY_VERSION = "version"
KEY_HAS_EMBEDDINGS = "has_embeddings"
KEY_N_SAMPLES = "n_samples"
KEY_EMBEDDING_DIM = "embedding_dim"
KEY_CHECKSUMS = "checksums"


class CacheManager:
    """
    Manages caching and loading of dataset components with integrity verification.
    
    Features:
    - Atomic writes with temporary files
    - File checksum verification
    - Comprehensive error handling
    - Support for all steering data types
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
        Check if cache directory exists with valid metadata.
        
        Returns:
            True if cache directory exists with metadata file
        """
        return self.cache_dir.exists() and (self.cache_dir / METADATA_FILE).exists()
    
    def compute_version(self, **kwargs) -> str:
        """
        Compute cache version hash from parameters.
        
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
        return metadata.get(KEY_VERSION) == expected_version
    
    def _compute_file_checksum(self, filepath: Path) -> str:
        """
        Compute SHA256 checksum of a file.
        
        Args:
            filepath: Path to file
            
        Returns:
            Hexadecimal checksum string
        """
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _atomic_write(self, filepath: Path, write_func, mode: str, *args, **kwargs):
        """
        Write file atomically using temporary file.
        
        If write fails midway, original file remains unchanged.
        
        Args:
            filepath: Target file path
            write_func: Function to write data (takes file object)
            mode: File open mode ('w' for text, 'wb' for binary)
            *args, **kwargs: Arguments passed to write_func
        """
        # Create temporary file in same directory for atomic rename
        temp_fd, temp_path = tempfile.mkstemp(
            dir=filepath.parent,
            prefix=f".tmp_{filepath.name}_"
        )
        
        try:
            # Close the file descriptor and reopen with correct mode
            import os
            os.close(temp_fd)
            
            # Write to temporary file with specified mode
            with open(temp_path, mode) as f:
                write_func(f, *args, **kwargs)
            
            # Atomic rename (POSIX guarantees atomicity)
            shutil.move(temp_path, filepath)
            
        except Exception as e:
            # Clean up temp file on failure
            try:
                Path(temp_path).unlink(missing_ok=True)
            except:
                pass
            raise RuntimeError(f"Atomic write failed for {filepath}: {e}") from e
    
    def save(self, data: Dict[str, Any], metadata: Dict[str, Any]):
        """
        Save all data to cache with atomic writes and checksums.
        
        Args:
            data: Dictionary of data to save (filename -> data)
            metadata: Metadata dictionary
            
        Raises:
            RuntimeError: If save operation fails
        """
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        checksums = {}
        
        try:
            # Save data files with atomic writes
            for filename, content in data.items():
                if content is None:
                    continue
                    
                filepath = self.cache_dir / filename
                
                if filename.endswith(".json"):
                    self._atomic_write(
                        filepath,
                        lambda f, c: json.dump(c, f, indent=2),
                        'w',  # Text mode for JSON
                        content
                    )
                elif filename.endswith(".npy"):
                    # Save numpy arrays as memory-mapped files
                    if isinstance(content, np.ndarray):
                        self._save_numpy_memmap(filepath, content)
                    else:
                        raise ValueError(f"Expected numpy array for {filename}, got {type(content)}")
                elif filename.endswith(".txt"):
                    # Save text files (e.g., clustering JSON path reference)
                    self._atomic_write(
                        filepath,
                        lambda f, c: f.write(c),
                        'w',  # Text mode
                        content
                    )
                else:
                    # Pickle for other data types
                    self._atomic_write(
                        filepath,
                        lambda f, c: pickle.dump(c, f, protocol=pickle.HIGHEST_PROTOCOL),
                        'wb',  # Binary mode for pickle
                        content
                    )
                
                # Compute checksum
                checksums[filename] = self._compute_file_checksum(filepath)
                logging.info(f"  ✓ Saved {filename}")
            
            # Add checksums to metadata
            metadata[KEY_CHECKSUMS] = checksums
            
            # Save metadata last (atomic)
            metadata_path = self.cache_dir / METADATA_FILE
            self._atomic_write(
                metadata_path,
                lambda f, m: json.dump(m, f, indent=2),
                'w',  # Text mode for JSON
                metadata
            )
            logging.info(f"  ✓ Saved {METADATA_FILE}")
            
            logging.info(f"✓ Cache saved successfully to {self.cache_dir}")
            
        except Exception as e:
            logging.error(f"Cache save failed: {e}")
            raise
    
    def _save_numpy_memmap(self, filepath: Path, array: np.ndarray):
        """
        Save numpy array as memory-mapped file using atomic write.
        
        Args:
            filepath: Target file path
            array: Numpy array to save
        """
        # Create temporary file
        temp_fd, temp_path = tempfile.mkstemp(
            dir=filepath.parent,
            prefix=f".tmp_{filepath.name}_"
        )
        
        try:
            import os
            os.close(temp_fd)
            
            # Save array to temp file
            np.save(temp_path, array, allow_pickle=False)
            
            # Atomic rename
            shutil.move(temp_path, filepath)
            
        except Exception as e:
            # Clean up temp file on failure
            try:
                Path(temp_path).unlink(missing_ok=True)
            except:
                pass
            raise RuntimeError(f"Failed to save numpy array to {filepath}: {e}") from e
    
    def _verify_checksum(self, filename: str, expected_checksum: str) -> bool:
        """
        Verify file checksum matches expected value.
        
        Args:
            filename: File name
            expected_checksum: Expected checksum
            
        Returns:
            True if checksum matches
        """
        filepath = self.cache_dir / filename
        if not filepath.exists():
            return False
        
        actual_checksum = self._compute_file_checksum(filepath)
        return actual_checksum == expected_checksum
    
    def load_metadata(self) -> Dict[str, Any]:
        """
        Load metadata from cache.
        
        Returns:
            Metadata dictionary
            
        Raises:
            ValueError: If metadata file not found or invalid
        """
        metadata_path = self.cache_dir / METADATA_FILE
        if not metadata_path.exists():
            raise ValueError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return metadata
    
    def load_pickle(self, filename: str, verify_checksum: bool = True) -> Any:
        """
        Load pickled data from cache with optional checksum verification.
        
        Args:
            filename: File name
            verify_checksum: Whether to verify file checksum
            
        Returns:
            Loaded data
            
        Raises:
            ValueError: If file not found or checksum mismatch
        """
        filepath = self.cache_dir / filename
        if not filepath.exists():
            raise ValueError(f"Cache file not found: {filepath}")
        
        # Verify checksum if enabled and available
        if verify_checksum:
            metadata = self.load_metadata()
            checksums = metadata.get(KEY_CHECKSUMS, {})
            if filename in checksums:
                if not self._verify_checksum(filename, checksums[filename]):
                    raise ValueError(
                        f"Checksum mismatch for {filename}. File may be corrupted."
                    )
        
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def load_numpy(self, filename: str, mmap_mode: str = 'r', verify_checksum: bool = True) -> np.ndarray:
        """
        Load numpy array from cache with optional memory mapping.
        
        Args:
            filename: File name (should be .npy)
            mmap_mode: Memory-map mode ('r' for read-only, None to load into memory)
            verify_checksum: Whether to verify file checksum
            
        Returns:
            Numpy array (memory-mapped if mmap_mode specified)
            
        Raises:
            ValueError: If file not found or checksum mismatch
        """
        filepath = self.cache_dir / filename
        if not filepath.exists():
            raise ValueError(f"Cache file not found: {filepath}")
        
        # Verify checksum if enabled and available
        if verify_checksum:
            metadata = self.load_metadata()
            checksums = metadata.get(KEY_CHECKSUMS, {})
            if filename in checksums:
                if not self._verify_checksum(filename, checksums[filename]):
                    raise ValueError(
                        f"Checksum mismatch for {filename}. File may be corrupted."
                    )
        
        # Load with memory mapping for efficient access
        return np.load(filepath, mmap_mode=mmap_mode, allow_pickle=False)
    
    def load_numpy_optional(self, filename: str, mmap_mode: str = 'r', verify_checksum: bool = True) -> Optional[np.ndarray]:
        """
        Load numpy array if file exists, otherwise return None.
        
        Args:
            filename: File name
            mmap_mode: Memory-map mode ('r' for read-only, None to load into memory)
            verify_checksum: Whether to verify file checksum
            
        Returns:
            Numpy array or None if file doesn't exist
        """
        filepath = self.cache_dir / filename
        if not filepath.exists():
            return None
        
        return self.load_numpy(filename, mmap_mode=mmap_mode, verify_checksum=verify_checksum)
    
    def load_pickle_optional(self, filename: str, verify_checksum: bool = True) -> Optional[Any]:
        """
        Load pickled data if file exists, otherwise return None.
        
        Args:
            filename: File name
            verify_checksum: Whether to verify file checksum
            
        Returns:
            Loaded data or None if file doesn't exist
        """
        try:
            return self.load_pickle(filename, verify_checksum=verify_checksum)
        except (ValueError, FileNotFoundError):
            return None
    
    def load(self, return_embeddings: bool = True) -> Dict[str, Any]:
        """
        Load all cached data with integrity verification.
        
        Embeddings are loaded as memory-mapped numpy arrays for efficiency.
        Clustering data is loaded from JSON reference if available.
        
        Args:
            return_embeddings: Whether to load embedding data
            
        Returns:
            Dictionary containing all cached data
            
        Raises:
            ValueError: If cache invalid or corrupted
        """
        if not self.exists():
            raise ValueError(f"Cache not found at {self.cache_dir}")
        
        logging.info(f"Loading dataset from cache: {self.cache_dir}")
        
        # Load metadata
        metadata = self.load_metadata()
        
        # Load required data
        data = {
            "metadata": metadata,
            "suggestions": self.load_pickle(SUGGESTIONS_FILE),
            "parsed_suggestions": self.load_pickle("parsed_suggestions.pkl"),
        }
        
        # Load embeddings if requested (memory-mapped for efficiency)
        if return_embeddings:
            has_embeddings = metadata.get(KEY_HAS_EMBEDDINGS, True)
            if not has_embeddings:
                raise ValueError(
                    "Cache was built without embeddings. "
                    "Set return_embeddings=False or rebuild cache with embeddings."
                )
            # Load as memory-mapped arrays (efficient for large datasets)
            data["text_embeddings"] = self.load_numpy(TEXT_EMBEDDINGS_FILE, mmap_mode='r')
            data["suggestion_embeddings"] = self.load_numpy(SUGGESTION_EMBEDDINGS_FILE, mmap_mode='r')
        else:
            data["text_embeddings"] = None
            data["suggestion_embeddings"] = None
        
        # Load clustering JSON path reference
        clustering_json_path_file = self.cache_dir / CLUSTERING_JSON_PATH_FILE
        if clustering_json_path_file.exists():
            with open(clustering_json_path_file, 'r') as f:
                clustering_json_path = f.read().strip()
                data["clustering_json_path"] = clustering_json_path
                # Load clustering data from JSON
                if Path(clustering_json_path).exists():
                    from ...clustering.clustering_data import ClusteringData
                    data["clustering_data"] = ClusteringData.from_json(clustering_json_path)
                    # Extract labels and descriptors
                    data["cluster_labels"] = data["clustering_data"].labels
                    data["cluster_descriptors"] = data["clustering_data"].descriptors
                else:
                    logging.warning(f"Clustering JSON not found: {clustering_json_path}")
                    data["clustering_data"] = None
                    data["cluster_labels"] = None
                    data["cluster_descriptors"] = None
        else:
            data["clustering_json_path"] = None
            data["clustering_data"] = None
            data["cluster_labels"] = None
            data["cluster_descriptors"] = None
        
        # Load LLM steering texts if available
        data["llm_steering_texts"] = self.load_pickle_optional(LLM_STEERING_TEXTS_FILE)
        
        # Load steering embeddings as memory-mapped arrays
        data["cluster_descriptor_embeddings"] = self.load_numpy_optional(
            CLUSTER_DESCRIPTOR_EMBEDDINGS_FILE, mmap_mode='r'
        )
        data["llm_steering_embeddings"] = self.load_numpy_optional(
            LLM_STEERING_EMBEDDINGS_FILE, mmap_mode='r'
        )
        data["steering_embeddings"] = self.load_numpy_optional(
            STEERING_EMBEDDINGS_FILE, mmap_mode='r'
        )
        data["sample_weights"] = self.load_pickle_optional(SAMPLE_WEIGHTS_FILE)
        
        logging.info("✓ Dataset loaded successfully with integrity verification")
        return data
