"""
Configuration dataclass for JASPER Steering Dataset Builder.

This module provides build configuration with JSON serialization support
for reproducible dataset construction.
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Union


@dataclass
class BuildConfig:
    """Configuration for building JASPER Steering Dataset.
    
    This configuration is saved as config.json in the output directory
    and serves as the single source of truth for dataset metadata.
    
    Parameters
    ----------
    embedding_dim : int
        Dimension of embeddings (e.g., 384 for all-MiniLM-L6-v2)
    n_neg : int
        Number of hard negatives per positive pair
    clustering_source : str
        Path to KeywordClusterer JSON results file
    split_ratios : List[float]
        Train/val/test split ratios, must sum to 1.0
    steering_probabilities : Dict[str, float]
        Probabilities for steering variants:
        - "zero": Zero vector (no steering)
        - "centroid": Cluster centroid direction
        - "keyword": Keyword-weighted direction
        - "residual": Residual from centroid
        Must sum to 1.0
    curriculum : Dict[str, Union[float, str]]
        Curriculum learning configuration:
        - "mode": "fixed" | "linear" | "cosine"
        - "start_distance": Initial centroid distance threshold
        - "end_distance": Final distance threshold
        - "warmup_epochs": Epochs to transition from start to end
    n_pairs : int, optional
        Total number of question-source pairs (computed during build)
    n_questions : int, optional
        Number of unique questions (computed during build)
    n_sources : int, optional
        Number of unique sources (computed during build)
    n_keywords : int, optional
        Number of unique keywords (computed during build)
    n_clusters : int, optional
        Number of clusters from KeywordClusterer (computed during build)
    storage_format : str
        Storage format: "pt" (PyTorch) or "hdf5"
    include_inspection_file : bool
        Whether to generate optional inspection.json for debugging
    random_seed : int
        Random seed for deterministic splitting and sampling
    
    Examples
    --------
    >>> config = BuildConfig(
    ...     embedding_dim=384,
    ...     n_neg=12,
    ...     clustering_source="clusters.json",
    ...     split_ratios=[0.8, 0.1, 0.1],
    ...     steering_probabilities={"zero": 0.25, "centroid": 0.25, 
    ...                             "keyword": 0.25, "residual": 0.25},
    ...     curriculum={"mode": "linear", "start_distance": 0.3, 
    ...                 "end_distance": 0.7, "warmup_epochs": 10}
    ... )
    >>> config.save("output/config.json")
    >>> loaded = BuildConfig.load("output/config.json")
    """
    
    embedding_dim: int
    n_neg: int
    clustering_source: str
    split_ratios: List[float] = field(default_factory=lambda: [0.8, 0.1, 0.1])
    steering_probabilities: Dict[str, float] = field(
        default_factory=lambda: {
            "zero": 0.25,
            "centroid": 0.25,
            "keyword": 0.25,
            "residual": 0.25
        }
    )
    curriculum: Dict[str, Union[float, str]] = field(
        default_factory=lambda: {
            "mode": "linear",
            "start_distance": 0.3,
            "end_distance": 0.7,
            "warmup_epochs": 10
        }
    )
    n_pairs: Optional[int] = None
    n_questions: Optional[int] = None
    n_sources: Optional[int] = None
    n_keywords: Optional[int] = None
    n_clusters: Optional[int] = None
    storage_format: str = "pt"
    include_inspection_file: bool = False
    random_seed: int = 42
    
    def __post_init__(self):
        """Validate configuration values."""
        # Validate split ratios
        if not abs(sum(self.split_ratios) - 1.0) < 1e-6:
            raise ValueError(
                f"split_ratios must sum to 1.0, got {sum(self.split_ratios)}"
            )
        
        if len(self.split_ratios) != 3:
            raise ValueError(
                f"split_ratios must have exactly 3 values (train/val/test), "
                f"got {len(self.split_ratios)}"
            )
        
        if any(r < 0 or r > 1 for r in self.split_ratios):
            raise ValueError(
                f"split_ratios must be in range [0, 1], got {self.split_ratios}"
            )
        
        # Validate steering probabilities
        if not abs(sum(self.steering_probabilities.values()) - 1.0) < 1e-6:
            raise ValueError(
                f"steering_probabilities must sum to 1.0, "
                f"got {sum(self.steering_probabilities.values())}"
            )
        
        required_steering_keys = {"zero", "centroid", "keyword", "residual"}
        if set(self.steering_probabilities.keys()) != required_steering_keys:
            raise ValueError(
                f"steering_probabilities must have keys {required_steering_keys}, "
                f"got {set(self.steering_probabilities.keys())}"
            )
        
        # Validate storage format
        if self.storage_format not in ("pt", "hdf5"):
            raise ValueError(
                f"storage_format must be 'pt' or 'hdf5', got {self.storage_format}"
            )
        
        # Validate curriculum
        if "mode" not in self.curriculum:
            raise ValueError("curriculum must have 'mode' key")
        
        if self.curriculum["mode"] not in ("fixed", "linear", "cosine"):
            raise ValueError(
                f"curriculum mode must be 'fixed', 'linear', or 'cosine', "
                f"got {self.curriculum['mode']}"
            )
        
        # Validate embedding dimension
        if self.embedding_dim <= 0:
            raise ValueError(
                f"embedding_dim must be positive, got {self.embedding_dim}"
            )
        
        # Validate n_neg
        if self.n_neg <= 0:
            raise ValueError(f"n_neg must be positive, got {self.n_neg}")
    
    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to JSON file.
        
        Parameters
        ----------
        path : str or Path
            Output path for config.json
        
        Examples
        --------
        >>> config.save("output_dir/config.json")
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "BuildConfig":
        """Load configuration from JSON file.
        
        Parameters
        ----------
        path : str or Path
            Path to config.json
        
        Returns
        -------
        BuildConfig
            Loaded configuration object
        
        Examples
        --------
        >>> config = BuildConfig.load("output_dir/config.json")
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return cls(**data)
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary.
        
        Returns
        -------
        Dict
            Configuration as dictionary
        """
        return asdict(self)
    
    def update_post_build(
        self,
        n_pairs: int,
        n_questions: int,
        n_sources: int,
        n_keywords: int,
        n_clusters: int
    ) -> None:
        """Update configuration with computed statistics after build.
        
        Called by build orchestrator after dataset construction is complete.
        
        Parameters
        ----------
        n_pairs : int
            Total number of question-source pairs
        n_questions : int
            Number of unique questions
        n_sources : int
            Number of unique sources
        n_keywords : int
            Number of unique keywords
        n_clusters : int
            Number of clusters
        """
        self.n_pairs = n_pairs
        self.n_questions = n_questions
        self.n_sources = n_sources
        self.n_keywords = n_keywords
        self.n_clusters = n_clusters
