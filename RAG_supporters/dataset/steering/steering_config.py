"""Steering configuration with embedded steering modes."""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class SteeringMode(Enum):
    """Steering embedding modes for cluster/subspace steering."""

    SUGGESTION = "suggestion"  # Use suggestion embeddings
    LLM_GENERATED = "llm_generated"  # LLM-generated steering text
    CLUSTER_DESCRIPTOR = "cluster_descriptor"  # Cluster/topic descriptor embeddings
    ZERO = "zero"  # Zero baseline (no steering)
    MIXED = "mixed"  # Weighted combination of multiple modes


@dataclass
class SteeringConfig:
    """
    Configuration for steering embeddings.
    
    Attributes:
        mode: List of (mode, probability) tuples for steering mode selection
        multi_label_mode: "hard" for one-hot, "soft" for probabilities
        mixed_weights: Weights for MIXED mode
        random_seed: Seed for reproducible mode selection
    """
    
    mode: List[Tuple[SteeringMode, float]] = field(default_factory=list)
    multi_label_mode: str = "hard"
    mixed_weights: Dict[str, float] = field(default_factory=dict)
    random_seed: int = 42
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate multi_label_mode
        if self.multi_label_mode not in ["hard", "soft"]:
            raise ValueError(
                f"multi_label_mode must be 'hard' or 'soft', got '{self.multi_label_mode}'"
            )
        
        # Validate and normalize mode probabilities
        if self.mode:
            if not isinstance(self.mode, list):
                raise ValueError("mode must be a list of (SteeringMode, probability) tuples")
            
            if len(self.mode) == 0:
                raise ValueError("mode list cannot be empty")
            
            # Validate all entries are (SteeringMode, float) tuples
            for i, entry in enumerate(self.mode):
                if not isinstance(entry, tuple) or len(entry) != 2:
                    raise ValueError(
                        f"mode[{i}] must be a (SteeringMode, float) tuple, got {type(entry)}"
                    )
                mode, prob = entry
                if not isinstance(mode, SteeringMode):
                    raise ValueError(
                        f"mode[{i}] first element must be SteeringMode instance, "
                        f"got {type(mode)}. Did you forget to import SteeringMode?"
                    )
                if not isinstance(prob, (int, float)):
                    raise ValueError(
                        f"mode[{i}] probability must be numeric, got {type(prob)}"
                    )
                if prob < 0:
                    raise ValueError(
                        f"mode[{i}] probability must be non-negative, got {prob}"
                    )
            
            # Check for zero total probability
            total_prob = sum(prob for _, prob in self.mode)
            if total_prob == 0:
                raise ValueError(
                    "All mode probabilities are zero. At least one mode must have positive probability."
                )
            
            # Normalize probabilities if needed
            if abs(total_prob - 1.0) > 1e-6:
                logging.warning(
                    f"Steering mode probabilities sum to {total_prob:.6f}, normalizing to 1.0"
                )
                self.mode = [(mode, prob / total_prob) for mode, prob in self.mode]
        
        logging.info(f"SteeringConfig validated: mode={self.mode}, multi_label={self.multi_label_mode}")
    
    @classmethod
    def from_single_mode(
        cls,
        mode: SteeringMode,
        multi_label_mode: str = "hard",
        mixed_weights: Optional[Dict[str, float]] = None,
        random_seed: int = 42
    ) -> "SteeringConfig":
        """
        Create config from a single steering mode.
        
        Args:
            mode: Steering mode
            multi_label_mode: Target label mode
            mixed_weights: Weights for MIXED mode
            random_seed: Random seed
            
        Returns:
            SteeringConfig instance
            
        Raises:
            ValueError: If mode is not a SteeringMode instance
        """
        if not isinstance(mode, SteeringMode):
            raise ValueError(
                f"mode must be SteeringMode instance, got {type(mode)}"
            )
        
        return cls(
            mode=[(mode, 1.0)],
            multi_label_mode=multi_label_mode,
            mixed_weights=mixed_weights or {},
            random_seed=random_seed
        )
    
    @classmethod
    def from_mode_list(
        cls,
        modes: List[Tuple[SteeringMode, float]],
        multi_label_mode: str = "hard",
        mixed_weights: Optional[Dict[str, float]] = None,
        random_seed: int = 42
    ) -> "SteeringConfig":
        """
        Create config from list of modes with probabilities.
        
        Args:
            modes: List of (mode, probability) tuples
            multi_label_mode: Target label mode
            mixed_weights: Weights for MIXED mode
            random_seed: Random seed
            
        Returns:
            SteeringConfig instance
        """
        return cls(
            mode=modes,
            multi_label_mode=multi_label_mode,
            mixed_weights=mixed_weights or {},
            random_seed=random_seed
        )
    
    def get_mode_probabilities(self) -> Dict[SteeringMode, float]:
        """
        Get mode probabilities as a dictionary.
        
        Returns:
            Dictionary mapping modes to probabilities
        """
        return {mode: prob for mode, prob in self.mode}
    
    def has_mode(self, mode: SteeringMode) -> bool:
        """
        Check if a specific mode is configured.
        
        Args:
            mode: Steering mode to check
            
        Returns:
            True if mode is in configuration
        """
        return any(m == mode for m, _ in self.mode)
