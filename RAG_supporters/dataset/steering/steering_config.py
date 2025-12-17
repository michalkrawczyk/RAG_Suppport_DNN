"""Steering configuration dataclass."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .steering_mode import SteeringMode


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
            
            # Normalize probabilities
            total_prob = sum(prob for _, prob in self.mode)
            if total_prob > 0 and abs(total_prob - 1.0) > 1e-6:
                logging.warning(
                    f"Steering mode probabilities sum to {total_prob}, normalizing to 1.0"
                )
                self.mode = [(mode, prob / total_prob) for mode, prob in self.mode]
            
            # Validate all probabilities are non-negative
            for mode, prob in self.mode:
                if prob < 0:
                    raise ValueError(f"Probability for {mode} must be non-negative, got {prob}")
        
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
        """
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
        Create config from a list of modes with probabilities.
        
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
    
    def get_single_mode(self) -> Optional[SteeringMode]:
        """
        Get single mode if only one mode is configured.
        
        Returns:
            SteeringMode if single mode, None otherwise
        """
        if self.mode and len(self.mode) == 1:
            return self.mode[0][0]
        return None
    
    def is_multi_mode(self) -> bool:
        """Check if multiple modes are configured."""
        return len(self.mode) > 1
