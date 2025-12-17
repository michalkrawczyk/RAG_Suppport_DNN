"""Steering embedding modes enum."""

from enum import Enum


class SteeringMode(Enum):
    """Steering embedding modes for cluster/subspace steering."""

    SUGGESTION = "suggestion"  # Use suggestion embeddings
    LLM_GENERATED = "llm_generated"  # LLM-generated steering text
    CLUSTER_DESCRIPTOR = "cluster_descriptor"  # Cluster/topic descriptor embeddings
    ZERO = "zero"  # Zero baseline (no steering)
    MIXED = "mixed"  # Weighted combination of multiple modes
