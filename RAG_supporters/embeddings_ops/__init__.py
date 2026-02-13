"""Embedding operations for batch generation and validation.

This module provides tools for generating embeddings with validation,
sanity checks, and batch processing support.

Key Features:
- Batch embedding generation with progress tracking
- Model type detection (sentence-transformers or LangChain)
- NaN/Inf detection and validation
- Centroid similarity validation
- Memory-efficient batch processing
- Steering embedding generation with multiple modes

Examples
--------
>>> from RAG_supporters.embeddings_ops import EmbeddingGenerator, generate_embeddings
>>> from sentence_transformers import SentenceTransformer
>>>
>>> model = SentenceTransformer("all-MiniLM-L6-v2")
>>> generator = EmbeddingGenerator(model, cluster_parser)
>>>
>>> # Generate embeddings for all dataset components
>>> embeddings = generator.generate_all_embeddings(df)
>>> print(embeddings.keys())
dict_keys(['question_embs', 'source_embs', 'keyword_embs', 'centroid_embs', ...])
"""

from .embed import EmbeddingGenerator, generate_embeddings
from .steering_config import SteeringConfig, SteeringMode
from .steering_embedding_generator import SteeringEmbeddingGenerator

__all__ = [
    "EmbeddingGenerator",
    "generate_embeddings",
    "SteeringConfig",
    "SteeringMode",
    "SteeringEmbeddingGenerator",
]
