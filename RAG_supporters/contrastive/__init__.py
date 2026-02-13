"""Contrastive learning tools for hard negative mining and steering signals.

This module provides highly reusable tools for contrastive learning:
- Hard negative mining with 4-tier sampling strategy
- Steering signal generation for curriculum learning
- Distance-based negative sampling
- Centroid-based steering computation

These tools can be used for any contrastive learning project including metric learning,
siamese networks, and triplet loss training.

Key Features:
- 4-tier negative sampling (in-cluster, adjacent, hard, random)
- Centroid steering (normalized direction to cluster center)
- Keyword-weighted steering (weighted average of keywords)
- Residual steering (off-center signals)
- Curriculum learning support via distance metrics

Examples
--------
>>> from RAG_supporters.contrastive import NegativeMiner, SteeringBuilder
>>>
>>> # Mine hard negatives
>>> miner = NegativeMiner(
...     source_embeddings=source_embs,
...     question_embeddings=question_embs,
...     centroid_embeddings=centroid_embs,
...     pair_indices=pair_indices,
...     pair_cluster_ids=pair_cluster_ids,
...     source_cluster_ids=source_cluster_ids,
...     n_neg=12,
...     tier_proportions=[3, 4, 3, 2]
... )
>>> results = miner.mine_all_negatives()
>>>
>>> # Build steering signals
>>> builder = SteeringBuilder(
...     question_embeddings=q_embs,
...     keyword_embeddings=k_embs,
...     centroid_embeddings=c_embs,
...     pair_indices=indices,
...     pair_cluster_ids=cluster_ids,
...     pair_keyword_ids=keyword_ids
... )
>>> steering_results = builder.build_all_steering()
"""

from .mine_negatives import NegativeMiner, mine_negatives
from .build_steering import SteeringBuilder, build_steering

__all__ = [
    "NegativeMiner",
    "mine_negatives",
    "SteeringBuilder",
    "build_steering",
]
