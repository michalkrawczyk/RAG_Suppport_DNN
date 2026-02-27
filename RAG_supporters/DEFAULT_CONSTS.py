"""Shared pipeline key constants for the JASPER dataset builder.

This module defines three groups of frozen-dataclass constants that act as the
single source of truth for string-keyed interfaces shared across sub-packages:

* :data:`DEFAULT_COL_KEYS` — normalised DataFrame column names produced by
  :class:`~RAG_supporters.data_prep.CSVMerger` and consumed through the whole
  build pipeline.
* :data:`DEFAULT_EMB_KEYS` — keys in the dict returned by
  :func:`~RAG_supporters.embeddings_ops.generate_embeddings`.
* :data:`DEFAULT_PA_KEYS` — keys in the pair-artifact dict produced by the
  internal ``_save_pair_level_artifacts`` helper in
  :mod:`RAG_supporters.jasper.build`.

Overriding defaults
-------------------
All three singletons are instances of ``frozen=True`` dataclasses, so they
cannot be mutated.  To use non-default column or key names for a single
pipeline call, create a modified copy with :func:`dataclasses.replace`::

    import dataclasses
    from RAG_supporters.DEFAULT_CONSTS import DEFAULT_COL_KEYS

    col_keys = dataclasses.replace(DEFAULT_COL_KEYS, question="q_text")

:func:`build_dataset` accepts ``col_key_overrides``, ``emb_key_overrides``,
and ``pa_key_overrides`` dicts that are forwarded to :func:`dataclasses.replace`
automatically.
"""

from dataclasses import dataclass
from typing import Dict, List

__all__ = [
    "ColKeys",
    "EmbKeys",
    "PairArtifactKeys",
    "DEFAULT_COL_KEYS",
    "DEFAULT_EMB_KEYS",
    "DEFAULT_PA_KEYS",
    "COLUMN_ALIASES",
    "DEFAULT_SUGGESTION_MIN_CONFIDENCE",
    "DEFAULT_TOPIC_MIN_PROBABILITY",
]


# ---------------------------------------------------------------------------
# Column key schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColKeys:
    """Normalised DataFrame column names produced by ``CSVMerger``.

    Attributes
    ----------
    question : str
        Column containing the question text.
    source : str
        Column containing the source/passage text.
    source_id : str
        Integer source index assigned during embedding generation.
    cluster_id : str
        Cluster assignment for each (question, source) pair.
    relevance_score : str
        Float relevance score in [0, 1].
    keywords : str
        Comma-separated or list-valued keyword field.
    split_tag : str
        Internal tag injected during CSV-split-aware merging.
    """

    question: str = "question"
    source: str = "source"
    source_id: str = "source_id"
    cluster_id: str = "cluster_id"
    relevance_score: str = "relevance_score"
    keywords: str = "keywords"
    split_tag: str = "_split_tag"


# ---------------------------------------------------------------------------
# Embedding dict key schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EmbKeys:
    """Keys in the dict returned by :func:`~RAG_supporters.embeddings_ops.generate_embeddings`.

    Attributes
    ----------
    question : str
        ``Tensor`` of shape ``[n_questions, dim]``.
    source : str
        ``Tensor`` of shape ``[n_sources, dim]``.
    keyword : str
        ``Tensor`` of shape ``[n_keywords, dim]``, or ``None`` when absent.
    centroid : str
        ``Tensor`` of shape ``[n_clusters, dim]``; absent when no
        ``cluster_parser`` / keywords are provided.
    question_to_id : str
        ``Dict[str, int]`` mapping question text to embedding row index.
    source_to_id : str
        ``Dict[str, int]`` mapping source text to embedding row index.
    keyword_to_id : str
        ``Dict[str, int]`` mapping keyword text to embedding row index.
    """

    question: str = "question_embs"
    source: str = "source_embs"
    keyword: str = "keyword_embs"
    centroid: str = "centroid_embs"
    question_to_id: str = "question_to_id"
    source_to_id: str = "source_to_id"
    keyword_to_id: str = "keyword_to_id"


# ---------------------------------------------------------------------------
# Pair-artifact dict key schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairArtifactKeys:
    """Keys in the pair-artifact dict produced by ``_save_pair_level_artifacts``.

    Attributes
    ----------
    index : str
        ``Tensor`` of shape ``[n_pairs, 2]`` — ``[question_id, source_id]``.
    cluster_id : str
        ``Tensor`` of shape ``[n_pairs]`` — per-pair cluster assignment.
    relevance : str
        ``Tensor`` of shape ``[n_pairs]`` — per-pair relevance score.
    keyword_ids : str
        ``List[List[int]]`` — per-pair keyword index lists.
    source_cluster_ids : str
        ``Tensor`` of shape ``[n_sources]`` — majority-vote cluster per source.
    """

    index: str = "pair_index"
    cluster_id: str = "pair_cluster_id"
    relevance: str = "pair_relevance"
    keyword_ids: str = "pair_keyword_ids"
    source_cluster_ids: str = "source_cluster_ids"


# ---------------------------------------------------------------------------
# Singleton defaults — import these in consumer modules
# ---------------------------------------------------------------------------

DEFAULT_COL_KEYS = ColKeys()
DEFAULT_EMB_KEYS = EmbKeys()
DEFAULT_PA_KEYS = PairArtifactKeys()

# ---------------------------------------------------------------------------
# Column alias map for CSVMerger
# ---------------------------------------------------------------------------

COLUMN_ALIASES: Dict[str, List[str]] = {
    "question": ["question", "question_text", "query"],
    "source": ["source_text", "source", "context", "passage"],
    "answer": ["answer", "answer_text", "response"],
    # text_source rows: structured LLM keyword extraction (extract_suggestions)
    # question rows: topic-relevance probability scores
    "keywords": [
        "extract_suggestions",
        "topic_relevance_prob_topic_scores",
        "keywords",
        "topics",
        "tags",
    ],
    "relevance_score": ["relevance_score", "score", "relevance", "label"],
}

# Default minimum confidence for extract_suggestions filtering
# (0.0 = include all terms)
DEFAULT_SUGGESTION_MIN_CONFIDENCE: float = 0.0

# Default minimum probability for topic_relevance_prob_topic_scores filtering.
# Topics below this threshold are excluded when building question keywords.
# Set conservatively (0.5) so only moderately relevant topics are kept;
# callers can lower this if topic coverage is too sparse.
DEFAULT_TOPIC_MIN_PROBABILITY: float = 0.5
