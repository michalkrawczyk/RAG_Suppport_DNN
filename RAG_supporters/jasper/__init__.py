"""JASPER-specific dataset builder orchestration.

This module implements the JASPER dataset builder pipeline (Tasks 1-9):
- Task 1: CSV merging and deduplication
- Task 2: Cluster parsing and keyword linking
- Task 3: Embedding generation
- Task 4: Steering signal computation
- Task 5: Hard negative mining
- Task 6: Dataset splitting
- Task 7: Artifact persistence
- Task 8: Validation and finalization
- Task 9: Orchestration and timing

These tools are specific to JASPER but demonstrate patterns for building
other contrastive learning datasets.

Key Features:
- End-to-end pipeline orchestration
- Task timing and profiling
- Configuration management (BuildConfig)
- SQLite storage backend
- Cross-validation and integrity checks
- Reproducible builds with random seeds

Examples
--------
>>> from RAG_supporters.jasper import build_dataset, BuildConfig
>>> from sentence_transformers import SentenceTransformer
>>>
>>> # Build complete JASPER dataset
>>> config = build_dataset(
...     csv_paths=["data1.csv", "data2.csv"],
...     cluster_json_path="clusters.json",
...     embedding_model=SentenceTransformer("all-MiniLM-L6-v2"),
...     output_dir="output/dataset",
...     n_neg=12,
...     split_ratios=[0.7, 0.15, 0.15],
...     random_seed=42
... )
>>> print(f"Built dataset with {config.n_pairs} pairs")
>>>
>>> # Load configuration later
>>> loaded_config = BuildConfig.load("output/dataset/config.json")
"""

from .build import build_dataset
from .builder_config import BuildConfig
from .finalize import DatasetFinalizer, finalize_dataset
from .sqlite_storage import SQLiteStorageManager

__all__ = [
    "build_dataset",
    "BuildConfig",
    "DatasetFinalizer",
    "finalize_dataset",
    "SQLiteStorageManager",
]
