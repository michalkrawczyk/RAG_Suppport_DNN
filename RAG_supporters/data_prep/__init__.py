"""Generic data preprocessing utilities for RAG_supporters.

This module provides reusable CSV merging, deduplication, and dataset splitting
functionality that works with any tabular data format.

Key Features:
- CSV merging with column aliasing
- Deduplication with configurable merge rules
- Stratified splitting with no-leakage guarantees
- Works with any CSV format via column mapping

Examples
--------
>>> from RAG_supporters.data_prep import merge_csv_files, DatasetSplitter
>>>
>>> # Merge CSV files with deduplication
>>> df = merge_csv_files(
...     csv_paths=["data1.csv", "data2.csv"],
...     output_path="merged.csv"
... )
>>>
>>> # Stratified split with question-level grouping
>>> splitter = DatasetSplitter(
...     pair_indices=pair_indices,
...     pair_cluster_ids=cluster_ids,
...     train_ratio=0.7,
...     val_ratio=0.15,
...     test_ratio=0.15
... )
>>> results = splitter.split()
"""

from .merge_csv import CSVMerger, merge_csv_files
from .split import DatasetSplitter, split_dataset

__all__ = [
    # CSV merging
    "CSVMerger",
    "merge_csv_files",
    # Stratified splitting (no-leakage)
    "DatasetSplitter",
    "split_dataset",
]
