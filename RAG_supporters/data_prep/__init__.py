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
>>> # Simple split (any integer-indexed dataset)
>>> splitter = DatasetSplitter(val_ratio=0.2, random_state=42)
>>> train_idx, val_idx = splitter.split(1000)
>>>
>>> # Stratified split with question-level grouping (JASPER pair tensors)
>>> splitter = DatasetSplitter(
...     pair_indices=pair_indices,
...     pair_cluster_ids=cluster_ids,
...     val_ratio=0.15,
...     test_ratio=0.15,
... )
>>> results = splitter.split()
"""

from .merge_csv import CSVMerger, merge_csv_files
from .dataset_splitter import DatasetSplitter, split_dataset, create_train_val_split

__all__ = [
    # CSV merging
    "CSVMerger",
    "merge_csv_files",
    # Dataset splitting (simple and stratified)
    "DatasetSplitter",
    "split_dataset",
    "create_train_val_split",
]
