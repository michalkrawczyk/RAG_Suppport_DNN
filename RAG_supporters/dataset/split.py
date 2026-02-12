"""
Dataset Splitter for JASPER Steering Dataset Builder.

This module performs question-level stratified splitting with no leakage:
- All pairs from the same question stay in the same split
- Stratified by cluster for balanced representation
- Supports train/val/test splits with configurable ratios
- Deterministic with random seed for reproducibility

Key Features:
- Question-level grouping (no question leakage across splits)
- Stratified sampling by cluster label
- Handles edge cases (small clusters, uneven distributions)
- Validation of split properties
- Outputs PyTorch tensor indices
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

from .validation_utils import (
    validate_tensor_2d,
    validate_tensor_1d,
    validate_length_consistency,
    validate_split_ratios
)

LOGGER = logging.getLogger(__name__)


class DatasetSplitter:
    """Split dataset into train/val/test at question level with stratification.
    
    Ensures no question leakage: all pairs belonging to the same question
    are assigned to the same split. Uses stratified sampling by cluster
    to maintain balanced cluster representation across splits.
    
    Parameters
    ----------
    pair_indices : torch.Tensor
        Pair indices [n_pairs, 2] mapping to (question_idx, source_idx)
    pair_cluster_ids : torch.Tensor
        Primary cluster ID per pair [n_pairs]
    train_ratio : float, optional
        Training set ratio (default: 0.7)
    val_ratio : float, optional
        Validation set ratio (default: 0.15)
    test_ratio : float, optional
        Test set ratio (default: 0.15)
    random_seed : int, optional
        Random seed for reproducibility (default: 42)
    show_progress : bool, optional
        Show progress bars (default: True)
    
    Attributes
    ----------
    pair_indices : torch.Tensor
        Pair index mapping
    pair_cluster_ids : torch.Tensor
        Cluster assignments for pairs
    n_pairs : int
        Total number of pairs
    n_questions : int
        Total number of unique questions
    train_ratio : float
        Training set ratio
    val_ratio : float
        Validation set ratio
    test_ratio : float
        Test set ratio
    random_seed : int
        Random seed
    show_progress : bool
        Progress bar visibility
    
    Examples
    --------
    >>> import torch
    >>> from RAG_supporters.dataset import DatasetSplitter
    >>> 
    >>> # Prepare data
    >>> pair_indices = torch.randint(0, 100, (500, 2))
    >>> pair_cluster_ids = torch.randint(0, 10, (500,))
    >>> 
    >>> # Split dataset
    >>> splitter = DatasetSplitter(
    ...     pair_indices=pair_indices,
    ...     pair_cluster_ids=pair_cluster_ids,
    ...     train_ratio=0.7,
    ...     val_ratio=0.15,
    ...     test_ratio=0.15
    ... )
    >>> 
    >>> # Generate splits
    >>> results = splitter.split()
    >>> print(results['train_idx'].shape)  # [~350]
    >>> print(results['val_idx'].shape)    # [~75]
    >>> print(results['test_idx'].shape)   # [~75]
    """
    
    def __init__(
        self,
        pair_indices: torch.Tensor,
        pair_cluster_ids: torch.Tensor,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42,
        show_progress: bool = True
    ):
        """Initialize dataset splitter."""
        # Validate inputs
        self._validate_inputs(
            pair_indices,
            pair_cluster_ids,
            train_ratio,
            val_ratio,
            test_ratio
        )
        
        self.pair_indices = pair_indices
        self.pair_cluster_ids = pair_cluster_ids
        self.n_pairs = len(pair_indices)
        
        # Extract unique questions
        self.question_ids = pair_indices[:, 0].unique()
        self.n_questions = len(self.question_ids)
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.show_progress = show_progress
        
        # Set random seed
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        LOGGER.info(
            f"Initialized DatasetSplitter: {self.n_pairs} pairs, "
            f"{self.n_questions} questions, "
            f"splits={train_ratio:.2f}/{val_ratio:.2f}/{test_ratio:.2f}"
        )
    
    def _validate_inputs(
        self,
        pair_indices: torch.Tensor,
        pair_cluster_ids: torch.Tensor,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float
    ) -> None:
        """Validate constructor inputs."""
        # Validate tensor structures
        validate_tensor_2d(pair_indices, "pair_indices", expected_cols=2, min_rows=1)
        validate_tensor_1d(pair_cluster_ids, "pair_cluster_ids", min_length=1)
        
        # Validate length consistency
        n_pairs = pair_indices.shape[0]
        validate_length_consistency(
            (pair_cluster_ids, "pair_cluster_ids", n_pairs)
        )
        
        # Validate split ratios
        validate_split_ratios(train_ratio, val_ratio, test_ratio)
        
        LOGGER.debug("Input validation passed")
    
    def _build_question_to_pairs(self) -> Dict[int, List[int]]:
        """Build mapping from question ID to list of pair indices.
        
        Returns
        -------
        Dict[int, List[int]]
            Mapping: question_id -> list of pair indices
        """
        question_to_pairs = {}
        
        iterator = enumerate(self.pair_indices)
        if self.show_progress:
            iterator = tqdm(
                iterator,
                total=self.n_pairs,
                desc="Building question-to-pairs mapping"
            )
        
        for pair_idx, (question_id, _) in iterator:
            question_id = question_id.item()
            if question_id not in question_to_pairs:
                question_to_pairs[question_id] = []
            question_to_pairs[question_id].append(pair_idx)
        
        LOGGER.info(
            f"Built question-to-pairs mapping: {len(question_to_pairs)} questions"
        )
        
        return question_to_pairs
    
    def _assign_question_clusters(
        self,
        question_to_pairs: Dict[int, List[int]]
    ) -> Dict[int, int]:
        """Assign each question to its primary cluster via majority voting.
        
        Parameters
        ----------
        question_to_pairs : Dict[int, List[int]]
            Mapping from question ID to pair indices
        
        Returns
        -------
        Dict[int, int]
            Mapping: question_id -> primary_cluster_id
        """
        question_clusters = {}
        
        iterator = question_to_pairs.items()
        if self.show_progress:
            iterator = tqdm(
                iterator,
                total=len(question_to_pairs),
                desc="Assigning question clusters"
            )
        
        for question_id, pair_indices_list in iterator:
            # Get cluster IDs for all pairs of this question
            pair_clusters = self.pair_cluster_ids[pair_indices_list]
            
            # Majority voting: most frequent cluster
            unique_clusters, counts = torch.unique(pair_clusters, return_counts=True)
            primary_cluster = unique_clusters[counts.argmax()].item()
            
            question_clusters[question_id] = primary_cluster
        
        LOGGER.info(
            f"Assigned {len(question_clusters)} questions to primary clusters"
        )
        
        return question_clusters
    
    def _stratified_split_questions(
        self,
        question_clusters: Dict[int, int]
    ) -> Tuple[List[int], List[int], List[int]]:
        """Perform stratified split at question level.
        
        Parameters
        ----------
        question_clusters : Dict[int, int]
            Mapping: question_id -> primary_cluster_id
        
        Returns
        -------
        Tuple[List[int], List[int], List[int]]
            (train_question_ids, val_question_ids, test_question_ids)
        """
        # Group questions by cluster
        cluster_to_questions = {}
        for question_id, cluster_id in question_clusters.items():
            if cluster_id not in cluster_to_questions:
                cluster_to_questions[cluster_id] = []
            cluster_to_questions[cluster_id].append(question_id)
        
        n_clusters = len(cluster_to_questions)
        LOGGER.info(f"Splitting across {n_clusters} clusters")
        
        train_questions = []
        val_questions = []
        test_questions = []
        
        # Split each cluster independently
        for cluster_id, question_ids_list in cluster_to_questions.items():
            n_q = len(question_ids_list)
            
            # Shuffle questions within cluster
            rng = np.random.RandomState(self.random_seed + cluster_id)
            shuffled = np.array(question_ids_list)
            rng.shuffle(shuffled)
            
            # Calculate split points
            n_train = max(1, int(n_q * self.train_ratio))
            n_val = max(1, int(n_q * self.val_ratio))
            n_test = n_q - n_train - n_val
            
            # Handle edge case: ensure test set has at least 1 question
            if n_test == 0 and n_q >= 3:
                n_test = 1
                n_val = max(1, n_val - 1)
            
            # Split
            train_end = n_train
            val_end = n_train + n_val
            
            train_questions.extend(shuffled[:train_end].tolist())
            val_questions.extend(shuffled[train_end:val_end].tolist())
            test_questions.extend(shuffled[val_end:].tolist())
            
            LOGGER.debug(
                f"Cluster {cluster_id}: {n_q} questions -> "
                f"train={n_train}, val={n_val}, test={n_test}"
            )
        
        LOGGER.info(
            f"Split questions: train={len(train_questions)}, "
            f"val={len(val_questions)}, test={len(test_questions)}"
        )
        
        return train_questions, val_questions, test_questions
    
    def _questions_to_pair_indices(
        self,
        question_ids_list: List[int],
        question_to_pairs: Dict[int, List[int]]
    ) -> torch.Tensor:
        """Convert question IDs to pair indices.
        
        Parameters
        ----------
        question_ids_list : List[int]
            List of question IDs
        question_to_pairs : Dict[int, List[int]]
            Mapping from question ID to pair indices
        
        Returns
        -------
        torch.Tensor
            Tensor of pair indices [n_pairs_in_split]
        """
        pair_indices_list = []
        for question_id in question_ids_list:
            pair_indices_list.extend(question_to_pairs[question_id])
        
        return torch.tensor(pair_indices_list, dtype=torch.long)
    
    def split(self) -> Dict[str, torch.Tensor]:
        """Perform stratified question-level split.
        
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary with keys:
            - 'train_idx': Training pair indices [n_train_pairs]
            - 'val_idx': Validation pair indices [n_val_pairs]
            - 'test_idx': Test pair indices [n_test_pairs]
        
        Examples
        --------
        >>> splitter = DatasetSplitter(pair_indices, pair_cluster_ids)
        >>> results = splitter.split()
        >>> train_idx = results['train_idx']
        >>> val_idx = results['val_idx']
        >>> test_idx = results['test_idx']
        """
        LOGGER.info("Starting question-level stratified split")
        
        # Build question-to-pairs mapping
        question_to_pairs = self._build_question_to_pairs()
        
        # Assign each question to primary cluster
        question_clusters = self._assign_question_clusters(question_to_pairs)
        
        # Perform stratified split at question level
        train_questions, val_questions, test_questions = \
            self._stratified_split_questions(question_clusters)
        
        # Convert question IDs to pair indices
        train_idx = self._questions_to_pair_indices(train_questions, question_to_pairs)
        val_idx = self._questions_to_pair_indices(val_questions, question_to_pairs)
        test_idx = self._questions_to_pair_indices(test_questions, question_to_pairs)
        
        # Validate splits
        self._validate_splits(train_idx, val_idx, test_idx)
        
        LOGGER.info(
            f"Split complete: train={len(train_idx)} pairs, "
            f"val={len(val_idx)} pairs, test={len(test_idx)} pairs"
        )
        
        return {
            'train_idx': train_idx,
            'val_idx': val_idx,
            'test_idx': test_idx
        }
    
    def _validate_splits(
        self,
        train_idx: torch.Tensor,
        val_idx: torch.Tensor,
        test_idx: torch.Tensor
    ) -> None:
        """Validate split properties.
        
        Parameters
        ----------
        train_idx : torch.Tensor
            Training pair indices
        val_idx : torch.Tensor
            Validation pair indices
        test_idx : torch.Tensor
            Test pair indices
        
        Raises
        ------
        ValueError
            If validation fails
        """
        # Check no overlap between splits
        train_set = set(train_idx.tolist())
        val_set = set(val_idx.tolist())
        test_set = set(test_idx.tolist())
        
        overlap_train_val = train_set & val_set
        overlap_train_test = train_set & test_set
        overlap_val_test = val_set & test_set
        
        if overlap_train_val:
            raise ValueError(
                f"Train and val splits overlap: {len(overlap_train_val)} pairs"
            )
        
        if overlap_train_test:
            raise ValueError(
                f"Train and test splits overlap: {len(overlap_train_test)} pairs"
            )
        
        if overlap_val_test:
            raise ValueError(
                f"Val and test splits overlap: {len(overlap_val_test)} pairs"
            )
        
        # Check all pairs are included
        all_indices = train_set | val_set | test_set
        expected_indices = set(range(self.n_pairs))
        
        if all_indices != expected_indices:
            missing = expected_indices - all_indices
            extra = all_indices - expected_indices
            raise ValueError(
                f"Split indices mismatch: missing={len(missing)}, extra={len(extra)}"
            )
        
        # Check non-empty splits
        if len(train_idx) == 0:
            raise ValueError("Training split is empty")
        
        if len(val_idx) == 0:
            raise ValueError("Validation split is empty")
        
        if len(test_idx) == 0:
            raise ValueError("Test split is empty")
        
        # Verify no question leakage
        self._validate_no_question_leakage(train_idx, val_idx, test_idx)
        
        LOGGER.info("Split validation passed")
    
    def _validate_no_question_leakage(
        self,
        train_idx: torch.Tensor,
        val_idx: torch.Tensor,
        test_idx: torch.Tensor
    ) -> None:
        """Verify no questions appear in multiple splits.
        
        Parameters
        ----------
        train_idx : torch.Tensor
            Training pair indices
        val_idx : torch.Tensor
            Validation pair indices
        test_idx : torch.Tensor
            Test pair indices
        
        Raises
        ------
        ValueError
            If question leakage detected
        """
        # Extract question IDs from each split
        train_questions = set(self.pair_indices[train_idx, 0].tolist())
        val_questions = set(self.pair_indices[val_idx, 0].tolist())
        test_questions = set(self.pair_indices[test_idx, 0].tolist())
        
        # Check for overlaps
        leak_train_val = train_questions & val_questions
        leak_train_test = train_questions & test_questions
        leak_val_test = val_questions & test_questions
        
        if leak_train_val:
            raise ValueError(
                f"Question leakage between train and val: {len(leak_train_val)} questions"
            )
        
        if leak_train_test:
            raise ValueError(
                f"Question leakage between train and test: {len(leak_train_test)} questions"
            )
        
        if leak_val_test:
            raise ValueError(
                f"Question leakage between val and test: {len(leak_val_test)} questions"
            )
        
        LOGGER.info("No question leakage detected")


def split_dataset(
    pair_indices: torch.Tensor,
    pair_cluster_ids: torch.Tensor,
    output_dir: Union[str, Path],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    show_progress: bool = True
) -> Dict[str, torch.Tensor]:
    """Split dataset and save to PyTorch tensor files.
    
    Convenience function that creates a splitter, performs the split,
    and saves results to disk.
    
    Parameters
    ----------
    pair_indices : torch.Tensor
        Pair indices [n_pairs, 2] mapping to (question_idx, source_idx)
    pair_cluster_ids : torch.Tensor
        Primary cluster ID per pair [n_pairs]
    output_dir : str or Path
        Directory to save split indices
    train_ratio : float, optional
        Training set ratio (default: 0.7)
    val_ratio : float, optional
        Validation set ratio (default: 0.15)
    test_ratio : float, optional
        Test set ratio (default: 0.15)
    random_seed : int, optional
        Random seed for reproducibility (default: 42)
    show_progress : bool, optional
        Show progress bars (default: True)
    
    Returns
    -------
    Dict[str, torch.Tensor]
        Dictionary with 'train_idx', 'val_idx', 'test_idx'
    
    Examples
    --------
    >>> import torch
    >>> from RAG_supporters.dataset import split_dataset
    >>> 
    >>> pair_indices = torch.randint(0, 100, (500, 2))
    >>> pair_cluster_ids = torch.randint(0, 10, (500,))
    >>> 
    >>> results = split_dataset(
    ...     pair_indices=pair_indices,
    ...     pair_cluster_ids=pair_cluster_ids,
    ...     output_dir="output/splits"
    ... )
    >>> # Files saved: train_idx.pt, val_idx.pt, test_idx.pt
    """
    # Create splitter
    splitter = DatasetSplitter(
        pair_indices=pair_indices,
        pair_cluster_ids=pair_cluster_ids,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed,
        show_progress=show_progress
    )
    
    # Perform split
    results = splitter.split()
    
    # Save to disk
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(results['train_idx'], output_dir / 'train_idx.pt')
    torch.save(results['val_idx'], output_dir / 'val_idx.pt')
    torch.save(results['test_idx'], output_dir / 'test_idx.pt')
    
    LOGGER.info(f"Saved split indices to {output_dir}")
    
    return results
