"""Dataset splitting utilities.

Provides a unified DatasetSplitter supporting two modes:

- ``'simple'``: ratio-based split for any integer-indexed dataset.
  Produces ``numpy.ndarray`` indices and persists to JSON.
- ``'stratified'``: question-level stratified split with no leakage for JASPER
  pair-tensor data.  Produces ``torch.Tensor`` indices and persists as ``.pt``
  files.

Mode is auto-detected: if ``pair_indices`` is supplied the mode defaults to
``'stratified'``, otherwise to ``'simple'``.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

from RAG_supporters.data_validation import (
    validate_length_consistency,
    validate_tensor_1d,
    validate_tensor_2d,
)

LOGGER = logging.getLogger(__name__)


class DatasetSplitter:
    """Unified dataset splitter supporting simple and stratified modes.

    Parameters
    ----------
    mode : str, optional
        ``'simple'`` or ``'stratified'``.  Auto-detected: ``'stratified'`` when
        *pair_indices* is supplied, otherwise ``'simple'``.
    val_ratio : float, optional
        Validation set ratio (default: ``0.1``).
    test_ratio : float, optional
        Test set ratio (default: ``0.0``).  When ``0.0`` no test split is
        produced.
    random_state : int, optional
        Random seed for reproducibility (default: ``42``).
    pair_indices : torch.Tensor, optional
        ``[n_pairs, 2]`` tensor for stratified mode.
    pair_cluster_ids : torch.Tensor, optional
        ``[n_pairs]`` cluster ID tensor for stratified mode.
    train_ratio : float, optional
        Explicit train ratio for stratified mode.
        Defaults to ``1 - val_ratio - test_ratio``.
    show_progress : bool, optional
        Show tqdm bars (stratified mode, default: ``True``).
    shuffle : bool, optional
        Shuffle before splitting (simple mode, default: ``True``).
    """

    def __init__(
        self,
        mode: Optional[str] = None,
        val_ratio: float = 0.1,
        test_ratio: float = 0.0,
        random_state: int = 42,
        # Stratified mode inputs
        pair_indices: Optional[torch.Tensor] = None,
        pair_cluster_ids: Optional[torch.Tensor] = None,
        train_ratio: Optional[float] = None,
        show_progress: bool = True,
        # Simple mode options
        shuffle: bool = True,
    ) -> None:
        # --- mode resolution ---
        if mode is None:
            mode = "stratified" if pair_indices is not None else "simple"
        if mode not in ("simple", "stratified"):
            raise ValueError(f"mode must be 'simple' or 'stratified', got {mode!r}")

        self.mode = mode
        self.random_state = random_state
        self.show_progress = show_progress
        self.shuffle = shuffle

        # --- ratio validation ---
        self._validate_ratios(val_ratio, test_ratio)
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.train_ratio: float = (
            train_ratio if train_ratio is not None else 1.0 - val_ratio - test_ratio
        )

        # --- simple-mode result attributes ---
        self.train_indices: Optional[np.ndarray] = None
        self.val_indices: Optional[np.ndarray] = None
        self.test_indices: Optional[np.ndarray] = None
        self.dataset_size: Optional[int] = None

        # --- stratified-mode setup ---
        if mode == "stratified":
            if pair_indices is None or pair_cluster_ids is None:
                raise ValueError(
                    "pair_indices and pair_cluster_ids are required for mode='stratified'"
                )
            self._init_stratified(pair_indices, pair_cluster_ids)
        else:
            self.pair_indices = None
            self.pair_cluster_ids = None
            self.n_pairs: Optional[int] = None
            self.n_questions: Optional[int] = None
            self.question_ids = None

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_ratios(self, val_ratio: float, test_ratio: float) -> None:
        if val_ratio < 0:
            raise ValueError(f"val_ratio must be >= 0, got {val_ratio}")
        if test_ratio < 0:
            raise ValueError(f"test_ratio must be >= 0, got {test_ratio}")
        if val_ratio + test_ratio >= 1.0:
            raise ValueError(
                f"val_ratio ({val_ratio}) + test_ratio ({test_ratio}) must be < 1.0"
            )

    def _init_stratified(
        self,
        pair_indices: torch.Tensor,
        pair_cluster_ids: torch.Tensor,
    ) -> None:
        validate_tensor_2d(pair_indices, "pair_indices", expected_cols=2, min_rows=1)
        validate_tensor_1d(pair_cluster_ids, "pair_cluster_ids", min_length=1)
        n_pairs = pair_indices.shape[0]
        validate_length_consistency((pair_cluster_ids, "pair_cluster_ids", n_pairs))

        self.pair_indices = pair_indices
        self.pair_cluster_ids = pair_cluster_ids
        self.n_pairs = len(pair_indices)
        self.question_ids = pair_indices[:, 0].unique()
        self.n_questions = len(self.question_ids)

        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

        LOGGER.info(
            "Initialized DatasetSplitter (stratified): %d pairs, %d questions, "
            "ratios=%.2f/%.2f/%.2f",
            self.n_pairs,
            self.n_questions,
            self.train_ratio,
            self.val_ratio,
            self.test_ratio,
        )

    # ------------------------------------------------------------------
    # Public split entry-point
    # ------------------------------------------------------------------

    def split(
        self,
        dataset_size: Optional[int] = None,
        val_ratio: Optional[float] = None,
        shuffle: Optional[bool] = None,
    ) -> Union[Tuple[np.ndarray, np.ndarray], Dict[str, torch.Tensor]]:
        """Perform the split.

        Simple mode
        -----------
        Requires *dataset_size*.  Stores results on ``self.train_indices``,
        ``self.val_indices``, and ``self.test_indices`` (``None`` when
        ``test_ratio == 0``).  Returns ``(train_indices, val_indices)``.

        *val_ratio* and *shuffle* may be passed here to override constructor
        values (backward compatibility).

        Stratified mode
        ---------------
        Returns ``Dict[str, torch.Tensor]`` with ``'train_idx'``,
        ``'val_idx'``, and ``'test_idx'`` (only when ``test_ratio > 0``).

        Parameters
        ----------
        dataset_size : int, optional
            Total samples – required for simple mode.
        val_ratio : float, optional
            Override for simple mode.
        shuffle : bool, optional
            Override for simple mode.
        """
        if self.mode == "simple":
            return self._split_simple(dataset_size, val_ratio, shuffle)
        return self._split_stratified()

    # ------------------------------------------------------------------
    # Simple mode
    # ------------------------------------------------------------------

    def _split_simple(
        self,
        dataset_size: Optional[int],
        val_ratio: Optional[float],
        shuffle: Optional[bool],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if dataset_size is None:
            raise ValueError("dataset_size is required for simple mode split")
        if dataset_size <= 0:
            raise ValueError(f"dataset_size must be positive, got {dataset_size}")

        _val_ratio = val_ratio if val_ratio is not None else self.val_ratio
        _test_ratio = self.test_ratio
        _shuffle = shuffle if shuffle is not None else self.shuffle

        val_size = int(dataset_size * _val_ratio)
        test_size = int(dataset_size * _test_ratio) if _test_ratio > 0 else 0
        train_size = dataset_size - val_size - test_size

        if val_size == 0:
            raise ValueError(
                f"Validation set would be empty with dataset_size={dataset_size} "
                f"and val_ratio={_val_ratio}."
            )
        if train_size <= 0:
            raise ValueError(
                f"Training set would be empty with dataset_size={dataset_size}, "
                f"val_ratio={_val_ratio}, test_ratio={_test_ratio}."
            )

        self.dataset_size = dataset_size

        indices = np.arange(dataset_size)
        if _shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(indices)

        # Slices: [val | test | train]
        self.val_indices = indices[:val_size]
        if test_size > 0:
            self.test_indices = indices[val_size : val_size + test_size]
        else:
            self.test_indices = None
        self.train_indices = indices[val_size + test_size :]

        LOGGER.info(
            "Split dataset of %d into train=%d, val=%d, test=%d",
            dataset_size,
            len(self.train_indices),
            len(self.val_indices),
            len(self.test_indices) if self.test_indices is not None else 0,
        )

        return self.train_indices, self.val_indices

    # ------------------------------------------------------------------
    # Stratified mode
    # ------------------------------------------------------------------

    def _split_stratified(self) -> Dict[str, torch.Tensor]:
        LOGGER.info("Starting question-level stratified split")

        question_to_pairs = self._build_question_to_pairs()
        question_clusters = self._assign_question_clusters(question_to_pairs)
        train_q, val_q, test_q = self._stratified_split_questions(question_clusters)

        train_idx = self._questions_to_pair_indices(train_q, question_to_pairs)
        val_idx = self._questions_to_pair_indices(val_q, question_to_pairs)
        test_idx = self._questions_to_pair_indices(test_q, question_to_pairs)

        self._validate_stratified_splits(train_idx, val_idx, test_idx)

        LOGGER.info(
            "Split complete: train=%d pairs, val=%d pairs, test=%d pairs",
            len(train_idx), len(val_idx), len(test_idx),
        )

        result: Dict[str, torch.Tensor] = {"train_idx": train_idx, "val_idx": val_idx}
        if self.test_ratio > 0:
            result["test_idx"] = test_idx
        return result

    def _build_question_to_pairs(self) -> Dict[int, List[int]]:
        question_to_pairs: Dict[int, List[int]] = {}
        iterator = enumerate(self.pair_indices)
        if self.show_progress:
            iterator = tqdm(
                iterator, total=self.n_pairs, desc="Building question-to-pairs mapping"
            )
        for pair_idx, (question_id, _) in iterator:
            qid = question_id.item()
            if qid not in question_to_pairs:
                question_to_pairs[qid] = []
            question_to_pairs[qid].append(pair_idx)
        LOGGER.info("Built question-to-pairs mapping: %d questions", len(question_to_pairs))
        return question_to_pairs

    def _assign_question_clusters(
        self, question_to_pairs: Dict[int, List[int]]
    ) -> Dict[int, int]:
        question_clusters: Dict[int, int] = {}
        iterator = question_to_pairs.items()
        if self.show_progress:
            iterator = tqdm(
                iterator,
                total=len(question_to_pairs),
                desc="Assigning question clusters",
            )
        for qid, pair_indices_list in iterator:
            pair_clusters = self.pair_cluster_ids[pair_indices_list]
            unique_clusters, counts = torch.unique(pair_clusters, return_counts=True)
            primary_cluster = unique_clusters[counts.argmax()].item()
            question_clusters[qid] = primary_cluster
        LOGGER.info("Assigned %d questions to primary clusters", len(question_clusters))
        return question_clusters

    def _stratified_split_questions(
        self, question_clusters: Dict[int, int]
    ) -> Tuple[List[int], List[int], List[int]]:
        cluster_to_questions: Dict[int, List[int]] = {}
        for qid, cid in question_clusters.items():
            if cid not in cluster_to_questions:
                cluster_to_questions[cid] = []
            cluster_to_questions[cid].append(qid)

        train_questions: List[int] = []
        val_questions: List[int] = []
        test_questions: List[int] = []

        for cluster_id, qids in cluster_to_questions.items():
            n_q = len(qids)
            rng = np.random.RandomState(self.random_state + cluster_id)
            shuffled = np.array(qids)
            rng.shuffle(shuffled)

            n_train = max(1, int(n_q * self.train_ratio))
            n_val = max(1, int(n_q * self.val_ratio))
            n_test = n_q - n_train - n_val

            # Force at least 1 test sample only when test_ratio > 0
            if self.test_ratio > 0 and n_test == 0 and n_q >= 3:
                n_test = 1
                if n_val > 1:
                    n_val -= 1
                elif n_train > 1:
                    n_train -= 1
                else:
                    n_test = 0

            train_end = n_train
            val_end = n_train + n_val

            train_questions.extend(shuffled[:train_end].tolist())
            val_questions.extend(shuffled[train_end:val_end].tolist())
            test_questions.extend(shuffled[val_end:].tolist())

            LOGGER.debug(
                "Cluster %d: %d questions -> train=%d, val=%d, test=%d",
                cluster_id, n_q, n_train, n_val, n_test,
            )

        LOGGER.info(
            "Split questions: train=%d, val=%d, test=%d",
            len(train_questions), len(val_questions), len(test_questions),
        )
        return train_questions, val_questions, test_questions

    def _questions_to_pair_indices(
        self, question_ids_list: List[int], question_to_pairs: Dict[int, List[int]]
    ) -> torch.Tensor:
        pair_idx_list: List[int] = []
        for qid in question_ids_list:
            pair_idx_list.extend(question_to_pairs[qid])
        return torch.tensor(pair_idx_list, dtype=torch.long)

    def _validate_stratified_splits(
        self,
        train_idx: torch.Tensor,
        val_idx: torch.Tensor,
        test_idx: torch.Tensor,
    ) -> None:
        train_set = set(train_idx.tolist())
        val_set = set(val_idx.tolist())
        test_set = set(test_idx.tolist())

        if train_set & val_set:
            raise ValueError(f"Train/val overlap: {len(train_set & val_set)} pairs")
        if train_set & test_set:
            raise ValueError(f"Train/test overlap: {len(train_set & test_set)} pairs")
        if val_set & test_set:
            raise ValueError(f"Val/test overlap: {len(val_set & test_set)} pairs")

        all_indices = train_set | val_set | test_set
        expected = set(range(self.n_pairs))
        if all_indices != expected:
            missing = expected - all_indices
            extra = all_indices - expected
            raise ValueError(
                f"Split indices mismatch: missing={len(missing)}, extra={len(extra)}"
            )

        if len(train_idx) == 0:
            raise ValueError("Training split is empty")
        if len(val_idx) == 0:
            raise ValueError("Validation split is empty")
        if self.test_ratio > 0 and len(test_idx) == 0:
            raise ValueError("Test split is empty (test_ratio > 0)")

        self._validate_no_question_leakage(train_idx, val_idx, test_idx)
        LOGGER.info("Split validation passed")

    def _validate_no_question_leakage(
        self,
        train_idx: torch.Tensor,
        val_idx: torch.Tensor,
        test_idx: torch.Tensor,
    ) -> None:
        train_q = set(self.pair_indices[train_idx, 0].tolist())
        val_q = set(self.pair_indices[val_idx, 0].tolist())
        test_q = set(self.pair_indices[test_idx, 0].tolist())

        if train_q & val_q:
            raise ValueError(
                f"Question leakage train/val: {len(train_q & val_q)} questions"
            )
        if train_q & test_q:
            raise ValueError(
                f"Question leakage train/test: {len(train_q & test_q)} questions"
            )
        if val_q & test_q:
            raise ValueError(
                f"Question leakage val/test: {len(val_q & test_q)} questions"
            )
        LOGGER.info("No question leakage detected")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(
        self,
        path: Union[str, Path],
        format: str = "auto",
        metadata: Optional[Dict] = None,
    ) -> None:
        """Save the split to disk.

        Parameters
        ----------
        path : str or Path
            JSON file path (simple mode) or directory for ``.pt`` files
            (stratified mode).
        format : str, optional
            ``'auto'``, ``'json'``, or ``'pt'``.  ``'auto'`` selects ``'json'``
            for simple mode.  Stratified ``.pt`` saving is handled by
            :func:`split_dataset`.
        metadata : dict, optional
            Extra fields for the JSON file (simple mode only).
        """
        resolved = format
        if resolved == "auto":
            resolved = "json" if self.mode == "simple" else "pt"

        if resolved == "json":
            self._save_json(path, metadata)
        elif resolved == "pt":
            raise NotImplementedError(
                "Saving stratified splits as .pt files is handled by "
                "split_dataset(output_dir=...) convenience function."
            )
        else:
            raise ValueError(f"format must be 'auto', 'json', or 'pt', got {resolved!r}")

    def _save_json(
        self,
        path: Union[str, Path],
        metadata: Optional[Dict] = None,
    ) -> None:
        if self.train_indices is None or self.val_indices is None:
            raise ValueError("No split generated. Call split() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data: Dict = {
            "mode": self.mode,
            "train_indices": self.train_indices.tolist(),
            "val_indices": self.val_indices.tolist(),
            "test_indices": (
                self.test_indices.tolist() if self.test_indices is not None else None
            ),
            "dataset_size": self.dataset_size,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "random_state": self.random_state,
            "metadata": metadata or {},
        }

        with open(path, "w") as fp:
            json.dump(data, fp, indent=2)

        LOGGER.info("Saved split to %s", path)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "DatasetSplitter":
        """Load a simple-mode DatasetSplitter from a JSON file.

        Parameters
        ----------
        path : str or Path
            Path to a JSON file written by :meth:`save`.

        Returns
        -------
        DatasetSplitter
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Split file not found: {path}")

        with open(path, "r") as fp:
            data = json.load(fp)

        required = ["train_indices", "val_indices", "dataset_size"]
        missing = [f for f in required if f not in data]
        if missing:
            raise ValueError(f"Invalid split file. Missing fields: {missing}")

        splitter = cls(
            mode="simple",
            val_ratio=data.get("val_ratio", 0.1),
            test_ratio=data.get("test_ratio", 0.0),
            random_state=data.get("random_state", 42),
        )
        splitter.train_indices = np.array(data["train_indices"])
        splitter.val_indices = np.array(data["val_indices"])
        splitter.test_indices = (
            np.array(data["test_indices"])
            if data.get("test_indices") is not None
            else None
        )
        splitter.dataset_size = data["dataset_size"]

        LOGGER.info(
            "Loaded split from %s: train=%d, val=%d",
            path, len(splitter.train_indices), len(splitter.val_indices),
        )
        return splitter

    # ------------------------------------------------------------------
    # Backward-compatibility aliases (used by ClusterLabeledDataset)
    # ------------------------------------------------------------------

    def save_split(
        self,
        output_path: Union[str, Path],
        metadata: Optional[Dict] = None,
    ) -> None:
        """Backward-compatible alias for :meth:`save` with JSON format."""
        self._save_json(output_path, metadata)

    @classmethod
    def load_split(cls, input_path: Union[str, Path]) -> "DatasetSplitter":
        """Backward-compatible alias for :meth:`from_file`."""
        return cls.from_file(input_path)

    def get_split(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(train_indices, val_indices)`` after a simple-mode split."""
        if self.train_indices is None or self.val_indices is None:
            raise ValueError("No split available. Call split() first or load from file.")
        return self.train_indices, self.val_indices

    def validate_split(self, dataset_size: int) -> bool:
        """Validate stored indices are compatible with *dataset_size*.

        Returns
        -------
        bool
            ``True`` if valid.
        """
        if self.train_indices is None or self.val_indices is None:
            raise ValueError("No split to validate.")

        all_idx: set = set(self.train_indices.tolist()) | set(self.val_indices.tolist())
        if self.test_indices is not None:
            all_idx |= set(self.test_indices.tolist())

        if all_idx and max(all_idx) >= dataset_size:
            raise ValueError(
                f"Split contains indices up to {max(all_idx)}, "
                f"but dataset size is {dataset_size}"
            )

        n_total = len(self.train_indices) + len(self.val_indices)
        if self.test_indices is not None:
            n_total += len(self.test_indices)
        if len(all_idx) != n_total:
            raise ValueError("Split contains duplicate indices")

        if self.dataset_size is not None and self.dataset_size != dataset_size:
            LOGGER.warning(
                "Dataset size mismatch: split created for %d, validating against %d",
                self.dataset_size, dataset_size,
            )

        return True


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def create_train_val_split(
    dataset_size: int,
    val_ratio: float = 0.2,
    random_state: Optional[int] = None,
    shuffle: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    metadata: Optional[Dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create a simple train/val split and return numpy arrays.

    Parameters
    ----------
    dataset_size : int
        Total number of samples.
    val_ratio : float, optional
        Fraction for validation (default: ``0.2``).
    random_state : int, optional
        Random seed.  Defaults to ``42``.
    shuffle : bool, optional
        Shuffle before splitting (default: ``True``).
    save_path : str or Path, optional
        If provided, save the split as JSON.
    metadata : dict, optional
        Extra metadata written to the JSON file.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray]
        ``(train_indices, val_indices)``
    """
    _seed = random_state if random_state is not None else 42
    splitter = DatasetSplitter(
        mode="simple",
        val_ratio=val_ratio,
        random_state=_seed,
        shuffle=shuffle,
    )
    train_indices, val_indices = splitter.split(dataset_size)

    if save_path is not None:
        splitter.save_split(save_path, metadata=metadata)

    return train_indices, val_indices


def split_dataset(
    pair_indices: torch.Tensor,
    pair_cluster_ids: torch.Tensor,
    output_dir: Optional[Union[str, Path]] = None,
    train_ratio: Optional[float] = None,
    val_ratio: float = 0.1,
    test_ratio: float = 0.0,
    random_state: int = 42,
    # Backward-compat alias (used in jasper/build.py)
    random_seed: Optional[int] = None,
    show_progress: bool = True,
) -> Dict[str, torch.Tensor]:
    """Stratified question-level split for JASPER pair tensors.

    Parameters
    ----------
    pair_indices : torch.Tensor
        ``[n_pairs, 2]`` tensor.
    pair_cluster_ids : torch.Tensor
        ``[n_pairs]`` cluster ID tensor.
    output_dir : str or Path, optional
        If provided, write ``train_idx.pt`` / ``val_idx.pt``
        (/ ``test_idx.pt``) here.
    train_ratio : float, optional
        Defaults to ``1 - val_ratio - test_ratio``.
    val_ratio : float, optional
        Default: ``0.1``.
    test_ratio : float, optional
        Default: ``0.0``.
    random_state : int, optional
        Random seed (default: ``42``).
    random_seed : int, optional
        Backward-compatible alias for *random_state*.
    show_progress : bool, optional
        Default: ``True``.

    Returns
    -------
    Dict[str, torch.Tensor]
        Keys: ``'train_idx'``, ``'val_idx'``, ``'test_idx'`` (when
        ``test_ratio > 0``).
    """
    _seed = random_seed if random_seed is not None else random_state
    _train = train_ratio if train_ratio is not None else 1.0 - val_ratio - test_ratio

    splitter = DatasetSplitter(
        mode="stratified",
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=_seed,
        pair_indices=pair_indices,
        pair_cluster_ids=pair_cluster_ids,
        train_ratio=_train,
        show_progress=show_progress,
    )

    results = splitter.split()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(results["train_idx"], output_dir / "train_idx.pt")
        torch.save(results["val_idx"], output_dir / "val_idx.pt")
        if "test_idx" in results:
            torch.save(results["test_idx"], output_dir / "test_idx.pt")
        LOGGER.info("Saved split indices to %s", output_dir)

    return results
