"""Build orchestrator for JASPER Steering Dataset pipeline.

This module implements Task 9 of the dataset builder pipeline:
- Run Tasks 1-8 in sequence
- Persist all required PT artifacts
- Log per-task timing and overall runtime
"""

import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd
import torch

from RAG_supporters.contrastive import build_steering
from .builder_config import BuildConfig
from RAG_supporters.embeddings_ops import generate_embeddings
from .finalize import finalize_dataset
from RAG_supporters.clustering_ops import link_sources
from RAG_supporters.data_prep import CSVMerger, merge_csv_files
from RAG_supporters.contrastive import mine_negatives
from RAG_supporters.clustering_ops import parse_clusters
from RAG_supporters.data_prep import split_dataset

LOGGER = logging.getLogger(__name__)


def _build_config_from_input(
    cluster_json_path: Union[str, Path],
    storage_format: str,
    include_inspection_file: bool,
    random_seed: int,
    n_neg: int,
    split_ratios: Sequence[float],
    steering_probabilities: Optional[Dict[str, float]],
    curriculum: Optional[Dict[str, Union[float, str]]],
    config: Optional[Union[BuildConfig, Dict[str, Any]]],
) -> BuildConfig:
    """Normalize user-provided config into BuildConfig."""
    if isinstance(config, BuildConfig):
        resolved = config
        resolved.clustering_source = str(cluster_json_path)
        return resolved

    if isinstance(config, dict):
        config_payload = dict(config)
        config_payload.setdefault("embedding_dim", 1)
        config_payload.setdefault("n_neg", n_neg)
        config_payload.setdefault("split_ratios", list(split_ratios))
        if steering_probabilities is not None:
            config_payload.setdefault("steering_probabilities", steering_probabilities)
        if curriculum is not None:
            config_payload.setdefault("curriculum", curriculum)
        config_payload.setdefault("storage_format", storage_format)
        config_payload.setdefault("include_inspection_file", include_inspection_file)
        config_payload.setdefault("random_seed", random_seed)
        config_payload["clustering_source"] = str(cluster_json_path)
        return BuildConfig(**config_payload)

    return BuildConfig(
        embedding_dim=1,
        n_neg=n_neg,
        clustering_source=str(cluster_json_path),
        split_ratios=list(split_ratios),
        steering_probabilities=(
            steering_probabilities
            if steering_probabilities is not None
            else {
                "zero": 0.25,
                "centroid": 0.25,
                "keyword": 0.25,
                "residual": 0.25,
            }
        ),
        curriculum=(
            curriculum
            if curriculum is not None
            else {
                "mode": "linear",
                "start_distance": 0.3,
                "end_distance": 0.7,
                "warmup_epochs": 10,
            }
        ),
        storage_format=storage_format,
        include_inspection_file=include_inspection_file,
        random_seed=random_seed,
    )


def _to_keyword_id_lists(
    keywords_series: pd.Series,
    keyword_to_id: Dict[str, int],
) -> List[List[int]]:
    """Convert pair keywords to keyword ID lists."""
    pair_keyword_ids: List[List[int]] = []

    for keywords in keywords_series.tolist():
        if isinstance(keywords, str):
            keyword_values = [kw.strip() for kw in keywords.split(",") if kw.strip()]
        elif isinstance(keywords, list):
            keyword_values = [str(kw).strip() for kw in keywords if str(kw).strip()]
        else:
            keyword_values = []

        keyword_ids = sorted({keyword_to_id[kw] for kw in keyword_values if kw in keyword_to_id})
        pair_keyword_ids.append(keyword_ids)

    return pair_keyword_ids


def _compute_source_cluster_ids(
    linked_df: pd.DataFrame,
    n_sources: int,
) -> torch.Tensor:
    """Compute a primary cluster assignment for each source.

    Majority voting over pair-level cluster assignments is used.
    Ties are broken by selecting the smallest cluster ID.
    """
    source_cluster_ids = torch.zeros(n_sources, dtype=torch.long)

    grouped = linked_df.groupby("source_id")["cluster_id"]
    for source_id, cluster_series in grouped:
        counts = cluster_series.value_counts()
        max_count = counts.max()
        top_clusters = counts[counts == max_count].index.tolist()
        source_cluster_ids[int(source_id)] = int(min(top_clusters))

    return source_cluster_ids


def _save_pair_level_artifacts(
    linked_df: pd.DataFrame,
    question_to_id: Dict[str, int],
    source_to_id: Dict[str, int],
    keyword_to_id: Dict[str, int],
    output_dir: Path,
) -> Dict[str, Union[torch.Tensor, List[List[int]]]]:
    """Build and save pair-level tensors required by downstream tasks."""
    pair_index = torch.tensor(
        [
            [question_to_id[question], source_to_id[source]]
            for question, source in zip(
                linked_df["question"].tolist(),
                linked_df["source"].tolist(),
            )
        ],
        dtype=torch.long,
    )

    pair_cluster_id = torch.tensor(
        linked_df["cluster_id"].astype(int).tolist(),
        dtype=torch.long,
    )

    pair_relevance = torch.tensor(
        linked_df["relevance_score"].astype(float).tolist(),
        dtype=torch.float32,
    )

    pair_keyword_ids = _to_keyword_id_lists(
        keywords_series=linked_df["keywords"],
        keyword_to_id=keyword_to_id,
    )

    source_cluster_ids = _compute_source_cluster_ids(
        linked_df=linked_df,
        n_sources=len(source_to_id),
    )

    torch.save(pair_index, output_dir / "pair_index.pt")
    torch.save(pair_cluster_id, output_dir / "pair_cluster_id.pt")
    torch.save(pair_relevance, output_dir / "pair_relevance.pt")
    torch.save(pair_keyword_ids, output_dir / "pair_keyword_ids.pt")

    return {
        "pair_index": pair_index,
        "pair_cluster_id": pair_cluster_id,
        "pair_relevance": pair_relevance,
        "pair_keyword_ids": pair_keyword_ids,
        "source_cluster_ids": source_cluster_ids,
    }


def _timed_task(
    task_name: str,
    fn: Any,
    timings: Dict[str, float],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Execute a task and record elapsed time in seconds."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    timings[task_name] = elapsed
    LOGGER.info("%s completed in %.3fs", task_name, elapsed)
    return result


def _csv_origin_split(
    linked_df: pd.DataFrame,
    output_dir: Path,
    show_progress: bool = True,
) -> Dict[str, torch.Tensor]:
    """Build and save split index tensors from the ``_split_tag`` column.

    Each row of *linked_df* maps 1-to-1 with a pair (insertion order from the
    tagged CSV merge).  Rows are grouped by their ``_split_tag`` value and
    saved as ``{tag}_idx.pt`` files inside *output_dir*.

    Parameters
    ----------
    linked_df : pd.DataFrame
        DataFrame produced by Task 3 that must contain a ``_split_tag`` column
        added during the CSV-tagged merge (Task 1).
    output_dir : Path
        Directory where ``{tag}_idx.pt`` files are written.
    show_progress : bool, optional
        Unused; present for a consistent signature with :func:`split_dataset`.

    Returns
    -------
    Dict[str, torch.Tensor]
        Mapping from ``"{tag}_idx"`` to a 1-D ``torch.long`` tensor of pair
        indices belonging to that split.

    Raises
    ------
    ValueError
        If the ``_split_tag`` column is missing or the DataFrame is empty.
    """
    if "_split_tag" not in linked_df.columns:
        raise ValueError(
            "linked_df must contain a '_split_tag' column for CSV-origin splitting. "
            "Pass csv_splits to build_dataset to enable tagged merging."
        )
    if linked_df.empty:
        raise ValueError("linked_df is empty; cannot build split indices.")

    split_to_indices: Dict[str, List[int]] = {}
    for i, tag in enumerate(linked_df["_split_tag"].tolist()):
        split_to_indices.setdefault(str(tag), []).append(i)

    result: Dict[str, torch.Tensor] = {}
    for split_name in sorted(split_to_indices):
        indices = split_to_indices[split_name]
        tensor = torch.tensor(indices, dtype=torch.long)
        result[f"{split_name}_idx"] = tensor
        torch.save(tensor, output_dir / f"{split_name}_idx.pt")
        LOGGER.info(
            "CSV-origin split '%s': %d pairs → %s_idx.pt",
            split_name,
            len(tensor),
            split_name,
        )

    return result


def build_dataset(
    csv_paths: List[Union[str, Path]],
    cluster_json_path: Union[str, Path],
    embedding_model: Any,
    output_dir: Union[str, Path],
    config: Optional[Union[BuildConfig, Dict[str, Any]]] = None,
    csv_splits: Optional[Dict[str, List[Union[str, Path]]]] = None,
    storage_format: str = "pt",
    include_inspection_file: bool = False,
    column_aliases: Optional[Dict[str, List[str]]] = None,
    n_neg: int = 12,
    split_ratios: Sequence[float] = (0.8, 0.1, 0.1),
    steering_probabilities: Optional[Dict[str, float]] = None,
    curriculum: Optional[Dict[str, Union[float, str]]] = None,
    link_fallback_strategy: str = "largest",
    parse_validate: bool = True,
    cosine_threshold: float = 0.7,
    embedding_batch_size: int = 32,
    normalize_embeddings: bool = False,
    validate_embeddings: bool = True,
    steering_normalize_residual: bool = False,
    steering_fallback_strategy: str = "centroid",
    tier_proportions: Optional[List[int]] = None,
    adjacent_k: int = 3,
    random_seed: int = 42,
    show_progress: bool = True,
) -> BuildConfig:
    """Build JASPER steering dataset by orchestrating Tasks 1-8.

    Parameters
    ----------
    csv_paths : List[str or Path]
        Input CSV files with question-source pairs.  Ignored when
        *csv_splits* is provided.
    cluster_json_path : str or Path
        Path to KeywordClusterer JSON output.
    embedding_model : Any
        Embedding model accepted by ``KeywordEmbedder``.
    output_dir : str or Path
        Target directory for built dataset artifacts.
    config : BuildConfig or Dict[str, Any], optional
        Optional preconfigured BuildConfig or dict payload.
    csv_splits : Dict[str, List[str or Path]], optional
        When provided, defines which CSV files belong to each split instead
        of using random stratified splitting.  Keys are split names
        (``"train"``, ``"val"``, ``"test"``); values are lists of CSV paths.
        Example::

            csv_splits={"train": ["train.csv"], "val": ["val.csv"]}

        When this argument is set, ``split_ratios`` is stored as metadata
        only; the actual train/val/test assignment is driven entirely by
        which CSV file each row originated from.
    storage_format : str, optional
        Storage format (currently only ``pt`` is supported end-to-end).
    include_inspection_file : bool, optional
        Whether to write optional ``inspection.json``.

    Returns
    -------
    BuildConfig
        Final validated configuration persisted to ``config.json``.
    """
    total_start = time.perf_counter()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Guard: exactly one of csv_paths / csv_splits must be provided
    # ------------------------------------------------------------------
    if csv_splits is not None:
        if csv_paths:
            LOGGER.warning(
                "Both csv_splits and csv_paths were provided; csv_paths will be ignored. "
                "All CSV files should be declared via csv_splits."
            )
        required_keys = {"train", "val"}
        missing = required_keys - set(csv_splits.keys())
        if missing:
            raise ValueError(
                f"csv_splits must contain at least 'train' and 'val' keys; missing: {missing}"
            )
    elif not csv_paths:
        raise ValueError("Either csv_paths or csv_splits must be provided.")

    resolved_config = _build_config_from_input(
        cluster_json_path=cluster_json_path,
        storage_format=storage_format,
        include_inspection_file=include_inspection_file,
        random_seed=random_seed,
        n_neg=n_neg,
        split_ratios=split_ratios,
        steering_probabilities=steering_probabilities,
        curriculum=curriculum,
        config=config,
    )

    # Store normalised csv_splits paths on the config for persistence in config.json
    if csv_splits is not None:
        resolved_config.csv_splits = {
            k: [str(p) for p in v] for k, v in csv_splits.items()
        }

    if resolved_config.storage_format != "pt":
        raise NotImplementedError(
            "Task 9 build orchestrator currently supports storage_format='pt' only"
        )

    timings: Dict[str, float] = {}

    LOGGER.info("Starting JASPER dataset build pipeline")

    # Task 1: CSV merge
    if csv_splits is not None:
        def _tagged_merge() -> pd.DataFrame:
            tagged_dfs: List[pd.DataFrame] = []
            for split_name, paths in csv_splits.items():
                split_df = merge_csv_files(
                    csv_paths=list(paths), column_aliases=column_aliases
                )
                split_df = split_df.copy()
                split_df["_split_tag"] = split_name
                tagged_dfs.append(split_df)
            merged = pd.concat(tagged_dfs, ignore_index=True)
            LOGGER.info(
                "CSV-split merge: %d total pairs from %d split groups (%s)",
                len(merged),
                len(csv_splits),
                ", ".join(f"{k}={len(v)}csv" for k, v in csv_splits.items()),
            )
            return merged

        merged_df = _timed_task("task_1_merge_csv", _tagged_merge, timings)
    else:
        merged_df = _timed_task(
            "task_1_merge_csv",
            merge_csv_files,
            timings,
            csv_paths=csv_paths,
            output_path=None,
            column_aliases=column_aliases,
        )

    # Optional inspection metadata
    if resolved_config.include_inspection_file:
        merger = CSVMerger(column_aliases=column_aliases)
        inspection_payload = merger.create_inspection_metadata(
            merged_df,
            clustering_source=str(cluster_json_path),
        )
        inspection_path = output_path / "inspection.json"
        import json

        with open(inspection_path, "w", encoding="utf-8") as inspection_file:
            json.dump(inspection_payload, inspection_file, indent=2)
        LOGGER.info("Saved inspection metadata to %s", inspection_path)

    # Task 2: Cluster parser
    cluster_parser = _timed_task(
        "task_2_parse_clusters",
        parse_clusters,
        timings,
        clustering_json_path=cluster_json_path,
        validate=parse_validate,
        cosine_threshold=cosine_threshold,
    )

    # Task 3: Source-cluster linking
    linked_df = _timed_task(
        "task_3_link_sources",
        link_sources,
        timings,
        df=merged_df,
        cluster_parser=cluster_parser,
        keywords_col="keywords",
        pair_id_col="pair_id",
        output_col="cluster_id",
        fallback_strategy=link_fallback_strategy,
        show_progress=show_progress,
    )

    # Task 4: Embeddings
    embeddings = _timed_task(
        "task_4_generate_embeddings",
        generate_embeddings,
        timings,
        df=linked_df,
        embedding_model=embedding_model,
        cluster_parser=cluster_parser,
        output_dir=output_path,
        question_col="question",
        source_col="source",
        keywords_col="keywords",
        batch_size=embedding_batch_size,
        normalize_embeddings=normalize_embeddings,
        validate=validate_embeddings,
    )

    pair_artifacts = _timed_task(
        "task_4b_save_pair_artifacts",
        _save_pair_level_artifacts,
        timings,
        linked_df=linked_df,
        question_to_id=embeddings["question_to_id"],
        source_to_id=embeddings["source_to_id"],
        keyword_to_id=embeddings["keyword_to_id"],
        output_dir=output_path,
    )

    # Task 5: Steering vectors
    _timed_task(
        "task_5_build_steering",
        build_steering,
        timings,
        question_embeddings=embeddings["question_embs"],
        keyword_embeddings=embeddings["keyword_embs"],
        centroid_embeddings=embeddings["centroid_embs"],
        pair_indices=pair_artifacts["pair_index"],
        pair_cluster_ids=pair_artifacts["pair_cluster_id"],
        pair_keyword_ids=pair_artifacts["pair_keyword_ids"],
        output_dir=output_path,
        normalize_residual=steering_normalize_residual,
        fallback_strategy=steering_fallback_strategy,
        show_progress=show_progress,
    )

    # Task 6: Hard negatives
    _timed_task(
        "task_6_mine_negatives",
        mine_negatives,
        timings,
        source_embeddings=embeddings["source_embs"],
        question_embeddings=embeddings["question_embs"],
        centroid_embeddings=embeddings["centroid_embs"],
        pair_indices=pair_artifacts["pair_index"],
        pair_cluster_ids=pair_artifacts["pair_cluster_id"],
        source_cluster_ids=pair_artifacts["source_cluster_ids"],
        n_neg=resolved_config.n_neg,
        output_dir=output_path,
        tier_proportions=tier_proportions,
        adjacent_k=adjacent_k,
        random_seed=resolved_config.random_seed,
        show_progress=show_progress,
    )

    # Task 7: Split
    if csv_splits is not None:
        _timed_task(
            "task_7_split",
            _csv_origin_split,
            timings,
            linked_df=linked_df,
            output_dir=output_path,
            show_progress=show_progress,
        )
    else:
        _timed_task(
            "task_7_split",
            split_dataset,
            timings,
            pair_indices=pair_artifacts["pair_index"],
            pair_cluster_ids=pair_artifacts["pair_cluster_id"],
            output_dir=output_path,
            train_ratio=resolved_config.split_ratios[0],
            val_ratio=resolved_config.split_ratios[1],
            test_ratio=resolved_config.split_ratios[2],
            random_seed=resolved_config.random_seed,
            show_progress=show_progress,
        )

    # Task 8: Finalize
    final_config = _timed_task(
        "task_8_finalize",
        finalize_dataset,
        timings,
        output_dir=output_path,
        config=resolved_config,
        clustering_source=cluster_json_path,
        save=True,
    )

    total_elapsed = time.perf_counter() - total_start
    LOGGER.info("Build finished in %.3fs", total_elapsed)
    LOGGER.info("Task timings: %s", {k: round(v, 3) for k, v in timings.items()})
    LOGGER.debug("Final config: %s", asdict(final_config))

    return final_config
