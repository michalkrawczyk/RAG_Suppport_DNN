"""Finalize and validate JASPER dataset builder outputs.

This module implements Task 8 of the dataset builder pipeline:
- Cross-validate all generated output tensors/files
- Verify referential integrity and dimensional consistency
- Produce final ``config.json`` as single source of truth
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch

from .builder_config import BuildConfig
from .tensor_utils import load_tensor_artifact, load_multiple_tensors
from .validation_utils import (
    validate_tensor_1d,
    validate_tensor_2d,
    validate_pair_indices_bounds,
    validate_cluster_ids_bounds,
    validate_length_consistency,
    validate_keyword_ids_list,
    validate_no_nan_inf,
    validate_embedding_dimensions,
    validate_values_in_range,
)

LOGGER = logging.getLogger(__name__)


class DatasetFinalizer:
    """Validate built dataset artifacts and write final config.

    Parameters
    ----------
    output_dir : str or Path
        Dataset output directory containing generated artifacts.
    """

    REQUIRED_PT_FILES = [
        "question_embs.pt",
        "source_embs.pt",
        "keyword_embs.pt",
        "centroid_embs.pt",
        "pair_index.pt",
        "pair_cluster_id.pt",
        "pair_relevance.pt",
        "pair_keyword_ids.pt",
        "steering_centroid.pt",
        "steering_keyword_weighted.pt",
        "steering_residual.pt",
        "centroid_distances.pt",
        "hard_negatives.pt",
        "negative_tiers.pt",
        "train_idx.pt",
        "val_idx.pt",
        "test_idx.pt",
    ]

    def __init__(self, output_dir: Union[str, Path]):
        """Initialize finalizer."""
        self.output_dir = Path(output_dir)
        if not self.output_dir.exists():
            raise FileNotFoundError(f"Output directory not found: {self.output_dir}")

    def _validate_required_files_pt(self) -> None:
        """Validate all required PT artifacts exist."""
        missing_files = [
            file_name
            for file_name in self.REQUIRED_PT_FILES
            if not (self.output_dir / file_name).exists()
        ]

        if missing_files:
            raise FileNotFoundError(
                f"Missing required dataset files: {missing_files}"
            )

    def _load_pt_artifacts(self) -> Dict[str, object]:
        """Load all PT artifacts.

        Returns
        -------
        Dict[str, object]
            Loaded tensors and pair keyword IDs list.
        """
        # Load embedding tensors
        embedding_specs = [
            ("question_embs", "question_embs.pt", True, (None, None)),
            ("source_embs", "source_embs.pt", True, (None, None)),
            ("keyword_embs", "keyword_embs.pt", True, (None, None)),
            ("centroid_embs", "centroid_embs.pt", True, (None, None)),
        ]
        
        # Load pair-level tensors
        pair_specs = [
            ("pair_index", "pair_index.pt", True, (None, 2)),
            ("pair_cluster_id", "pair_cluster_id.pt", True, (None,)),
            ("pair_relevance", "pair_relevance.pt", True, (None,)),
        ]
        
        # Load steering tensors
        steering_specs = [
            ("steering_centroid", "steering_centroid.pt", True, (None, None)),
            ("steering_keyword_weighted", "steering_keyword_weighted.pt", True, (None, None)),
            ("steering_residual", "steering_residual.pt", True, (None, None)),
            ("centroid_distances", "centroid_distances.pt", True, (None,)),
        ]
        
        # Load negative tensors
        negative_specs = [
            ("hard_negatives", "hard_negatives.pt", True, (None, None)),
            ("negative_tiers", "negative_tiers.pt", True, (None, None)),
        ]
        
        # Load split indices
        split_specs = [
            ("train_idx", "train_idx.pt", True, (None,)),
            ("val_idx", "val_idx.pt", True, (None,)),
            ("test_idx", "test_idx.pt", True, (None,)),
        ]
        
        # Combine all specs
        all_specs = (
            embedding_specs + pair_specs + steering_specs + negative_specs + split_specs
        )
        
        # Load all tensors
        artifacts = load_multiple_tensors(self.output_dir, all_specs)
        
        # Load pair_keyword_ids separately (list not tensor)
        artifacts["pair_keyword_ids"] = load_tensor_artifact(
            self.output_dir, "pair_keyword_ids.pt",
            weights_only=False
        )

        return artifacts

    def _validate_embedding_tensor(
        self,
        tensor: torch.Tensor,
        name: str,
        expected_dim: Optional[int] = None,
    ) -> int:
        """Validate embedding tensor shape and return embedding dim."""
        validate_tensor_2d(tensor, name, expected_cols=expected_dim, min_rows=1)
        validate_no_nan_inf(tensor, name)
        return int(tensor.shape[1])

    def _validate_pair_keyword_ids(
        self,
        pair_keyword_ids: object,
        n_pairs: int,
        n_keywords: int,
    ) -> None:
        """Validate pair keyword IDs list structure and bounds."""
        validate_keyword_ids_list(
            pair_keyword_ids,
            n_pairs=n_pairs,
            n_keywords=n_keywords,
            name="pair_keyword_ids"
        )

    def _validate_splits(
        self,
        train_idx: torch.Tensor,
        val_idx: torch.Tensor,
        test_idx: torch.Tensor,
        n_pairs: int,
    ) -> Tuple[int, int, int]:
        """Validate split tensors cover all pairs without overlap."""
        # Validate structure of each split
        for split_name, split_tensor in (
            ("train_idx", train_idx),
            ("val_idx", val_idx),
            ("test_idx", test_idx),
        ):
            validate_tensor_1d(split_tensor, split_name, min_length=1)
            validate_values_in_range(
                split_tensor, split_name,
                min_value=0, max_value=n_pairs - 1, inclusive=True
            )

        # Check for overlaps
        train_set = set(train_idx.tolist())
        val_set = set(val_idx.tolist())
        test_set = set(test_idx.tolist())

        if train_set & val_set:
            raise ValueError("train_idx and val_idx overlap")
        if train_set & test_set:
            raise ValueError("train_idx and test_idx overlap")
        if val_set & test_set:
            raise ValueError("val_idx and test_idx overlap")

        # Verify complete coverage
        all_indices = train_set | val_set | test_set
        if len(all_indices) != n_pairs:
            raise ValueError(
                f"Split indices do not cover all pairs exactly once: "
                f"covered={len(all_indices)}, n_pairs={n_pairs}"
            )

        return len(train_idx), len(val_idx), len(test_idx)

    def _validate_pt_artifacts(self, artifacts: Dict[str, object]) -> Dict[str, int]:
        """Validate loaded PT artifacts and return computed statistics."""
        question_embs = artifacts["question_embs"]
        source_embs = artifacts["source_embs"]
        keyword_embs = artifacts["keyword_embs"]
        centroid_embs = artifacts["centroid_embs"]

        # Validate all embeddings have consistent dimensions
        embedding_dim = validate_embedding_dimensions(
            (question_embs, "question_embs"),
            (source_embs, "source_embs"),
            (keyword_embs, "keyword_embs"),
            (centroid_embs, "centroid_embs"),
        )
        
        # Validate no NaN/Inf in embeddings
        for emb_name, emb_tensor in [
            ("question_embs", question_embs),
            ("source_embs", source_embs),
            ("keyword_embs", keyword_embs),
            ("centroid_embs", centroid_embs),
        ]:
            validate_no_nan_inf(emb_tensor, emb_name)

        n_questions = int(question_embs.shape[0])
        n_sources = int(source_embs.shape[0])
        n_keywords = int(keyword_embs.shape[0])
        n_clusters = int(centroid_embs.shape[0])

        pair_index = artifacts["pair_index"]
        pair_cluster_id = artifacts["pair_cluster_id"]
        pair_relevance = artifacts["pair_relevance"]
        pair_keyword_ids = artifacts["pair_keyword_ids"]

        validate_tensor_2d(pair_index, "pair_index", expected_cols=2, min_rows=1)
        n_pairs = int(pair_index.shape[0])
        
        validate_pair_indices_bounds(
            pair_index,
            n_questions=n_questions,
            n_sources=n_sources,
            name="pair_index"
        )

        validate_tensor_1d(pair_cluster_id, "pair_cluster_id", expected_length=n_pairs)
        validate_cluster_ids_bounds(
            pair_cluster_id,
            n_clusters=n_clusters,
            name="pair_cluster_id"
        )

        validate_tensor_1d(pair_relevance, "pair_relevance", expected_length=n_pairs)
        validate_no_nan_inf(pair_relevance, "pair_relevance")
        validate_values_in_range(
            pair_relevance, "pair_relevance",
            min_value=0.0, max_value=1.0, inclusive=True
        )

        # Validate pair_keyword_ids
        self._validate_pair_keyword_ids(pair_keyword_ids, n_pairs=n_pairs, n_keywords=n_keywords)

        # Validate steering tensors
        for steering_name in (
            "steering_centroid",
            "steering_keyword_weighted",
            "steering_residual",
        ):
            steering_tensor = artifacts[steering_name]
            validate_tensor_2d(
                steering_tensor, steering_name,
                expected_cols=embedding_dim, min_rows=n_pairs
            )
            validate_length_consistency(
                (steering_tensor, steering_name, n_pairs)
            )
            validate_no_nan_inf(steering_tensor, steering_name)

        centroid_distances = artifacts["centroid_distances"]
        validate_tensor_1d(centroid_distances, "centroid_distances", expected_length=n_pairs)
        validate_no_nan_inf(centroid_distances, "centroid_distances")

        hard_negatives = artifacts["hard_negatives"]
        negative_tiers = artifacts["negative_tiers"]
        
        validate_tensor_2d(hard_negatives, "hard_negatives", min_rows=1)
        validate_tensor_2d(negative_tiers, "negative_tiers", min_rows=1)
        
        if negative_tiers.shape != hard_negatives.shape:
            raise ValueError(
                "negative_tiers shape must match hard_negatives shape"
            )
        
        validate_length_consistency(
            (hard_negatives, "hard_negatives", n_pairs)
        )

        n_neg = int(hard_negatives.shape[1])
        if n_neg <= 0:
            raise ValueError("hard_negatives must contain at least one negative per pair")
        
        # Validate negative_tiers matches structure
        validate_tensor_2d(
            negative_tiers,
            "negative_tiers",
            expected_cols=n_neg,
            min_rows=n_pairs
        )
        if negative_tiers.shape[0] != n_pairs:
            raise ValueError(
                f"negative_tiers must have {n_pairs} rows, "
                f"got {negative_tiers.shape[0]}"
            )
        
        # Validate value ranges
        validate_values_in_range(
            hard_negatives,
            "hard_negatives",
            min_value=0,
            max_value=n_sources - 1,
            inclusive=True
        )
        validate_values_in_range(
            negative_tiers,
            "negative_tiers",
            min_value=1,
            max_value=4,
            inclusive=True
        )

        validate_values_in_range(
            hard_negatives, "hard_negatives",
            min_value=0, max_value=n_sources - 1, inclusive=True
        )
        validate_values_in_range(
            negative_tiers, "negative_tiers",
            min_value=1, max_value=4, inclusive=True
        )

        true_sources = pair_index[:, 1].view(-1, 1)
        if torch.any(hard_negatives == true_sources):
            raise ValueError("hard_negatives contains true source for at least one pair")

        self._validate_splits(
            artifacts["train_idx"],
            artifacts["val_idx"],
            artifacts["test_idx"],
            n_pairs=n_pairs,
        )

        return {
            "embedding_dim": embedding_dim,
            "n_neg": n_neg,
            "n_pairs": n_pairs,
            "n_questions": n_questions,
            "n_sources": n_sources,
            "n_keywords": n_keywords,
            "n_clusters": n_clusters,
        }

    def _resolve_config(
        self,
        config: Optional[BuildConfig],
        clustering_source: Optional[Union[str, Path]],
    ) -> Tuple[BuildConfig, bool]:
        """Resolve config source: argument, existing file, or minimal constructor."""
        is_placeholder = False

        if config is not None:
            resolved = config
        else:
            config_path = self.output_dir / "config.json"
            if config_path.exists():
                resolved = BuildConfig.load(config_path)
            else:
                if clustering_source is None:
                    raise ValueError(
                        "clustering_source is required when config is not provided "
                        "and config.json does not exist"
                    )
                resolved = BuildConfig(
                    embedding_dim=1,
                    n_neg=1,
                    clustering_source=str(clustering_source),
                )
                is_placeholder = True

        if clustering_source is not None:
            resolved.clustering_source = str(clustering_source)

        return resolved, is_placeholder

    def finalize(
        self,
        config: Optional[BuildConfig] = None,
        clustering_source: Optional[Union[str, Path]] = None,
        save: bool = True,
    ) -> BuildConfig:
        """Validate outputs and write final config.json.

        Parameters
        ----------
        config : BuildConfig, optional
            Existing build configuration to enrich with computed statistics.
        clustering_source : str or Path, optional
            Optional override for clustering JSON path in final config.
        save : bool, optional
            Whether to persist final config.json (default: True).

        Returns
        -------
        BuildConfig
            Final validated configuration.
        """
        resolved_config, is_placeholder = self._resolve_config(config, clustering_source)

        if resolved_config.storage_format != "pt":
            raise NotImplementedError(
                "Task 8 finalization currently supports storage_format='pt' only"
            )

        self._validate_required_files_pt()
        artifacts = self._load_pt_artifacts()
        stats = self._validate_pt_artifacts(artifacts)

        # Allow placeholder values (embedding_dim=1, n_neg=1) to be auto-updated
        is_embedding_dim_placeholder = resolved_config.embedding_dim == 1
        is_n_neg_placeholder = resolved_config.n_neg == 1
        
        if not is_placeholder and not is_embedding_dim_placeholder and (
            resolved_config.embedding_dim != stats["embedding_dim"]
        ):
            raise ValueError(
                f"Config embedding_dim ({resolved_config.embedding_dim}) does not match "
                f"artifacts ({stats['embedding_dim']})"
            )

        if not is_placeholder and not is_n_neg_placeholder and resolved_config.n_neg != stats["n_neg"]:
            raise ValueError(
                f"Config n_neg ({resolved_config.n_neg}) does not match "
                f"artifacts ({stats['n_neg']})"
            )

        resolved_config.embedding_dim = stats["embedding_dim"]
        resolved_config.n_neg = stats["n_neg"]
        resolved_config.update_post_build(
            n_pairs=stats["n_pairs"],
            n_questions=stats["n_questions"],
            n_sources=stats["n_sources"],
            n_keywords=stats["n_keywords"],
            n_clusters=stats["n_clusters"],
        )

        if save:
            config_path = self.output_dir / "config.json"
            resolved_config.save(config_path)
            LOGGER.info(f"Saved final config to {config_path}")

        LOGGER.info(
            "Finalization successful: "
            f"pairs={resolved_config.n_pairs}, "
            f"questions={resolved_config.n_questions}, "
            f"sources={resolved_config.n_sources}, "
            f"keywords={resolved_config.n_keywords}, "
            f"clusters={resolved_config.n_clusters}, "
            f"embedding_dim={resolved_config.embedding_dim}, "
            f"n_neg={resolved_config.n_neg}"
        )

        return resolved_config


def finalize_dataset(
    output_dir: Union[str, Path],
    config: Optional[BuildConfig] = None,
    clustering_source: Optional[Union[str, Path]] = None,
    save: bool = True,
) -> BuildConfig:
    """Convenience function for dataset finalization.

    Parameters
    ----------
    output_dir : str or Path
        Dataset output directory.
    config : BuildConfig, optional
        Optional BuildConfig to enrich and validate.
    clustering_source : str or Path, optional
        Optional clustering JSON path override.
    save : bool, optional
        Whether to write ``config.json`` to disk (default: True).

    Returns
    -------
    BuildConfig
        Final validated configuration.
    """
    finalizer = DatasetFinalizer(output_dir=output_dir)
    return finalizer.finalize(
        config=config,
        clustering_source=clustering_source,
        save=save,
    )
