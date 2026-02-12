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
from .validation_utils import validate_tensor_2d, validate_keyword_ids_list

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
        
        embedding_dim = int(tensor.shape[1])
        
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            raise ValueError(f"{name} contains NaN or Inf values")

        return embedding_dim

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
        for split_name, split_tensor in (
            ("train_idx", train_idx),
            ("val_idx", val_idx),
            ("test_idx", test_idx),
        ):
            if not isinstance(split_tensor, torch.Tensor):
                raise TypeError(
                    f"{split_name} must be torch.Tensor, got {type(split_tensor)}"
                )
            if split_tensor.ndim != 1:
                raise ValueError(
                    f"{split_name} must be 1D, got shape {tuple(split_tensor.shape)}"
                )
            if len(split_tensor) == 0:
                raise ValueError(f"{split_name} cannot be empty")
            if split_tensor.min().item() < 0 or split_tensor.max().item() >= n_pairs:
                raise ValueError(
                    f"{split_name} has indices outside valid range [0, {n_pairs - 1}]"
                )

        train_set = set(train_idx.tolist())
        val_set = set(val_idx.tolist())
        test_set = set(test_idx.tolist())

        if train_set & val_set:
            raise ValueError("train_idx and val_idx overlap")
        if train_set & test_set:
            raise ValueError("train_idx and test_idx overlap")
        if val_set & test_set:
            raise ValueError("val_idx and test_idx overlap")

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

        embedding_dim = self._validate_embedding_tensor(question_embs, "question_embs")
        self._validate_embedding_tensor(source_embs, "source_embs", embedding_dim)
        self._validate_embedding_tensor(keyword_embs, "keyword_embs", embedding_dim)
        self._validate_embedding_tensor(centroid_embs, "centroid_embs", embedding_dim)

        n_questions = int(question_embs.shape[0])
        n_sources = int(source_embs.shape[0])
        n_keywords = int(keyword_embs.shape[0])
        n_clusters = int(centroid_embs.shape[0])

        pair_index = artifacts["pair_index"]
        pair_cluster_id = artifacts["pair_cluster_id"]
        pair_relevance = artifacts["pair_relevance"]
        pair_keyword_ids = artifacts["pair_keyword_ids"]

        if not isinstance(pair_index, torch.Tensor):
            raise TypeError(f"pair_index must be torch.Tensor, got {type(pair_index)}")
        if pair_index.ndim != 2 or pair_index.shape[1] != 2:
            raise ValueError(
                f"pair_index must have shape [n_pairs, 2], got {tuple(pair_index.shape)}"
            )

        n_pairs = int(pair_index.shape[0])
        if n_pairs <= 0:
            raise ValueError("pair_index must contain at least one pair")

        if pair_index[:, 0].min().item() < 0 or pair_index[:, 0].max().item() >= n_questions:
            raise ValueError("pair_index question indices are out of range")
        if pair_index[:, 1].min().item() < 0 or pair_index[:, 1].max().item() >= n_sources:
            raise ValueError("pair_index source indices are out of range")

        if not isinstance(pair_cluster_id, torch.Tensor):
            raise TypeError(
                f"pair_cluster_id must be torch.Tensor, got {type(pair_cluster_id)}"
            )
        if pair_cluster_id.ndim != 1 or pair_cluster_id.shape[0] != n_pairs:
            raise ValueError(
                "pair_cluster_id must be 1D and match pair count from pair_index"
            )
        if pair_cluster_id.min().item() < 0 or pair_cluster_id.max().item() >= n_clusters:
            raise ValueError("pair_cluster_id contains out-of-range cluster IDs")

        if not isinstance(pair_relevance, torch.Tensor):
            raise TypeError(f"pair_relevance must be torch.Tensor, got {type(pair_relevance)}")
        if pair_relevance.ndim != 1 or pair_relevance.shape[0] != n_pairs:
            raise ValueError("pair_relevance must be 1D and match n_pairs")
        if torch.isnan(pair_relevance).any() or torch.isinf(pair_relevance).any():
            raise ValueError("pair_relevance contains NaN or Inf")
        if pair_relevance.min().item() < 0.0 or pair_relevance.max().item() > 1.0:
            raise ValueError("pair_relevance values must be in range [0, 1]")

        self._validate_pair_keyword_ids(pair_keyword_ids, n_pairs=n_pairs, n_keywords=n_keywords)

        for steering_name in (
            "steering_centroid",
            "steering_keyword_weighted",
            "steering_residual",
        ):
            steering_tensor = artifacts[steering_name]
            if not isinstance(steering_tensor, torch.Tensor):
                raise TypeError(
                    f"{steering_name} must be torch.Tensor, got {type(steering_tensor)}"
                )
            if steering_tensor.shape != (n_pairs, embedding_dim):
                raise ValueError(
                    f"{steering_name} must have shape {(n_pairs, embedding_dim)}, "
                    f"got {tuple(steering_tensor.shape)}"
                )
            if torch.isnan(steering_tensor).any() or torch.isinf(steering_tensor).any():
                raise ValueError(f"{steering_name} contains NaN or Inf")

        centroid_distances = artifacts["centroid_distances"]
        if not isinstance(centroid_distances, torch.Tensor):
            raise TypeError(
                f"centroid_distances must be torch.Tensor, got {type(centroid_distances)}"
            )
        if centroid_distances.ndim != 1 or centroid_distances.shape[0] != n_pairs:
            raise ValueError("centroid_distances must be 1D and match n_pairs")
        if torch.isnan(centroid_distances).any() or torch.isinf(centroid_distances).any():
            raise ValueError("centroid_distances contains NaN or Inf")

        hard_negatives = artifacts["hard_negatives"]
        negative_tiers = artifacts["negative_tiers"]
        if not isinstance(hard_negatives, torch.Tensor):
            raise TypeError(
                f"hard_negatives must be torch.Tensor, got {type(hard_negatives)}"
            )
        if not isinstance(negative_tiers, torch.Tensor):
            raise TypeError(
                f"negative_tiers must be torch.Tensor, got {type(negative_tiers)}"
            )
        if hard_negatives.ndim != 2:
            raise ValueError(
                f"hard_negatives must be 2D, got shape {tuple(hard_negatives.shape)}"
            )
        if negative_tiers.shape != hard_negatives.shape:
            raise ValueError(
                "negative_tiers shape must match hard_negatives shape"
            )
        if hard_negatives.shape[0] != n_pairs:
            raise ValueError("hard_negatives first dimension must match n_pairs")

        n_neg = int(hard_negatives.shape[1])
        if n_neg <= 0:
            raise ValueError("hard_negatives must contain at least one negative per pair")

        if hard_negatives.min().item() < 0 or hard_negatives.max().item() >= n_sources:
            raise ValueError("hard_negatives contains out-of-range source IDs")
        if negative_tiers.min().item() < 1 or negative_tiers.max().item() > 4:
            raise ValueError("negative_tiers values must be in range [1, 4]")

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

        if not is_placeholder and (
            resolved_config.embedding_dim != stats["embedding_dim"]
        ):
            raise ValueError(
                f"Config embedding_dim ({resolved_config.embedding_dim}) does not match "
                f"artifacts ({stats['embedding_dim']})"
            )

        if not is_placeholder and resolved_config.n_neg != stats["n_neg"]:
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
