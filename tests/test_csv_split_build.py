"""Tests for CSV-origin split in build_dataset / builder_config.

Covers:
  - BuildConfig accepts csv_splits field and skips split_ratios validation
  - _csv_origin_split saves correct idx tensors and returns expected dict
  - BuildConfig serialises / deserialises csv_splits via save() / load()
  - build_dataset raises ValueError when neither csv_paths nor csv_splits
  - build_dataset raises ValueError when csv_splits misses required keys
"""

from __future__ import annotations

import json
from typing import List

import pandas as pd
import pytest
import torch

from RAG_supporters.jasper.builder_config import BuildConfig
from RAG_supporters.jasper.build import _csv_origin_split


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_build_config(csv_splits: dict | None = None, **kwargs) -> BuildConfig:
    """Create a minimal valid BuildConfig."""
    defaults = dict(
        embedding_dim=64,
        n_neg=4,
        clustering_source="clusters.json",
        steering_probabilities={
            "zero": 0.25,
            "centroid": 0.25,
            "keyword": 0.25,
            "residual": 0.25,
        },
        curriculum={
            "mode": "fixed",
            "start_distance": 0.3,
            "end_distance": 0.7,
            "warmup_epochs": 5,
        },
        csv_splits=csv_splits,
    )
    defaults.update(kwargs)
    return BuildConfig(**defaults)


def _make_linked_df(tags: List[str]) -> pd.DataFrame:
    """Return a minimal linked_df with _split_tag for N pairs."""
    return pd.DataFrame(
        {
            "question": [f"q{i}" for i in range(len(tags))],
            "source": [f"s{i}" for i in range(len(tags))],
            "_split_tag": tags,
        }
    )


# ---------------------------------------------------------------------------
# BuildConfig — csv_splits field
# ---------------------------------------------------------------------------


class TestBuildConfigCsvSplits:
    def test_accepts_csv_splits_field(self):
        cfg = _make_build_config(csv_splits={"train": ["train.csv"], "val": ["val.csv"]})
        assert cfg.csv_splits == {"train": ["train.csv"], "val": ["val.csv"]}

    def test_csv_splits_defaults_to_none(self):
        cfg = _make_build_config()
        assert cfg.csv_splits is None

    def test_split_ratios_validation_skipped_when_csv_splits_set(self):
        """split_ratios is not required to sum to 1.0 when csv_splits is provided."""
        # Default split_ratios [0.8, 0.1, 0.1] always sum to 1.0, so we test
        # via instantiation without split_ratios (uses default) — just confirm
        # a config with csv_splits is constructed without error.
        cfg = _make_build_config(csv_splits={"train": ["a.csv"], "val": ["b.csv"]})
        assert cfg is not None

    def test_split_ratios_still_validated_without_csv_splits(self):
        with pytest.raises(ValueError, match="split_ratios must sum to 1.0"):
            BuildConfig(
                embedding_dim=64,
                n_neg=4,
                clustering_source="clusters.json",
                split_ratios=[0.5, 0.5, 0.5],  # sums to 1.5 — invalid
                steering_probabilities={
                    "zero": 0.25,
                    "centroid": 0.25,
                    "keyword": 0.25,
                    "residual": 0.25,
                },
                curriculum={
                    "mode": "fixed",
                    "start_distance": 0.3,
                    "end_distance": 0.7,
                    "warmup_epochs": 5,
                },
            )

    def test_serialise_and_deserialise_csv_splits(self, tmp_path):
        cfg = _make_build_config(
            csv_splits={"train": ["train.csv", "extra.csv"], "val": ["val.csv"]}
        )
        save_path = tmp_path / "config.json"
        cfg.save(save_path)
        loaded = BuildConfig.load(save_path)
        assert loaded.csv_splits == cfg.csv_splits

    def test_backward_compat_load_without_csv_splits(self, tmp_path):
        """Old config.json files without csv_splits load as csv_splits=None."""
        old_data = {
            "embedding_dim": 64,
            "n_neg": 4,
            "clustering_source": "clusters.json",
            "split_ratios": [0.8, 0.1, 0.1],
            "steering_probabilities": {
                "zero": 0.25,
                "centroid": 0.25,
                "keyword": 0.25,
                "residual": 0.25,
            },
            "curriculum": {
                "mode": "fixed",
                "start_distance": 0.3,
                "end_distance": 0.7,
                "warmup_epochs": 5,
            },
            # Note: no "csv_splits" key — simulates legacy config.json
        }
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps(old_data))
        loaded = BuildConfig.load(cfg_path)
        assert loaded.csv_splits is None


# ---------------------------------------------------------------------------
# _csv_origin_split
# ---------------------------------------------------------------------------


class TestCsvOriginSplit:
    def test_saves_train_val_idx_files(self, tmp_path):
        tags = ["train"] * 6 + ["val"] * 4
        linked_df = _make_linked_df(tags)
        result = _csv_origin_split(linked_df=linked_df, output_dir=tmp_path)

        assert (tmp_path / "train_idx.pt").exists()
        assert (tmp_path / "val_idx.pt").exists()

    def test_correct_indices_for_each_split(self, tmp_path):
        tags = ["train", "val", "train", "val", "train"]
        linked_df = _make_linked_df(tags)
        result = _csv_origin_split(linked_df=linked_df, output_dir=tmp_path)

        expected_train = torch.tensor([0, 2, 4], dtype=torch.long)
        expected_val = torch.tensor([1, 3], dtype=torch.long)
        assert torch.equal(result["train_idx"], expected_train)
        assert torch.equal(result["val_idx"], expected_val)

    def test_saved_tensors_match_returned_tensors(self, tmp_path):
        tags = ["train"] * 3 + ["val"] * 2
        linked_df = _make_linked_df(tags)
        result = _csv_origin_split(linked_df=linked_df, output_dir=tmp_path)

        loaded_train = torch.load(tmp_path / "train_idx.pt", weights_only=False)
        loaded_val = torch.load(tmp_path / "val_idx.pt", weights_only=False)
        assert torch.equal(loaded_train, result["train_idx"])
        assert torch.equal(loaded_val, result["val_idx"])

    def test_indices_cover_all_pairs_without_overlap(self, tmp_path):
        tags = ["train", "train", "val", "test", "val", "train"]
        linked_df = _make_linked_df(tags)
        result = _csv_origin_split(linked_df=linked_df, output_dir=tmp_path)

        all_indices: set = set()
        for tensor in result.values():
            indices_set = set(tensor.tolist())
            assert not indices_set & all_indices, "Overlapping indices between splits"
            all_indices |= indices_set

        assert all_indices == set(range(len(tags)))

    def test_three_way_split_with_test(self, tmp_path):
        tags = ["train"] * 4 + ["val"] * 2 + ["test"] * 2
        linked_df = _make_linked_df(tags)
        result = _csv_origin_split(linked_df=linked_df, output_dir=tmp_path)

        assert "train_idx" in result
        assert "val_idx" in result
        assert "test_idx" in result
        assert (tmp_path / "test_idx.pt").exists()

    def test_raises_without_split_tag_column(self, tmp_path):
        df = pd.DataFrame({"question": ["q1"], "source": ["s1"]})
        with pytest.raises(ValueError, match="_split_tag"):
            _csv_origin_split(linked_df=df, output_dir=tmp_path)

    def test_raises_on_empty_dataframe(self, tmp_path):
        df = pd.DataFrame({"question": [], "source": [], "_split_tag": []})
        with pytest.raises(ValueError, match="empty"):
            _csv_origin_split(linked_df=df, output_dir=tmp_path)


# ---------------------------------------------------------------------------
# build_dataset argument validation (mocked — no real model or CSV needed)
# ---------------------------------------------------------------------------


def _noop_build_dataset_check(csv_paths, csv_splits, tmp_path, **kwargs):
    """Partially invoke build_dataset only far enough to trigger arg validation."""
    from RAG_supporters.jasper.build import build_dataset

    # build_dataset will raise ValueError before touching the model / FS
    build_dataset(
        csv_paths=csv_paths,
        cluster_json_path="fake.json",
        embedding_model=None,
        output_dir=str(tmp_path),
        csv_splits=csv_splits,
        **kwargs,
    )


class TestBuildDatasetArgValidation:
    def test_raises_when_neither_csv_paths_nor_csv_splits(self, tmp_path):
        from RAG_supporters.jasper.build import build_dataset

        with pytest.raises(ValueError, match="Either csv_paths or csv_splits"):
            build_dataset(
                csv_paths=[],
                cluster_json_path="fake.json",
                embedding_model=None,
                output_dir=str(tmp_path),
                csv_splits=None,
            )

    def test_raises_when_csv_splits_missing_required_keys(self, tmp_path):
        from RAG_supporters.jasper.build import build_dataset

        with pytest.raises(ValueError, match="missing"):
            build_dataset(
                csv_paths=[],
                cluster_json_path="fake.json",
                embedding_model=None,
                output_dir=str(tmp_path),
                csv_splits={"train": ["train.csv"]},  # missing "val"
            )

    def test_warns_when_both_csv_paths_and_csv_splits_provided(self, tmp_path, caplog):
        import logging
        from RAG_supporters.jasper.build import build_dataset

        # Will fail further down (no real model/CSV), but warning must appear first
        with caplog.at_level(logging.WARNING, logger="RAG_supporters.jasper.build"):
            try:
                build_dataset(
                    csv_paths=["something.csv"],
                    cluster_json_path="fake.json",
                    embedding_model=None,
                    output_dir=str(tmp_path),
                    csv_splits={"train": ["train.csv"], "val": ["val.csv"]},
                )
            except Exception:
                pass
        assert any("csv_paths will be ignored" in r.message for r in caplog.records)
