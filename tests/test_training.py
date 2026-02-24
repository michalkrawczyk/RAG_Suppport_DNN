"""Tests for training infrastructure.

Merged from:
- test_jasper_trainer.py  (JASPERTrainer, JASPERTrainerConfig)
- test_monitoring.py      (TrainingMonitor)
"""

from __future__ import annotations

import copy
import csv
import json
import tempfile
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from RAG_supporters.nn.models.jasper_predictor import JASPERPredictor, JASPERPredictorConfig
from RAG_supporters.nn.models.ema_encoder import EMAEncoder
from RAG_supporters.nn.losses.jasper_losses import JASPERMultiObjectiveLoss
from RAG_supporters.nn.training.jasper_trainer import JASPERTrainer, JASPERTrainerConfig
from RAG_supporters.nn.training.monitoring import TrainingMonitor, _MATPLOTLIB_AVAILABLE


# ===========================================================================
# JASPERTrainer
# ===========================================================================

_TB, _TD, _TK, _TC = 8, 32, 4, 6  # batch, embedding dim, negatives, clusters


def _make_trainer_batch(batch_size: int = _TB) -> Dict[str, torch.Tensor]:
    torch.manual_seed(7)
    return {
        "question_emb": torch.randn(batch_size, _TD),
        "target_source_emb": torch.randn(batch_size, _TD),
        "steering": torch.randn(batch_size, _TD),
        "negative_embs": torch.randn(batch_size, _TK, _TD),
        "cluster_id": torch.randint(0, _TC, (batch_size,)),
        "relevance": torch.rand(batch_size),
        "centroid_distance": torch.rand(batch_size),
        "steering_variant": torch.randint(0, 4, (batch_size,)),
        "negative_tiers": torch.randint(0, 4, (batch_size, _TK)),
    }


class _MockDataset(torch.utils.data.Dataset):
    """Tiny in-memory dataset that mimics the JASPER batch structure."""

    def __init__(self, n_samples: int = 32):
        self.n = n_samples
        self.centroid_embs = torch.randn(_TC, _TD)
        torch.manual_seed(42)
        self._data = [_make_trainer_batch(batch_size=1) for _ in range(n_samples)]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {k: v.squeeze(0) for k, v in self._data[idx % len(self._data)].items()}

    def set_epoch(self, epoch: int) -> None:
        pass


def _make_trainer_loader(n_samples: int = 32, batch_size: int = _TB) -> DataLoader:
    dataset = _MockDataset(n_samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    loader.dataset_obj = dataset
    return loader


def _make_source_encoder(dim: int = _TD) -> nn.Module:
    return nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))


@pytest.fixture
def trainer_components(tmp_path):
    torch.manual_seed(0)
    model = JASPERPredictor(JASPERPredictorConfig(embedding_dim=_TD, hidden_dim=_TD, num_layers=1))
    ema = EMAEncoder(_make_source_encoder(_TD), tau_min=0.9, tau_max=0.99)
    loss_fn = JASPERMultiObjectiveLoss(
        lambda_jasper=1.0, lambda_contrastive=0.5, lambda_centroid=0.1, lambda_vicreg=0.1
    )
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(ema.online_encoder.parameters()), lr=1e-3
    )
    train_loader = _make_trainer_loader()
    val_loader = _make_trainer_loader()
    config = JASPERTrainerConfig(
        max_grad_norm=1.0,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        save_every_n_epochs=0,
        keep_last_n_checkpoints=2,
        device="cpu",
        mixed_precision=False,
    )
    trainer = JASPERTrainer(
        config=config,
        model=model,
        ema_encoder=ema,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    return trainer, model, ema


class TestJASPERTrainerInit:
    def test_device_is_cpu(self, trainer_components):
        trainer, *_ = trainer_components
        assert (
            trainer.device.type == "cpu"
        ), "Trainer device should be 'cpu' when config.device='cpu'"

    def test_checkpoint_dir_created(self, trainer_components, tmp_path):
        trainer, *_ = trainer_components
        assert Path(
            trainer.config.checkpoint_dir
        ).exists(), "Trainer __init__ should create the checkpoint directory"

    def test_centroids_loaded_from_dataset(self, trainer_components):
        trainer, *_ = trainer_components
        assert (
            trainer._centroid_embs is not None
        ), "Trainer should load centroid embeddings from the dataset"
        assert trainer._centroid_embs.shape == (
            _TC,
            _TD,
        ), f"Centroid embeddings should have shape ({_TC}, {_TD})"

    def test_from_dict_config(self, tmp_path):
        model = JASPERPredictor({"embedding_dim": _TD, "hidden_dim": _TD, "num_layers": 1})
        ema = EMAEncoder(_make_source_encoder(_TD))
        trainer = JASPERTrainer(
            config={"checkpoint_dir": str(tmp_path), "device": "cpu"},
            model=model,
            ema_encoder=ema,
            loss_fn=JASPERMultiObjectiveLoss(),
            optimizer=torch.optim.AdamW(model.parameters()),
            train_loader=_make_trainer_loader(),
            val_loader=_make_trainer_loader(),
        )
        assert (
            trainer.device.type == "cpu"
        ), "Trainer should resolve to CPU when config.device='cpu'"


class TestJASPERTrainerStep:
    def test_single_step_returns_metrics(self, trainer_components):
        trainer, *_ = trainer_components
        trainer._max_steps = 100
        batch = _make_trainer_batch()
        metrics = trainer._train_step(batch)
        assert "total" in metrics, "_train_step should return a dict with 'total' key"
        assert isinstance(metrics["total"], float), "metrics['total'] should be a Python float"

    def test_parameters_updated_after_step(self, trainer_components):
        trainer, model, _ = trainer_components
        trainer._max_steps = 100
        params_before = {n: p.data.clone() for n, p in model.named_parameters()}
        trainer._train_step(_make_trainer_batch())
        changed = any(
            not torch.equal(p.data, params_before[n]) for n, p in model.named_parameters()
        )
        assert changed, "No model parameters changed after training step"

    def test_ema_target_updated_after_step(self, trainer_components):
        trainer, _, ema = trainer_components
        trainer._max_steps = 100
        target_before = {n: p.data.clone() for n, p in ema.target_encoder.named_parameters()}
        trainer._train_step(_make_trainer_batch())
        changed = any(
            not torch.equal(p.data, target_before[n])
            for n, p in ema.target_encoder.named_parameters()
        )
        assert changed, "EMA target not updated after step"

    def test_loss_is_finite(self, trainer_components):
        trainer, *_ = trainer_components
        trainer._max_steps = 100
        metrics = trainer._train_step(_make_trainer_batch())
        assert not any(
            v != v or abs(v) == float("inf") for v in metrics.values() if isinstance(v, float)
        )


class TestJASPERTrainerEpoch:
    def test_train_epoch_returns_dict(self, trainer_components):
        trainer, *_ = trainer_components
        trainer._max_steps = len(trainer.train_loader)
        metrics = trainer.train_epoch(epoch=0)
        assert isinstance(metrics, dict), "train_epoch should return a dict of metrics"
        assert "total" in metrics, "train_epoch metrics dict must contain 'total' key"

    def test_validate_returns_dict(self, trainer_components):
        trainer, *_ = trainer_components
        metrics = trainer.validate()
        assert isinstance(metrics, dict), "validate() should return a dict of metrics"
        assert "total" in metrics, "validate() metrics dict must contain 'total' key"

    def test_global_step_increments(self, trainer_components):
        trainer, *_ = trainer_components
        trainer._max_steps = len(trainer.train_loader)
        before = trainer.global_step
        trainer.train_epoch(epoch=0)
        assert trainer.global_step == before + len(
            trainer.train_loader
        ), f"global_step should increase by {len(trainer.train_loader)} after one epoch"

    def test_model_in_train_mode_during_epoch(self, trainer_components):
        trainer, model, _ = trainer_components
        trainer._max_steps = 10
        trainer.train_epoch(epoch=0)
        assert model.training, "Model should still be in training mode after train_epoch completes"

    def test_validate_model_in_eval_mode(self, trainer_components):
        trainer, model, _ = trainer_components
        model.train()
        trainer.validate()
        assert not model.training, "Model should be in eval mode after validate() completes"


class TestJASPERTrainerCheckpoint:
    def test_save_creates_file(self, trainer_components, tmp_path):
        trainer, *_ = trainer_components
        ckpt_path = tmp_path / "test.pt"
        trainer.save_checkpoint(ckpt_path, epoch=3)
        assert ckpt_path.exists(), "save_checkpoint should create the checkpoint file on disk"

    def test_checkpoint_contains_required_keys(self, trainer_components, tmp_path):
        trainer, *_ = trainer_components
        ckpt_path = tmp_path / "test.pt"
        trainer.save_checkpoint(ckpt_path, epoch=3)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        for key in (
            "epoch",
            "global_step",
            "model_state_dict",
            "ema_state_dict",
            "optimizer_state_dict",
        ):
            assert key in ckpt, f"Missing key: {key}"

    def test_load_restores_epoch(self, trainer_components, tmp_path):
        trainer, *_ = trainer_components
        ckpt_path = tmp_path / "epoch_5.pt"
        trainer._global_step = 50
        trainer.save_checkpoint(ckpt_path, epoch=5, metrics={"val/total": 0.42})
        epoch, metrics = trainer.load_checkpoint(ckpt_path)
        assert epoch == 5, "load_checkpoint should restore the saved epoch number"
        assert (
            abs(metrics.get("val/total", 0) - 0.42) < 1e-6
        ), "load_checkpoint should restore the saved metrics"

    def test_load_restores_model_weights(self, trainer_components, tmp_path):
        trainer, model, _ = trainer_components
        ckpt_path = tmp_path / "weights.pt"
        original_weights = {n: p.data.clone() for n, p in model.named_parameters()}
        trainer.save_checkpoint(ckpt_path, epoch=0)
        with torch.no_grad():
            for p in model.parameters():
                p.data.fill_(999.0)
        trainer.load_checkpoint(ckpt_path)
        for n, p in model.named_parameters():
            assert torch.allclose(p.data, original_weights[n]), f"Weights not restored for {n}"

    def test_load_restores_global_step(self, trainer_components, tmp_path):
        trainer, *_ = trainer_components
        trainer._global_step = 123
        ckpt_path = tmp_path / "step.pt"
        trainer.save_checkpoint(ckpt_path, epoch=0)
        trainer._global_step = 0
        trainer.load_checkpoint(ckpt_path)
        assert (
            trainer.global_step == 123
        ), "load_checkpoint should restore global_step to the saved value"

    def test_checkpoint_rotation(self, trainer_components, tmp_path):
        trainer, *_ = trainer_components
        trainer.config.keep_last_n_checkpoints = 2
        paths = []
        for i in range(4):
            p = tmp_path / f"epoch_{i:04d}.pt"
            trainer.save_checkpoint(p, epoch=i)
            paths.append(p)
        existing = [p for p in paths if p.exists()]
        assert (
            len(existing) == 2
        ), f"With keep_last_n_checkpoints=2, only 2 files should remain; found {len(existing)}"
        assert paths[-1] in existing, "The most recent checkpoint should be kept"
        assert paths[-2] in existing, "The second most recent checkpoint should be kept"


class TestJASPERTrainerCurriculum:
    def test_set_epoch_called_on_dataset(self, trainer_components):
        trainer, *_ = trainer_components
        trainer._max_steps = len(trainer.train_loader)
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=32)
        mock_dataset.__getitem__ = trainer.train_loader.dataset_obj.__getitem__
        mock_dataset.centroid_embs = trainer.train_loader.dataset_obj.centroid_embs
        trainer.train_loader.dataset_obj = mock_dataset
        trainer.train_epoch(epoch=3)
        mock_dataset.set_epoch.assert_called_with(3)


# ===========================================================================
# TrainingMonitor
# ===========================================================================

requires_matplotlib = pytest.mark.skipif(
    not _MATPLOTLIB_AVAILABLE,
    reason="matplotlib not installed; plotting tests skipped",
)


def _make_monitor_metrics(epoch: int, *, with_steering: bool = False) -> dict:
    m = {
        "train/total_loss": 1.5 - epoch * 0.1,
        "train/jasper_loss": 0.8 - epoch * 0.05,
        "train/contrastive_loss": 0.4 - epoch * 0.02,
        "train/centroid_loss": 0.2,
        "train/vicreg_loss": 0.1,
        "val/total_loss": 1.6 - epoch * 0.08,
        "val/jasper_loss": 0.9 - epoch * 0.04,
        "centroid_accuracy": 0.7 + epoch * 0.01,
    }
    if with_steering:
        m["train/steering_variant_0_frac"] = 0.4
        m["train/steering_variant_1_frac"] = 0.3
        m["train/steering_variant_2_frac"] = 0.2
        m["train/steering_variant_3_frac"] = 0.1
    return m


class TestTrainingMonitorInit:
    def test_output_dir_created(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "monitor_runs"
        assert not output_dir.exists()
        TrainingMonitor(output_dir=str(output_dir))
        assert output_dir.exists(), "TrainingMonitor should create output_dir on init"

    def test_initial_epoch_count_is_zero(self, tmp_path: Path) -> None:
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        assert monitor.num_epochs_logged == 0, "No epochs should be logged at construction"

    def test_history_initially_empty(self, tmp_path: Path) -> None:
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        assert monitor.history == [], "History should be empty list at construction"

    def test_repr_contains_expected_info(self, tmp_path: Path) -> None:
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        r = repr(monitor)
        assert "epochs_logged=0" in r, "repr should contain epochs_logged count"
        assert "disabled" in r, "repr should indicate wandb is disabled by default"

    def test_no_wandb_by_default(self, tmp_path: Path) -> None:
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        assert monitor._wandb_run is None, "W&B run should be None when use_wandb=False"


class TestTrainingMonitorLogMetrics:
    def test_history_grows_with_each_call(self, tmp_path: Path) -> None:
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        for epoch in range(3):
            monitor.log_metrics(epoch, _make_monitor_metrics(epoch))
        assert monitor.num_epochs_logged == 3, "Should have logged exactly 3 epochs"

    def test_epoch_key_stored_in_record(self, tmp_path: Path) -> None:
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        monitor.log_metrics(5, {"train/total_loss": 0.9})
        assert monitor.history[0]["epoch"] == 5, "Epoch index must be stored in history record"

    def test_all_metric_keys_stored(self, tmp_path: Path) -> None:
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        metrics = _make_monitor_metrics(0)
        monitor.log_metrics(0, metrics)
        for key in metrics:
            assert key in monitor.history[0], f"Key '{key}' should be present in stored record"

    def test_bool_excluded_from_wandb_payload(self, tmp_path: Path) -> None:
        """bool is a subclass of int — must be excluded from W&B float payload."""
        mock_run = MagicMock()
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        monitor._wandb_run = mock_run
        monitor.log_metrics(0, {"train/loss": 1.0, "is_best": True, "count": 3})
        mock_run.log.assert_called_once()
        call_kwargs = mock_run.log.call_args[0][0]
        assert "is_best" not in call_kwargs, "bool values must be excluded from W&B payload"
        assert "train/loss" in call_kwargs, "float values must be included in W&B payload"
        assert "count" in call_kwargs, "int values must be included in W&B payload"

    def test_wandb_failure_does_not_raise(self, tmp_path: Path) -> None:
        mock_run = MagicMock()
        mock_run.log.side_effect = RuntimeError("W&B unavailable")
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        monitor._wandb_run = mock_run
        monitor.log_metrics(0, {"train/total_loss": 0.5})
        assert monitor.num_epochs_logged == 1, "History should still be updated even if W&B fails"


class TestTrainingMonitorPlotLosses:
    def test_returns_none_when_history_empty(self, tmp_path: Path) -> None:
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        assert (
            monitor.plot_losses() is None
        ), "plot_losses should return None when no metrics have been logged"

    def test_returns_none_when_no_recognised_loss_keys(self, tmp_path: Path) -> None:
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        monitor.log_metrics(0, {"centroid_accuracy": 0.8, "lr": 1e-3})
        assert (
            monitor.plot_losses() is None
        ), "plot_losses should return None when history has no recognised loss keys"

    @requires_matplotlib
    def test_creates_png_file(self, tmp_path: Path) -> None:
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        for epoch in range(3):
            monitor.log_metrics(epoch, _make_monitor_metrics(epoch))
        save_path = str(tmp_path / "losses.png")
        result = monitor.plot_losses(save_path=save_path)
        assert result is not None, "plot_losses should return a path string on success"
        assert Path(result).exists(), "The returned path must point to an existing file"
        assert result == save_path, "Returned path should match the requested save_path"

    @requires_matplotlib
    def test_default_filename_used_when_no_save_path(self, tmp_path: Path) -> None:
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        for epoch in range(2):
            monitor.log_metrics(epoch, _make_monitor_metrics(epoch))
        result = monitor.plot_losses()
        expected = str(tmp_path / "loss_curves.png")
        assert result == expected, "Default save path should be <output_dir>/loss_curves.png"
        assert Path(result).exists()

    @requires_matplotlib
    def test_handles_none_values_in_history(self, tmp_path: Path) -> None:
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        monitor.log_metrics(0, {"train/total_loss": 1.0})
        monitor.log_metrics(1, {"train/total_loss": None})
        monitor.log_metrics(2, {"train/total_loss": 0.8})
        result = monitor.plot_losses(save_path=str(tmp_path / "l.png"))
        assert result is not None, "plot_losses should handle None metric values via NaN conversion"


class TestTrainingMonitorPlotSteering:
    def test_returns_none_when_no_steering_keys(self, tmp_path: Path) -> None:
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        for epoch in range(3):
            monitor.log_metrics(epoch, _make_monitor_metrics(epoch, with_steering=False))
        assert (
            monitor.plot_steering_distribution() is None
        ), "plot_steering_distribution should return None when no steering_variant_*_frac keys exist"

    def test_returns_none_when_history_empty(self, tmp_path: Path) -> None:
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        assert (
            monitor.plot_steering_distribution() is None
        ), "plot_steering_distribution should return None on empty history"

    @requires_matplotlib
    def test_creates_png_when_steering_keys_present(self, tmp_path: Path) -> None:
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        for epoch in range(3):
            monitor.log_metrics(epoch, _make_monitor_metrics(epoch, with_steering=True))
        save_path = str(tmp_path / "steering.png")
        result = monitor.plot_steering_distribution(save_path=save_path)
        assert result is not None, "plot_steering_distribution should return path when data exists"
        assert Path(result).exists()


class TestTrainingMonitorExportHistory:
    def test_csv_and_json_sidecar_created(self, tmp_path: Path) -> None:
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        for epoch in range(3):
            monitor.log_metrics(epoch, _make_monitor_metrics(epoch))
        csv_path = str(tmp_path / "history.csv")
        result = monitor.export_history(save_path=csv_path)
        assert result == csv_path, "export_history should return the CSV path"
        assert Path(csv_path).exists(), "CSV file must exist after export_history"
        assert (
            Path(csv_path).with_suffix(".json").exists()
        ), "JSON sidecar must be created alongside the CSV"

    def test_csv_has_correct_row_count(self, tmp_path: Path) -> None:
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        n_epochs = 5
        for epoch in range(n_epochs):
            monitor.log_metrics(epoch, _make_monitor_metrics(epoch))
        csv_path = str(tmp_path / "history.csv")
        monitor.export_history(save_path=csv_path)
        with open(csv_path) as f:
            rows = list(csv.reader(f))
        assert (
            len(rows[1:]) == n_epochs
        ), f"CSV should have {n_epochs} data rows, got {len(rows[1:])}"

    def test_json_sidecar_is_valid_json(self, tmp_path: Path) -> None:
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        monitor.log_metrics(0, {"train/total_loss": 0.9, "epoch_flag": False})
        csv_path = str(tmp_path / "history.csv")
        monitor.export_history(save_path=csv_path)
        with open(Path(csv_path).with_suffix(".json")) as f:
            data = json.load(f)
        assert isinstance(data, list), "JSON sidecar should be a list of epoch records"
        assert len(data) == 1, "JSON should contain exactly one record for one logged epoch"

    def test_default_path_used_when_not_specified(self, tmp_path: Path) -> None:
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        monitor.log_metrics(0, _make_monitor_metrics(0))
        result = monitor.export_history()
        expected = str(tmp_path / "training_history.csv")
        assert result == expected, "Default CSV path should be <output_dir>/training_history.csv"
        assert Path(result).exists()

    def test_empty_history_still_creates_json(self, tmp_path: Path) -> None:
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        csv_path = str(tmp_path / "history.csv")
        monitor.export_history(save_path=csv_path)
        json_path = Path(csv_path).with_suffix(".json")
        assert json_path.exists(), "JSON sidecar must be created even when history is empty"
        with open(json_path) as f:
            data = json.load(f)
        assert data == [], "JSON sidecar should contain empty list for empty history"

    def test_csv_contains_epoch_column(self, tmp_path: Path) -> None:
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        monitor.log_metrics(7, {"train/total_loss": 0.5})
        csv_path = str(tmp_path / "history.csv")
        monitor.export_history(save_path=csv_path)
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert "epoch" in rows[0], "CSV must contain the 'epoch' column"
        assert rows[0]["epoch"] == "7", "epoch column value should match the logged epoch"

    @patch("RAG_supporters.nn.training.monitoring._PANDAS_AVAILABLE", False)
    def test_fallback_csv_without_pandas(self, tmp_path: Path) -> None:
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        for epoch in range(2):
            monitor.log_metrics(epoch, {"train/total_loss": 1.0 - epoch * 0.1})
        csv_path = str(tmp_path / "history_no_pandas.csv")
        monitor.export_history(save_path=csv_path)
        assert Path(csv_path).exists(), "Fallback CSV must exist even without pandas"
        with open(csv_path) as f:
            rows = list(csv.reader(f))
        assert len(rows) == 3, "Fallback CSV should have 1 header + 2 data rows"

    @patch("RAG_supporters.nn.training.monitoring._PANDAS_AVAILABLE", False)
    def test_empty_history_no_pandas_does_not_create_csv(self, tmp_path: Path) -> None:
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        csv_path = str(tmp_path / "empty.csv")
        monitor.export_history(save_path=csv_path)
        json_path = Path(csv_path).with_suffix(".json")
        assert json_path.exists(), "JSON sidecar must be written regardless of pandas availability"
        assert not Path(
            csv_path
        ).exists(), "Regression: empty history without pandas does not write CSV (known limitation)"


class TestTrainingMonitorGetSummaryTable:
    def test_returns_data_matching_history(self, tmp_path: Path) -> None:
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        for epoch in range(4):
            monitor.log_metrics(epoch, _make_monitor_metrics(epoch))
        table = monitor.get_summary_table()
        try:
            import pandas as pd

            assert isinstance(table, pd.DataFrame), "Should return DataFrame when pandas available"
            assert len(table) == 4, "DataFrame should have 4 rows for 4 logged epochs"
        except ImportError:
            assert (
                isinstance(table, list) and len(table) == 4
            ), "Should return list of 4 entries when pandas unavailable"

    def test_empty_history_returns_empty_structure(self, tmp_path: Path) -> None:
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        table = monitor.get_summary_table()
        try:
            import pandas as pd

            assert isinstance(table, pd.DataFrame) and len(table) == 0
        except ImportError:
            assert table == [], "Empty history should produce empty list"


class TestTrainingMonitorFinish:
    def test_finish_without_wandb_does_not_raise(self, tmp_path: Path) -> None:
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        monitor.finish()
        assert monitor._wandb_run is None

    def test_finish_closes_wandb_run(self, tmp_path: Path) -> None:
        mock_run = MagicMock()
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        monitor._wandb_run = mock_run
        monitor.finish()
        mock_run.finish.assert_called_once()
        assert monitor._wandb_run is None

    def test_finish_clears_run_even_if_finish_raises(self, tmp_path: Path) -> None:
        mock_run = MagicMock()
        mock_run.finish.side_effect = RuntimeError("W&B error")
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        monitor._wandb_run = mock_run
        monitor.finish()
        assert (
            monitor._wandb_run is None
        ), "W&B run reference must be cleared in finally block even when finish() raises"
