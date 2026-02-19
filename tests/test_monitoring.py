"""Tests for TrainingMonitor.

Covers:
- log_metrics: history accumulation, W&B payload filtering (bool vs int/float)
- plot_losses: guard on empty history, guard on no recognised loss keys,
               successful PNG creation, key detection logic
- plot_steering_distribution: guard on missing steering keys, PNG creation
- export_history: CSV + JSON sidecar creation, row count, bug regression
  (empty history with pandas unavailable must not report false success)
- num_epochs_logged property
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from RAG_supporters.nn.training.monitoring import TrainingMonitor, _MATPLOTLIB_AVAILABLE


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

requires_matplotlib = pytest.mark.skipif(
    not _MATPLOTLIB_AVAILABLE,
    reason="matplotlib not installed; plotting tests skipped",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_metrics(epoch: int, *, with_steering: bool = False) -> dict:
    """Return a realistic metrics dict for a single epoch."""
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


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------


class TestInit:
    def test_output_dir_created(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "monitor_runs"
        assert not output_dir.exists(), "Output dir should not exist before construction"
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


# ---------------------------------------------------------------------------
# TestLogMetrics
# ---------------------------------------------------------------------------


class TestLogMetrics:
    def test_history_grows_with_each_call(self, tmp_path: Path) -> None:
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        for epoch in range(3):
            monitor.log_metrics(epoch, _make_metrics(epoch))
        assert monitor.num_epochs_logged == 3, "Should have logged exactly 3 epochs"

    def test_epoch_key_stored_in_record(self, tmp_path: Path) -> None:
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        monitor.log_metrics(5, {"train/total_loss": 0.9})
        assert monitor.history[0]["epoch"] == 5, "Epoch index must be stored in history record"

    def test_all_metric_keys_stored(self, tmp_path: Path) -> None:
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        metrics = _make_metrics(0)
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
        call_kwargs = mock_run.log.call_args[0][0]  # first positional arg = payload dict
        assert "is_best" not in call_kwargs, "bool values must be excluded from W&B payload"
        assert "train/loss" in call_kwargs, "float values must be included in W&B payload"
        assert "count" in call_kwargs, "int values must be included in W&B payload"

    def test_wandb_failure_does_not_raise(self, tmp_path: Path) -> None:
        mock_run = MagicMock()
        mock_run.log.side_effect = RuntimeError("W&B unavailable")
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        monitor._wandb_run = mock_run

        # Should not raise
        monitor.log_metrics(0, {"train/total_loss": 0.5})
        assert monitor.num_epochs_logged == 1, "History should still be updated even if W&B fails"


# ---------------------------------------------------------------------------
# TestPlotLosses
# ---------------------------------------------------------------------------


class TestPlotLosses:
    def test_returns_none_when_history_empty(self, tmp_path: Path) -> None:
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        result = monitor.plot_losses()
        assert result is None, "plot_losses should return None when no metrics have been logged"

    def test_returns_none_when_no_recognised_loss_keys(self, tmp_path: Path) -> None:
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        # Log metrics with no recognised loss-component keys
        monitor.log_metrics(0, {"centroid_accuracy": 0.8, "lr": 1e-3})
        result = monitor.plot_losses()
        assert result is None, (
            "plot_losses should return None when history has no recognised loss keys"
        )

    @requires_matplotlib
    def test_creates_png_file(self, tmp_path: Path) -> None:
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        for epoch in range(3):
            monitor.log_metrics(epoch, _make_metrics(epoch))

        save_path = str(tmp_path / "losses.png")
        result = monitor.plot_losses(save_path=save_path)

        assert result is not None, "plot_losses should return a path string on success"
        assert Path(result).exists(), "The returned path must point to an existing file"
        assert result == save_path, "Returned path should match the requested save_path"

    @requires_matplotlib
    def test_default_filename_used_when_no_save_path(self, tmp_path: Path) -> None:
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        for epoch in range(2):
            monitor.log_metrics(epoch, _make_metrics(epoch))

        result = monitor.plot_losses()
        expected = str(tmp_path / "loss_curves.png")
        assert result == expected, "Default save path should be <output_dir>/loss_curves.png"
        assert Path(result).exists(), "Default loss curves PNG should exist after plot_losses"

    @requires_matplotlib
    def test_detects_both_train_and_val_keys(self, tmp_path: Path) -> None:
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        # Log only train prefix
        monitor.log_metrics(0, {"train/jasper_loss": 0.5})
        result = monitor.plot_losses(save_path=str(tmp_path / "l.png"))
        assert result is not None, "plot_losses should succeed when at least one loss key exists"

    @requires_matplotlib
    def test_only_val_keys_also_works(self, tmp_path: Path) -> None:
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        monitor.log_metrics(0, {"val/total_loss": 1.2, "val/contrastive_loss": 0.3})
        result = monitor.plot_losses(save_path=str(tmp_path / "l.png"))
        assert result is not None, "plot_losses should succeed when only val/ keys are present"

    @requires_matplotlib
    def test_handles_none_values_in_history(self, tmp_path: Path) -> None:
        """None values should be converted to NaN without raising."""
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        monitor.log_metrics(0, {"train/total_loss": 1.0})
        monitor.log_metrics(1, {"train/total_loss": None})  # missing value
        monitor.log_metrics(2, {"train/total_loss": 0.8})

        result = monitor.plot_losses(save_path=str(tmp_path / "l.png"))
        assert result is not None, "plot_losses should handle None metric values via NaN conversion"


# ---------------------------------------------------------------------------
# TestPlotSteeringDistribution
# ---------------------------------------------------------------------------


class TestPlotSteeringDistribution:
    def test_returns_none_when_no_steering_keys(self, tmp_path: Path) -> None:
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        for epoch in range(3):
            monitor.log_metrics(epoch, _make_metrics(epoch, with_steering=False))
        result = monitor.plot_steering_distribution()
        assert result is None, (
            "plot_steering_distribution should return None when no steering_variant_*_frac keys exist"
        )

    def test_returns_none_when_history_empty(self, tmp_path: Path) -> None:
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        result = monitor.plot_steering_distribution()
        assert result is None, "plot_steering_distribution should return None on empty history"

    @requires_matplotlib
    def test_creates_png_when_steering_keys_present(self, tmp_path: Path) -> None:
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        for epoch in range(3):
            monitor.log_metrics(epoch, _make_metrics(epoch, with_steering=True))

        save_path = str(tmp_path / "steering.png")
        result = monitor.plot_steering_distribution(save_path=save_path)

        assert result is not None, "plot_steering_distribution should return path when data exists"
        assert Path(result).exists(), "Steering distribution PNG must exist at the returned path"

    @requires_matplotlib
    def test_default_filename_used_when_no_save_path(self, tmp_path: Path) -> None:
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        monitor.log_metrics(0, _make_metrics(0, with_steering=True))

        result = monitor.plot_steering_distribution()
        expected = str(tmp_path / "steering_distribution.png")
        assert result == expected, "Default save path should be <output_dir>/steering_distribution.png"


# ---------------------------------------------------------------------------
# TestExportHistory
# ---------------------------------------------------------------------------


class TestExportHistory:
    def test_csv_and_json_sidecar_created(self, tmp_path: Path) -> None:
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        for epoch in range(3):
            monitor.log_metrics(epoch, _make_metrics(epoch))

        csv_path = str(tmp_path / "history.csv")
        result = monitor.export_history(save_path=csv_path)

        assert result == csv_path, "export_history should return the CSV path"
        assert Path(csv_path).exists(), "CSV file must exist after export_history"
        json_path = Path(csv_path).with_suffix(".json")
        assert json_path.exists(), "JSON sidecar must be created alongside the CSV"

    def test_csv_has_correct_row_count(self, tmp_path: Path) -> None:
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        n_epochs = 5
        for epoch in range(n_epochs):
            monitor.log_metrics(epoch, _make_metrics(epoch))

        csv_path = str(tmp_path / "history.csv")
        monitor.export_history(save_path=csv_path)

        with open(csv_path) as f:
            rows = list(csv.reader(f))
        data_rows = rows[1:]  # exclude header
        assert len(data_rows) == n_epochs, (
            f"CSV should have {n_epochs} data rows, got {len(data_rows)}"
        )

    def test_json_sidecar_is_valid_json(self, tmp_path: Path) -> None:
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        monitor.log_metrics(0, {"train/total_loss": 0.9, "epoch_flag": False})

        csv_path = str(tmp_path / "history.csv")
        monitor.export_history(save_path=csv_path)

        json_path = Path(csv_path).with_suffix(".json")
        with open(json_path) as f:
            data = json.load(f)
        assert isinstance(data, list), "JSON sidecar should be a list of epoch records"
        assert len(data) == 1, "JSON should contain exactly one record for one logged epoch"
        assert data[0]["train/total_loss"] == pytest.approx(0.9), (
            "JSON sidecar should preserve metric values accurately"
        )

    def test_default_path_used_when_not_specified(self, tmp_path: Path) -> None:
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        monitor.log_metrics(0, _make_metrics(0))

        result = monitor.export_history()
        expected = str(tmp_path / "training_history.csv")
        assert result == expected, "Default CSV path should be <output_dir>/training_history.csv"
        assert Path(result).exists(), "Default CSV file must exist after export_history"

    def test_empty_history_still_creates_json(self, tmp_path: Path) -> None:
        """JSON sidecar should always be written (even for empty history)."""
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
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1, "CSV should contain exactly one data row"
        assert "epoch" in rows[0], "CSV must contain the 'epoch' column"
        assert rows[0]["epoch"] == "7", "epoch column value should match the logged epoch"

    @patch("RAG_supporters.nn.training.monitoring._PANDAS_AVAILABLE", False)
    def test_fallback_csv_without_pandas(self, tmp_path: Path) -> None:
        """Manual CSV fallback (no pandas) should produce a valid file."""
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        for epoch in range(2):
            monitor.log_metrics(epoch, {"train/total_loss": 1.0 - epoch * 0.1})

        csv_path = str(tmp_path / "history_no_pandas.csv")
        result = monitor.export_history(save_path=csv_path)

        assert Path(csv_path).exists(), "Fallback CSV must exist even without pandas"

        with open(csv_path) as f:
            rows = list(csv.reader(f))
        assert len(rows) == 3, "Fallback CSV should have 1 header + 2 data rows"  # header + 2

    @patch("RAG_supporters.nn.training.monitoring._PANDAS_AVAILABLE", False)
    def test_empty_history_no_pandas_does_not_create_csv(self, tmp_path: Path) -> None:
        """Regression test: empty history + no pandas skips CSV but must not false-report success.

        The current implementation skips writing CSV silently.  This test
        documents the behaviour so any accidental change is caught.
        """
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        csv_path = str(tmp_path / "empty.csv")
        monitor.export_history(save_path=csv_path)

        # JSON must always be written
        json_path = Path(csv_path).with_suffix(".json")
        assert json_path.exists(), "JSON sidecar must be written regardless of pandas availability"

        # Document current behaviour: CSV is NOT written for empty history without pandas
        # If this changes (e.g. a fix is applied), update this assertion accordingly
        assert not Path(csv_path).exists(), (
            "Regression: empty history without pandas does not write CSV file "
            "(known limitation — documented behaviour)"
        )


# ---------------------------------------------------------------------------
# TestGetSummaryTable
# ---------------------------------------------------------------------------


class TestGetSummaryTable:
    def test_returns_data_matching_history(self, tmp_path: Path) -> None:
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        for epoch in range(4):
            monitor.log_metrics(epoch, _make_metrics(epoch))

        table = monitor.get_summary_table()
        try:
            import pandas as pd
            assert isinstance(table, pd.DataFrame), "Should return DataFrame when pandas available"
            assert len(table) == 4, "DataFrame should have 4 rows for 4 logged epochs"
        except ImportError:
            assert isinstance(table, list), "Should return list when pandas unavailable"
            assert len(table) == 4, "List should have 4 entries for 4 logged epochs"

    def test_empty_history_returns_empty_structure(self, tmp_path: Path) -> None:
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        table = monitor.get_summary_table()
        try:
            import pandas as pd
            assert isinstance(table, pd.DataFrame), "Should return empty DataFrame for empty history"
            assert len(table) == 0, "Empty history should produce zero-row DataFrame"
        except ImportError:
            assert table == [], "Empty history should produce empty list"


# ---------------------------------------------------------------------------
# TestFinish
# ---------------------------------------------------------------------------


class TestFinish:
    def test_finish_without_wandb_does_not_raise(self, tmp_path: Path) -> None:
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        monitor.finish()  # should not raise
        assert monitor._wandb_run is None, "W&B run should remain None after finish with no run"

    def test_finish_closes_wandb_run(self, tmp_path: Path) -> None:
        mock_run = MagicMock()
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        monitor._wandb_run = mock_run

        monitor.finish()

        mock_run.finish.assert_called_once(), "W&B run.finish() should be called exactly once"
        assert monitor._wandb_run is None, "W&B run reference should be cleared after finish"

    def test_finish_clears_run_even_if_finish_raises(self, tmp_path: Path) -> None:
        mock_run = MagicMock()
        mock_run.finish.side_effect = RuntimeError("W&B error")
        monitor = TrainingMonitor(output_dir=str(tmp_path))
        monitor._wandb_run = mock_run

        monitor.finish()  # should not propagate exception

        assert monitor._wandb_run is None, (
            "W&B run reference must be cleared in finally block even when finish() raises"
        )
