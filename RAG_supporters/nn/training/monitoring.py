"""Training monitor: metric collection, plotting, CSV export, and optional W&B logging."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger(__name__)

# Optional dependencies
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend — safe on servers
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False
    LOGGER.warning("matplotlib not installed; plotting will be skipped.")

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


class TrainingMonitor:
    """Collect, visualise, and optionally sync training metrics.

    Features
    --------
    - Per-epoch metric logging (internal history)
    - Loss curve plots (all loss components in separate sub-plots)
    - Steering variant distribution plot over epochs
    - Full history export to CSV and JSON
    - Optional Weights & Biases (W&B) integration — gracefully disabled when
      ``wandb`` is not installed or ``use_wandb=False``

    Args:
        output_dir: Directory where plots and CSV are saved.
        use_wandb: Whether to log to Weights & Biases.
        wandb_project: W&B project name.
        wandb_name: W&B run name (auto-generated if ``None``).
        wandb_config: Config dict to log to W&B.
    """

    # Keys that contain loss values (used for loss curve plot)
    _LOSS_KEYS = (
        "total",
        "jasper",
        "contrastive",
        "centroid",
        "vicreg",
        "vicreg_v",
        "vicreg_i",
        "vicreg_c",
    )

    def __init__(
        self,
        output_dir: str = "runs",
        use_wandb: bool = False,
        wandb_project: Optional[str] = "jasper-rag",
        wandb_name: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.history: List[Dict[str, Any]] = []
        self._wandb_run = None

        if use_wandb:
            if not _WANDB_AVAILABLE:
                LOGGER.warning("W&B requested but `wandb` is not installed; skipping W&B logging.")
            else:
                self._wandb_run = wandb.init(
                    project=wandb_project,
                    name=wandb_name,
                    config=wandb_config or {},
                    reinit=True,
                )
                LOGGER.info("W&B run initialised: %s", self._wandb_run.url)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_metrics(self, epoch: int, metrics_dict: Dict[str, Any]) -> None:
        """Record metrics for one epoch.

        Also syncs to W&B if enabled.

        Args:
            epoch: Epoch index (used as W&B step).
            metrics_dict: Dict of metric name → scalar value.
        """
        record = {"epoch": epoch, **metrics_dict}
        self.history.append(record)

        if self._wandb_run is not None:
            try:
                # Filter non-serialisable values before sending
                wandb_payload = {
                    k: float(v) for k, v in metrics_dict.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                }
                self._wandb_run.log(wandb_payload, step=epoch)
            except Exception as exc:
                LOGGER.warning("W&B log failed: %s", exc)

    def plot_losses(self, save_path: Optional[str] = None) -> Optional[str]:
        """Plot train/val loss curves for all tracked components.

        Args:
            save_path: File path for the saved figure.  Defaults to
                ``<output_dir>/loss_curves.png``.

        Returns:
            Absolute path to the saved figure, or ``None`` if matplotlib
            is unavailable or there is no data.
        """
        if not _MATPLOTLIB_AVAILABLE:
            LOGGER.warning("matplotlib not available; skipping plot_losses.")
            return None
        if not self.history:
            LOGGER.warning("No metrics logged yet; skipping plot_losses.")
            return None

        save_path = save_path or str(self.output_dir / "loss_curves.png")
        epochs = [r["epoch"] for r in self.history]

        # Collect all loss keys that appear in the data (both train/ and val/ prefixes)
        prefixes = ("train/", "val/")
        loss_components = sorted({
            k.split("/", 1)[1]
            for r in self.history
            for k in r
            if any(k.startswith(p) for p in prefixes)
            and any(lk in k for lk in self._LOSS_KEYS)
        })

        if not loss_components:
            LOGGER.warning("No recognised loss keys found; skipping plot_losses.")
            return None

        ncols = 2
        nrows = max((len(loss_components) + 1) // ncols, 1)
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)

        for idx, component in enumerate(loss_components):
            ax = axes[idx // ncols][idx % ncols]
            for prefix, style in (("train/", "-"), ("val/", "--")):
                key = f"{prefix}{component}"
                values = [r.get(key) for r in self.history]
                if any(v is not None for v in values):
                    y = [v if v is not None else float("nan") for v in values]
                    ax.plot(epochs, y, style, label=prefix.rstrip("/"))
            ax.set_title(component)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(len(loss_components), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        fig.suptitle("JASPER Training Loss Curves", fontsize=14)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

        LOGGER.info("Loss curves saved to %s", save_path)

        # Sync plot to W&B
        if self._wandb_run is not None:
            try:
                self._wandb_run.log({"loss_curves": wandb.Image(save_path)})
            except Exception as exc:
                LOGGER.warning("W&B image log failed: %s", exc)

        return save_path

    def plot_steering_distribution(self, save_path: Optional[str] = None) -> Optional[str]:
        """Plot the steering variant distribution over epochs.

        Expects metric keys like ``train/steering_variant_<N>_frac`` to be
        logged.  If no such keys are present the plot is skipped.

        Args:
            save_path: File path for the saved figure.

        Returns:
            Absolute path to the saved figure, or ``None``.
        """
        if not _MATPLOTLIB_AVAILABLE or not self.history:
            return None

        save_path = save_path or str(self.output_dir / "steering_distribution.png")
        epochs = [r["epoch"] for r in self.history]

        # Look for steering_variant_N_frac keys
        variant_keys = sorted({
            k
            for r in self.history
            for k in r
            if "steering_variant" in k and "frac" in k
        })

        if not variant_keys:
            LOGGER.debug("No steering variant fraction keys found; skipping steering distribution plot.")
            return None

        fig, ax = plt.subplots(figsize=(10, 5))
        for key in variant_keys:
            values = [r.get(key, float("nan")) for r in self.history]
            ax.plot(epochs, values, label=key.split("/", 1)[-1])

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Fraction")
        ax.set_title("Steering Variant Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

        LOGGER.info("Steering distribution plot saved to %s", save_path)

        if self._wandb_run is not None:
            try:
                self._wandb_run.log({"steering_distribution": wandb.Image(save_path)})
            except Exception as exc:
                LOGGER.warning("W&B image log failed: %s", exc)

        return save_path

    def export_history(self, save_path: Optional[str] = None) -> str:
        """Export the full metric history to CSV (and JSON sidecar).

        Args:
            save_path: CSV file path.  Defaults to
                ``<output_dir>/training_history.csv``.

        Returns:
            Path to the saved CSV file.
        """
        save_path = save_path or str(self.output_dir / "training_history.csv")
        json_path = Path(save_path).with_suffix(".json")

        # JSON export (always works)
        with open(json_path, "w") as f:
            json.dump(self.history, f, indent=2, default=_json_default)
        LOGGER.info("History JSON saved to %s", json_path)

        # CSV export (requires pandas)
        if _PANDAS_AVAILABLE:
            df = pd.DataFrame(self.history)
            df.to_csv(save_path, index=False)
            LOGGER.info("History CSV saved to %s", save_path)
        else:
            # Fallback: write CSV manually
            if self.history:
                all_keys = sorted({k for r in self.history for k in r})
                with open(save_path, "w") as f:
                    f.write(",".join(all_keys) + "\n")
                    for row in self.history:
                        f.write(",".join(str(row.get(k, "")) for k in all_keys) + "\n")
            LOGGER.info("History CSV saved to %s (pandas not available; basic CSV written)", save_path)

        return save_path

    def get_summary_table(self) -> Any:
        """Return training history as a pandas DataFrame.

        Returns:
            ``pandas.DataFrame`` if pandas is installed, otherwise a plain
            list of dicts.
        """
        if _PANDAS_AVAILABLE:
            return pd.DataFrame(self.history)
        return self.history

    def finish(self) -> None:
        """Finalise the monitoring session.

        Closes the W&B run if one is active.  Should be called at the end of
        training.
        """
        if self._wandb_run is not None:
            try:
                self._wandb_run.finish()
                LOGGER.info("W&B run finished.")
            except Exception as exc:
                LOGGER.warning("W&B finish failed: %s", exc)
            finally:
                self._wandb_run = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_epochs_logged(self) -> int:
        """Number of epochs recorded so far."""
        return len(self.history)

    def __repr__(self) -> str:
        wandb_status = "enabled" if self._wandb_run is not None else "disabled"
        return (
            f"TrainingMonitor(epochs_logged={self.num_epochs_logged}, "
            f"wandb={wandb_status}, output_dir='{self.output_dir}')"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    """JSON serialiser for numpy/torch scalars."""
    try:
        import numpy as np
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
    except ImportError:
        pass
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")
