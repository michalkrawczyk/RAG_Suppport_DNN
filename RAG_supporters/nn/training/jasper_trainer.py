"""JASPER training orchestrator with EMA updates, curriculum, and checkpointing."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from RAG_supporters.nn.models.ema_encoder import EMAEncoder
from RAG_supporters.nn.losses.jasper_losses import JASPERMultiObjectiveLoss


# Lazy import to avoid pulling in the heavy dataset/sklearn chain at module level.
# set_epoch is only needed during actual training, not at import time.
def _set_epoch(loader: DataLoader, epoch: int) -> None:  # noqa: E302
    from RAG_supporters.pytorch_datasets.loader import set_epoch as _se

    _se(loader, epoch)


LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class JASPERTrainerConfig:
    """Configuration for :class:`JASPERTrainer`.

    Args:
        max_grad_norm: Maximum gradient norm for clipping.  Set to 0 to disable.
        log_every_n_steps: Log metrics every N optimiser steps.
        checkpoint_dir: Directory to save checkpoints.  Created if it does not exist.
        save_every_n_epochs: Save a checkpoint every N epochs.  0 disables periodic saves.
        keep_last_n_checkpoints: How many epoch checkpoints to keep (oldest deleted).
            0 means keep all.
        device: Torch device string or ``None`` to auto-detect.
        mixed_precision: Whether to use ``torch.cuda.amp`` mixed precision training.
    """

    max_grad_norm: float = 1.0
    log_every_n_steps: int = 50
    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 1
    keep_last_n_checkpoints: int = 5
    device: Optional[str] = None
    mixed_precision: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "JASPERTrainerConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# JASPERTrainer
# ---------------------------------------------------------------------------


class JASPERTrainer:
    """Training orchestrator for the JASPER predictor model.

    Responsibilities
    ----------------
    - EMA target encoder updates (once per optimiser step)
    - Multi-objective loss computation
    - Curriculum learning via :func:`set_epoch` on the dataset
    - Mixed-precision training (optional)
    - Gradient clipping
    - Checkpoint save / load (model + EMA + optimiser + scheduler + epoch)
    - Validation loop (no EMA updates, no gradient accumulation)
    - Structured logging of all loss components

    Args:
        config: Trainer configuration.
        model: The :class:`~RAG_supporters.nn.models.jasper_predictor.JASPERPredictor`
            (or any ``nn.Module`` with ``forward(question_emb, steering_emb)``).
        ema_encoder: :class:`~RAG_supporters.nn.models.ema_encoder.EMAEncoder`
            wrapping a source-side encoder.
        loss_fn: :class:`~RAG_supporters.nn.losses.jasper_losses.JASPERMultiObjectiveLoss`.
        optimizer: PyTorch optimiser bound to ``model.parameters()``.
        train_loader: Training :class:`~torch.utils.data.DataLoader`.
        val_loader: Validation :class:`~torch.utils.data.DataLoader`.
        scheduler: Optional LR scheduler.  ``step()`` is called once per epoch.
        monitor: Optional :class:`~RAG_supporters.nn.training.monitoring.TrainingMonitor`.
    """

    def __init__(
        self,
        config: JASPERTrainerConfig | dict,
        model: nn.Module,
        ema_encoder: EMAEncoder,
        loss_fn: JASPERMultiObjectiveLoss,
        optimizer: Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        scheduler: Optional[_LRScheduler] = None,
        monitor: Optional[Any] = None,  # TrainingMonitor (avoids circular import)
    ) -> None:
        if isinstance(config, dict):
            config = JASPERTrainerConfig.from_dict(config)
        self.config = config

        # Resolve device
        if config.device:
            self.device = torch.device(config.device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.ema_encoder = ema_encoder.to(self.device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.monitor = monitor

        # Checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Mixed precision scaler
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None
        if config.mixed_precision and self.device.type == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()

        # State
        self._global_step = 0
        self._best_val_loss = float("inf")
        self._checkpoint_paths: List[Path] = []

        # Cache centroid embeddings from dataset (moved to device once)
        self._centroid_embs: Optional[torch.Tensor] = self._load_centroids()

        # Total steps needed for EMA tau schedule
        self._max_steps: int = 0  # set in fit()

        # Warn when running on CUDA with a single-threaded DataLoader —
        # num_workers=0 will saturate the CPU→GPU transfer pipeline.
        if self.device.type == "cuda" and getattr(self.train_loader, "num_workers", 0) == 0:
            LOGGER.warning(
                "DataLoader num_workers=0 on CUDA device — consider num_workers >= 2 "
                "to avoid CPU-bound data loading bottlenecks."
            )

        LOGGER.info(
            "JASPERTrainer ready | device=%s | mixed_precision=%s",
            self.device,
            config.mixed_precision,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, num_epochs: int) -> List[Dict[str, float]]:
        """Train for ``num_epochs`` epochs and return training history.

        Args:
            num_epochs: Number of epochs to train.

        Returns:
            List of per-epoch metric dicts (train + val metrics).
        """
        self._max_steps = num_epochs * len(self.train_loader)
        history: List[Dict[str, float]] = []

        LOGGER.info(
            "Starting training: %d epochs, %d steps/epoch", num_epochs, len(self.train_loader)
        )

        for epoch in range(num_epochs):
            t0 = time.time()

            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()

            epoch_metrics = {
                "epoch": epoch,
                **{f"train/{k}": v for k, v in train_metrics.items()},
                **{f"val/{k}": v for k, v in val_metrics.items()},
                "epoch_time_s": time.time() - t0,
            }

            if self.scheduler is not None:
                self.scheduler.step()
                epoch_metrics["lr"] = self.scheduler.get_last_lr()[0]

            if self.monitor is not None:
                self.monitor.log_metrics(epoch, epoch_metrics)

            history.append(epoch_metrics)
            self._log_epoch(epoch, epoch_metrics)

            # Save periodic checkpoint
            if (
                self.config.save_every_n_epochs > 0
                and (epoch + 1) % self.config.save_every_n_epochs == 0
            ):
                self.save_checkpoint(
                    self.checkpoint_dir / f"epoch_{epoch:04d}.pt",
                    epoch=epoch,
                    metrics=epoch_metrics,
                )

            # Save best
            if val_metrics.get("total", float("inf")) < self._best_val_loss:
                self._best_val_loss = val_metrics.get("total", float("inf"))
                self.save_checkpoint(
                    self.checkpoint_dir / "best.pt",
                    epoch=epoch,
                    metrics=epoch_metrics,
                )

        LOGGER.info("Training complete. Best val loss: %.4f", self._best_val_loss)
        return history

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch.

        Args:
            epoch: Current epoch index (0-based).

        Returns:
            Dict of averaged loss metrics over the epoch.
        """
        self.model.train()
        self.ema_encoder.train()

        # Update curriculum weights
        _set_epoch(self.train_loader, epoch)

        running: Dict[str, float] = {}
        n_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            step_metrics = self._train_step(batch)
            self._global_step += 1

            for k, v in step_metrics.items():
                running[k] = running.get(k, 0.0) + v
            n_batches += 1

            if self._global_step % self.config.log_every_n_steps == 0:
                LOGGER.debug(
                    "epoch=%d step=%d %s",
                    epoch,
                    self._global_step,
                    "  ".join(f"{k}={v:.4f}" for k, v in step_metrics.items() if "acc" not in k),
                )

        return {k: v / max(n_batches, 1) for k, v in running.items()}

    def validate(self) -> Dict[str, float]:
        """Run one validation epoch (no EMA updates, no grad).

        Returns:
            Dict of averaged loss metrics over the validation set.
        """
        self.model.eval()
        self.ema_encoder.eval()

        running: Dict[str, float] = {}
        n_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                step_metrics = self._eval_step(batch)
                for k, v in step_metrics.items():
                    running[k] = running.get(k, 0.0) + v
                n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in running.items()}

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        path: str | Path,
        epoch: int,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save full training state to a checkpoint file.

        Saved state includes: model weights, EMA encoder state (both online and
        target + tau config), optimiser state, scheduler state, global step,
        epoch, and optional metrics.

        Args:
            path: File path for the checkpoint (``.pt`` / ``.pth``).
            epoch: Current epoch index.
            metrics: Optional metrics dict to store alongside the checkpoint.
        """
        path = Path(path)
        checkpoint = {
            "epoch": epoch,
            "global_step": self._global_step,
            "model_state_dict": self.model.state_dict(),
            "ema_state_dict": self.ema_encoder.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self._best_val_loss,
            "metrics": metrics or {},
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, path)
        LOGGER.info("Checkpoint saved: %s (epoch=%d)", path, epoch)

        # Track periodic checkpoints for rotation
        if path.name != "best.pt":
            self._checkpoint_paths.append(path)
            self._rotate_checkpoints()

    def load_checkpoint(self, path: str | Path) -> Tuple[int, Dict[str, Any]]:
        """Load training state from a checkpoint file.

        Args:
            path: Path to a checkpoint saved by :meth:`save_checkpoint`.

        Returns:
            Tuple of ``(epoch, metrics)`` restored from the checkpoint.
        """
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.ema_encoder.load_state_dict(checkpoint["ema_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self._global_step = checkpoint.get("global_step", 0)
        self._best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        epoch = checkpoint.get("epoch", 0)
        metrics = checkpoint.get("metrics", {})

        LOGGER.info("Checkpoint loaded: %s (epoch=%d)", path, epoch)
        return epoch, metrics

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Run a single forward + backward step and return loss metrics."""
        batch = self._to_device(batch)

        question_emb = batch["question_emb"]
        target_source_emb = batch["target_source_emb"]
        steering_emb = batch["steering"]
        negatives = batch["negative_embs"]
        cluster_ids = batch["cluster_id"]

        centroid_embs = self._get_centroid_embs(target_source_emb)

        self.optimizer.zero_grad()

        if self.scaler is not None:
            with torch.cuda.amp.autocast():
                predicted = self.model(question_emb, steering_emb)
                ema_target = self.ema_encoder.encode_target(target_source_emb)
                total_loss, loss_dict = self.loss_fn(
                    predicted, ema_target, negatives, centroid_embs, cluster_ids
                )
            self.scaler.scale(total_loss).backward()
            if self.config.max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            predicted = self.model(question_emb, steering_emb)
            ema_target = self.ema_encoder.encode_target(target_source_emb)
            total_loss, loss_dict = self.loss_fn(
                predicted, ema_target, negatives, centroid_embs, cluster_ids
            )
            total_loss.backward()
            if self.config.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

        # EMA update after each optimiser step
        self.ema_encoder.update_target(self._global_step, self._max_steps)

        return {k: v.item() for k, v in loss_dict.items()}

    def _eval_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Run a single validation step and return loss metrics."""
        batch = self._to_device(batch)

        question_emb = batch["question_emb"]
        target_source_emb = batch["target_source_emb"]
        steering_emb = batch["steering"]
        negatives = batch["negative_embs"]
        cluster_ids = batch["cluster_id"]

        centroid_embs = self._get_centroid_embs(target_source_emb)

        predicted = self.model(question_emb, steering_emb)
        ema_target = self.ema_encoder.encode_target(target_source_emb)
        _, loss_dict = self.loss_fn(predicted, ema_target, negatives, centroid_embs, cluster_ids)

        return {k: v.item() for k, v in loss_dict.items()}

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
        }

    def _get_centroid_embs(self, source_emb: torch.Tensor) -> torch.Tensor:
        """Return cached centroid embeddings (or fall back to source mean)."""
        if self._centroid_embs is not None:
            return self._centroid_embs.to(self.device)
        # Fallback: treat the source emb batch mean as a single centroid
        LOGGER.warning("No centroid embeddings found; using batch mean as fallback centroid.")
        return source_emb.mean(dim=0, keepdim=True)

    def _load_centroids(self) -> Optional[torch.Tensor]:
        """Try to extract centroid embeddings from the training dataset."""
        dataset = getattr(self.train_loader, "dataset_obj", None) or self.train_loader.dataset
        centroid_embs = getattr(dataset, "centroid_embs", None)
        if centroid_embs is None:
            LOGGER.warning(
                "Dataset does not expose 'centroid_embs'; CentroidLoss will use a fallback. "
                "Make sure to pass centroid embeddings explicitly if needed."
            )
            return None
        if isinstance(centroid_embs, torch.Tensor):
            return centroid_embs.to(self.device)
        # numpy or list
        return torch.as_tensor(centroid_embs, dtype=torch.float32).to(self.device)

    def _rotate_checkpoints(self) -> None:
        """Delete oldest checkpoint files if over the keep limit."""
        limit = self.config.keep_last_n_checkpoints
        if limit <= 0:
            return
        while len(self._checkpoint_paths) > limit:
            old_path = self._checkpoint_paths.pop(0)
            if old_path.exists():
                old_path.unlink()
                LOGGER.debug("Removed old checkpoint: %s", old_path)

    def _log_epoch(self, epoch: int, metrics: Dict[str, Any]) -> None:
        train_loss = metrics.get("train/total", float("nan"))
        val_loss = metrics.get("val/total", float("nan"))
        t = metrics.get("epoch_time_s", 0.0)
        LOGGER.info(
            "Epoch %d | train_loss=%.4f  val_loss=%.4f  time=%.1fs",
            epoch,
            train_loss,
            val_loss,
            t,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def global_step(self) -> int:
        """Total number of optimiser steps taken so far."""
        return self._global_step

    @property
    def best_val_loss(self) -> float:
        """Best validation loss seen during training."""
        return self._best_val_loss
