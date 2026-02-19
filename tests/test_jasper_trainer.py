"""Tests for JASPERTrainer."""

from __future__ import annotations

import copy
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


# ---------------------------------------------------------------------------
# Mock dataset / loader helpers
# ---------------------------------------------------------------------------


B, D, K, C = 8, 32, 4, 6   # batch size, embedding dim, negatives, clusters


def _make_batch(batch_size: int = B) -> Dict[str, torch.Tensor]:
    torch.manual_seed(7)
    return {
        "question_emb": torch.randn(batch_size, D),
        "target_source_emb": torch.randn(batch_size, D),
        "steering": torch.randn(batch_size, D),
        "negative_embs": torch.randn(batch_size, K, D),
        "cluster_id": torch.randint(0, C, (batch_size,)),
        "relevance": torch.rand(batch_size),
        "centroid_distance": torch.rand(batch_size),
        "steering_variant": torch.randint(0, 4, (batch_size,)),
        "negative_tiers": torch.randint(0, 4, (batch_size, K)),
    }


class _MockDataset(torch.utils.data.Dataset):
    """Tiny in-memory dataset that mimics the JASPER batch structure."""

    def __init__(self, n_samples: int = 32):
        self.n = n_samples
        self.centroid_embs = torch.randn(C, D)   # expose centroids like real dataset
        torch.manual_seed(42)
        self._data = [_make_batch(batch_size=1) for _ in range(n_samples)]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {k: v.squeeze(0) for k, v in self._data[idx % len(self._data)].items()}

    def set_epoch(self, epoch: int) -> None:
        pass  # no-op for mock


def _make_loader(n_samples: int = 32, batch_size: int = B) -> DataLoader:
    dataset = _MockDataset(n_samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    loader.dataset_obj = dataset  # mimic create_loader convention
    return loader


def _make_source_encoder(dim: int = D) -> nn.Module:
    return nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))


@pytest.fixture
def trainer_components(tmp_path):
    torch.manual_seed(0)
    model = JASPERPredictor(JASPERPredictorConfig(embedding_dim=D, hidden_dim=D, num_layers=1))
    ema = EMAEncoder(_make_source_encoder(D), tau_min=0.9, tau_max=0.99)
    loss_fn = JASPERMultiObjectiveLoss(
        lambda_jasper=1.0, lambda_contrastive=0.5, lambda_centroid=0.1, lambda_vicreg=0.1
    )
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(ema.online_encoder.parameters()), lr=1e-3
    )
    train_loader = _make_loader()
    val_loader = _make_loader()
    config = JASPERTrainerConfig(
        max_grad_norm=1.0,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        save_every_n_epochs=0,   # disable auto-save during tests
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


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestInit:
    def test_device_is_cpu(self, trainer_components):
        trainer, *_ = trainer_components
        assert trainer.device.type == "cpu", \
            "Trainer device should be 'cpu' when config.device='cpu'"

    def test_checkpoint_dir_created(self, trainer_components, tmp_path):
        trainer, *_ = trainer_components
        assert Path(trainer.config.checkpoint_dir).exists(), \
            "Trainer __init__ should create the checkpoint directory"

    def test_centroids_loaded_from_dataset(self, trainer_components):
        trainer, *_ = trainer_components
        assert trainer._centroid_embs is not None, \
            "Trainer should load centroid embeddings from the dataset"
        assert trainer._centroid_embs.shape == (C, D), \
            f"Centroid embeddings should have shape ({C}, {D})"

    def test_from_dict_config(self, tmp_path):
        model = JASPERPredictor({"embedding_dim": D, "hidden_dim": D, "num_layers": 1})
        ema = EMAEncoder(_make_source_encoder(D))
        trainer = JASPERTrainer(
            config={"checkpoint_dir": str(tmp_path), "device": "cpu"},
            model=model,
            ema_encoder=ema,
            loss_fn=JASPERMultiObjectiveLoss(),
            optimizer=torch.optim.AdamW(model.parameters()),
            train_loader=_make_loader(),
            val_loader=_make_loader(),
        )
        assert trainer.device.type == "cpu", \
            "Trainer should resolve to CPU when config.device='cpu'"


# ---------------------------------------------------------------------------
# Single training step
# ---------------------------------------------------------------------------


class TestTrainStep:
    def test_single_step_returns_metrics(self, trainer_components):
        trainer, *_ = trainer_components
        trainer._max_steps = 100
        batch = _make_batch()
        metrics = trainer._train_step(batch)
        assert "total" in metrics, "_train_step should return a dict with 'total' key"
        assert isinstance(metrics["total"], float), \
            "metrics['total'] should be a Python float"

    def test_parameters_updated_after_step(self, trainer_components):
        trainer, model, _ = trainer_components
        trainer._max_steps = 100
        params_before = {n: p.data.clone() for n, p in model.named_parameters()}
        batch = _make_batch()
        trainer._train_step(batch)
        changed = any(
            not torch.equal(p.data, params_before[n])
            for n, p in model.named_parameters()
        )
        assert changed, "No model parameters changed after training step"

    def test_ema_target_updated_after_step(self, trainer_components):
        trainer, _, ema = trainer_components
        trainer._max_steps = 100
        target_before = {n: p.data.clone() for n, p in ema.target_encoder.named_parameters()}
        batch = _make_batch()
        trainer._train_step(batch)
        changed = any(
            not torch.equal(p.data, target_before[n])
            for n, p in ema.target_encoder.named_parameters()
        )
        assert changed, "EMA target not updated after step"

    def test_loss_is_finite(self, trainer_components):
        trainer, *_ = trainer_components
        trainer._max_steps = 100
        batch = _make_batch()
        metrics = trainer._train_step(batch)
        assert not any(
            v != v or abs(v) == float("inf")
            for v in metrics.values()
            if isinstance(v, float)
        )


# ---------------------------------------------------------------------------
# Train / validate epoch
# ---------------------------------------------------------------------------


class TestEpoch:
    def test_train_epoch_returns_dict(self, trainer_components):
        trainer, *_ = trainer_components
        trainer._max_steps = len(trainer.train_loader)
        metrics = trainer.train_epoch(epoch=0)
        assert isinstance(metrics, dict), \
            "train_epoch should return a dict of metrics"
        assert "total" in metrics, \
            "train_epoch metrics dict must contain 'total' key"

    def test_validate_returns_dict(self, trainer_components):
        trainer, *_ = trainer_components
        metrics = trainer.validate()
        assert isinstance(metrics, dict), \
            "validate() should return a dict of metrics"
        assert "total" in metrics, \
            "validate() metrics dict must contain 'total' key"

    def test_global_step_increments(self, trainer_components):
        trainer, *_ = trainer_components
        trainer._max_steps = len(trainer.train_loader)
        before = trainer.global_step
        trainer.train_epoch(epoch=0)
        assert trainer.global_step == before + len(trainer.train_loader), \
            f"global_step should increase by {len(trainer.train_loader)} after one epoch"

    def test_model_in_train_mode_during_epoch(self, trainer_components):
        trainer, model, _ = trainer_components
        trainer._max_steps = 10
        # After calling train_epoch the model should have been in train mode
        original_train = model.training
        trainer.train_epoch(epoch=0)
        # validate switches to eval; but after train_epoch model should be in training mode
        # (it gets set at start of train_epoch)
        assert model.training, \
            "Model should still be in training mode after train_epoch completes"

    def test_validate_model_in_eval_mode(self, trainer_components):
        trainer, model, _ = trainer_components
        model.train()
        trainer.validate()
        assert not model.training, \
            "Model should be in eval mode after validate() completes"


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------


class TestCheckpoint:
    def test_save_creates_file(self, trainer_components, tmp_path):
        trainer, *_ = trainer_components
        ckpt_path = tmp_path / "test.pt"
        trainer.save_checkpoint(ckpt_path, epoch=3)
        assert ckpt_path.exists(), \
            "save_checkpoint should create the checkpoint file on disk"

    def test_checkpoint_contains_required_keys(self, trainer_components, tmp_path):
        trainer, *_ = trainer_components
        ckpt_path = tmp_path / "test.pt"
        trainer.save_checkpoint(ckpt_path, epoch=3)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        for key in ("epoch", "global_step", "model_state_dict", "ema_state_dict", "optimizer_state_dict"):
            assert key in ckpt, f"Missing key: {key}"

    def test_load_restores_epoch(self, trainer_components, tmp_path):
        trainer, *_ = trainer_components
        ckpt_path = tmp_path / "epoch_5.pt"
        trainer._global_step = 50
        trainer.save_checkpoint(ckpt_path, epoch=5, metrics={"val/total": 0.42})
        epoch, metrics = trainer.load_checkpoint(ckpt_path)
        assert epoch == 5, "load_checkpoint should restore the saved epoch number"
        assert abs(metrics.get("val/total", 0) - 0.42) < 1e-6, \
            "load_checkpoint should restore the saved metrics"

    def test_load_restores_model_weights(self, trainer_components, tmp_path):
        trainer, model, _ = trainer_components
        ckpt_path = tmp_path / "weights.pt"
        # Save
        original_weights = {n: p.data.clone() for n, p in model.named_parameters()}
        trainer.save_checkpoint(ckpt_path, epoch=0)
        # Corrupt weights
        with torch.no_grad():
            for p in model.parameters():
                p.data.fill_(999.0)
        # Restore
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
        assert trainer.global_step == 123, \
            "load_checkpoint should restore global_step to the saved value"

    def test_checkpoint_rotation(self, trainer_components, tmp_path):
        trainer, *_ = trainer_components
        trainer.config.keep_last_n_checkpoints = 2
        paths = []
        for i in range(4):
            p = tmp_path / f"epoch_{i:04d}.pt"
            trainer.save_checkpoint(p, epoch=i)
            paths.append(p)
        # After 4 saves with keep=2, only the last 2 should remain
        existing = [p for p in paths if p.exists()]
        assert len(existing) == 2, \
            f"With keep_last_n_checkpoints=2, only 2 files should remain; found {len(existing)}"
        assert paths[-1] in existing, "The most recent checkpoint should be kept"
        assert paths[-2] in existing, "The second most recent checkpoint should be kept"


# ---------------------------------------------------------------------------
# Curriculum (set_epoch integration)
# ---------------------------------------------------------------------------


class TestCurriculum:
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
