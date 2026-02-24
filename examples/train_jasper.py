#!/usr/bin/env python3
"""Standalone JASPER training script — real dataset, no YAML required.

All hyperparameters are defined inline as Python dicts. Override any of them
via CLI flags. The only required argument is ``--dataset-dir``.

Usage
-----
Minimal::

    python examples/train_jasper.py \\
        --dataset-dir /path/to/jasper_dataset

Full overrides::

    python examples/train_jasper.py \\
        --dataset-dir /path/to/jasper_dataset \\
        --output-dir  runs/jasper_run01 \\
        --epochs      50 \\
        --batch-size  64 \\
        --lr          3e-4 \\
        --embedding-dim 768 \\
        --device      cuda:0

Resume from checkpoint::

    python examples/train_jasper.py \\
        --dataset-dir /path/to/jasper_dataset \\
        --resume      runs/jasper_run01/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from RAG_supporters.nn.models.jasper_predictor import JASPERPredictor, JASPERPredictorConfig
from RAG_supporters.nn.models.ema_encoder import EMAEncoder
from RAG_supporters.nn.losses.jasper_losses import JASPERMultiObjectiveLoss
from RAG_supporters.nn.training.jasper_trainer import JASPERTrainer, JASPERTrainerConfig
from RAG_supporters.nn.training.monitoring import TrainingMonitor
from RAG_supporters.pytorch_datasets.loader import create_loader, set_epoch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("train_jasper_phase1")


# ---------------------------------------------------------------------------
# Default hyperparameters (override via CLI flags)
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    # Model
    embedding_dim=768,
    hidden_dim=512,
    num_layers=3,
    dropout=0.1,
    activation="GELU",
    use_layer_norm=True,
    normalize_output=False,
    # EMA
    tau_min=0.996,
    tau_max=0.999,
    ema_schedule="cosine",
    # Loss
    lambda_jasper=1.0,
    lambda_contrastive=0.5,
    lambda_centroid=0.1,
    lambda_vicreg=0.1,
    jasper_beta=1.0,
    contrastive_temperature=0.07,
    centroid_temperature=0.1,
    vicreg_lambda_v=25.0,
    vicreg_lambda_i=25.0,
    vicreg_lambda_c=1.0,
    # Training
    epochs=50,
    batch_size=64,
    lr=3e-4,
    weight_decay=1e-4,
    warmup_epochs=2,
    num_workers=4,
    max_grad_norm=1.0,
    save_every_n_epochs=5,
    log_every_n_steps=50,
    mixed_precision=False,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def read_dataset_embedding_dim(dataset_dir: str) -> Optional[int]:
    """Read embedding_dim from the dataset's config.json if present."""
    cfg_path = Path(dataset_dir) / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            return json.load(f).get("embedding_dim")
    return None


def build_warmup_scheduler(
    optimizer: optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    base_lr: float,
) -> optim.lr_scheduler.SequentialLR:
    """Linear warmup followed by cosine annealing."""
    warmup = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-6 / max(base_lr, 1e-10),
        total_iters=max(warmup_epochs, 1),
    )
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(total_epochs - warmup_epochs, 1),
        eta_min=base_lr * 0.01,
    )
    return optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 1 JASPER training (JASPERPredictor, real dataset)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset-dir", required=True, help="Path to JASPER dataset directory")
    p.add_argument("--output-dir", default="runs/phase1", help="Output directory")
    p.add_argument("--resume", default=None, help="Checkpoint path to resume from")
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    p.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    p.add_argument("--warmup-epochs", type=int, default=DEFAULTS["warmup_epochs"])
    p.add_argument("--num-workers", type=int, default=DEFAULTS["num_workers"])
    p.add_argument(
        "--embedding-dim",
        type=int,
        default=None,
        help="Embedding dim (auto-read from dataset config.json if omitted)",
    )
    p.add_argument("--hidden-dim", type=int, default=DEFAULTS["hidden_dim"])
    p.add_argument("--num-layers", type=int, default=DEFAULTS["num_layers"])
    p.add_argument("--mixed-precision", action="store_true", default=DEFAULTS["mixed_precision"])
    p.add_argument("--device", default=None, help="Device override, e.g. 'cuda:0'")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"

    # ------------------------------------------------------------------
    # Resolve embedding_dim — prefer CLI arg, else dataset config.json
    # ------------------------------------------------------------------
    D = (
        args.embedding_dim
        or read_dataset_embedding_dim(args.dataset_dir)
        or DEFAULTS["embedding_dim"]
    )
    LOGGER.info("embedding_dim = %d", D)

    # ------------------------------------------------------------------
    # 1. Data loaders
    # ------------------------------------------------------------------
    LOGGER.info("Loading dataset from %s", args.dataset_dir)
    train_loader = create_loader(
        dataset_dir=args.dataset_dir,
        split="train",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = create_loader(
        dataset_dir=args.dataset_dir,
        split="val",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    LOGGER.info(
        "Train: %d samples | Val: %d samples",
        len(train_loader.dataset),
        len(val_loader.dataset),
    )

    # ------------------------------------------------------------------
    # 2. Model + EMA encoder
    # ------------------------------------------------------------------
    predictor_cfg = JASPERPredictorConfig(
        embedding_dim=D,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=DEFAULTS["dropout"],
        activation=DEFAULTS["activation"],
        use_layer_norm=DEFAULTS["use_layer_norm"],
        normalize_output=DEFAULTS["normalize_output"],
    )
    model = JASPERPredictor(predictor_cfg)
    LOGGER.info("%s", model.get_model_summary())

    # Shallow source-side projection (same width as predictor output)
    source_encoder = nn.Sequential(
        nn.Linear(D, D),
        nn.GELU(),
        nn.Linear(D, D),
    )
    ema_encoder = EMAEncoder(
        base_encoder=source_encoder,
        tau_min=DEFAULTS["tau_min"],
        tau_max=DEFAULTS["tau_max"],
        schedule=DEFAULTS["ema_schedule"],
    )

    # ------------------------------------------------------------------
    # 3. Loss
    # ------------------------------------------------------------------
    loss_fn = JASPERMultiObjectiveLoss(
        lambda_jasper=DEFAULTS["lambda_jasper"],
        lambda_contrastive=DEFAULTS["lambda_contrastive"],
        lambda_centroid=DEFAULTS["lambda_centroid"],
        lambda_vicreg=DEFAULTS["lambda_vicreg"],
        jasper_beta=DEFAULTS["jasper_beta"],
        contrastive_temperature=DEFAULTS["contrastive_temperature"],
        centroid_temperature=DEFAULTS["centroid_temperature"],
        vicreg_lambda_v=DEFAULTS["vicreg_lambda_v"],
        vicreg_lambda_i=DEFAULTS["vicreg_lambda_i"],
        vicreg_lambda_c=DEFAULTS["vicreg_lambda_c"],
    )

    # ------------------------------------------------------------------
    # 4. Optimiser + scheduler
    # ------------------------------------------------------------------
    optimizer = optim.AdamW(
        list(model.parameters()) + list(ema_encoder.online_encoder.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = build_warmup_scheduler(optimizer, args.warmup_epochs, args.epochs, args.lr)

    # ------------------------------------------------------------------
    # 5. Monitor
    # ------------------------------------------------------------------
    monitor = TrainingMonitor(
        output_dir=str(output_dir),
        use_wandb=False,
    )

    # ------------------------------------------------------------------
    # 6. Trainer
    # ------------------------------------------------------------------
    trainer_config = JASPERTrainerConfig(
        max_grad_norm=DEFAULTS["max_grad_norm"],
        log_every_n_steps=DEFAULTS["log_every_n_steps"],
        checkpoint_dir=str(checkpoint_dir),
        save_every_n_epochs=DEFAULTS["save_every_n_epochs"],
        keep_last_n_checkpoints=3,
        device=args.device,
        mixed_precision=args.mixed_precision,
    )
    trainer = JASPERTrainer(
        config=trainer_config,
        model=model,
        ema_encoder=ema_encoder,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        monitor=monitor,
    )

    # ------------------------------------------------------------------
    # 7. Resume
    # ------------------------------------------------------------------
    start_epoch = 0
    if args.resume:
        start_epoch, _ = trainer.load_checkpoint(args.resume)
        start_epoch += 1
        LOGGER.info("Resumed from epoch %d", start_epoch)
        remaining = args.epochs - start_epoch
        if remaining <= 0:
            LOGGER.warning("Already trained for %d epochs; nothing to do.", args.epochs)
            return
        num_epochs = remaining
    else:
        num_epochs = args.epochs

    # ------------------------------------------------------------------
    # 8. Train
    # ------------------------------------------------------------------
    LOGGER.info("Starting Phase 1 training for %d epochs → %s", num_epochs, output_dir)
    history = trainer.fit(num_epochs)

    # ------------------------------------------------------------------
    # 9. Finalise
    # ------------------------------------------------------------------
    final_ckpt = output_dir / "final.pt"
    trainer.save_checkpoint(final_ckpt, epoch=start_epoch + len(history) - 1)
    monitor.export_history(str(output_dir / "training_history.csv"))
    monitor.plot_losses(str(output_dir / "loss_curves.png"))
    monitor.finish()

    final_loss = history[-1].get("val/total", history[-1].get("train/total", float("nan")))
    LOGGER.info("Done. Final val loss: %.4f  |  checkpoint: %s", final_loss, final_ckpt)


if __name__ == "__main__":
    main()
