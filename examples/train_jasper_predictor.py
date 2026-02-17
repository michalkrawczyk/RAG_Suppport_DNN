#!/usr/bin/env python3
"""Train JASPER Predictor on JASPER Steering Dataset.

Usage
-----
Basic::

    python examples/train_jasper_predictor.py \\
        --config configs/jasper_base.yaml \\
        --dataset-dir /path/to/jasper_dataset \\
        --output-dir runs/jasper_run_01

Resume from checkpoint::

    python examples/train_jasper_predictor.py \\
        --config configs/jasper_base.yaml \\
        --dataset-dir /path/to/jasper_dataset \\
        --output-dir runs/jasper_run_01 \\
        --resume runs/jasper_run_01/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path

import torch
import torch.optim as optim
import yaml

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from RAG_supporters.nn.models.jasper_predictor import JASPERPredictor, JASPERPredictorConfig
from RAG_supporters.nn.models.ema_encoder import EMAEncoder
from RAG_supporters.nn.losses.jasper_losses import JASPERMultiObjectiveLoss
from RAG_supporters.nn.training.jasper_trainer import JASPERTrainer, JASPERTrainerConfig
from RAG_supporters.nn.training.monitoring import TrainingMonitor
from RAG_supporters.pytorch_datasets.loader import create_loader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("train_jasper")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_warmup_scheduler(
    optimizer: optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    base_lr: float,
) -> optim.lr_scheduler.SequentialLR:
    """Linear warmup followed by cosine annealing."""
    warmup = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-6 / max(base_lr, 1e-10), total_iters=max(warmup_epochs, 1)
    )
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_epochs - warmup_epochs, 1), eta_min=base_lr * 0.01
    )
    return optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train JASPER Predictor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--dataset-dir", required=True, help="Path to JASPER dataset directory")
    parser.add_argument("--output-dir", default="runs/jasper", help="Output directory for checkpoints and logs")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs from config")
    parser.add_argument("--device", default=None, help="Override device (e.g. 'cuda:0', 'cpu')")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # ------------------------------------------------------------------
    # 1. Load config
    # ------------------------------------------------------------------
    cfg = load_config(args.config)
    LOGGER.info("Config loaded from %s", args.config)

    model_cfg = cfg.get("model", {})
    ema_cfg = cfg.get("ema", {})
    loss_cfg = cfg.get("loss", {})
    train_cfg = cfg.get("training", {})
    dataset_cfg = cfg.get("dataset", {})
    monitor_cfg = cfg.get("monitoring", {})

    # CLI overrides
    if args.epochs is not None:
        train_cfg["num_epochs"] = args.epochs

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"

    num_epochs: int = train_cfg.get("num_epochs", 50)
    batch_size: int = train_cfg.get("batch_size", 64)
    lr: float = float(train_cfg.get("learning_rate", 3e-4))
    weight_decay: float = float(train_cfg.get("weight_decay", 1e-4))
    num_workers: int = train_cfg.get("num_workers", 4)
    warmup_epochs: int = train_cfg.get("warmup_epochs", 2)

    # ------------------------------------------------------------------
    # 2. Data loaders
    # ------------------------------------------------------------------
    LOGGER.info("Loading dataset from %s", args.dataset_dir)
    train_loader = create_loader(
        dataset_dir=args.dataset_dir,
        split=dataset_cfg.get("split_train", "train"),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=train_cfg.get("pin_memory", True),
    )
    val_loader = create_loader(
        dataset_dir=args.dataset_dir,
        split=dataset_cfg.get("split_val", "val"),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=train_cfg.get("pin_memory", True),
    )
    LOGGER.info("Train: %d samples | Val: %d samples", len(train_loader.dataset), len(val_loader.dataset))

    # ------------------------------------------------------------------
    # 3. Model + EMA encoder
    # ------------------------------------------------------------------
    predictor_cfg = JASPERPredictorConfig.from_dict(model_cfg)
    model = JASPERPredictor(predictor_cfg)
    LOGGER.info("%s", model.get_model_summary())

    # The EMA encoder wraps a shallow source-side projection (same width as predictor output)
    D = predictor_cfg.embedding_dim
    source_encoder = torch.nn.Sequential(
        torch.nn.Linear(D, D),
        torch.nn.GELU(),
        torch.nn.Linear(D, D),
    )
    ema_encoder = EMAEncoder(
        base_encoder=source_encoder,
        tau_min=float(ema_cfg.get("tau_min", 0.996)),
        tau_max=float(ema_cfg.get("tau_max", 0.999)),
        schedule=ema_cfg.get("schedule", "cosine"),
    )

    # ------------------------------------------------------------------
    # 4. Loss
    # ------------------------------------------------------------------
    loss_fn = JASPERMultiObjectiveLoss(**{
        k: float(v) if isinstance(v, (int, float)) else v
        for k, v in loss_cfg.items()
    })
    LOGGER.info("Loss: %s", loss_fn)

    # ------------------------------------------------------------------
    # 5. Optimiser + scheduler
    # ------------------------------------------------------------------
    optimizer = optim.AdamW(
        list(model.parameters()) + list(ema_encoder.online_encoder.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = build_warmup_scheduler(optimizer, warmup_epochs, num_epochs, lr)

    # ------------------------------------------------------------------
    # 6. Monitor
    # ------------------------------------------------------------------
    wandb_config = {**cfg} if monitor_cfg.get("use_wandb", False) else None
    monitor = TrainingMonitor(
        output_dir=str(output_dir),
        use_wandb=monitor_cfg.get("use_wandb", False),
        wandb_project=monitor_cfg.get("wandb_project", "jasper-rag"),
        wandb_name=monitor_cfg.get("wandb_name"),
        wandb_config=wandb_config,
    )

    # ------------------------------------------------------------------
    # 7. Trainer
    # ------------------------------------------------------------------
    trainer_config = JASPERTrainerConfig(
        max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
        log_every_n_steps=train_cfg.get("log_every_n_steps", 50),
        checkpoint_dir=str(checkpoint_dir),
        save_every_n_epochs=train_cfg.get("save_every_n_epochs", 5),
        keep_last_n_checkpoints=train_cfg.get("keep_last_n_checkpoints", 3),
        device=args.device,
        mixed_precision=train_cfg.get("mixed_precision", False),
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
    # 8. Resume
    # ------------------------------------------------------------------
    start_epoch = 0
    if args.resume:
        start_epoch, _ = trainer.load_checkpoint(args.resume)
        start_epoch += 1
        LOGGER.info("Resumed from epoch %d", start_epoch)
        remaining = num_epochs - start_epoch
        if remaining <= 0:
            LOGGER.warning("Already trained for %d epochs; nothing to do.", num_epochs)
            return
        num_epochs = remaining

    # ------------------------------------------------------------------
    # 9. Train
    # ------------------------------------------------------------------
    LOGGER.info("Starting training for %d epochs → output: %s", num_epochs, output_dir)
    history = trainer.fit(num_epochs)

    # ------------------------------------------------------------------
    # 10. Finalise
    # ------------------------------------------------------------------
    trainer.save_checkpoint(output_dir / "final.pt", epoch=start_epoch + len(history) - 1)
    monitor.export_history(str(output_dir / "training_history.csv"))
    monitor.plot_losses(str(output_dir / "loss_curves.png"))
    monitor.finish()

    LOGGER.info("Done. Final model saved to %s/final.pt", output_dir)


if __name__ == "__main__":
    main()
