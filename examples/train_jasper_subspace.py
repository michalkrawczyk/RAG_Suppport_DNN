#!/usr/bin/env python3
"""Train Subspace-Routed JASPER Predictor (DecomposedJASPERPredictor).

Usage
-----
Basic::

    python examples/train_jasper_subspace.py \
        --config configs/subspace_jasper.yaml \
        --dataset-dir /path/to/jasper_dataset \
        --output-dir runs/subspace_run_01 \
        --centroids-path /path/to/centroids.pt

With dataset build step (Tasks 1-8 run automatically before training)::

    python examples/train_jasper_subspace.py \
        --config configs/subspace_jasper.yaml \
        --dataset-dir /path/to/jasper_dataset \
        --centroids-path /path/to/centroids.pt \
        --build-csv-paths data/train.csv data/val.csv \
        --build-cluster-json data/clusters.json

    Override the embedding model (uses LangChain OpenAI text-embedding-3-small by default)::

        python examples/train_jasper_subspace.py \
            --config configs/subspace_jasper.yaml \
            --dataset-dir /path/to/jasper_dataset \
            --centroids-path /path/to/centroids.pt \
            --build-csv-paths data/train.csv data/val.csv \
            --build-cluster-json data/clusters.json \
            --build-embedding-model sentence-transformers/all-MiniLM-L6-v2

Resume from checkpoint::

    python examples/train_jasper_subspace.py \
        --config configs/subspace_jasper.yaml \
        --dataset-dir /path/to/jasper_dataset \
        --output-dir runs/subspace_run_01 \
        --centroids-path /path/to/centroids.pt \
        --resume runs/subspace_run_01/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import yaml

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from RAG_supporters.jasper import build_dataset
from RAG_supporters.nn.models.decomposed_predictor import (
    DecomposedJASPERPredictor,
    DecomposedJASPERConfig,
)
from RAG_supporters.nn.models.ema_encoder import EMAEncoder
from RAG_supporters.nn.losses.jasper_losses import JASPERMultiObjectiveLoss
from RAG_supporters.nn.losses.routing_losses import (
    RoutingLoss,
    EntropyRegularization,
    ResidualPenalty,
    DisentanglementLoss,
)
from RAG_supporters.nn.training.jasper_trainer import JASPERTrainer, JASPERTrainerConfig
from RAG_supporters.nn.training.monitoring import TrainingMonitor
from RAG_supporters.nn.inference.xai_interface import XAIInterface
from RAG_supporters.pytorch_datasets.loader import create_loader, set_epoch  # noqa: F401
from RAG_supporters.pytorch_datasets.jasper_steering_dataset import JASPERSteeringDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger("train_subspace_jasper")


# ---------------------------------------------------------------------------
# SubspaceJASPERTrainer
# ---------------------------------------------------------------------------


class SubspaceJASPERTrainer(JASPERTrainer):
    """JASPERTrainer subclass for DecomposedJASPERPredictor.

    Overrides :meth:`_train_step` and :meth:`_eval_step` to:

    1. Unpack ``(prediction, explanation_dict)`` from the model forward.
    2. Add routing-specific losses on top of the base multi-objective loss.
    3. Log routing accuracy, routing entropy, residual norm, and disentanglement.

    Args:
        routing_loss_fn: :class:`RoutingLoss` instance.
        entropy_reg_fn: :class:`EntropyRegularization` instance.
        residual_penalty_fn: :class:`ResidualPenalty` instance.
        disentanglement_fn: :class:`DisentanglementLoss` instance.
        lambda_routing: Scalar weight for :class:`RoutingLoss`.
        lambda_entropy: Scalar weight for :class:`EntropyRegularization`.
        lambda_residual: Scalar weight for :class:`ResidualPenalty`.
        lambda_disentangle: Scalar weight for :class:`DisentanglementLoss`.
        xai_interface: Optional :class:`XAIInterface` for validation-set XAI export.
        xai_every_n_epochs: How often to run val-set XAI output.
        xai_output_dir: Directory for XAI JSON files.
        reload_negatives_every_n_epochs: Reload hard negatives from disk every
            N *absolute* epochs (0 = disabled).
        All other args forwarded to :class:`JASPERTrainer`.
    """

    def __init__(
        self,
        routing_loss_fn: RoutingLoss,
        entropy_reg_fn: EntropyRegularization,
        residual_penalty_fn: ResidualPenalty,
        disentanglement_fn: DisentanglementLoss,
        lambda_routing: float = 1.0,
        lambda_entropy: float = 0.1,
        lambda_residual: float = 0.1,
        lambda_disentangle: float = 0.01,
        xai_interface: Optional[XAIInterface] = None,
        xai_every_n_epochs: int = 5,
        xai_output_dir: Optional[str] = None,
        reload_negatives_every_n_epochs: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize SubspaceJASPERTrainer."""
        super().__init__(**kwargs)

        self.routing_loss_fn = routing_loss_fn
        self.entropy_reg_fn = entropy_reg_fn
        self.residual_penalty_fn = residual_penalty_fn
        self.disentanglement_fn = disentanglement_fn

        self.lambda_routing = lambda_routing
        self.lambda_entropy = lambda_entropy
        self.lambda_residual = lambda_residual
        self.lambda_disentangle = lambda_disentangle

        self.xai_interface = xai_interface
        self.xai_every_n_epochs = xai_every_n_epochs
        self.xai_output_dir = Path(xai_output_dir) if xai_output_dir else None

        self._current_epoch: int = 0
        self.reload_negatives_every_n_epochs = reload_negatives_every_n_epochs

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Store current epoch before calling the base train_epoch."""
        self._current_epoch = epoch
        return super().train_epoch(epoch)

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single forward+backward step for DecomposedJASPERPredictor."""
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
                prediction, xai = self.model(
                    question_emb, steering_emb, centroid_embs, training=True
                )
                ema_target = self.ema_encoder.encode_target(target_source_emb)
                base_loss, loss_dict = self.loss_fn(
                    prediction, ema_target, negatives, centroid_embs, cluster_ids
                )
                routing_dict = self.routing_loss_fn(xai["concept_logits"], cluster_ids)
                entropy_dict = self.entropy_reg_fn(xai["routing_weights"], self._current_epoch)
                residual_dict = self.residual_penalty_fn(xai["fine"])
                disentangle_dict = self.disentanglement_fn(xai["routing_weights"])

                total_loss = (
                    base_loss
                    + self.lambda_routing * routing_dict["routing"]
                    + self.lambda_entropy * entropy_dict["entropy_reg"]
                    + self.lambda_residual * residual_dict["residual_penalty"]
                    + self.lambda_disentangle * disentangle_dict["disentanglement"]
                )

            self.scaler.scale(total_loss).backward()
            if self.config.max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            prediction, xai = self.model(question_emb, steering_emb, centroid_embs, training=True)
            ema_target = self.ema_encoder.encode_target(target_source_emb)
            base_loss, loss_dict = self.loss_fn(
                prediction, ema_target, negatives, centroid_embs, cluster_ids
            )
            routing_dict = self.routing_loss_fn(xai["concept_logits"], cluster_ids)
            entropy_dict = self.entropy_reg_fn(xai["routing_weights"], self._current_epoch)
            residual_dict = self.residual_penalty_fn(xai["fine"])
            disentangle_dict = self.disentanglement_fn(xai["routing_weights"])

            total_loss = (
                base_loss
                + self.lambda_routing * routing_dict["routing"]
                + self.lambda_entropy * entropy_dict["entropy_reg"]
                + self.lambda_residual * residual_dict["residual_penalty"]
                + self.lambda_disentangle * disentangle_dict["disentanglement"]
            )

            total_loss.backward()
            if self.config.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

        # EMA update after each optimiser step
        self.ema_encoder.update_target(self._global_step, self._max_steps)

        # Merge all metric dicts for logging
        all_metrics: Dict[str, float] = {k: v.item() for k, v in loss_dict.items()}
        all_metrics["total"] = total_loss.item()
        all_metrics.update({k: v.item() for k, v in routing_dict.items()})
        all_metrics.update({k: v.item() for k, v in entropy_dict.items()})
        all_metrics.update({k: v.item() for k, v in residual_dict.items()})
        all_metrics.update({k: v.item() for k, v in disentangle_dict.items()})

        return all_metrics

    def _eval_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single validation step for DecomposedJASPERPredictor."""
        batch = self._to_device(batch)

        question_emb = batch["question_emb"]
        target_source_emb = batch["target_source_emb"]
        steering_emb = batch["steering"]
        negatives = batch["negative_embs"]
        cluster_ids = batch["cluster_id"]

        centroid_embs = self._get_centroid_embs(target_source_emb)

        prediction, xai = self.model(question_emb, steering_emb, centroid_embs, training=False)
        ema_target = self.ema_encoder.encode_target(target_source_emb)
        _, loss_dict = self.loss_fn(prediction, ema_target, negatives, centroid_embs, cluster_ids)
        routing_dict = self.routing_loss_fn(xai["concept_logits"], cluster_ids)
        entropy_dict = self.entropy_reg_fn(xai["routing_weights"], self._current_epoch)
        residual_dict = self.residual_penalty_fn(xai["fine"])
        disentangle_dict = self.disentanglement_fn(xai["routing_weights"])

        all_metrics: Dict[str, float] = {k: v.item() for k, v in loss_dict.items()}
        all_metrics.update({k: v.item() for k, v in routing_dict.items()})
        all_metrics.update({k: v.item() for k, v in entropy_dict.items()})
        all_metrics.update({k: v.item() for k, v in residual_dict.items()})
        all_metrics.update({k: v.item() for k, v in disentangle_dict.items()})

        return all_metrics

    def save_val_xai(self, epoch: int) -> Optional[str]:
        """Run XAI over validation set and save JSON.

        Returns:
            Path to the saved JSON file, or ``None`` if XAI is not configured.
        """
        if self.xai_interface is None:
            return None

        self.model.eval()
        xai_results = []

        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._to_device(batch)
                q = batch["question_emb"]
                s = batch["steering"]
                for i in range(q.shape[0]):
                    try:
                        result = self.xai_interface.explain_prediction(q[i], s[i])
                        xai_results.append(result)
                    except Exception as exc:
                        LOGGER.warning("XAI explain failed for sample %d: %s", i, exc)

        out_dir = self.xai_output_dir or (Path(self.config.checkpoint_dir).parent / "xai")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"xai_epoch_{epoch:04d}.json"
        self.xai_interface.save_xai_outputs(xai_results, str(out_path))
        return str(out_path)

    def fit(self, num_epochs: int, start_epoch: int = 0) -> List[Dict[str, float]]:
        """Extend training loop with periodic XAI export after select epochs.

        Args:
            num_epochs: Number of epochs to train.
            start_epoch: Absolute epoch offset for curriculum and negative-reload
                scheduling.  Pass ``start_epoch`` from the resume block so the
                steering curriculum continues from the correct position.
        """
        self._max_steps = num_epochs * len(self.train_loader)
        import time

        history: List[Dict[str, float]] = []

        LOGGER.info("Starting subspace training: %d epochs (offset=%d)", num_epochs, start_epoch)

        for epoch in range(start_epoch, start_epoch + num_epochs):
            t0 = time.time()
            # Reload hard negatives periodically (skip the very first epoch)
            if (
                self.reload_negatives_every_n_epochs > 0
                and epoch > start_epoch
                and epoch % self.reload_negatives_every_n_epochs == 0
            ):
                dataset: JASPERSteeringDataset = self.train_loader.dataset_obj
                LOGGER.info("Epoch %d: reloading hard negatives from disk.", epoch)
                dataset.reload_negatives()
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

            # Periodic checkpoint
            if (
                self.config.save_every_n_epochs > 0
                and (epoch + 1) % self.config.save_every_n_epochs == 0
            ):
                self.save_checkpoint(
                    self.checkpoint_dir / f"epoch_{epoch:04d}.pt",
                    epoch=epoch,
                    metrics=epoch_metrics,
                )

            # Best checkpoint
            if val_metrics.get("total", float("inf")) < self._best_val_loss:
                self._best_val_loss = val_metrics.get("total", float("inf"))
                self.save_checkpoint(
                    self.checkpoint_dir / "best.pt",
                    epoch=epoch,
                    metrics=epoch_metrics,
                )

            # XAI export
            if (
                self.xai_interface is not None
                and self.xai_every_n_epochs > 0
                and (epoch + 1) % self.xai_every_n_epochs == 0
            ):
                xai_path = self.save_val_xai(epoch)
                if xai_path:
                    LOGGER.info("XAI saved: %s", xai_path)

        LOGGER.info("Subspace training complete. Best val loss: %.4f", self._best_val_loss)
        return history


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_config(config_path: str | Path) -> dict:
    """Load a YAML config file and return its contents as a dict."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_centroids(centroids_path: str) -> torch.Tensor:
    """Load centroid embeddings from a .pt or .npy file."""
    path = Path(centroids_path)
    if path.suffix == ".npy":
        import numpy as np

        arr = np.load(str(path))
        return torch.from_numpy(arr).float()
    return torch.load(str(path), map_location="cpu")


def load_cluster_names(names_file: Optional[str], K: int) -> List[str]:
    """Load cluster names from a JSON file or generate defaults."""
    if names_file is None:
        return [f"subspace_{i}" for i in range(K)]
    path = Path(names_file)
    if path.suffix == ".json":
        with open(path) as f:
            return json.load(f)
    # Plain text: one name per line
    lines = path.read_text().strip().splitlines()
    if len(lines) != K:
        LOGGER.warning("cluster_names_file has %d lines but K=%d; using defaults.", len(lines), K)
        return [f"subspace_{i}" for i in range(K)]
    return lines


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
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for subspace JASPER training."""
    parser = argparse.ArgumentParser(
        description="Train Subspace-Routed JASPER Predictor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--dataset-dir", required=True, help="Path to JASPER dataset directory")
    parser.add_argument("--output-dir", default="runs/subspace", help="Output directory")
    parser.add_argument(
        "--centroids-path",
        default=None,
        help="Path to centroid embeddings (.pt or .npy). "
        "Falls back to dataset.centroid_embs if not provided.",
    )
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs")
    parser.add_argument("--device", default=None, help="Device (e.g. 'cuda:0', 'cpu')")
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")

    # ------------------------------------------------------------------
    # Dataset build arguments (optional — run Tasks 1-8 before training)
    # ------------------------------------------------------------------
    build = parser.add_argument_group(
        "dataset build",
        "When --build-csv-paths is provided the dataset is built from raw CSV files "
        "before training.  The result is written to --dataset-dir.",
    )
    build.add_argument(
        "--build-csv-paths",
        nargs="+",
        default=None,
        metavar="CSV",
        help="One or more raw CSV files to merge and process (triggers build step).",
    )
    build.add_argument(
        "--build-cluster-json",
        default=None,
        metavar="JSON",
        help="Path to KeywordClusterer JSON (required when --build-csv-paths is set).",
    )
    build.add_argument(
        "--build-embedding-model",
        default="text-embedding-3-small",
        metavar="MODEL",
        help=(
            "Embedding model to use during the build step. "
            "Defaults to 'text-embedding-3-small' (LangChain OpenAI). "
            "Pass a HuggingFace/SentenceTransformer path (e.g. "
            "'sentence-transformers/all-MiniLM-L6-v2') to use a local model instead."
        ),
    )
    build.add_argument(
        "--build-embedding-batch-size",
        type=int,
        default=32,
        metavar="INT",
        help="Batch size used when generating embeddings during the build step (default: 32).",
    )
    build.add_argument(
        "--build-n-neg",
        type=int,
        default=12,
        metavar="N",
        help="Number of hard negatives per sample (default: 12).",
    )
    build.add_argument(
        "--build-normalize-embeddings",
        action="store_true",
        default=False,
        help="L2-normalise embeddings during the build step.",
    )
    build.add_argument(
        "--build-storage-format",
        type=str,
        default="pt",
        choices=["pt", "hdf5"],
        dest="build_storage_format",
        help='Storage format for the built dataset: "pt" (PyTorch tensors, default) or "hdf5".',
    )
    return parser.parse_args()


def main() -> None:
    """Run subspace JASPER training end-to-end."""
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # ------------------------------------------------------------------
    # 0. Dataset build  (Tasks 1-8, skipped when --build-csv-paths is absent)
    # ------------------------------------------------------------------
    if args.build_csv_paths:
        if not args.build_cluster_json:
            raise ValueError("--build-cluster-json is required when --build-csv-paths is set.")

        LOGGER.info(
            "Step 0: Building JASPER dataset from %d CSV file(s) → %s",
            len(args.build_csv_paths),
            args.dataset_dir,
        )
        if "/" in args.build_embedding_model:
            # HuggingFace / SentenceTransformer model path
            try:
                from sentence_transformers import SentenceTransformer

                embedding_model = SentenceTransformer(args.build_embedding_model)
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "sentence-transformers is required for this embedding model. "
                    "Install it with: pip install sentence-transformers"
                ) from exc
        else:
            # OpenAI model via LangChain (default: text-embedding-3-small)
            try:
                from langchain_openai import OpenAIEmbeddings

                embedding_model = OpenAIEmbeddings(model=args.build_embedding_model)
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "langchain-openai is required for the default embedding model. "
                    "Install it with: pip install langchain-openai"
                ) from exc

        build_dataset(
            csv_paths=args.build_csv_paths,
            cluster_json_path=args.build_cluster_json,
            embedding_model=embedding_model,
            output_dir=args.dataset_dir,
            n_neg=args.build_n_neg,
            normalize_embeddings=args.build_normalize_embeddings,
            embedding_batch_size=args.build_embedding_batch_size,
            storage_format=args.build_storage_format,
        )
        LOGGER.info("Step 0: Dataset build complete.")

    # ------------------------------------------------------------------
    # 1. Load config
    # ------------------------------------------------------------------
    cfg = load_config(args.config)
    LOGGER.info("Config loaded from %s", args.config)

    model_cfg = cfg.get("model", {})
    router_cfg = cfg.get("router", {})
    ema_cfg = cfg.get("ema", {})
    loss_cfg = cfg.get("loss", {})
    routing_loss_cfg = cfg.get("routing_loss", {})
    train_cfg = cfg.get("training", {})
    dataset_cfg = cfg.get("dataset", {})
    monitor_cfg = cfg.get("monitoring", {})
    xai_cfg = cfg.get("xai", {})

    if args.epochs is not None:
        train_cfg["num_epochs"] = args.epochs

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"

    num_epochs: int = train_cfg.get("num_epochs", 80)
    batch_size: int = train_cfg.get("batch_size", 64)
    lr: float = float(train_cfg.get("learning_rate", 2e-4))
    weight_decay: float = float(train_cfg.get("weight_decay", 1e-4))
    num_workers: int = train_cfg.get("num_workers", 4)
    warmup_epochs: int = train_cfg.get("warmup_epochs", 3)
    K: int = router_cfg.get("num_subspaces", 8)

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
    train_dataset: JASPERSteeringDataset = train_loader.dataset_obj
    val_dataset: JASPERSteeringDataset = val_loader.dataset_obj  # noqa: F841
    LOGGER.info(
        "Train: %d samples | Val: %d samples | n_neg: %d",
        len(train_loader.dataset),
        len(val_loader.dataset),
        train_dataset.n_neg,
    )
    assert train_dataset.n_neg >= 1, f"Dataset n_neg={train_dataset.n_neg} must be \u2265 1"

    # ------------------------------------------------------------------
    # 3. Centroid embeddings
    # ------------------------------------------------------------------
    if args.centroids_path:
        centroid_embs = load_centroids(args.centroids_path)
        LOGGER.info("Centroids loaded from %s: shape=%s", args.centroids_path, centroid_embs.shape)
    else:
        centroid_embs = train_dataset.centroid_embs
        if not isinstance(centroid_embs, torch.Tensor):
            centroid_embs = torch.as_tensor(centroid_embs, dtype=torch.float32)
        LOGGER.info("Centroids from dataset: shape=%s", centroid_embs.shape)
    assert centroid_embs.shape[0] == K, (
        f"centroid_embs has {centroid_embs.shape[0]} rows but router.num_subspaces K={K}. "
        "Ensure --centroids-path or the dataset centroid_embs.pt matches your config."
    )

    cluster_names = load_cluster_names(xai_cfg.get("cluster_names_file"), K)

    # ------------------------------------------------------------------
    # 4. Model + EMA encoder
    # ------------------------------------------------------------------
    D = model_cfg.get("embedding_dim", 768)
    if train_dataset.embedding_dim != D:
        LOGGER.warning(
            "Config embedding_dim=%d differs from dataset embedding_dim=%d; using dataset value.",
            D,
            train_dataset.embedding_dim,
        )
        D = train_dataset.embedding_dim

    decomposed_cfg = DecomposedJASPERConfig(
        embedding_dim=D,
        hidden_dim=model_cfg.get("hidden_dim", 512),
        num_subspaces=K,
        num_layers=model_cfg.get("num_layers", 3),
        dropout=model_cfg.get("dropout", 0.1),
        activation=model_cfg.get("activation", "GELU"),
        use_layer_norm=model_cfg.get("use_layer_norm", True),
        normalize_output=model_cfg.get("normalize_output", False),
        router_hidden_dim=router_cfg.get("hidden_dim", 256),
        router_temperature=float(router_cfg.get("temperature", 1.0)),
        router_gumbel_hard=router_cfg.get("gumbel_hard", False),
        router_normalize_input=router_cfg.get("normalize_input", True),
        fine_input_mode=router_cfg.get("fine_input_mode", "concat"),
    )
    model = DecomposedJASPERPredictor(decomposed_cfg)
    LOGGER.info("%s", model.get_model_summary())

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
    # 5. Loss functions
    # ------------------------------------------------------------------
    loss_fn = JASPERMultiObjectiveLoss(
        **{k: float(v) if isinstance(v, (int, float)) else v for k, v in loss_cfg.items()}
    )

    routing_loss_fn = RoutingLoss(
        weight=float(routing_loss_cfg.get("lambda_routing", 1.0)),
        label_smoothing=float(routing_loss_cfg.get("routing_label_smoothing", 0.0)),
    )
    entropy_reg_fn = EntropyRegularization(
        entropy_high=routing_loss_cfg.get("entropy_high"),  # None → log(K)
        entropy_low=float(routing_loss_cfg.get("entropy_low", 0.1)),
        anneal_epochs=int(routing_loss_cfg.get("anneal_epochs", 20)),
        weight=float(routing_loss_cfg.get("lambda_entropy", 0.1)),
    )
    residual_penalty_fn = ResidualPenalty(
        margin=float(routing_loss_cfg.get("residual_margin", 1.0)),
        weight=float(routing_loss_cfg.get("lambda_residual", 0.1)),
    )
    disentanglement_fn = DisentanglementLoss(
        weight=float(routing_loss_cfg.get("lambda_disentangle", 0.01)),
    )

    # ------------------------------------------------------------------
    # 6. Optimiser + scheduler
    # ------------------------------------------------------------------
    optimizer = optim.AdamW(
        list(model.parameters()) + list(ema_encoder.online_encoder.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = build_warmup_scheduler(optimizer, warmup_epochs, num_epochs, lr)

    # ------------------------------------------------------------------
    # 7. Monitor
    # ------------------------------------------------------------------
    wandb_config = {**cfg} if monitor_cfg.get("use_wandb", False) else None
    monitor = TrainingMonitor(
        output_dir=str(output_dir),
        use_wandb=monitor_cfg.get("use_wandb", False),
        wandb_project=monitor_cfg.get("wandb_project", "jasper-rag-subspace"),
        wandb_name=monitor_cfg.get("wandb_name"),
        wandb_config=wandb_config,
    )

    # ------------------------------------------------------------------
    # 8. XAI Interface
    # ------------------------------------------------------------------
    xai_interface: Optional[XAIInterface] = None
    if xai_cfg.get("save_val_xai", True):
        xai_interface = XAIInterface(
            model=model,
            centroid_embs=centroid_embs,
            cluster_names=cluster_names,
        )
        LOGGER.info("XAIInterface ready: K=%d", K)

    xai_output_dir = xai_cfg.get("xai_output_dir") or str(output_dir / "xai")

    # ------------------------------------------------------------------
    # 9. Trainer
    # ------------------------------------------------------------------
    device = args.device
    trainer_config = JASPERTrainerConfig(
        max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
        log_every_n_steps=train_cfg.get("log_every_n_steps", 50),
        checkpoint_dir=str(checkpoint_dir),
        save_every_n_epochs=train_cfg.get("save_every_n_epochs", 5),
        keep_last_n_checkpoints=train_cfg.get("keep_last_n_checkpoints", 3),
        device=device,
        mixed_precision=train_cfg.get("mixed_precision", False),
    )
    trainer = SubspaceJASPERTrainer(
        config=trainer_config,
        model=model,
        ema_encoder=ema_encoder,
        loss_fn=loss_fn,
        routing_loss_fn=routing_loss_fn,
        entropy_reg_fn=entropy_reg_fn,
        residual_penalty_fn=residual_penalty_fn,
        disentanglement_fn=disentanglement_fn,
        lambda_routing=float(routing_loss_cfg.get("lambda_routing", 1.0)),
        lambda_entropy=float(routing_loss_cfg.get("lambda_entropy", 0.1)),
        lambda_residual=float(routing_loss_cfg.get("lambda_residual", 0.1)),
        lambda_disentangle=float(routing_loss_cfg.get("lambda_disentangle", 0.01)),
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        monitor=monitor,
        xai_interface=xai_interface,
        xai_every_n_epochs=xai_cfg.get("xai_every_n_epochs", 5),
        xai_output_dir=xai_output_dir,
        reload_negatives_every_n_epochs=int(train_cfg.get("reload_negatives_every_n_epochs", 0)),
    )

    # Move centroids to trainer device
    trainer._centroid_embs = centroid_embs.to(trainer.device)

    # ------------------------------------------------------------------
    # 10. Resume
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
    # 11. Train
    # ------------------------------------------------------------------
    LOGGER.info("Starting subspace training for %d epochs → %s", num_epochs, output_dir)
    history = trainer.fit(num_epochs, start_epoch=start_epoch)

    # ------------------------------------------------------------------
    # 12. Finalise
    # ------------------------------------------------------------------
    trainer.save_checkpoint(output_dir / "final.pt", epoch=start_epoch + len(history) - 1)
    monitor.export_history(str(output_dir / "training_history.csv"))
    monitor.plot_losses(str(output_dir / "loss_curves.png"))
    monitor.finish()

    LOGGER.info("Done. Final model saved to %s/final.pt", output_dir)


if __name__ == "__main__":
    main()
