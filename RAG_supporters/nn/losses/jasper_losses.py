"""Multi-objective loss functions for JASPER model training."""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# L_jasper  — prediction loss in latent space
# ---------------------------------------------------------------------------


class JASPERLoss(nn.Module):
    """SmoothL1 (Huber) loss between predicted and EMA-target embeddings.

    Measures how accurately the predictor reconstructs the target encoder's
    representation.  SmoothL1 is preferred over MSE because it is less
    sensitive to outliers in the embedding space.

    Args:
        beta: Transition point between L1 and L2 behaviour (SmoothL1 ``beta``).
        reduction: ``"mean"`` or ``"sum"``.
    """

    def __init__(self, beta: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be 'mean' or 'sum', got '{reduction}'")
        self.beta = beta
        self.reduction = reduction
        self._loss_fn = nn.SmoothL1Loss(beta=beta, reduction=reduction)

    def forward(
        self,
        predicted: torch.Tensor,
        ema_target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute JASPER prediction loss.

        Args:
            predicted: Predicted source embeddings ``[B, D]``.
            ema_target: EMA target encoder output ``[B, D]``.

        Returns:
            Dict with key ``"jasper"`` containing the scalar loss.
        """
        loss = self._loss_fn(predicted, ema_target)
        return {"jasper": loss}


# ---------------------------------------------------------------------------
# L_contrastive  — InfoNCE with hard negatives
# ---------------------------------------------------------------------------


class ContrastiveLoss(nn.Module):
    """InfoNCE contrastive loss with hard negatives.

    Pulls the predicted embedding toward the positive target while pushing it
    away from hard negative examples in the batch.

    Args:
        temperature: Softmax temperature τ.  Lower = sharper distribution.
        reduction: ``"mean"`` or ``"sum"``.
    """

    def __init__(self, temperature: float = 0.07, reduction: str = "mean") -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be 'mean' or 'sum', got '{reduction}'")
        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        negatives: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute InfoNCE contrastive loss.

        Args:
            predicted: Predicted embeddings ``[B, D]``.
            target: Positive target embeddings ``[B, D]``.
            negatives: Hard negative embeddings ``[B, K, D]`` where K is the
                number of negatives per sample.

        Returns:
            Dict with key ``"contrastive"`` containing the scalar loss.
        """
        B, D = predicted.shape
        K = negatives.shape[1]

        # Normalise to unit sphere for cosine similarity
        pred_norm = F.normalize(predicted, dim=-1)          # [B, D]
        tgt_norm = F.normalize(target, dim=-1)              # [B, D]
        neg_norm = F.normalize(negatives, dim=-1)           # [B, K, D]

        # Positive similarity: [B]
        pos_sim = (pred_norm * tgt_norm).sum(dim=-1, keepdim=True) / self.temperature  # [B, 1]

        # Negative similarities: [B, K]
        neg_sim = torch.bmm(neg_norm, pred_norm.unsqueeze(-1)).squeeze(-1) / self.temperature  # [B, K]

        # Concatenate: [B, 1+K] — positive is always the 0th column
        logits = torch.cat([pos_sim, neg_sim], dim=-1)  # [B, 1+K]
        labels = torch.zeros(B, dtype=torch.long, device=predicted.device)

        loss = F.cross_entropy(logits, labels, reduction=self.reduction)
        return {"contrastive": loss}


# ---------------------------------------------------------------------------
# L_centroid  — auxiliary cluster classification
# ---------------------------------------------------------------------------


class CentroidLoss(nn.Module):
    """Cross-entropy loss that asks: does the prediction land in the right cluster?

    Uses cosine similarity between the predicted embedding and each cluster
    centroid as logits, then applies cross-entropy against the ground-truth
    cluster ID.

    Args:
        temperature: Softmax temperature applied to centroid similarities.
    """

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.temperature = temperature

    def forward(
        self,
        predicted_emb: torch.Tensor,
        centroid_embs: torch.Tensor,
        cluster_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute centroid classification loss.

        Args:
            predicted_emb: Predicted source embeddings ``[B, D]``.
            centroid_embs: Cluster centroid embeddings ``[C, D]`` where C is
                the number of clusters.
            cluster_ids: Ground-truth cluster indices ``[B]`` (long tensor).

        Returns:
            Dict with keys ``"centroid"`` (total loss) and
            ``"centroid_acc"`` (top-1 accuracy scalar, no grad).
        """
        # Normalise
        pred_norm = F.normalize(predicted_emb, dim=-1)        # [B, D]
        cent_norm = F.normalize(centroid_embs, dim=-1)        # [C, D]

        # Cosine similarity → logits: [B, C]
        logits = (pred_norm @ cent_norm.T) / self.temperature

        loss = F.cross_entropy(logits, cluster_ids)

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            acc = (preds == cluster_ids).float().mean()

        return {"centroid": loss, "centroid_acc": acc}


# ---------------------------------------------------------------------------
# L_vicreg  — variance + invariance + covariance regularisation
# ---------------------------------------------------------------------------


class VICRegLoss(nn.Module):
    """VICReg regularisation: Variance + Invariance + Covariance.

    Prevents embedding collapse without requiring negative pairs.

    - **Variance**: keeps per-dimension std above a threshold γ.
    - **Invariance**: SmoothL1 between two views (here predicted vs. target).
    - **Covariance**: penalises off-diagonal covariance entries to decorrelate dims.

    Reference: Bardes et al. 2022 "VICReg: Variance-Invariance-Covariance
    Regularization for Self-Supervised Learning".

    Args:
        lambda_v: Weight for variance term.
        lambda_i: Weight for invariance term.
        lambda_c: Weight for covariance term.
        gamma: Target std threshold for variance term.
        eps: Small constant for numerical stability.
    """

    def __init__(
        self,
        lambda_v: float = 25.0,
        lambda_i: float = 25.0,
        lambda_c: float = 1.0,
        gamma: float = 1.0,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.lambda_v = lambda_v
        self.lambda_i = lambda_i
        self.lambda_c = lambda_c
        self.gamma = gamma
        self.eps = eps

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute VICReg loss.

        Args:
            predicted: Predicted embeddings ``[B, D]``.
            target: Target (e.g. EMA) embeddings ``[B, D]``.

        Returns:
            Dict with keys ``"vicreg"``, ``"vicreg_v"``, ``"vicreg_i"``,
            ``"vicreg_c"`` for total and individual components.
        """
        B, D = predicted.shape

        # --- Invariance ---
        inv_loss = F.smooth_l1_loss(predicted, target)

        # --- Variance ---
        v_loss = self._variance_loss(predicted) + self._variance_loss(target)

        # --- Covariance ---
        c_loss = self._covariance_loss(predicted) + self._covariance_loss(target)

        total = self.lambda_v * v_loss + self.lambda_i * inv_loss + self.lambda_c * c_loss

        return {
            "vicreg": total,
            "vicreg_v": v_loss.detach(),
            "vicreg_i": inv_loss.detach(),
            "vicreg_c": c_loss.detach(),
        }

    def _variance_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Hinge loss on per-dimension standard deviation."""
        std = torch.sqrt(z.var(dim=0) + self.eps)             # [D]
        loss = F.relu(self.gamma - std).mean()
        return loss

    def _covariance_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Off-diagonal covariance penalty."""
        B, D = z.shape
        z_centered = z - z.mean(dim=0)                         # [B, D]
        cov = (z_centered.T @ z_centered) / (B - 1)           # [D, D]
        # Zero out diagonal, penalise squared off-diagonal entries
        off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
        loss = off_diag / D
        return loss


# ---------------------------------------------------------------------------
# JASPERMultiObjectiveLoss  — combined loss
# ---------------------------------------------------------------------------


class JASPERMultiObjectiveLoss(nn.Module):
    """Combined multi-objective loss for JASPER training.

    Combines :class:`JASPERLoss`, :class:`ContrastiveLoss`,
    :class:`CentroidLoss`, and :class:`VICRegLoss` with configurable
    scalar weights.

    Args:
        lambda_jasper: Weight for the JASPER prediction loss.
        lambda_contrastive: Weight for the contrastive (InfoNCE) loss.
        lambda_centroid: Weight for the centroid classification loss.
        lambda_vicreg: Weight for VICReg regularisation.
        jasper_beta: SmoothL1 beta for :class:`JASPERLoss`.
        contrastive_temperature: Temperature for :class:`ContrastiveLoss`.
        centroid_temperature: Temperature for :class:`CentroidLoss`.
        vicreg_lambda_v: VICReg variance weight.
        vicreg_lambda_i: VICReg invariance weight.
        vicreg_lambda_c: VICReg covariance weight.
    """

    def __init__(
        self,
        lambda_jasper: float = 1.0,
        lambda_contrastive: float = 0.5,
        lambda_centroid: float = 0.1,
        lambda_vicreg: float = 0.1,
        jasper_beta: float = 1.0,
        contrastive_temperature: float = 0.07,
        centroid_temperature: float = 0.1,
        vicreg_lambda_v: float = 25.0,
        vicreg_lambda_i: float = 25.0,
        vicreg_lambda_c: float = 1.0,
    ) -> None:
        super().__init__()

        self.lambda_jasper = lambda_jasper
        self.lambda_contrastive = lambda_contrastive
        self.lambda_centroid = lambda_centroid
        self.lambda_vicreg = lambda_vicreg

        self.jasper_loss = JASPERLoss(beta=jasper_beta)
        self.contrastive_loss = ContrastiveLoss(temperature=contrastive_temperature)
        self.centroid_loss = CentroidLoss(temperature=centroid_temperature)
        self.vicreg_loss = VICRegLoss(
            lambda_v=vicreg_lambda_v,
            lambda_i=vicreg_lambda_i,
            lambda_c=vicreg_lambda_c,
        )

    def forward(
        self,
        predicted: torch.Tensor,
        ema_target: torch.Tensor,
        negatives: torch.Tensor,
        centroid_embs: torch.Tensor,
        cluster_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute weighted multi-objective loss.

        Args:
            predicted: Predicted source embeddings ``[B, D]``.
            ema_target: EMA target encoder output ``[B, D]``.
            negatives: Hard negative embeddings ``[B, K, D]``.
            centroid_embs: Cluster centroid embeddings ``[C, D]``.
            cluster_ids: Ground-truth cluster indices ``[B]``.

        Returns:
            Tuple of:
            - ``total_loss``: Weighted scalar loss (gradient-bearing).
            - ``loss_dict``: Dict of all individual loss components
              (for logging; detached except total).
        """
        j_dict = self.jasper_loss(predicted, ema_target)
        c_dict = self.contrastive_loss(predicted, ema_target, negatives)
        cen_dict = self.centroid_loss(predicted, centroid_embs, cluster_ids)
        v_dict = self.vicreg_loss(predicted, ema_target)

        total = (
            self.lambda_jasper * j_dict["jasper"]
            + self.lambda_contrastive * c_dict["contrastive"]
            + self.lambda_centroid * cen_dict["centroid"]
            + self.lambda_vicreg * v_dict["vicreg"]
        )

        loss_dict: Dict[str, torch.Tensor] = {
            "total": total,
            **{k: v.detach() for k, v in j_dict.items()},
            **{k: v.detach() for k, v in c_dict.items()},
            **{k: v.detach() for k, v in cen_dict.items()},
            **{k: v.detach() for k, v in v_dict.items()},
        }

        return total, loss_dict

    def __repr__(self) -> str:
        return (
            f"JASPERMultiObjectiveLoss("
            f"λ_jasper={self.lambda_jasper}, "
            f"λ_contrastive={self.lambda_contrastive}, "
            f"λ_centroid={self.lambda_centroid}, "
            f"λ_vicreg={self.lambda_vicreg})"
        )
