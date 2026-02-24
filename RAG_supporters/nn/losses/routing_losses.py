"""Routing-specific loss functions for the subspace-routed JASPER model."""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RoutingLoss  — supervised cluster routing
# ---------------------------------------------------------------------------


class RoutingLoss(nn.Module):
    """Cross-entropy loss supervising the subspace router.

    Encourages the router to assign each sample to its ground-truth cluster
    (as provided by the dataset's ``cluster_id`` field).

    Args:
        weight: Scalar multiplier applied to the loss before returning.
        label_smoothing: Label smoothing factor in ``[0, 1)``. 0 = disabled.
    """

    def __init__(self, weight: float = 1.0, label_smoothing: float = 0.0) -> None:
        super().__init__()
        if not (0.0 <= label_smoothing < 1.0):
            raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}")
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(
        self,
        routing_logits: torch.Tensor,
        true_cluster_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute routing cross-entropy loss.

        Args:
            routing_logits: Raw (pre-softmax) router logits ``[B, K]``.
            true_cluster_ids: Ground-truth cluster indices ``[B]`` (long tensor).

        Returns:
            Dict with:
                - ``"routing"``: scalar loss (gradient-bearing).
                - ``"routing_acc"``: top-1 accuracy (detached, no grad).
        """
        loss = (
            F.cross_entropy(
                routing_logits,
                true_cluster_ids,
                label_smoothing=self.label_smoothing,
            )
            * self.weight
        )

        with torch.no_grad():
            preds = routing_logits.argmax(dim=-1)
            acc = (preds == true_cluster_ids).float().mean()

        return {"routing": loss, "routing_acc": acc}


# ---------------------------------------------------------------------------
# EntropyRegularization  — annealed routing entropy
# ---------------------------------------------------------------------------


class EntropyRegularization(nn.Module):
    """Entropy-annealing regulariser for routing weights.

    Steers routing entropy along a linear schedule:
    - **Early training** (epoch 0): pushes entropy toward ``entropy_high``
      (uniform routing encourages exploration).
    - **Late training** (epoch >= ``anneal_epochs``): pushes entropy toward
      ``entropy_low`` (confident, peaked routing).

    Loss:

        ``L = weight × (H(w) − target_entropy(epoch))²``

    where ``H(w) = −∑_k w_k log(w_k)`` (per sample, averaged over the batch).

    Args:
        entropy_high: Target entropy at epoch 0 (nats).  ``None`` → defaults to
            ``log(K)`` (the maximum entropy for K categories), inferred from the
            first ``forward()`` call.
        entropy_low: Target entropy after ``anneal_epochs`` epochs (nats).
        anneal_epochs: Number of epochs to anneal from high to low.
        weight: Scalar multiplier on the loss.
        eps: Small constant for numerical stability inside the log.
    """

    def __init__(
        self,
        entropy_high: Optional[float] = None,
        entropy_low: float = 0.1,
        anneal_epochs: int = 20,
        weight: float = 0.1,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if entropy_low < 0:
            raise ValueError(f"entropy_low must be >= 0, got {entropy_low}")
        if anneal_epochs < 1:
            raise ValueError(f"anneal_epochs must be >= 1, got {anneal_epochs}")
        self._entropy_high = entropy_high  # None → will be set on first call
        self.entropy_low = entropy_low
        self.anneal_epochs = anneal_epochs
        self.weight = weight
        self.eps = eps

    def get_target_entropy(self, current_epoch: int) -> float:
        """Return the scheduled target entropy for this epoch.

        Args:
            current_epoch: Current epoch index (0-based).

        Returns:
            Target entropy in nats.
        """
        if self._entropy_high is None:
            # Fallback if forward() hasn't been called yet; caller should ensure
            # entropy_high is initialised first via a forward() call.
            return self.entropy_low
        progress = min(1.0, current_epoch / self.anneal_epochs)
        return self._entropy_high + (self.entropy_low - self._entropy_high) * progress

    def forward(
        self,
        routing_weights: torch.Tensor,
        current_epoch: int,
    ) -> Dict[str, torch.Tensor]:
        """Compute entropy regularisation loss.

        Args:
            routing_weights: Soft routing probabilities ``[B, K]`` (post-Gumbel /
                post-Softmax).  Values should be >= 0 and sum to 1 along dim=-1.
            current_epoch: Current epoch index (0-based).

        Returns:
            Dict with:
                - ``"entropy_reg"``: scalar loss (gradient-bearing).
                - ``"routing_entropy"``: mean batch entropy in nats (detached).
        """
        K = routing_weights.shape[-1]

        # Lazy initialisation of entropy_high
        if self._entropy_high is None:
            self._entropy_high = math.log(K)

        # Per-sample Shannon entropy: H(w) = −∑_k w_k log(w_k + ε)
        entropy = -(routing_weights * (routing_weights + self.eps).log()).sum(dim=-1)  # [B]
        mean_entropy = entropy.mean()  # scalar

        target = self.get_target_entropy(current_epoch)
        loss = self.weight * (mean_entropy - target).pow(2)

        return {
            "entropy_reg": loss,
            "routing_entropy": mean_entropy.detach(),
        }


# ---------------------------------------------------------------------------
# ResidualPenalty  — hinge loss on fine-vector norm
# ---------------------------------------------------------------------------


class ResidualPenalty(nn.Module):
    """Hinge loss penalising large fine (residual) vectors.

    Encourages the model to rely on subspace centroids rather than producing
    a large residual correction that bypasses the routing mechanism.

    Loss:

        ``L = weight × mean(relu(‖fine‖ − margin))``

    Args:
        margin: Allowed residual L2 norm.  Samples below this threshold incur
            zero penalty.  Scale with the typical embedding norm in your dataset:
            use ~0.5–1.5 × median(‖target_emb‖).  Default ``1.0`` is appropriate
            for unit-normalised embeddings.
        weight: Scalar multiplier on the loss.
    """

    def __init__(self, margin: float = 1.0, weight: float = 0.1) -> None:
        super().__init__()
        if margin <= 0:
            raise ValueError(f"margin must be > 0, got {margin}")
        self.margin = margin
        self.weight = weight

    def forward(
        self,
        fine_vector: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute residual penalty.

        Args:
            fine_vector: Fine (residual) embeddings ``[B, D]``.

        Returns:
            Dict with:
                - ``"residual_penalty"``: scalar hinge loss (gradient-bearing).
                - ``"residual_norm_mean"``: mean residual norm (detached, for monitoring).
        """
        residual_norms = fine_vector.norm(dim=-1)  # [B]
        hinge = F.relu(residual_norms - self.margin)  # [B]
        loss = self.weight * hinge.mean()

        return {
            "residual_penalty": loss,
            "residual_norm_mean": residual_norms.mean().detach(),
        }


# ---------------------------------------------------------------------------
# DisentanglementLoss  — routing axis decorrelation
# ---------------------------------------------------------------------------


class DisentanglementLoss(nn.Module):
    """Covariance penalty across routing weight axes.

    Encourages different routing dimensions to be statistically independent.
    Uses the same off-diagonal covariance formula as VICReg (Bardes et al. 2022),
    applied to the routing weight matrix ``[B, K]`` instead of embeddings.

    Loss:

        ``L = weight × off_diag_cov(routing_weights).pow(2).sum() / K``

    Args:
        weight: Scalar multiplier on the loss.
        eps: Added to batch variance for numerical stability.
    """

    def __init__(self, weight: float = 0.01, eps: float = 1e-4) -> None:
        super().__init__()
        self.weight = weight
        self.eps = eps

    def forward(
        self,
        routing_weights: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute disentanglement (routing covariance) loss.

        Args:
            routing_weights: Soft routing probabilities ``[B, K]``.

        Returns:
            Dict with ``"disentanglement"``: scalar loss (gradient-bearing).
        """
        B, K = routing_weights.shape
        if B < 2:
            # Covariance is undefined for a single sample; return zero
            return {"disentanglement": routing_weights.sum() * 0.0}

        loss = self.weight * self._covariance_loss(routing_weights)
        return {"disentanglement": loss}

    def _covariance_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Off-diagonal covariance penalty (same formula as VICRegLoss._covariance_loss)."""
        B, K = z.shape
        z_centered = z - z.mean(dim=0)  # [B, K]
        cov = (z_centered.T @ z_centered) / (B - 1)  # [K, K]
        off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
        return off_diag / K
