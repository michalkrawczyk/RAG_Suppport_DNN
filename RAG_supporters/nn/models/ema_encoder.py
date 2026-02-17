"""EMA (Exponential Moving Average) target encoder wrapper for JASPER training."""

from __future__ import annotations

import copy
import logging
import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

LOGGER = logging.getLogger(__name__)


class EMAEncoder(nn.Module):
    """Exponential Moving Average wrapper that maintains a slowly-updating target encoder.

    The *target encoder* is a copy of ``base_encoder`` whose parameters are
    updated via momentum rather than backpropagation.  This prevents
    representation collapse — the predictor must match a stable, slowly-drifting
    target rather than a moving one it can trivially align with.

    Used in JEPA, BYOL, MoCo-style self-supervised architectures.

    Architecture
    ------------
    ::

        online_encoder  ←── gradient descent (normal training)
        target_encoder  ←── EMA update:  θ_t ← τ·θ_t + (1-τ)·θ_online

    The momentum coefficient τ (tau) is annealed from ``tau_min`` to
    ``tau_max`` over ``max_steps`` using a cosine schedule, so the target
    encoder starts updating faster and gradually slows down.

    Args:
        base_encoder: The encoder module to wrap.  A deep copy is made for the
            target; the original becomes the ``online_encoder``.
        tau_min: Starting (lower) momentum value. Typical: 0.996.
        tau_max: Ending (upper) momentum value. Typical: 0.999.
        schedule: Annealing schedule for tau.  Currently only ``"cosine"``
            is supported.
    """

    def __init__(
        self,
        base_encoder: nn.Module,
        tau_min: float = 0.996,
        tau_max: float = 0.999,
        schedule: str = "cosine",
    ) -> None:
        super().__init__()

        if not (0.0 < tau_min <= tau_max < 1.0):
            raise ValueError(
                f"Require 0 < tau_min <= tau_max < 1, got tau_min={tau_min} tau_max={tau_max}"
            )
        if schedule not in ("cosine",):
            raise ValueError(f"Unsupported schedule '{schedule}'. Only 'cosine' is supported.")

        self.tau_min = tau_min
        self.tau_max = tau_max
        self.schedule = schedule

        # online_encoder is trained normally via backprop
        self.online_encoder = base_encoder

        # target_encoder is a deep copy; gradients are always disabled
        self.target_encoder = copy.deepcopy(base_encoder)
        self._freeze_target()

        LOGGER.debug(
            "EMAEncoder initialised: tau_min=%.4f tau_max=%.4f schedule=%s",
            tau_min,
            tau_max,
            schedule,
        )

    # ------------------------------------------------------------------
    # Target encoder management
    # ------------------------------------------------------------------

    def _freeze_target(self) -> None:
        """Disable gradients on the target encoder (called once at init)."""
        for param in self.target_encoder.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update_target(self, step: int, max_steps: int) -> None:
        """Perform one EMA update of the target encoder.

        Should be called **once per training step** after the optimizer step.

        Args:
            step: Current global training step (0-indexed).
            max_steps: Total number of training steps.
        """
        tau = self.get_tau(step, max_steps)
        for online_p, target_p in zip(
            self.online_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            target_p.data.mul_(tau).add_(online_p.data, alpha=1.0 - tau)

    def get_tau(self, step: int, max_steps: int) -> float:
        """Compute the current momentum coefficient using the configured schedule.

        Args:
            step: Current global training step (0-indexed).
            max_steps: Total number of training steps.

        Returns:
            Current tau value in ``[tau_min, tau_max]``.
        """
        if max_steps <= 0:
            return self.tau_max

        progress = min(step / max(max_steps, 1), 1.0)  # clamp to [0, 1]

        if self.schedule == "cosine":
            # Cosine annealing: starts at tau_min, ends at tau_max
            tau = self.tau_max - (self.tau_max - self.tau_min) * (
                0.5 * (1.0 + math.cos(math.pi * progress))
            )
            return tau

        # Unreachable given __init__ validation, but keeps mypy happy
        return self.tau_max  # pragma: no cover

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the *online* encoder (trainable, with grad).

        Args:
            x: Input tensor.

        Returns:
            Online encoder output.
        """
        return self.online_encoder(x)

    @torch.no_grad()
    def encode_target(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the *target* encoder (no gradient).

        Args:
            x: Input tensor.

        Returns:
            Target encoder output (detached).
        """
        return self.target_encoder(x)

    # ------------------------------------------------------------------
    # State dict overrides — include both encoders + tau params
    # ------------------------------------------------------------------

    def state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:  # type: ignore[override]
        """Return state dict containing both encoders and tau configuration."""
        base = super().state_dict(*args, **kwargs)
        base["_ema_tau_min"] = self.tau_min
        base["_ema_tau_max"] = self.tau_max
        base["_ema_schedule"] = self.schedule
        return base

    def load_state_dict(  # type: ignore[override]
        self, state_dict: Dict[str, Any], strict: bool = True
    ) -> Any:
        """Load state dict, restoring tau configuration if present."""
        # Pop meta keys before delegating to nn.Module
        tau_min = state_dict.pop("_ema_tau_min", self.tau_min)
        tau_max = state_dict.pop("_ema_tau_max", self.tau_max)
        schedule = state_dict.pop("_ema_schedule", self.schedule)

        self.tau_min = tau_min
        self.tau_max = tau_max
        self.schedule = schedule

        result = super().load_state_dict(state_dict, strict=strict)
        # Ensure target encoder gradients are still disabled after load
        self._freeze_target()
        return result

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_tau_info(self, step: int, max_steps: int) -> Dict[str, float]:
        """Return a dict with tau diagnostics for logging."""
        return {
            "tau": self.get_tau(step, max_steps),
            "tau_min": self.tau_min,
            "tau_max": self.tau_max,
            "step": step,
            "max_steps": max_steps,
        }

    def __repr__(self) -> str:
        return (
            f"EMAEncoder(tau_min={self.tau_min}, tau_max={self.tau_max}, "
            f"schedule='{self.schedule}')"
        )
