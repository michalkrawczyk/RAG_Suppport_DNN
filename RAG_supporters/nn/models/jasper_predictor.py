"""JASPER Predictor model: question + steering → predicted source embedding."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class JASPERPredictorConfig:
    """Configuration for JASPERPredictor.

    Args:
        embedding_dim: Dimensionality of input question/steering/output embeddings (D).
        hidden_dim: Width of hidden layers (H).
        num_layers: Number of Linear blocks in each encoder and in the predictor head.
            Must be >= 1.
        dropout: Dropout probability applied after each hidden activation. 0 disables it.
        activation: Name of ``torch.nn`` activation class to use (e.g. ``"GELU"``).
        use_layer_norm: Whether to apply LayerNorm after each hidden activation.
        normalize_output: Whether to L2-normalize the output embedding.
    """

    embedding_dim: int = 768
    hidden_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.1
    activation: str = "GELU"
    use_layer_norm: bool = True
    normalize_output: bool = False

    def __post_init__(self) -> None:
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be > 0, got {self.embedding_dim}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {self.hidden_dim}")
        if not hasattr(nn, self.activation):
            raise ValueError(f"Unknown activation '{self.activation}'. Must be a torch.nn class.")

    @classmethod
    def from_dict(cls, d: dict) -> "JASPERPredictorConfig":
        """Create config from a plain dictionary (e.g. loaded from YAML)."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def _make_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    num_layers: int,
    dropout: float,
    activation: str,
    use_layer_norm: bool,
) -> nn.Sequential:
    """Build a configurable MLP block.

    For ``num_layers == 1`` the result is a single ``Linear(in_dim, out_dim)``
    with no activation/dropout.  For ``num_layers > 1`` the intermediate layers
    map ``in_dim → hidden_dim → … → hidden_dim`` each followed by activation,
    optional LayerNorm, and optional Dropout; the final layer maps
    ``hidden_dim → out_dim`` with no activation.
    """
    activation_cls = getattr(nn, activation)
    layers: list[nn.Module] = []

    if num_layers == 1:
        layers.append(nn.Linear(in_dim, out_dim))
        return nn.Sequential(*layers)

    # First hidden layer
    layers.append(nn.Linear(in_dim, hidden_dim))
    layers.append(activation_cls())
    if use_layer_norm:
        layers.append(nn.LayerNorm(hidden_dim))
    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))

    # Intermediate hidden layers
    for _ in range(num_layers - 2):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(activation_cls())
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))

    # Output projection
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# JASPERPredictor
# ---------------------------------------------------------------------------


class JASPERPredictor(nn.Module):
    """JASPER predictor: ``question_emb + steering_emb → predicted_source_emb``.

    Architecture
    ------------
    ::

        question_emb  [B, D] ──► question_encoder  [B, H]  ──┐
                                                               ├─ concat [B, 2H] ──► predictor_head ──► [B, D]
        steering_emb  [B, D] ──► steering_encoder  [B, H]  ──┘

    Each sub-network is a configurable MLP (see :class:`JASPERPredictorConfig`).

    Args:
        config: Model configuration.  Can also be passed as a plain ``dict``
            which will be coerced via :meth:`JASPERPredictorConfig.from_dict`.
    """

    def __init__(self, config: JASPERPredictorConfig | dict) -> None:
        super().__init__()

        if isinstance(config, dict):
            config = JASPERPredictorConfig.from_dict(config)
        self.config = config

        D = config.embedding_dim
        H = config.hidden_dim
        n = config.num_layers
        p = config.dropout
        act = config.activation
        ln = config.use_layer_norm

        self.question_encoder = _make_mlp(D, H, H, n, p, act, ln)
        self.steering_encoder = _make_mlp(D, H, H, n, p, act, ln)
        # Predictor head: concat of two H-dim latents → output embedding of dim D
        self.predictor_head = _make_mlp(2 * H, H, D, n, p, act, ln)

        self._normalize_output = config.normalize_output
        self._latents: Dict[str, torch.Tensor] = {}

        self._init_weights()
        LOGGER.debug(
            "JASPERPredictor initialised: D=%d H=%d layers=%d params=%d",
            D,
            H,
            n,
            sum(p.numel() for p in self.parameters()),
        )

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Xavier-uniform init for Linear layers, zeros for biases."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0 / math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        question_emb: torch.Tensor,
        steering_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Predict source embedding from question and steering.

        Args:
            question_emb: Float tensor of shape ``[B, D]``.
            steering_emb: Float tensor of shape ``[B, D]``.

        Returns:
            Predicted source embedding of shape ``[B, D]``.
        """
        q_latent = self.question_encoder(question_emb)  # [B, H]
        s_latent = self.steering_encoder(steering_emb)  # [B, H]

        # Cache for introspection (detached to avoid holding the graph)
        self._latents = {
            "question_latent": q_latent.detach(),
            "steering_latent": s_latent.detach(),
        }

        combined = torch.cat([q_latent, s_latent], dim=-1)  # [B, 2H]
        prediction = self.predictor_head(combined)  # [B, D]

        if self._normalize_output:
            prediction = nn.functional.normalize(prediction, dim=-1)

        return prediction

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_latent_representations(self) -> Dict[str, torch.Tensor]:
        """Return cached intermediate activations from the last forward pass.

        Keys: ``"question_latent"``, ``"steering_latent"``.

        Returns:
            Dict of detached CPU tensors (empty if no forward pass has run).
        """
        return {k: v.cpu() for k, v in self._latents.items()}

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of input and output embeddings."""
        return self.config.embedding_dim

    @property
    def hidden_dim(self) -> int:
        """Width of hidden layers."""
        return self.config.hidden_dim

    def get_model_summary(self) -> str:
        """Return a human-readable parameter summary."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"JASPERPredictor | D={self.config.embedding_dim} H={self.config.hidden_dim} "
            f"layers={self.config.num_layers} | "
            f"total params: {total:,} | trainable: {trainable:,}"
        )

    def __repr__(self) -> str:
        return (
            f"JASPERPredictor(embedding_dim={self.config.embedding_dim}, "
            f"hidden_dim={self.config.hidden_dim}, "
            f"num_layers={self.config.num_layers})"
        )
