"""SubspaceRouter: routes question+steering embeddings to concept subspaces."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from RAG_supporters.nn.models.jasper_predictor import _make_mlp

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SubspaceRouterConfig:
    """Configuration for :class:`SubspaceRouter`.

    Args:
        embedding_dim: Dimensionality of question/steering input embeddings (D).
        hidden_dim: Width of hidden layers in the routing MLP (H).
            Can be smaller than the main predictor hidden_dim.
        num_subspaces: Number of concept subspaces to route to (K). Must be >= 2.
        num_layers: Number of Linear blocks in the routing MLP. Must be >= 1.
        dropout: Dropout probability. 0 disables it.
        activation: Name of ``torch.nn`` activation class (e.g. ``"GELU"``).
        use_layer_norm: Whether to apply LayerNorm after hidden activations.
        temperature: Temperature for Gumbel-Softmax (training) and Softmax (inference).
            Lower values produce sharper distributions. Must be > 0.
        gumbel_hard: If ``True``, use straight-through hard Gumbel-Softmax
            (one-hot during forward, but gradients flow through soft version).
            Default ``False`` — soft Gumbel weights.
        normalize_input: If ``True``, L2-normalise the concatenated
            ``[question_emb; steering_emb]`` vector before feeding the MLP.
    """

    embedding_dim: int = 768
    hidden_dim: int = 256
    num_subspaces: int = 8
    num_layers: int = 2
    dropout: float = 0.1
    activation: str = "GELU"
    use_layer_norm: bool = True
    temperature: float = 1.0
    gumbel_hard: bool = False
    normalize_input: bool = True

    def __post_init__(self) -> None:
        if self.num_subspaces < 2:
            raise ValueError(f"num_subspaces must be >= 2, got {self.num_subspaces}")
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
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
    def from_dict(cls, d: dict) -> "SubspaceRouterConfig":
        """Create config from a plain dictionary (e.g. loaded from YAML)."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# SubspaceRouter
# ---------------------------------------------------------------------------


class SubspaceRouter(nn.Module):
    """Differentiable subspace router: routes (question, steering) to K concept subspaces.

    During training, Gumbel-Softmax produces stochastic-but-differentiable
    routing weights.  During inference, plain Softmax gives deterministic
    soft assignments (or hard argmax when ``config.gumbel_hard=True``).

    Args:
        config: Router configuration.  Can be passed as a plain ``dict``
            which will be coerced via :meth:`SubspaceRouterConfig.from_dict`.

    Example::

        router = SubspaceRouter(SubspaceRouterConfig(embedding_dim=768, num_subspaces=8))
        weights, logits = router(question_emb, steering_emb)   # [B,K], [B,K]
    """

    def __init__(self, config: SubspaceRouterConfig | dict) -> None:
        super().__init__()

        if isinstance(config, dict):
            config = SubspaceRouterConfig.from_dict(config)
        self.config = config

        D = config.embedding_dim
        H = config.hidden_dim
        K = config.num_subspaces

        # Input is concat([q, s]) → [B, 2D]
        self.router_mlp = _make_mlp(
            in_dim=2 * D,
            hidden_dim=H,
            out_dim=K,
            num_layers=config.num_layers,
            dropout=config.dropout,
            activation=config.activation,
            use_layer_norm=config.use_layer_norm,
        )

        self._init_weights()
        LOGGER.debug(
            "SubspaceRouter initialised: D=%d H=%d K=%d layers=%d params=%d",
            D, H, K, config.num_layers,
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
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute routing weights and logits from question and steering embeddings.

        Args:
            question_emb: Float tensor ``[B, D]``.
            steering_emb: Float tensor ``[B, D]``.
            training: If ``True``, apply Gumbel-Softmax noise.
                If ``False``, use deterministic Softmax.
                Defaults to ``True``; callers can pass ``self.training`` explicitly.

        Returns:
            Tuple of:
                - ``routing_weights`` ``[B, K]``: soft routing probabilities summing to 1.
                - ``concept_logits`` ``[B, K]``: raw unnormalised logits before (Gumbel-)Softmax.
        """
        x = torch.cat([question_emb, steering_emb], dim=-1)  # [B, 2D]

        if self.config.normalize_input:
            x = F.normalize(x, dim=-1)

        concept_logits = self.router_mlp(x)  # [B, K]

        if training:
            routing_weights = F.gumbel_softmax(
                concept_logits,
                tau=self.config.temperature,
                hard=self.config.gumbel_hard,
                dim=-1,
            )
        else:
            routing_weights = F.softmax(concept_logits / self.config.temperature, dim=-1)

        return routing_weights, concept_logits

    # ------------------------------------------------------------------
    # High-level helpers
    # ------------------------------------------------------------------

    def get_primary_subspace(
        self,
        question_emb: torch.Tensor,
        steering_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the dominant subspace index and confidence for each sample.

        Args:
            question_emb: ``[B, D]``
            steering_emb: ``[B, D]``

        Returns:
            Tuple of:
                - ``cluster_ids`` ``[B]`` (long): argmax of routing weights.
                - ``confidences`` ``[B]`` (float): corresponding max weight value.
        """
        with torch.no_grad():
            routing_weights, _ = self.forward(question_emb, steering_emb, training=False)
        confidences, cluster_ids = routing_weights.max(dim=-1)
        return cluster_ids, confidences

    def explain(
        self,
        question_emb: torch.Tensor,
        steering_emb: torch.Tensor,
        cluster_names: List[str],
    ) -> Dict[str, Any]:
        """Return a human-readable routing explanation for a batch.

        Args:
            question_emb: ``[B, D]`` or ``[D]`` (single sample).
            steering_emb: ``[B, D]`` or ``[D]``.
            cluster_names: List of K human-readable subspace names.

        Returns:
            Dict with keys:
                - ``"routing_weights"``: list of K floats (mean over batch).
                - ``"primary_subspace"``: name of the dominant subspace.
                - ``"primary_confidence"``: float confidence of the dominant subspace.
                - ``"entropy"``: routing entropy in nats (mean over batch).
                - ``"cluster_names"``: the provided cluster names list.
        """
        if len(cluster_names) != self.config.num_subspaces:
            raise ValueError(
                f"len(cluster_names)={len(cluster_names)} must equal "
                f"num_subspaces={self.config.num_subspaces}"
            )

        if question_emb.dim() == 1:
            question_emb = question_emb.unsqueeze(0)
            steering_emb = steering_emb.unsqueeze(0)

        with torch.no_grad():
            routing_weights, _ = self.forward(question_emb, steering_emb, training=False)

        mean_weights = routing_weights.mean(dim=0)  # [K]
        entropy = -(routing_weights * (routing_weights + 1e-8).log()).sum(dim=-1).mean()
        primary_idx = mean_weights.argmax().item()

        return {
            "routing_weights": mean_weights.cpu().tolist(),
            "primary_subspace": cluster_names[primary_idx],
            "primary_confidence": mean_weights[primary_idx].item(),
            "entropy": entropy.item(),
            "cluster_names": cluster_names,
        }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def num_subspaces(self) -> int:
        """Number of concept subspaces K."""
        return self.config.num_subspaces

    @property
    def embedding_dim(self) -> int:
        """Input embedding dimension D."""
        return self.config.embedding_dim

    def get_model_summary(self) -> str:
        """Return a human-readable parameter summary."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"SubspaceRouter | D={self.config.embedding_dim} H={self.config.hidden_dim} "
            f"K={self.config.num_subspaces} layers={self.config.num_layers} | "
            f"total params: {total:,} | trainable: {trainable:,}"
        )

    def __repr__(self) -> str:
        return (
            f"SubspaceRouter(embedding_dim={self.config.embedding_dim}, "
            f"hidden_dim={self.config.hidden_dim}, "
            f"num_subspaces={self.config.num_subspaces})"
        )
