"""DecomposedJASPERPredictor: coarse subspace routing + fine residual correction."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from RAG_supporters.nn.models.jasper_predictor import _make_mlp
from RAG_supporters.nn.models.subspace_router import SubspaceRouter, SubspaceRouterConfig

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DecomposedJASPERConfig:
    """Configuration for :class:`DecomposedJASPERPredictor`.

    Args:
        embedding_dim: Dimensionality of input/output embeddings (D).
        hidden_dim: Width of hidden layers for question/steering encoders (H).
        num_subspaces: Number of routing subspaces (K).
            Must match the number of centroid embeddings passed to ``forward()``.
        num_layers: MLP depth for question/steering encoders and fine MLP.
        dropout: Dropout probability. 0 disables it.
        activation: Name of ``torch.nn`` activation class (e.g. ``"GELU"``).
        use_layer_norm: Whether to apply LayerNorm after hidden activations.
        normalize_output: Whether to L2-normalise the final prediction.
        router_hidden_dim: Hidden width for the :class:`SubspaceRouter` MLP.
            Can differ from ``hidden_dim``.
        router_temperature: Gumbel-Softmax / Softmax temperature for the router.
        router_gumbel_hard: Whether to use straight-through hard Gumbel-Softmax.
        router_normalize_input: Whether to L2-normalise the router's input.
        fine_input_mode: How to combine latents and coarse vector in the fine MLP.
            ``"concat"`` (default): ``fine_mlp`` receives ``[q_latent; s_latent; coarse]``
            of size ``[B, 2H+D]``.
            ``"add"``: ``fine_mlp`` receives the elementwise sum of projected latents
            of size ``[B, H]``.
    """

    embedding_dim: int = 768
    hidden_dim: int = 512
    num_subspaces: int = 8
    num_layers: int = 2
    dropout: float = 0.1
    activation: str = "GELU"
    use_layer_norm: bool = True
    normalize_output: bool = False
    router_hidden_dim: int = 256
    router_temperature: float = 1.0
    router_gumbel_hard: bool = False
    router_normalize_input: bool = True
    fine_input_mode: str = "concat"  # "concat" | "add"

    def __post_init__(self) -> None:
        if self.fine_input_mode not in ("concat", "add"):
            raise ValueError(
                f"fine_input_mode must be 'concat' or 'add', got '{self.fine_input_mode}'"
            )
        if self.num_subspaces < 2:
            raise ValueError(f"num_subspaces must be >= 2, got {self.num_subspaces}")
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
    def from_dict(cls, d: dict) -> "DecomposedJASPERConfig":
        """Create config from a plain dictionary (e.g. loaded from YAML)."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# DecomposedJASPERPredictor
# ---------------------------------------------------------------------------


class DecomposedJASPERPredictor(nn.Module):
    """Two-stage JASPER predictor with explicit subspace routing.

    Architecture
    ------------
    ::

        question_emb [B,D] ──► question_encoder ──► q_latent [B,H]  ──┐
                                                                        ├─► fine_mlp ──► fine [B,D]
        steering_emb [B,D] ──► steering_encoder ──► s_latent [B,H]  ──┤        ┌─────────────┘
                                                                        │        │
        centroid_embs [K,D] ◄── routing_weights [B,K] ◄── router ◄────┘  coarse[B,D]
                │
                └─► coarse = routing_weights @ centroid_embs [B,D]
                                                         │
                                    prediction = coarse + fine [B,D]

    The **coarse** vector is a soft weighted sum of concept centroids, giving
    the prediction a subspace-anchored starting point.  The **fine** (residual)
    vector refines it using the full question and steering context.

    ``atypicality = ‖fine‖`` measures how much the sample deviates from its
    assigned subspace.

    Args:
        config: Model configuration.  Can also be passed as a plain ``dict``.
    """

    def __init__(self, config: DecomposedJASPERConfig | dict) -> None:
        super().__init__()

        if isinstance(config, dict):
            config = DecomposedJASPERConfig.from_dict(config)
        self.config = config

        D = config.embedding_dim
        H = config.hidden_dim
        K = config.num_subspaces
        n = config.num_layers
        p = config.dropout
        act = config.activation
        ln = config.use_layer_norm

        # --- Encoders (same architecture as JASPERPredictor) ---
        self.question_encoder = _make_mlp(D, H, H, n, p, act, ln)
        self.steering_encoder = _make_mlp(D, H, H, n, p, act, ln)

        # --- Subspace router ---
        router_cfg = SubspaceRouterConfig(
            embedding_dim=D,
            hidden_dim=config.router_hidden_dim,
            num_subspaces=K,
            num_layers=n,
            dropout=p,
            activation=act,
            use_layer_norm=ln,
            temperature=config.router_temperature,
            gumbel_hard=config.router_gumbel_hard,
            normalize_input=config.router_normalize_input,
        )
        self.router = SubspaceRouter(router_cfg)

        # --- Fine (residual) MLP ---
        if config.fine_input_mode == "concat":
            # Input: [q_latent; s_latent; coarse] = [B, 2H+D]
            fine_in_dim = 2 * H + D
            self.coarse_projector: Optional[nn.Linear] = None
        else:
            # Input: q_latent + s_latent + projected(coarse) = [B, H]
            fine_in_dim = H
            self.coarse_projector = nn.Linear(D, H, bias=True)

        self.fine_mlp = _make_mlp(fine_in_dim, H, D, n, p, act, ln)

        self._normalize_output = config.normalize_output
        self._latents: Dict[str, torch.Tensor] = {}

        self._init_weights()
        LOGGER.debug(
            "DecomposedJASPERPredictor initialised: D=%d H=%d K=%d mode=%s params=%d",
            D,
            H,
            K,
            config.fine_input_mode,
            sum(p_.numel() for p_ in self.parameters()),
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
        centroid_embs: torch.Tensor,
        training: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Predict source embedding via subspace routing + residual refinement.

        Args:
            question_emb: ``[B, D]`` question embedding.
            steering_emb: ``[B, D]`` steering embedding.
            centroid_embs: ``[K, D]`` cluster centroid embeddings.
                K must equal ``config.num_subspaces``.
            training: If ``None`` (default), uses ``self.training``.
                Pass ``True`` to force Gumbel-Softmax or ``False`` for Softmax.

        Returns:
            Tuple of:
                - ``prediction`` ``[B, D]``: predicted source embedding.
                - ``explanation_dict``: diagnostic tensors (all **detached**):
                    - ``"routing_weights"`` ``[B, K]``: soft routing distribution.
                    - ``"concept_logits"``  ``[B, K]``: raw router logits.
                    - ``"coarse"``          ``[B, D]``: centroid-anchored estimate.
                    - ``"fine"``            ``[B, D]``: residual correction.
                    - ``"atypicality"``     ``[B]``:   ``‖fine‖`` per sample.
        """
        K = centroid_embs.shape[0]
        if K != self.config.num_subspaces:
            raise ValueError(
                f"centroid_embs has {K} rows but config.num_subspaces={self.config.num_subspaces}. "
                "These must match."
            )

        use_training = self.training if training is None else training

        # --- Encode question and steering ---
        q_latent = self.question_encoder(question_emb)  # [B, H]
        s_latent = self.steering_encoder(steering_emb)  # [B, H]

        # --- Route ---
        routing_weights, concept_logits = self.router(
            question_emb, steering_emb, training=use_training
        )  # [B, K], [B, K]

        # --- Coarse: weighted centroid sum ---
        coarse = routing_weights @ centroid_embs  # [B, D]

        # --- Fine: residual MLP ---
        if self.config.fine_input_mode == "concat":
            fine_in = torch.cat([q_latent, s_latent, coarse], dim=-1)  # [B, 2H+D]
        else:
            coarse_proj = self.coarse_projector(coarse)  # [B, H]
            fine_in = q_latent + s_latent + coarse_proj  # [B, H]

        fine = self.fine_mlp(fine_in)  # [B, D]

        # --- Prediction ---
        prediction = coarse + fine  # [B, D]

        if self._normalize_output:
            prediction = F.normalize(prediction, dim=-1)

        # --- Atypicality ---
        atypicality = fine.detach().norm(dim=-1)  # [B]

        # --- Cache for introspection ---
        self._latents = {
            "question_latent": q_latent.detach(),
            "steering_latent": s_latent.detach(),
            "routing_weights": routing_weights.detach(),
            "coarse": coarse.detach(),
            "fine": fine.detach(),
            "atypicality": atypicality,
        }

        explanation_dict = {
            "routing_weights": routing_weights.detach(),
            "concept_logits": concept_logits.detach(),
            "coarse": coarse.detach(),
            "fine": fine.detach(),
            "atypicality": atypicality,
        }

        return prediction, explanation_dict

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_routing_weights(
        self,
        question_emb: torch.Tensor,
        steering_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Return soft routing weights ``[B, K]`` under the current training mode.

        Args:
            question_emb: ``[B, D]``
            steering_emb: ``[B, D]``

        Returns:
            Routing probabilities ``[B, K]`` summing to 1 along dim=-1.
        """
        weights, _ = self.router(question_emb, steering_emb, training=self.training)
        return weights

    def get_latent_representations(self) -> Dict[str, torch.Tensor]:
        """Return cached intermediate activations from the last forward pass.

        All tensors are detached and moved to CPU.

        Returns:
            Dict with keys: ``"question_latent"``, ``"steering_latent"``,
            ``"routing_weights"``, ``"coarse"``, ``"fine"``, ``"atypicality"``.
            Empty if no forward pass has run yet.
        """
        return {k: v.cpu() for k, v in self._latents.items()}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def embedding_dim(self) -> int:
        """Input/output embedding dimension D."""
        return self.config.embedding_dim

    @property
    def hidden_dim(self) -> int:
        """Hidden layer width H."""
        return self.config.hidden_dim

    @property
    def num_subspaces(self) -> int:
        """Number of routing subspaces K."""
        return self.config.num_subspaces

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_model_summary(self) -> str:
        """Return a human-readable parameter summary."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"DecomposedJASPERPredictor | D={self.config.embedding_dim} "
            f"H={self.config.hidden_dim} K={self.config.num_subspaces} "
            f"mode={self.config.fine_input_mode} | "
            f"total params: {total:,} | trainable: {trainable:,}"
        )

    def __repr__(self) -> str:
        return (
            f"DecomposedJASPERPredictor(embedding_dim={self.config.embedding_dim}, "
            f"hidden_dim={self.config.hidden_dim}, "
            f"num_subspaces={self.config.num_subspaces}, "
            f"fine_input_mode='{self.config.fine_input_mode}')"
        )
