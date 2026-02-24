"""XAIInterface: explainability wrapper for JASPERPredictor and DecomposedJASPERPredictor."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)

# Optional matplotlib
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False


def _json_default(obj: Any) -> Any:
    """JSON serialiser for numpy/torch scalars."""
    try:
        import numpy as np

        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    try:
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
    except Exception:
        pass
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


class XAIInterface:
    """Explainability interface for JASPER models.

    Works with both :class:`~RAG_supporters.nn.models.decomposed_predictor.DecomposedJASPERPredictor`
    (full XAI) and :class:`~RAG_supporters.nn.models.jasper_predictor.JASPERPredictor`
    (limited XAI via proxy routing from centroid similarity).

    Args:
        model: A ``DecomposedJASPERPredictor`` or ``JASPERPredictor`` instance.
        centroid_embs: Cluster centroid embeddings ``[K, D]``.
        cluster_names: K human-readable subspace names.
        training_pairs: Optional list of ``(question_emb [D], steering_emb [D],
            source_emb [D])`` tuples used for nearest-neighbour retrieval in
            :meth:`explain_prediction`.
        device: Torch device.  ``None`` → auto-detect.
    """

    def __init__(
        self,
        model: nn.Module,
        centroid_embs: torch.Tensor,
        cluster_names: List[str],
        training_pairs: Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        K = centroid_embs.shape[0]
        if len(cluster_names) != K:
            raise ValueError(
                f"len(cluster_names)={len(cluster_names)} must equal " f"centroid_embs.shape[0]={K}"
            )

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.centroid_embs = centroid_embs.to(self.device)
        self.cluster_names = cluster_names

        # Determine model type
        try:
            from RAG_supporters.nn.models.decomposed_predictor import DecomposedJASPERPredictor

            self._is_decomposed = isinstance(model, DecomposedJASPERPredictor)
        except ImportError:
            self._is_decomposed = False

        # Stacked training pairs for nearest-neighbour lookup
        self._train_questions: Optional[torch.Tensor] = None
        self._train_steerings: Optional[torch.Tensor] = None
        self._train_sources: Optional[torch.Tensor] = None

        if training_pairs:
            qs = torch.stack([p[0] for p in training_pairs]).to(self.device)
            ss = torch.stack([p[1] for p in training_pairs]).to(self.device)
            srcs = torch.stack([p[2] for p in training_pairs]).to(self.device)
            self._train_questions = qs
            self._train_steerings = ss
            self._train_sources = srcs

        LOGGER.debug(
            "XAIInterface ready: model=%s K=%d training_pairs=%d",
            type(model).__name__,
            K,
            len(training_pairs) if training_pairs else 0,
        )

    # ------------------------------------------------------------------
    # Primary explanation
    # ------------------------------------------------------------------

    def explain_prediction(
        self,
        question_emb: torch.Tensor,
        steering_emb: torch.Tensor,
    ) -> Dict[str, Any]:
        """Produce a full explanation for one question/steering pair.

        Args:
            question_emb: ``[D]`` or ``[1, D]``.
            steering_emb: ``[D]`` or ``[1, D]``.

        Returns:
            Dict with keys:

            - ``"prediction"`` — predicted source embedding as a Python list.
            - ``"primary_subspace"`` — name of the dominant routing cluster.
            - ``"primary_confidence"`` — float confidence of the dominant cluster.
            - ``"routing_distribution"`` — ``{cluster_name: float}`` dict.
            - ``"routing_entropy"`` — routing entropy in nats.
            - ``"atypicality"`` — ‖fine‖ (or proxy for JASPERPredictor).
            - ``"coarse_vector"`` — ``[D]`` list (``None`` for JASPERPredictor).
            - ``"fine_vector"`` — ``[D]`` list (``None`` for JASPERPredictor).
            - ``"steering_influence"`` — KL divergence measuring how much the
              steering vector shifts the routing.
            - ``"similar_pairs"`` — up to 5 nearest training pairs (list of dicts).
            - ``"actionable_signal"`` — human-readable summary string.
        """
        # Ensure [1, D]
        q = question_emb.view(1, -1).to(self.device)
        s = steering_emb.view(1, -1).to(self.device)

        with torch.no_grad():
            if self._is_decomposed:
                result = self._explain_decomposed(q, s)
            else:
                result = self._explain_jasper(q, s)

        # Nearest-neighbour similar pairs
        result["similar_pairs"] = self._find_similar_pairs(q, s, k=5)

        # Actionable signal
        result["actionable_signal"] = self._build_actionable_signal(result)

        return result

    # ------------------------------------------------------------------
    # Steering influence comparison
    # ------------------------------------------------------------------

    def compare_steering_influence(
        self,
        question_emb: torch.Tensor,
        steering_embs: List[torch.Tensor],
        labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compare how different steering vectors affect routing and prediction.

        Args:
            question_emb: ``[D]`` single question embedding.
            steering_embs: List of N steering embeddings, each ``[D]``.
            labels: Optional list of N human-readable labels.

        Returns:
            Dict with:
                - ``"labels"``: list of N labels.
                - ``"routing_matrices"``: list of N routing distributions (list of K floats).
                - ``"predictions"``: list of N predicted embeddings (as Python lists).
                - ``"atypicalities"``: list of N floats.
                - ``"routing_kl_from_first"``: KL(steer_i ‖ steer_0) for i in 1..N-1.
        """
        q = question_emb.view(1, -1).to(self.device)
        labels = labels or [f"steering_{i}" for i in range(len(steering_embs))]

        routing_matrices: List[List[float]] = []
        predictions: List[List[float]] = []
        atypicalities: List[float] = []

        for s_emb in steering_embs:
            s = s_emb.view(1, -1).to(self.device)
            xai = self.explain_prediction(q.squeeze(0), s.squeeze(0))
            routing_matrices.append(list(xai["routing_distribution"].values()))
            predictions.append(xai["prediction"])
            atypicalities.append(xai["atypicality"])

        # KL divergence relative to first steering
        kl_from_first: List[float] = []
        if len(routing_matrices) > 1:
            ref = torch.tensor(routing_matrices[0], dtype=torch.float32)
            for rm in routing_matrices[1:]:
                cmp = torch.tensor(rm, dtype=torch.float32)
                kl = F.kl_div(
                    ref.clamp(min=1e-8).log(),
                    cmp.clamp(min=1e-8),
                    reduction="sum",
                ).item()
                kl_from_first.append(kl)

        return {
            "labels": labels,
            "routing_matrices": routing_matrices,
            "predictions": predictions,
            "atypicalities": atypicalities,
            "routing_kl_from_first": kl_from_first,
        }

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def visualize_explanation(
        self,
        xai_dict: Dict[str, Any],
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> Optional[Any]:
        """Plot a three-panel explanation figure.

        Panels:
        1. Routing distribution (bar chart, primary subspace highlighted).
        2. Coarse vs. Fine magnitude (bar or single-value indicator).
        3. Steering influence (KL divergence gauge).

        Args:
            xai_dict: Output of :meth:`explain_prediction`.
            title: Optional figure title.
            save_path: If given, saves the figure to this path.

        Returns:
            ``matplotlib.figure.Figure`` or ``None`` if matplotlib is unavailable.
        """
        if not _MATPLOTLIB_AVAILABLE:
            LOGGER.warning("matplotlib not available; skipping visualize_explanation.")
            return None

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # --- Panel 1: Routing distribution ---
        ax = axes[0]
        routing_dist = xai_dict.get("routing_distribution", {})
        names = list(routing_dist.keys())
        weights = list(routing_dist.values())
        primary = xai_dict.get("primary_subspace", "")
        colors = ["#e05c5c" if n == primary else "#5c9be0" for n in names]
        ax.bar(range(len(names)), weights, color=colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Routing weight")
        ax.set_title("Routing Distribution")
        ax.set_ylim(0, 1)
        ax.axhline(
            1.0 / max(len(names), 1), color="gray", linestyle="--", alpha=0.5, label="uniform"
        )
        ax.legend(fontsize=7)
        ax.grid(True, axis="y", alpha=0.3)

        # --- Panel 2: Coarse / Fine magnitude ---
        ax = axes[1]
        atypicality = xai_dict.get("atypicality", 0.0)
        coarse_vec = xai_dict.get("coarse_vector")
        if coarse_vec is not None:
            import numpy as np

            coarse_norm = float(np.linalg.norm(coarse_vec))
            ax.bar(
                ["Coarse ‖c‖", "Fine ‖f‖ (atypicality)"],
                [coarse_norm, atypicality],
                color=["#5cc85c", "#e05c5c"],
            )
            ax.set_ylabel("L2 norm")
        else:
            ax.bar(["Atypicality"], [atypicality], color=["#e05c5c"])
            ax.set_ylabel("L2 norm")
        ax.set_title("Coarse vs. Fine Magnitude")
        ax.grid(True, axis="y", alpha=0.3)

        # --- Panel 3: Steering influence ---
        ax = axes[2]
        kl = xai_dict.get("steering_influence", 0.0)
        ax.barh(["Steering\ninfluence (KL)"], [kl], color="#c85cc8")
        ax.set_xlabel("KL divergence (nats)")
        ax.set_title("Steering Influence")
        ax.grid(True, axis="x", alpha=0.3)

        if title:
            fig.suptitle(title, fontsize=12)

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            LOGGER.info("XAI figure saved to %s", save_path)

        return fig

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save_xai_outputs(
        self,
        xai_results: List[Dict[str, Any]],
        output_path: str,
    ) -> None:
        """Serialise a list of XAI explanation dicts to JSON.

        Args:
            xai_results: List of :meth:`explain_prediction` outputs.
            output_path: Path for the JSON file.
        """
        output_path = str(output_path)
        with open(output_path, "w") as f:
            json.dump(xai_results, f, indent=2, default=_json_default)
        LOGGER.info("XAI outputs saved to %s (%d items)", output_path, len(xai_results))

    # ------------------------------------------------------------------
    # Internal helpers — DecomposedJASPERPredictor path
    # ------------------------------------------------------------------

    def _explain_decomposed(self, q: torch.Tensor, s: torch.Tensor) -> Dict[str, Any]:
        """Full XAI for DecomposedJASPERPredictor."""
        prediction, xai = self.model(q, s, self.centroid_embs, training=False)

        routing_weights = xai["routing_weights"].squeeze(0)  # [K]
        routing_dist = {
            name: float(w) for name, w in zip(self.cluster_names, routing_weights.tolist())
        }
        primary_idx = routing_weights.argmax().item()
        entropy = -(routing_weights * (routing_weights + 1e-8).log()).sum().item()

        # Steering influence via KL(w_without ‖ w_with)
        steering_influence = self._compute_steering_influence_decomposed(q, s)

        return {
            "prediction": prediction.squeeze(0).cpu().tolist(),
            "primary_subspace": self.cluster_names[primary_idx],
            "primary_confidence": float(routing_weights[primary_idx]),
            "routing_distribution": routing_dist,
            "routing_entropy": entropy,
            "atypicality": float(xai["atypicality"].squeeze(0)),
            "coarse_vector": xai["coarse"].squeeze(0).cpu().tolist(),
            "fine_vector": xai["fine"].squeeze(0).cpu().tolist(),
            "steering_influence": steering_influence,
        }

    def _compute_steering_influence_decomposed(self, q: torch.Tensor, s: torch.Tensor) -> float:
        """KL divergence measuring how much the steering changes the routing."""
        s_zero = torch.zeros_like(s)
        # router.forward returns (routing_weights, concept_logits)
        p_with, _ = self.model.router(q, s, training=False)  # [1, K]
        p_without, _ = self.model.router(q, s_zero, training=False)  # [1, K]
        p_with = p_with.squeeze(0)
        p_without = p_without.squeeze(0)
        kl = F.kl_div(
            p_without.clamp(min=1e-8).log(),
            p_with.clamp(min=1e-8),
            reduction="sum",
        ).item()
        return kl

    # ------------------------------------------------------------------
    # Internal helpers — JASPERPredictor path (limited XAI)
    # ------------------------------------------------------------------

    def _explain_jasper(self, q: torch.Tensor, s: torch.Tensor) -> Dict[str, Any]:
        """Limited XAI for base JASPERPredictor via proxy routing."""
        prediction = self.model(q, s)  # [1, D]
        pred = prediction.squeeze(0)  # [D]

        # Proxy routing: cosine similarity between prediction and each centroid
        pred_norm = F.normalize(pred.unsqueeze(0), dim=-1)  # [1, D]
        cent_norm = F.normalize(self.centroid_embs, dim=-1)  # [K, D]
        sims = (pred_norm @ cent_norm.T).squeeze(0)  # [K]
        routing_weights = F.softmax(sims, dim=-1)  # [K] proxy

        routing_dist = {
            name: float(w) for name, w in zip(self.cluster_names, routing_weights.tolist())
        }
        primary_idx = routing_weights.argmax().item()
        entropy = -(routing_weights * (routing_weights + 1e-8).log()).sum().item()

        # Proxy atypicality: distance to the nearest centroid
        nearest_centroid = self.centroid_embs[primary_idx]
        atypicality = float((pred - nearest_centroid).norm().item())

        # Steering influence: difference in nearest centroid without steering
        steering_influence = self._compute_steering_influence_jasper(q, s)

        return {
            "prediction": pred.cpu().tolist(),
            "primary_subspace": self.cluster_names[primary_idx],
            "primary_confidence": float(routing_weights[primary_idx]),
            "routing_distribution": routing_dist,
            "routing_entropy": entropy,
            "atypicality": atypicality,
            "coarse_vector": None,
            "fine_vector": None,
            "steering_influence": steering_influence,
        }

    def _compute_steering_influence_jasper(self, q: torch.Tensor, s: torch.Tensor) -> float:
        """Proxy steering influence for JASPERPredictor."""
        s_zero = torch.zeros_like(s)
        pred_with = self.model(q, s).squeeze(0)
        pred_without = self.model(q, s_zero).squeeze(0)

        cent_norm = F.normalize(self.centroid_embs, dim=-1)

        def proxy_routing(pred: torch.Tensor) -> torch.Tensor:
            p_norm = F.normalize(pred.unsqueeze(0), dim=-1)
            sims = (p_norm @ cent_norm.T).squeeze(0)
            return F.softmax(sims, dim=-1)

        w_with = proxy_routing(pred_with)
        w_without = proxy_routing(pred_without)
        kl = F.kl_div(
            w_without.clamp(min=1e-8).log(),
            w_with.clamp(min=1e-8),
            reduction="sum",
        ).item()
        return kl

    # ------------------------------------------------------------------
    # Nearest-neighbour lookup
    # ------------------------------------------------------------------

    def _find_similar_pairs(
        self,
        q: torch.Tensor,
        s: torch.Tensor,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Find training pairs similar to this query by cosine similarity."""
        if self._train_questions is None:
            return []

        q_norm = F.normalize(q, dim=-1)  # [1, D]
        train_q_norm = F.normalize(self._train_questions, dim=-1)  # [N, D]

        sims = (train_q_norm @ q_norm.T).squeeze(-1)  # [N]
        topk_vals, topk_idx = sims.topk(min(k, len(sims)))

        results = []
        for sim_val, idx in zip(topk_vals.tolist(), topk_idx.tolist()):
            results.append(
                {
                    "similarity": round(sim_val, 4),
                    "index": idx,
                }
            )
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_actionable_signal(self, xai_dict: Dict[str, Any]) -> str:
        """Build a concise human-readable summary from an XAI dict."""
        primary = xai_dict.get("primary_subspace", "unknown")
        confidence = xai_dict.get("primary_confidence", 0.0)
        atypicality = xai_dict.get("atypicality", 0.0)
        steering_kl = xai_dict.get("steering_influence", 0.0)

        confidence_str = f"{confidence:.0%}"
        parts = [f"Routed to '{primary}' with {confidence_str} confidence."]

        if atypicality > 2.0:
            parts.append(
                f"High atypicality (‖fine‖={atypicality:.2f}): "
                "sample deviates significantly from its assigned subspace."
            )
        elif atypicality < 0.5:
            parts.append(
                f"Low atypicality (‖fine‖={atypicality:.2f}): "
                "sample is well-represented by its subspace centroid."
            )

        if steering_kl > 1.0:
            parts.append(
                f"Steering has high influence (KL={steering_kl:.2f}): "
                "the steering signal substantially changed the routing."
            )
        elif steering_kl < 0.05:
            parts.append(
                f"Steering has low influence (KL={steering_kl:.2f}): "
                "routing is primarily driven by the question, not the steering."
            )

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        model_type = "decomposed" if self._is_decomposed else "base"
        return (
            f"XAIInterface(model={type(self.model).__name__} [{model_type}], "
            f"K={len(self.cluster_names)}, "
            f"training_pairs={self._train_questions.shape[0] if self._train_questions is not None else 0})"
        )
