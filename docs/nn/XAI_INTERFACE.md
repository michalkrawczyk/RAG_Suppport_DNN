# XAI Interface

## Overview

`XAIInterface` provides human-interpretable explanations for JASPER model
predictions.  It works with both:

- **`DecomposedJASPERPredictor`** (full XAI): exact routing weights, coarse/fine
  decomposition, atypicality.
- **`JASPERPredictor`** (limited XAI): proxy routing from centroid similarity,
  proxy atypicality from distance to nearest centroid.

---

## Concepts

### Routing distribution
Which concept subspace(s) the model assigned this sample to.  A peaked
distribution (high `primary_confidence`) means the sample clearly belongs to
one subspace.  A flat distribution means the sample is ambiguous.

### Atypicality (`‖fine‖`)
How much the model needed to deviate from the subspace centroid.  High
atypicality signals that the sample is unusual within its assigned subspace.

### Steering influence (KL divergence)
How much the steering vector shifted the routing.  Computed as:

```
KL(routing_without_steering ‖ routing_with_steering)
```

A high value means the steering dramatically changed which subspace was chosen.
A low value means routing was driven primarily by the question content.

---

## Quick Start

```python
from RAG_supporters.nn.inference.xai_interface import XAIInterface
from RAG_supporters.nn.models.decomposed_predictor import DecomposedJASPERPredictor, DecomposedJASPERConfig
import torch

# Build model
model = DecomposedJASPERPredictor(DecomposedJASPERConfig(embedding_dim=768, num_subspaces=8))
centroid_embs = torch.load("centroids.pt")         # [K, D]
cluster_names = ["factual", "procedural", "causal", "analogical",
                 "comparative", "hypothetical", "temporal", "spatial"]

# Wrap with XAI interface
xai = XAIInterface(
    model=model,
    centroid_embs=centroid_embs,
    cluster_names=cluster_names,
)

# Explain one prediction
question_emb = torch.randn(768)
steering_emb = torch.randn(768)
result = xai.explain_prediction(question_emb, steering_emb)

print(result["primary_subspace"])      # e.g. "factual"
print(result["primary_confidence"])    # e.g. 0.73
print(result["atypicality"])           # e.g. 0.42
print(result["steering_influence"])    # e.g. 0.18
print(result["actionable_signal"])
# "Routed to 'factual' with 73% confidence. Low atypicality (‖fine‖=0.42): ..."
```

---

## `explain_prediction` Output Reference

| Key | Type | Description |
|-----|------|-------------|
| `"prediction"` | `List[float]` (len D) | Predicted source embedding |
| `"primary_subspace"` | `str` | Name of the dominant subspace |
| `"primary_confidence"` | `float` | Max routing weight (0–1) |
| `"routing_distribution"` | `Dict[str, float]` | Per-subspace weight (sums to 1) |
| `"routing_entropy"` | `float` | Shannon entropy of routing in nats |
| `"atypicality"` | `float` | ‖fine‖ (or proxy for JASPERPredictor) |
| `"coarse_vector"` | `List[float]` or `None` | Centroid-anchored estimate |
| `"fine_vector"` | `List[float]` or `None` | Residual correction |
| `"steering_influence"` | `float` | KL divergence (nats) |
| `"similar_pairs"` | `List[Dict]` | Up to K nearest training pairs |
| `"actionable_signal"` | `str` | Human-readable summary |

---

## `compare_steering_influence`

Compare how different steering vectors affect routing for the same question:

```python
steerings = [
    zero_steering,
    centroid_steering,
    keyword_steering,
    residual_steering,
]
labels = ["zero", "centroid", "keyword", "residual"]

comparison = xai.compare_steering_influence(
    question_emb=question_emb,
    steering_embs=steerings,
    labels=labels,
)

# comparison keys:
#   "labels":              ["zero", "centroid", "keyword", "residual"]
#   "routing_matrices":    [[0.72, 0.09, ...], [0.18, 0.55, ...], ...]  (one per steering)
#   "predictions":         [[...], [...], ...]
#   "atypicalities":       [0.42, 0.31, 0.56, 0.28]
#   "routing_kl_from_first": [0.83, 0.61, 0.44]  (KL from zero-steering baseline)
```

---

## Visualisation

```python
fig = xai.visualize_explanation(
    result,
    title="Question 42 — Factual Query",
    save_path="runs/xai_example.png",
)
# fig is a matplotlib.figure.Figure or None if matplotlib unavailable
```

The figure has three panels:
1. **Routing distribution** — bar chart with primary subspace highlighted in red.
2. **Coarse / Fine magnitude** — compares ‖coarse‖ and ‖fine‖ (atypicality).
3. **Steering influence** — horizontal bar showing KL divergence.

---

## Batch XAI Export

Save XAI results for an entire validation set:

```python
results = []
for batch in val_loader:
    for i in range(batch["question_emb"].shape[0]):
        result = xai.explain_prediction(
            batch["question_emb"][i],
            batch["steering"][i],
        )
        results.append(result)

xai.save_xai_outputs(results, "runs/xai/epoch_050.json")
```

The output is a JSON array where each element is one `explain_prediction` dict.

---

## Programmatic access

```python
from RAG_supporters.nn.inference.xai_interface import XAIInterface

iface = XAIInterface(
    model=model,
    centroid_embs=centroid_embs,
    cluster_names=cluster_names,
    training_pairs=training_pairs,   # optional: list of (q, s, src) for k-NN lookup
    device=torch.device("cuda"),
)
```

### `training_pairs` for nearest-neighbour retrieval

Passing `training_pairs` enables the `"similar_pairs"` field in
`explain_prediction`.  Each pair should be a tuple of
`(question_emb [D], steering_emb [D], source_emb [D])` tensors.  The interface
finds the K closest training questions (by cosine similarity) and returns their
indices and similarity scores.

```python
training_pairs = [
    (q_train[i], s_train[i], src_train[i])
    for i in range(len(q_train))
]
iface = XAIInterface(model, centroid_embs, cluster_names, training_pairs=training_pairs)
```

---

## JASPERPredictor compatibility

`XAIInterface` accepts the base `JASPERPredictor` for a limited but useful XAI
experience.  In this mode:

- `coarse_vector` and `fine_vector` are `None`.
- Routing is approximated from cosine similarities between the prediction and each centroid.
- `atypicality` is approximated as the L2 distance from the prediction to its nearest centroid.
- `steering_influence` is estimated from the shift in the proxy routing when steering is zeroed.

This allows existing Phase 1 checkpoints to benefit from XAI analysis without retraining.

---

## Troubleshooting

### `primary_confidence` always near `1/K`
The routing distribution is nearly uniform, meaning the router has not yet learned
meaningful subspace assignments.  Check:
- `routing_acc` in training logs — is it above random chance?
- Is `lambda_routing` set to a positive value?

### High atypicality for all samples
The fine MLP is overriding the coarse vector.  Increase `lambda_residual` or
lower `residual_margin`.

### `steering_influence` ≈ 0 for all samples
The router ignores steering — likely because the question signal dominates.
Try increasing `lambda_routing` or using a stronger steering signal in the dataset.
