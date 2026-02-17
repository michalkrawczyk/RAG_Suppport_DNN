# Subspace-Routed JASPER Architecture

## Overview

`DecomposedJASPERPredictor` extends the base `JASPERPredictor` with an explicit
subspace routing mechanism.  Each prediction is decomposed into:

- **Coarse vector**: a soft-weighted centroid selected by the router.
- **Fine vector**: a learned residual MLP that refines the coarse estimate.
- **Prediction**: `coarse + fine`.

```
question_emb  [B,D] ──► question_encoder  ──► q_latent [B,H]  ──┐
                                                                   ├─► fine_mlp ──► fine [B,D]
steering_emb  [B,D] ──► steering_encoder  ──► s_latent [B,H]  ──┤       ┌───────────────┘
                                                                   │       │
                        ┌── SubspaceRouter ◄──────────────────────┘  coarse[B,D]
                        │   [B,D]+[B,D]→[B,K] Gumbel-Softmax
                        │
centroid_embs [K,D] ◄───┘ routing_weights [B,K]
      │
      └─► coarse = routing_weights @ centroid_embs  [B,D]

                   prediction = coarse + fine  [B,D]
```

| Symbol | Meaning |
|--------|---------|
| B | Batch size |
| D | Embedding dimension |
| H | Hidden dimension |
| K | Number of subspaces |

---

## SubspaceRouter

`SubspaceRouter` decides which concept subspace(s) are relevant for a given
question+steering pair.

### Router math

1. Concatenate inputs: `x = concat([question_emb, steering_emb])` → `[B, 2D]`.
2. Optionally L2-normalise `x`.
3. Pass through a configurable MLP → `concept_logits [B, K]`.
4. **During training**: `routing_weights = GumbelSoftmax(concept_logits, τ, hard)`.
5. **During inference**: `routing_weights = Softmax(concept_logits / τ)`.

### Gumbel-Softmax

Gumbel-Softmax provides a continuous (differentiable) relaxation of discrete
routing.  The temperature τ controls the sharpness:

| τ | Behaviour |
|---|-----------|
| High (e.g. 2.0) | Soft, nearly uniform weights — good early in training |
| Low  (e.g. 0.1) | Sharp, near-one-hot weights — closer to hard routing |

The `temperature` config field sets the base value; it can be annealed externally
from the training script.

With `gumbel_hard=True`, straight-through estimation is used: the forward pass
produces hard one-hot weights, but gradients flow through the soft weights.

---

## `DecomposedJASPERConfig`

```python
from RAG_supporters.nn.models.decomposed_predictor import DecomposedJASPERPredictor, DecomposedJASPERConfig

cfg = DecomposedJASPERConfig(
    embedding_dim=768,        # D — must match your embedding model
    hidden_dim=512,           # H for encoders and fine MLP
    num_subspaces=8,          # K — must match number of centroid_embs rows
    num_layers=3,
    dropout=0.1,
    activation="GELU",
    use_layer_norm=True,
    normalize_output=False,
    router_hidden_dim=256,    # separate width for SubspaceRouter
    router_temperature=1.0,
    router_gumbel_hard=False,
    router_normalize_input=True,
    fine_input_mode="concat", # "concat" | "add"
)
model = DecomposedJASPERPredictor(cfg)
# or from dict / YAML:
model = DecomposedJASPERPredictor(yaml_config["model"])
```

### `fine_input_mode`

| Mode | Fine MLP input | Input size |
|------|---------------|------------|
| `"concat"` (default) | `[q_latent; s_latent; coarse]` | `2H + D` |
| `"add"` | `q_latent + s_latent + coarse_projector(coarse)` | `H` |

`"concat"` gives the fine MLP full context at the cost of a larger input.
`"add"` is more parameter-efficient.

---

## Usage

### Forward pass

```python
import torch
from RAG_supporters.nn.models.decomposed_predictor import DecomposedJASPERPredictor, DecomposedJASPERConfig

model = DecomposedJASPERPredictor(DecomposedJASPERConfig(embedding_dim=768, num_subspaces=8))

B, D, K = 32, 768, 8
question_emb  = torch.randn(B, D)
steering_emb  = torch.randn(B, D)
centroid_embs = torch.randn(K, D)   # provided at runtime — not stored in model

prediction, xai = model(question_emb, steering_emb, centroid_embs)
# prediction: [B, D]
# xai dict (all detached):
#   "routing_weights" [B, K]
#   "concept_logits"  [B, K]
#   "coarse"          [B, D]
#   "fine"            [B, D]
#   "atypicality"     [B]     — ||fine||, a per-sample deviation measure
```

### Inspecting routing

```python
# Which subspace is primary for each sample?
cluster_ids, confidences = model.router.get_primary_subspace(question_emb, steering_emb)
# cluster_ids: [B] long   — argmax of routing weights
# confidences: [B] float  — max routing weight value
```

### Routing explanation (single sample)

```python
cluster_names = ["factual", "procedural", "causal", "analogical", ...]
explanation = model.router.explain(
    question_emb[0:1], steering_emb[0:1], cluster_names
)
# {
#   "routing_weights": [0.72, 0.09, 0.12, 0.07, ...],
#   "primary_subspace": "factual",
#   "primary_confidence": 0.72,
#   "entropy": 0.83,
#   "cluster_names": ["factual", ...]
# }
```

---

## Routing Losses

Four losses supplement the base JASPER multi-objective loss:

| Loss | Class | Purpose |
|------|-------|---------|
| Routing XEnt | `RoutingLoss` | Supervise router to assign correct cluster |
| Entropy annealing | `EntropyRegularization` | Diversity early → confidence late |
| Residual hinge | `ResidualPenalty` | Keep `‖fine‖` below a margin |
| Axis decorrelation | `DisentanglementLoss` | Decorrelate routing dimensions |

```python
from RAG_supporters.nn.losses.routing_losses import (
    RoutingLoss, EntropyRegularization, ResidualPenalty, DisentanglementLoss
)

routing_loss  = RoutingLoss(weight=1.0)
entropy_reg   = EntropyRegularization(entropy_low=0.1, anneal_epochs=20, weight=0.1)
residual_pen  = ResidualPenalty(margin=1.0, weight=0.1)
disentangle   = DisentanglementLoss(weight=0.01)

# In training step:
routing_dict  = routing_loss(xai["concept_logits"], batch["cluster_id"])
entropy_dict  = entropy_reg(xai["routing_weights"], current_epoch)
residual_dict = residual_pen(xai["fine"])
dis_dict      = disentangle(xai["routing_weights"])

extra_loss = (
    routing_dict["routing"]
    + entropy_dict["entropy_reg"]
    + residual_dict["residual_penalty"]
    + dis_dict["disentanglement"]
)
```

### Entropy annealing schedule

```
target_entropy(epoch) = entropy_high + (entropy_low - entropy_high) × min(1, epoch / anneal_epochs)

epoch 0:              target ≈ log(K)   (uniform → explore)
epoch anneal_epochs:  target = 0.1      (confident → exploit)
```

Loss: `weight × (H(routing_weights) − target_entropy)²`

---

## Training Script

```bash
python examples/train_subspace_jasper.py \
    --config configs/subspace_jasper.yaml \
    --dataset-dir /path/to/jasper_dataset \
    --output-dir runs/subspace_run_01 \
    --centroids-path /path/to/centroids.pt
```

Resume:

```bash
python examples/train_subspace_jasper.py \
    --config configs/subspace_jasper.yaml \
    --dataset-dir /path/to/jasper_dataset \
    --output-dir runs/subspace_run_01 \
    --centroids-path /path/to/centroids.pt \
    --resume runs/subspace_run_01/checkpoints/best.pt
```

---

## Config Reference (`configs/subspace_jasper.yaml`)

New sections beyond `jasper_base.yaml`:

```yaml
router:
  hidden_dim: 256          # Router MLP hidden width
  num_subspaces: 8         # K — must match centroid file
  temperature: 1.0         # Gumbel-Softmax temperature
  gumbel_hard: false       # true → hard one-hot routing during training
  normalize_input: true
  fine_input_mode: concat  # "concat" | "add"

routing_loss:
  lambda_routing: 1.0
  lambda_entropy: 0.1
  lambda_residual: 0.1
  lambda_disentangle: 0.01
  entropy_low: 0.1
  anneal_epochs: 20
  residual_margin: 1.0     # Set to ~median(||target_emb||)

xai:
  save_val_xai: true
  xai_every_n_epochs: 5
  cluster_names_file: null # JSON/txt with K names; null → "subspace_0", ...
```

---

## Key Signals to Monitor

| Metric | Healthy Range | Problem |
|--------|--------------|---------|
| `train/routing_acc` | Increasing above `1/K` | Flat → routing not learning |
| `train/routing_entropy` | Decreases from `log(K)` to ~`0.1-0.5` | Always flat → entropy reg not working |
| `train/residual_norm_mean` | < `residual_margin` | > margin → model ignoring coarse |
| `train/disentanglement` | < 0.1 | High → routing axes are correlated |
| `train/atypicality` (mean) | Bounded, not exploding | Growing → residual penalty too low |

---

## Key Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Router collapse (all → one subspace) | `EntropyRegularization`; balanced cluster sampling |
| Residual dominance (ignores coarse) | `ResidualPenalty`; lower `residual_margin` |
| Gumbel noise destabilises training | Increase temperature; switch to `gumbel_hard=False` |
| Routing accuracy stays at `1/K` | Check cluster_id labels in dataset; increase `lambda_routing` |
