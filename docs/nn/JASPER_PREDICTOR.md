# JASPER Predictor Architecture

## Overview

`JASPERPredictor` implements the core prediction step of the JASPER
(Joint Architecture for Subspace Prediction with Explainable Routing)
model.  Given a question embedding and a steering signal, it predicts
the source embedding that a retrieval system should target.

```
question_emb  [B, D] ──► question_encoder  [B, H]  ──┐
                                                        ├─ concat [B, 2H] ──► predictor_head ──► [B, D]
steering_emb  [B, D] ──► steering_encoder  [B, H]  ──┘
```

| Symbol | Meaning |
|--------|---------|
| B | Batch size |
| D | Embedding dimension (same for input and output) |
| H | Hidden dimension |

---

## EMA Target Encoder

Training uses an **Exponential Moving Average (EMA)** target encoder to
prevent representation collapse — a common failure mode in self-supervised
architectures where the model learns to output a constant vector.

```
online_encoder  ←── gradient descent  (updated each step)
target_encoder  ←── EMA momentum      (no backprop)
                     θ_t ← τ·θ_t + (1-τ)·θ_online
```

The momentum coefficient **τ** (tau) is cosine-annealed from `tau_min` to
`tau_max` over training, so the target starts updating faster and gradually
slows to a near-freeze.

```
τ(step) = tau_max - (tau_max - tau_min) × 0.5 × (1 + cos(π × progress))
```

Typical values: `tau_min = 0.996`, `tau_max = 0.999`.

---

## Configuration

### `JASPERPredictorConfig`

```python
from RAG_supporters.nn.models.jasper_predictor import JASPERPredictorConfig

cfg = JASPERPredictorConfig(
    embedding_dim=768,      # D — must match your embedding model output
    hidden_dim=512,         # H
    num_layers=3,           # depth of each sub-network (≥1)
    dropout=0.1,            # 0.0 = disabled
    activation="GELU",      # any torch.nn activation class name
    use_layer_norm=True,    # LayerNorm after each hidden activation
    normalize_output=False, # L2-normalise output (useful with cosine losses)
)
```

Or load from a dict / YAML:

```python
cfg = JASPERPredictorConfig.from_dict(yaml_config["model"])
model = JASPERPredictor(cfg)
# or pass the dict directly:
model = JASPERPredictor(yaml_config["model"])
```

### Parameter Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embedding_dim` | 768 | Input/output embedding size (D) |
| `hidden_dim` | 512 | Hidden layer width (H) |
| `num_layers` | 2 | MLP depth — 1 = single linear; ≥2 = hidden layers + output |
| `dropout` | 0.1 | Dropout after each hidden activation |
| `activation` | `"GELU"` | `torch.nn` activation class |
| `use_layer_norm` | `True` | LayerNorm after hidden activations |
| `normalize_output` | `False` | L2-normalize the output vector |

---

## Usage Examples

### Basic forward pass

```python
import torch
from RAG_supporters.nn.models.jasper_predictor import JASPERPredictor, JASPERPredictorConfig

model = JASPERPredictor(JASPERPredictorConfig(embedding_dim=768, hidden_dim=512))

B, D = 32, 768
question_emb = torch.randn(B, D)
steering_emb = torch.randn(B, D)

predicted_source_emb = model(question_emb, steering_emb)  # [B, D]
print(predicted_source_emb.shape)   # torch.Size([32, 768])
```

### Inspecting latent representations

```python
model(question_emb, steering_emb)
latents = model.get_latent_representations()
# latents["question_latent"]: [B, H]  (detached, on CPU)
# latents["steering_latent"]: [B, H]
```

### With EMA encoder

```python
import copy, torch.nn as nn
from RAG_supporters.nn.models.ema_encoder import EMAEncoder

# Source encoder: maps raw source embeddings to a stable latent target
source_enc = nn.Sequential(nn.Linear(768, 768), nn.GELU(), nn.Linear(768, 768))
ema = EMAEncoder(source_enc, tau_min=0.996, tau_max=0.999)

# Training loop (simplified)
for step, batch in enumerate(train_loader):
    predicted = model(batch["question_emb"], batch["steering_emb"])
    ema_target = ema.encode_target(batch["target_source_emb"])  # no_grad

    loss = criterion(predicted, ema_target)
    loss.backward()
    optimizer.step()

    # Must call AFTER optimizer.step()
    ema.update_target(step, max_steps=total_steps)
```

### Saving and loading

```python
# Save
torch.save({
    "model": model.state_dict(),
    "ema": ema.state_dict(),
    "config": cfg.__dict__,
}, "jasper_checkpoint.pt")

# Load
ckpt = torch.load("jasper_checkpoint.pt")
cfg = JASPERPredictorConfig(**ckpt["config"])
model = JASPERPredictor(cfg)
model.load_state_dict(ckpt["model"])
ema.load_state_dict(ckpt["ema"])
```

---

## Multi-Objective Loss

Four loss components are combined during training:

| Component | Class | Purpose |
|-----------|-------|---------|
| `L_jasper` | `JASPERLoss` | SmoothL1 prediction accuracy |
| `L_contrastive` | `ContrastiveLoss` | InfoNCE — pull toward target, push from negatives |
| `L_centroid` | `CentroidLoss` | Does prediction land in the right cluster? |
| `L_vicreg` | `VICRegLoss` | Prevent embedding collapse (variance + covariance) |

```python
from RAG_supporters.nn.losses.jasper_losses import JASPERMultiObjectiveLoss

loss_fn = JASPERMultiObjectiveLoss(
    lambda_jasper=1.0,
    lambda_contrastive=0.5,
    lambda_centroid=0.1,
    lambda_vicreg=0.1,
)

total_loss, loss_dict = loss_fn(
    predicted=predicted,
    ema_target=ema_target,
    negatives=batch["negative_embs"],    # [B, K, D]
    centroid_embs=centroid_embs,          # [C, D]
    cluster_ids=batch["cluster_id"],      # [B]
)
# loss_dict keys: total, jasper, contrastive, centroid, centroid_acc,
#                 vicreg, vicreg_v, vicreg_i, vicreg_c
```

---

## Architecture Diagrams

### Full JASPER forward pass

```
Inputs
──────
question_emb  [B, D]
steering_emb  [B, D]

Encoders (separate MLPs)
────────────────────────
question_encoder:
    Linear(D→H) → GELU → LayerNorm → (× n_layers-2) → Linear(H→H)
                                                              │
steering_encoder:                                             │
    Linear(D→H) → GELU → LayerNorm → (× n_layers-2) → Linear(H→H)
                                                              │
Concatenation                                                 │
─────────────────────────────────────────────────────────────▼
    concat([q_latent, s_latent]) → [B, 2H]

Predictor head
──────────────
    Linear(2H→H) → GELU → LayerNorm → Linear(H→D) → [B, D]

Output
──────
    predicted_source_emb  [B, D]
```

### EMA update flow

```
             ┌─── backprop gradient ───┐
             ▼                         │
online_enc ──────► forward ──► loss ───┘
             │
             │  EMA momentum
             ▼
target_enc  (no grad, stable representation target)
```
