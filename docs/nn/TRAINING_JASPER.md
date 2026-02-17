# Training JASPER Predictor

## Quick Start

```bash
python examples/train_jasper_predictor.py \
    --config configs/jasper_base.yaml \
    --dataset-dir /path/to/jasper_dataset \
    --output-dir runs/jasper_run_01
```

Resume from a checkpoint:

```bash
python examples/train_jasper_predictor.py \
    --config configs/jasper_base.yaml \
    --dataset-dir /path/to/jasper_dataset \
    --output-dir runs/jasper_run_01 \
    --resume runs/jasper_run_01/checkpoints/best.pt
```

---

## Config File Format

The YAML config is divided into six sections:

```yaml
model:
  embedding_dim: 768      # Must match your embedding model (e.g. 768 for BERT-base)
  hidden_dim: 512
  num_layers: 3
  dropout: 0.1
  activation: GELU
  use_layer_norm: true
  normalize_output: false

ema:
  tau_min: 0.996          # Starting momentum (lower = faster initial updates)
  tau_max: 0.999          # Ending momentum
  schedule: cosine        # Only 'cosine' supported

loss:
  lambda_jasper: 1.0
  lambda_contrastive: 0.5
  lambda_centroid: 0.1
  lambda_vicreg: 0.1
  jasper_beta: 1.0
  contrastive_temperature: 0.07
  centroid_temperature: 0.1
  vicreg_lambda_v: 25.0
  vicreg_lambda_i: 25.0
  vicreg_lambda_c: 1.0

training:
  batch_size: 64
  num_epochs: 50
  warmup_epochs: 2
  learning_rate: 3.0e-4
  weight_decay: 1.0e-4
  num_workers: 4
  pin_memory: true
  max_grad_norm: 1.0       # Set to 0 to disable gradient clipping
  mixed_precision: false   # Set true for CUDA AMP
  save_every_n_epochs: 5
  keep_last_n_checkpoints: 3
  log_every_n_steps: 50

dataset:
  split_train: train
  split_val: val
  n_neg: 8                 # Must match the dataset's hard-negative count

monitoring:
  use_wandb: false
  wandb_project: jasper-rag
  wandb_name: null         # Auto-generated if null
```

---

## Loss Function Tuning Guide

Start simple, then add complexity:

### Stage 1 — Prediction only (epochs 1-5)
```yaml
loss:
  lambda_jasper: 1.0
  lambda_contrastive: 0.0
  lambda_centroid: 0.0
  lambda_vicreg: 0.0
```
Verify that `train/jasper` decreases.  If it diverges, halve the learning rate.

### Stage 2 — Add contrastive (epochs 6-15)
```yaml
lambda_contrastive: 0.5
```
Watch `train/contrastive` — it should decrease slowly.  If it spikes, increase `contrastive_temperature` (e.g. `0.1 → 0.3`).

### Stage 3 — Add centroid classification (epochs 16-25)
```yaml
lambda_centroid: 0.1
```
Monitor `train/centroid_acc` — expect 30-60% early on.  Low accuracy means the model isn't routing to the right cluster; try increasing `lambda_centroid`.

### Stage 4 — Add VICReg (ongoing)
```yaml
lambda_vicreg: 0.1
```
`vicreg_v` (variance) should stay low (~0.01 for healthy embeddings).  A spike means collapse is starting — increase `lambda_vicreg`.

### Key signals

| Metric | Healthy Range | Problem |
|--------|--------------|---------|
| `train/jasper` | Decreasing | Flat → reduce LR |
| `vicreg_v` | < 0.1 | > 1.0 → collapse starting |
| `centroid_acc` | 30-80% | Near 0 → routing broken |
| `contrastive` | Decreasing | Oscillates → raise temperature |

---

## Curriculum Learning Setup

The `JASPERSteeringDataset` adjusts the mix of steering variants over epochs
automatically when `set_epoch(loader, epoch)` is called.  The trainer does
this at the start of each epoch — no configuration needed.

To inspect the current steering variant distribution, log it manually:

```python
from collections import Counter
variant_counts = Counter()
for batch in train_loader:
    for v in batch["steering_variant"].tolist():
        variant_counts[v] += 1
print(variant_counts)
# {0: 'zero', 1: 'centroid', 2: 'keyword-weighted', 3: 'residual'}
```

---

## Checkpoint Management

Checkpoints are saved to `<output_dir>/checkpoints/` and contain:

| Key | Contents |
|-----|----------|
| `model_state_dict` | JASPERPredictor weights |
| `ema_state_dict` | Both online + target encoder weights + tau config |
| `optimizer_state_dict` | AdamW state (momentum, variance accumulators) |
| `scheduler_state_dict` | LR schedule state |
| `global_step` | Total optimiser steps so far |
| `epoch` | Last completed epoch |
| `metrics` | All loss values for the saved epoch |

Special files:
- `best.pt` — lowest validation loss seen so far (overwritten each time)
- `final.pt` — state after the last epoch
- `epoch_NNNN.pt` — periodic saves (rotated, keeping `keep_last_n_checkpoints`)

### Manual checkpoint loading

```python
from RAG_supporters.nn.training.jasper_trainer import JASPERTrainer

epoch, metrics = trainer.load_checkpoint("runs/jasper/checkpoints/best.pt")
print(f"Restored from epoch {epoch}, val_loss={metrics['val/total']:.4f}")
```

---

## Monitoring with W&B

Enable by setting `monitoring.use_wandb: true` in your config.  Make sure
`wandb` is installed:

```bash
pip install wandb
wandb login
```

All metrics are synced automatically each epoch.  Plots (loss curves,
steering distribution) are uploaded as W&B images.

### Programmatic access

```python
from RAG_supporters.nn.training.monitoring import TrainingMonitor

monitor = TrainingMonitor(
    output_dir="runs/my_run",
    use_wandb=True,
    wandb_project="jasper-rag",
    wandb_name="experiment-01",
    wandb_config=cfg,
)

# Log manually
monitor.log_metrics(epoch=0, metrics_dict={"train/total": 1.23, "val/total": 1.45})

# Export at the end
monitor.export_history("runs/my_run/history.csv")
monitor.plot_losses("runs/my_run/losses.png")
monitor.finish()  # closes W&B run
```

---

## Troubleshooting

### Loss diverges (NaN / Inf)

1. Enable gradient clipping: `max_grad_norm: 1.0`
2. Reduce learning rate: try `1e-4` → `3e-5`
3. Add warmup: `warmup_epochs: 5`
4. Disable VICReg temporarily to isolate the cause

### EMA target encoder collapses

Symptom: `vicreg_v` spikes; all predictions converge to the same vector.

Fix:
- Verify `ema.update_target()` is called **after** `optimizer.step()` (not before)
- Increase VICReg weight: `lambda_vicreg: 0.5`
- Ensure `tau_min ≥ 0.99` (avoid too-fast target updates)

### Router produces uniform centroid assignments

Symptom: `centroid_acc` stays near `1/C` (random chance level).

Fix:
- Confirm centroids are from the same embedding space as `target_source_emb`
- Increase `lambda_centroid: 0.5`
- Lower `centroid_temperature: 0.05` to sharpen the classification

### Training is slow

- Set `num_workers: 8` (one per CPU core up to the number of cores)
- Enable `pin_memory: true` for GPU training
- Enable `mixed_precision: true` for CUDA AMP (roughly 2× speedup)
- Use `HDF5` storage format for datasets > 10 GB (see dataset docs)

### Import errors

If you see `ModuleNotFoundError: No module named 'sklearn'` when running
the trainer on a clean environment, install the full dependency set:

```bash
pip install -e ".[dev]"
# or specifically:
pip install scikit-learn scipy sentence-transformers
```
