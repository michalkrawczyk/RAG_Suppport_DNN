# JEPA Stage 2: Concise Coding Plan

## Status: Phase 1 ✅ Complete | Phase 2 ✅ Complete | All tasks complete ✅

> **Note:** All files were implemented with the `JASPER` naming convention instead of `JEPA`
> (e.g. `jasper_predictor.py` instead of `jepa_predictor.py`).

---

## Phase 1: Core JEPA Architecture (MVP)

**Goal:** Train JEPA predictor with EMA and multi-objective losses on JASPERSteeringDataset

### Task 1: JEPA Predictor Model ✅ DONE
**File:** `RAG_supporters/nn/models/jasper_predictor.py` ✅ DONE (implemented as `jasper_predictor.py`)  
**Dependencies:** None

```python
class JEPAPredictor(nn.Module):
    """question_emb + steering_emb → predicted_source_emb"""
    # Components: question_encoder, steering_encoder, predictor_head
    # Config: hidden_dim, num_layers (YAML)
    # forward(question_emb, steering_emb) → predicted_source_emb [B, D]
```

**Done when:**
- ✅ Forward pass: [B, D] inputs → [B, D] output
- ✅ YAML config instantiation
- ✅ Test: correct shapes, config loading

---

### Task 2: EMA Target Encoder ✅ DONE
**File:** `RAG_supporters/nn/models/ema_encoder.py` ✅ DONE  
**Dependencies:** None

```python
class EMAEncoder(nn.Module):
    """Exponential Moving Average wrapper for target encoder"""
    # tau_schedule: cosine (tau_min=0.996 → tau_max=0.999)
    # update_target(step, max_steps) - momentum update
    # encode_target(x) - no_grad forward
    # state_dict() includes both online + target encoders
```

**Done when:**
- ✅ EMA updates work correctly
- ✅ Tau cosine annealing schedule
- ✅ Target encoder gradients disabled
- ✅ Test: tau progression, state dict save/load

---

### Task 3: Multi-Objective Losses ✅ DONE
**File:** `RAG_supporters/nn/losses/jasper_losses.py` ✅ DONE (implemented as `jasper_losses.py`)  
**Dependencies:** None

```python
JEPALoss            # SmoothL1(predicted, ema_target)
ContrastiveLoss     # InfoNCE with hard negatives
CentroidLoss        # Cross-entropy cluster classification
VICRegLoss          # Variance + Invariance + Covariance
JEPAMultiObjectiveLoss  # Combined: λ_jepa, λ_contrastive, λ_centroid, λ_vicreg
```

**Done when:**
- ✅ All 4 losses implemented + combined
- ✅ Returns dict with sub-components for logging
- ✅ Test: losses compute correctly, VICReg prevents collapse

---

### Task 4: Training Orchestrator ✅ DONE
**File:** `RAG_supporters/nn/training/jasper_trainer.py` ✅ DONE (implemented as `jasper_trainer.py`)  
**Dependencies:** Tasks 1-3

```python
class JEPATrainer:
    """Training loop with EMA updates, curriculum, checkpointing"""
    # train_epoch(epoch) → metrics_dict
    # validate() → metrics_dict
    # save_checkpoint(path, epoch, metrics) - includes EMA state
    # load_checkpoint(path)
    # fit(num_epochs)
```

**Done when:**
- ✅ EMA updated each step
- ✅ Curriculum: `set_epoch(loader, epoch)`
- ✅ Checkpoint: model + EMA + optimizer + metrics
- ✅ Logging: all losses, steering distribution, centroid accuracy
- ✅ Test: 1 epoch on mock data, checkpoint save/load

---

### Task 5: Training Script ✅ DONE
**File:** `examples/train_jasper_predictor.py` ✅ DONE (implemented as `train_jasper_predictor.py`)  
**Config:** `configs/jasper_base.yaml` ✅ DONE (implemented as `jasper_base.yaml`)  
**Dependencies:** Task 4

```bash
python examples/train_jepa_predictor.py --config configs/jepa_base.yaml
```

**Config structure:**
```yaml
model: {embedding_dim, hidden_dim, num_layers}
ema: {tau_min, tau_max, schedule}
loss: {lambda_jepa, lambda_contrastive, lambda_centroid, lambda_vicreg}
training: {lr, batch_size, epochs, warmup_epochs}
dataset: {path, n_neg}
```

**Done when:**
- ✅ CLI training works end-to-end
- ✅ Resume from checkpoint
- ✅ Example config provided
- ✅ Test: 2 epochs on mock dataset

---

### Task 6: Monitoring & Visualization ✅ DONE (test missing)
**File:** `RAG_supporters/nn/training/monitoring.py` ✅ DONE  
**Dependencies:** Task 4

```python
class TrainingMonitor:
    """Collect metrics, plot curves, export history"""
    # log_metrics(epoch, metrics_dict)
    # plot_losses(save_path) - all components
    # plot_steering_distribution(save_path)
    # export_history(save_path) - CSV/JSON
```

**Done when:**
- ✅ Loss curve plotting (subplots per component)
- ✅ Steering distribution evolution
- ✅ CSV export with full history
- ✅ Test: mock metrics → plots

---

### Task 7: Unit Tests (Phase 1) ✅ DONE (test_monitoring.py missing)
**Files:** `tests/test_jasper_*.py` ✅ DONE (uses `jasper_` prefix)  
**Dependencies:** Tasks 1-6

```
test_jasper_predictor.py   ✅ - init, forward, gradients, config
test_ema_encoder.py        ✅ - EMA update, tau schedule, state dict
test_jasper_losses.py      ✅ - individual + combined losses, VICReg
test_jasper_trainer.py     ✅ - single step, EMA update, checkpoint, curriculum
test_monitoring.py         ✅ - metric logging, W&B payload filtering, plotting guards, CSV/JSON export
```

**Done when:**
- ✅ All tests pass, >90% coverage
- ✅ Mock datasets (no API calls)
- ✅ Runs in <30 seconds

---

### Task 8: Documentation (Phase 1) ✅ DONE
**Files:** ✅ DONE  
- `docs/nn/JASPER_PREDICTOR.md` ✅ - Architecture, config, EMA, examples
- `docs/nn/TRAINING_JASPER.md` ✅ - Training script, config format, tuning, troubleshooting

**Done when:**
- ✅ Complete usage examples with outputs
- ✅ Config parameter reference
- ✅ Troubleshooting section

---

## Phase 2: Subspace-Routed JEPA with XAI

**Goal:** Add subspace routing, coarse+fine decomposition, XAI explanations

### Task 9: Subspace Router ✅ DONE
**File:** `RAG_supporters/nn/models/subspace_router.py` ✅ DONE  
**Dependencies:** Task 1 (parallel OK)

```python
class SubspaceRouter(nn.Module):
    """Concept bottleneck: question+steering → routing_weights [K]"""
    # Gumbel-Softmax (training), Softmax/Argmax (inference)
    # forward(question_emb, steering_emb, training) → routing_weights, logits
    # explain(question_emb, steering_emb, cluster_names) → XAI_dict
```

**Done when:**
- ✅ Valid probability distribution (sum to 1)
- ✅ Gumbel-Softmax differentiable
- ✅ XAI method returns cluster names + confidences
- ✅ Test: routing_weights validity, differentiation

---

### Task 10: Coarse + Fine Decomposition ✅ DONE
**File:** `RAG_supporters/nn/models/decomposed_predictor.py` ✅ DONE  
**Dependencies:** Task 9

```python
class DecomposedJEPAPredictor(nn.Module):
    """prediction = coarse (anchor) + fine (residual)"""
    # Components: SubspaceRouter, residual_predictor
    # coarse = weighted_sum(routing_weights, centroid_embs)
    # forward() → prediction, explanation_dict
    # explanation_dict: routing_weights, coarse_vector, fine_vector, atypicality
```

**Done when:**
- ✅ Prediction = coarse + fine
- ✅ Coarse uses exact centroids
- ✅ Atypicality = ||fine_vector||
- ✅ Test: decomposition validity, centroid alignment

---

### Task 11: Routing Losses ✅ DONE
**File:** `RAG_supporters/nn/losses/routing_losses.py` ✅ DONE  
**Dependencies:** Tasks 9-10

```python
RoutingLoss              # Cross-entropy: router classification
EntropyRegularization    # Schedule: diversity early → confidence late
ResidualPenalty          # L = max(0, ||residual|| - margin)
DisentanglementLoss      # Covariance penalty (multi-axis)
```

**Done when:**
- ✅ All routing losses implemented
- ✅ Entropy schedule (diversity → confidence)
- ✅ Test: losses compute correctly

---

### Task 12: XAI Interface ✅ DONE
**File:** `RAG_supporters/nn/inference/xai_interface.py` ✅ DONE  
**Dependencies:** Tasks 9-10

```python
class XAIInterface:
    """Inference-time explainability"""
    # explain_prediction(question_emb, steering_emb) → XAI_dict
    # XAI_dict: predicted_source_emb, primary_subspace, confidence,
    #           routing_distribution, steering_influence (KL),
    #           atypicality, similar_known_pairs, actionable_signal
    # compare_steering_influence() - with vs without steering
```

**Done when:**
- ✅ Complete XAI outputs
- ✅ Steering influence via KL/L2
- ✅ Similar pairs via nearest neighbors
- ✅ Test: XAI output structure, batch processing

---

### Task 13: Training Script (Subspace Model) ✅ DONE
**File:** `examples/train_subspace_jasper.py` ✅ DONE (implemented as `train_subspace_jasper.py`)  
**Dependencies:** Tasks 9-12

Similar to Task 5 but with:
- DecomposedJEPAPredictor
- Routing losses
- XAI validation (routing accuracy, entropy tracking)
- Residual magnitude logging

**Done when:**
- ✅ Subspace model training works
- ✅ Routing accuracy logged
- ✅ XAI outputs for validation set

---

### Task 14: Unit Tests (Phase 2) ✅ DONE
**Files:** `tests/test_subspace_*.py` ✅ DONE  
**Dependencies:** Tasks 9-13

```
test_subspace_router.py        ✅ - routing validity, Gumbel-Softmax, XAI
test_decomposed_predictor.py   ✅ - decomposition, atypicality
test_routing_losses.py         ✅ - all routing losses
test_xai_interface.py          ✅ - XAI structure, steering influence
```

**Done when:**
- ✅ All tests pass
- ✅ XAI validated against ground truth

---

### Task 15: Documentation (Phase 2) ✅ DONE
**Files:** ✅ DONE  
- `docs/nn/SUBSPACE_JASPER.md` ✅ - Routing architecture, coarse+fine, training
- `docs/nn/XAI_INTERFACE.md` ✅ - Output format, interpretation guide, use cases

**Done when:**
- ✅ XAI examples with real questions
- ✅ Use case scenarios
- ✅ Visualization examples

---

## Execution Paths

### Quick Win (Get Training Working) ✅ DONE
**Timeline:** 5-6 days
1. Tasks 1-3 (parallel) → 2-3 days ✅
2. Task 4 (Trainer) → 2 days ✅
3. Task 5 (Script) → 1 day ✅
4. **Checkpoint:** Run 10 epochs, validate convergence

### Full MVP (Phase 1) ✅ DONE
**Timeline:** 1-2 weeks
- Tasks 1-8 complete ✅ (except test_monitoring.py)
- **Milestone:** Publish basic JASPER predictor

### Advanced XAI (Phase 2) ✅ DONE
**Timeline:** 2-3 weeks
- Tasks 9-15 complete ✅
- **Milestone:** Published with XAI capabilities

**Total Effort:** 4-6 weeks

---

## Critical Decisions Before Starting

| Decision | Recommendation |
|----------|---------------|
| **Config format** | YAML (match existing model_builder) |
| **Anchor updates** | Frozen centroids initially (max interpretability) |
| **Gumbel-Softmax temp** | Cosine anneal: 1.0 → 0.1 |
| **Loss weights** | Fixed initially, add learned weighting later |
| **Routing axes** | Single axis (topic only) for MVP |

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| **EMA collapse** | Proven tau schedule (0.996→0.999), VICReg regularization |
| **Router collapse** | Entropy regularization, balanced batch sampling |
| **Residual dominance** | Residual penalty, two-stage training (freeze anchors) |
| **Training instability** | Gradient clipping, warmup, start with L_jepa only |

---

## File Impact Summary

### New Files
**Phase 1:** 8 core + 5 test + 2 docs = 15 files  
**Phase 2:** 7 core + 4 test + 2 docs = 13 files  
**Total:** 28 new files

### Modified Files
- `RAG_supporters/nn/__init__.py` - Export new models/losses
- `RAG_supporters/nn/models/__init__.py` - Export model classes
- `agents_notes/PROJECT_STRUCTURE.md` - Document new files
- `docs/README.md` - Add documentation links

---

## Next Action
**All tasks complete.** Both Phase 1 and Phase 2 are fully implemented and tested.
