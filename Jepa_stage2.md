# JEPA Stage 2: Implementation Plan

## Executive Summary

This document outlines the implementation plan for a JEPA-like (Joint-Embedding Predictive Architecture) steering model for RAG support. The model will predict source embeddings from question embeddings conditioned on steering signals, with optional subspace routing for explainability.

**Current Status:** Dataset infrastructure complete ✅  
**Next Step:** Implement JEPA model architecture and training pipeline

---

## Current State Assessment

### ✅ **What's Already Implemented**

1. **JASPER Dataset Infrastructure** (Complete)
   - [JASPERSteeringDataset](RAG_supporters/pytorch_datasets/jasper_steering_dataset.py) - PyTorch dataset with curriculum learning
   - [Dataset Builder Pipeline](RAG_supporters/jasper/build.py) - Full orchestration (Tasks 1-9)
   - [Embedding Generation](RAG_supporters/embeddings_ops/embed.py) - Batch processing for all text types
   - [Steering Signal Generation](RAG_supporters/contrastive/build_steering.py) - 4 variants (zero, centroid, keyword-weighted, residual)
   - [Hard Negative Mining](RAG_supporters/contrastive/mine_negatives.py) - Multi-tier negatives
   - [DataLoader Utilities](RAG_supporters/pytorch_datasets/loader.py) - `create_loader`, `set_epoch`, curriculum support

2. **Basic Model Infrastructure**
   - [ConfigurableModel](RAG_supporters/nn/models/model_builder.py) - YAML-based model builder
   - [Simple JASPER Predictor](docs/dataset/JASPER_TRAINING_EXAMPLE.md#L8-L27) - Example in documentation (concat question+steering → MLP → prediction)

3. **Documentation & Examples**
   - [JASPER_TRAINING_EXAMPLE.md](docs/dataset/JASPER_TRAINING_EXAMPLE.md) - Basic training loop
   - [JASPER_STEERING_DATASET.md](docs/dataset/JASPER_STEERING_DATASET.md) - Dataset documentation

### ❌ **What's Missing** (Based on Your Plans)

1. **JEPA-Specific Model Architectures**
   - Predictor with EMA Target Encoder
   - Subspace Router (concept bottleneck for XAI)
   - Coarse + Fine prediction decomposition
   - Multi-axis disentangled subspaces

2. **JEPA-Specific Loss Functions**
   - L_jepa (SmoothL1 in latent space)
   - L_routing (subspace classification + entropy regularization)
   - L_disentangle (covariance penalty across axes)
   - VICReg regularization (variance + invariance + covariance)

3. **Training Infrastructure**
   - Training script with EMA updates
   - Router monitoring/logging
   - XAI output generation for inference
   - Steering influence tracking

---

## Implementation Plan: JEPA Architecture for RAG Support

### **Scope Definition**

This plan implements the JEPA-like steering model in **two phases**:
- **Phase 1 (MVP)**: Core JEPA predictor with EMA and multi-objective losses
- **Phase 2 (Advanced)**: Subspace-routed architecture with XAI capabilities

---

## **Phase 1: Core JEPA Architecture** (MVP)

### Definition of Done
- [ ] JEPA predictor model with EMA target encoder implemented
- [ ] Multi-objective loss module (L_jepa + L_contrastive + L_centroid + L_vicreg)
- [ ] Training script with EMA updates, curriculum, and validation
- [ ] Model checkpoint saving/loading with EMA state
- [ ] Basic monitoring (loss curves, steering distribution)
- [ ] Unit tests for all components
- [ ] Documentation updated

**Success looks like:** A training script that successfully trains on JASPERSteeringDataset, shows convergence on validation loss, and produces a model that predicts source embeddings from question+steering better than baseline.

---

### Step 1: Implement JEPA Predictor Model
**File:** `RAG_supporters/nn/models/jepa_predictor.py` (CREATE)  
**Action:** Create

**Details:**
```python
class JEPAPredictor(nn.Module):
    """
    JEPA predictor: question_emb + steering_emb → predicted_source_emb
    
    Architecture:
    - Context encoder: processes question embedding
    - Steering processor: processes steering signal
    - Predictor head: combines both → predicts target in latent space
    """
    
    Components:
    - question_encoder: nn.Sequential (question_emb → latent)
    - steering_encoder: nn.Sequential (steering_emb → latent)
    - predictor_head: nn.Sequential (concat → prediction)
    - Configuration via YAML or dataclass
    
    Methods:
    - forward(question_emb, steering_emb) → predicted_source_emb
    - get_latent_representations() → dict of intermediate activations
```

**✅ Done when:**
- Model accepts [B, D] question and steering tensors
- Outputs [B, D] predicted source embeddings
- Configurable hidden dimensions and depth
- Test: forward pass with dummy data produces correct shapes
- Test: model can be instantiated from config
- **Dependencies:** None

---

### Step 2: Implement EMA Target Encoder
**File:** `RAG_supporters/nn/models/ema_encoder.py` (CREATE)  
**Action:** Create

**Details:**
```python
class EMAEncoder(nn.Module):
    """
    Exponential Moving Average wrapper for target encoder.
    
    Prevents representation collapse by maintaining a slowly-updating 
    target that the predictor tries to match.
    
    Used in JEPA, BYOL, MoCo architectures.
    """
    
    def __init__(self, base_encoder, tau_schedule='cosine', tau_min=0.996, tau_max=0.999):
        - self.online_encoder (predictor parameters, updated via backprop)
        - self.target_encoder (EMA copy, updated via momentum)
        - tau schedule (starts at tau_min, anneals to tau_max)
    
    Methods:
    - update_target(step, max_steps) → updates target with current tau
    - encode_target(x) → forward through target encoder (no_grad)
    - get_tau(step, max_steps) → compute current tau value
    - state_dict() / load_state_dict() → save/load both encoders
```

**✅ Done when:**
- EMA wrapper correctly updates target encoder momentum
- Tau annealing schedule (cosine) works correctly
- Target encoder gradients are disabled
- State dict includes both encoders + tau schedule
- Test: target encoder parameters drift slower than online encoder
- Test: tau schedule progresses correctly over training steps
- **Dependencies:** None

---

### Step 3: Implement Multi-Objective Loss Module
**File:** `RAG_supporters/nn/losses/jepa_losses.py` (CREATE)  
**Action:** Create

**Details:**
```python
# L_jepa: Prediction loss in latent space
class JEPALoss(nn.Module):
    """SmoothL1 loss between predicted and EMA target embeddings."""
    forward(predicted, ema_target) → loss

# L_contrastive: InfoNCE with hard negatives
class ContrastiveLoss(nn.Module):
    """InfoNCE: pull predicted toward target, push away from negatives."""
    forward(predicted, target, negatives, temperature=0.07) → loss

# L_centroid: Auxiliary cluster classification
class CentroidLoss(nn.Module):
    """Cross-entropy: does prediction land in correct cluster region?"""
    forward(predicted_emb, centroid_embs, cluster_id) → loss

# L_vicreg: Variance + Invariance + Covariance regularization
class VICRegLoss(nn.Module):
    """Prevents embedding collapse via 3 components."""
    forward(predicted_embs: [B, D]) → loss_dict (v, i, c components)

# Combined loss
class JEPAMultiObjectiveLoss(nn.Module):
    """Combines all losses with configurable weights."""
    __init__(lambda_jepa, lambda_contrastive, lambda_centroid, lambda_vicreg)
    forward(batch, predicted, ema_target, centroid_embs) → total_loss, loss_dict
```

**✅ Done when:**
- All 4 loss components implemented
- Combined loss with configurable λ weights
- Each loss returns dict with sub-components for logging
- Test: losses compute correctly on dummy data
- Test: loss gradients flow to predictor
- Test: VICReg prevents collapse in toy scenario
- **Dependencies:** None

---

### Step 4: Implement Training Loop with EMA
**File:** `RAG_supporters/nn/training/jepa_trainer.py` (CREATE)  
**Action:** Create

**Details:**
```python
class JEPATrainer:
    """
    Training orchestrator for JEPA models.
    
    Handles:
    - EMA target encoder updates
    - Multi-objective loss computation
    - Curriculum learning (via dataset.set_epoch)
    - Checkpointing with EMA state
    - Logging (loss curves, steering distribution)
    - Validation loop
    """
    
    def __init__(config, model, ema_encoder, loss_fn, optimizer, train_loader, val_loader):
        ...
    
    def train_epoch(epoch) → metrics_dict
    def validate() → metrics_dict
    def save_checkpoint(path, epoch, metrics)
    def load_checkpoint(path) → epoch, metrics
    def fit(num_epochs) → training_history
```

**✅ Done when:**
- Training loop correctly updates predictor + EMA encoder
- Curriculum learning via `set_epoch(loader, epoch)`
- Checkpoints save: model, EMA, optimizer, scheduler, epoch, metrics
- Validation runs without EMA updates
- Logging captures: all loss components, steering variant distribution, centroid accuracy
- Test: can train for 1 epoch on mock dataset
- Test: checkpoint can be saved and loaded
- **Dependencies:** Steps 1-3

---

### Step 5: Create Training Script
**File:** `examples/train_jepa_predictor.py` (CREATE)  
**Action:** Create

**Details:**
```python
#!/usr/bin/env python3
"""
Train JEPA Predictor on JASPER Steering Dataset.

Usage:
    python examples/train_jepa_predictor.py --config configs/jepa_base.yaml
"""

Entry point script that:
1. Parses CLI arguments (config path, dataset dir, output dir)
2. Loads config (model, training, loss weights)
3. Creates dataset loaders via create_loader()
4. Instantiates model, EMA, losses, optimizer
5. Creates JEPATrainer
6. Runs training with checkpointing
7. Saves final model
8. Plots training curves (optional, via matplotlib)

Config structure (YAML):
- model: {embedding_dim, hidden_dim, num_layers}
- ema: {tau_min, tau_max, schedule}
- loss: {lambda_jepa, lambda_contrastive, lambda_centroid, lambda_vicreg}
- training: {lr, batch_size, epochs, warmup_epochs}
- dataset: {path, n_neg}
```

**✅ Done when:**
- Script runs end-to-end training from CLI
- Config loaded from YAML file
- Checkpointing works (resume from checkpoint)
- Training curves logged (to file or tensorboard)
- Example config file provided: `configs/jepa_base.yaml`
- Test: can run 2 epochs on mock dataset
- **Dependencies:** Step 4

---

### Step 6: Add Monitoring & Visualization
**File:** `RAG_supporters/nn/training/monitoring.py` (CREATE)  
**Action:** Create

**Details:**
```python
class TrainingMonitor:
    """
    Collect and visualize training metrics.
    
    Features:
    - Loss curve plotting (all components)
    - Steering variant distribution over epochs
    - Centroid accuracy tracking
    - EMA tau schedule visualization
    - Validation loss comparison
    - Export to CSV / JSON
    """
    
    def log_metrics(epoch, metrics_dict)
    def plot_losses(save_path)
    def plot_steering_distribution(save_path)
    def export_history(save_path)
    def get_summary_table() → pandas.DataFrame
```

**✅ Done when:**
- Monitor logs all metrics per epoch
- Can plot loss curves (separate subplots for each component)
- Can plot steering distribution evolution
- Exports CSV with full training history
- Test: mock metrics produce correct plots
- **Dependencies:** Step 4

---

### Step 7: Unit Tests for Phase 1
**Files:** `tests/test_jepa_*.py` (CREATE)  
**Action:** Create

**Test Coverage:**
```python
# tests/test_jepa_predictor.py
- Test model initialization
- Test forward pass shapes
- Test gradient flow
- Test config-based instantiation

# tests/test_ema_encoder.py
- Test EMA update mechanics
- Test tau schedule (cosine annealing)
- Test state dict save/load
- Test target encoder has no_grad

# tests/test_jepa_losses.py
- Test each loss component individually
- Test combined loss
- Test loss weighting
- Test VICReg prevents collapse

# tests/test_jepa_trainer.py
- Test single training step
- Test EMA update in training loop
- Test checkpoint save/load
- Test curriculum updates
```

**✅ Done when:**
- All tests pass with >90% coverage
- Mock datasets used (no real LLM calls)
- Tests run in <30 seconds total
- **Dependencies:** Steps 1-6

---

### Step 8: Documentation for Phase 1
**Files:** `docs/nn/JEPA_PREDICTOR.md`, `docs/nn/TRAINING_JEPA.md` (CREATE)  
**Action:** Create

**Content:**
```markdown
# JEPA_PREDICTOR.md
- Architecture overview
- Model configuration options
- EMA mechanism explanation
- Usage examples

# TRAINING_JEPA.md
- Training script usage
- Config file format
- Loss function tuning guide
- Curriculum learning setup
- Checkpoint management
- Troubleshooting
```

**✅ Done when:**
- Documentation includes architecture diagrams (ASCII art OK)
- Complete usage examples with expected outputs
- Config parameter reference table
- Troubleshooting section (common issues)
- **Dependencies:** Steps 1-7

---

## **Phase 2: Subspace-Routed JEPA with XAI** (Advanced)

### Definition of Done
- [ ] Subspace router (concept bottleneck) implemented
- [ ] Coarse + fine prediction decomposition working
- [ ] Routing-specific losses (L_routing, entropy regularization)
- [ ] XAI interface for inference (routing explanations)
- [ ] Multi-axis disentanglement (optional)
- [ ] Training script for subspace model
- [ ] XAI output examples in documentation

**Success looks like:** A model that not only predicts source embeddings accurately but also provides interpretable routing decisions ("This question belongs 62% to topic A, 28% to topic B"), with XAI outputs that can be validated against ground-truth cluster assignments.

---

### Step 9: Implement Subspace Router
**File:** `RAG_supporters/nn/models/subspace_router.py` (CREATE)  
**Action:** Create

**Details:**
```python
class SubspaceRouter(nn.Module):
    """
    Concept bottleneck: routes questions to subspaces (clusters).
    
    Inputs: question_emb + steering_emb
    Outputs: routing_weights [K] (soft assignment to K clusters)
    
    Architecture:
    - Input projection
    - MLP → concept_logits [K]
    - Gumbel-Softmax (training) or Softmax/Argmax (inference)
    
    XAI Output:
    - routing_weights: soft assignment to clusters
    - primary_subspace: argmax cluster
    - confidence: max(routing_weights)
    """
    
    Methods:
    - forward(question_emb, steering_emb, training=True) → routing_weights, concept_logits
    - explain(question_emb, steering_emb, cluster_names) → XAI_dict
    - get_primary_subspace() → cluster_id, confidence
```

**✅ Done when:**
- Router produces valid probability distribution over clusters
- Gumbel-Softmax used during training (differentiable)
- Argmax or top-k used during inference
- XAI method returns interpretable cluster names + confidences
- Test: routing_weights sum to 1.0
- Test: Gumbel-Softmax is differentiable
- **Dependencies:** Step 1 (can be developed in parallel)

---

### Step 10: Implement Coarse + Fine Decomposition
**File:** `RAG_supporters/nn/models/decomposed_predictor.py` (CREATE)  
**Action:** Create

**Details:**
```python
class DecomposedJEPAPredictor(nn.Module):
    """
    Prediction = Coarse (anchor retrieval) + Fine (residual)
    
    Components:
    1. SubspaceRouter → routing_weights [K]
    2. Anchor retrieval: weighted_sum(routing_weights, centroid_embs) → coarse_vector
    3. Residual predictor: f(question, steering, selected_subspace) → fine_vector
    4. Prediction: coarse_vector + fine_vector
    
    Interpretability:
    - coarse_vector: "go to this region" (tied to cluster centroids)
    - fine_vector: "offset from prototype" (learned refinement)
    - ||fine_vector|| = atypicality score
    """
    
    def __init__(router, residual_predictor, centroid_embs):
        ...
    
    def forward(question_emb, steering_emb) → prediction, explanation_dict
        explanation_dict = {
            'routing_weights': [K],
            'coarse_vector': [D],
            'fine_vector': [D],
            'atypicality': scalar (||fine_vector||)
        }
```

**✅ Done when:**
- Prediction decomposes into coarse + fine
- Coarse vector uses exact cluster centroids (interpretable anchors)
- Residual magnitude tracked (atypicality)
- Test: coarse + fine = prediction
- Test: coarse vector aligns with weighted centroid
- **Dependencies:** Step 9

---

### Step 11: Implement Routing Losses
**File:** `RAG_supporters/nn/losses/routing_losses.py` (CREATE)  
**Action:** Create

**Details:**
```python
# L_routing: Router classification loss
class RoutingLoss(nn.Module):
    """Cross-entropy: did router pick correct subspace?"""
    forward(routing_logits, true_cluster_id) → loss

# Entropy regularization
class EntropyRegularization(nn.Module):
    """
    Penalize flat routing (early) or peaked routing (late).
    Schedule: encourage diversity early, confidence late.
    """
    forward(routing_weights, epoch, max_epochs, mode='anneal') → loss

# Residual magnitude penalty
class ResidualPenalty(nn.Module):
    """
    Encourage small residuals (coarse explains most).
    L = max(0, ||residual|| - margin)
    """
    forward(fine_vector, margin=0.5) → loss

# Disentanglement loss (for multi-axis)
class DisentanglementLoss(nn.Module):
    """Covariance penalty between routing axes."""
    forward(routing_weights_axis1, routing_weights_axis2) → loss
```

**✅ Done when:**
- All routing-specific losses implemented
- Entropy regularization schedule works (early diversity → late confidence)
- Residual penalty encourages interpretable coarse vectors
- Test: losses compute correctly
- **Dependencies:** Step 9-10

---

### Step 12: Implement XAI Interface
**File:** `RAG_supporters/nn/inference/xai_interface.py` (CREATE)  
**Action:** Create

**Details:**
```python
class XAIInterface:
    """
    Inference-time explainability for JEPA predictions.
    
    Outputs:
    - Primary subspace (cluster name)
    - Routing distribution (all clusters)
    - Steering influence (with vs without steering)
    - Atypicality score (residual magnitude)
    - Similar training pairs (nearest neighbors in subspace)
    - Confidence score
    """
    
    def __init__(model, centroid_embs, cluster_names, training_pairs):
        ...
    
    def explain_prediction(question_emb, steering_emb=None) → XAI_dict
        Returns:
        {
            'predicted_source_emb': [D],
            'primary_subspace': 'Contract Law',
            'subspace_confidence': 0.87,
            'routing_distribution': {'Legal': 0.62, 'Financial': 0.28, ...},
            'steering_influence': 0.34,  # KL(with_steer || without_steer)
            'atypicality': 0.12,
            'similar_known_pairs': [idx1, idx2, ...],
            'actionable_signal': 'High confidence, small residual → Trust retrieval'
        }
    
    def compare_steering_influence(question_emb, steering_emb) → dict
        Shows routing difference with vs without steering
    
    def visualize_explanation(xai_dict) → matplotlib figure
```

**✅ Done when:**
- XAI interface produces all interpretability outputs
- Steering influence computed via KL divergence or L2 distance
- Similar pairs retrieved via nearest neighbor search
- Actionable signals generated based on confidence + atypicality
- Test: XAI output has expected structure
- Test: can run on batch of questions
- **Dependencies:** Steps 9-10

---

### Step 13: Training Script for Subspace Model
**File:** `examples/train_subspace_jepa.py` (CREATE)  
**Action:** Create

**Details:**
Similar to Step 5, but with:
- DecomposedJEPAPredictor instead of JEPAPredictor
- Additional routing losses
- XAI validation (log routing accuracy, entropy over epochs)
- Residual magnitude tracking

**✅ Done when:**
- Script trains subspace-routed model
- Logs routing accuracy to clusters
- Tracks residual magnitude evolution
- Saves XAI outputs for validation set
- **Dependencies:** Steps 9-12

---

### Step 14: Unit Tests for Phase 2
**Files:** `tests/test_subspace_*.py` (CREATE)  
**Action:** Create

**Test Coverage:**
```python
# tests/test_subspace_router.py
- Test routing weight validity (sum to 1)
- Test Gumbel-Softmax differentiability
- Test XAI explain() method
- Test top-k subspace selection

# tests/test_decomposed_predictor.py
- Test coarse + fine decomposition
- Test atypicality computation
- Test anchor alignment with centroids

# tests/test_routing_losses.py
- Test routing classification loss
- Test entropy regularization schedule
- Test residual penalty
- Test disentanglement loss

# tests/test_xai_interface.py
- Test XAI output structure
- Test steering influence computation
- Test similar pair retrieval
```

**✅ Done when:**
- All tests pass
- XAI outputs validated against ground truth
- **Dependencies:** Steps 9-13

---

### Step 15: Documentation for Phase 2
**Files:** `docs/nn/SUBSPACE_JEPA.md`, `docs/nn/XAI_INTERFACE.md` (CREATE)  
**Action:** Create

**Content:**
```markdown
# SUBSPACE_JEPA.md
- Subspace routing architecture
- Coarse + fine decomposition explanation
- Router training strategies
- Anchor initialization from centroids
- Multi-axis disentanglement (advanced)

# XAI_INTERFACE.md
- XAI output format specification
- Interpretation guide (what each metric means)
- Actionable signals decision tree
- Use cases (debugging, trust assessment, user feedback)
- Visualization examples
```

**✅ Done when:**
- Complete XAI output examples with real questions
- Use case scenarios documented
- Visualization examples included
- **Dependencies:** Steps 9-14

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| **EMA collapse** (target encoder doesn't stabilize) | Medium | High | Start with proven tau schedule (0.996→0.999), add VICReg regularization, monitor target encoder variance |
| **Router collapse** (all questions → one subspace) | Medium | High | Entropy regularization, balanced batch sampling per cluster, validate routing accuracy early |
| **Residual dominance** (model ignores coarse anchors) | Medium | Medium | Residual penalty, two-stage training (freeze anchors initially), monitor ‖residual‖ / ‖coarse‖ ratio |
| **XAI outputs uninterpretable** (routing doesn't align with clusters) | Low | Medium | Initialize anchors from actual centroids, validate routing accuracy against ground truth, freeze anchors during early training |
| **Training instability** (loss diverges) | Low | High | Gradient clipping, warmup learning rate, start with simple loss (L_jepa only) then add components |

---

## Recommended Execution Order

### **Quick Win Path** (Get training working ASAP):
1. Steps 1-3 (Model + Losses) → 2-3 days
2. Step 4 (Trainer) → 2 days
3. Step 5 (Training Script) → 1 day
4. **Checkpoint**: Run 10 epochs on real data, validate convergence

### **Full MVP Path** (Phase 1 Complete):
1. Steps 1-8 → 1-2 weeks
2. **Milestone**: Publish Phase 1 with basic JEPA predictor

### **Advanced Path** (Phase 2):
1. Steps 9-15 → 2-3 weeks
2. **Milestone**: Publish Phase 2 with XAI capabilities

---

## Critical Design Decisions to Make Before Starting

1. **Model Configuration Format**: YAML (like existing model_builder) or Python dataclass?
   - **Recommendation**: YAML for consistency with existing code

2. **Anchor Update Strategy**: Frozen, slow-updated, or fully learnable centroids?
   - **Recommendation**: Start frozen (max interpretability), add slow-update option later

3. **Routing Temperature Annealing**: How to schedule Gumbel-Softmax temperature?
   - **Recommendation**: Cosine schedule from 1.0 → 0.1 over training

4. **Loss Weight Tuning**: Fixed or learned (GradNorm, uncertainty weighting)?
   - **Recommendation**: Start fixed, add learned weighting as improvement

5. **Multi-Axis Routing**: Single axis (topic only) or multiple (topic + intent + specificity)?
   - **Recommendation**: Single axis for MVP, multi-axis in Phase 2 extension

---

## File Changes Summary

### **New Files (Phase 1)**: 8 files
- `RAG_supporters/nn/models/jasper_predictor.py`
- `RAG_supporters/nn/models/ema_encoder.py`
- `RAG_supporters/nn/losses/jasper_losses.py`
- `RAG_supporters/nn/training/jasper_trainer.py`
- `RAG_supporters/nn/training/monitoring.py`
- `examples/train_jasper_predictor.py`
- `configs/jasper_base.yaml`
- `docs/nn/JASPER_PREDICTOR.md`

### **New Files (Phase 2)**: 7 files
- `RAG_supporters/nn/models/subspace_router.py`
- `RAG_supporters/nn/models/decomposed_predictor.py`
- `RAG_supporters/nn/losses/routing_losses.py`
- `RAG_supporters/nn/inference/xai_interface.py`
- `examples/train_subspace_jasper.py`
- `docs/nn/SUBSPACE_JASPER.md`
- `docs/nn/XAI_INTERFACE.md`

### **Test Files**: 9 files
- `tests/test_jasper_predictor.py`
- `tests/test_ema_encoder.py`
- `tests/test_jasper_losses.py`
- `tests/test_jasper_trainer.py`
- `tests/test_subspace_router.py`
- `tests/test_decomposed_predictor.py`
- `tests/test_routing_losses.py`
- `tests/test_xai_interface.py`
- `tests/test_monitoring.py`

### **Modified Files**:
- `RAG_supporters/nn/__init__.py` - Export new models and losses
- `RAG_supporters/nn/models/__init__.py` - Export new model classes
- `agents_notes/PROJECT_STRUCTURE.md` - Document new files
- `docs/README.md` - Add links to new documentation

---

## Total Estimated Effort

- **Phase 1**: 1-2 weeks (core functionality)
- **Phase 2**: 2-3 weeks (advanced XAI features)
- **Testing & Documentation**: 1 week (across both phases)
- **Total**: 4-6 weeks for complete implementation

---

## Conceptual Background: JEPA Architecture

### Core Insight

JEPA's power comes from **predicting in representation space rather than reconstructing raw inputs**. Instead of predicting pixel values or token sequences, JEPA predicts embeddings—abstract representations that capture semantic meaning.

### Mapping to RAG Problem

| JEPA Concept | RAG Steering Implementation |
|--------------|----------------------------|
| **Context input** `x` | `question_text` embedding |
| **Mask / conditioning signal** | `steering_embedding` (topic keywords, zeros, or blend) |
| **Target** `y` | `source_text` embedding (in latent space) |
| **Predictor** | `f(q_emb, steering) → ŝ_emb` |
| **Target encoder** | EMA-updated encoder for source (prevents collapse) |

The steering embedding is analogous to JEPA's **mask specification** — it tells the predictor *which region of semantic space* to aim for, not the answer itself.

### Why Subspace Routing?

Predicting directly in full **D**-dimensional space conflates two fundamentally different tasks:

1. **"Where should I look?"** → subspace/region selection (coarse, discrete-ish)
2. **"What exactly should I find there?"** → within-region refinement (fine, continuous)

Separating these gives you both **better navigation** and **natural XAI hooks**.

**Decomposed Prediction:**
```
prediction = anchor(subspace) + residual(within subspace)
         ↑                        ↑
    INTERPRETABLE             FINE-GRAINED
    "go to this region"       "go to this exact point"
```

### Interpretability Benefits

The subspace router is a **concept bottleneck** — a layer where routing decisions are explicitly represented as probabilities over named clusters. This provides:

1. **Routing explanation**: "This question is 62% topic_A, 28% topic_B, ..."
2. **Primary subspace**: "The model chose THIS region"
3. **Atypicality score**: Large residual = answer is unusual for that subspace
4. **Steering influence**: Delta caused by steering signal
5. **Confidence signal**: High routing confidence + small residual = trust retrieval

This is **intrinsically interpretable**, not post-hoc explanation — the routing decision is causally part of the computation.

---

## Next Steps

1. **Review this plan** and provide feedback
2. **Make design decisions** on the 5 critical choices listed above
3. **Begin Phase 1 implementation** starting with Step 1 (JEPA Predictor Model)
4. **Set up project tracking** for the 15 implementation steps

**Recommended first action:** Implement Steps 1-3 in parallel (Model + EMA + Losses) as they have no dependencies on each other.
