# Module Reference

> Read this file when implementing or modifying code in `nn/`, `pytorch_datasets/`, `jasper/`, or `contrastive/`.

## Neural Network (`RAG_supporters/nn/`)

**Core model:** `JASPERPredictor` — JEPA-style predictor: `(question_emb + steering_emb) → predicted_source_emb`
**EMA:** `EMAEncoder` wraps the encoder with cosine tau schedule (0.996→0.999)
**Phase 2:** `SubspaceRouter` (Gumbel-Softmax), `DecomposedJASPERPredictor` (coarse+fine decomposition), `XAIInterface`

Losses: `JASPERLoss`, `ContrastiveLoss` (InfoNCE), `CentroidLoss`, `VICRegLoss`, `RoutingLoss`, `EntropyRegularization`, `ResidualPenalty`, `DisentanglementLoss`

**Training:** `JASPERTrainer` handles curriculum learning, EMA updates, and checkpointing. `TrainingMonitor` handles metrics and optional W&B integration.

## PyTorch Datasets (`RAG_supporters/pytorch_datasets/`)

`JASPERSteeringDataset` — primary dataset:
- Zero I/O training (embeddings in memory or HDF5)
- Curriculum learning: call `dataset.set_epoch(epoch)` each epoch
- 4-tier hard negative sampling: in-cluster → adjacent → high-similarity → random
- Supports HDF5 and memmap storage formats

## JASPER Dataset Builder (`RAG_supporters/jasper/`)

Pipeline Tasks 1–9: CSV merge → cluster parsing → embedding generation → steering computation → negative mining → train/val/test split → persistence → validation → finalization. Configured by `BuildConfig` (validates and serializes to JSON). `SQLiteStorageManager` is the storage backend.

## Contrastive Tools (`RAG_supporters/contrastive/`)

`NegativeMiner`: 4-tier hard negative sampling with configurable proportions.
`SteeringBuilder`: Generates steering signals (centroid, keyword-weighted, residual).
