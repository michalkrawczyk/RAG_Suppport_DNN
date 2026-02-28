# NN Coding Rules

> Read this file when writing or reviewing code in `nn/`, `pytorch_datasets/`, `jasper/`, or `contrastive/`.

## Layer Boundaries

Applies to `nn/`, `pytorch_datasets/`, `jasper/`, `contrastive/`.

| Layer | Location | Allowed | Forbidden |
|-------|----------|---------|----------|
| Models | `RAG_supporters/nn/models/` | `torch`, `typing`, stdlib | `agents`, `prompts_templates`, `pandas` |
| Datasets | `RAG_supporters/dataset/*_dataset.py` | `torch`, `pandas`, `numpy`, dataset utils | `agents` (except type hints), `prompts_templates` |
| Training | Scripts / notebooks | `torch`, models, datasets, agents | Direct LLM calls, prompts in training |
| Utils | `RAG_supporters/utils/` | stdlib, `numpy` | `torch` (unless `utils/torch_utils.py`) |

Exceptions require: a comment explaining necessity + `TODO(TICKET-#)` for refactoring + reviewer approval in PR.

---

## Mandatory Tech Stack

| Component | Required | Reject If Using |
|-----------|----------|----------------|
| NN Framework | PyTorch `torch.nn.Module` | TensorFlow, JAX, custom frameworks |
| Dataset classes | `torch.utils.data.Dataset` with `__len__`, `__getitem__` | Manual batching, custom iterators |
| DataLoader | `torch.utils.data.DataLoader` | Manual batch generation |
| Tensors | `torch.Tensor`, `.to(device)` with explicit device param | NumPy arrays to model, hardcoded `.cuda()` |
| Loss functions | PyTorch loss modules or validated custom | Manual gradient computation |
| Optimizers | `torch.optim` modules | Untested custom optimizers |
| Type hints | All public methods, parameters, returns | Missing type annotations |
| Data splits | `DatasetSplitter` with saved JSON indices | Manual random splits without persistence |

---

## PyTorch Dataset Requirements

Every custom `Dataset` MUST:

| Requirement | Details |
|-------------|---------|
| Inherit from | `torch.utils.data.Dataset` |
| Implement | `__len__() -> int` and `__getitem__(idx: int) -> Dict[str, torch.Tensor]` |
| Validate | Required columns in `__init__`, index bounds in `__getitem__` |
| Document | Docstring with tensor shapes, dtypes, return structure |
| Error handling | `IndexError` for invalid index, `ValueError` for invalid data |

**Docstring template:**
```python
"""Returns (per sample): Dict[str, torch.Tensor]:
    - 'input': shape (D,), dtype float32
    - 'target': shape (C,), dtype long
"""
```

---

## Model Architecture Requirements

Every `nn.Module` MUST:
- Inherit from `torch.nn.Module`
- Document input/output shapes in docstring
- Validate input shapes in `forward()` (check dimensions and sizes)
- Use type hints on all public methods
- Be device agnostic — explicit `.to(device)`, no hardcoded `.cuda()`

---

## Data Integrity & Reproducibility

| Area | Required | Reject If |
|------|----------|-----------|
| **Train/Val Splits** | `DatasetSplitter` from `RAG_supporters.dataset` with saved JSON indices | Manual random splits, no seed persistence |
| **Data Leakage** | Strict train/val/test separation, zero overlap | Validation data in training batches |
| **Tensor Shapes** | Validate shapes before operations | Assuming shapes without checks |
| **Device Consistency** | All tensors on same device, explicit `.to(device)` | Mixed CPU/GPU, hardcoded `.cuda()` |
| **Random Seeds** | Set seeds (torch, numpy, random) | Non-deterministic operations |
| **Gradient Management** | `.zero_grad()` before backward pass | Not clearing gradients |

---

## Efficiency Requirements

| Category | Red Flag (Reject) | Yellow Flag (Suggest) | Required Pattern |
|----------|-------------------|----------------------|------------------|
| **GPU Usage** | Moving tensors in loop, no batching | Not using mixed precision AMP | Batch tensor transfers, avoid CPU↔GPU in loops |
| **DataLoader** | `num_workers=0` without justification | No `pin_memory=True` | `num_workers ≥ 2`, `pin_memory=True` for GPU |
| **Memory** | Unnecessary `.clone()` in loops | Not using `.detach()` when needed | Minimize copies, reuse buffers |
| **Batch Processing** | Hardcoded batch size | Fixed batch size without tuning | Configurable via parameter |
