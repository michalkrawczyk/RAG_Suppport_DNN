````markdown
# Training Example: JASPER Steering Dataset

This example demonstrates how to train a JASPER predictor (Joint Architecture for Subspace Prediction with Explainable Routing) using the JASPERSteeringDataset.

## Model Architecture

```python
import torch.nn as nn

class SimpleJASPERPredictor(nn.Module):
    """Predict target embedding from question and steering."""
    
    def __init__(self, embedding_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
    
    def forward(self, question_emb, steering):
        x = torch.cat([question_emb, steering], dim=-1)
        return self.predictor(x)
```

## Loss Function

```python
def contrastive_loss(predicted, target, negatives, temperature=0.07):
    """InfoNCE contrastive loss."""
    B, D = predicted.shape
    
    # Normalize
    predicted = F.normalize(predicted, dim=-1)
    target = F.normalize(target, dim=-1)
    negatives = F.normalize(negatives, dim=-1)
    
    # Positive similarity
    pos_sim = (predicted * target).sum(dim=-1) / temperature  # [B]
    
    # Negative similarities
    neg_sim = torch.bmm(negatives, predicted.unsqueeze(-1)).squeeze(-1)
    neg_sim = neg_sim / temperature  # [B, N_neg]
    
    # Logits: [positive, negative1, negative2, ...]
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
    
    # Labels: position 0 is positive
    labels = torch.zeros(B, dtype=torch.long, device=predicted.device)
    
    return F.cross_entropy(logits, labels)
```

## Training Loop

### Standard Training (CPU Dataset)

```python
from RAG_supporters.dataset import create_loader, set_epoch

# Create dataloaders
train_loader = create_loader(
    dataset_dir="./my_dataset",
    split="train",
    batch_size=128,
    num_workers=4,
)

val_loader = create_loader(
    dataset_dir="./my_dataset",
    split="val",
    batch_size=128,
    num_workers=4,
)

# Create model
embedding_dim = train_loader.dataset_obj.embedding_dim
model = SimpleJASPERPredictor(embedding_dim).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Training
for epoch in range(100):
    # Update curriculum
    set_epoch(train_loader, epoch)
    
    model.train()
    for batch in train_loader:
        question_emb = batch["question_emb"].cuda()
        target_emb = batch["target_source_emb"].cuda()
        steering = batch["steering"].cuda()
        negatives = batch["negative_embs"].cuda()
        
        # Forward + backward
        predicted = model(question_emb, steering)
        loss = contrastive_loss(predicted, target_emb, negatives)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Evaluate
    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            val_loss = evaluate(model, val_loader)
        print(f"Epoch {epoch}: Val Loss = {val_loss:.4f}")
```

### GPU Preloading (Faster Training)

```python
import torch
from RAG_supporters.pytorch_datasets import JASPERSteeringDataset, create_loader
from torch.utils.data import DataLoader

# Load datasets directly to GPU (10-20% faster)
train_dataset = JASPERSteeringDataset(
    dataset_dir="./my_dataset",
    split="train",
    epoch=0,
    device=torch.device("cuda")  # All tensors preloaded to GPU
)

val_dataset = JASPERSteeringDataset(
    dataset_dir="./my_dataset",
    split="val",
    epoch=0,
    device=torch.device("cuda")
)

print(f"Train dataset: {train_dataset.memory_usage_mb:.2f} MB on {train_dataset.device}")
print(f"Val dataset: {val_dataset.memory_usage_mb:.2f} MB on {val_dataset.device}")

# Create dataloaders (no pin_memory needed, already on GPU)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# Training loop - NO .cuda() calls needed!
for epoch in range(100):
    train_dataset.set_epoch(epoch)
    
    model.train()
    for batch in train_loader:
        # All data already on GPU!
        question_emb = batch["question_emb"]
        target_emb = batch["target_source_emb"]
        steering = batch["steering"]
        negatives = batch["negative_embs"]
        
        predicted = model(question_emb, steering)
        loss = contrastive_loss(predicted, target_emb, negatives)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Context Manager Pattern (Clean Resource Management)

```python
from RAG_supporters.pytorch_datasets import JASPERSteeringDataset
from torch.utils.data import DataLoader

# Automatic cleanup with context manager
with JASPERSteeringDataset("./my_dataset", split="train", device="cuda") as train_dataset:
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    for epoch in range(100):
        train_dataset.set_epoch(epoch)
        
        for batch in train_loader:
            # Training code...
            pass
# Automatic cleanup and statistics logging
```

### Load All Splits at Once

```python
from RAG_supporters.pytorch_datasets import JASPERSteeringDataset
from torch.utils.data import DataLoader

# Load train/val/test in one call
splits = JASPERSteeringDataset.create_combined_splits(
    dataset_dir="./my_dataset",
    epoch=0,
    device=torch.device("cuda")
)

# Create loaders
train_loader = DataLoader(splits["train"], batch_size=128, shuffle=True)
val_loader = DataLoader(splits["val"], batch_size=128, shuffle=False)
test_loader = DataLoader(splits["test"], batch_size=128, shuffle=False)

print(f"Loaded {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test samples")
```
```

## Validation with Different Steering

```python
# Test with zero steering (baseline)
val_loader.dataset_obj.force_steering("zero")
zero_loss = evaluate(model, val_loader)

# Test with centroid steering (with guidance)
val_loader.dataset_obj.force_steering("centroid")
centroid_loss = evaluate(model, val_loader)

# Restore stochastic steering
val_loader.dataset_obj.force_steering(None)

print(f"Zero steering loss: {zero_loss:.4f}")
print(f"Centroid steering loss: {centroid_loss:.4f}")
print(f"Improvement: {(zero_loss - centroid_loss) / zero_loss * 100:.1f}%")
```

## Distributed Training

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

# Create distributed loader
train_loader = create_loader(
    dataset_dir="./my_dataset",
    split="train",
    batch_size=128,
    num_workers=4,
    distributed=True,
)

# Wrap model
model = SimpleJASPERPredictor(embedding_dim).cuda()
model = DDP(model, device_ids=[local_rank])

# Training loop
for epoch in range(100):
    set_epoch(train_loader, epoch)  # Updates both sampler and dataset
    
    for batch in train_loader:
        # Train...
```

## Hot-Reloading Negatives

```python
# Every 10 epochs, refresh negatives
for epoch in range(100):
    if epoch % 10 == 0 and epoch > 0:
        # External script updates hard_negatives.pt
        # (e.g., re-mine with current model)
        
        train_loader.dataset_obj.reload_negatives()
        print(f"Reloaded negatives at epoch {epoch}")
    
    # Train...
```

## Monitoring Steering Distribution

```python
from collections import Counter

# Check steering variant distribution
variants = []
for batch in train_loader:
    variants.extend(batch["steering_variant"].tolist())

dist = Counter(variants)
print("Steering distribution:")
print(f"  Zero: {dist[0] / len(variants) * 100:.1f}%")
print(f"  Centroid: {dist[1] / len(variants) * 100:.1f}%")
print(f"  Keyword: {dist[2] / len(variants) * 100:.1f}%")
print(f"  Residual: {dist[3] / len(variants) * 100:.1f}%")
```

## Complete Example

See `RAG_supporters/dataset/dataset_builder_README.md` for building the dataset.

Full training script structure:

```python
#!/usr/bin/env python3
"""Train JASPER predictor on steering dataset."""

import torch
from RAG_supporters.dataset import create_loader, set_epoch, validate_first_batch

def main():
    # Configuration
    DATASET_DIR = "./my_dataset"
    BATCH_SIZE = 128
    NUM_EPOCHS = 100
    
    # Create loaders
    train_loader = create_loader(DATASET_DIR, "train", BATCH_SIZE, num_workers=4)
    val_loader = create_loader(DATASET_DIR, "val", BATCH_SIZE, num_workers=4)
    
    # Validate
    validate_first_batch(train_loader)
    
    # Create model
    embedding_dim = train_loader.dataset_obj.embedding_dim
    model = SimpleJASPERPredictor(embedding_dim).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    # Train
    for epoch in range(NUM_EPOCHS):
        set_epoch(train_loader, epoch)
        train_loss = train_epoch(model, train_loader, optimizer)
        
        if epoch % 10 == 0:
            val_loss = evaluate(model, val_loader)
            print(f"Epoch {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}")

if __name__ == "__main__":
    main()
```

## Tips

1. **GPU preloading for speed**: Use `device=torch.device("cuda")` for 10-20% faster training (datasets < 10GB)
2. **Context manager for cleanup**: Use `with` statement to ensure proper resource cleanup
3. **Load all splits at once**: Use `create_combined_splits()` for cleaner code
4. **Start small**: Use small batch size and embedding dim for debugging
5. **Monitor curriculum**: Log steering probabilities per epoch
6. **Ablation studies**: Compare with/without steering
7. **Negative tiers**: Analyze which tier negatives are hardest
8. **Cluster analysis**: Check if model learns cluster structure
9. **Memory management**: Monitor `memory_usage_mb` attribute to avoid OOM
10. **Index validation**: Dataset automatically validates indices and raises clear errors

````
