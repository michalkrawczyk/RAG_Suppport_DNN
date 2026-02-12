"""Quick validation script for Task 6 (Hard Negative Miner).

This script validates that:
1. NegativeMiner can be imported and instantiated
2. Mining produces correct shapes and valid outputs
3. Negatives integrate with existing pipeline components
"""

import tempfile
from pathlib import Path

import torch
import numpy as np

def validate_task6():
    """Validate Task 6 implementation."""
    print("=" * 60)
    print("Task 6 Validation: Hard Negative Miner")
    print("=" * 60)
    
    # Import the module
    print("\n[1/5] Testing imports...")
    try:
        from RAG_supporters.dataset import NegativeMiner, mine_negatives
        print("✓ Imports successful")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    
    # Create sample data
    print("\n[2/5] Creating sample data...")
    torch.manual_seed(42)
    np.random.seed(42)
    
    n_sources = 100
    n_questions = 20
    n_clusters = 5
    n_pairs = 50
    dim = 64
    n_neg = 12
    
    source_embs = torch.randn(n_sources, dim)
    question_embs = torch.randn(n_questions, dim)
    centroid_embs = torch.randn(n_clusters, dim)
    
    pair_indices = torch.stack([
        torch.randint(0, n_questions, (n_pairs,)),
        torch.randint(0, n_sources, (n_pairs,))
    ], dim=1)
    
    pair_cluster_ids = torch.randint(0, n_clusters, (n_pairs,))
    
    # Distribute sources across clusters
    source_cluster_ids = torch.zeros(n_sources, dtype=torch.long)
    sources_per_cluster = n_sources // n_clusters
    for i in range(n_clusters):
        start = i * sources_per_cluster
        end = start + sources_per_cluster if i < n_clusters - 1 else n_sources
        source_cluster_ids[start:end] = i
    
    print(f"✓ Created sample data: {n_pairs} pairs, {n_sources} sources, {n_clusters} clusters")
    
    # Initialize miner
    print("\n[3/5] Initializing NegativeMiner...")
    try:
        miner = NegativeMiner(
            source_embeddings=source_embs,
            question_embeddings=question_embs,
            centroid_embeddings=centroid_embs,
            pair_indices=pair_indices,
            pair_cluster_ids=pair_cluster_ids,
            source_cluster_ids=source_cluster_ids,
            n_neg=n_neg,
            tier_proportions=[3, 4, 3, 2],
            adjacent_k=3,
            random_seed=42,
            show_progress=False
        )
        print("✓ NegativeMiner initialized successfully")
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return False
    
    # Mine negatives
    print("\n[4/5] Mining hard negatives...")
    try:
        results = miner.mine_all_negatives()
        
        hard_negatives = results['hard_negatives']
        negative_tiers = results['negative_tiers']
        
        print(f"✓ Mining successful")
        print(f"  - hard_negatives shape: {hard_negatives.shape}")
        print(f"  - negative_tiers shape: {negative_tiers.shape}")
        
        # Validate shapes
        assert hard_negatives.shape == (n_pairs, n_neg), \
            f"Wrong shape: expected {(n_pairs, n_neg)}, got {hard_negatives.shape}"
        assert negative_tiers.shape == (n_pairs, n_neg), \
            f"Wrong shape: expected {(n_pairs, n_neg)}, got {negative_tiers.shape}"
        
        # Validate indices
        assert hard_negatives.min() >= 0, "Negative indices < 0"
        assert hard_negatives.max() < n_sources, f"Negative indices >= n_sources ({n_sources})"
        
        # Validate tiers
        assert negative_tiers.min() >= 1, "Tier labels < 1"
        assert negative_tiers.max() <= 4, "Tier labels > 4"
        
        # Check no true sources in negatives
        violations = 0
        for pair_idx in range(n_pairs):
            true_source = pair_indices[pair_idx, 1].item()
            if true_source in hard_negatives[pair_idx].tolist():
                violations += 1
        
        assert violations == 0, f"Found {violations} true sources in negatives"
        
        print(f"✓ Validation checks passed")
        print(f"  - Index range: [{hard_negatives.min()}, {hard_negatives.max()}]")
        print(f"  - Tier range: [{negative_tiers.min()}, {negative_tiers.max()}]")
        print(f"  - No true sources in negatives: ✓")
        
    except Exception as e:
        print(f"✗ Mining failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test save functionality
    print("\n[5/5] Testing save functionality...")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            miner.save(output_dir, results)
            
            # Check files exist
            assert (output_dir / "hard_negatives.pt").exists(), \
                "hard_negatives.pt not saved"
            assert (output_dir / "negative_tiers.pt").exists(), \
                "negative_tiers.pt not saved"
            
            # Load and verify
            loaded_negatives = torch.load(
                output_dir / "hard_negatives.pt", weights_only=True
            )
            loaded_tiers = torch.load(
                output_dir / "negative_tiers.pt", weights_only=True
            )
            
            assert torch.equal(loaded_negatives, hard_negatives), \
                "Loaded negatives don't match"
            assert torch.equal(loaded_tiers, negative_tiers), \
                "Loaded tiers don't match"
            
            print("✓ Save/load successful")
    
    except Exception as e:
        print(f"✗ Save/load failed: {e}")
        return False
    
    # Test convenience function
    print("\n[BONUS] Testing convenience function...")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            results2 = mine_negatives(
                source_embeddings=source_embs,
                question_embeddings=question_embs,
                centroid_embeddings=centroid_embs,
                pair_indices=pair_indices,
                pair_cluster_ids=pair_cluster_ids,
                source_cluster_ids=source_cluster_ids,
                n_neg=n_neg,
                output_dir=tmpdir,
                tier_proportions=[3, 4, 3, 2],
                random_seed=42,
                show_progress=False
            )
            
            print("✓ Convenience function works")
    except Exception as e:
        print(f"✗ Convenience function failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✓ All validation checks passed!")
    print("=" * 60)
    print("\nTask 6 implementation is working correctly.")
    print("Ready to proceed with Task 7 (Dataset Splitting).")
    
    return True


if __name__ == "__main__":
    success = validate_task6()
    exit(0 if success else 1)
