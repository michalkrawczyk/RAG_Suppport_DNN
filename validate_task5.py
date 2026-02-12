#!/usr/bin/env python3
"""Quick validation script for Task 5: Steering Builder."""

import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from RAG_supporters.dataset import SteeringBuilder, build_steering


def validate_task5():
    """Validate Task 5 implementation."""
    print("=== Task 5: Steering Builder Validation ===\n")
    
    # Create sample data
    print("Creating sample embeddings...")
    torch.manual_seed(42)
    
    n_questions = 10
    n_keywords = 5
    n_clusters = 3
    n_pairs = 20
    dim = 64
    
    question_embs = torch.randn(n_questions, dim)
    question_embs = question_embs / torch.norm(question_embs, dim=1, keepdim=True)
    
    keyword_embs = torch.randn(n_keywords, dim)
    keyword_embs = keyword_embs / torch.norm(keyword_embs, dim=1, keepdim=True)
    
    centroid_embs = torch.randn(n_clusters, dim)
    centroid_embs = centroid_embs / torch.norm(centroid_embs, dim=1, keepdim=True)
    
    pair_indices = torch.stack([
        torch.randint(0, n_questions, (n_pairs,)),
        torch.randint(0, n_questions, (n_pairs,))
    ], dim=1)
    
    pair_cluster_ids = torch.randint(0, n_clusters, (n_pairs,))
    
    # Create variable-length keyword lists
    pair_keyword_ids = []
    for i in range(n_pairs):
        if i % 5 == 0:
            pair_keyword_ids.append([])
        elif i % 3 == 0:
            pair_keyword_ids.append([i % n_keywords])
        else:
            n_kw = 2 + (i % 2)
            pair_keyword_ids.append([j % n_keywords for j in range(i, i + n_kw)])
    
    print(f"✓ Created embeddings: {n_questions} questions, {n_keywords} keywords, {n_clusters} clusters")
    print(f"✓ Created {n_pairs} pairs\n")
    
    # Test SteeringBuilder
    print("Testing SteeringBuilder...")
    builder = SteeringBuilder(
        question_embeddings=question_embs,
        keyword_embeddings=keyword_embs,
        centroid_embeddings=centroid_embs,
        pair_indices=pair_indices,
        pair_cluster_ids=pair_cluster_ids,
        pair_keyword_ids=pair_keyword_ids,
        show_progress=False
    )
    print(f"✓ Initialized SteeringBuilder\n")
    
    # Build all steering variants
    print("Building steering signals...")
    results = builder.build_all_steering()
    
    print(f"✓ Centroid steering: {results['centroid'].shape}")
    print(f"✓ Keyword-weighted steering: {results['keyword_weighted'].shape}")
    print(f"✓ Residual steering: {results['residual'].shape}")
    print(f"✓ Centroid distances: {results['distances'].shape}\n")
    
    # Validate properties
    print("Validating steering properties...")
    
    # Check shapes
    assert results['centroid'].shape == (n_pairs, dim), "Centroid steering shape mismatch"
    assert results['keyword_weighted'].shape == (n_pairs, dim), "Keyword steering shape mismatch"
    assert results['residual'].shape == (n_pairs, dim), "Residual steering shape mismatch"
    assert results['distances'].shape == (n_pairs,), "Distances shape mismatch"
    print("✓ All shapes correct")
    
    # Check normalization
    centroid_norms = torch.norm(results['centroid'], dim=1)
    keyword_norms = torch.norm(results['keyword_weighted'], dim=1)
    
    non_zero_centroid = centroid_norms[centroid_norms > 1e-8]
    non_zero_keyword = keyword_norms[keyword_norms > 1e-8]
    
    assert torch.allclose(non_zero_centroid, torch.ones_like(non_zero_centroid), atol=1e-5), \
        "Centroid steering not normalized"
    assert torch.allclose(non_zero_keyword, torch.ones_like(non_zero_keyword), atol=1e-5), \
        "Keyword steering not normalized"
    print("✓ Steering vectors are unit normalized")
    
    # Check distance range
    assert (results['distances'] >= -1e-6).all(), "Distances below 0"
    assert (results['distances'] <= 2.0 + 1e-6).all(), "Distances above 2"
    print("✓ Distances in valid range [0, 2]")
    
    # Check no NaN/Inf
    for key, tensor in results.items():
        assert not torch.isnan(tensor).any(), f"{key} contains NaN"
        assert not torch.isinf(tensor).any(), f"{key} contains Inf"
    print("✓ No NaN or Inf values")
    
    print("\n=== Validation Summary ===")
    print(f"Distance statistics:")
    print(f"  Mean: {results['distances'].mean():.3f}")
    print(f"  Std:  {results['distances'].std():.3f}")
    print(f"  Min:  {results['distances'].min():.3f}")
    print(f"  Max:  {results['distances'].max():.3f}")
    
    n_no_keywords = sum(1 for kw_ids in pair_keyword_ids if len(kw_ids) == 0)
    print(f"\nPairs with no keywords: {n_no_keywords}/{n_pairs} ({100*n_no_keywords/n_pairs:.1f}%)")
    
    print("\n✅ All validations passed!")
    print("Task 5: Steering Builder implementation is complete and working correctly.")
    
    return True


if __name__ == "__main__":
    try:
        validate_task5()
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
