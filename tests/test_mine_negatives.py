"""Tests for Hard Negative Miner."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from RAG_supporters.contrastive import (
    NegativeMiner,
    mine_negatives,
)


@pytest.fixture
def sample_data():
    """Create sample data for testing.
    
    Creates:
    - 100 sources with embeddings
    - 20 questions with embeddings
    - 5 clusters with centroids
    - 50 pairs with cluster assignments
    """
    torch.manual_seed(42)
    np.random.seed(42)
    
    n_sources = 100
    n_questions = 20
    n_clusters = 5
    n_pairs = 50
    dim = 64
    
    # Generate normalized embeddings
    source_embs = torch.randn(n_sources, dim)
    source_embs = source_embs / torch.norm(source_embs, dim=1, keepdim=True)
    
    question_embs = torch.randn(n_questions, dim)
    question_embs = question_embs / torch.norm(question_embs, dim=1, keepdim=True)
    
    centroid_embs = torch.randn(n_clusters, dim)
    centroid_embs = centroid_embs / torch.norm(centroid_embs, dim=1, keepdim=True)
    
    # Create pair indices (question_idx, source_idx)
    pair_indices = torch.stack([
        torch.randint(0, n_questions, (n_pairs,)),
        torch.randint(0, n_sources, (n_pairs,))
    ], dim=1)
    
    # Assign clusters to pairs
    pair_cluster_ids = torch.randint(0, n_clusters, (n_pairs,))
    
    # Assign clusters to sources (distribute evenly)
    source_cluster_ids = torch.zeros(n_sources, dtype=torch.long)
    sources_per_cluster = n_sources // n_clusters
    for i in range(n_clusters):
        start = i * sources_per_cluster
        end = start + sources_per_cluster if i < n_clusters - 1 else n_sources
        source_cluster_ids[start:end] = i
    
    return {
        "source_embeddings": source_embs,
        "question_embeddings": question_embs,
        "centroid_embeddings": centroid_embs,
        "pair_indices": pair_indices,
        "pair_cluster_ids": pair_cluster_ids,
        "source_cluster_ids": source_cluster_ids,
        "n_sources": n_sources,
        "n_questions": n_questions,
        "n_clusters": n_clusters,
        "n_pairs": n_pairs,
        "dim": dim
    }


class TestNegativeMinerInit:
    """Test NegativeMiner initialization."""
    
    def test_init_valid(self, sample_data):
        """Test initialization with valid inputs."""
        miner = NegativeMiner(
            source_embeddings=sample_data["source_embeddings"],
            question_embeddings=sample_data["question_embeddings"],
            centroid_embeddings=sample_data["centroid_embeddings"],
            pair_indices=sample_data["pair_indices"],
            pair_cluster_ids=sample_data["pair_cluster_ids"],
            source_cluster_ids=sample_data["source_cluster_ids"],
            n_neg=12
        )
        
        assert miner.n_pairs == sample_data["n_pairs"], \
            "Should have correct number of pairs"
        assert miner.n_sources == sample_data["n_sources"], \
            "Should have correct number of sources"
        assert miner.n_clusters == sample_data["n_clusters"], \
            "Should have correct number of clusters"
        assert miner.n_neg == 12, \
            "Should have correct n_neg"
        assert miner.embedding_dim == sample_data["dim"], \
            "Should have correct embedding dimension"
    
    def test_init_with_tier_proportions(self, sample_data):
        """Test initialization with custom tier proportions."""
        miner = NegativeMiner(
            source_embeddings=sample_data["source_embeddings"],
            question_embeddings=sample_data["question_embeddings"],
            centroid_embeddings=sample_data["centroid_embeddings"],
            pair_indices=sample_data["pair_indices"],
            pair_cluster_ids=sample_data["pair_cluster_ids"],
            source_cluster_ids=sample_data["source_cluster_ids"],
            n_neg=12,
            tier_proportions=[3, 4, 3, 2]
        )
        
        assert miner.tier_proportions == [3, 4, 3, 2], \
            "Should have custom tier proportions"
    
    def test_init_auto_tier_proportions(self, sample_data):
        """Test automatic tier proportion distribution."""
        miner = NegativeMiner(
            source_embeddings=sample_data["source_embeddings"],
            question_embeddings=sample_data["question_embeddings"],
            centroid_embeddings=sample_data["centroid_embeddings"],
            pair_indices=sample_data["pair_indices"],
            pair_cluster_ids=sample_data["pair_cluster_ids"],
            source_cluster_ids=sample_data["source_cluster_ids"],
            n_neg=12
        )
        
        assert sum(miner.tier_proportions) == 12, \
            "Auto tier proportions should sum to n_neg"
        assert len(miner.tier_proportions) == 4, \
            "Should have 4 tiers"
    
    def test_init_invalid_source_embeddings_type(self, sample_data):
        """Test initialization with invalid source_embeddings type."""
        with pytest.raises(TypeError, match="source_embeddings must be torch.Tensor"):
            NegativeMiner(
                source_embeddings=sample_data["source_embeddings"].numpy(),
                question_embeddings=sample_data["question_embeddings"],
                centroid_embeddings=sample_data["centroid_embeddings"],
                pair_indices=sample_data["pair_indices"],
                pair_cluster_ids=sample_data["pair_cluster_ids"],
                source_cluster_ids=sample_data["source_cluster_ids"],
                n_neg=12
            )
    
    def test_init_invalid_dimension_mismatch(self, sample_data):
        """Test initialization with dimension mismatch."""
        wrong_dim_embs = torch.randn(20, 32)  # Wrong dimension
        
        with pytest.raises(ValueError, match="Embedding dimensions must match"):
            NegativeMiner(
                source_embeddings=sample_data["source_embeddings"],
                question_embeddings=wrong_dim_embs,
                centroid_embeddings=sample_data["centroid_embeddings"],
                pair_indices=sample_data["pair_indices"],
                pair_cluster_ids=sample_data["pair_cluster_ids"],
                source_cluster_ids=sample_data["source_cluster_ids"],
                n_neg=12
            )
    
    def test_init_invalid_n_neg(self, sample_data):
        """Test initialization with invalid n_neg."""
        with pytest.raises(ValueError, match="n_neg must be a positive integer"):
            NegativeMiner(
                source_embeddings=sample_data["source_embeddings"],
                question_embeddings=sample_data["question_embeddings"],
                centroid_embeddings=sample_data["centroid_embeddings"],
                pair_indices=sample_data["pair_indices"],
                pair_cluster_ids=sample_data["pair_cluster_ids"],
                source_cluster_ids=sample_data["source_cluster_ids"],
                n_neg=0
            )
    
    def test_init_invalid_tier_proportions_sum(self, sample_data):
        """Test initialization with tier proportions that don't sum to n_neg."""
        with pytest.raises(ValueError, match="tier_proportions must sum to n_neg"):
            NegativeMiner(
                source_embeddings=sample_data["source_embeddings"],
                question_embeddings=sample_data["question_embeddings"],
                centroid_embeddings=sample_data["centroid_embeddings"],
                pair_indices=sample_data["pair_indices"],
                pair_cluster_ids=sample_data["pair_cluster_ids"],
                source_cluster_ids=sample_data["source_cluster_ids"],
                n_neg=12,
                tier_proportions=[3, 3, 3, 2]  # Sums to 11, not 12
            )
    
    def test_init_invalid_tier_proportions_length(self, sample_data):
        """Test initialization with wrong number of tier proportions."""
        with pytest.raises(ValueError, match="tier_proportions must have exactly 4 values"):
            NegativeMiner(
                source_embeddings=sample_data["source_embeddings"],
                question_embeddings=sample_data["question_embeddings"],
                centroid_embeddings=sample_data["centroid_embeddings"],
                pair_indices=sample_data["pair_indices"],
                pair_cluster_ids=sample_data["pair_cluster_ids"],
                source_cluster_ids=sample_data["source_cluster_ids"],
                n_neg=12,
                tier_proportions=[3, 4, 5]  # Only 3 values
            )
    
    def test_init_cluster_structures_built(self, sample_data):
        """Test that cluster structures are built during initialization."""
        miner = NegativeMiner(
            source_embeddings=sample_data["source_embeddings"],
            question_embeddings=sample_data["question_embeddings"],
            centroid_embeddings=sample_data["centroid_embeddings"],
            pair_indices=sample_data["pair_indices"],
            pair_cluster_ids=sample_data["pair_cluster_ids"],
            source_cluster_ids=sample_data["source_cluster_ids"],
            n_neg=12
        )
        
        assert hasattr(miner, 'cluster_sources'), \
            "Should have cluster_sources mapping"
        assert len(miner.cluster_sources) == sample_data["n_clusters"], \
            "Should have entry for each cluster"
        
        # Check that all sources are assigned
        total_sources = sum(len(sources) for sources in miner.cluster_sources.values())
        assert total_sources == sample_data["n_sources"], \
            "All sources should be assigned to clusters"


class TestNegativeMinerMining:
    """Test negative mining functionality."""
    
    def test_mine_all_negatives_shape(self, sample_data):
        """Test that mine_all_negatives returns correct shapes."""
        miner = NegativeMiner(
            source_embeddings=sample_data["source_embeddings"],
            question_embeddings=sample_data["question_embeddings"],
            centroid_embeddings=sample_data["centroid_embeddings"],
            pair_indices=sample_data["pair_indices"],
            pair_cluster_ids=sample_data["pair_cluster_ids"],
            source_cluster_ids=sample_data["source_cluster_ids"],
            n_neg=12,
            show_progress=False
        )
        
        results = miner.mine_all_negatives()
        
        assert 'hard_negatives' in results, \
            "Results should contain hard_negatives"
        assert 'negative_tiers' in results, \
            "Results should contain negative_tiers"
        
        assert results['hard_negatives'].shape == (sample_data["n_pairs"], 12), \
            f"hard_negatives shape should be {(sample_data['n_pairs'], 12)}"
        assert results['negative_tiers'].shape == (sample_data["n_pairs"], 12), \
            f"negative_tiers shape should be {(sample_data['n_pairs'], 12)}"
    
    def test_mine_all_negatives_indices_valid(self, sample_data):
        """Test that all negative indices are valid."""
        miner = NegativeMiner(
            source_embeddings=sample_data["source_embeddings"],
            question_embeddings=sample_data["question_embeddings"],
            centroid_embeddings=sample_data["centroid_embeddings"],
            pair_indices=sample_data["pair_indices"],
            pair_cluster_ids=sample_data["pair_cluster_ids"],
            source_cluster_ids=sample_data["source_cluster_ids"],
            n_neg=12,
            show_progress=False
        )
        
        results = miner.mine_all_negatives()
        hard_negatives = results['hard_negatives']
        
        assert hard_negatives.min() >= 0, \
            "All negative indices should be >= 0"
        assert hard_negatives.max() < sample_data["n_sources"], \
            f"All negative indices should be < n_sources={sample_data['n_sources']}"
    
    def test_mine_all_negatives_tiers_valid(self, sample_data):
        """Test that all tier labels are valid."""
        miner = NegativeMiner(
            source_embeddings=sample_data["source_embeddings"],
            question_embeddings=sample_data["question_embeddings"],
            centroid_embeddings=sample_data["centroid_embeddings"],
            pair_indices=sample_data["pair_indices"],
            pair_cluster_ids=sample_data["pair_cluster_ids"],
            source_cluster_ids=sample_data["source_cluster_ids"],
            n_neg=12,
            show_progress=False
        )
        
        results = miner.mine_all_negatives()
        negative_tiers = results['negative_tiers']
        
        assert negative_tiers.min() >= 1, \
            "All tier labels should be >= 1"
        assert negative_tiers.max() <= 4, \
            "All tier labels should be <= 4"
    
    def test_mine_all_negatives_no_true_source(self, sample_data):
        """Test that true source is never in negatives."""
        miner = NegativeMiner(
            source_embeddings=sample_data["source_embeddings"],
            question_embeddings=sample_data["question_embeddings"],
            centroid_embeddings=sample_data["centroid_embeddings"],
            pair_indices=sample_data["pair_indices"],
            pair_cluster_ids=sample_data["pair_cluster_ids"],
            source_cluster_ids=sample_data["source_cluster_ids"],
            n_neg=12,
            show_progress=False
        )
        
        results = miner.mine_all_negatives()
        hard_negatives = results['hard_negatives']
        
        violations = 0
        for pair_idx in range(sample_data["n_pairs"]):
            true_source = sample_data["pair_indices"][pair_idx, 1].item()
            pair_negatives = hard_negatives[pair_idx].tolist()
            if true_source in pair_negatives:
                violations += 1
        
        assert violations == 0, \
            f"True source should never be in negatives, found {violations} violations"
    
    def test_mine_all_negatives_tier_distribution(self, sample_data):
        """Test that tier distribution matches proportions."""
        tier_proportions = [3, 4, 3, 2]
        miner = NegativeMiner(
            source_embeddings=sample_data["source_embeddings"],
            question_embeddings=sample_data["question_embeddings"],
            centroid_embeddings=sample_data["centroid_embeddings"],
            pair_indices=sample_data["pair_indices"],
            pair_cluster_ids=sample_data["pair_cluster_ids"],
            source_cluster_ids=sample_data["source_cluster_ids"],
            n_neg=12,
            tier_proportions=tier_proportions,
            show_progress=False
        )
        
        results = miner.mine_all_negatives()
        negative_tiers = results['negative_tiers']
        
        # Count occurrences of each tier
        for tier_id, expected_count in enumerate(tier_proportions, start=1):
            actual_count = (negative_tiers == tier_id).sum().item()
            expected_total = expected_count * sample_data["n_pairs"]
            
            # Allow some tolerance due to insufficient negatives in some tiers
            assert actual_count <= expected_total * 1.2, \
                f"Tier {tier_id} count {actual_count} exceeds expected {expected_total} by too much"
    
    def test_mine_single_pair(self, sample_data):
        """Test mining negatives for single pair."""
        # Create dataset with single pair
        pair_indices = sample_data["pair_indices"][:1]
        pair_cluster_ids = sample_data["pair_cluster_ids"][:1]
        
        miner = NegativeMiner(
            source_embeddings=sample_data["source_embeddings"],
            question_embeddings=sample_data["question_embeddings"],
            centroid_embeddings=sample_data["centroid_embeddings"],
            pair_indices=pair_indices,
            pair_cluster_ids=pair_cluster_ids,
            source_cluster_ids=sample_data["source_cluster_ids"],
            n_neg=12,
            show_progress=False
        )
        
        results = miner.mine_all_negatives()
        
        assert results['hard_negatives'].shape == (1, 12), \
            "Should mine 12 negatives for single pair"
        assert results['negative_tiers'].shape == (1, 12), \
            "Should have tier labels for single pair"
    
    def test_mine_with_small_n_neg(self, sample_data):
        """Test mining with small number of negatives."""
        miner = NegativeMiner(
            source_embeddings=sample_data["source_embeddings"],
            question_embeddings=sample_data["question_embeddings"],
            centroid_embeddings=sample_data["centroid_embeddings"],
            pair_indices=sample_data["pair_indices"],
            pair_cluster_ids=sample_data["pair_cluster_ids"],
            source_cluster_ids=sample_data["source_cluster_ids"],
            n_neg=4,
            tier_proportions=[1, 1, 1, 1],
            show_progress=False
        )
        
        results = miner.mine_all_negatives()
        
        assert results['hard_negatives'].shape[1] == 4, \
            "Should mine 4 negatives per pair"
    
    def test_mine_with_large_n_neg(self, sample_data):
        """Test mining with large number of negatives."""
        miner = NegativeMiner(
            source_embeddings=sample_data["source_embeddings"],
            question_embeddings=sample_data["question_embeddings"],
            centroid_embeddings=sample_data["centroid_embeddings"],
            pair_indices=sample_data["pair_indices"],
            pair_cluster_ids=sample_data["pair_cluster_ids"],
            source_cluster_ids=sample_data["source_cluster_ids"],
            n_neg=50,
            show_progress=False
        )
        
        results = miner.mine_all_negatives()
        
        assert results['hard_negatives'].shape[1] == 50, \
            "Should mine 50 negatives per pair"


class TestNegativeMinerEdgeCases:
    """Test edge cases for negative mining."""
    
    def test_small_cluster(self):
        """Test mining with very small clusters."""
        torch.manual_seed(42)
        
        # Create minimal dataset: 3 clusters with 2 sources each
        n_sources = 6
        n_clusters = 3
        dim = 64
        
        source_embs = torch.randn(n_sources, dim)
        question_embs = torch.randn(2, dim)
        centroid_embs = torch.randn(n_clusters, dim)
        
        # 2 sources per cluster
        source_cluster_ids = torch.tensor([0, 0, 1, 1, 2, 2])
        
        # Create pairs
        pair_indices = torch.tensor([[0, 0], [0, 2]])  # 2 pairs
        pair_cluster_ids = torch.tensor([0, 1])
        
        miner = NegativeMiner(
            source_embeddings=source_embs,
            question_embeddings=question_embs,
            centroid_embeddings=centroid_embs,
            pair_indices=pair_indices,
            pair_cluster_ids=pair_cluster_ids,
            source_cluster_ids=source_cluster_ids,
            n_neg=4,
            tier_proportions=[1, 1, 1, 1],
            show_progress=False
        )
        
        results = miner.mine_all_negatives()
        
        # Should still produce correct shape
        assert results['hard_negatives'].shape == (2, 4), \
            "Should mine 4 negatives even with small clusters"
    
    def test_single_cluster(self):
        """Test mining when all sources in single cluster."""
        torch.manual_seed(42)
        
        n_sources = 10
        dim = 64
        
        source_embs = torch.randn(n_sources, dim)
        question_embs = torch.randn(2, dim)
        centroid_embs = torch.randn(1, dim)  # Single cluster
        
        # All sources in cluster 0
        source_cluster_ids = torch.zeros(n_sources, dtype=torch.long)
        
        pair_indices = torch.tensor([[0, 0], [0, 1]])
        pair_cluster_ids = torch.zeros(2, dtype=torch.long)
        
        miner = NegativeMiner(
            source_embeddings=source_embs,
            question_embeddings=question_embs,
            centroid_embeddings=centroid_embs,
            pair_indices=pair_indices,
            pair_cluster_ids=pair_cluster_ids,
            source_cluster_ids=source_cluster_ids,
            n_neg=8,
            show_progress=False
        )
        
        results = miner.mine_all_negatives()
        
        # Should gracefully handle lack of adjacent clusters
        assert results['hard_negatives'].shape == (2, 8), \
            "Should mine negatives even with single cluster"
    
    def test_deterministic_with_seed(self, sample_data):
        """Test that results are deterministic with same seed."""
        miner1 = NegativeMiner(
            source_embeddings=sample_data["source_embeddings"],
            question_embeddings=sample_data["question_embeddings"],
            centroid_embeddings=sample_data["centroid_embeddings"],
            pair_indices=sample_data["pair_indices"],
            pair_cluster_ids=sample_data["pair_cluster_ids"],
            source_cluster_ids=sample_data["source_cluster_ids"],
            n_neg=12,
            random_seed=123,
            show_progress=False
        )
        
        miner2 = NegativeMiner(
            source_embeddings=sample_data["source_embeddings"],
            question_embeddings=sample_data["question_embeddings"],
            centroid_embeddings=sample_data["centroid_embeddings"],
            pair_indices=sample_data["pair_indices"],
            pair_cluster_ids=sample_data["pair_cluster_ids"],
            source_cluster_ids=sample_data["source_cluster_ids"],
            n_neg=12,
            random_seed=123,
            show_progress=False
        )
        
        results1 = miner1.mine_all_negatives()
        results2 = miner2.mine_all_negatives()
        
        assert torch.equal(results1['hard_negatives'], results2['hard_negatives']), \
            "Results should be identical with same seed"
        assert torch.equal(results1['negative_tiers'], results2['negative_tiers']), \
            "Tier labels should be identical with same seed"


class TestNegativeMinerSaveLoad:
    """Test saving functionality."""
    
    def test_save(self, sample_data):
        """Test saving negatives to disk."""
        miner = NegativeMiner(
            source_embeddings=sample_data["source_embeddings"],
            question_embeddings=sample_data["question_embeddings"],
            centroid_embeddings=sample_data["centroid_embeddings"],
            pair_indices=sample_data["pair_indices"],
            pair_cluster_ids=sample_data["pair_cluster_ids"],
            source_cluster_ids=sample_data["source_cluster_ids"],
            n_neg=12,
            show_progress=False
        )
        
        results = miner.mine_all_negatives()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            miner.save(output_dir, results)
            
            # Check files exist
            assert (output_dir / "hard_negatives.pt").exists(), \
                "Should save hard_negatives.pt"
            assert (output_dir / "negative_tiers.pt").exists(), \
                "Should save negative_tiers.pt"
            
            # Load and verify
            loaded_negatives = torch.load(output_dir / "hard_negatives.pt", weights_only=True)
            loaded_tiers = torch.load(output_dir / "negative_tiers.pt", weights_only=True)
            
            assert torch.equal(loaded_negatives, results['hard_negatives']), \
                "Loaded negatives should match original"
            assert torch.equal(loaded_tiers, results['negative_tiers']), \
                "Loaded tiers should match original"


class TestMineNegativesFunction:
    """Test convenience function."""
    
    def test_mine_negatives_function(self, sample_data):
        """Test mine_negatives convenience function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            
            results = mine_negatives(
                source_embeddings=sample_data["source_embeddings"],
                question_embeddings=sample_data["question_embeddings"],
                centroid_embeddings=sample_data["centroid_embeddings"],
                pair_indices=sample_data["pair_indices"],
                pair_cluster_ids=sample_data["pair_cluster_ids"],
                source_cluster_ids=sample_data["source_cluster_ids"],
                n_neg=12,
                output_dir=output_dir,
                tier_proportions=[3, 4, 3, 2],
                show_progress=False
            )
            
            # Check results
            assert 'hard_negatives' in results, \
                "Should return hard_negatives"
            assert 'negative_tiers' in results, \
                "Should return negative_tiers"
            
            # Check files saved
            assert (output_dir / "hard_negatives.pt").exists(), \
                "Should save hard_negatives.pt"
            assert (output_dir / "negative_tiers.pt").exists(), \
                "Should save negative_tiers.pt"
    
    def test_mine_negatives_function_returns_correct_shape(self, sample_data):
        """Test that convenience function returns correct shapes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = mine_negatives(
                source_embeddings=sample_data["source_embeddings"],
                question_embeddings=sample_data["question_embeddings"],
                centroid_embeddings=sample_data["centroid_embeddings"],
                pair_indices=sample_data["pair_indices"],
                pair_cluster_ids=sample_data["pair_cluster_ids"],
                source_cluster_ids=sample_data["source_cluster_ids"],
                n_neg=12,
                output_dir=tmpdir,
                show_progress=False
            )
            
            assert results['hard_negatives'].shape == (sample_data["n_pairs"], 12), \
                "Should return correct hard_negatives shape"
            assert results['negative_tiers'].shape == (sample_data["n_pairs"], 12), \
                "Should return correct negative_tiers shape"
