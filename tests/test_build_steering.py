"""Tests for Steering Signal Builder."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from RAG_supporters.contrastive import (
    SteeringBuilder,
    build_steering,
)


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing.

    Creates:
    - 10 questions with embeddings
    - 5 keywords with embeddings
    - 3 clusters with centroids
    - 20 pairs with cluster and keyword assignments
    """
    torch.manual_seed(42)
    np.random.seed(42)

    n_questions = 10
    n_keywords = 5
    n_clusters = 3
    n_pairs = 20
    dim = 64

    # Generate normalized embeddings
    question_embs = torch.randn(n_questions, dim)
    question_embs = question_embs / torch.norm(question_embs, dim=1, keepdim=True)

    keyword_embs = torch.randn(n_keywords, dim)
    keyword_embs = keyword_embs / torch.norm(keyword_embs, dim=1, keepdim=True)

    centroid_embs = torch.randn(n_clusters, dim)
    centroid_embs = centroid_embs / torch.norm(centroid_embs, dim=1, keepdim=True)

    # Create pair indices (question_idx, source_idx)
    # We don't actually use source_idx for steering, but need it for completeness
    pair_indices = torch.stack(
        [
            torch.randint(0, n_questions, (n_pairs,)),
            torch.randint(0, n_questions, (n_pairs,)),  # Just reuse questions as sources
        ],
        dim=1,
    )

    # Assign clusters
    pair_cluster_ids = torch.randint(0, n_clusters, (n_pairs,))

    # Assign keywords (variable length, some empty)
    pair_keyword_ids = []
    for i in range(n_pairs):
        if i % 5 == 0:
            # Every 5th pair has no keywords
            pair_keyword_ids.append([])
        elif i % 3 == 0:
            # Every 3rd pair has 1 keyword
            pair_keyword_ids.append([i % n_keywords])
        else:
            # Others have 2-3 keywords
            n_kw = 2 + (i % 2)
            kw_ids = [j % n_keywords for j in range(i, i + n_kw)]
            pair_keyword_ids.append(kw_ids)

    return {
        "question_embeddings": question_embs,
        "keyword_embeddings": keyword_embs,
        "centroid_embeddings": centroid_embs,
        "pair_indices": pair_indices,
        "pair_cluster_ids": pair_cluster_ids,
        "pair_keyword_ids": pair_keyword_ids,
        "n_pairs": n_pairs,
        "dim": dim,
    }


class TestSteeringBuilderInit:
    """Test SteeringBuilder initialization."""

    def test_init_valid(self, sample_embeddings):
        """Test initialization with valid inputs."""
        builder = SteeringBuilder(
            question_embeddings=sample_embeddings["question_embeddings"],
            keyword_embeddings=sample_embeddings["keyword_embeddings"],
            centroid_embeddings=sample_embeddings["centroid_embeddings"],
            pair_indices=sample_embeddings["pair_indices"],
            pair_cluster_ids=sample_embeddings["pair_cluster_ids"],
            pair_keyword_ids=sample_embeddings["pair_keyword_ids"],
        )

        assert (
            builder.n_pairs == sample_embeddings["n_pairs"]
        ), "Should have correct number of pairs"
        assert (
            builder.embedding_dim == sample_embeddings["dim"]
        ), "Should have correct embedding dimension"
        assert builder.fallback_strategy == "centroid", "Should have default fallback strategy"

    def test_init_custom_fallback(self, sample_embeddings):
        """Test initialization with custom fallback strategy."""
        builder = SteeringBuilder(
            question_embeddings=sample_embeddings["question_embeddings"],
            keyword_embeddings=sample_embeddings["keyword_embeddings"],
            centroid_embeddings=sample_embeddings["centroid_embeddings"],
            pair_indices=sample_embeddings["pair_indices"],
            pair_cluster_ids=sample_embeddings["pair_cluster_ids"],
            pair_keyword_ids=sample_embeddings["pair_keyword_ids"],
            fallback_strategy="zero",
        )

        assert builder.fallback_strategy == "zero", "Should use custom fallback strategy"

    def test_init_invalid_fallback(self, sample_embeddings):
        """Test initialization fails with invalid fallback strategy."""
        with pytest.raises(ValueError, match="Invalid fallback_strategy"):
            SteeringBuilder(
                question_embeddings=sample_embeddings["question_embeddings"],
                keyword_embeddings=sample_embeddings["keyword_embeddings"],
                centroid_embeddings=sample_embeddings["centroid_embeddings"],
                pair_indices=sample_embeddings["pair_indices"],
                pair_cluster_ids=sample_embeddings["pair_cluster_ids"],
                pair_keyword_ids=sample_embeddings["pair_keyword_ids"],
                fallback_strategy="invalid",
            )

    def test_init_wrong_tensor_type(self, sample_embeddings):
        """Test initialization fails with non-tensor inputs."""
        with pytest.raises(TypeError, match="must be torch.Tensor"):
            SteeringBuilder(
                question_embeddings=sample_embeddings["question_embeddings"].numpy(),
                keyword_embeddings=sample_embeddings["keyword_embeddings"],
                centroid_embeddings=sample_embeddings["centroid_embeddings"],
                pair_indices=sample_embeddings["pair_indices"],
                pair_cluster_ids=sample_embeddings["pair_cluster_ids"],
                pair_keyword_ids=sample_embeddings["pair_keyword_ids"],
            )

    def test_init_dimension_mismatch(self, sample_embeddings):
        """Test initialization fails when embedding dimensions don't match."""
        # Create keyword embeddings with different dimension
        wrong_dim_keywords = torch.randn(5, 32)  # dim=32 instead of 64

        with pytest.raises(ValueError, match="Embedding dimensions must match"):
            SteeringBuilder(
                question_embeddings=sample_embeddings["question_embeddings"],
                keyword_embeddings=wrong_dim_keywords,
                centroid_embeddings=sample_embeddings["centroid_embeddings"],
                pair_indices=sample_embeddings["pair_indices"],
                pair_cluster_ids=sample_embeddings["pair_cluster_ids"],
                pair_keyword_ids=sample_embeddings["pair_keyword_ids"],
            )

    def test_init_pair_count_mismatch(self, sample_embeddings):
        """Test initialization fails when pair counts don't match."""
        # Create cluster IDs with wrong length
        wrong_cluster_ids = torch.randint(0, 3, (10,))  # Only 10 instead of 20

        with pytest.raises(ValueError, match="pair_cluster_ids length.*must equal"):
            SteeringBuilder(
                question_embeddings=sample_embeddings["question_embeddings"],
                keyword_embeddings=sample_embeddings["keyword_embeddings"],
                centroid_embeddings=sample_embeddings["centroid_embeddings"],
                pair_indices=sample_embeddings["pair_indices"],
                pair_cluster_ids=wrong_cluster_ids,
                pair_keyword_ids=sample_embeddings["pair_keyword_ids"],
            )

    def test_init_invalid_index_bounds(self, sample_embeddings):
        """Test initialization fails with out-of-bounds indices."""
        # Create pair indices with question index out of bounds
        invalid_indices = sample_embeddings["pair_indices"].clone()
        invalid_indices[0, 0] = 999  # Out of bounds

        with pytest.raises(ValueError, match="contains question index"):
            SteeringBuilder(
                question_embeddings=sample_embeddings["question_embeddings"],
                keyword_embeddings=sample_embeddings["keyword_embeddings"],
                centroid_embeddings=sample_embeddings["centroid_embeddings"],
                pair_indices=invalid_indices,
                pair_cluster_ids=sample_embeddings["pair_cluster_ids"],
                pair_keyword_ids=sample_embeddings["pair_keyword_ids"],
            )


class TestCentroidSteering:
    """Test centroid steering generation."""

    def test_centroid_steering_shape(self, sample_embeddings):
        """Test centroid steering has correct shape."""
        builder = SteeringBuilder(
            question_embeddings=sample_embeddings["question_embeddings"],
            keyword_embeddings=sample_embeddings["keyword_embeddings"],
            centroid_embeddings=sample_embeddings["centroid_embeddings"],
            pair_indices=sample_embeddings["pair_indices"],
            pair_cluster_ids=sample_embeddings["pair_cluster_ids"],
            pair_keyword_ids=sample_embeddings["pair_keyword_ids"],
            show_progress=False,
        )

        steering, distances = builder.build_centroid_steering()

        assert steering.shape == (
            sample_embeddings["n_pairs"],
            sample_embeddings["dim"],
        ), "Steering vectors should have shape [n_pairs, dim]"
        assert distances.shape == (
            sample_embeddings["n_pairs"],
        ), "Distances should have shape [n_pairs]"

    def test_centroid_steering_normalized(self, sample_embeddings):
        """Test centroid steering vectors are normalized."""
        builder = SteeringBuilder(
            question_embeddings=sample_embeddings["question_embeddings"],
            keyword_embeddings=sample_embeddings["keyword_embeddings"],
            centroid_embeddings=sample_embeddings["centroid_embeddings"],
            pair_indices=sample_embeddings["pair_indices"],
            pair_cluster_ids=sample_embeddings["pair_cluster_ids"],
            pair_keyword_ids=sample_embeddings["pair_keyword_ids"],
            show_progress=False,
        )

        steering, _ = builder.build_centroid_steering()

        # Check normalization (unit length)
        norms = torch.norm(steering, dim=1)
        non_zero_norms = norms[norms > 1e-8]

        assert torch.allclose(
            non_zero_norms, torch.ones_like(non_zero_norms), atol=1e-5
        ), "Non-zero steering vectors should be unit normalized"

    def test_centroid_distances_range(self, sample_embeddings):
        """Test centroid distances are in valid range [0, 2]."""
        builder = SteeringBuilder(
            question_embeddings=sample_embeddings["question_embeddings"],
            keyword_embeddings=sample_embeddings["keyword_embeddings"],
            centroid_embeddings=sample_embeddings["centroid_embeddings"],
            pair_indices=sample_embeddings["pair_indices"],
            pair_cluster_ids=sample_embeddings["pair_cluster_ids"],
            pair_keyword_ids=sample_embeddings["pair_keyword_ids"],
            show_progress=False,
        )

        _, distances = builder.build_centroid_steering()

        assert (distances >= -1e-6).all(), "Cosine distances should be >= 0"
        assert (distances <= 2.0 + 1e-6).all(), "Cosine distances should be <= 2"

    def test_centroid_steering_no_nan_inf(self, sample_embeddings):
        """Test centroid steering contains no NaN or Inf values."""
        builder = SteeringBuilder(
            question_embeddings=sample_embeddings["question_embeddings"],
            keyword_embeddings=sample_embeddings["keyword_embeddings"],
            centroid_embeddings=sample_embeddings["centroid_embeddings"],
            pair_indices=sample_embeddings["pair_indices"],
            pair_cluster_ids=sample_embeddings["pair_cluster_ids"],
            pair_keyword_ids=sample_embeddings["pair_keyword_ids"],
            show_progress=False,
        )

        steering, distances = builder.build_centroid_steering()

        assert not torch.isnan(steering).any(), "Steering vectors should not contain NaN"
        assert not torch.isinf(steering).any(), "Steering vectors should not contain Inf"
        assert not torch.isnan(distances).any(), "Distances should not contain NaN"
        assert not torch.isinf(distances).any(), "Distances should not contain Inf"


class TestKeywordWeightedSteering:
    """Test keyword-weighted steering generation."""

    def test_keyword_steering_shape(self, sample_embeddings):
        """Test keyword steering has correct shape."""
        builder = SteeringBuilder(
            question_embeddings=sample_embeddings["question_embeddings"],
            keyword_embeddings=sample_embeddings["keyword_embeddings"],
            centroid_embeddings=sample_embeddings["centroid_embeddings"],
            pair_indices=sample_embeddings["pair_indices"],
            pair_cluster_ids=sample_embeddings["pair_cluster_ids"],
            pair_keyword_ids=sample_embeddings["pair_keyword_ids"],
            show_progress=False,
        )

        steering = builder.build_keyword_weighted_steering()

        assert steering.shape == (
            sample_embeddings["n_pairs"],
            sample_embeddings["dim"],
        ), "Keyword steering should have shape [n_pairs, dim]"

    def test_keyword_steering_normalized(self, sample_embeddings):
        """Test keyword steering vectors are normalized."""
        builder = SteeringBuilder(
            question_embeddings=sample_embeddings["question_embeddings"],
            keyword_embeddings=sample_embeddings["keyword_embeddings"],
            centroid_embeddings=sample_embeddings["centroid_embeddings"],
            pair_indices=sample_embeddings["pair_indices"],
            pair_cluster_ids=sample_embeddings["pair_cluster_ids"],
            pair_keyword_ids=sample_embeddings["pair_keyword_ids"],
            show_progress=False,
        )

        steering = builder.build_keyword_weighted_steering()

        # Check normalization
        norms = torch.norm(steering, dim=1)
        non_zero_norms = norms[norms > 1e-8]

        assert torch.allclose(
            non_zero_norms, torch.ones_like(non_zero_norms), atol=1e-5
        ), "Non-zero keyword steering vectors should be unit normalized"

    def test_keyword_steering_fallback_centroid(self, sample_embeddings):
        """Test keyword steering uses centroid fallback for pairs with no keywords."""
        builder = SteeringBuilder(
            question_embeddings=sample_embeddings["question_embeddings"],
            keyword_embeddings=sample_embeddings["keyword_embeddings"],
            centroid_embeddings=sample_embeddings["centroid_embeddings"],
            pair_indices=sample_embeddings["pair_indices"],
            pair_cluster_ids=sample_embeddings["pair_cluster_ids"],
            pair_keyword_ids=sample_embeddings["pair_keyword_ids"],
            fallback_strategy="centroid",
            show_progress=False,
        )

        steering = builder.build_keyword_weighted_steering()

        # Check that pairs with no keywords still have steering vectors
        for i, keyword_ids in enumerate(sample_embeddings["pair_keyword_ids"]):
            if len(keyword_ids) == 0:
                assert (
                    steering[i].norm() > 1e-8
                ), f"Pair {i} with no keywords should have non-zero fallback steering"

    def test_keyword_steering_fallback_zero(self, sample_embeddings):
        """Test keyword steering uses zero fallback."""
        builder = SteeringBuilder(
            question_embeddings=sample_embeddings["question_embeddings"],
            keyword_embeddings=sample_embeddings["keyword_embeddings"],
            centroid_embeddings=sample_embeddings["centroid_embeddings"],
            pair_indices=sample_embeddings["pair_indices"],
            pair_cluster_ids=sample_embeddings["pair_cluster_ids"],
            pair_keyword_ids=sample_embeddings["pair_keyword_ids"],
            fallback_strategy="zero",
            show_progress=False,
        )

        steering = builder.build_keyword_weighted_steering()

        # Check that pairs with no keywords have zero vectors
        for i, keyword_ids in enumerate(sample_embeddings["pair_keyword_ids"]):
            if len(keyword_ids) == 0:
                assert (
                    steering[i].norm() < 1e-8
                ), f"Pair {i} with no keywords should have zero fallback steering"

    def test_keyword_steering_fallback_random(self, sample_embeddings):
        """Test keyword steering uses random fallback."""
        builder = SteeringBuilder(
            question_embeddings=sample_embeddings["question_embeddings"],
            keyword_embeddings=sample_embeddings["keyword_embeddings"],
            centroid_embeddings=sample_embeddings["centroid_embeddings"],
            pair_indices=sample_embeddings["pair_indices"],
            pair_cluster_ids=sample_embeddings["pair_cluster_ids"],
            pair_keyword_ids=sample_embeddings["pair_keyword_ids"],
            fallback_strategy="random",
            show_progress=False,
        )

        steering = builder.build_keyword_weighted_steering()

        # Check that pairs with no keywords have unit random vectors
        for i, keyword_ids in enumerate(sample_embeddings["pair_keyword_ids"]):
            if len(keyword_ids) == 0:
                norm = steering[i].norm()
                assert torch.isclose(
                    norm, torch.tensor(1.0), atol=1e-5
                ), f"Pair {i} with no keywords should have unit random fallback steering"


class TestResidualSteering:
    """Test residual steering generation."""

    def test_residual_steering_shape(self, sample_embeddings):
        """Test residual steering has correct shape."""
        builder = SteeringBuilder(
            question_embeddings=sample_embeddings["question_embeddings"],
            keyword_embeddings=sample_embeddings["keyword_embeddings"],
            centroid_embeddings=sample_embeddings["centroid_embeddings"],
            pair_indices=sample_embeddings["pair_indices"],
            pair_cluster_ids=sample_embeddings["pair_cluster_ids"],
            pair_keyword_ids=sample_embeddings["pair_keyword_ids"],
            show_progress=False,
        )

        steering = builder.build_residual_steering()

        assert steering.shape == (
            sample_embeddings["n_pairs"],
            sample_embeddings["dim"],
        ), "Residual steering should have shape [n_pairs, dim]"

    def test_residual_steering_unnormalized(self, sample_embeddings):
        """Test residual steering is not normalized by default."""
        builder = SteeringBuilder(
            question_embeddings=sample_embeddings["question_embeddings"],
            keyword_embeddings=sample_embeddings["keyword_embeddings"],
            centroid_embeddings=sample_embeddings["centroid_embeddings"],
            pair_indices=sample_embeddings["pair_indices"],
            pair_cluster_ids=sample_embeddings["pair_cluster_ids"],
            pair_keyword_ids=sample_embeddings["pair_keyword_ids"],
            normalize_residual=False,
            show_progress=False,
        )

        steering = builder.build_residual_steering()

        # Residual should not necessarily be unit length
        norms = torch.norm(steering, dim=1)

        # At least some vectors should not be unit length
        non_unit = (norms - 1.0).abs() > 1e-4
        assert non_unit.sum() > 0, "Unnormalized residual steering should have non-unit vectors"

    def test_residual_steering_normalized(self, sample_embeddings):
        """Test residual steering can be normalized."""
        builder = SteeringBuilder(
            question_embeddings=sample_embeddings["question_embeddings"],
            keyword_embeddings=sample_embeddings["keyword_embeddings"],
            centroid_embeddings=sample_embeddings["centroid_embeddings"],
            pair_indices=sample_embeddings["pair_indices"],
            pair_cluster_ids=sample_embeddings["pair_cluster_ids"],
            pair_keyword_ids=sample_embeddings["pair_keyword_ids"],
            normalize_residual=True,
            show_progress=False,
        )

        steering = builder.build_residual_steering()

        # Check normalization
        norms = torch.norm(steering, dim=1)
        non_zero_norms = norms[norms > 1e-8]

        assert torch.allclose(
            non_zero_norms, torch.ones_like(non_zero_norms), atol=1e-5
        ), "Normalized residual steering should be unit length"


class TestBuildAllSteering:
    """Test build_all_steering method."""

    def test_build_all_steering_keys(self, sample_embeddings):
        """Test build_all_steering returns all expected keys."""
        builder = SteeringBuilder(
            question_embeddings=sample_embeddings["question_embeddings"],
            keyword_embeddings=sample_embeddings["keyword_embeddings"],
            centroid_embeddings=sample_embeddings["centroid_embeddings"],
            pair_indices=sample_embeddings["pair_indices"],
            pair_cluster_ids=sample_embeddings["pair_cluster_ids"],
            pair_keyword_ids=sample_embeddings["pair_keyword_ids"],
            show_progress=False,
        )

        results = builder.build_all_steering()

        expected_keys = {"centroid", "keyword_weighted", "residual", "distances"}
        assert (
            set(results.keys()) == expected_keys
        ), "Should return all steering variants and distances"

    def test_build_all_steering_shapes(self, sample_embeddings):
        """Test build_all_steering returns correct shapes."""
        builder = SteeringBuilder(
            question_embeddings=sample_embeddings["question_embeddings"],
            keyword_embeddings=sample_embeddings["keyword_embeddings"],
            centroid_embeddings=sample_embeddings["centroid_embeddings"],
            pair_indices=sample_embeddings["pair_indices"],
            pair_cluster_ids=sample_embeddings["pair_cluster_ids"],
            pair_keyword_ids=sample_embeddings["pair_keyword_ids"],
            show_progress=False,
        )

        results = builder.build_all_steering()

        n_pairs = sample_embeddings["n_pairs"]
        dim = sample_embeddings["dim"]

        assert results["centroid"].shape == (
            n_pairs,
            dim,
        ), "Centroid steering should have correct shape"
        assert results["keyword_weighted"].shape == (
            n_pairs,
            dim,
        ), "Keyword steering should have correct shape"
        assert results["residual"].shape == (
            n_pairs,
            dim,
        ), "Residual steering should have correct shape"
        assert results["distances"].shape == (n_pairs,), "Distances should have correct shape"


class TestSteeringSave:
    """Test saving steering tensors."""

    def test_save_creates_files(self, sample_embeddings):
        """Test save creates all expected files."""
        builder = SteeringBuilder(
            question_embeddings=sample_embeddings["question_embeddings"],
            keyword_embeddings=sample_embeddings["keyword_embeddings"],
            centroid_embeddings=sample_embeddings["centroid_embeddings"],
            pair_indices=sample_embeddings["pair_indices"],
            pair_cluster_ids=sample_embeddings["pair_cluster_ids"],
            pair_keyword_ids=sample_embeddings["pair_keyword_ids"],
            show_progress=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "steering"
            builder.save(output_dir)

            # Check all files exist
            assert (
                output_dir / "steering_centroid.pt"
            ).exists(), "Should create centroid steering file"
            assert (
                output_dir / "steering_keyword_weighted.pt"
            ).exists(), "Should create keyword steering file"
            assert (
                output_dir / "steering_residual.pt"
            ).exists(), "Should create residual steering file"
            assert (output_dir / "centroid_distances.pt").exists(), "Should create distances file"

    def test_save_with_precomputed_results(self, sample_embeddings):
        """Test save with pre-computed steering results."""
        builder = SteeringBuilder(
            question_embeddings=sample_embeddings["question_embeddings"],
            keyword_embeddings=sample_embeddings["keyword_embeddings"],
            centroid_embeddings=sample_embeddings["centroid_embeddings"],
            pair_indices=sample_embeddings["pair_indices"],
            pair_cluster_ids=sample_embeddings["pair_cluster_ids"],
            pair_keyword_ids=sample_embeddings["pair_keyword_ids"],
            show_progress=False,
        )

        results = builder.build_all_steering()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "steering"
            builder.save(output_dir, steering_results=results)

            # Load and verify
            loaded_centroid = torch.load(output_dir / "steering_centroid.pt")
            assert torch.allclose(
                loaded_centroid, results["centroid"]
            ), "Saved centroid steering should match computed"


class TestBuildSteeringFunction:
    """Test build_steering convenience function."""

    def test_build_steering_function(self, sample_embeddings):
        """Test build_steering convenience function works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "steering"

            results = build_steering(
                question_embeddings=sample_embeddings["question_embeddings"],
                keyword_embeddings=sample_embeddings["keyword_embeddings"],
                centroid_embeddings=sample_embeddings["centroid_embeddings"],
                pair_indices=sample_embeddings["pair_indices"],
                pair_cluster_ids=sample_embeddings["pair_cluster_ids"],
                pair_keyword_ids=sample_embeddings["pair_keyword_ids"],
                output_dir=output_dir,
                show_progress=False,
            )

            # Check results returned
            assert "centroid" in results, "Should return centroid steering"
            assert "keyword_weighted" in results, "Should return keyword steering"
            assert "residual" in results, "Should return residual steering"
            assert "distances" in results, "Should return distances"

            # Check files created
            assert (output_dir / "steering_centroid.pt").exists(), "Should create steering files"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_all_pairs_same_cluster(self, sample_embeddings):
        """Test steering when all pairs belong to same cluster."""
        # Assign all pairs to cluster 0
        all_same_cluster = torch.zeros(sample_embeddings["n_pairs"], dtype=torch.long)

        builder = SteeringBuilder(
            question_embeddings=sample_embeddings["question_embeddings"],
            keyword_embeddings=sample_embeddings["keyword_embeddings"],
            centroid_embeddings=sample_embeddings["centroid_embeddings"],
            pair_indices=sample_embeddings["pair_indices"],
            pair_cluster_ids=all_same_cluster,
            pair_keyword_ids=sample_embeddings["pair_keyword_ids"],
            show_progress=False,
        )

        results = builder.build_all_steering()

        assert (
            results["centroid"].shape[0] == sample_embeddings["n_pairs"]
        ), "Should handle all pairs in same cluster"

    def test_all_pairs_no_keywords(self, sample_embeddings):
        """Test steering when no pairs have keywords."""
        # Remove all keywords
        no_keywords = [[] for _ in range(sample_embeddings["n_pairs"])]

        builder = SteeringBuilder(
            question_embeddings=sample_embeddings["question_embeddings"],
            keyword_embeddings=sample_embeddings["keyword_embeddings"],
            centroid_embeddings=sample_embeddings["centroid_embeddings"],
            pair_indices=sample_embeddings["pair_indices"],
            pair_cluster_ids=sample_embeddings["pair_cluster_ids"],
            pair_keyword_ids=no_keywords,
            fallback_strategy="centroid",
            show_progress=False,
        )

        steering = builder.build_keyword_weighted_steering()

        # All steering should use fallback
        assert (
            steering.shape[0] == sample_embeddings["n_pairs"]
        ), "Should handle all pairs with no keywords using fallback"

    def test_single_pair(self):
        """Test steering with single pair."""
        question_embs = torch.randn(1, 64)
        keyword_embs = torch.randn(2, 64)
        centroid_embs = torch.randn(1, 64)
        pair_indices = torch.tensor([[0, 0]])
        pair_cluster_ids = torch.tensor([0])
        pair_keyword_ids = [[0, 1]]

        builder = SteeringBuilder(
            question_embeddings=question_embs,
            keyword_embeddings=keyword_embs,
            centroid_embeddings=centroid_embs,
            pair_indices=pair_indices,
            pair_cluster_ids=pair_cluster_ids,
            pair_keyword_ids=pair_keyword_ids,
            show_progress=False,
        )

        results = builder.build_all_steering()

        assert results["centroid"].shape == (1, 64), "Should handle single pair"
        assert results["distances"].shape == (1,), "Should compute distance for single pair"
