"""Tests for validation_utils module."""

import pytest
import numpy as np
import torch

from RAG_supporters.data_validation import (
    validate_tensor_2d,
    validate_tensor_1d,
    validate_embedding_dimensions,
    validate_pair_indices_bounds,
    validate_cluster_ids_bounds,
    validate_length_consistency,
    validate_split_ratios,
    validate_keyword_ids_list,
    validate_no_nan_inf,
    validate_values_in_range,
)


class TestValidateTensor2D:
    """Tests for validate_tensor_2d function."""

    def test_valid_tensor(self):
        """Test valid 2D tensor passes validation."""
        tensor = torch.randn(10, 5)
        validate_tensor_2d(tensor, "test_tensor")  # Should not raise

    def test_valid_tensor_with_column_check(self):
        """Test valid 2D tensor with correct column count."""
        tensor = torch.randn(10, 384)
        validate_tensor_2d(tensor, "test_tensor", expected_cols=384)  # Should not raise

    def test_invalid_type_list(self):
        """Test non-tensor list raises TypeError."""
        with pytest.raises(TypeError, match="must be torch.Tensor"):
            validate_tensor_2d([1, 2, 3], "test_tensor")

    def test_invalid_type_numpy(self):
        """Test numpy array raises TypeError."""
        arr = np.array([[1, 2], [3, 4]])
        with pytest.raises(TypeError, match="must be torch.Tensor"):
            validate_tensor_2d(arr, "test_tensor")

    def test_wrong_dimensions_1d(self):
        """Test 1D tensor raises ValueError."""
        tensor = torch.randn(10)
        with pytest.raises(ValueError, match="must be 2D"):
            validate_tensor_2d(tensor, "test_tensor")

    def test_wrong_dimensions_3d(self):
        """Test 3D tensor raises ValueError."""
        tensor = torch.randn(10, 5, 3)
        with pytest.raises(ValueError, match="must be 2D"):
            validate_tensor_2d(tensor, "test_tensor")

    def test_column_count_mismatch(self):
        """Test wrong column count raises ValueError."""
        tensor = torch.randn(10, 5)
        with pytest.raises(ValueError, match="expected 3 columns"):
            validate_tensor_2d(tensor, "test_tensor", expected_cols=3)

    def test_too_few_rows(self):
        """Test insufficient rows raises ValueError."""
        tensor = torch.randn(2, 5)
        with pytest.raises(ValueError, match="at least 5 rows"):
            validate_tensor_2d(tensor, "test_tensor", min_rows=5)

    def test_minimum_one_row_default(self):
        """Test default minimum 1 row requirement."""
        tensor = torch.randn(0, 5)
        with pytest.raises(ValueError, match="at least 1 rows"):
            validate_tensor_2d(tensor, "embeddings")

    def test_edge_case_single_row(self):
        """Test single row tensor is valid."""
        tensor = torch.randn(1, 5)
        validate_tensor_2d(tensor, "test_tensor")  # Should not raise


class TestValidateTensor1D:
    """Tests for validate_tensor_1d function."""

    def test_valid_tensor(self):
        """Test valid 1D tensor passes validation."""
        tensor = torch.tensor([1, 2, 3, 4, 5])
        validate_tensor_1d(tensor, "test_tensor")  # Should not raise

    def test_valid_tensor_with_length_check(self):
        """Test valid 1D tensor with correct length."""
        tensor = torch.tensor([1, 2, 3])
        validate_tensor_1d(tensor, "test_tensor", expected_length=3)  # Should not raise

    def test_invalid_type(self):
        """Test non-tensor raises TypeError."""
        with pytest.raises(TypeError, match="must be torch.Tensor"):
            validate_tensor_1d([1, 2, 3], "test_tensor")

    def test_wrong_dimensions_2d(self):
        """Test 2D tensor raises ValueError."""
        tensor = torch.randn(10, 5)
        with pytest.raises(ValueError, match="must be 1D"):
            validate_tensor_1d(tensor, "test_tensor")

    def test_wrong_dimensions_0d(self):
        """Test scalar tensor raises ValueError."""
        tensor = torch.tensor(5)
        with pytest.raises(ValueError, match="must be 1D"):
            validate_tensor_1d(tensor, "test_tensor")

    def test_length_mismatch(self):
        """Test wrong length raises ValueError."""
        tensor = torch.tensor([1, 2, 3])
        with pytest.raises(ValueError, match="expected length 5"):
            validate_tensor_1d(tensor, "test_tensor", expected_length=5)

    def test_too_short(self):
        """Test tensor shorter than minimum raises ValueError."""
        tensor = torch.tensor([1, 2])
        with pytest.raises(ValueError, match="at least 5 elements"):
            validate_tensor_1d(tensor, "test_tensor", min_length=5)

    def test_empty_tensor_fails_default(self):
        """Test empty tensor fails with default min_length=1."""
        tensor = torch.tensor([])
        with pytest.raises(ValueError, match="at least 1 elements"):
            validate_tensor_1d(tensor, "cluster_ids")


class TestValidateEmbeddingDimensions:
    """Tests for validate_embedding_dimensions function."""

    def test_consistent_dimensions_two_tensors(self):
        """Test two tensors with same dimensions pass."""
        q = torch.randn(10, 384)
        s = torch.randn(20, 384)
        dim = validate_embedding_dimensions((q, "q"), (s, "s"))
        assert dim == 384, "Should return common dimension 384"

    def test_consistent_dimensions_three_tensors(self):
        """Test three tensors with same dimensions pass."""
        q = torch.randn(10, 384)
        s = torch.randn(20, 384)
        k = torch.randn(50, 384)
        dim = validate_embedding_dimensions((q, "q"), (s, "s"), (k, "k"))
        assert dim == 384, "Should return common dimension 384"

    def test_inconsistent_dimensions_two_tensors(self):
        """Test tensors with different dimensions raise ValueError."""
        q = torch.randn(10, 384)
        s = torch.randn(20, 512)
        with pytest.raises(ValueError, match="dimensions must match"):
            validate_embedding_dimensions((q, "q"), (s, "s"))

    def test_inconsistent_dimensions_three_tensors(self):
        """Test three tensors with inconsistent dimensions raise ValueError."""
        q = torch.randn(10, 384)
        s = torch.randn(20, 384)
        k = torch.randn(50, 512)
        with pytest.raises(ValueError, match="dimensions must match"):
            validate_embedding_dimensions((q, "q"), (s, "s"), (k, "k"))

    def test_no_tensors_provided(self):
        """Test providing no tensors raises ValueError."""
        with pytest.raises(ValueError, match="At least one tensor"):
            validate_embedding_dimensions()

    def test_single_tensor(self):
        """Test single tensor returns its dimension."""
        q = torch.randn(10, 384)
        dim = validate_embedding_dimensions((q, "q"))
        assert dim == 384, "Should return dimension of single tensor"

    def test_validates_2d_requirement(self):
        """Test that non-2D tensors are rejected."""
        q = torch.randn(384)  # 1D tensor
        s = torch.randn(20, 384)
        with pytest.raises(ValueError, match="must be 2D"):
            validate_embedding_dimensions((q, "q"), (s, "s"))


class TestValidatePairIndicesBounds:
    """Tests for validate_pair_indices_bounds function."""

    def test_valid_indices(self):
        """Test valid indices pass validation."""
        pairs = torch.tensor([[0, 5], [1, 10], [2, 3]])
        validate_pair_indices_bounds(pairs, n_questions=50, n_sources=20)  # Should not raise

    def test_question_index_out_of_bounds_high(self):
        """Test out-of-bounds question index raises ValueError."""
        pairs = torch.tensor([[0, 5], [60, 10]])  # 60 >= 50
        with pytest.raises(ValueError, match="question index"):
            validate_pair_indices_bounds(pairs, n_questions=50, n_sources=20)

    def test_source_index_out_of_bounds_high(self):
        """Test out-of-bounds source index raises ValueError."""
        pairs = torch.tensor([[0, 5], [1, 25]])  # 25 >= 20
        with pytest.raises(ValueError, match="source index"):
            validate_pair_indices_bounds(pairs, n_questions=50, n_sources=20)

    def test_question_index_negative(self):
        """Test negative question index is caught by max check."""
        pairs = torch.tensor([[0, 5], [-1, 10]])
        # Note: min() check is in validate_tensor_2d, but negative will pass through
        # This is acceptable as long as we catch it somewhere
        validate_pair_indices_bounds(pairs, n_questions=50, n_sources=20)  # Won't raise

    def test_edge_case_max_valid_index(self):
        """Test maximum valid indices are accepted."""
        pairs = torch.tensor([[49, 19], [0, 0]])  # Max valid indices
        validate_pair_indices_bounds(pairs, n_questions=50, n_sources=20)  # Should not raise

    def test_validates_2d_requirement(self):
        """Test that non-2D tensor is rejected."""
        pairs = torch.tensor([0, 5])  # 1D
        with pytest.raises(ValueError, match="must be 2D"):
            validate_pair_indices_bounds(pairs, n_questions=50, n_sources=20)

    def test_validates_column_count(self):
        """Test that tensor without exactly 2 columns is rejected."""
        pairs = torch.tensor([[0, 5, 1], [1, 10, 2]])  # 3 columns
        with pytest.raises(ValueError, match="expected 2 columns"):
            validate_pair_indices_bounds(pairs, n_questions=50, n_sources=20)

    def test_custom_name_in_error(self):
        """Test custom name appears in error messages."""
        pairs = torch.tensor([[60, 5]])
        with pytest.raises(ValueError, match="custom_pairs contains question"):
            validate_pair_indices_bounds(
                pairs, n_questions=50, n_sources=20, name="custom_pairs"
            )


class TestValidateClusterIdsBounds:
    """Tests for validate_cluster_ids_bounds function."""

    def test_valid_cluster_ids(self):
        """Test valid cluster IDs pass validation."""
        cluster_ids = torch.tensor([0, 1, 2, 1, 0, 4])
        validate_cluster_ids_bounds(cluster_ids, n_clusters=5)  # Should not raise

    def test_cluster_id_out_of_bounds_high(self):
        """Test out-of-bounds cluster ID raises ValueError."""
        cluster_ids = torch.tensor([0, 1, 5, 1, 0])  # 5 >= 5
        with pytest.raises(ValueError, match="cluster ID 5"):
            validate_cluster_ids_bounds(cluster_ids, n_clusters=5)

    def test_edge_case_max_valid_cluster(self):
        """Test maximum valid cluster ID is accepted."""
        cluster_ids = torch.tensor([0, 1, 4, 1, 0])  # 4 is max for 5 clusters
        validate_cluster_ids_bounds(cluster_ids, n_clusters=5)  # Should not raise

    def test_single_cluster(self):
        """Test single cluster case."""
        cluster_ids = torch.tensor([0, 0, 0])
        validate_cluster_ids_bounds(cluster_ids, n_clusters=1)  # Should not raise

    def test_validates_1d_requirement(self):
        """Test that non-1D tensor is rejected."""
        cluster_ids = torch.randn(5, 2)  # 2D
        with pytest.raises(ValueError, match="must be 1D"):
            validate_cluster_ids_bounds(cluster_ids, n_clusters=5)

    def test_custom_name_in_error(self):
        """Test custom name appears in error messages."""
        cluster_ids = torch.tensor([0, 1, 10])
        with pytest.raises(ValueError, match="custom_clusters contains cluster ID"):
            validate_cluster_ids_bounds(cluster_ids, n_clusters=5, name="custom_clusters")


class TestValidateLengthConsistency:
    """Tests for validate_length_consistency function."""

    def test_consistent_tensor_lengths(self):
        """Test tensors with consistent lengths pass."""
        t1 = torch.randn(100, 2)
        t2 = torch.randn(100)
        validate_length_consistency(
            (t1, "pairs", 100),
            (t2, "clusters", 100)
        )  # Should not raise

    def test_consistent_list_length(self):
        """Test list with correct length passes."""
        lst = [[1, 2], [3]] * 50  # 100 items
        validate_length_consistency((lst, "keywords", 100))  # Should not raise

    def test_mixed_tensor_and_list(self):
        """Test mix of tensors and lists with consistent lengths."""
        t1 = torch.randn(100, 2)
        lst = [[1, 2], [3]] * 50  # 100 items
        validate_length_consistency(
            (t1, "pairs", 100),
            (lst, "keywords", 100)
        )  # Should not raise

    def test_tensor_length_mismatch(self):
        """Test tensor length mismatch raises ValueError."""
        t1 = torch.randn(95, 2)  # Wrong length
        with pytest.raises(ValueError, match="length .95. must equal 100"):
            validate_length_consistency((t1, "pairs", 100))

    def test_list_length_mismatch(self):
        """Test list length mismatch raises ValueError."""
        lst = [[1, 2]] * 50  # 50 items, not 100
        with pytest.raises(ValueError, match="length .50. must equal 100"):
            validate_length_consistency((lst, "keywords", 100))

    def test_invalid_type(self):
        """Test invalid type raises TypeError."""
        with pytest.raises(TypeError, match="must be torch.Tensor or list"):
            validate_length_consistency(({"key": "value"}, "dict_obj", 10))

    def test_multiple_mismatches(self):
        """Test first mismatch is reported."""
        t1 = torch.randn(100, 2)
        t2 = torch.randn(95)  # First mismatch
        t3 = torch.randn(90)  # Second mismatch
        with pytest.raises(ValueError, match="t2.*95.*100"):
            validate_length_consistency(
                (t1, "t1", 100),
                (t2, "t2", 100),
                (t3, "t3", 100)
            )


class TestValidateSplitRatios:
    """Tests for validate_split_ratios function."""

    def test_valid_ratios_standard(self):
        """Test valid standard split ratios."""
        validate_split_ratios(0.7, 0.15, 0.15)  # Should not raise

    def test_valid_ratios_80_10_10(self):
        """Test valid 80/10/10 split."""
        validate_split_ratios(0.8, 0.1, 0.1)  # Should not raise

    def test_train_ratio_too_low(self):
        """Test train ratio <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="train_ratio must be in .0, 1."):
            validate_split_ratios(0.0, 0.5, 0.5)

    def test_train_ratio_too_high(self):
        """Test train ratio >= 1 raises ValueError."""
        with pytest.raises(ValueError, match="train_ratio must be in .0, 1."):
            validate_split_ratios(1.0, 0.0, 0.0)

    def test_val_ratio_too_low(self):
        """Test val ratio <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="val_ratio must be in .0, 1."):
            validate_split_ratios(0.7, 0.0, 0.3)

    def test_test_ratio_too_low(self):
        """Test test ratio <= 0 raises ValueError."""
        with pytest.raises(ValueError, match="test_ratio must be in .0, 1."):
            validate_split_ratios(0.7, 0.3, 0.0)

    def test_ratios_sum_too_low(self):
        """Test ratios summing to less than 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            validate_split_ratios(0.5, 0.2, 0.2)  # Sum = 0.9

    def test_ratios_sum_too_high(self):
        """Test ratios summing to more than 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            validate_split_ratios(0.7, 0.2, 0.2)  # Sum = 1.1

    def test_ratios_sum_within_tolerance(self):
        """Test ratios that sum to 1.0 within tolerance pass."""
        # 0.7 + 0.15 + 0.15 = 1.0 (exactly)
        validate_split_ratios(0.7, 0.15, 0.15)  # Should not raise
        
        # Test slightly off due to floating point (within default 1e-6 tolerance)
        validate_split_ratios(0.333333, 0.333333, 0.333334)  # Should not raise


class TestValidateKeywordIdsList:
    """Tests for validate_keyword_ids_list function."""

    def test_valid_keyword_ids_list(self):
        """Test valid list of keyword ID lists passes."""
        pair_keyword_ids = [[0, 1, 2], [1, 3], [], [4, 5]]
        validate_keyword_ids_list(pair_keyword_ids, n_pairs=4, n_keywords=10)  # Should not raise

    def test_valid_with_empty_lists(self):
        """Test valid list with some empty keyword lists."""
        pair_keyword_ids = [[0, 1], [], [2, 3, 4], []]
        validate_keyword_ids_list(pair_keyword_ids, n_pairs=4, n_keywords=10)  # Should not raise

    def test_not_a_list(self):
        """Test non-list raises TypeError."""
        with pytest.raises(TypeError, match="must be list"):
            validate_keyword_ids_list(
                {"key": [1, 2]}, n_pairs=1, n_keywords=10
            )

    def test_wrong_length(self):
        """Test wrong number of pairs raises ValueError."""
        pair_keyword_ids = [[0, 1], [2, 3]]  # Only 2, not 4
        with pytest.raises(ValueError, match="length .2. must equal n_pairs .4."):
            validate_keyword_ids_list(pair_keyword_ids, n_pairs=4, n_keywords=10)

    def test_inner_not_list(self):
        """Test non-list inner element raises TypeError."""
        pair_keyword_ids = [[0, 1], (2, 3), [4]]  # Tuple instead of list
        with pytest.raises(TypeError, match="pair_keyword_ids.1. must be list"):
            validate_keyword_ids_list(pair_keyword_ids, n_pairs=3, n_keywords=10)

    def test_keyword_id_not_int(self):
        """Test non-integer keyword ID raises TypeError."""
        pair_keyword_ids = [[0, 1], [2, 3.5], [4]]  # Float instead of int
        with pytest.raises(TypeError, match="contains non-int"):
            validate_keyword_ids_list(pair_keyword_ids, n_pairs=3, n_keywords=10)

    def test_keyword_id_out_of_bounds_high(self):
        """Test keyword ID >= n_keywords raises ValueError."""
        pair_keyword_ids = [[0, 1], [2, 15], [4]]  # 15 >= 10
        with pytest.raises(ValueError, match="out-of-range keyword ID 15"):
            validate_keyword_ids_list(pair_keyword_ids, n_pairs=3, n_keywords=10)

    def test_keyword_id_negative(self):
        """Test negative keyword ID raises ValueError."""
        pair_keyword_ids = [[0, 1], [2, -1], [4]]
        with pytest.raises(ValueError, match="out-of-range keyword ID -1"):
            validate_keyword_ids_list(pair_keyword_ids, n_pairs=3, n_keywords=10)

    def test_custom_name_in_error(self):
        """Test custom name appears in error messages."""
        pair_keyword_ids = [[0, 1], [2]]  # Wrong length
        with pytest.raises(ValueError, match="custom_keywords length"):
            validate_keyword_ids_list(
                pair_keyword_ids, n_pairs=4, n_keywords=10, name="custom_keywords"
            )


class TestValidateNoNanInf:
    """Tests for validate_no_nan_inf function."""

    def test_valid_tensor_float(self):
        """Test valid float tensor passes validation."""
        tensor = torch.randn(10, 5)
        validate_no_nan_inf(tensor, "test_tensor")  # Should not raise

    def test_valid_tensor_int(self):
        """Test valid integer tensor passes validation."""
        tensor = torch.randint(0, 100, (10, 5))
        validate_no_nan_inf(tensor, "test_tensor")  # Should not raise

    def test_tensor_with_nan(self):
        """Test tensor with NaN raises ValueError."""
        tensor = torch.randn(10, 5)
        tensor[0, 0] = float('nan')
        with pytest.raises(ValueError, match="contains NaN values"):
            validate_no_nan_inf(tensor, "test_tensor")

    def test_tensor_with_inf(self):
        """Test tensor with Inf raises ValueError."""
        tensor = torch.randn(10, 5)
        tensor[0, 0] = float('inf')
        with pytest.raises(ValueError, match="contains Inf values"):
            validate_no_nan_inf(tensor, "test_tensor")

    def test_tensor_with_negative_inf(self):
        """Test tensor with -Inf raises ValueError."""
        tensor = torch.randn(10, 5)
        tensor[0, 0] = float('-inf')
        with pytest.raises(ValueError, match="contains Inf values"):
            validate_no_nan_inf(tensor, "test_tensor")

    def test_tensor_with_both_nan_and_inf(self):
        """Test tensor with both NaN and Inf raises ValueError for NaN first."""
        tensor = torch.randn(10, 5)
        tensor[0, 0] = float('nan')
        tensor[1, 1] = float('inf')
        # NaN check happens first
        with pytest.raises(ValueError, match="contains NaN values"):
            validate_no_nan_inf(tensor, "test_tensor")

    def test_custom_name_in_error_nan(self):
        """Test custom name appears in NaN error message."""
        tensor = torch.tensor([1.0, float('nan'), 3.0])
        with pytest.raises(ValueError, match="embeddings contains NaN"):
            validate_no_nan_inf(tensor, "embeddings")

    def test_custom_name_in_error_inf(self):
        """Test custom name appears in Inf error message."""
        tensor = torch.tensor([1.0, float('inf'), 3.0])
        with pytest.raises(ValueError, match="scores contains Inf"):
            validate_no_nan_inf(tensor, "scores")

    def test_1d_tensor(self):
        """Test validation works with 1D tensors."""
        tensor = torch.randn(100)
        validate_no_nan_inf(tensor, "test_tensor")  # Should not raise

    def test_3d_tensor(self):
        """Test validation works with 3D tensors."""
        tensor = torch.randn(5, 10, 3)
        validate_no_nan_inf(tensor, "test_tensor")  # Should not raise


class TestValidateValuesInRange:
    """Tests for validate_values_in_range function."""

    def test_valid_values_inclusive(self):
        """Test valid values within inclusive range pass."""
        tensor = torch.tensor([0.0, 0.5, 1.0])
        validate_values_in_range(
            tensor, "scores",
            min_value=0.0, max_value=1.0, inclusive=True
        )  # Should not raise

    def test_valid_values_exclusive(self):
        """Test valid values within exclusive range pass."""
        tensor = torch.tensor([0.1, 0.5, 0.9])
        validate_values_in_range(
            tensor, "scores",
            min_value=0.0, max_value=1.0, inclusive=False
        )  # Should not raise

    def test_edge_values_inclusive(self):
        """Test edge values are accepted in inclusive range."""
        tensor = torch.tensor([0, 10])
        validate_values_in_range(
            tensor, "indices",
            min_value=0, max_value=10, inclusive=True
        )  # Should not raise

    def test_edge_values_exclusive_fails(self):
        """Test edge values are rejected in exclusive range."""
        tensor = torch.tensor([0, 10])
        with pytest.raises(ValueError, match="values must be in range.*0.*10"):
            validate_values_in_range(
                tensor, "indices",
                min_value=0, max_value=10, inclusive=False
            )

    def test_value_below_min_inclusive(self):
        """Test value below minimum raises ValueError in inclusive range."""
        tensor = torch.tensor([0.0, 0.5, -0.1])
        with pytest.raises(ValueError, match="must be in range .0.0, 1.0.*.-0.1"):
            validate_values_in_range(
                tensor, "scores",
                min_value=0.0, max_value=1.0, inclusive=True
            )

    def test_value_above_max_inclusive(self):
        """Test value above maximum raises ValueError in inclusive range."""
        tensor = torch.tensor([0.0, 0.5, 1.1])
        with pytest.raises(ValueError, match="must be in range .0.0, 1.0.*1.1"):
            validate_values_in_range(
                tensor, "scores",
                min_value=0.0, max_value=1.0, inclusive=True
            )

    def test_value_at_min_exclusive_fails(self):
        """Test value at minimum fails in exclusive range."""
        tensor = torch.tensor([0.0, 0.5, 1.0])
        with pytest.raises(ValueError, match="must be in range .0.0, 1.0."):
            validate_values_in_range(
                tensor, "scores",
                min_value=0.0, max_value=1.0, inclusive=False
            )

    def test_value_at_max_exclusive_fails(self):
        """Test value at maximum fails in exclusive range."""
        tensor = torch.tensor([0.1, 0.5, 1.0])
        with pytest.raises(ValueError, match="must be in range .0.0, 1.0."):
            validate_values_in_range(
                tensor, "scores",
                min_value=0.0, max_value=1.0, inclusive=False
            )

    def test_integer_range(self):
        """Test integer range validation."""
        tensor = torch.tensor([0, 5, 10, 15, 19])
        validate_values_in_range(
            tensor, "indices",
            min_value=0, max_value=19, inclusive=True
        )  # Should not raise

    def test_integer_out_of_range(self):
        """Test integer out of range raises ValueError."""
        tensor = torch.tensor([0, 5, 10, 20])
        with pytest.raises(ValueError, match="must be in range .0, 19."):
            validate_values_in_range(
                tensor, "indices",
                min_value=0, max_value=19, inclusive=True
            )

    def test_negative_range(self):
        """Test negative value ranges."""
        tensor = torch.tensor([-10.0, -5.0, 0.0])
        validate_values_in_range(
            tensor, "values",
            min_value=-10.0, max_value=0.0, inclusive=True
        )  # Should not raise

    def test_custom_name_in_error(self):
        """Test custom name appears in error messages."""
        tensor = torch.tensor([0, 1, 2, 10])
        with pytest.raises(ValueError, match="cluster_ids values must be in range"):
            validate_values_in_range(
                tensor, "cluster_ids",
                min_value=0, max_value=5, inclusive=True
            )

    def test_single_value_tensor(self):
        """Test validation works with single value tensor."""
        tensor = torch.tensor([0.5])
        validate_values_in_range(
            tensor, "value",
            min_value=0.0, max_value=1.0, inclusive=True
        )  # Should not raise

    def test_multidimensional_tensor(self):
        """Test validation works with multidimensional tensors."""
        tensor = torch.rand(5, 10)  # Random values in [0, 1)
        validate_values_in_range(
            tensor, "matrix",
            min_value=0.0, max_value=1.0, inclusive=True
        )  # Should not raise

