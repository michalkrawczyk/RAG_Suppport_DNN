"""Tests for tensor_utils module."""

import pytest
import torch
from pathlib import Path
import tempfile

from RAG_supporters.dataset.tensor_utils import (
    load_tensor_artifact,
    load_multiple_tensors,
    save_tensor_artifact,
)


class TestLoadTensorArtifact:
    """Tests for load_tensor_artifact function."""

    def test_load_existing_tensor(self, tmp_path):
        """Test loading existing tensor succeeds."""
        tensor = torch.randn(10, 5)
        torch.save(tensor, tmp_path / "test.pt")
        
        loaded = load_tensor_artifact(tmp_path, "test.pt")
        assert loaded.shape == (10, 5), "Should load tensor with correct shape"
        assert torch.allclose(loaded, tensor), "Should load tensor with same values"

    def test_load_existing_tensor_path_object(self, tmp_path):
        """Test loading with Path object works."""
        tensor = torch.randn(10, 5)
        torch.save(tensor, tmp_path / "test.pt")
        
        loaded = load_tensor_artifact(Path(tmp_path), "test.pt")
        assert loaded.shape == (10, 5), "Should work with Path object"

    def test_load_missing_required_file(self, tmp_path):
        """Test loading missing required file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Required tensor file not found"):
            load_tensor_artifact(tmp_path, "missing.pt", required=True)

    def test_load_missing_optional_file(self, tmp_path):
        """Test loading missing optional file returns None."""
        result = load_tensor_artifact(tmp_path, "missing.pt", required=False)
        assert result is None, "Should return None for optional missing file"

    def test_load_with_weights_only_true(self, tmp_path):
        """Test loading with weights_only=True."""
        tensor = torch.randn(10, 5)
        torch.save(tensor, tmp_path / "test.pt")
        
        loaded = load_tensor_artifact(tmp_path, "test.pt", weights_only=True)
        assert loaded.shape == (10, 5), "Should load with weights_only=True"

    def test_load_with_weights_only_false(self, tmp_path):
        """Test loading with weights_only=False for non-tensor data."""
        data = [[1, 2, 3], [4, 5]]
        torch.save(data, tmp_path / "data.pt")
        
        loaded = load_tensor_artifact(tmp_path, "data.pt", weights_only=False)
        assert loaded == [[1, 2, 3], [4, 5]], "Should load non-tensor data with weights_only=False"

    def test_shape_validation_correct_2d(self, tmp_path):
        """Test shape validation with correct 2D shape."""
        tensor = torch.randn(10, 384)
        torch.save(tensor, tmp_path / "test.pt")
        
        loaded = load_tensor_artifact(
            tmp_path, "test.pt",
            expected_shape=(None, 384)
        )
        assert loaded.shape == (10, 384), "Should accept correct shape"

    def test_shape_validation_correct_1d(self, tmp_path):
        """Test shape validation with correct 1D shape."""
        tensor = torch.randn(100)
        torch.save(tensor, tmp_path / "test.pt")
        
        loaded = load_tensor_artifact(
            tmp_path, "test.pt",
            expected_shape=(None,)
        )
        assert loaded.shape == (100,), "Should accept correct 1D shape"

    def test_shape_validation_exact_match(self, tmp_path):
        """Test shape validation with exact dimensions."""
        tensor = torch.randn(10, 384)
        torch.save(tensor, tmp_path / "test.pt")
        
        loaded = load_tensor_artifact(
            tmp_path, "test.pt",
            expected_shape=(10, 384)
        )
        assert loaded.shape == (10, 384), "Should accept exact shape match"

    def test_shape_validation_wrong_ndim(self, tmp_path):
        """Test shape validation with wrong number of dimensions raises ValueError."""
        tensor = torch.randn(10, 384)
        torch.save(tensor, tmp_path / "test.pt")
        
        with pytest.raises(ValueError, match="has 2 dimensions, expected 1"):
            load_tensor_artifact(
                tmp_path, "test.pt",
                expected_shape=(None,)  # Expected 1D, got 2D
            )

    def test_shape_validation_wrong_columns(self, tmp_path):
        """Test shape validation with wrong column count raises ValueError."""
        tensor = torch.randn(10, 384)
        torch.save(tensor, tmp_path / "test.pt")
        
        with pytest.raises(ValueError, match="dimension 1 has size 384, expected 512"):
            load_tensor_artifact(
                tmp_path, "test.pt",
                expected_shape=(None, 512)  # Wrong dimension
            )

    def test_shape_validation_wrong_rows(self, tmp_path):
        """Test shape validation with wrong row count raises ValueError."""
        tensor = torch.randn(10, 384)
        torch.save(tensor, tmp_path / "test.pt")
        
        with pytest.raises(ValueError, match="dimension 0 has size 10, expected 20"):
            load_tensor_artifact(
                tmp_path, "test.pt",
                expected_shape=(20, 384)  # Wrong first dimension
            )

    def test_no_shape_validation(self, tmp_path):
        """Test that no validation is performed when expected_shape is None."""
        tensor = torch.randn(10, 384)
        torch.save(tensor, tmp_path / "test.pt")
        
        loaded = load_tensor_artifact(tmp_path, "test.pt", expected_shape=None)
        assert loaded.shape == (10, 384), "Should load without validation"


class TestLoadMultipleTensors:
    """Tests for load_multiple_tensors function."""

    def test_batch_loading_two_tensors(self, tmp_path):
        """Test loading multiple tensors at once."""
        t1 = torch.randn(10, 5)
        t2 = torch.randn(20, 3)
        torch.save(t1, tmp_path / "t1.pt")
        torch.save(t2, tmp_path / "t2.pt")
        
        specs = [
            ("tensor1", "t1.pt", True, (None, 5)),
            ("tensor2", "t2.pt", True, (None, 3)),
        ]
        
        result = load_multiple_tensors(tmp_path, specs)
        assert "tensor1" in result, "Should load first tensor"
        assert "tensor2" in result, "Should load second tensor"
        assert result["tensor1"].shape == (10, 5), "Should have correct shape for tensor1"
        assert result["tensor2"].shape == (20, 3), "Should have correct shape for tensor2"
        assert torch.allclose(result["tensor1"], t1), "Should load correct values for tensor1"

    def test_batch_loading_multiple_tensors(self, tmp_path):
        """Test loading many tensors efficiently."""
        tensors = {
            "q_embs": torch.randn(100, 384),
            "s_embs": torch.randn(200, 384),
            "k_embs": torch.randn(50, 384),
            "pairs": torch.randint(0, 100, (500, 2)),
        }
        
        for name, tensor in tensors.items():
            torch.save(tensor, tmp_path / f"{name}.pt")
        
        specs = [
            ("question_embs", "q_embs.pt", True, (None, 384)),
            ("source_embs", "s_embs.pt", True, (None, 384)),
            ("keyword_embs", "k_embs.pt", True, (None, 384)),
            ("pair_index", "pairs.pt", True, (None, 2)),
        ]
        
        result = load_multiple_tensors(tmp_path, specs)
        assert len(result) == 4, "Should load all 4 tensors"
        assert result["question_embs"].shape == (100, 384), "Should have correct shape"
        assert result["pair_index"].shape == (500, 2), "Should have correct shape"

    def test_batch_loading_with_non_tensor(self, tmp_path):
        """Test loading mix of tensors and non-tensors."""
        t1 = torch.randn(10, 5)
        data = [[1, 2], [3, 4]]
        torch.save(t1, tmp_path / "t1.pt")
        torch.save(data, tmp_path / "data.pt")
        
        specs = [
            ("tensor1", "t1.pt", True, (None, 5)),
            ("data", "data.pt", False, None),  # weights_only=False, no shape check
        ]
        
        result = load_multiple_tensors(tmp_path, specs)
        assert isinstance(result["tensor1"], torch.Tensor), "Should load tensor"
        assert result["data"] == [[1, 2], [3, 4]], "Should load list data"

    def test_empty_specs_list(self, tmp_path):
        """Test loading with empty specs returns empty dict."""
        specs = []
        result = load_multiple_tensors(tmp_path, specs)
        assert result == {}, "Should return empty dict for empty specs"

    def test_missing_file_in_batch(self, tmp_path):
        """Test missing file in batch raises FileNotFoundError."""
        t1 = torch.randn(10, 5)
        torch.save(t1, tmp_path / "t1.pt")
        
        specs = [
            ("tensor1", "t1.pt", True, (None, 5)),
            ("missing", "missing.pt", True, None),
        ]
        
        with pytest.raises(FileNotFoundError, match="Required tensor file not found"):
            load_multiple_tensors(tmp_path, specs)

    def test_shape_validation_failure_in_batch(self, tmp_path):
        """Test shape validation failure in batch raises ValueError."""
        t1 = torch.randn(10, 5)
        t2 = torch.randn(20, 3)
        torch.save(t1, tmp_path / "t1.pt")
        torch.save(t2, tmp_path / "t2.pt")
        
        specs = [
            ("tensor1", "t1.pt", True, (None, 5)),
            ("tensor2", "t2.pt", True, (None, 10)),  # Wrong shape
        ]
        
        with pytest.raises(ValueError, match="dimension 1 has size 3, expected 10"):
            load_multiple_tensors(tmp_path, specs)


class TestSaveTensorArtifact:
    """Tests for save_tensor_artifact function."""

    def test_save_valid_tensor_2d(self, tmp_path):
        """Test saving valid 2D tensor succeeds."""
        tensor = torch.randn(10, 5)
        save_tensor_artifact(tensor, tmp_path, "output.pt")
        
        assert (tmp_path / "output.pt").exists(), "Should create file"
        loaded = torch.load(tmp_path / "output.pt", weights_only=True)
        assert torch.allclose(loaded, tensor), "Should save correctly"

    def test_save_valid_tensor_1d(self, tmp_path):
        """Test saving valid 1D tensor succeeds."""
        tensor = torch.randn(100)
        save_tensor_artifact(tensor, tmp_path, "output.pt")
        
        assert (tmp_path / "output.pt").exists(), "Should create file"
        loaded = torch.load(tmp_path / "output.pt", weights_only=True)
        assert torch.allclose(loaded, tensor), "Should save correctly"

    def test_save_creates_directory(self, tmp_path):
        """Test saving creates nested directories if needed."""
        nested_dir = tmp_path / "nested" / "dir"
        tensor = torch.randn(10, 5)
        
        save_tensor_artifact(tensor, nested_dir, "output.pt")
        
        assert (nested_dir / "output.pt").exists(), "Should create nested directories"

    def test_save_tensor_with_nan_validation_enabled(self, tmp_path):
        """Test saving tensor with NaN raises ValueError when validation enabled."""
        tensor = torch.randn(10, 5)
        tensor[0, 0] = float('nan')
        
        with pytest.raises(ValueError, match="tensor contains NaN values"):
            save_tensor_artifact(tensor, tmp_path, "bad.pt", validate=True)

    def test_save_tensor_with_inf_validation_enabled(self, tmp_path):
        """Test saving tensor with Inf raises ValueError when validation enabled."""
        tensor = torch.randn(10, 5)
        tensor[0, 0] = float('inf')
        
        with pytest.raises(ValueError, match="tensor contains Inf values"):
            save_tensor_artifact(tensor, tmp_path, "bad.pt", validate=True)

    def test_save_tensor_with_negative_inf(self, tmp_path):
        """Test saving tensor with -Inf raises ValueError."""
        tensor = torch.randn(10, 5)
        tensor[0, 0] = float('-inf')
        
        with pytest.raises(ValueError, match="tensor contains Inf values"):
            save_tensor_artifact(tensor, tmp_path, "bad.pt", validate=True)

    def test_save_without_validation_allows_nan(self, tmp_path):
        """Test saving with validation disabled allows NaN."""
        tensor = torch.randn(10, 5)
        tensor[0, 0] = float('nan')
        
        # Should not raise
        save_tensor_artifact(tensor, tmp_path, "allow_nan.pt", validate=False)
        assert (tmp_path / "allow_nan.pt").exists(), "Should save even with NaN"
        
        loaded = torch.load(tmp_path / "allow_nan.pt", weights_only=True)
        assert torch.isnan(loaded[0, 0]), "Should preserve NaN value"

    def test_save_without_validation_allows_inf(self, tmp_path):
        """Test saving with validation disabled allows Inf."""
        tensor = torch.randn(10, 5)
        tensor[0, 0] = float('inf')
        
        # Should not raise
        save_tensor_artifact(tensor, tmp_path, "allow_inf.pt", validate=False)
        assert (tmp_path / "allow_inf.pt").exists(), "Should save even with Inf"
        
        loaded = torch.load(tmp_path / "allow_inf.pt", weights_only=True)
        assert torch.isinf(loaded[0, 0]), "Should preserve Inf value"

    def test_save_with_validation_default_enabled(self, tmp_path):
        """Test validation is enabled by default."""
        tensor = torch.randn(10, 5)
        tensor[0, 0] = float('nan')
        
        # Should raise with default validate=True
        with pytest.raises(ValueError, match="NaN"):
            save_tensor_artifact(tensor, tmp_path, "bad.pt")

    def test_save_overwrites_existing_file(self, tmp_path):
        """Test saving overwrites existing file."""
        tensor1 = torch.ones(5, 3)
        tensor2 = torch.zeros(5, 3)
        
        save_tensor_artifact(tensor1, tmp_path, "overwrite.pt")
        save_tensor_artifact(tensor2, tmp_path, "overwrite.pt")
        
        loaded = torch.load(tmp_path / "overwrite.pt", weights_only=True)
        assert torch.allclose(loaded, tensor2), "Should overwrite with new tensor"
        assert not torch.allclose(loaded, tensor1), "Should not contain old tensor"

    def test_save_path_object(self, tmp_path):
        """Test saving with Path object works."""
        tensor = torch.randn(10, 5)
        save_tensor_artifact(tensor, Path(tmp_path), "output.pt")
        
        assert (tmp_path / "output.pt").exists(), "Should work with Path object"

    def test_save_int_tensor(self, tmp_path):
        """Test saving integer tensor."""
        tensor = torch.randint(0, 100, (10, 5))
        save_tensor_artifact(tensor, tmp_path, "int_tensor.pt")
        
        loaded = torch.load(tmp_path / "int_tensor.pt", weights_only=True)
        assert torch.equal(loaded, tensor), "Should save integer tensor correctly"

    def test_save_long_tensor(self, tmp_path):
        """Test saving long tensor."""
        tensor = torch.randint(0, 1000, (100,), dtype=torch.long)
        save_tensor_artifact(tensor, tmp_path, "long_tensor.pt")
        
        loaded = torch.load(tmp_path / "long_tensor.pt", weights_only=True)
        assert torch.equal(loaded, tensor), "Should save long tensor correctly"
