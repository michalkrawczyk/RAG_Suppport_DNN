"""
Unit tests for cluster steering agent and LLM-driven features.

Tests cover:
- Prompt template validation
- Cluster steering agent initialization
- Validation utilities
- Mock LLM integration
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

# Test prompt templates
from RAG_supporters.prompts_templates.cluster_steering import (
    AMBIGUITY_RESOLUTION_PROMPT,
    CLUSTER_ACTIVATION_PROMPT,
    MULTI_CLUSTER_REPHRASE_PROMPT,
    QUESTION_REPHRASE_PROMPT,
)

# Test validation utilities
from RAG_supporters.utils.llm_output_validation import (
    analyze_descriptor_usage,
    compute_confidence_statistics,
    validate_ambiguity_resolution,
    validate_question_rephrase,
    validate_steering_text,
)


class TestPromptTemplates:
    """Test prompt template structure."""

    def test_cluster_activation_prompt_placeholders(self):
        """Test that cluster activation prompt has required placeholders."""
        required_placeholders = ["{question}", "{cluster_id}", "{cluster_descriptors}"]
        for placeholder in required_placeholders:
            assert (
                placeholder in CLUSTER_ACTIVATION_PROMPT
            ), f"Missing placeholder: {placeholder}"

    def test_question_rephrase_prompt_placeholders(self):
        """Test that question rephrase prompt has required placeholders."""
        required_placeholders = [
            "{question}",
            "{cluster_id}",
            "{cluster_descriptors}",
            "{alternate_clusters}",
        ]
        for placeholder in required_placeholders:
            assert (
                placeholder in QUESTION_REPHRASE_PROMPT
            ), f"Missing placeholder: {placeholder}"

    def test_multi_cluster_rephrase_prompt_placeholders(self):
        """Test that multi-cluster rephrase prompt has required placeholders."""
        required_placeholders = ["{question}", "{cluster_info}", "{num_variations}"]
        for placeholder in required_placeholders:
            assert (
                placeholder in MULTI_CLUSTER_REPHRASE_PROMPT
            ), f"Missing placeholder: {placeholder}"

    def test_ambiguity_resolution_prompt_placeholders(self):
        """Test that ambiguity resolution prompt has required placeholders."""
        required_placeholders = ["{question}", "{cluster_assignments}"]
        for placeholder in required_placeholders:
            assert (
                placeholder in AMBIGUITY_RESOLUTION_PROMPT
            ), f"Missing placeholder: {placeholder}"


class TestValidationUtilities:
    """Test validation utilities for LLM outputs."""

    def test_validate_steering_text_valid(self):
        """Test validation of valid steering text result."""
        valid_result = {
            "cluster_id": 0,
            "steering_text": "Test steering text for cluster activation",
            "incorporated_descriptors": ["deep learning", "AI"],
            "confidence": 0.85,
        }
        is_valid, errors = validate_steering_text(valid_result)
        assert is_valid
        assert len(errors) == 0

    def test_validate_steering_text_missing_field(self):
        """Test validation fails with missing field."""
        invalid_result = {
            "cluster_id": 0,
            "steering_text": "Test steering text",
            # Missing: incorporated_descriptors and confidence
        }
        is_valid, errors = validate_steering_text(invalid_result)
        assert not is_valid
        assert len(errors) > 0
        assert any("Missing required field" in err for err in errors)

    def test_validate_steering_text_invalid_confidence(self):
        """Test validation fails with invalid confidence score."""
        invalid_result = {
            "cluster_id": 0,
            "steering_text": "Test steering text",
            "incorporated_descriptors": ["test"],
            "confidence": 1.5,  # Invalid: > 1.0
        }
        is_valid, errors = validate_steering_text(invalid_result)
        assert not is_valid
        assert any("confidence must be in [0, 1]" in err for err in errors)

    def test_validate_steering_text_empty_text(self):
        """Test validation fails with empty steering text."""
        invalid_result = {
            "cluster_id": 0,
            "steering_text": "   ",  # Empty/whitespace only
            "incorporated_descriptors": ["test"],
            "confidence": 0.8,
        }
        is_valid, errors = validate_steering_text(invalid_result)
        assert not is_valid
        assert any("empty" in err.lower() for err in errors)

    def test_validate_question_rephrase_valid(self):
        """Test validation of valid question rephrase result."""
        valid_result = {
            "original_question": "What is AI?",
            "target_cluster_id": 1,
            "rephrased_question": "Can you explain artificial intelligence concepts?",
            "genre_shift": "Shifted to educational tone",
            "preserved_intent": "Understanding AI basics",
            "confidence": 0.9,
        }
        is_valid, errors = validate_question_rephrase(valid_result)
        assert is_valid
        assert len(errors) == 0

    def test_validate_question_rephrase_identical(self):
        """Test validation fails when rephrased is identical to original."""
        invalid_result = {
            "original_question": "What is AI?",
            "target_cluster_id": 1,
            "rephrased_question": "What is AI?",  # Identical
            "genre_shift": "No shift",
            "preserved_intent": "Same",
            "confidence": 0.9,
        }
        is_valid, errors = validate_question_rephrase(invalid_result)
        assert not is_valid
        assert any("identical" in err.lower() for err in errors)

    def test_validate_ambiguity_resolution_valid(self):
        """Test validation of valid ambiguity resolution result."""
        valid_result = {
            "question": "Test question",
            "is_ambiguous": True,
            "primary_cluster": {
                "cluster_id": 0,
                "reason": "Primary topic",
                "confidence": 0.7,
            },
            "secondary_clusters": [
                {"cluster_id": 1, "reason": "Related topic", "confidence": 0.5}
            ],
            "recommendation": "multi-domain",
            "explanation": "Question spans multiple domains",
        }
        is_valid, errors = validate_ambiguity_resolution(valid_result)
        assert is_valid
        assert len(errors) == 0

    def test_validate_ambiguity_resolution_invalid_recommendation(self):
        """Test validation fails with invalid recommendation value."""
        invalid_result = {
            "question": "Test question",
            "is_ambiguous": True,
            "primary_cluster": {"cluster_id": 0, "reason": "Test", "confidence": 0.7},
            "secondary_clusters": [],
            "recommendation": "invalid-value",  # Invalid
            "explanation": "Test",
        }
        is_valid, errors = validate_ambiguity_resolution(invalid_result)
        assert not is_valid
        assert any("recommendation must be" in err for err in errors)

    def test_compute_confidence_statistics(self):
        """Test confidence statistics computation."""
        results = [
            {"confidence": 0.9},
            {"confidence": 0.8},
            {"confidence": 0.7},
            {"confidence": 0.85},
        ]
        stats = compute_confidence_statistics(results)

        assert stats["count"] == 4
        assert 0.8 <= stats["mean"] <= 0.9
        assert stats["min"] == 0.7
        assert stats["max"] == 0.9
        assert stats["std"] > 0

    def test_compute_confidence_statistics_empty(self):
        """Test confidence statistics with empty input."""
        stats = compute_confidence_statistics([])
        assert stats["count"] == 0
        assert stats["mean"] == 0.0

    def test_analyze_descriptor_usage(self):
        """Test descriptor usage analysis."""
        results = [
            {"incorporated_descriptors": ["AI", "machine learning"]},
            {"incorporated_descriptors": ["deep learning", "AI"]},
            {"incorporated_descriptors": ["AI", "neural networks"]},
        ]
        analysis = analyze_descriptor_usage(results)

        assert analysis["total_results"] == 3
        assert analysis["results_with_descriptors"] == 3
        assert analysis["total_descriptors_used"] == 6
        assert analysis["unique_descriptors"] == 4  # AI, ML, DL, NN
        assert analysis["avg_descriptors_per_result"] == 2.0

        # Check most common
        most_common = dict(analysis["most_common_descriptors"])
        assert most_common["AI"] == 3

    def test_analyze_descriptor_usage_with_validation(self):
        """Test descriptor usage analysis with available descriptors validation."""
        results = [
            {"incorporated_descriptors": ["valid1", "invalid1"]},
            {"incorporated_descriptors": ["valid2", "valid1"]},
        ]
        available = {"valid1", "valid2", "valid3"}

        analysis = analyze_descriptor_usage(results, available)

        assert "invalid_descriptors" in analysis
        assert "invalid1" in analysis["invalid_descriptors"]
        assert analysis["invalid_count"] == 1


class TestClusterSteeringAgent:
    """Test ClusterSteeringAgent with mock LLM."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing."""
        llm = Mock()
        return llm

    @pytest.fixture
    def mock_response_steering(self):
        """Mock LLM response for steering text generation."""
        response = Mock()
        response.content = json.dumps(
            {
                "cluster_id": 0,
                "steering_text": "Test steering text with keywords",
                "incorporated_descriptors": ["keyword1", "keyword2"],
                "confidence": 0.85,
            }
        )
        return response

    @pytest.fixture
    def mock_response_rephrase(self):
        """Mock LLM response for question rephrasing."""
        response = Mock()
        response.content = json.dumps(
            {
                "original_question": "Original question",
                "target_cluster_id": 1,
                "rephrased_question": "Rephrased version of the question",
                "genre_shift": "Technical to conversational",
                "preserved_intent": "Core information need",
                "confidence": 0.9,
            }
        )
        return response

    def test_agent_initialization_with_dependencies(self, mock_llm):
        """Test agent initialization when dependencies are available."""
        try:
            from RAG_supporters.agents.cluster_steering import ClusterSteeringAgent

            agent = ClusterSteeringAgent(llm=mock_llm, max_retries=2)
            assert agent.llm == mock_llm
            assert agent.max_retries == 2
        except ImportError:
            pytest.skip("ClusterSteeringAgent dependencies not available")

    def test_generate_steering_text_with_mock(
        self, mock_llm, mock_response_steering
    ):
        """Test steering text generation with mock LLM."""
        try:
            from RAG_supporters.agents.cluster_steering import ClusterSteeringAgent

            mock_llm.invoke.return_value = mock_response_steering
            agent = ClusterSteeringAgent(llm=mock_llm, max_retries=1)

            result = agent.generate_steering_text(
                question="What is deep learning?",
                cluster_id=0,
                cluster_descriptors=["AI", "neural networks"],
            )

            assert result is not None
            assert result["cluster_id"] == 0
            assert "steering_text" in result
            assert len(result["incorporated_descriptors"]) > 0
            assert 0 <= result["confidence"] <= 1
        except ImportError:
            pytest.skip("ClusterSteeringAgent dependencies not available")

    def test_rephrase_question_with_mock(self, mock_llm, mock_response_rephrase):
        """Test question rephrasing with mock LLM."""
        try:
            from RAG_supporters.agents.cluster_steering import ClusterSteeringAgent

            mock_llm.invoke.return_value = mock_response_rephrase
            agent = ClusterSteeringAgent(llm=mock_llm, max_retries=1)

            result = agent.rephrase_question(
                question="Original question",
                target_cluster_id=1,
                cluster_descriptors=["keyword1", "keyword2"],
            )

            assert result is not None
            assert result["target_cluster_id"] == 1
            assert "rephrased_question" in result
            assert result["rephrased_question"] != result["original_question"]
        except ImportError:
            pytest.skip("ClusterSteeringAgent dependencies not available")


class TestJSONFileValidation:
    """Test JSON file validation utilities."""

    def test_validate_json_file_with_tempfile(self):
        """Test validation of JSON file."""
        # Create temporary file with test data
        test_data = [
            {
                "cluster_id": 0,
                "steering_text": "Test 1",
                "incorporated_descriptors": ["desc1"],
                "confidence": 0.8,
            },
            {
                "cluster_id": 1,
                "steering_text": "Test 2",
                "incorporated_descriptors": ["desc2"],
                "confidence": 0.9,
            },
        ]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(test_data, f)
            temp_path = Path(f.name)

        try:
            from RAG_supporters.utils.llm_output_validation import validate_json_file

            total, valid, errors = validate_json_file(temp_path, validate_steering_text)
            assert total == 2
            assert valid == 2
            assert len(errors) == 0
        finally:
            temp_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
