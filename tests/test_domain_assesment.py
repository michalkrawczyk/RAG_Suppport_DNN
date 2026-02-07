"""Tests for domain assessment agent."""

import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# Skip all tests if required dependencies are not installed
pytest.importorskip("langchain")
pytest.importorskip("langchain_core")
pytest.importorskip("langgraph")
pytest.importorskip("pydantic")
pytest.importorskip("tqdm")
pytest.importorskip("numpy")
pytest.importorskip("pandas")

import numpy as np
import pandas as pd


def test_pydantic_models_import():
    """Test that Pydantic models can be imported."""
    from RAG_supporters.agents.domain_assesment import (
        DomainAssessmentResult,
        DomainExtractionResult,
        DomainGuessResult,
        DomainSuggestion,
        QuestionTopicRelevanceResult,
        SelectedTerm,
        TopicRelevanceScore,
    )

    # Test DomainSuggestion
    suggestion = DomainSuggestion(
        term="machine learning",
        type="domain",
        confidence=0.9,
        reason="Core topic of the text",
    )
    assert suggestion.term == "machine learning"
    assert suggestion.confidence == 0.9

    # Test DomainExtractionResult
    extraction = DomainExtractionResult(
        suggestions=[suggestion],
        total_suggestions=1,
        primary_theme="Artificial Intelligence",
    )
    assert extraction.total_suggestions == 1
    assert extraction.primary_theme == "Artificial Intelligence"

    # Test SelectedTerm
    selected = SelectedTerm(
        term="database",
        type="keyword",
        relevance_score=0.8,
        reason="Mentioned in context",
    )
    assert selected.term == "database"
    assert selected.relevance_score == 0.8

    # Test TopicRelevanceScore
    topic_score = TopicRelevanceScore(
        topic_descriptor="neural networks",
        probability=0.75,
        reason="Related to ML concepts",
    )
    assert topic_score.topic_descriptor == "neural networks"
    assert topic_score.probability == 0.75


def test_domain_suggestion_validation():
    """Test DomainSuggestion validation."""
    from RAG_supporters.agents.domain_assesment import DomainSuggestion

    # Valid suggestion
    suggestion = DomainSuggestion(term="test", type="domain", confidence=0.5, reason="test reason")
    assert suggestion.confidence == 0.5

    # Test confidence bounds
    with pytest.raises(Exception):  # Pydantic validation error
        DomainSuggestion(term="test", type="domain", confidence=1.5, reason="test")  # > 1.0

    with pytest.raises(Exception):
        DomainSuggestion(term="test", type="domain", confidence=-0.1, reason="test")  # < 0.0


def test_domain_extraction_result_validation():
    """Test DomainExtractionResult auto-correction of total_suggestions."""
    from RAG_supporters.agents.domain_assesment import (
        DomainExtractionResult,
        DomainSuggestion,
    )

    suggestions = [
        DomainSuggestion(term="test1", type="domain", confidence=0.9, reason="r1"),
        DomainSuggestion(term="test2", type="domain", confidence=0.8, reason="r2"),
    ]

    # Mismatched total_suggestions should be corrected
    result = DomainExtractionResult(
        suggestions=suggestions,
        total_suggestions=5,  # Wrong count
        primary_theme="Test Theme",
    )

    # Should be auto-corrected to 2
    assert result.total_suggestions == 2


def test_question_topic_relevance_result_validation():
    """Test QuestionTopicRelevanceResult validation."""
    from RAG_supporters.agents.domain_assesment import (
        QuestionTopicRelevanceResult,
        TopicRelevanceScore,
    )

    scores = [
        TopicRelevanceScore(topic_descriptor="topic1", probability=0.8),
        TopicRelevanceScore(topic_descriptor="topic2", probability=0.6),
    ]

    result = QuestionTopicRelevanceResult(
        topic_scores=scores,
        total_topics=2,
        question_summary="Test summary",
    )

    assert result.total_topics == 2
    assert len(result.topic_scores) == 2

    # Test auto-correction
    result_wrong = QuestionTopicRelevanceResult(
        topic_scores=scores,
        total_topics=10,  # Wrong count
        question_summary="Test",
    )
    assert result_wrong.total_topics == 2  # Should be corrected


def test_operation_mode_enum():
    """Test OperationMode enum."""
    from RAG_supporters.agents.domain_assesment import OperationMode

    assert OperationMode.EXTRACT.value == "extract"
    assert OperationMode.GUESS.value == "guess"
    assert OperationMode.ASSESS.value == "assess"
    assert OperationMode.TOPIC_RELEVANCE_PROB.value == "topic_relevance_prob"


def create_mock_llm():
    """Create a mock LLM for testing."""
    mock_llm = MagicMock()
    mock_llm.invoke = MagicMock()
    mock_llm.batch = MagicMock()
    return mock_llm


def test_domain_analysis_agent_initialization():
    """Test DomainAnalysisAgent initialization."""
    from RAG_supporters.agents.domain_assesment import DomainAnalysisAgent

    mock_llm = create_mock_llm()

    agent = DomainAnalysisAgent(
        llm=mock_llm,
        max_retries=3,
        batch_size=10,
        include_reason=False,
    )

    assert agent.llm == mock_llm
    assert agent.max_retries == 3
    assert agent.batch_size == 10
    assert agent.include_reason is False
    assert agent.extraction_parser is not None
    assert agent.guess_parser is not None
    assert agent.assessment_parser is not None
    assert agent.topic_relevance_prob_parser is not None


def test_parse_topic_descriptors_list_of_strings():
    """Test parsing topic descriptors from list of strings."""
    from RAG_supporters.agents.domain_assesment import DomainAnalysisAgent

    mock_llm = create_mock_llm()
    agent = DomainAnalysisAgent(llm=mock_llm)

    descriptors = ["machine learning", "databases", "web development"]
    result = agent._parse_topic_descriptors(descriptors)

    assert result == descriptors
    assert len(result) == 3


def test_parse_topic_descriptors_json_string():
    """Test parsing topic descriptors from JSON string."""
    from RAG_supporters.agents.domain_assesment import DomainAnalysisAgent

    mock_llm = create_mock_llm()
    agent = DomainAnalysisAgent(llm=mock_llm)

    descriptors = ["machine learning", "databases", "web development"]
    json_string = json.dumps(descriptors)

    result = agent._parse_topic_descriptors(json_string)

    assert result == descriptors


def test_parse_topic_descriptors_keywordclusterer_dict():
    """Test parsing topic descriptors from KeywordClusterer dict format."""
    from RAG_supporters.agents.domain_assesment import DomainAnalysisAgent

    mock_llm = create_mock_llm()
    agent = DomainAnalysisAgent(llm=mock_llm)

    clusterer_data = {
        "cluster_stats": {
            "0": {
                "topic_descriptors": ["machine learning", "AI", "neural networks"],
                "size": 10,
            },
            "1": {
                "topic_descriptors": ["database", "SQL", "storage"],
                "size": 8,
            },
            "2": {
                "topic_descriptors": ["web development", "frontend"],
                "size": 12,
            },
        }
    }

    result = agent._parse_topic_descriptors(clusterer_data)

    # Should extract all unique topic descriptors
    expected = [
        "machine learning",
        "AI",
        "neural networks",
        "database",
        "SQL",
        "storage",
        "web development",
        "frontend",
    ]
    assert len(result) == len(expected)
    assert set(result) == set(expected)


def test_parse_topic_descriptors_file_path():
    """Test parsing topic descriptors from file path."""
    from RAG_supporters.agents.domain_assesment import DomainAnalysisAgent

    mock_llm = create_mock_llm()
    agent = DomainAnalysisAgent(llm=mock_llm)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create JSON file
        descriptors_file = tmpdir / "descriptors.json"
        descriptors = ["machine learning", "databases", "web development"]

        with open(descriptors_file, "w", encoding="utf-8") as f:
            json.dump(descriptors, f)

        # Parse from file path
        result = agent._parse_topic_descriptors(str(descriptors_file))

        assert result == descriptors


def test_parse_topic_descriptors_keywordclusterer_file():
    """Test parsing topic descriptors from KeywordClusterer JSON file."""
    from RAG_supporters.agents.domain_assesment import DomainAnalysisAgent

    mock_llm = create_mock_llm()
    agent = DomainAnalysisAgent(llm=mock_llm)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create KeywordClusterer JSON file
        clusterer_file = tmpdir / "clusters.json"
        clusterer_data = {
            "cluster_stats": {
                "0": {"topic_descriptors": ["topic1", "topic2"], "size": 10},
                "1": {"topic_descriptors": ["topic3", "topic4"], "size": 8},
            }
        }

        with open(clusterer_file, "w", encoding="utf-8") as f:
            json.dump(clusterer_data, f)

        # Parse from file path
        result = agent._parse_topic_descriptors(str(clusterer_file))

        assert set(result) == {"topic1", "topic2", "topic3", "topic4"}


def test_parse_topic_descriptors_invalid_dict():
    """Test error handling for invalid dict format."""
    from RAG_supporters.agents.domain_assesment import DomainAnalysisAgent

    mock_llm = create_mock_llm()
    agent = DomainAnalysisAgent(llm=mock_llm)

    # Dict without cluster_stats or clusters
    invalid_dict = {"some_key": "some_value"}

    with pytest.raises(ValueError, match="no 'cluster_stats' or 'clusters' keys found"):
        agent._parse_topic_descriptors(invalid_dict)


def test_parse_topic_descriptors_invalid_file():
    """Test handling of non-existent file path (treated as single descriptor)."""
    from RAG_supporters.agents.domain_assesment import DomainAnalysisAgent

    mock_llm = create_mock_llm()
    agent = DomainAnalysisAgent(llm=mock_llm)

    # Non-existent file paths are treated as single descriptors (not validated as files)
    result = agent._parse_topic_descriptors("/nonexistent/path/to/file.json")
    assert result == [
        "/nonexistent/path/to/file.json"
    ], "Non-existent paths should be treated as single descriptors"


def test_parse_topic_descriptors_single_string():
    """Test parsing single string as single descriptor."""
    from RAG_supporters.agents.domain_assesment import DomainAnalysisAgent

    mock_llm = create_mock_llm()
    agent = DomainAnalysisAgent(llm=mock_llm)

    # Non-JSON string should be treated as single descriptor
    result = agent._parse_topic_descriptors("machine learning")

    assert result == ["machine learning"]


def test_parse_topic_descriptors_fallback_to_clusters():
    """Test fallback to 'clusters' key when 'cluster_stats' is missing."""
    from RAG_supporters.agents.domain_assesment import DomainAnalysisAgent

    mock_llm = create_mock_llm()
    agent = DomainAnalysisAgent(llm=mock_llm)

    # KeywordClusterer dict with only 'clusters' key
    clusterer_data = {
        "clusters": {
            "0": ["keyword1", "keyword2", "keyword3"],
            "1": ["keyword4", "keyword5"],
        }
    }

    result = agent._parse_topic_descriptors(clusterer_data)

    # Should extract all unique keywords
    expected = ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"]
    assert set(result) == set(expected)


def test_create_relevance_json_mapping():
    """Test creating JSON mapping from topic scores."""
    from RAG_supporters.agents.domain_assesment import DomainAnalysisAgent

    mock_llm = create_mock_llm()
    agent = DomainAnalysisAgent(llm=mock_llm)

    topic_scores = [
        {"topic_descriptor": "machine learning", "probability": 0.9},
        {"topic_descriptor": "databases", "probability": 0.5},
        {"topic_descriptor": "web development", "probability": 0.3},
    ]

    json_mapping = agent._create_relevance_json_mapping(topic_scores)

    # Parse JSON
    mapping = json.loads(json_mapping)

    assert isinstance(mapping, dict)
    assert mapping["machine learning"] == 0.9
    assert mapping["databases"] == 0.5
    assert mapping["web development"] == 0.3


def test_get_column_prefix():
    """Test _get_column_prefix method."""
    from RAG_supporters.agents.domain_assesment import (
        DomainAnalysisAgent,
        OperationMode,
    )

    mock_llm = create_mock_llm()
    agent = DomainAnalysisAgent(llm=mock_llm)

    assert agent._get_column_prefix(OperationMode.EXTRACT) == "extract"
    assert agent._get_column_prefix(OperationMode.GUESS) == "guess"
    assert agent._get_column_prefix(OperationMode.ASSESS) == "assess"
    assert agent._get_column_prefix(OperationMode.TOPIC_RELEVANCE_PROB) == "topic_relevance_prob"


def test_extract_result_dict():
    """Test _extract_result_dict method."""
    from RAG_supporters.agents.domain_assesment import (
        DomainAnalysisAgent,
        DomainSuggestion,
        DomainExtractionResult,
    )

    mock_llm = create_mock_llm()
    agent = DomainAnalysisAgent(llm=mock_llm)

    # Test with None
    assert agent._extract_result_dict(None) is None

    # Test with dict
    test_dict = {"key": "value"}
    assert agent._extract_result_dict(test_dict) == test_dict

    # Test with Pydantic model
    suggestion = DomainSuggestion(term="test", type="domain", confidence=0.9, reason="test reason")
    result = DomainExtractionResult(
        suggestions=[suggestion],
        total_suggestions=1,
        primary_theme="Test",
    )

    extracted = agent._extract_result_dict(result)
    assert isinstance(extracted, dict)
    assert "suggestions" in extracted
    assert "total_suggestions" in extracted
    assert "primary_theme" in extracted


def test_check_openai_llm():
    """Test _check_openai_llm method."""
    from RAG_supporters.agents.domain_assesment import DomainAnalysisAgent

    mock_llm = create_mock_llm()
    agent = DomainAnalysisAgent(llm=mock_llm)

    # With mock LLM, should return False (not OpenAI)
    # This may vary based on whether langchain_openai is installed
    assert isinstance(agent._is_openai_llm, bool)


def test_extract_domains_success():
    """Test extract_domains method with successful result."""
    from RAG_supporters.agents.domain_assesment import DomainAnalysisAgent

    mock_llm = create_mock_llm()
    agent = DomainAnalysisAgent(llm=mock_llm)

    # Mock successful result
    mock_result = {
        "result": {
            "suggestions": [{"term": "AI", "type": "domain", "confidence": 0.9, "reason": "test"}],
            "total_suggestions": 1,
            "primary_theme": "Artificial Intelligence",
        },
        "error": None,
    }

    mock_graph = MagicMock()
    mock_graph.invoke = MagicMock(return_value=mock_result)
    agent.graph = mock_graph

    result = agent.extract_domains("Machine learning is a subset of AI")

    assert result is not None
    assert "suggestions" in result
    assert "primary_theme" in result


def test_extract_domains_failure():
    """Test extract_domains method with failure."""
    from RAG_supporters.agents.domain_assesment import DomainAnalysisAgent

    mock_llm = create_mock_llm()
    agent = DomainAnalysisAgent(llm=mock_llm)

    # Mock failure result
    mock_result = {
        "result": None,
        "error": "Parsing failed",
    }

    mock_graph = MagicMock()
    mock_graph.invoke = MagicMock(return_value=mock_result)
    agent.graph = mock_graph

    result = agent.extract_domains("Test text")

    assert result is None


def test_guess_domains():
    """Test guess_domains method."""
    from RAG_supporters.agents.domain_assesment import DomainAnalysisAgent

    mock_llm = create_mock_llm()
    agent = DomainAnalysisAgent(llm=mock_llm)

    # Mock successful result
    mock_result = {
        "result": {
            "suggestions": [
                {
                    "term": "physics",
                    "type": "domain",
                    "confidence": 0.8,
                    "reason": "test",
                }
            ],
            "total_suggestions": 1,
            "question_category": "Science",
        },
        "error": None,
    }

    mock_graph = MagicMock()
    mock_graph.invoke = MagicMock(return_value=mock_result)
    agent.graph = mock_graph

    result = agent.guess_domains("What is quantum mechanics?")

    assert result is not None
    assert "suggestions" in result
    assert "question_category" in result


def test_assess_domains():
    """Test assess_domains method."""
    from RAG_supporters.agents.domain_assesment import DomainAnalysisAgent

    mock_llm = create_mock_llm()
    agent = DomainAnalysisAgent(llm=mock_llm)

    # Mock successful result
    mock_result = {
        "result": {
            "selected_terms": [
                {
                    "term": "biology",
                    "type": "domain",
                    "relevance_score": 0.9,
                    "reason": "test",
                }
            ],
            "total_selected": 1,
            "question_intent": "Understanding photosynthesis",
            "primary_topics": ["biology"],
        },
        "error": None,
    }

    mock_graph = MagicMock()
    mock_graph.invoke = MagicMock(return_value=mock_result)
    agent.graph = mock_graph

    available_terms = ["physics", "chemistry", "biology"]
    result = agent.assess_domains("What is photosynthesis?", available_terms)

    assert result is not None
    assert "selected_terms" in result
    assert "primary_topics" in result


def test_assess_domains_invalid_json():
    """Test assess_domains with invalid JSON string."""
    from RAG_supporters.agents.domain_assesment import DomainAnalysisAgent

    mock_llm = create_mock_llm()
    agent = DomainAnalysisAgent(llm=mock_llm)

    # Invalid JSON string
    invalid_json = "this is not valid JSON"

    with pytest.raises(ValueError, match="Invalid JSON"):
        agent.assess_domains("What is photosynthesis?", invalid_json)


def test_assess_topic_relevance_prob():
    """Test assess_topic_relevance_prob method."""
    from RAG_supporters.agents.domain_assesment import DomainAnalysisAgent

    mock_llm = create_mock_llm()
    agent = DomainAnalysisAgent(llm=mock_llm)

    # Mock successful result
    mock_result = {
        "result": {
            "topic_scores": [
                {
                    "topic_descriptor": "machine learning",
                    "probability": 0.9,
                    "reason": "test",
                },
                {"topic_descriptor": "databases", "probability": 0.3, "reason": "test"},
            ],
            "total_topics": 2,
            "question_summary": "Test summary",
        },
        "error": None,
    }

    mock_graph = MagicMock()
    mock_graph.invoke = MagicMock(return_value=mock_result)
    agent.graph = mock_graph

    descriptors = ["machine learning", "databases", "web development"]
    result = agent.assess_topic_relevance_prob("What is gradient descent?", descriptors)

    assert result is not None
    assert "topic_scores" in result
    assert "total_topics" in result


def test_assess_topic_relevance_prob_with_keywordclusterer_dict():
    """Test assess_topic_relevance_prob with KeywordClusterer dict."""
    from RAG_supporters.agents.domain_assesment import DomainAnalysisAgent

    mock_llm = create_mock_llm()
    agent = DomainAnalysisAgent(llm=mock_llm)

    # Mock the graph invoke method
    mock_result = {
        "result": {
            "topic_scores": [
                {"topic_descriptor": "machine learning", "probability": 0.9},
                {"topic_descriptor": "AI", "probability": 0.8},
            ],
            "total_topics": 2,
            "question_summary": "Test",
        },
        "error": None,
    }

    with patch.object(agent.graph, "invoke", return_value=mock_result):
        clusterer_data = {
            "cluster_stats": {"0": {"topic_descriptors": ["machine learning", "AI"], "size": 10}}
        }

        result = agent.assess_topic_relevance_prob("What is gradient descent?", clusterer_data)

        assert result is not None
        assert "topic_scores" in result


def test_batch_processing_extract_domains_sequential():
    """Test extract_domains_batch with sequential processing."""
    from RAG_supporters.agents.domain_assesment import DomainAnalysisAgent

    mock_llm = create_mock_llm()
    agent = DomainAnalysisAgent(llm=mock_llm)

    # Mock _is_openai_llm to False to force sequential processing
    agent._is_openai_llm = False

    # Mock extract_domains method
    mock_result = {
        "suggestions": [{"term": "test", "type": "domain", "confidence": 0.9, "reason": "test"}],
        "total_suggestions": 1,
        "primary_theme": "Test",
    }

    with patch.object(agent, "extract_domains", return_value=mock_result):
        texts = ["Text 1", "Text 2", "Text 3"]
        results = agent.extract_domains_batch(texts, show_progress=False)

        assert len(results) == 3
        assert all(r == mock_result for r in results)


def test_batch_processing_guess_domains_sequential():
    """Test guess_domains_batch with sequential processing."""
    from RAG_supporters.agents.domain_assesment import DomainAnalysisAgent

    mock_llm = create_mock_llm()
    agent = DomainAnalysisAgent(llm=mock_llm)

    # Force sequential processing
    agent._is_openai_llm = False

    mock_result = {
        "suggestions": [{"term": "test", "type": "domain", "confidence": 0.9, "reason": "test"}],
        "total_suggestions": 1,
        "question_category": "Test",
    }

    with patch.object(agent, "guess_domains", return_value=mock_result):
        questions = ["Question 1?", "Question 2?"]
        results = agent.guess_domains_batch(questions, show_progress=False)

        assert len(results) == 2
        assert all(r == mock_result for r in results)


def test_batch_processing_assess_domains_sequential():
    """Test assess_domains_batch with sequential processing."""
    from RAG_supporters.agents.domain_assesment import DomainAnalysisAgent

    mock_llm = create_mock_llm()
    agent = DomainAnalysisAgent(llm=mock_llm)

    # Force sequential processing
    agent._is_openai_llm = False

    mock_result = {
        "selected_terms": [
            {"term": "test", "type": "domain", "relevance_score": 0.9, "reason": "test"}
        ],
        "total_selected": 1,
        "question_intent": "Test",
        "primary_topics": ["test"],
    }

    with patch.object(agent, "assess_domains", return_value=mock_result):
        questions = ["Question 1?", "Question 2?"]
        available_terms = ["term1", "term2", "term3"]
        results = agent.assess_domains_batch(questions, available_terms, show_progress=False)

        assert len(results) == 2
        assert all(r == mock_result for r in results)


def test_batch_processing_topic_relevance_sequential():
    """Test assess_topic_relevance_prob_batch with sequential processing."""
    from RAG_supporters.agents.domain_assesment import DomainAnalysisAgent

    mock_llm = create_mock_llm()
    agent = DomainAnalysisAgent(llm=mock_llm)

    # Force sequential processing
    agent._is_openai_llm = False

    mock_result = {
        "topic_scores": [{"topic_descriptor": "topic1", "probability": 0.9, "reason": "test"}],
        "total_topics": 1,
        "question_summary": "Test",
    }

    with patch.object(agent, "assess_topic_relevance_prob", return_value=mock_result):
        questions = ["Question 1?", "Question 2?"]
        descriptors = ["topic1", "topic2"]
        results = agent.assess_topic_relevance_prob_batch(
            questions, descriptors, show_progress=False
        )

        assert len(results) == 2
        assert all(r == mock_result for r in results)


def test_get_parser_for_mode():
    """Test _get_parser_for_mode method."""
    from RAG_supporters.agents.domain_assesment import (
        DomainAnalysisAgent,
        OperationMode,
    )

    mock_llm = create_mock_llm()
    agent = DomainAnalysisAgent(llm=mock_llm)

    # Test each mode
    parser, fixing_parser = agent._get_parser_for_mode(OperationMode.EXTRACT)
    assert parser is not None
    assert fixing_parser is not None

    parser, fixing_parser = agent._get_parser_for_mode(OperationMode.GUESS)
    assert parser is not None
    assert fixing_parser is not None

    parser, fixing_parser = agent._get_parser_for_mode(OperationMode.ASSESS)
    assert parser is not None
    assert fixing_parser is not None

    parser, fixing_parser = agent._get_parser_for_mode(OperationMode.TOPIC_RELEVANCE_PROB)
    assert parser is not None
    assert fixing_parser is not None


def test_get_template_for_mode():
    """Test _get_template_for_mode method."""
    from RAG_supporters.agents.domain_assesment import (
        DomainAnalysisAgent,
        OperationMode,
    )

    mock_llm = create_mock_llm()
    agent = DomainAnalysisAgent(llm=mock_llm)

    # Test each mode
    template = agent._get_template_for_mode(OperationMode.EXTRACT)
    assert template is not None

    template = agent._get_template_for_mode(OperationMode.GUESS)
    assert template is not None

    template = agent._get_template_for_mode(OperationMode.ASSESS)
    assert template is not None

    template = agent._get_template_for_mode(OperationMode.TOPIC_RELEVANCE_PROB)
    assert template is not None

    # Test invalid mode
    with pytest.raises(ValueError, match="Unknown operation mode"):
        agent._get_template_for_mode("invalid_mode")


def test_include_reason_parameter():
    """Test that include_reason parameter is stored correctly."""
    from RAG_supporters.agents.domain_assesment import DomainAnalysisAgent

    mock_llm = create_mock_llm()

    # Test with include_reason=True
    agent_with_reason = DomainAnalysisAgent(llm=mock_llm, include_reason=True)
    assert agent_with_reason.include_reason is True

    # Test with include_reason=False (default)
    agent_without_reason = DomainAnalysisAgent(llm=mock_llm, include_reason=False)
    assert agent_without_reason.include_reason is False


def test_max_retries_parameter():
    """Test that max_retries parameter is stored correctly."""
    from RAG_supporters.agents.domain_assesment import DomainAnalysisAgent

    mock_llm = create_mock_llm()

    agent = DomainAnalysisAgent(llm=mock_llm, max_retries=5)
    assert agent.max_retries == 5


def test_batch_size_parameter():
    """Test that batch_size parameter is stored correctly."""
    from RAG_supporters.agents.domain_assesment import DomainAnalysisAgent

    mock_llm = create_mock_llm()

    agent = DomainAnalysisAgent(llm=mock_llm, batch_size=20)
    assert agent.batch_size == 20


def test_agent_state_initialization():
    """Test AgentState initialization."""
    from RAG_supporters.agents.domain_assesment import AgentState, OperationMode

    state = AgentState(
        mode=OperationMode.EXTRACT,
        text_source="Test text",
        max_retries=3,
    )

    assert state.mode == OperationMode.EXTRACT
    assert state.text_source == "Test text"
    assert state.max_retries == 3
    assert state.retry_count == 0
    assert state.result is None
    assert state.error is None


def test_parse_topic_descriptors_with_duplicate_descriptors():
    """Test that duplicate topic descriptors are deduplicated."""
    from RAG_supporters.agents.domain_assesment import DomainAnalysisAgent

    mock_llm = create_mock_llm()
    agent = DomainAnalysisAgent(llm=mock_llm)

    # KeywordClusterer data with duplicate descriptors across clusters
    clusterer_data = {
        "cluster_stats": {
            "0": {
                "topic_descriptors": ["machine learning", "AI", "common_term"],
                "size": 10,
            },
            "1": {
                "topic_descriptors": ["databases", "common_term", "SQL"],
                "size": 8,
            },
        }
    }

    result = agent._parse_topic_descriptors(clusterer_data)

    # Should deduplicate to unique descriptors only
    assert "common_term" in result, "common_term should be in the result"
    # Should have only one occurrence (deduplication)
    assert (
        result.count("common_term") == 1
    ), "Duplicates should be removed, expecting unique descriptors only"
    # Verify all expected unique descriptors are present
    expected_descriptors = {"machine learning", "AI", "common_term", "databases", "SQL"}
    assert set(result) == expected_descriptors, "All unique descriptors should be present"


# Tests for GROUP_TOPIC_RELEVANCE_PROB mode


def test_group_topic_relevance_pydantic_models():
    """Test GroupTopicRelevance and GroupTopicRelevanceResult Pydantic models."""
    from RAG_supporters.agents.domain_assesment import (
        GroupTopicRelevance,
        GroupTopicRelevanceResult,
    )

    # Test GroupTopicRelevance
    group = GroupTopicRelevance(
        cluster_id=0,
        descriptors=["machine learning", "AI"],
        probability=0.85,
        reason="Test reason",
    )
    assert group.cluster_id == 0, "Cluster ID should be 0"
    assert len(group.descriptors) == 2, "Should have 2 descriptors"
    assert group.probability == 0.85, "Probability should be 0.85"
    assert group.reason == "Test reason", "Reason should be set"

    # Test with string cluster_id
    group_str = GroupTopicRelevance(
        cluster_id="cluster_a",
        descriptors=["web dev"],
        probability=0.5,
    )
    assert group_str.cluster_id == "cluster_a", "String cluster IDs should be allowed"

    # Test GroupTopicRelevanceResult
    groups = [
        GroupTopicRelevance(cluster_id=0, descriptors=["ml", "ai"], probability=0.9),
        GroupTopicRelevance(cluster_id=1, descriptors=["db", "sql"], probability=0.3),
    ]

    result = GroupTopicRelevanceResult(
        question_text="What is gradient descent?",
        group_probs=groups,
        total_groups=2,
        question_summary="ML question",
    )

    assert result.question_text == "What is gradient descent?", "Question text should match"
    assert len(result.group_probs) == 2, "Should have 2 groups"
    assert result.total_groups == 2, "Total groups should be 2"
    assert result.question_summary == "ML question", "Summary should match"


def test_group_topic_relevance_result_validation():
    """Test GroupTopicRelevanceResult auto-correction."""
    from RAG_supporters.agents.domain_assesment import (
        GroupTopicRelevance,
        GroupTopicRelevanceResult,
    )

    groups = [
        GroupTopicRelevance(cluster_id=0, descriptors=["ml"], probability=0.8),
        GroupTopicRelevance(cluster_id=1, descriptors=["db"], probability=0.2),
    ]

    # Mismatched total_groups should be corrected
    result = GroupTopicRelevanceResult(
        question_text="Test question",
        group_probs=groups,
        total_groups=10,  # Wrong count
    )

    # Should be auto-corrected to 2
    assert result.total_groups == 2, "total_groups should be auto-corrected to actual count"


def test_operation_mode_enum_includes_group_topic_relevance_prob():
    """Test that OperationMode enum includes GROUP_TOPIC_RELEVANCE_PROB."""
    from RAG_supporters.agents.domain_assesment import OperationMode

    assert hasattr(OperationMode, "GROUP_TOPIC_RELEVANCE_PROB"), "GROUP_TOPIC_RELEVANCE_PROB should exist"
    assert (
        OperationMode.GROUP_TOPIC_RELEVANCE_PROB.value == "group_topic_relevance_prob"
    ), "Value should be 'group_topic_relevance_prob'"


def test_prepare_cluster_data_from_dict():
    """Test _prepare_cluster_data with KeywordClusterer dict."""
    from RAG_supporters.agents.domain_assesment import DomainAnalysisAgent

    mock_llm = create_mock_llm()
    agent = DomainAnalysisAgent(llm=mock_llm)

    cluster_data = {
        "cluster_stats": {
            "0": {"topic_descriptors": ["ml", "ai"], "size": 10},
            "1": {"topic_descriptors": ["db", "sql"], "size": 8},
            "2": {"topic_descriptors": ["web"], "size": 5},
        }
    }

    result = agent._prepare_cluster_data(cluster_data)

    assert isinstance(result, list), "Result should be a list"
    assert len(result) == 3, "Should have 3 clusters"
    assert all("cluster_id" in cluster for cluster in result), "Each cluster should have cluster_id"
    assert all("descriptors" in cluster for cluster in result), "Each cluster should have descriptors"

    # Check cluster IDs are integers
    cluster_ids = [cluster["cluster_id"] for cluster in result]
    assert all(isinstance(cid, int) for cid in cluster_ids), "Cluster IDs should be integers"

    # Check descriptors
    assert result[0]["descriptors"] == ["ml", "ai"], "First cluster descriptors should match"
    assert result[1]["descriptors"] == ["db", "sql"], "Second cluster descriptors should match"


def test_prepare_cluster_data_from_file():
    """Test _prepare_cluster_data with file path."""
    from RAG_supporters.agents.domain_assesment import DomainAnalysisAgent

    mock_llm = create_mock_llm()
    agent = DomainAnalysisAgent(llm=mock_llm)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create cluster data file
        cluster_file = tmpdir / "clusters.json"
        cluster_data = {
            "cluster_stats": {
                "0": {"topic_descriptors": ["topic1", "topic2"], "size": 10},
                "1": {"topic_descriptors": ["topic3"], "size": 5},
            }
        }

        with open(cluster_file, "w", encoding="utf-8") as f:
            json.dump(cluster_data, f)

        # Parse from file path
        result = agent._prepare_cluster_data(str(cluster_file))

        assert len(result) == 2, "Should have 2 clusters"
        assert result[0]["descriptors"] == ["topic1", "topic2"], "First cluster descriptors should match"


def test_prepare_cluster_data_missing_cluster_stats():
    """Test _prepare_cluster_data error when cluster_stats is missing."""
    from RAG_supporters.agents.domain_assesment import DomainAnalysisAgent

    mock_llm = create_mock_llm()
    agent = DomainAnalysisAgent(llm=mock_llm)

    # Dict without cluster_stats
    invalid_data = {"some_key": "some_value"}

    with pytest.raises(ValueError, match="cluster_data must contain 'cluster_stats'"):
        agent._prepare_cluster_data(invalid_data)


def test_prepare_cluster_data_invalid_cluster():
    """Test _prepare_cluster_data skips invalid clusters."""
    from RAG_supporters.agents.domain_assesment import DomainAnalysisAgent

    mock_llm = create_mock_llm()
    agent = DomainAnalysisAgent(llm=mock_llm)

    cluster_data = {
        "cluster_stats": {
            "0": {"topic_descriptors": ["valid1"], "size": 10},
            "1": {},  # Invalid - no topic_descriptors
            "2": {"topic_descriptors": "not_a_list", "size": 5},  # Invalid - not a list
            "3": {"topic_descriptors": ["valid2"], "size": 3},
        }
    }

    result = agent._prepare_cluster_data(cluster_data)

    # Should only include valid clusters (0 and 3)
    assert len(result) == 2, "Should only include valid clusters with topic_descriptors as lists"
    assert result[0]["descriptors"] == ["valid1"], "Should include cluster 0"
    assert result[1]["descriptors"] == ["valid2"], "Should include cluster 3"


def test_assess_group_topic_relevance_prob_success():
    """Test assess_group_topic_relevance_prob with successful result."""
    from RAG_supporters.agents.domain_assesment import DomainAnalysisAgent

    mock_llm = create_mock_llm()
    agent = DomainAnalysisAgent(llm=mock_llm)

    # Mock successful result
    mock_result = {
        "result": {
            "question_text": "What is gradient descent?",
            "group_probs": [
                {"cluster_id": 0, "descriptors": ["ml", "ai"], "probability": 0.9},
                {"cluster_id": 1, "descriptors": ["db", "sql"], "probability": 0.1},
            ],
            "total_groups": 2,
            "question_summary": "ML question",
        },
        "error": None,
    }

    mock_graph = MagicMock()
    mock_graph.invoke = MagicMock(return_value=mock_result)
    agent.graph = mock_graph

    cluster_data = {
        "cluster_stats": {
            "0": {"topic_descriptors": ["ml", "ai"], "size": 10},
            "1": {"topic_descriptors": ["db", "sql"], "size": 8},
        }
    }

    result = agent.assess_group_topic_relevance_prob("What is gradient descent?", cluster_data)

    assert result is not None, "Result should not be None for successful assessment"
    assert "group_probs" in result, "Result should contain group_probs"
    assert "total_groups" in result, "Result should contain total_groups"
    assert "question_text" in result, "Result should contain question_text"
    assert result["total_groups"] == 2, "Should have 2 groups"


def test_assess_group_topic_relevance_prob_failure():
    """Test assess_group_topic_relevance_prob with failure."""
    from RAG_supporters.agents.domain_assesment import DomainAnalysisAgent

    mock_llm = create_mock_llm()
    agent = DomainAnalysisAgent(llm=mock_llm)

    # Mock failure result
    mock_result = {
        "result": None,
        "error": "Parsing failed",
    }

    mock_graph = MagicMock()
    mock_graph.invoke = MagicMock(return_value=mock_result)
    agent.graph = mock_graph

    cluster_data = {
        "cluster_stats": {
            "0": {"topic_descriptors": ["ml"], "size": 10},
        }
    }

    result = agent.assess_group_topic_relevance_prob("Test question", cluster_data)

    assert result is None, "Result should be None when assessment fails"


def test_assess_group_topic_relevance_prob_batch_sequential():
    """Test assess_group_topic_relevance_prob_batch with sequential processing."""
    from RAG_supporters.agents.domain_assesment import DomainAnalysisAgent

    mock_llm = create_mock_llm()
    agent = DomainAnalysisAgent(llm=mock_llm)

    # Force sequential processing
    agent._is_openai_llm = False

    mock_result = {
        "question_text": "Test question",
        "group_probs": [
            {"cluster_id": 0, "descriptors": ["ml"], "probability": 0.9},
        ],
        "total_groups": 1,
    }

    with patch.object(agent, "assess_group_topic_relevance_prob", return_value=mock_result):
        questions = ["Question 1?", "Question 2?"]
        cluster_data = {
            "cluster_stats": {
                "0": {"topic_descriptors": ["ml"], "size": 10},
            }
        }
        results = agent.assess_group_topic_relevance_prob_batch(
            questions, cluster_data, show_progress=False
        )

        assert len(results) == 2, "Should return results for both questions"
        assert all(r == mock_result for r in results), "All results should match mock"


def test_get_column_prefix_group_topic_relevance_prob():
    """Test _get_column_prefix for GROUP_TOPIC_RELEVANCE_PROB mode."""
    from RAG_supporters.agents.domain_assesment import (
        DomainAnalysisAgent,
        OperationMode,
    )

    mock_llm = create_mock_llm()
    agent = DomainAnalysisAgent(llm=mock_llm)

    assert (
        agent._get_column_prefix(OperationMode.GROUP_TOPIC_RELEVANCE_PROB)
        == "group_topic_relevance_prob"
    ), "Column prefix should match mode value"


def test_get_parser_for_mode_group_topic_relevance_prob():
    """Test _get_parser_for_mode for GROUP_TOPIC_RELEVANCE_PROB mode."""
    from RAG_supporters.agents.domain_assesment import (
        DomainAnalysisAgent,
        OperationMode,
    )

    mock_llm = create_mock_llm()
    agent = DomainAnalysisAgent(llm=mock_llm)

    parser, fixing_parser = agent._get_parser_for_mode(OperationMode.GROUP_TOPIC_RELEVANCE_PROB)

    assert parser is not None, "Parser should not be None"
    assert fixing_parser is not None, "Fixing parser should not be None"
    assert parser == agent.group_topic_relevance_prob_parser, "Should return correct parser"


def test_get_template_for_mode_group_topic_relevance_prob():
    """Test _get_template_for_mode for GROUP_TOPIC_RELEVANCE_PROB mode."""
    from RAG_supporters.agents.domain_assesment import (
        DomainAnalysisAgent,
        OperationMode,
    )

    mock_llm = create_mock_llm()
    agent = DomainAnalysisAgent(llm=mock_llm)

    template = agent._get_template_for_mode(OperationMode.GROUP_TOPIC_RELEVANCE_PROB)

    assert template is not None, "Template should not be None"
    assert (
        template == agent.group_topic_relevance_prob_template
    ), "Should return correct template"


def test_agent_initialization_includes_group_parsers():
    """Test that agent initialization includes GROUP_TOPIC_RELEVANCE_PROB parsers and templates."""
    from RAG_supporters.agents.domain_assesment import DomainAnalysisAgent

    mock_llm = create_mock_llm()
    agent = DomainAnalysisAgent(llm=mock_llm)

    assert hasattr(agent, "group_topic_relevance_prob_parser"), "Should have group parser"
    assert hasattr(
        agent, "group_topic_relevance_prob_fixing_parser"
    ), "Should have group fixing parser"
    assert hasattr(
        agent, "group_topic_relevance_prob_template"
    ), "Should have group template"
    assert agent.group_topic_relevance_prob_parser is not None, "Parser should be initialized"
    assert (
        agent.group_topic_relevance_prob_fixing_parser is not None
    ), "Fixing parser should be initialized"
    assert agent.group_topic_relevance_prob_template is not None, "Template should be initialized"


def test_prepare_cluster_data_preserves_cluster_id_types():
    """Test that _prepare_cluster_data preserves or converts cluster IDs appropriately."""
    from RAG_supporters.agents.domain_assesment import DomainAnalysisAgent

    mock_llm = create_mock_llm()
    agent = DomainAnalysisAgent(llm=mock_llm)

    cluster_data = {
        "cluster_stats": {
            "0": {"topic_descriptors": ["ml"], "size": 10},
            "cluster_a": {"topic_descriptors": ["db"], "size": 5},
        }
    }

    result = agent._prepare_cluster_data(cluster_data)

    assert len(result) == 2, "Should have 2 clusters"

    # Check that numeric strings are converted to ints
    cluster_ids = [cluster["cluster_id"] for cluster in result]
    assert 0 in cluster_ids, "Numeric string '0' should be converted to int 0"
    assert "cluster_a" in cluster_ids, "Non-numeric string should remain as string"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
