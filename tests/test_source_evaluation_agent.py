"""Tests for SourceEvaluationAgent."""

import logging
from unittest.mock import MagicMock, Mock, patch

import pytest

# Skip all tests if required dependencies are not installed
pytest.importorskip("langchain")
pytest.importorskip("langchain_core")
pytest.importorskip("langgraph")
pytest.importorskip("pydantic")
pytest.importorskip("tqdm")
pytest.importorskip("pandas")

import pandas as pd


def test_agent_import():
    """Test that SourceEvaluationAgent can be imported."""
    from RAG_supporters.agents.source_assesment import SourceEvaluationAgent
    
    assert SourceEvaluationAgent is not None


def test_pydantic_models_import():
    """Test that Pydantic models can be imported."""
    from RAG_supporters.agents.source_assesment import (
        AgentState,
        ScoreRange,
        SourceEvaluation,
    )
    
    # Test ScoreRange
    score_range = ScoreRange(score=8, reasoning="Good quality")
    assert score_range.score == 8
    assert score_range.reasoning == "Good quality"
    
    # Test score validation
    with pytest.raises(Exception):  # Should fail validation
        ScoreRange(score=15)  # Score > 10
    
    with pytest.raises(Exception):  # Should fail validation
        ScoreRange(score=-1)  # Score < 0


class TestSourceEvaluationAgentInit:
    """Test SourceEvaluationAgent initialization."""

    def test_init_with_llm(self):
        """Test initialization with a language model."""
        from RAG_supporters.agents.source_assesment import SourceEvaluationAgent
        from langchain_core.language_models import BaseChatModel
        
        mock_llm = Mock(spec=BaseChatModel)
        agent = SourceEvaluationAgent(llm=mock_llm)
        
        assert agent._llm == mock_llm
        assert agent._max_retries == 3

    def test_init_with_custom_retries(self):
        """Test initialization with custom max_retries."""
        from RAG_supporters.agents.source_assesment import SourceEvaluationAgent
        from langchain_core.language_models import BaseChatModel
        
        mock_llm = Mock(spec=BaseChatModel)
        agent = SourceEvaluationAgent(llm=mock_llm, max_retries=5)
        
        assert agent._max_retries == 5


class TestSourceEvaluationAgentEvaluateSource:
    """Test evaluate_source method."""

    def test_evaluate_source_basic(self):
        """Test basic source evaluation."""
        from RAG_supporters.agents.source_assesment import SourceEvaluationAgent
        from langchain_core.language_models import BaseChatModel
        
        mock_llm = Mock(spec=BaseChatModel)
        
        # Create a mock response that will be parsed correctly
        mock_response = Mock()
        mock_response.content = """{
            "inferred_domain": "biology",
            "relevance": {"score": 9, "reasoning": "Highly relevant"},
            "expertise_authority": {"score": 8, "reasoning": "Good authority"},
            "depth_specificity": {"score": 7, "reasoning": "Sufficient depth"},
            "clarity_conciseness": {"score": 9, "reasoning": "Very clear"},
            "objectivity_bias": {"score": 8, "reasoning": "Mostly objective"},
            "completeness": {"score": 7, "reasoning": "Reasonably complete"}
        }"""
        
        mock_llm.invoke = Mock(return_value=mock_response)
        
        agent = SourceEvaluationAgent(llm=mock_llm)
        
        result = agent.evaluate_source(
            question="What is photosynthesis?",
            source_content="Photosynthesis is the process by which plants convert light into energy."
        )
        
        assert result is not None
        assert 'evaluation' in result
        assert result['error'] is None

    def test_evaluate_source_with_error(self):
        """Test source evaluation with LLM error."""
        from RAG_supporters.agents.source_assesment import SourceEvaluationAgent
        from langchain_core.language_models import BaseChatModel
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_llm.invoke = Mock(side_effect=Exception("LLM error"))
        
        agent = SourceEvaluationAgent(llm=mock_llm)
        
        result = agent.evaluate_source(
            question="What is AI?",
            source_content="AI is artificial intelligence."
        )
        
        # Should handle error gracefully
        assert result is not None
        assert 'error' in result


class TestSourceEvaluationAgentProcessDataFrame:
    """Test process_dataframe method."""

    def test_process_dataframe_basic(self):
        """Test basic dataframe processing."""
        from RAG_supporters.agents.source_assesment import SourceEvaluationAgent
        from langchain_core.language_models import BaseChatModel
        
        mock_llm = Mock(spec=BaseChatModel)
        
        # Create mock response
        mock_response = Mock()
        mock_response.content = """{
            "inferred_domain": "science",
            "relevance": {"score": 8, "reasoning": "Relevant"},
            "expertise_authority": {"score": 7, "reasoning": "Good"},
            "depth_specificity": {"score": 8, "reasoning": "Detailed"},
            "clarity_conciseness": {"score": 9, "reasoning": "Clear"},
            "objectivity_bias": {"score": 8, "reasoning": "Objective"},
            "completeness": {"score": 7, "reasoning": "Complete"}
        }"""
        
        mock_llm.invoke = Mock(return_value=mock_response)
        
        agent = SourceEvaluationAgent(llm=mock_llm)
        
        df = pd.DataFrame({
            'question_text': ['Q1?', 'Q2?'],
            'source_text': ['S1', 'S2']
        })
        
        result_df = agent.process_dataframe(
            df,
            question_col='question_text',
            source_col='source_text'
        )
        
        assert len(result_df) == 2
        # Check that evaluation columns are added
        assert 'inferred_domain' in result_df.columns or 'evaluation_error' in result_df.columns

    def test_process_dataframe_with_include_reasoning(self):
        """Test dataframe processing with reasoning included."""
        from RAG_supporters.agents.source_assesment import SourceEvaluationAgent
        from langchain_core.language_models import BaseChatModel
        
        mock_llm = Mock(spec=BaseChatModel)
        
        mock_response = Mock()
        mock_response.content = """{
            "inferred_domain": "biology",
            "relevance": {"score": 9, "reasoning": "Very relevant to question"},
            "expertise_authority": {"score": 8, "reasoning": "Expert source"},
            "depth_specificity": {"score": 7, "reasoning": "Good depth"},
            "clarity_conciseness": {"score": 9, "reasoning": "Very clear"},
            "objectivity_bias": {"score": 8, "reasoning": "Unbiased"},
            "completeness": {"score": 7, "reasoning": "Mostly complete"}
        }"""
        
        mock_llm.invoke = Mock(return_value=mock_response)
        
        agent = SourceEvaluationAgent(llm=mock_llm)
        
        df = pd.DataFrame({
            'question_text': ['What is photosynthesis?'],
            'source_text': ['Photosynthesis is...']
        })
        
        result_df = agent.process_dataframe(
            df,
            question_col='question_text',
            source_col='source_text',
            include_reasoning=True
        )
        
        assert len(result_df) == 1
        # Should have reasoning columns
        # (exact column names depend on implementation)

    def test_process_dataframe_skip_existing(self):
        """Test that already evaluated rows are skipped."""
        from RAG_supporters.agents.source_assesment import SourceEvaluationAgent
        from langchain_core.language_models import BaseChatModel
        
        mock_llm = Mock(spec=BaseChatModel)
        
        mock_response = Mock()
        mock_response.content = """{
            "inferred_domain": "biology",
            "relevance": {"score": 9, "reasoning": "Relevant"},
            "expertise_authority": {"score": 8, "reasoning": "Good"},
            "depth_specificity": {"score": 7, "reasoning": "Sufficient"},
            "clarity_conciseness": {"score": 9, "reasoning": "Clear"},
            "objectivity_bias": {"score": 8, "reasoning": "Objective"},
            "completeness": {"score": 7, "reasoning": "Complete"}
        }"""
        
        mock_llm.invoke = Mock(return_value=mock_response)
        
        agent = SourceEvaluationAgent(llm=mock_llm)
        
        df = pd.DataFrame({
            'question_text': ['Q1?', 'Q2?'],
            'source_text': ['S1', 'S2'],
            'relevance_score': [8.0, None]  # First row already evaluated
        })
        
        result_df = agent.process_dataframe(
            df,
            question_col='question_text',
            source_col='source_text',
            skip_existing=True
        )
        
        assert len(result_df) == 2
        # First row score should be unchanged
        # Second row should have new evaluation


class TestSourceEvaluationAgentScoreExtraction:
    """Test score extraction from evaluation results."""

    def test_score_extraction_with_valid_evaluation(self):
        """Test extracting scores from valid evaluation."""
        from RAG_supporters.agents.source_assesment import (
            ScoreRange,
            SourceEvaluation,
        )
        
        evaluation = SourceEvaluation(
            inferred_domain="biology",
            relevance=ScoreRange(score=9, reasoning="Very relevant"),
            expertise_authority=ScoreRange(score=8, reasoning="Good authority"),
            depth_specificity=ScoreRange(score=7, reasoning="Good depth"),
            clarity_conciseness=ScoreRange(score=9, reasoning="Clear"),
            objectivity_bias=ScoreRange(score=8, reasoning="Objective"),
            completeness=ScoreRange(score=7, reasoning="Complete")
        )
        
        assert evaluation.relevance.score == 9
        assert evaluation.expertise_authority.score == 8
        assert evaluation.depth_specificity.score == 7
        assert evaluation.clarity_conciseness.score == 9
        assert evaluation.objectivity_bias.score == 8
        assert evaluation.completeness.score == 7

    def test_score_validation(self):
        """Test that invalid scores are rejected."""
        from RAG_supporters.agents.source_assesment import ScoreRange
        
        # Valid score
        valid_score = ScoreRange(score=5, reasoning="Average")
        assert valid_score.score == 5
        
        # Invalid scores
        with pytest.raises(Exception):
            ScoreRange(score=11)  # Too high
        
        with pytest.raises(Exception):
            ScoreRange(score=-1)  # Too low


class TestSourceEvaluationAgentIntegration:
    """Integration tests for SourceEvaluationAgent."""

    @pytest.mark.skip(reason="Requires actual LLM API access")
    def test_real_llm_integration(self):
        """Test with real LLM (requires API key)."""
        from langchain_openai import ChatOpenAI
        from RAG_supporters.agents.source_assesment import SourceEvaluationAgent
        
        llm = ChatOpenAI(model="gpt-4", temperature=0.0)
        agent = SourceEvaluationAgent(llm=llm)
        
        result = agent.evaluate_source(
            question="What is photosynthesis?",
            source_content="Photosynthesis is the process by which plants convert light energy into chemical energy using chlorophyll."
        )
        
        assert result is not None
        assert result['error'] is None
        assert result['evaluation'] is not None
        
        # Check that all scores are in valid range
        eval_data = result['evaluation']
        assert 0 <= eval_data.relevance.score <= 10
        assert 0 <= eval_data.expertise_authority.score <= 10
        assert 0 <= eval_data.depth_specificity.score <= 10
        assert 0 <= eval_data.clarity_conciseness.score <= 10
        assert 0 <= eval_data.objectivity_bias.score <= 10
        assert 0 <= eval_data.completeness.score <= 10
