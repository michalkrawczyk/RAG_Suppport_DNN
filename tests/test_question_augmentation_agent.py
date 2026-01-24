"""Tests for QuestionAugmentationAgent."""

import logging
from unittest.mock import MagicMock, Mock, patch

import pytest

# Skip all tests if required dependencies are not installed
pytest.importorskip("langchain")
pytest.importorskip("langchain_core")
pytest.importorskip("tqdm")
pytest.importorskip("pandas")

import pandas as pd


def test_agent_import():
    """Test that QuestionAugmentationAgent can be imported."""
    from RAG_supporters.agents.question_augmentation_agent import QuestionAugmentationAgent
    
    assert QuestionAugmentationAgent is not None


class TestQuestionAugmentationAgentInit:
    """Test QuestionAugmentationAgent initialization."""

    def test_init_with_llm(self):
        """Test initialization with a language model."""
        from RAG_supporters.agents.question_augmentation_agent import QuestionAugmentationAgent
        from langchain_core.language_models import BaseChatModel
        
        mock_llm = Mock(spec=BaseChatModel)
        agent = QuestionAugmentationAgent(llm=mock_llm)
        
        assert agent._llm == mock_llm
        assert agent._max_retries == 3
        assert agent.batch_size == 10

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        from RAG_supporters.agents.question_augmentation_agent import QuestionAugmentationAgent
        from langchain_core.language_models import BaseChatModel
        
        mock_llm = Mock(spec=BaseChatModel)
        agent = QuestionAugmentationAgent(
            llm=mock_llm,
            max_retries=5,
            batch_size=20
        )
        
        assert agent._max_retries == 5
        assert agent.batch_size == 20

    def test_init_none_llm_raises_error(self):
        """Test that None LLM raises ValueError."""
        from RAG_supporters.agents.question_augmentation_agent import QuestionAugmentationAgent
        
        with pytest.raises(ValueError, match="llm parameter cannot be None"):
            QuestionAugmentationAgent(llm=None)

    def test_init_invalid_llm_type_raises_error(self):
        """Test that invalid LLM type raises TypeError."""
        from RAG_supporters.agents.question_augmentation_agent import QuestionAugmentationAgent
        
        with pytest.raises(TypeError, match="llm must be a BaseChatModel instance"):
            QuestionAugmentationAgent(llm="not_an_llm")


class TestQuestionAugmentationAgentRephraseWithSource:
    """Test rephrase_question_with_source method."""

    def test_rephrase_question_with_source_basic(self):
        """Test basic question rephrasing with source."""
        from RAG_supporters.agents.question_augmentation_agent import QuestionAugmentationAgent
        from langchain_core.language_models import BaseChatModel
        from langchain_core.messages import AIMessage
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_llm.invoke = Mock(return_value=AIMessage(
            content="What process do plants use to convert light energy into chemical energy?"
        ))
        
        agent = QuestionAugmentationAgent(llm=mock_llm)
        
        result = agent.rephrase_question_with_source(
            question="What does it do?",
            source="Photosynthesis is the process by which plants convert light energy into chemical energy."
        )
        
        assert result is not None
        assert "photosynthesis" in result.lower() or "light energy" in result.lower()

    def test_rephrase_question_with_source_returns_none_on_error(self):
        """Test that None is returned when LLM fails."""
        from RAG_supporters.agents.question_augmentation_agent import QuestionAugmentationAgent
        from langchain_core.language_models import BaseChatModel
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_llm.invoke = Mock(side_effect=Exception("LLM error"))
        
        agent = QuestionAugmentationAgent(llm=mock_llm)
        
        result = agent.rephrase_question_with_source(
            question="What is this?",
            source="Some source text"
        )
        
        assert result is None


class TestQuestionAugmentationAgentRephraseWithDomain:
    """Test rephrase_question_with_domain method."""

    def test_rephrase_question_with_domain_basic(self):
        """Test basic question rephrasing with domain."""
        from RAG_supporters.agents.question_augmentation_agent import QuestionAugmentationAgent
        from langchain_core.language_models import BaseChatModel
        from langchain_core.messages import AIMessage
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_llm.invoke = Mock(return_value=AIMessage(
            content="What is the mechanism of cellular respiration in eukaryotic cells?"
        ))
        
        agent = QuestionAugmentationAgent(llm=mock_llm)
        
        result = agent.rephrase_question_with_domain(
            question="How does it work?",
            domain="molecular biology"
        )
        
        assert result is not None
        assert isinstance(result, str)

    def test_rephrase_question_with_domain_returns_none_on_error(self):
        """Test that None is returned when LLM fails."""
        from RAG_supporters.agents.question_augmentation_agent import QuestionAugmentationAgent
        from langchain_core.language_models import BaseChatModel
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_llm.invoke = Mock(side_effect=Exception("LLM error"))
        
        agent = QuestionAugmentationAgent(llm=mock_llm)
        
        result = agent.rephrase_question_with_domain(
            question="What is this?",
            domain="physics"
        )
        
        assert result is None


class TestQuestionAugmentationAgentGenerateAlternatives:
    """Test generate_alternative_questions method."""

    def test_generate_alternative_questions_basic(self):
        """Test basic alternative question generation."""
        from RAG_supporters.agents.question_augmentation_agent import QuestionAugmentationAgent
        from langchain_core.language_models import BaseChatModel
        from langchain_core.messages import AIMessage
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_llm.invoke = Mock(return_value=AIMessage(
            content='{"questions": ["Question 1?", "Question 2?", "Question 3?"]}'
        ))
        
        agent = QuestionAugmentationAgent(llm=mock_llm)
        
        result = agent.generate_alternative_questions(
            source="Photosynthesis converts light into energy.",
            n=3
        )
        
        assert result is not None
        assert isinstance(result, list)
        assert len(result) <= 3

    def test_generate_alternative_questions_with_allow_vague(self):
        """Test alternative question generation with allow_vague parameter."""
        from RAG_supporters.agents.question_augmentation_agent import QuestionAugmentationAgent
        from langchain_core.language_models import BaseChatModel
        from langchain_core.messages import AIMessage
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_llm.invoke = Mock(return_value=AIMessage(
            content='{"questions": ["What is photosynthesis?", "How do plants produce energy?"]}'
        ))
        
        agent = QuestionAugmentationAgent(llm=mock_llm)
        
        result = agent.generate_alternative_questions(
            source="Photosynthesis is important.",
            n=2,
            allow_vague=True
        )
        
        assert result is not None
        assert isinstance(result, list)

    def test_generate_alternative_questions_returns_none_on_error(self):
        """Test that None is returned when LLM fails."""
        from RAG_supporters.agents.question_augmentation_agent import QuestionAugmentationAgent
        from langchain_core.language_models import BaseChatModel
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_llm.invoke = Mock(side_effect=Exception("LLM error"))
        
        agent = QuestionAugmentationAgent(llm=mock_llm)
        
        result = agent.generate_alternative_questions(
            source="Some source",
            n=3
        )
        
        assert result is None


class TestQuestionAugmentationAgentProcessDataFrame:
    """Test process_dataframe_rephrasing method."""

    def test_process_dataframe_rephrasing_basic(self):
        """Test basic dataframe processing for rephrasing."""
        from RAG_supporters.agents.question_augmentation_agent import QuestionAugmentationAgent
        from langchain_core.language_models import BaseChatModel
        from langchain_core.messages import AIMessage
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_llm.invoke = Mock(return_value=AIMessage(
            content="Rephrased question"
        ))
        
        agent = QuestionAugmentationAgent(llm=mock_llm)
        
        df = pd.DataFrame({
            'question_text': ['What is this?', 'How does it work?'],
            'source_text': ['Source 1', 'Source 2']
        })
        
        # Uses default column names: question_text, source_text, rephrased_question
        result_df = agent.process_dataframe_rephrasing(
            df,
            columns_mapping={
                'question_text': 'question_text',
                'source_text': 'source_text',
                'rephrased_question': 'rephrased_question'
            }
        )
        
        assert 'rephrased_question' in result_df.columns
        assert len(result_df) == 2


class TestQuestionAugmentationAgentProcessDataFrameGeneration:
    """Test process_dataframe_generation method."""

    def test_process_dataframe_generation_basic(self):
        """Test basic dataframe processing for question generation."""
        from RAG_supporters.agents.question_augmentation_agent import QuestionAugmentationAgent
        from langchain_core.language_models import BaseChatModel
        from langchain_core.messages import AIMessage
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_llm.invoke = Mock(return_value=AIMessage(
            content='{"questions": ["Question 1?", "Question 2?"]}'
        ))
        
        agent = QuestionAugmentationAgent(llm=mock_llm)
        
        df = pd.DataFrame({
            'source_text': ['Source 1', 'Source 2']
        })
        
        # Uses default column name 'source_text'
        result_df = agent.process_dataframe_generation(
            df,
            n_questions=2
        )
        
        assert 'question_text' in result_df.columns
        assert 'source_text' in result_df.columns
        assert len(result_df) >= 2  # At least 2 questions generated

    def test_process_dataframe_generation_with_custom_columns(self):
        """Test question generation with custom column names."""
        from RAG_supporters.agents.question_augmentation_agent import QuestionAugmentationAgent
        from langchain_core.language_models import BaseChatModel
        from langchain_core.messages import AIMessage
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_llm.invoke = Mock(return_value=AIMessage(
            content='{"questions": ["Custom column question?"]}'
        ))
        
        agent = QuestionAugmentationAgent(llm=mock_llm)
        
        df = pd.DataFrame({
            'my_source': ['Source about biology']
        })
        
        # Uses columns_mapping to specify custom column name
        result_df = agent.process_dataframe_generation(
            df,
            columns_mapping={'source_text': 'my_source'},
            n_questions=1
        )
        
        assert len(result_df) >= 1
        assert 'my_source' in result_df.columns


class TestQuestionAugmentationAgentIntegration:
    """Integration tests for QuestionAugmentationAgent."""

    @pytest.mark.skip(reason="Requires actual LLM API access")
    def test_real_llm_integration(self):
        """Test with real LLM (requires API key)."""
        from langchain_openai import ChatOpenAI
        from RAG_supporters.agents.question_augmentation_agent import QuestionAugmentationAgent
        
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        agent = QuestionAugmentationAgent(llm=llm)
        
        result = agent.rephrase_question_with_source(
            question="What is it?",
            source="Photosynthesis is the process plants use to convert sunlight into energy."
        )
        
        assert result is not None
        assert len(result) > 0
