"""Tests for TextAugmentationAgent."""

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
    """Test that TextAugmentationAgent can be imported."""
    from RAG_supporters.agents.text_augmentation import TextAugmentationAgent
    
    assert TextAugmentationAgent is not None


class TestTextAugmentationAgentInit:
    """Test TextAugmentationAgent initialization."""

    def test_init_with_llm(self):
        """Test initialization with a language model."""
        from RAG_supporters.agents.text_augmentation import TextAugmentationAgent
        from langchain_core.language_models import BaseChatModel
        
        mock_llm = Mock(spec=BaseChatModel)
        agent = TextAugmentationAgent(llm=mock_llm)
        
        assert agent._llm == mock_llm
        assert agent._verify_meaning is False
        assert agent._max_retries == 3
        assert agent.batch_size == 10

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        from RAG_supporters.agents.text_augmentation import TextAugmentationAgent
        from langchain_core.language_models import BaseChatModel
        
        mock_llm = Mock(spec=BaseChatModel)
        agent = TextAugmentationAgent(
            llm=mock_llm,
            verify_meaning=True,
            max_retries=5,
            batch_size=20
        )
        
        assert agent._verify_meaning is True
        assert agent._max_retries == 5
        assert agent.batch_size == 20


class TestTextAugmentationAgentRephraseText:
    """Test rephrase_text method."""

    def test_rephrase_text_full_mode(self):
        """Test text rephrasing in full mode."""
        from RAG_supporters.agents.text_augmentation import TextAugmentationAgent
        from langchain_core.language_models import BaseChatModel
        from langchain_core.messages import AIMessage
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_llm.invoke = Mock(return_value=AIMessage(
            content="Plants utilize photosynthesis to transform sunlight into energy."
        ))
        
        agent = TextAugmentationAgent(llm=mock_llm)
        
        result = agent.rephrase_text(
            text="Plants use photosynthesis to convert sunlight into energy.",
            mode="full"
        )
        
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    def test_rephrase_text_sentence_mode(self):
        """Test text rephrasing in sentence mode."""
        from RAG_supporters.agents.text_augmentation import TextAugmentationAgent
        from langchain_core.language_models import BaseChatModel
        from langchain_core.messages import AIMessage
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_llm.invoke = Mock(return_value=AIMessage(
            content="Plants utilize photosynthesis."
        ))
        
        agent = TextAugmentationAgent(llm=mock_llm)
        
        result = agent.rephrase_text(
            text="Plants use photosynthesis. It converts light to energy.",
            mode="sentence"
        )
        
        assert result is not None
        assert isinstance(result, str)

    def test_rephrase_text_random_mode(self):
        """Test text rephrasing in random mode."""
        from RAG_supporters.agents.text_augmentation import TextAugmentationAgent
        from langchain_core.language_models import BaseChatModel
        from langchain_core.messages import AIMessage
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_llm.invoke = Mock(return_value=AIMessage(
            content="Rephrased text"
        ))
        
        agent = TextAugmentationAgent(llm=mock_llm)
        
        result = agent.rephrase_text(
            text="Original text to rephrase.",
            mode="random"
        )
        
        assert result is not None
        assert isinstance(result, str)

    def test_rephrase_text_returns_none_on_error(self):
        """Test that None is returned when LLM fails."""
        from RAG_supporters.agents.text_augmentation import TextAugmentationAgent
        from langchain_core.language_models import BaseChatModel
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_llm.invoke = Mock(side_effect=Exception("LLM error"))
        
        agent = TextAugmentationAgent(llm=mock_llm)
        
        result = agent.rephrase_text(
            text="Some text",
            mode="full"
        )
        
        assert result is None


class TestTextAugmentationAgentAugmentDataFrame:
    """Test augment_dataframe method."""

    def test_augment_dataframe_questions_only(self):
        """Test augmenting only questions in dataframe."""
        from RAG_supporters.agents.text_augmentation import TextAugmentationAgent
        from langchain_core.language_models import BaseChatModel
        from langchain_core.messages import AIMessage
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_llm.invoke = Mock(return_value=AIMessage(
            content="Rephrased question?"
        ))
        
        agent = TextAugmentationAgent(llm=mock_llm)
        
        df = pd.DataFrame({
            'question_text': ['Q1?', 'Q2?'],
            'source_text': ['S1', 'S2']
        })
        
        result_df = agent.augment_dataframe(
            df,
            rephrase_question=True,
            rephrase_source=False,
            probability=1.0
        )
        
        # Should have more rows due to augmentation
        assert len(result_df) >= len(df)
        assert 'question_text' in result_df.columns
        assert 'source_text' in result_df.columns

    def test_augment_dataframe_sources_only(self):
        """Test augmenting only sources in dataframe."""
        from RAG_supporters.agents.text_augmentation import TextAugmentationAgent
        from langchain_core.language_models import BaseChatModel
        from langchain_core.messages import AIMessage
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_llm.invoke = Mock(return_value=AIMessage(
            content="Rephrased source text."
        ))
        
        agent = TextAugmentationAgent(llm=mock_llm)
        
        df = pd.DataFrame({
            'question_text': ['Q1?', 'Q2?'],
            'source_text': ['S1', 'S2']
        })
        
        result_df = agent.augment_dataframe(
            df,
            rephrase_question=False,
            rephrase_source=True,
            probability=1.0
        )
        
        assert len(result_df) >= len(df)

    def test_augment_dataframe_both(self):
        """Test augmenting both questions and sources."""
        from RAG_supporters.agents.text_augmentation import TextAugmentationAgent
        from langchain_core.language_models import BaseChatModel
        from langchain_core.messages import AIMessage
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_llm.invoke = Mock(return_value=AIMessage(
            content="Rephrased text"
        ))
        
        agent = TextAugmentationAgent(llm=mock_llm)
        
        df = pd.DataFrame({
            'question_text': ['Q1?', 'Q2?'],
            'source_text': ['S1', 'S2']
        })
        
        result_df = agent.augment_dataframe(
            df,
            rephrase_question=True,
            rephrase_source=True,
            probability=1.0
        )
        
        # Should have more rows due to augmentation
        assert len(result_df) >= len(df)

    def test_augment_dataframe_with_probability(self):
        """Test augmentation with probability < 1."""
        from RAG_supporters.agents.text_augmentation import TextAugmentationAgent
        from langchain_core.language_models import BaseChatModel
        from langchain_core.messages import AIMessage
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_llm.invoke = Mock(return_value=AIMessage(
            content="Rephrased"
        ))
        
        agent = TextAugmentationAgent(llm=mock_llm)
        
        df = pd.DataFrame({
            'question_text': ['Q1?', 'Q2?'],
            'source_text': ['S1', 'S2']
        })
        
        result_df = agent.augment_dataframe(
            df,
            rephrase_question=True,
            probability=0.5
        )
        
        # With probability 0.5, may or may not add rows
        assert len(result_df) >= len(df)

    def test_augment_dataframe_custom_columns(self):
        """Test augmentation with custom column names."""
        from RAG_supporters.agents.text_augmentation import TextAugmentationAgent
        from langchain_core.language_models import BaseChatModel
        from langchain_core.messages import AIMessage
        
        mock_llm = Mock(spec=BaseChatModel)
        mock_llm.invoke = Mock(return_value=AIMessage(
            content="Rephrased"
        ))
        
        agent = TextAugmentationAgent(llm=mock_llm)
        
        df = pd.DataFrame({
            'query': ['Q1?', 'Q2?'],
            'answer': ['A1', 'A2']
        })
        
        result_df = agent.augment_dataframe(
            df,
            columns_mapping={'question_text': 'query', 'source_text': 'answer'},
            rephrase_question=True,
            probability=1.0
        )
        
        assert 'query' in result_df.columns
        assert len(result_df) >= len(df)


class TestTextAugmentationAgentVerifyMeaning:
    """Test meaning verification functionality."""

    def test_verify_meaning_preserved(self):
        """Test that meaning verification works when enabled."""
        from RAG_supporters.agents.text_augmentation import TextAugmentationAgent
        from langchain_core.language_models import BaseChatModel
        from langchain_core.messages import AIMessage
        
        mock_llm = Mock(spec=BaseChatModel)
        # First call for rephrasing, second for verification
        mock_llm.invoke = Mock(side_effect=[
            AIMessage(content="Rephrased text"),
            AIMessage(content="EQUIVALENT")  # Meaning preserved
        ])
        
        agent = TextAugmentationAgent(llm=mock_llm, verify_meaning=True)
        
        result = agent.rephrase_text(
            text="Original text",
            mode="full"
        )
        
        # Should make 2 LLM calls when verify_meaning is True
        assert mock_llm.invoke.call_count == 2
        # Verify the result is returned correctly
        assert result is not None
        assert isinstance(result, str)
        assert result == "Rephrased text"


class TestTextAugmentationAgentIntegration:
    """Integration tests for TextAugmentationAgent."""

    @pytest.mark.skip(reason="Requires actual LLM API access")
    def test_real_llm_integration(self):
        """Test with real LLM (requires API key)."""
        from langchain_openai import ChatOpenAI
        from RAG_supporters.agents.text_augmentation import TextAugmentationAgent
        
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        agent = TextAugmentationAgent(llm=llm)
        
        result = agent.rephrase_text(
            text="Plants convert sunlight into energy through photosynthesis.",
            mode="full"
        )
        
        assert result is not None
        assert len(result) > 0
        assert result != "Plants convert sunlight into energy through photosynthesis."
