"""Tests for DatasetCheckAgent."""

import logging
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

# Skip all tests if required dependencies are not installed
pytest.importorskip("langchain")
pytest.importorskip("langchain_core")
pytest.importorskip("langgraph")


def test_agent_import():
    """Test that DatasetCheckAgent can be imported."""
    from RAG_supporters.agents.dataset_check import DatasetCheckAgent
    
    assert DatasetCheckAgent is not None


def test_check_agent_state_import():
    """Test that CheckAgentState can be imported."""
    from RAG_supporters.agents.dataset_check import CheckAgentState
    
    assert CheckAgentState is not None


class TestDatasetCheckAgentInit:
    """Test DatasetCheckAgent initialization."""

    def test_init_with_llm(self):
        """Test initialization with a language model."""
        from RAG_supporters.agents.dataset_check import DatasetCheckAgent
        
        mock_llm = Mock()
        agent = DatasetCheckAgent(llm=mock_llm)
        
        assert agent._llm == mock_llm
        assert agent._executor is not None
        assert agent.compare_prompt is not None

    def test_init_with_custom_prompt(self):
        """Test initialization with custom prompt."""
        from RAG_supporters.agents.dataset_check import DatasetCheckAgent
        
        mock_llm = Mock()
        custom_prompt = "Custom prompt: {question} {source1_content} {source2_content}"
        agent = DatasetCheckAgent(llm=mock_llm, compare_prompt=custom_prompt)
        
        assert agent.compare_prompt == custom_prompt


class TestDatasetCheckAgentCompareTextSources:
    """Test compare_text_sources method."""

    def test_compare_text_sources_source1_better(self):
        """Test comparison when source1 is better."""
        from RAG_supporters.agents.dataset_check import DatasetCheckAgent
        from langchain_core.messages import AIMessage
        
        # Create mock LLM that returns predetermined responses
        mock_llm = Mock()
        mock_llm.invoke = Mock(side_effect=[
            AIMessage(content="Source 1 provides more detailed and accurate information."),
            AIMessage(content="source 1")
        ])
        
        agent = DatasetCheckAgent(llm=mock_llm)
        
        result = agent.compare_text_sources(
            question="What is photosynthesis?",
            source1="Photosynthesis is the process by which plants convert light energy into chemical energy.",
            source2="Plants make food.",
            return_analysis=True
        )
        
        assert result["label"] == 1
        assert result["analysis"] is not None
        assert "Source 1" in result["analysis"]

    def test_compare_text_sources_source2_better(self):
        """Test comparison when source2 is better."""
        from RAG_supporters.agents.dataset_check import DatasetCheckAgent
        from langchain_core.messages import AIMessage
        
        mock_llm = Mock()
        mock_llm.invoke = Mock(side_effect=[
            AIMessage(content="Source 2 is more comprehensive."),
            AIMessage(content="source 2")
        ])
        
        agent = DatasetCheckAgent(llm=mock_llm)
        
        result = agent.compare_text_sources(
            question="What is AI?",
            source1="AI is smart.",
            source2="Artificial Intelligence is the simulation of human intelligence by machines.",
            return_analysis=True
        )
        
        assert result["label"] == 2

    def test_compare_text_sources_neither(self):
        """Test comparison when neither source is good."""
        from RAG_supporters.agents.dataset_check import DatasetCheckAgent
        from langchain_core.messages import AIMessage
        
        mock_llm = Mock()
        mock_llm.invoke = Mock(side_effect=[
            AIMessage(content="Both sources are inadequate."),
            AIMessage(content="neither")
        ])
        
        agent = DatasetCheckAgent(llm=mock_llm)
        
        result = agent.compare_text_sources(
            question="What is quantum mechanics?",
            source1="It's complicated.",
            source2="It's hard to understand."
        )
        
        assert result["label"] == 0

    def test_compare_text_sources_with_messages(self):
        """Test that messages are returned when requested."""
        from RAG_supporters.agents.dataset_check import DatasetCheckAgent
        from langchain_core.messages import AIMessage
        
        mock_llm = Mock()
        mock_llm.invoke = Mock(side_effect=[
            AIMessage(content="Analysis text"),
            AIMessage(content="source 1")
        ])
        
        agent = DatasetCheckAgent(llm=mock_llm)
        
        result = agent.compare_text_sources(
            question="Test question",
            source1="Test source 1",
            source2="Test source 2",
            return_messages=True
        )
        
        assert len(result["messages"]) > 0

    def test_compare_text_sources_error_handling(self):
        """Test error handling when LLM fails."""
        from RAG_supporters.agents.dataset_check import DatasetCheckAgent
        
        mock_llm = Mock()
        mock_llm.invoke = Mock(side_effect=Exception("LLM error"))
        
        agent = DatasetCheckAgent(llm=mock_llm)
        
        result = agent.compare_text_sources(
            question="Test question",
            source1="Test source 1",
            source2="Test source 2"
        )
        
        assert result["label"] == -1


class TestDatasetCheckAgentProcessDataFrame:
    """Test process_dataframe method."""

    def test_process_dataframe_basic(self):
        """Test basic dataframe processing."""
        from RAG_supporters.agents.dataset_check import DatasetCheckAgent
        from langchain_core.messages import AIMessage
        
        mock_llm = Mock()
        mock_llm.invoke = Mock(side_effect=[
            AIMessage(content="Analysis"),
            AIMessage(content="source 1"),
            AIMessage(content="Analysis"),
            AIMessage(content="source 2"),
        ])
        
        agent = DatasetCheckAgent(llm=mock_llm)
        
        df = pd.DataFrame({
            'question_text': ['Q1', 'Q2'],
            'answer_text_1': ['A1', 'A2'],
            'answer_text_2': ['B1', 'B2'],
            'label': [-1, -1]
        })
        
        result_df = agent.process_dataframe(df)
        
        assert 'label' in result_df.columns
        assert len(result_df) == 2

    def test_process_dataframe_skip_labeled(self):
        """Test that already labeled rows are skipped when skip_labeled=True."""
        from RAG_supporters.agents.dataset_check import DatasetCheckAgent
        from langchain_core.messages import AIMessage
        
        mock_llm = Mock()
        # Should only be called once for the unlabeled row
        mock_llm.invoke = Mock(side_effect=[
            AIMessage(content="Analysis"),
            AIMessage(content="source 1"),
        ])
        
        agent = DatasetCheckAgent(llm=mock_llm)
        
        df = pd.DataFrame({
            'question_text': ['Q1', 'Q2'],
            'answer_text_1': ['A1', 'A2'],
            'answer_text_2': ['B1', 'B2'],
            'label': [1, -1]
        })
        
        result_df = agent.process_dataframe(df, skip_labeled=True)
        
        assert result_df.iloc[0]['label'] == 1  # Should be unchanged
        assert mock_llm.invoke.call_count == 2  # Only called for second row

    def test_process_dataframe_start_index(self):
        """Test processing with start_index."""
        from RAG_supporters.agents.dataset_check import DatasetCheckAgent
        from langchain_core.messages import AIMessage
        
        mock_llm = Mock()
        # Should only be called for rows from start_index
        mock_llm.invoke = Mock(side_effect=[
            AIMessage(content="Analysis"),
            AIMessage(content="source 2"),
        ])
        
        agent = DatasetCheckAgent(llm=mock_llm)
        
        df = pd.DataFrame({
            'question_text': ['Q1', 'Q2'],
            'answer_text_1': ['A1', 'A2'],
            'answer_text_2': ['B1', 'B2'],
            'label': [-1, -1]
        })
        
        result_df = agent.process_dataframe(df, start_index=1)
        
        assert result_df.iloc[0]['label'] == -1  # First row unchanged
        assert mock_llm.invoke.call_count == 2  # Only called for second row

    def test_process_dataframe_invalid_start_index(self):
        """Test processing with invalid start_index."""
        from RAG_supporters.agents.dataset_check import DatasetCheckAgent
        
        mock_llm = Mock()
        agent = DatasetCheckAgent(llm=mock_llm)
        
        df = pd.DataFrame({
            'question_text': ['Q1', 'Q2'],
            'answer_text_1': ['A1', 'A2'],
            'answer_text_2': ['B1', 'B2'],
            'label': [-1, -1]
        })
        
        result_df = agent.process_dataframe(df, start_index=100)
        
        assert len(result_df) == 2
        # DataFrame should be unchanged
        assert all(result_df['label'] == -1)


class TestDatasetCheckAgentIntegration:
    """Integration tests for DatasetCheckAgent."""

    @pytest.mark.skip(reason="Requires actual LLM API access")
    def test_real_llm_integration(self):
        """Test with real LLM (requires API key)."""
        from langchain_openai import ChatOpenAI
        from RAG_supporters.agents.dataset_check import DatasetCheckAgent
        
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
        agent = DatasetCheckAgent(llm=llm)
        
        result = agent.compare_text_sources(
            question="What is the capital of France?",
            source1="Paris is the capital of France.",
            source2="The capital of France is Paris."
        )
        
        # Both sources are essentially the same, so should be neither or equal
        assert result["label"] in [0, 1, 2]
