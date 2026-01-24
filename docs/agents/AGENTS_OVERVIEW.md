# RAG Supporters Agents - Overview

## Introduction

The RAG Supporters agents are a collection of specialized LLM-powered agents designed to support the creation, curation, and enhancement of RAG (Retrieval-Augmented Generation) datasets. Each agent focuses on specific tasks in the RAG pipeline, from question generation to source evaluation and domain analysis.

## Available Agents

### 1. QuestionAugmentationAgent
**File:** `RAG_supporters/agents/question_augmentation_agent.py`  
**Documentation:** [CSV_QUESTION_AGENT.md](CSV_QUESTION_AGENT.md)

**Purpose:** Question generation and rephrasing in CSV/DataFrame contexts.

**Key Features:**
- Rephrase questions to align with source content
- Adapt questions to specific domains
- Generate alternative questions from sources
- Batch processing for CSV files

**Use Cases:**
- Improving question-source alignment
- Dataset augmentation
- Domain-specific question adaptation
- Generating training data for RAG systems

**Quick Example:**
```python
from langchain_openai import ChatOpenAI
from RAG_supporters.agents.question_augmentation_agent import QuestionAugmentationAgent

llm = ChatOpenAI(model="gpt-3.5-turbo")
agent = QuestionAugmentationAgent(llm=llm)

# Rephrase with source context
rephrased = agent.rephrase_question_with_source(
    question="What does it do?",
    source="Mitochondria produce ATP through cellular respiration..."
)

# Generate alternative questions
questions = agent.generate_alternative_questions(
    source="Photosynthesis converts light into chemical energy...",
    n=5
)
```

---

### 2. TextAugmentationAgent
**File:** `RAG_supporters/agents/text_augmentation.py`  
**Documentation:** [TEXT_AUGMENTATION.md](TEXT_AUGMENTATION.md)

**Purpose:** Generate alternative versions of questions and sources while preserving meaning.

**Key Features:**
- Full text rephrasing
- Sentence-level rephrasing
- Automatic mode selection
- Meaning verification
- CSV batch processing

**Use Cases:**
- Dataset augmentation
- Creating diverse training examples
- Paraphrasing for variety
- Expanding limited datasets

**Quick Example:**
```python
from langchain_openai import ChatOpenAI
from RAG_supporters.agents.text_augmentation import TextAugmentationAgent

llm = ChatOpenAI(model="gpt-3.5-turbo")
agent = TextAugmentationAgent(llm=llm)

# Augment a DataFrame
augmented_df = agent.augment_dataframe(
    df,
    rephrase_question=True,
    rephrase_source=True,
    rephrase_mode="random",
    probability=0.5
)
```

---

### 3. DatasetCheckAgent
**File:** `RAG_supporters/agents/dataset_check.py`  
**Documentation:** [DATASET_CHECK_AGENT.md](DATASET_CHECK_AGENT.md)

**Purpose:** Compare and evaluate text sources to determine which is better for answering questions.

**Key Features:**
- LLM-powered source comparison
- Duplicate detection
- Structured workflow with LangGraph
- Batch processing with checkpoints
- Resumable processing

**Use Cases:**
- Quality control
- Source selection
- Dataset curation
- Duplicate removal

**Quick Example:**
```python
from langchain_openai import ChatOpenAI
from RAG_supporters.agents.dataset_check import DatasetCheckAgent

llm = ChatOpenAI(model="gpt-4")
agent = DatasetCheckAgent(llm=llm)

# Compare sources
result = agent.compare_text_sources(
    question="What is photosynthesis?",
    source1="Detailed scientific explanation...",
    source2="Brief description...",
    return_analysis=True
)

# Result: label = 1 (source1), 2 (source2), 0 (neither), -1 (error)
```

---

### 4. DomainAnalysisAgent
**File:** `RAG_supporters/agents/domain_assessment.py`  
**Documentation:** [DOMAIN_ANALYSIS_AGENT.md](DOMAIN_ANALYSIS_AGENT.md)

**Purpose:** Unified agent for domain extraction, guessing, and assessment.

**Three Operation Modes:**

1. **EXTRACT Mode**: Extract domains from text sources
2. **GUESS Mode**: Identify domains needed for questions
3. **ASSESS Mode**: Evaluate relevance of available terms to questions

**Key Features:**
- Multi-mode operation
- Structured Pydantic output
- Batch processing
- LangGraph workflow
- Automatic retry logic

**Use Cases:**
- Document categorization
- Question routing
- Knowledge base organization
- Domain taxonomy building

**Quick Example:**
```python
from langchain_openai import ChatOpenAI
from RAG_supporters.agents.domain_assesment import DomainAnalysisAgent, OperationMode

llm = ChatOpenAI(model="gpt-4")
agent = DomainAnalysisAgent(llm=llm)

# Extract domains from text
result = agent.extract_domains("Machine learning enables systems to learn from data...")

# Guess domains for question
result = agent.guess_domains("How does gradient descent work?")

# Assess question against available terms
result = agent.assess_domains(
    question="What is photosynthesis?",
    available_terms=["biology", "chemistry", "physics"]
)
```

---

### 5. SourceEvaluationAgent
**File:** `RAG_supporters/agents/source_assesment.py`  
**Documentation:** [SOURCE_EVALUATION_AGENT.md](SOURCE_EVALUATION_AGENT.md)

**Purpose:** Multi-dimensional quality evaluation of sources against questions.

**Six Evaluation Dimensions:**
1. **Relevance** (0-10): How well the source addresses the question
2. **Expertise/Authority** (0-10): Credibility and authoritativeness
3. **Depth and Specificity** (0-10): Level of detail
4. **Clarity and Conciseness** (0-10): Readability
5. **Objectivity/Bias** (0-10): Neutrality
6. **Completeness** (0-10): Comprehensiveness

**Key Features:**
- Detailed multi-dimensional scoring
- Structured Pydantic output
- Batch processing
- Optional reasoning text
- Checkpoint support

**Use Cases:**
- Source quality assessment
- Source ranking
- Dataset quality control
- Filtering low-quality sources

**Quick Example:**
```python
from langchain_openai import ChatOpenAI
from RAG_supporters.agents.source_assesment import SourceEvaluationAgent

llm = ChatOpenAI(model="gpt-4")
agent = SourceEvaluationAgent(llm=llm)

# Evaluate a source
result = agent.evaluate(
    question="What is climate change?",
    source_content="Climate change refers to long-term shifts in temperatures..."
)

# Access scores
print(result['scores'])  # All six dimensions
print(result['inferred_domain'])
```

---

## Agent Comparison Matrix

| Agent | Primary Task | Input | Output | Batch Support | LangGraph |
|-------|-------------|-------|--------|---------------|-----------|
| QuestionAugmentationAgent | Question rephrasing & generation | Question, Source/Domain | Rephrased question or list of questions | ✅ | ❌ |
| TextAugmentationAgent | Text augmentation | Text | Rephrased text | ✅ | ❌ |
| DatasetCheckAgent | Source comparison | Question, 2 Sources | Label (0/1/2/-1) | ✅ | ✅ |
| DomainAnalysisAgent | Domain analysis | Text/Question/Terms | Domain suggestions | ✅ | ✅ |
| SourceEvaluationAgent | Source quality scoring | Question, Source | 6 dimension scores | ✅ | ✅ |

## Common Patterns

### Initialization

All agents follow a similar initialization pattern:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4", temperature=0.3)

# Initialize any agent
agent = AgentClass(llm=llm, max_retries=3)
```

### DataFrame Processing

Most agents support DataFrame processing:

```python
import pandas as pd

df = pd.read_csv("data.csv")

result_df = agent.process_dataframe(
    df,
    # Agent-specific parameters
    save_path="results.csv",
    skip_existing=True,
    use_batch_processing=True,
    batch_size=20
)
```

### Batch Processing

For OpenAI models, batch processing is significantly faster:

```python
# Single items
result = agent.method(input)

# Batch items (OpenAI only)
results = agent.method_batch(inputs)
```

## Workflow Examples

### Complete RAG Dataset Creation Pipeline

```python
from langchain_openai import ChatOpenAI
from RAG_supporters.agents import (
    QuestionAugmentationAgent,
    TextAugmentationAgent,
    DomainAnalysisAgent,
    DatasetCheckAgent,
    SourceEvaluationAgent
)
import pandas as pd

llm = ChatOpenAI(model="gpt-4", temperature=0.3)

# 1. Generate alternative questions from sources
qa_agent = QuestionAugmentationAgent(llm=llm)
sources_df = pd.read_csv("sources.csv")
qa_pairs = qa_agent.process_dataframe_generation(
    sources_df,
    n_questions=5
)

# 2. Augment with rephrased versions
text_agent = TextAugmentationAgent(llm=llm)
augmented = text_agent.augment_dataframe(
    qa_pairs,
    rephrase_question=True,
    rephrase_source=True,
    probability=0.3
)

# 3. Extract domains for categorization
domain_agent = DomainAnalysisAgent(llm=llm)
categorized = domain_agent.process_dataframe(
    augmented,
    mode=OperationMode.EXTRACT,
    text_source_col="source_text"
)

# 4. Evaluate source quality
eval_agent = SourceEvaluationAgent(llm=llm)
evaluated = eval_agent.process_dataframe(
    categorized,
    question_col="question_text",
    source_col="source_text",
    include_reasoning=False
)

# 5. Filter high-quality pairs
evaluated['avg_score'] = evaluated[[
    'relevance_score', 'expertise_authority_score',
    'depth_specificity_score', 'clarity_conciseness_score',
    'objectivity_bias_score', 'completeness_score'
]].mean(axis=1)

final_dataset = evaluated[evaluated['avg_score'] >= 7.0]
final_dataset.to_csv("high_quality_rag_dataset.csv", index=False)

print(f"Created {len(final_dataset)} high-quality QA pairs")
```

### Source Comparison and Selection

```python
# Compare multiple sources for each question
check_agent = DatasetCheckAgent(llm=llm)

# DataFrame with question, source1, source2
comparisons_df = pd.read_csv("sources_to_compare.csv")
labeled = check_agent.process_dataframe(comparisons_df)

# Select winners
winners = []
for idx, row in labeled.iterrows():
    if row['label'] == 1:
        winners.append({'question': row['question_text'], 'source': row['answer_text_1']})
    elif row['label'] == 2:
        winners.append({'question': row['question_text'], 'source': row['answer_text_2']})

winners_df = pd.DataFrame(winners)

# Evaluate winners for quality
eval_agent = SourceEvaluationAgent(llm=llm)
final = eval_agent.process_dataframe(
    winners_df,
    question_col="question",
    source_col="source"
)
```

### Domain-Based Question Routing

```python
# Define expert domains
expert_domains = {
    "machine_learning": ["ML expert queue"],
    "quantum_physics": ["Physics expert queue"],
    "molecular_biology": ["Biology expert queue"]
}

# Analyze incoming questions
domain_agent = DomainAnalysisAgent(llm=llm)
questions_df = pd.read_csv("incoming_questions.csv")

assessed = domain_agent.process_dataframe(
    questions_df,
    mode=OperationMode.ASSESS,
    question_col="question_text",
    available_terms=list(expert_domains.keys())
)

# Route questions
for idx, row in assessed.iterrows():
    import ast
    selected = ast.literal_eval(row['selected_terms'])
    if selected:
        primary_domain = selected[0]['term']
        queue = expert_domains.get(primary_domain, ["General queue"])
        print(f"Route to: {queue[0]}")
```

## Best Practices

### 1. Model Selection

- **GPT-4**: Best for evaluation and comparison tasks (DatasetCheckAgent, SourceEvaluationAgent)
- **GPT-3.5-turbo**: Sufficient for generation and augmentation (QuestionAugmentationAgent, TextAugmentationAgent)
- **Temperature**: 
  - Low (0.2-0.3) for consistent evaluation
  - Medium (0.5-0.7) for rephrasing
  - Higher (0.7-0.9) for creative generation

### 2. Batch Processing

- Always use `use_batch_processing=True` for OpenAI models
- Adjust `batch_size` (10-50) based on your rate limits
- Monitor API costs - batch processing saves on time but not necessarily costs

### 3. Large Dataset Processing

- Enable checkpoints: `checkpoint_batch_size=100`
- Use `skip_existing=True` to resume interrupted processing
- Save intermediate results with `save_path`
- Process in chunks if memory is limited

### 4. Error Handling

- Set appropriate `max_retries` (3-5) for unreliable networks
- Check for None returns from single operations
- Monitor `evaluation_error` or `domain_analysis_error` columns in DataFrames
- Enable logging to track progress and errors

### 5. Cost Optimization

- Use batch processing to reduce latency
- Start with smaller models for experimentation
- Use `skip_existing=True` to avoid reprocessing
- Consider using cheaper models for less critical tasks

## Installation

### Quick Start

```bash
# Install all agent dependencies
pip install -r RAG_supporters/requirements_agents.txt

# Or install individually
pip install langchain langgraph langchain-core pydantic pandas tqdm

# For OpenAI
pip install langchain-openai
```

### Verify Installation

```python
# Test imports
from RAG_supporters.agents.question_augmentation_agent import QuestionAugmentationAgent
from RAG_supporters.agents.text_augmentation import TextAugmentationAgent
from RAG_supporters.agents.dataset_check import DatasetCheckAgent
from RAG_supporters.agents.domain_assesment import DomainAnalysisAgent
from RAG_supporters.agents.source_assesment import SourceEvaluationAgent

print("All agents imported successfully!")
```

## Troubleshooting

### Common Issues Across All Agents

**Issue: ImportError for dependencies**
```bash
# Solution: Install agent dependencies
pip install -r RAG_supporters/requirements_agents.txt
```

**Issue: LLM connection errors**
- Check API keys are set correctly
- Verify network connectivity
- Check rate limits haven't been exceeded

**Issue: Slow processing**
- Enable batch processing for OpenAI models
- Use faster models (gpt-3.5-turbo)
- Increase batch_size

**Issue: Inconsistent results**
- Lower temperature (0.2-0.3)
- Use more capable model (GPT-4)
- Ensure prompts are clear

## Performance Comparison

| Agent | Single Operation | Batch (OpenAI) | Memory Usage |
|-------|-----------------|----------------|--------------|
| QuestionAugmentationAgent | ~1-2s | ~0.2-0.5s | Low |
| TextAugmentationAgent | ~1-2s | ~0.2-0.5s | Low |
| DatasetCheckAgent | ~2-4s | N/A | Low |
| DomainAnalysisAgent | ~2-3s | ~0.3-0.5s | Low |
| SourceEvaluationAgent | ~2-3s | ~0.3-0.5s | Low |

*Times are approximate and depend on model, network, and complexity*

## API Cost Estimates

Based on GPT-4 pricing (approximate):

| Agent | Cost per Operation | Notes |
|-------|-------------------|-------|
| QuestionAugmentationAgent | $0.002-0.004 | 1 LLM call per operation |
| TextAugmentationAgent | $0.002-0.006 | 1-2 calls (if verification enabled) |
| DatasetCheckAgent | $0.004-0.008 | 2 LLM calls (analysis + verdict) |
| DomainAnalysisAgent | $0.002-0.006 | 1 call + retries |
| SourceEvaluationAgent | $0.002-0.006 | 1 call + retries |

*Costs are estimates and will vary based on actual usage and pricing*

## Logging

Enable logging across all agents:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Or configure per agent
logger = logging.getLogger('RAG_supporters.agents')
logger.setLevel(logging.INFO)
```

## Testing

### Running Tests

All agents should have comprehensive unit tests in the `tests/` directory. To run the tests:

```bash
# Run all tests
pytest tests/

# Run tests for a specific agent
pytest tests/test_dataset_check_agent.py
pytest tests/test_question_augmentation_agent.py
pytest tests/test_text_augmentation_agent.py
pytest tests/test_source_evaluation_agent.py
pytest tests/test_domain_assesment.py

# Run tests with coverage
pytest tests/ --cov=RAG_supporters --cov-report=html

# Run tests with verbose output
pytest tests/ -v

# Run tests and stop at first failure
pytest tests/ -x
```

### Test Examples

Each agent has corresponding test files that demonstrate proper usage patterns:

- `tests/test_dataset_check_agent.py` - Tests for DatasetCheckAgent
- `tests/test_question_augmentation_agent.py` - Tests for QuestionAugmentationAgent
- `tests/test_text_augmentation_agent.py` - Tests for TextAugmentationAgent
- `tests/test_source_evaluation_agent.py` - Tests for SourceEvaluationAgent
- `tests/test_domain_assesment.py` - Tests for DomainAnalysisAgent
- `tests/test_dataset_splitter.py` - Tests for DatasetSplitter
- `tests/test_topic_distance_calculator.py` - Tests for TopicDistanceCalculator

### Writing Tests for New Agents

When adding a new agent or modifying an existing one, follow these guidelines:

1. **Create a test file** in `tests/` with the naming pattern `test_<agent_name>.py`
2. **Test initialization** - Verify agent can be initialized with valid parameters
3. **Test core methods** - Test all public methods with various inputs
4. **Test error handling** - Ensure graceful error handling and recovery
5. **Test DataFrame processing** - Verify batch processing functionality
6. **Mock LLM calls** - Use mocks to avoid API costs during testing
7. **Add integration tests** - Optional tests with real LLM (marked with @pytest.mark.skip)

Example test structure:

```python
"""Tests for NewAgent."""

import pytest
from unittest.mock import Mock
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

# Skip if dependencies not available
pytest.importorskip("langchain")
pytest.importorskip("langchain_core")

def test_agent_import():
    """Test that agent can be imported."""
    from RAG_supporters.agents.new_agent import NewAgent
    assert NewAgent is not None

class TestNewAgentInit:
    """Test agent initialization."""
    
    def test_init_with_llm(self):
        """Test basic initialization."""
        from RAG_supporters.agents.new_agent import NewAgent
        mock_llm = Mock(spec=BaseChatModel)
        agent = NewAgent(llm=mock_llm)
        assert agent._llm == mock_llm

class TestNewAgentMethods:
    """Test agent methods."""
    
    def test_process_method(self):
        """Test main processing method."""
        from RAG_supporters.agents.new_agent import NewAgent
        mock_llm = Mock(spec=BaseChatModel)
        mock_llm.invoke = Mock(return_value=AIMessage(content="Result"))
        
        agent = NewAgent(llm=mock_llm)
        result = agent.process("input")
        
        assert result is not None
```

## Contributing

To add a new agent:

1. Create agent file in `RAG_supporters/agents/`
2. Follow existing agent patterns (initialization, batch processing, DataFrame support)
3. Add comprehensive documentation in `docs/`
4. Include usage examples
5. **Write unit tests in `tests/`** - This is mandatory for all new agents
6. Update this overview document

## Repository Etiquette

When contributing to this repository:

### Pull Request Requirements

1. **Add tests for new functionality** - All new agents and significant features must include unit tests
2. **Ensure tests pass** - Run `pytest tests/` before submitting PR
3. **Update documentation** - Add or update relevant documentation files
4. **Follow code style** - Maintain consistency with existing code patterns
5. **Add type hints** - Use type annotations for function parameters and returns
6. **Include docstrings** - Document all public classes and methods

### Code Quality Standards

- All tests must pass before merging
- Test coverage should be maintained or improved
- Code should follow PEP 8 style guidelines
- Complex logic should include inline comments
- Breaking changes must be documented

### Testing Best Practices

- Mock external dependencies (LLM calls, API requests)
- Test both success and failure cases
- Use descriptive test names that explain what is being tested
- Keep tests focused and independent
- Add integration tests for critical workflows (marked as skippable)

## Notes for AI Coding Agents

When working with this codebase as an AI coding agent:

### Step-by-Step Workflow

1. **Understand requirements** - Read issue description and acceptance criteria carefully
2. **Explore existing patterns** - Review similar agents and their implementations
3. **Plan implementation** - Outline the agent structure before coding
4. **Write the agent code** - Follow existing patterns for consistency
5. **Add documentation** - Create comprehensive documentation with examples
6. **Write unit tests** - Create thorough test coverage for all agent functionality
   - Test initialization with various parameters
   - Test all public methods with different inputs
   - Test error handling and edge cases
   - Mock LLM calls to avoid API costs
   - Add integration tests (marked as skippable) if appropriate
7. **Run tests locally** - Verify all tests pass with `pytest tests/`
8. **Update overview documentation** - Add agent to this file and relevant READMEs

### Testing Guidelines for AI Agents

**IMPORTANT:** Every new or modified agent **must** have corresponding unit tests. This is not optional.

- Create test file: `tests/test_<agent_name>.py`
- Use pytest framework and follow existing test patterns
- Mock LLM responses to avoid API costs during testing
- Test both successful operations and error conditions
- Ensure tests can run without external dependencies (API keys, network)
- Add descriptive docstrings to each test explaining what it validates
- Include integration tests with `@pytest.mark.skip` for optional real LLM testing

### Common Pitfalls to Avoid

1. **Forgetting to add tests** - Always create unit tests for new agents
2. **Not mocking LLM calls** - Tests should not require real API keys
3. **Skipping error handling** - Always test failure scenarios
4. **Insufficient test coverage** - Test all public methods and edge cases
5. **Not updating documentation** - Documentation should match implementation
6. **Breaking existing tests** - Run full test suite before submitting

## Resources

- **Individual Agent Documentation**: See links in each agent section above
- **Prompt Templates**: `RAG_supporters/prompts_templates/`
- **Requirements**: `RAG_supporters/requirements_agents.txt`
- **Test Examples**: `tests/` directory
- **Examples**: Included in each agent's documentation

## Support

For issues, questions, or contributions:
- Check individual agent documentation for specific issues
- Review troubleshooting sections
- Enable logging for detailed error information
- Review test files for usage examples

---

**Last Updated**: January 2026  
**Version**: 1.1
