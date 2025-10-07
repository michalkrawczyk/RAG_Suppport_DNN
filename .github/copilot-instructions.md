# GitHub Copilot Instructions for RAG_Suppport_DNN

## Repository Overview

This repository implements a RAG (Retrieval-Augmented Generation) Support system with Deep Neural Networks for evaluating and comparing text sources in question-answering scenarios. The project uses LangChain and LangGraph to build intelligent agents that assess source quality and relevance.

## Key Technologies

- **LangChain** (v0.3.20+): Framework for building LLM applications
- **LangGraph** (v0.3.8): Library for building stateful, multi-actor applications with LLMs
- **Pydantic** (v2.10.3): Data validation using Python type annotations
- **PyTorch** (v2.3.0+): Deep learning framework for neural network models
- **Pandas**: Data manipulation and analysis
- **Datasets**: HuggingFace datasets library for dataset handling

## Project Structure

```
RAG_supporters/
├── agents/                     # LangGraph-based agents for source evaluation
│   ├── dataset_check.py       # Agent for comparing two text sources
│   └── source_assesment.py    # Agent for evaluating single sources
├── dataset/                    # Dataset handling and RAG dataset implementations
│   ├── rag_dataset.py         # Core RAG dataset classes
│   ├── templates/             # Dataset templates for specific formats
│   └── utils/                 # Dataset utilities and loaders
├── nn/                        # Neural network models
│   └── models/                # Model builders and architectures
├── prompts_templates/         # LLM prompt templates
│   ├── rag_verifiers.py      # Prompts for source verification and scoring
│   └── rag_generators.py     # Prompts for text generation tasks
├── requirements.txt           # Main dependencies
└── requirements_agents.txt    # Agent-specific dependencies
```

## Architecture & Design Patterns

### Agents

The project uses **LangGraph** for building stateful agents with retry logic and error handling:

1. **DatasetCheckAgent** (`agents/dataset_check.py`):
   - Compares two text sources for a given question
   - Returns a label indicating which source is better (0 for source1, 1 for source2, -1 for error)
   - Uses `CheckAgentState` TypedDict for state management
   - Implements a graph-based workflow for source comparison

2. **SourceEvaluationAgent** (`agents/source_assesment.py`):
   - Evaluates a single source against a question
   - Provides detailed scores (0-10) across multiple criteria:
     - Inferred Domain
     - Relevance
     - Expertise/Authority
     - Depth and Specificity
     - Clarity and Conciseness
     - Objectivity/Bias
     - Completeness
   - Uses Pydantic models for validation and structured outputs
   - Implements retry logic for handling LLM failures

### State Management

- **TypedDict** for agent states (e.g., `CheckAgentState`, `AgentState`)
- States contain messages, questions, source content, and evaluation results
- Immutable state transitions through LangGraph nodes

### Error Handling

- Try-except blocks with fallback classes when dependencies are missing
- Optional imports to handle missing dependencies gracefully
- Retry mechanisms in agents with configurable `max_retries`
- Logging via Python's `logging` module (use `LOGGER` variable)

## Coding Standards

### Python Style

- **Type hints**: Use type annotations for all function parameters and return values
- **Docstrings**: Follow NumPy docstring format with Parameters, Returns, and Raises sections
- **Formatting**: Code is formatted with `black` (see requirements.txt)
- **Imports**: Group imports (standard library, third-party, local) with blank lines between groups
- **Error handling**: Use specific exception types, avoid bare `except` clauses

### Pydantic Models (v2.10.3)

- Use `BaseModel` for validation
- Use `Field` for field descriptions and constraints (e.g., `ge=0, le=10` for score ranges)
- Use `field_validator` and `model_validator` decorators for custom validation
- All scores should be integers between 0 and 10

### LangChain & LangGraph Patterns

- **Messages**: Use `HumanMessage`, `AIMessage`, `SystemMessage` from `langchain_core.messages`
- **State Graphs**: Build workflows using `StateGraph` from `langgraph.graph`
- **Prompts**: Use `PromptTemplate` for structured prompts with format instructions
- **Output Parsers**: Use `PydanticOutputParser` and `OutputFixingParser` for structured LLM outputs
- **Chat Models**: Accept `BaseChatModel` for LLM flexibility

### Prompt Engineering

- Prompts are stored in `prompts_templates/` directory
- Use multi-criteria evaluation (relevance, expertise, depth, clarity, objectivity, completeness)
- Always include "inferred domain" concept - agents should infer the domain from context
- Scores use consistent scales (0-10 for single source, 1-5 for comparative scoring)
- Include clear instructions and examples in prompts

### Dataset Handling

- Use `SamplePairingType` enum for different pairing strategies (relevant, all_existing, embedding_similarity)
- RAG datasets use triplets: question + source1 + source2
- Support for ChromaDB vector stores via `langchain_chroma`
- Use `tqdm` for progress bars in batch operations

## Common Patterns to Follow

1. **Optional Dependencies**: Wrap imports in try-except blocks with fallback classes for optional features
2. **Batch Processing**: Use `batch_size` parameter and `tqdm` for progress tracking
3. **Empty Text Checking**: Use `_is_empty_text()` helper to check for empty/whitespace/NaN values
4. **Logging**: Always use module-level `LOGGER = logging.getLogger(__name__)`
5. **Configuration**: Use YAML for model configurations (see `nn/models/model_builder.py`)

## Testing & Validation

- Validate LLM outputs using Pydantic models
- Use `OutputFixingParser` to automatically fix minor formatting issues in LLM responses
- Implement retry logic for transient failures
- Check for empty text before processing

## Important Notes

- **LLM Providers**: The code checks for OpenAI LLMs using `_check_openai_llm()` but is designed to work with any `BaseChatModel`
- **Score Validation**: All evaluation scores must be integers within defined ranges (use Pydantic `Field` with `ge` and `le`)
- **Domain Inference**: A key feature is inferring the domain from question/source context rather than requiring explicit domain specification
- **TODOs**: Check inline TODO comments for areas needing improvement (e.g., plugin architecture for custom layers)

## Examples for Common Tasks

### Creating a New Agent

```python
from typing import Any, Dict
from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field

class MyAgentState(BaseModel):
    """State for the agent"""
    input_data: str
    result: Optional[str] = None
    error: Optional[str] = None
    retry_count: int = 0

class MyAgent:
    def __init__(self, llm: BaseChatModel, max_retries: int = 3):
        self._llm = llm
        self._max_retries = max_retries
        self._graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        # Build LangGraph workflow
        pass
```

### Adding a New Evaluation Criterion

1. Add the criterion to the prompt template in `prompts_templates/rag_verifiers.py`
2. Add a corresponding Pydantic field to the evaluation model
3. Update the scoring logic in the agent

### Working with Prompts

```python
from prompts_templates.rag_verifiers import SINGLE_SRC_SCORE_PROMPT
from langchain_core.prompts import PromptTemplate

prompt_template = PromptTemplate(
    template=SINGLE_SRC_SCORE_PROMPT,
    input_variables=["question", "source_content", "format_instructions"]
)
```

## Development Workflow

1. Install dependencies: `pip install -r requirements.txt -r requirements_agents.txt`
2. Format code with black: `black RAG_supporters/`
3. Ensure proper type hints and docstrings for all new functions/classes
4. Add logging for important operations
5. Handle errors gracefully with appropriate fallbacks
