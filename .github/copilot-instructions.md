# GitHub Copilot Instructions for RAG_Suppport_DNN

## Repository Overview

This repository is about experiments with enhancing RAG (Retrieval-Augmented Generation) systems using Deep Neural Networks. The project focuses on evaluating and comparing text sources in question-answering scenarios. It uses LangChain and LangGraph to build intelligent agents that assess source quality and relevance.

## Key Technologies

- **LangChain** (v0.3.20+): Framework for building LLM applications
- **LangGraph** (v0.3.8): Library for building stateful, multi-actor applications with LLMs
- **Pydantic** (v2.10.3): Data validation using Python type annotations
- **PyTorch** (v2.3.0+): Deep learning framework for neural network models
- **Pandas**: Data manipulation and analysis
- **Datasets**: HuggingFace datasets library for dataset handling
- **ChromaDB** (`langchain_chroma`): Vector store for embeddings and similarity search

## Project Structure

```
RAG_supporters/
├── agents/                     # LangGraph-based agents for source evaluation
│   ├── __init__.py
│   ├── dataset_check.py       # Agent for comparing two text sources
│   └── source_assesment.py    # Agent for evaluating single sources
├── dataset/                    # Dataset handling and RAG dataset implementations
│   ├── __init__.py
│   ├── rag_dataset.py         # Core RAG dataset classes (BaseRAGDatasetGenerator)
│   ├── templates/             # Dataset templates for specific formats
│   │   ├── __init__.py
│   │   └── rag_mini_bioasq.py # BioASQ dataset implementation
│   └── utils/                 # Dataset utilities and loaders
│       ├── __init__.py
│       ├── dataset_loader.py  # PyTorch DataLoader implementations
│       └── text_splitters.py  # Text chunking utilities
├── nn/                        # Neural network models
│   ├── __init__.py
│   └── models/                # Model builders and architectures
│       ├── __init__.py
│       └── model_builder.py   # YAML-based configurable model builder
├── prompts_templates/         # LLM prompt templates
│   ├── __init__.py
│   ├── rag_verifiers.py      # Prompts for source verification and scoring
│   └── rag_generators.py     # Prompts for text generation tasks
├── requirements.txt           # Main dependencies
└── requirements_agents.txt    # Agent-specific dependencies (LangChain, LangGraph, Pydantic)
```

## Architecture & Design Patterns

### Agents

The project uses **LangGraph** for building stateful agents with retry logic and error handling:

1. **DatasetCheckAgent** (`agents/dataset_check.py`):
   - Compares two text sources for a given question
   - Returns a label indicating which source is better (0 for neither, 1 for source1, 2 for source2, -1 for error)
   - Uses `CheckAgentState` TypedDict for state management
   - Implements a graph-based workflow with `source_check` → `assign_label` nodes
   - Supports DataFrame and CSV processing with `process_dataframe()` and `process_csv()`

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
   - Supports batch processing for OpenAI LLMs via `evaluate_batch()`
   - DataFrame processing with `process_dataframe()` including checkpoint saves

### Dataset Classes

1. **BaseRAGDatasetGenerator** (`dataset/rag_dataset.py`):
   - Abstract base class for RAG dataset generation
   - Uses ChromaDB for question and text corpus storage
   - Generates triplet samples: question + source1 + source2
   - Supports three sample types: positive, contrastive, similar
   - Key methods: `generate_samples()`, `validate_triplet_samples()`, `evaluate_pair_samples()`

2. **SampleTripletRAGChroma** dataclass:
   - `question_id`, `source_id_1`, `source_id_2`, `label`
   - Labels: -1 (unlabeled), 0 (both irrelevant), 1 (source1 better), 2 (source2 better)

3. **SamplePairingType** enum:
   - `RELEVANT`: Passages assigned to the same question
   - `ALL_EXISTING`: All passages in database
   - `EMBEDDING_SIMILARITY`: Based on vector similarity search

### Neural Network Models

**ConfigurableModel** (`nn/models/model_builder.py`):
- Builds PyTorch models from YAML configuration files
- Supports nested Sequential layers
- Validates model with warmup passes
- Uses `yaml.safe_load()` for security

### State Management

- **TypedDict** for agent states (e.g., `CheckAgentState`)
- **Pydantic BaseModel** for complex states (e.g., `AgentState` in source_assesment.py)
- States contain messages, questions, source content, and evaluation results
- Immutable state transitions through LangGraph nodes

### Error Handling

- Try-except blocks with fallback classes when dependencies are missing
- Optional imports to handle missing dependencies gracefully (see pattern in agents)
- Retry mechanisms in agents with configurable `max_retries`
- Logging via Python's `logging` module (use `LOGGER` variable)
- KeyboardInterrupt handling for long-running processes with checkpoint saves

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
- RAG datasets use triplets (question + source1 + source2) and pairs (question + source)
- Support for ChromaDB vector stores via `langchain_chroma`
- Use `tqdm` for progress bars in batch operations and large iteration operations
- Checkpoint saves during long processing with `checkpoint_batch_size` parameter
- CSV export/import for triplet samples with `save_triplets_to_csv()`

### Prompt Templates

The repository uses several prompt templates in `prompts_templates/`:

1. **Source Comparison Prompts** (`rag_verifiers.py`):
   - `SRC_COMPARE_PROMPT`: Basic comparison without scores
   - `SRC_COMPARE_PROMPT_WITH_SCORES`: Comparison with 1-5 scores per criterion
   - `FINAL_VERDICT_PROMPT`: Extract final decision (Source 1/Source 2/Neither)
   - `SINGLE_SRC_SCORE_PROMPT`: Single source evaluation with 0-10 scores
   - `CONTEXT_SUFFICIENCY_PROMPT`: Rate context sufficiency 0-5

2. **Generator Prompts** (`rag_generators.py`):
   - `SUB_TEXT_SPLIT_PROMPT`: Extract relevant vs irrelevant text sections
   - `QUESTIONS_FROM_2_SOURCES_PROMPT`: Generate questions one source can answer but another cannot

## Common Patterns to Follow

1. **Optional Dependencies**: Wrap imports in try-except blocks with fallback classes for optional features
   ```python
   try:
       from langchain_openai import OpenAIEmbeddings
   except ImportError:
       def OpenAIEmbeddings(*args, **kwargs):
           raise ImportError("...")
   ```

2. **Batch Processing**: Use `batch_size` parameter and `tqdm` for progress tracking

3. **Empty Text Checking**: Use `_is_empty_text()` helper to check for empty/whitespace/NaN values
   ```python
   def _is_empty_text(text: str) -> bool:
       if not text or text.strip() == "":
           return True
       if text.lower() == "nan":
           return True
       return False
   ```

4. **Logging**: Always use module-level `LOGGER = logging.getLogger(__name__)`

5. **Configuration**: Use YAML for model configurations (see `nn/models/model_builder.py`)

6. **ChromaDB Integration**: Use embedding functions with ChromaDB for vector storage
   ```python
   self._question_db = Chroma(
       embedding_function=self._embed_function,
       persist_directory=os.path.join(dataset_dir, "question_db"),
   )
   ```

7. **DataFrame Processing with Checkpoints**:
   ```python
   if save_path and checkpoint_batch_size and processed_count % checkpoint_batch_size == 0:
       result_df.to_csv(save_path, index=False)
   ```

8. **Fallback Classes for Missing Agent Dependencies** (only for agents and code using `requirements_agents.txt`):
   ```python
   except ImportError as e:
       LOGGER.warning(f"ImportError: {str(e)}...")
       class MyAgent:
           def __init__(self, *args, **kwargs):
               raise ImportError("MyAgent requires X to be installed.")
   ```

## Testing & Validation

- Validate LLM outputs using Pydantic models
- Use `OutputFixingParser` to automatically fix minor formatting issues in LLM responses
- Implement retry logic for transient failures
- Check for empty text before processing
- Use `model_validator` for cross-field validation in Pydantic models

## Important Notes

- **LLM Providers**: The code checks for OpenAI LLMs using `_check_openai_llm()` but is designed to work with any `BaseChatModel`
- **Score Validation**: Evaluation scores can be integers or floats within defined ranges (use Pydantic `Field` with `ge` and `le`)
- **Domain Inference**: A key feature is inferring the domain from question/source context rather than requiring explicit domain specification
- **TODOs**: Check inline TODO comments for areas needing improvement (e.g., plugin architecture for custom layers)
- **Label Conventions**: For triplet comparisons: -1 = unlabeled/error, 0 = neither/both irrelevant, 1 = source1 better, 2 = source2 better
- **Embedding Model**: Default is OpenAI's `text-embedding-3-small`
- **HuggingFace Datasets**: Uses `enelpol/rag-mini-bioasq` dataset for primary testing, more datasets will be added

## Examples for Common Tasks

### Creating a New Agent

```python
from typing import Any, Dict, Optional
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

class MyAgentState(BaseModel):
    """State for the agent"""
    input_data: str
    result: Optional[str] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

class MyAgent:
    def __init__(self, llm: BaseChatModel, max_retries: int = 3):
        self._llm = llm
        self._max_retries = max_retries
        self._graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(MyAgentState)
        workflow.add_node("process", self._process)
        workflow.set_entry_point("process")
        workflow.add_edge("process", END)
        return workflow.compile()
```

### Creating a Dataset Generator

```python
from dataset.rag_dataset import BaseRAGDatasetGenerator, SampleTripletRAGChroma

class MyDatasetGenerator(BaseRAGDatasetGenerator):
    def __init__(self, dataset_dir: str, embed_function):
        self._dataset_dir = dataset_dir
        self._embed_function = embed_function
        self.load_dataset()
    
    def load_dataset(self):
        self._question_db = Chroma(
            embedding_function=self._embed_function,
            persist_directory=os.path.join(self._dataset_dir, "question_db"),
        )
        self._text_corpus_db = Chroma(
            embedding_function=self._embed_function,
            persist_directory=os.path.join(self._dataset_dir, "text_corpus_db"),
        )
```

### Working with Prompts

```python
from prompts_templates.rag_verifiers import SINGLE_SRC_SCORE_PROMPT
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser

# Create parser for structured output
parser = PydanticOutputParser(pydantic_object=SourceEvaluation)

# Create prompt template with format instructions
prompt_template = PromptTemplate(
    template=SINGLE_SRC_SCORE_PROMPT,
    input_variables=["question", "source_content"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)
```

### Processing DataFrames with Evaluation

```python
from agents.source_assesment import SourceEvaluationAgent

evaluator = SourceEvaluationAgent(llm=llm, max_retries=3)
result_df = evaluator.process_dataframe(
    df=pairs_df,
    question_col="question_text",
    source_col="source_text",
    include_reasoning=True,
    save_path="results.csv",
    checkpoint_batch_size=100,
)
```

### Similarity Search with ChromaDB

```python
# Raw similarity search using embeddings
results = self._raw_similarity_search(
    embedding_or_text=question_embedding,
    search_db="text",  # or "question"
    k=5,
    include=["distances", "documents"],
)
```

## Development Workflow

1. Install dependencies: `pip install -r requirements.txt -r requirements_agents.txt`
2. Format code with black: `black RAG_supporters/`
3. Ensure proper type hints and docstrings for all new functions/classes
4. Add logging for important operations
5. Handle errors gracefully with appropriate fallbacks
6. Use checkpoint saves for long-running batch operations
7. Test with both OpenAI and non-OpenAI LLMs to ensure compatibility
