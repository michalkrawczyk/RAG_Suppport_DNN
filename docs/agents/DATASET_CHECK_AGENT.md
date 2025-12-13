# DatasetCheckAgent Documentation

## Overview

The `DatasetCheckAgent` is a specialized agent for comparing and evaluating text sources in datasets. It uses LLM-based analysis to determine which of two sources is better for answering a given question, or if they are duplicates. This is particularly useful for quality control and dataset curation in RAG (Retrieval-Augmented Generation) applications.

### What It Does

The agent addresses the need for automated source comparison by:
- **Comparing Two Sources**: Analyzes which source is better for answering a specific question
- **Duplicate Detection**: Identifies when two sources are essentially the same
- **Quality Assessment**: Provides structured analysis of source relevance and quality
- **Batch Processing**: Efficiently processes entire datasets for source comparison

### Key Features

- **LLM-Powered Comparison**: Uses language models for intelligent source analysis
- **Structured Workflow**: Built on LangGraph for robust state management
- **Batch Processing**: Process entire DataFrames or CSV files
- **Checkpoint Support**: Resume processing from where you left off
- **Flexible Output**: Returns labels and optional detailed analysis
- **Error Handling**: Graceful handling of LLM failures

## Installation

The agent requires the following dependencies:

```bash
pip install langchain langgraph pandas tqdm
```

For running with specific LLM providers:
```bash
# For OpenAI
pip install langchain-openai

# For other providers, install appropriate langchain integration
```

## Basic Usage

### Initialize the Agent

```python
from langchain_openai import ChatOpenAI
from RAG_supporters.agents.dataset_check import DatasetCheckAgent

# Initialize with an LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.3)
agent = DatasetCheckAgent(llm=llm)
```

### Compare Two Sources

```python
question = "What is the capital of France?"

source1 = "Paris is the capital and most populous city of France."

source2 = "France is a country in Western Europe. Its capital is Paris, known for the Eiffel Tower."

# Compare sources
result = agent.compare_text_sources(
    question=question,
    source1=source1,
    source2=source2,
    return_analysis=True
)

print(f"Best source: {result['label']}")  # 1 for source1, 2 for source2, 0 for neither
print(f"Analysis: {result['analysis']}")
```

### Understanding Labels

The agent returns one of the following labels:
- **0**: Neither source is satisfactory (duplicates or both inadequate)
- **1**: Source 1 is better
- **2**: Source 2 is better
- **-1**: Error occurred during comparison

## DataFrame/CSV Processing

### Process a DataFrame

```python
import pandas as pd

# Your DataFrame should have columns: question_text, answer_text_1, answer_text_2, label
df = pd.read_csv("data/sources_to_compare.csv")

# Process and label all rows
result_df = agent.process_dataframe(
    df,
    save_path="data/labeled_sources.csv",
    skip_labeled=True,  # Skip rows that already have labels
    start_index=0
)

# Check results
print(f"Processed {len(result_df)} rows")
print(result_df[['question_text', 'label']].head())
```

### Process a CSV File

```python
# Direct CSV processing
agent.process_csv(
    csv_path="data/sources_to_compare.csv",
    skip_labeled=True,
    start_index=0
)
```

**Note**: The CSV file will be overwritten with the labeled results.

## Advanced Usage

### Get Detailed Analysis

```python
# Get both label and detailed reasoning
result = agent.compare_text_sources(
    question=question,
    source1=source1,
    source2=source2,
    return_analysis=True,  # Get detailed analysis
    return_messages=True   # Get full message history
)

print(f"Label: {result['label']}")
print(f"Analysis:\n{result['analysis']}")
print(f"Message history: {result['messages']}")
```

### Resume Processing from Checkpoint

```python
# If processing was interrupted, resume from a specific index
result_df = agent.process_dataframe(
    df,
    save_path="data/sources.csv",
    skip_labeled=True,
    start_index=100  # Resume from row 100
)
```

### Custom Comparison Prompts

```python
from prompts_templates.rag_verifiers import SRC_COMPARE_PROMPT_WITH_SCORES

# Initialize with custom prompt
agent = DatasetCheckAgent(
    llm=llm,
    compare_prompt=SRC_COMPARE_PROMPT_WITH_SCORES
)
```

## Expected Input Format

### For DataFrame Processing

The DataFrame should contain these columns:

- `question_text`: The question to evaluate sources against
- `answer_text_1`: First source text
- `answer_text_2`: Second source text
- `label` (optional): Existing label (will be updated if not skipped)

### Example CSV Format

```csv
question_text,answer_text_1,answer_text_2,label
"What is machine learning?","Machine learning is a type of AI...","ML is a subset of artificial intelligence...",-1
"How does photosynthesis work?","Photosynthesis uses sunlight...","Plants convert light energy...",-1
```

## Complete Examples

### Example 1: Basic Source Comparison

```python
from langchain_openai import ChatOpenAI
from RAG_supporters.agents.dataset_check import DatasetCheckAgent

# Initialize
llm = ChatOpenAI(model="gpt-4", temperature=0.3)
agent = DatasetCheckAgent(llm=llm)

# Define question and sources
question = "What is the process of photosynthesis?"

source1 = """
Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide
to produce oxygen and energy in the form of sugar. It occurs primarily in the leaves
of plants, specifically in chloroplasts containing chlorophyll.
"""

source2 = """
Plants make food through photosynthesis. They use light from the sun.
"""

# Compare sources
result = agent.compare_text_sources(
    question=question,
    source1=source1,
    source2=source2,
    return_analysis=True
)

if result['label'] == 1:
    print("Source 1 is better - more comprehensive and detailed")
elif result['label'] == 2:
    print("Source 2 is better")
elif result['label'] == 0:
    print("Neither source is satisfactory or they are duplicates")
else:
    print("Error in comparison")

print(f"\nDetailed Analysis:\n{result['analysis']}")
```

### Example 2: Dataset Quality Control

```python
import pandas as pd
from langchain_openai import ChatOpenAI
from RAG_supporters.agents.dataset_check import DatasetCheckAgent

# Create sample dataset
data = {
    'question_text': [
        "What is the Eiffel Tower?",
        "Who invented the telephone?",
        "What causes earthquakes?"
    ],
    'answer_text_1': [
        "The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
        "Alexander Graham Bell invented the telephone in 1876.",
        "Earthquakes are caused by tectonic plate movements."
    ],
    'answer_text_2': [
        "Located in Paris, the Eiffel Tower was built for the 1889 World's Fair.",
        "The telephone was invented by Alexander Graham Bell.",
        "Tectonic plates shifting cause earthquakes along fault lines."
    ],
    'label': [-1, -1, -1]
}

df = pd.DataFrame(data)

# Initialize agent
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
agent = DatasetCheckAgent(llm=llm)

# Process dataset
print("Processing dataset...")
result_df = agent.process_dataframe(
    df,
    save_path="labeled_dataset.csv"
)

# Analyze results
print("\n=== Results Summary ===")
print(f"Total rows: {len(result_df)}")
print(f"Source 1 better: {(result_df['label'] == 1).sum()}")
print(f"Source 2 better: {(result_df['label'] == 2).sum()}")
print(f"Neither/Duplicates: {(result_df['label'] == 0).sum()}")
print(f"Errors: {(result_df['label'] == -1).sum()}")

# Show detailed results
print("\n=== Detailed Results ===")
for idx, row in result_df.iterrows():
    label_desc = {
        0: "Neither/Duplicate",
        1: "Source 1",
        2: "Source 2",
        -1: "Error"
    }
    print(f"\nQuestion: {row['question_text'][:50]}...")
    print(f"Best source: {label_desc.get(row['label'], 'Unknown')}")
```

### Example 3: Resuming Interrupted Processing

```python
import pandas as pd
from langchain_openai import ChatOpenAI
from RAG_supporters.agents.dataset_check import DatasetCheckAgent

# Load large dataset
df = pd.read_csv("large_dataset.csv")
print(f"Total rows to process: {len(df)}")

# Initialize agent
llm = ChatOpenAI(model="gpt-4", temperature=0.3)
agent = DatasetCheckAgent(llm=llm)

try:
    # Process dataset with checkpoints
    result_df = agent.process_dataframe(
        df,
        save_path="large_dataset.csv",  # Overwrites with progress
        skip_labeled=True,  # Skip already labeled rows
        start_index=0
    )
    print("Processing complete!")
    
except KeyboardInterrupt:
    print("\nProcessing interrupted by user")
    print("Progress has been saved to large_dataset.csv")
    print("Resume processing by running again with skip_labeled=True")

# To resume later, just run again:
# result_df = agent.process_dataframe(
#     df,
#     save_path="large_dataset.csv",
#     skip_labeled=True  # Will skip already processed rows
# )
```

## Workflow Architecture

The agent uses LangGraph to manage a structured workflow:

1. **Source Check**: Analyzes both sources against the question
2. **Assign Label**: Determines which source is better based on the analysis

This two-step process ensures consistent and reliable comparisons.

### State Management

The agent maintains state through a `CheckAgentState` TypedDict containing:
- `messages`: Message history
- `question`: The question being analyzed
- `source1`: First source text
- `source2`: Second source text
- `analysis`: Analysis result
- `final_choice`: Final label (0, 1, 2, or -1)

## API Reference

### DatasetCheckAgent

```python
class DatasetCheckAgent:
    def __init__(
        self,
        llm: BaseChatModel,
        compare_prompt: str = SRC_COMPARE_PROMPT_WITH_SCORES
    )
```

**Parameters:**
- `llm`: Language model instance for performing analysis
- `compare_prompt`: Custom prompt template for comparison (optional)

#### Methods

##### compare_text_sources

```python
compare_text_sources(
    question: str,
    source1: str,
    source2: str,
    return_analysis: bool = False,
    return_messages: bool = False
) -> dict
```

Compare two text sources for a given question.

**Parameters:**
- `question`: The question to analyze sources against
- `source1`: First source text to compare
- `source2`: Second source text to compare
- `return_analysis`: Whether to return detailed analysis text
- `return_messages`: Whether to return message history

**Returns:** Dictionary containing:
- `label`: int (0, 1, 2, or -1)
- `analysis`: str or None
- `messages`: list or empty list

##### process_dataframe

```python
process_dataframe(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    skip_labeled: bool = True,
    start_index: int = 0
) -> pd.DataFrame
```

Process a DataFrame to compare sources and assign labels.

**Parameters:**
- `df`: DataFrame with columns: question_text, answer_text_1, answer_text_2, label
- `save_path`: Path to save results as CSV
- `skip_labeled`: Skip rows with existing labels (≠ -1)
- `start_index`: Index to start processing from

**Returns:** DataFrame with updated label column

##### process_csv

```python
process_csv(
    csv_path: str,
    skip_labeled: bool = True,
    start_index: int = 0
)
```

Process a CSV file directly (overwrites the file with results).

**Parameters:**
- `csv_path`: Path to CSV file to process
- `skip_labeled`: Skip rows with existing labels (≠ -1)
- `start_index`: Index to start processing from

## Best Practices

1. **Use Appropriate LLM**: GPT-4 provides better analysis than GPT-3.5-turbo
2. **Set Low Temperature**: Use temperature 0.2-0.3 for consistent comparisons
3. **Enable Checkpointing**: Always use `save_path` for large datasets
4. **Skip Labeled Rows**: Use `skip_labeled=True` to avoid reprocessing
5. **Monitor Progress**: Use tqdm progress bars to track processing
6. **Handle Interruptions**: Design for resumable processing with checkpoints

## Performance Considerations

- **Processing Speed**: ~1-3 seconds per comparison (depends on LLM)
- **API Costs**: Each comparison requires 2 LLM calls (analysis + verdict)
- **Memory Usage**: Loads entire DataFrame into memory
- **Batch Processing**: Progress bar with tqdm for large datasets

## Troubleshooting

### Common Issues

**Issue: All labels return -1**
- Check LLM API credentials
- Verify network connectivity
- Ensure question and sources are not empty

**Issue: Processing is too slow**
- Use faster LLM model (gpt-3.5-turbo instead of gpt-4)
- Process in smaller batches
- Consider parallel processing for very large datasets

**Issue: Inconsistent results**
- Lower the temperature (closer to 0)
- Use more capable LLM model
- Ensure questions are clear and specific

**Issue: KeyboardInterrupt not saving progress**
- Ensure `save_path` is specified
- Check file write permissions
- Verify disk space availability

## Integration with RAG Pipelines

The DatasetCheckAgent can be integrated into RAG data preparation workflows:

```python
from RAG_supporters.dataset.templates.rag_mini_bioasq import RagMiniBioASQBase
from RAG_supporters.agents.dataset_check import DatasetCheckAgent

# Generate candidate sources
dataset = RagMiniBioASQBase(dataset_dir="./data/bioasq")
candidates_df = dataset.generate_candidates()

# Use DatasetCheckAgent to select best sources
llm = ChatOpenAI(model="gpt-4", temperature=0.3)
agent = DatasetCheckAgent(llm=llm)

result_df = agent.process_dataframe(
    candidates_df,
    save_path="./data/bioasq/selected_sources.csv"
)

# Filter for best sources
best_sources = result_df[result_df['label'].isin([1, 2])]
print(f"Selected {len(best_sources)} best sources from {len(result_df)} candidates")
```

## Logging

The agent uses Python's logging module. Enable logging to track progress:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## See Also

- **SourceEvaluationAgent**: For evaluating individual source quality with detailed scores
- **DomainAnalysisAgent**: For extracting and assessing domain information
- **QuestionAugmentationAgent**: For generating and rephrasing questions
