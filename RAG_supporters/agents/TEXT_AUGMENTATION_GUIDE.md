# TextAugmentationAgent Guide

The `TextAugmentationAgent` provides functionality to generate alternative versions of questions and sources while preserving their original meaning. This is useful for data augmentation in RAG datasets.

## Features

- **Full Text Rephrasing**: Rephrase entire questions or sources
- **Sentence-Level Rephrasing**: Randomly select and rephrase individual sentences
- **Automatic Mode**: Randomly choose between full and sentence rephrasing
- **CSV Processing**: Batch process CSV files containing question-source pairs
- **Meaning Verification**: Optional verification that rephrased text preserves meaning
- **Flexible Configuration**: Customizable probability, modes, and column mappings

## Installation

The agent requires the following dependencies:

```bash
pip install langchain langchain-core pandas tqdm
```

For running the examples with OpenAI:
```bash
pip install langchain-openai
```

## Basic Usage

### Initialize the Agent

```python
from langchain_openai import ChatOpenAI
from agents.text_augmentation import TextAugmentationAgent

# Initialize with an LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
agent = TextAugmentationAgent(llm=llm, verify_meaning=False)
```

### Rephrase Full Text

```python
question = "What is the primary function of mitochondria?"
rephrased = agent.rephrase_full_text(question)
print(rephrased)
# Output: "What role do mitochondria primarily serve?"
```

### Rephrase Random Sentence

```python
source = (
    "Mitochondria are organelles found in cells. "
    "They produce ATP through oxidative phosphorylation. "
    "This occurs in the inner membrane."
)
rephrased = agent.rephrase_random_sentence(source)
print(rephrased)
# Output: One sentence will be rephrased while others remain unchanged
```

## CSV Processing

### Basic CSV Augmentation

```python
# Process a CSV file
augmented_df = agent.process_csv(
    input_csv_path="data/questions.csv",
    output_csv_path="data/augmented_questions.csv",
    rephrase_question=True,
    rephrase_source=True,
    rephrase_mode="random",  # Can be "full", "sentence", or "random"
    probability=0.5,  # Augment 50% of rows
)
```

### Using DataFrame Directly

```python
import pandas as pd

# Load your data
df = pd.read_csv("data/questions.csv")

# Augment the DataFrame
augmented_df = agent.augment_dataframe(
    df,
    rephrase_question=True,
    rephrase_source=True,
    rephrase_mode="full",
    probability=1.0,  # Augment all rows
)

# Save results
augmented_df.to_csv("data/augmented.csv", index=False)
```

## Configuration Options

### Rephrasing Modes

- **`"full"`**: Rephrase entire text
- **`"sentence"`**: Rephrase a randomly selected sentence
- **`"random"`**: Randomly choose between full and sentence mode for each row

### Custom Column Mapping

If your CSV uses different column names:

```python
augmented_df = agent.process_csv(
    input_csv_path="data/custom.csv",
    output_csv_path="data/augmented_custom.csv",
    columns_mapping={
        "question_text": "query",    # Your column name
        "source_text": "context",    # Your column name
    }
)
```

### Selective Augmentation

Control what to rephrase:

```python
# Rephrase only questions
augmented_df = agent.augment_dataframe(
    df,
    rephrase_question=True,
    rephrase_source=False,  # Don't rephrase sources
)

# Rephrase only sources
augmented_df = agent.augment_dataframe(
    df,
    rephrase_question=False,  # Don't rephrase questions
    rephrase_source=True,
)
```

### Probability Control

```python
# Augment 30% of rows
augmented_df = agent.augment_dataframe(
    df,
    probability=0.3
)

# Augment all rows
augmented_df = agent.augment_dataframe(
    df,
    probability=1.0
)
```

## Advanced Features

### Meaning Verification

Enable verification to ensure rephrased text preserves meaning:

```python
agent = TextAugmentationAgent(
    llm=llm,
    verify_meaning=True  # Enable verification
)

# Or verify specific rephrasing operations
rephrased = agent.rephrase_full_text(text, verify=True)
```

**Note**: Enabling verification doubles the LLM calls (one for rephrasing, one for verification).

### Retry Logic

Configure retry attempts for LLM failures:

```python
agent = TextAugmentationAgent(
    llm=llm,
    max_retries=5  # Retry up to 5 times on failure
)
```

## Expected Input Format

The agent expects CSV files or DataFrames with the following default columns:

- `question_text`: The question text to potentially rephrase
- `source_text`: The source/context text to potentially rephrase

Additional columns (like `question_id`, `source_id`, `answer`) are preserved in the output.

### Example Input CSV

```csv
question_id,question_text,source_text,answer
q1,"What causes photosynthesis?","Photosynthesis is driven by light energy.","Light energy"
q2,"How does DNA replicate?","DNA replication uses enzymes.","Through enzymes"
```

### Example Output CSV

The output will contain all original rows plus augmented versions:

```csv
question_id,question_text,source_text,answer
q1,"What causes photosynthesis?","Photosynthesis is driven by light energy.","Light energy"
q2,"How does DNA replicate?","DNA replication uses enzymes.","Through enzymes"
q1,"What triggers photosynthesis?","Light energy drives the process of photosynthesis.","Light energy"
q2,"What is the mechanism of DNA replication?","Enzymes are used in DNA replication.","Through enzymes"
```

## Integration with RAG Dataset

The agent can be used with existing RAG dataset workflows:

```python
from dataset.templates.rag_mini_bioasq import RagMiniBioASQBase
from agents.text_augmentation import TextAugmentationAgent

# Generate pairs from RAG dataset
dataset = RagMiniBioASQBase(
    dataset_dir="./data/bioasq",
    embed_function=your_embedding_function
)
pairs_df = dataset.generate_samples("pairs_relevant")

# Augment the pairs
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
agent = TextAugmentationAgent(llm=llm)
augmented_pairs = agent.augment_dataframe(
    pairs_df,
    rephrase_question=True,
    rephrase_source=True,
    rephrase_mode="random",
    probability=0.5,
)

# Save augmented dataset
augmented_pairs.to_csv("./data/bioasq/augmented_pairs.csv", index=False)
```

## Best Practices

1. **Use Appropriate Temperature**: Set LLM temperature to 0.7-0.9 for diverse rephrasing
2. **Start with Low Probability**: Test with `probability=0.1` before augmenting entire dataset
3. **Verify Critical Data**: Enable `verify_meaning=True` for important datasets
4. **Monitor Costs**: Each row requires 1-2 LLM calls (2 if verification enabled)
5. **Use Random Mode**: `rephrase_mode="random"` provides good variety
6. **Preserve IDs**: The agent preserves all existing columns including IDs

## Performance Considerations

- **LLM Calls**: Each augmented row requires 1-2 LLM calls
- **Processing Time**: Depends on LLM speed and dataset size
- **Cost**: Consider API costs for large datasets
- **Batch Processing**: Process in smaller batches for very large datasets

## Troubleshooting

### No Rows Augmented

Check that:
- `probability` is > 0
- Input texts are not empty
- LLM is responding correctly
- Columns exist in DataFrame

### Rephrasing Failures

If rephrasing fails frequently:
- Increase `max_retries`
- Check LLM connection and API key
- Verify text length is within LLM limits
- Check for malformed input text

### Verification Always Fails

If meaning verification always fails:
- Disable verification initially to test rephrasing
- Check that verification prompt is appropriate for your LLM
- Try different LLM models

## Examples

See `examples/text_augmentation_example.py` for complete working examples.

## API Reference

### TextAugmentationAgent

```python
class TextAugmentationAgent:
    def __init__(
        self,
        llm: BaseChatModel,
        verify_meaning: bool = False,
        max_retries: int = 3,
    )
```

#### Methods

- `rephrase_full_text(text: str, verify: Optional[bool] = None) -> Optional[str]`
- `rephrase_random_sentence(text: str, verify: Optional[bool] = None) -> Optional[str]`
- `augment_dataframe(df, ...) -> pd.DataFrame`
- `process_csv(input_csv_path, ...) -> pd.DataFrame`

See docstrings in the code for detailed parameter descriptions.
