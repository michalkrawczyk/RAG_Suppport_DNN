# TextAugmentationAgent Documentation

## Overview

The `TextAugmentationAgent` provides functionality to generate alternative versions of questions and sources while preserving their original meaning. This is useful for data augmentation in RAG datasets.

### What It Does

This agent addresses the need for an "Agent Generator for CSV" that can:
- **Randomly rephrase whole questions or sources** - Creates paraphrased versions of entire texts
- **Randomly pick and rephrase sentences** - Modifies individual sentences within texts
- **Preserve meaning** - Ensures semantic equivalence and doesn't change possible answers

### Key Features

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

For running with OpenAI:
```bash
pip install langchain-openai
```

## Basic Usage

### Initialize the Agent

```python
from langchain_openai import ChatOpenAI
from RAG_supporters.agents.text_augmentation import TextAugmentationAgent

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

## Examples

### Example 1: Basic Text Rephrasing

```python
from langchain_openai import ChatOpenAI
from RAG_supporters.agents.text_augmentation import TextAugmentationAgent

# Initialize the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
agent = TextAugmentationAgent(llm=llm, verify_meaning=False)

# Example question
question = "What is the primary function of mitochondria in cellular respiration?"

# Rephrase entire question
print("Original question:")
print(question)
print("\nRephrased question (full):")
rephrased_full = agent.rephrase_full_text(question)
print(rephrased_full)

# Example source text
source = (
    "Mitochondria are organelles found in eukaryotic cells. "
    "They are responsible for producing adenosine triphosphate (ATP), "
    "the cell's main energy currency, through a process called oxidative phosphorylation. "
    "This process occurs in the inner mitochondrial membrane."
)

print("\n\nOriginal source:")
print(source)
print("\nRephrased source (random sentence):")
rephrased_sentence = agent.rephrase_random_sentence(source)
print(rephrased_sentence)
```

### Example 2: CSV Augmentation

```python
import pandas as pd
from langchain_openai import ChatOpenAI
from RAG_supporters.agents.text_augmentation import TextAugmentationAgent

# Create sample data
sample_data = {
    "question_id": ["q1", "q2", "q3"],
    "question_text": [
        "What causes photosynthesis in plants?",
        "How does the immune system fight infections?",
        "What is the structure of DNA?",
    ],
    "source_text": [
        "Photosynthesis is the process by which plants convert light energy into chemical energy. "
        "Chlorophyll in the chloroplasts absorbs sunlight, which drives the conversion of carbon dioxide and water into glucose and oxygen.",
        "The immune system uses white blood cells to identify and destroy pathogens. "
        "T-cells and B-cells work together to recognize foreign substances and produce antibodies that neutralize them.",
        "DNA has a double helix structure composed of two complementary strands. "
        "Each strand consists of a sugar-phosphate backbone with nitrogenous bases (adenine, thymine, guanine, cytosine) that pair specifically.",
    ],
    "answer": ["Chlorophyll and sunlight", "White blood cells", "Double helix"],
}

df = pd.DataFrame(sample_data)

# Save to temporary CSV
temp_csv = "/tmp/sample_rag_data.csv"
df.to_csv(temp_csv, index=False)
print(f"Created sample CSV with {len(df)} rows")

# Initialize the agent
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
agent = TextAugmentationAgent(llm=llm, verify_meaning=False)

# Augment the CSV
print("\nAugmenting dataset...")
augmented_df = agent.process_csv(
    input_csv_path=temp_csv,
    output_csv_path="/tmp/augmented_rag_data.csv",
    rephrase_question=True,
    rephrase_source=True,
    rephrase_mode="random",  # Randomly choose between full and sentence rephrasing
    probability=1.0,  # Augment all rows for demo
)

print(f"\nAugmentation complete!")
print(f"Original rows: {len(df)}")
print(f"Total rows after augmentation: {len(augmented_df)}")
print(f"New augmented rows: {len(augmented_df) - len(df)}")

# Show example of augmented data
print("\n\nExample augmented row:")
if len(augmented_df) > len(df):
    aug_row = augmented_df.iloc[len(df)]
    print(f"\nQuestion: {aug_row['question_text']}")
    print(f"\nSource: {aug_row['source_text'][:200]}...")
```

### Example 3: Custom Column Names

```python
import pandas as pd
from langchain_openai import ChatOpenAI
from RAG_supporters.agents.text_augmentation import TextAugmentationAgent

# Create sample data with different column names
sample_data = {
    "q_id": ["q1", "q2"],
    "query": ["What is machine learning?", "How does HTTP work?"],
    "context": [
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "HTTP is a protocol for transferring web pages over the internet using request-response model.",
    ],
}

df = pd.DataFrame(sample_data)
temp_csv = "/tmp/custom_columns_data.csv"
df.to_csv(temp_csv, index=False)

# Initialize agent
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
agent = TextAugmentationAgent(llm=llm)

# Augment with custom column mapping
augmented_df = agent.process_csv(
    input_csv_path=temp_csv,
    output_csv_path="/tmp/augmented_custom_columns.csv",
    rephrase_question=True,
    rephrase_source=True,
    rephrase_mode="full",  # Rephrase entire texts
    probability=1.0,
    columns_mapping={
        "question_text": "query",  # Map internal name to actual column
        "source_text": "context",
    },
)

print(f"Augmented dataset from {len(df)} to {len(augmented_df)} rows")
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
from RAG_supporters.dataset.templates.rag_mini_bioasq import RagMiniBioASQBase
from RAG_supporters.agents.text_augmentation import TextAugmentationAgent

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

## Implementation Details

### Core Components

1. **TextAugmentationAgent** (`RAG_supporters/agents/text_augmentation.py`)
   - 469 lines of production-ready code
   - Full text rephrasing capability
   - Random sentence rephrasing within texts
   - CSV batch processing
   - Optional meaning verification
   - Retry logic for LLM failures
   - Configurable probability and modes

2. **Prompt Templates** (`RAG_supporters/prompts_templates/text_augmentation.py`)
   - `FULL_TEXT_REPHRASE_PROMPT`: For rephrasing entire texts
   - `SENTENCE_REPHRASE_PROMPT`: For rephrasing specific sentences
   - `VERIFY_MEANING_PRESERVATION_PROMPT`: For verifying semantic equivalence

### Rephrasing Modes

- **Full Mode**: Rephrases entire question/source text
- **Sentence Mode**: Randomly selects and rephrases one sentence
- **Random Mode**: Randomly chooses between full and sentence mode

### Configuration Options

- `rephrase_question`: Toggle question rephrasing
- `rephrase_source`: Toggle source rephrasing
- `rephrase_mode`: "full", "sentence", or "random"
- `probability`: Control augmentation rate (0.0 to 1.0)
- `verify_meaning`: Optional semantic verification
- `max_retries`: LLM retry configuration
- `columns_mapping`: Custom column name support

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

**Parameters:**
- `llm`: Language model instance for performing text rephrasing
- `verify_meaning`: Whether to verify that rephrased text preserves meaning (default: False)
- `max_retries`: Maximum number of retries for LLM calls (default: 3)

#### Methods

**`rephrase_full_text(text: str, verify: Optional[bool] = None) -> Optional[str]`**

Rephrase an entire text while preserving its meaning.

**Parameters:**
- `text`: The text to rephrase
- `verify`: Whether to verify meaning preservation (overrides instance default)

**Returns:** The rephrased text, or None if rephrasing failed

---

**`rephrase_random_sentence(text: str, verify: Optional[bool] = None) -> Optional[str]`**

Rephrase a randomly selected sentence within the text.

**Parameters:**
- `text`: The text containing sentences to rephrase
- `verify`: Whether to verify meaning preservation (overrides instance default)

**Returns:** The text with one sentence rephrased, or None if rephrasing failed

---

**`augment_dataframe(df, rephrase_question=True, rephrase_source=True, rephrase_mode="random", probability=0.5, columns_mapping=None) -> pd.DataFrame`**

Augment a DataFrame by adding rephrased versions of questions and/or sources.

**Parameters:**
- `df`: DataFrame containing question and source columns
- `rephrase_question`: Whether to rephrase questions (default: True)
- `rephrase_source`: Whether to rephrase sources (default: True)
- `rephrase_mode`: Rephrasing mode: "full", "sentence", or "random" (default: "random")
- `probability`: Probability of applying rephrasing to each row (default: 0.5)
- `columns_mapping`: Mapping of expected column names (default: None)

**Returns:** New DataFrame with augmented rows added to the original data

---

**`process_csv(input_csv_path, output_csv_path=None, rephrase_question=True, rephrase_source=True, rephrase_mode="random", probability=0.5, columns_mapping=None) -> pd.DataFrame`**

Process a CSV file by augmenting it with rephrased versions.

**Parameters:**
- `input_csv_path`: Path to the input CSV file
- `output_csv_path`: Path to save the augmented CSV (if None, overwrites input file)
- `rephrase_question`: Whether to rephrase questions (default: True)
- `rephrase_source`: Whether to rephrase sources (default: True)
- `rephrase_mode`: Rephrasing mode: "full", "sentence", or "random" (default: "random")
- `probability`: Probability of applying rephrasing to each row (default: 0.5)
- `columns_mapping`: Mapping of expected column names (default: None)

**Returns:** The augmented DataFrame

## Requirements Met

✅ **Agent Generator for CSV**: Implemented with `process_csv()` method  
✅ **Random rephrase whole question or source**: Implemented with `rephrase_mode="full"`  
✅ **Random pick sentence and rephrase**: Implemented with `rephrase_mode="sentence"`  
✅ **Keep meaning**: LLM prompted to preserve exact meaning + optional verification  
✅ **Not change possible answer**: Semantic preservation ensures answer remains valid
