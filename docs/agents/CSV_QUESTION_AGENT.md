# QuestionAugmentationAgent Documentation

## Overview

The `QuestionAugmentationAgent` provides functionality to work with questions in CSV/DataFrame contexts, specifically for RAG (Retrieval-Augmented Generation) datasets. It supports two main features:

1. **Question Rephrasing**: Rephrase questions to align with source content and domain terminology
2. **Alternative Question Generation**: Generate multiple relevant questions based on source text

### What It Does

This agent addresses the need for an "Agent Generator for CSV" that can:
- **Contextualize Questions to Sources** - Rephrase questions using terminology from the source
- **Domain-Aware Question Rephrasing** - Adapt questions to specific domain contexts
- **Generate Alternative Questions** - Propose multiple questions answerable by a given source
- **Batch Processing** - Process entire CSV files or DataFrames efficiently

### Key Features

- **Source-Based Rephrasing**: Rephrase questions to match the terminology and context of associated sources
- **Domain-Based Rephrasing**: Adapt questions to specific domain vocabularies
- **Alternative Question Generation**: Create 'n' diverse questions from a single source
- **CSV/DataFrame Support**: Direct integration with pandas DataFrames and CSV files
- **Flexible Column Mapping**: Customize input/output column names
- **Robust Error Handling**: Graceful handling of LLM failures with retry logic

## Installation

The agent requires the following dependencies:

```bash
pip install langchain langchain-core pandas tqdm
```

For running with OpenAI:
```bash
pip install langchain-openai
```

For other LLM providers, install the appropriate langchain integration package.

## Basic Usage

### Initialize the Agent

```python
from langchain_openai import ChatOpenAI
from RAG_supporters.agents.question_augmentation_agent import QuestionAugmentationAgent

# Initialize with an LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
agent = QuestionAugmentationAgent(llm=llm, max_retries=3)
```

### Feature 1: Question Rephrasing

#### Rephrase with Source Context

```python
# Original generic question
question = "What does it do?"

# Detailed source text
source = """
Mitochondria are membrane-bound organelles found in most eukaryotic cells.
They generate most of the cell's supply of adenosine triphosphate (ATP),
which is used as a source of chemical energy. This process is called
cellular respiration.
"""

# Rephrase to align with source terminology
rephrased = agent.rephrase_question_with_source(question, source)
print(rephrased)
# Output: "What is the primary function of mitochondria in eukaryotic cells?"
```

#### Rephrase with Domain Context

```python
# Generic question
question = "How do you make the system learn patterns?"

# Specify domain
domain = "machine learning"

# Rephrase to fit domain
rephrased = agent.rephrase_question_with_domain(question, domain)
print(rephrased)
# Output: "How do you train a machine learning model to recognize patterns?"
```

### Feature 2: Alternative Question Generation

```python
source = """
Photosynthesis is a process used by plants and other organisms to convert
light energy into chemical energy. During photosynthesis, light energy is
captured and used to convert water, carbon dioxide, and minerals into
oxygen and energy-rich organic compounds.
"""

# Generate 5 alternative questions
questions = agent.generate_alternative_questions(source, n=5)

for i, q in enumerate(questions, 1):
    print(f"{i}. {q}")

# Output example:
# 1. What is photosynthesis?
# 2. What energy transformation occurs during photosynthesis?
# 3. What are the inputs required for photosynthesis?
# 4. What products are generated during photosynthesis?
# 5. How do plants convert light energy into chemical energy?
```

## CSV/DataFrame Processing

### Batch Question Rephrasing (Source-Based)

```python
import pandas as pd

# Your CSV should have columns: question_text, source_text
df = pd.read_csv("data/questions.csv")

# Rephrase all questions based on their sources
result_df = agent.process_dataframe_rephrasing(
    df,
    rephrase_mode="source",  # Use source text for context
)

# Result will have a new column: rephrased_question
result_df.to_csv("data/rephrased_questions.csv", index=False)
```

### Batch Question Rephrasing (Domain-Based)

```python
# Rephrase questions to fit a specific domain
result_df = agent.process_dataframe_rephrasing(
    df,
    rephrase_mode="domain",
    domain="biology",  # Apply biology terminology
)

# Save results
result_df.to_csv("data/biology_questions.csv", index=False)
```

### Generate Alternative Questions from Sources

```python
# Your CSV should have column: source_text
df = pd.read_csv("data/sources.csv")

# Generate 3 questions per source
result_df = agent.process_dataframe_generation(
    df,
    n_questions=3,  # Customize number of questions
)

# Result will have multiple rows per source, each with a generated question
result_df.to_csv("data/generated_questions.csv", index=False)
```

### Direct CSV Processing

The agent provides convenience methods for direct CSV file processing:

```python
# Rephrase questions in a CSV file
agent.process_csv_rephrasing(
    input_csv_path="data/input.csv",
    output_csv_path="data/rephrased.csv",
    rephrase_mode="source"
)

# Generate questions from sources in a CSV file
agent.process_csv_generation(
    input_csv_path="data/sources.csv",
    output_csv_path="data/generated.csv",
    n_questions=5
)
```

## Advanced Usage

### Custom Column Mapping

If your CSV uses different column names, specify a mapping:

```python
# Your CSV has: my_question, my_source columns
columns_mapping = {
    "question_text": "my_question",
    "source_text": "my_source",
    "rephrased_question": "new_question"  # Output column name
}

result_df = agent.process_dataframe_rephrasing(
    df,
    rephrase_mode="source",
    columns_mapping=columns_mapping
)
```

### Configuring Retry Logic

```python
# Initialize with custom retry settings
agent = QuestionAugmentationAgent(llm=llm, max_retries=5)
```

### Handling Large Datasets

```python
# Process in chunks to manage memory
chunk_size = 100
results = []

for chunk in pd.read_csv("large_file.csv", chunksize=chunk_size):
    processed_chunk = agent.process_dataframe_rephrasing(chunk)
    results.append(processed_chunk)

final_df = pd.concat(results, ignore_index=True)
final_df.to_csv("processed_large_file.csv", index=False)
```

## Complete Examples

### Example 1: Improving Question-Source Alignment

```python
from langchain_openai import ChatOpenAI
from RAG_supporters.agents.question_augmentation_agent import QuestionAugmentationAgent
import pandas as pd

# Initialize
llm = ChatOpenAI(model="gpt-4", temperature=0.5)
agent = QuestionAugmentationAgent(llm=llm)

# Sample data with generic questions
data = {
    "question_text": [
        "What is this?",
        "How does it work?",
        "When was it discovered?"
    ],
    "source_text": [
        "The Higgs boson is an elementary particle in the Standard Model of particle physics...",
        "CRISPR-Cas9 is a genome editing tool that uses RNA-guided nucleases...",
        "Penicillin was first discovered by Alexander Fleming in 1928..."
    ]
}
df = pd.DataFrame(data)

# Rephrase to align with sources
result = agent.process_dataframe_rephrasing(df, rephrase_mode="source")

print(result[["question_text", "rephrased_question"]])
# Output shows questions now use proper terminology from sources
```

### Example 2: Dataset Augmentation with Alternative Questions

```python
from langchain_openai import ChatOpenAI
from RAG_supporters.agents.question_augmentation_agent import QuestionAugmentationAgent
import pandas as pd

# Initialize
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.8)
agent = QuestionAugmentationAgent(llm=llm)

# Load sources
sources_df = pd.read_csv("scientific_articles.csv")

# Generate 10 questions per article
augmented_df = agent.process_dataframe_generation(
    sources_df,
    n_questions=10
)

print(f"Original sources: {len(sources_df)}")
print(f"Generated pairs: {len(augmented_df)}")
# From 100 sources -> 1000 question-source pairs
```

### Example 3: Domain-Specific Question Adaptation

```python
from langchain_openai import ChatOpenAI
from RAG_supporters.agents.question_augmentation_agent import QuestionAugmentationAgent
import pandas as pd

# Initialize
llm = ChatOpenAI(model="gpt-4", temperature=0.3)
agent = QuestionAugmentationAgent(llm=llm)

# Load general questions
df = pd.read_csv("general_questions.csv")

# Adapt to medical domain
medical_df = agent.process_dataframe_rephrasing(
    df,
    rephrase_mode="domain",
    domain="medical science"
)

# Adapt to computer science domain
cs_df = agent.process_dataframe_rephrasing(
    df,
    rephrase_mode="domain",
    domain="computer science"
)

medical_df.to_csv("medical_questions.csv", index=False)
cs_df.to_csv("cs_questions.csv", index=False)
```

## Expected CSV Format

### For Question Rephrasing (Source Mode)

```csv
question_text,source_text
"What does it do?","Mitochondria are organelles that produce ATP..."
"How does this work?","Neural networks consist of layers of nodes..."
```

### For Question Rephrasing (Domain Mode)

```csv
question_text
"What is this technology?"
"How do you train it?"
```

### For Alternative Question Generation

```csv
source_text
"Photosynthesis is the process by which plants convert light into energy..."
"The water cycle describes the continuous movement of water on Earth..."
```

## API Reference

### QuestionAugmentationAgent Class

#### Constructor

```python
QuestionAugmentationAgent(llm: BaseChatModel, max_retries: int = 3)
```

**Parameters:**
- `llm`: LangChain chat model instance
- `max_retries`: Maximum number of retry attempts for LLM calls (default: 3)

#### Methods

##### rephrase_question_with_source

```python
rephrase_question_with_source(question: str, source: str) -> Optional[str]
```

Rephrase a question based on source context.

**Parameters:**
- `question`: Original question to rephrase
- `source`: Source text providing context

**Returns:** Rephrased question or None if failed

##### rephrase_question_with_domain

```python
rephrase_question_with_domain(question: str, domain: str) -> Optional[str]
```

Rephrase a question for a specific domain.

**Parameters:**
- `question`: Original question to rephrase
- `domain`: Domain context (e.g., "biology", "physics")

**Returns:** Rephrased question or None if failed

##### generate_alternative_questions

```python
generate_alternative_questions(source: str, n: int = 5) -> Optional[List[str]]
```

Generate n alternative questions from a source.

**Parameters:**
- `source`: Source text to generate questions from
- `n`: Number of questions to generate (default: 5)

**Returns:** List of generated questions or None if failed

##### process_dataframe_rephrasing

```python
process_dataframe_rephrasing(
    df: pd.DataFrame,
    rephrase_mode: str = "source",
    domain: Optional[str] = None,
    columns_mapping: Optional[dict] = None
) -> pd.DataFrame
```

Process DataFrame by rephrasing questions.

**Parameters:**
- `df`: Input DataFrame
- `rephrase_mode`: "source" or "domain"
- `domain`: Required for domain mode
- `columns_mapping`: Custom column name mapping

**Returns:** DataFrame with rephrased questions column

##### process_dataframe_generation

```python
process_dataframe_generation(
    df: pd.DataFrame,
    n_questions: int = 5,
    columns_mapping: Optional[dict] = None
) -> pd.DataFrame
```

Generate alternative questions for each source in DataFrame.

**Parameters:**
- `df`: Input DataFrame with sources
- `n_questions`: Number of questions per source (default: 5)
- `columns_mapping`: Custom column name mapping

**Returns:** New DataFrame with generated question-source pairs

##### process_csv_rephrasing

```python
process_csv_rephrasing(
    input_csv_path: str,
    output_csv_path: str,
    rephrase_mode: str = "source",
    domain: Optional[str] = None,
    columns_mapping: Optional[dict] = None
) -> pd.DataFrame
```

Process CSV file by rephrasing questions.

##### process_csv_generation

```python
process_csv_generation(
    input_csv_path: str,
    output_csv_path: str,
    n_questions: int = 5,
    columns_mapping: Optional[dict] = None
) -> pd.DataFrame
```

Process CSV file by generating alternative questions.

## Best Practices

1. **Choose Appropriate LLM Temperature:**
   - Lower (0.3-0.5) for consistent rephrasing
   - Higher (0.7-0.9) for diverse alternative questions

2. **Handle Failures Gracefully:**
   - Check for None returns from methods
   - Use appropriate retry settings for unreliable networks

3. **Validate Generated Content:**
   - Review samples of generated questions
   - Ensure questions align with your use case

4. **Optimize for Cost:**
   - Use cheaper models (gpt-3.5-turbo) for initial experimentation
   - Switch to more capable models (gpt-4) for production

5. **Process in Batches:**
   - For large datasets, process in chunks to manage memory
   - Save intermediate results to prevent data loss

## Troubleshooting

### Common Issues

**Issue: All rephrasing returns None**
- Check LLM API credentials
- Verify network connectivity
- Increase max_retries

**Issue: Generated questions are repetitive**
- Increase LLM temperature
- Try a more capable model
- Reduce n_questions per source

**Issue: Column not found errors**
- Verify CSV column names match expected defaults
- Use columns_mapping to specify custom names

**Issue: Out of memory on large datasets**
- Process in smaller chunks
- Reduce batch size
- Use streaming CSV reading

## Logging

The agent uses Python's logging module. Enable logging to track progress:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## Performance Considerations

- **LLM Calls**: Each question requires 1 LLM call
- **Batch Processing**: Uses tqdm progress bars
- **Memory Usage**: Loads entire DataFrame into memory
- **Network Latency**: Dominant factor in processing time

## See Also

- **TextAugmentationAgent**: For general text rephrasing without source context
- **DomainAnalysisAgent**: For domain extraction and assessment
- **SourceEvaluationAgent**: For evaluating source quality
