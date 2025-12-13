# SourceEvaluationAgent Documentation

## Overview

The `SourceEvaluationAgent` is a sophisticated LangGraph-based agent for evaluating source quality across multiple dimensions. It provides detailed, multi-dimensional scoring of how well a source answers a given question, making it invaluable for quality assessment, source ranking, and dataset curation in RAG applications.

### What It Does

The agent evaluates sources across six key dimensions:
- **Relevance**: How well the source addresses the question
- **Expertise/Authority**: Credibility and authoritativeness of the source
- **Depth and Specificity**: Level of detail and precision
- **Clarity and Conciseness**: Readability and presentation quality
- **Objectivity/Bias**: Neutrality and balance of the content
- **Completeness**: Comprehensiveness in covering the topic

### Key Features

- **Multi-Dimensional Scoring**: Six evaluation dimensions with 0-10 scores
- **Structured Output**: Validated Pydantic models with scores and reasoning
- **Batch Processing**: Efficient batch evaluation for OpenAI models
- **Automatic Retry Logic**: Built-in retry with OutputFixingParser
- **DataFrame Integration**: Process entire datasets with progress tracking
- **Checkpoint Support**: Resume processing for large datasets
- **Flexible Options**: Include/exclude detailed reasoning

## Installation

The agent requires the following dependencies:

```bash
pip install langchain langgraph pydantic pandas tqdm
```

For specific LLM providers:
```bash
# For OpenAI
pip install langchain-openai

# Install all agent dependencies
pip install -r RAG_supporters/requirements_agents.txt
```

## Evaluation Dimensions

### 1. Relevance (0-10)
How directly the source addresses the question. Higher scores indicate more direct relevance.

### 2. Expertise/Authority (0-10)
Credibility and authoritativeness. Considers citations, expert terminology, and depth of knowledge.

### 3. Depth and Specificity (0-10)
Level of detail and precision. Higher scores for specific, detailed information.

### 4. Clarity and Conciseness (0-10)
Readability and clear presentation. Balances clarity with completeness.

### 5. Objectivity/Bias (0-10)
Neutrality and balance. Higher scores for unbiased, fact-based content.

### 6. Completeness (0-10)
Comprehensiveness in covering the topic. Considers whether all aspects are addressed.

## Basic Usage

### Initialize the Agent

```python
from langchain_openai import ChatOpenAI
from RAG_supporters.agents.source_assesment import SourceEvaluationAgent

# Initialize with an LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.3)
agent = SourceEvaluationAgent(llm=llm, max_retries=3)
```

### Evaluate a Single Source

```python
question = "What is photosynthesis?"

source = """
Photosynthesis is the process by which plants, algae, and some bacteria 
convert light energy into chemical energy. During photosynthesis, organisms 
use carbon dioxide and water to produce glucose and oxygen. This process 
occurs primarily in chloroplasts and involves two main stages: the light-dependent 
reactions and the Calvin cycle.
"""

# Evaluate the source
result = agent.evaluate(question, source)

if result:
    print(f"Inferred Domain: {result['inferred_domain']}")
    print(f"\nScores:")
    for dimension, score in result['scores'].items():
        print(f"  {dimension}: {score}/10")
    print(f"\nScore Summary:\n{result['score_summary']}")
```

**Output:**
```
Inferred Domain: Biology / Plant Science

Scores:
  relevance: 9/10
  expertise_authority: 8/10
  depth_specificity: 7/10
  clarity_conciseness: 8/10
  objectivity_bias: 9/10
  completeness: 7/10

Score Summary:
- Relevance: 9/10
- Expertise/Authority: 8/10
- Depth and Specificity: 7/10
- Clarity and Conciseness: 8/10
- Objectivity/Bias: 9/10
- Completeness: 7/10
```

### Include Detailed Reasoning

To understand why each score was assigned, you can access the reasoning:

```python
result = agent.evaluate(question, source)

if result:
    print(f"Relevance Score: {result['scores']['relevance']}/10")
    print(f"Reasoning: {result['reasoning']['relevance']}")
```

## Batch Processing

### Evaluate Multiple Sources

```python
questions = [
    "What is machine learning?",
    "How does DNA replication work?",
    "What causes earthquakes?"
]

sources = [
    "Machine learning is a subset of AI that enables systems to learn from data...",
    "DNA replication is the process by which DNA makes a copy of itself...",
    "Earthquakes occur when tectonic plates shift along fault lines..."
]

# Batch evaluate (faster for OpenAI models)
results = agent.evaluate_batch(questions, sources)

for i, result in enumerate(results):
    if result:
        avg_score = sum(result['scores'].values()) / len(result['scores'])
        print(f"\nQuestion {i+1}: Average score = {avg_score:.1f}/10")
```

## DataFrame/CSV Processing

### Process DataFrame with Scores Only

```python
import pandas as pd

# Your DataFrame should have: question_text, source_text columns
df = pd.read_csv("qa_pairs.csv")

# Evaluate all sources
result_df = agent.process_dataframe(
    df,
    question_col="question_text",
    source_col="source_text",
    include_reasoning=False,  # Scores only, no reasoning text
    save_path="evaluated_sources.csv"
)

# Check results
print(result_df[['question_text', 'relevance_score', 'completeness_score']].head())
```

### Process DataFrame with Reasoning

```python
# Include detailed reasoning for each score
result_df = agent.process_dataframe(
    df,
    question_col="question_text",
    source_col="source_text",
    include_reasoning=True,  # Include reasoning text
    save_path="evaluated_sources_detailed.csv"
)

# Check reasoning
print(result_df[['question_text', 'relevance_score', 'relevance_reasoning']].head())
```

### Advanced Processing Options

```python
# Process with all advanced options
result_df = agent.process_dataframe(
    df,
    question_col="question_text",
    source_col="source_text",
    include_reasoning=False,
    progress_bar=True,  # Show progress
    save_path="results.csv",
    skip_existing=True,  # Skip already evaluated rows
    checkpoint_batch_size=100,  # Save every 100 rows
    use_batch_processing=True,  # Use batch API for OpenAI
    batch_size=20  # Process 20 at a time
)
```

## Complete Examples

### Example 1: Source Quality Assessment

```python
from langchain_openai import ChatOpenAI
from RAG_supporters.agents.source_assesment import SourceEvaluationAgent

# Initialize
llm = ChatOpenAI(model="gpt-4", temperature=0.3)
agent = SourceEvaluationAgent(llm=llm)

# Question and candidate sources
question = "What are the causes of climate change?"

source_high_quality = """
Climate change is primarily driven by human activities that increase greenhouse 
gas concentrations in the atmosphere. The main causes include: (1) Burning fossil 
fuels for energy, which releases CO2; (2) Deforestation, which reduces CO2 absorption; 
(3) Industrial processes and agriculture, which emit methane and nitrous oxide. 
Scientific consensus, supported by extensive peer-reviewed research, confirms that 
these anthropogenic factors are the dominant drivers of current climate change.
"""

source_low_quality = """
Climate change happens because of pollution and stuff. People drive cars and 
that's bad for the environment. Also trees are good but we cut them down.
"""

# Evaluate both sources
result_hq = agent.evaluate(question, source_high_quality)
result_lq = agent.evaluate(question, source_low_quality)

print("High Quality Source:")
print(f"  Average Score: {sum(result_hq['scores'].values()) / 6:.1f}/10")
print(f"  Relevance: {result_hq['scores']['relevance']}/10")
print(f"  Expertise: {result_hq['scores']['expertise_authority']}/10")

print("\nLow Quality Source:")
print(f"  Average Score: {sum(result_lq['scores'].values()) / 6:.1f}/10")
print(f"  Relevance: {result_lq['scores']['relevance']}/10")
print(f"  Expertise: {result_lq['scores']['expertise_authority']}/10")
```

### Example 2: Dataset Quality Control

```python
import pandas as pd
from langchain_openai import ChatOpenAI
from RAG_supporters.agents.source_assesment import SourceEvaluationAgent

# Initialize agent
llm = ChatOpenAI(model="gpt-4", temperature=0.3)
agent = SourceEvaluationAgent(llm=llm, batch_size=20)

# Load dataset
df = pd.read_csv("rag_dataset.csv")
print(f"Total pairs to evaluate: {len(df)}")

# Evaluate all sources
result_df = agent.process_dataframe(
    df,
    question_col="question_text",
    source_col="source_text",
    include_reasoning=False,
    save_path="evaluated_dataset.csv",
    use_batch_processing=True,
    batch_size=20,
    progress_bar=True
)

# Calculate quality metrics
result_df['avg_score'] = result_df[[
    'relevance_score',
    'expertise_authority_score',
    'depth_specificity_score',
    'clarity_conciseness_score',
    'objectivity_bias_score',
    'completeness_score'
]].mean(axis=1)

# Filter high-quality pairs
high_quality = result_df[result_df['avg_score'] >= 7.0]
medium_quality = result_df[(result_df['avg_score'] >= 5.0) & (result_df['avg_score'] < 7.0)]
low_quality = result_df[result_df['avg_score'] < 5.0]

print(f"\n=== Quality Distribution ===")
print(f"High quality (≥7.0): {len(high_quality)} ({len(high_quality)/len(df)*100:.1f}%)")
print(f"Medium quality (5.0-6.9): {len(medium_quality)} ({len(medium_quality)/len(df)*100:.1f}%)")
print(f"Low quality (<5.0): {len(low_quality)} ({len(low_quality)/len(df)*100:.1f}%)")

# Save filtered datasets
high_quality.to_csv("high_quality_pairs.csv", index=False)
print(f"\nSaved {len(high_quality)} high-quality pairs")
```

### Example 3: Source Ranking and Selection

```python
import pandas as pd
from langchain_openai import ChatOpenAI
from RAG_supporters.agents.source_assesment import SourceEvaluationAgent

# Initialize
llm = ChatOpenAI(model="gpt-4", temperature=0.3)
agent = SourceEvaluationAgent(llm=llm)

# Question with multiple candidate sources
question = "How do neural networks learn?"

candidates = pd.DataFrame({
    'source_id': ['s1', 's2', 's3'],
    'source_text': [
        "Neural networks learn through backpropagation, adjusting weights based on error gradients...",
        "Neural nets learn from examples. They adjust their parameters to minimize errors...",
        "Deep learning models use gradient descent to optimize their internal representations..."
    ]
})

# Create full dataframe for evaluation
eval_df = pd.DataFrame({
    'question_text': [question] * len(candidates),
    'source_text': candidates['source_text'],
    'source_id': candidates['source_id']
})

# Evaluate all candidates
result_df = agent.process_dataframe(
    eval_df,
    question_col="question_text",
    source_col="source_text",
    include_reasoning=False
)

# Calculate overall scores
result_df['overall_score'] = (
    result_df['relevance_score'] * 0.3 +  # Weight relevance higher
    result_df['expertise_authority_score'] * 0.2 +
    result_df['depth_specificity_score'] * 0.2 +
    result_df['clarity_conciseness_score'] * 0.15 +
    result_df['objectivity_bias_score'] * 0.05 +
    result_df['completeness_score'] * 0.1
)

# Rank sources
ranked = result_df.sort_values('overall_score', ascending=False)

print("=== Source Ranking ===")
for idx, row in ranked.iterrows():
    print(f"\nRank: {idx + 1}")
    print(f"Source ID: {row['source_id']}")
    print(f"Overall Score: {row['overall_score']:.2f}/10")
    print(f"Text: {row['source_text'][:60]}...")

# Select best source
best_source = ranked.iloc[0]
print(f"\n=== Best Source ===")
print(f"ID: {best_source['source_id']}")
print(f"Score: {best_source['overall_score']:.2f}/10")
```

### Example 4: Resumable Large Dataset Processing

```python
import pandas as pd
from langchain_openai import ChatOpenAI
from RAG_supporters.agents.source_assesment import SourceEvaluationAgent

# Initialize
llm = ChatOpenAI(model="gpt-4", temperature=0.3)
agent = SourceEvaluationAgent(llm=llm, batch_size=20)

# Load large dataset
df = pd.read_csv("large_dataset.csv")
print(f"Total rows: {len(df)}")

# Process with checkpoints
try:
    result_df = agent.process_dataframe(
        df,
        question_col="question_text",
        source_col="source_text",
        include_reasoning=False,
        save_path="large_dataset_evaluated.csv",
        skip_existing=True,  # Skip already evaluated
        checkpoint_batch_size=100,  # Save every 100 rows
        use_batch_processing=True,
        batch_size=20,
        progress_bar=True
    )
    print("Processing complete!")
    
except KeyboardInterrupt:
    print("\nProcessing interrupted")
    print("Progress saved. Resume by running again with skip_existing=True")

# To resume, just run the same code again
# The agent will skip rows that already have scores
```

## Output Format

### Without Reasoning

When `include_reasoning=False`, the following columns are added:
- `inferred_domain`: Identified domain for the question-source pair
- `relevance_score`: Score 0-10
- `expertise_authority_score`: Score 0-10
- `depth_specificity_score`: Score 0-10
- `clarity_conciseness_score`: Score 0-10
- `objectivity_bias_score`: Score 0-10
- `completeness_score`: Score 0-10
- `evaluation_error`: Error message if evaluation failed

### With Reasoning

When `include_reasoning=True`, additional columns are added:
- `relevance_reasoning`: Explanation for relevance score
- `expertise_authority_reasoning`: Explanation for expertise score
- `depth_specificity_reasoning`: Explanation for depth score
- `clarity_conciseness_reasoning`: Explanation for clarity score
- `objectivity_bias_reasoning`: Explanation for objectivity score
- `completeness_reasoning`: Explanation for completeness score

## Workflow Architecture

The agent uses LangGraph for robust evaluation:

1. **Evaluate Source**: Sends prompt to LLM for multi-dimensional evaluation
2. **Validate Response**: Validates against Pydantic SourceEvaluation model
3. **Retry on Error**: Automatically retries up to max_retries times

### State Management

The agent maintains state through `AgentState` containing:
- `question`: Question text
- `source_content`: Source text to evaluate
- `evaluation`: Parsed SourceEvaluation result
- `error`: Error message if any
- `retry_count`: Current retry count
- `max_retries`: Maximum retry attempts

### Pydantic Models

**SourceEvaluation Model:**
- `inferred_domain`: str
- `relevance`: ScoreRange (score + reasoning)
- `expertise_authority`: ScoreRange
- `depth_specificity`: ScoreRange
- `clarity_conciseness`: ScoreRange
- `objectivity_bias`: ScoreRange
- `completeness`: ScoreRange

**ScoreRange Model:**
- `score`: int (0-10)
- `reasoning`: Optional[str]

## API Reference

### SourceEvaluationAgent

```python
class SourceEvaluationAgent:
    def __init__(
        self,
        llm: BaseChatModel = None,
        max_retries: int = 3,
        evaluation_prompt: str = SINGLE_SRC_SCORE_PROMPT,
        batch_size: int = 10
    )
```

**Parameters:**
- `llm`: Language model for evaluation
- `max_retries`: Maximum retries for parsing errors (default: 3)
- `evaluation_prompt`: Custom evaluation prompt (optional)
- `batch_size`: Default batch size for batch processing (default: 10)

#### Single Evaluation

##### evaluate

```python
evaluate(
    question: str,
    source_content: str
) -> Optional[Dict[str, Any]]
```

Evaluate a source for a given question.

**Returns:** Dictionary with:
- `inferred_domain`: str
- `scores`: Dict with all six dimension scores (0-10)
- `reasoning`: Dict with reasoning for each dimension
- `score_summary`: Formatted string with all scores

#### Batch Evaluation

##### evaluate_batch

```python
evaluate_batch(
    questions: List[str],
    source_contents: List[str]
) -> List[Optional[Dict[str, Any]]]
```

Evaluate multiple question-source pairs in batch.

**Note:** Only available for OpenAI LLMs. Falls back to sequential for others.

**Returns:** List of evaluation results

#### DataFrame Processing

##### process_dataframe

```python
process_dataframe(
    df: pd.DataFrame,
    question_col: str = "question_text",
    source_col: str = "source_text",
    include_reasoning: bool = False,
    progress_bar: bool = True,
    save_path: Optional[str] = None,
    skip_existing: bool = True,
    checkpoint_batch_size: Optional[int] = None,
    use_batch_processing: bool = True,
    batch_size: Optional[int] = None
) -> pd.DataFrame
```

Process entire DataFrame with evaluation scores.

**Parameters:**
- `df`: DataFrame containing questions and sources
- `question_col`: Column name for questions (default: "question_text")
- `source_col`: Column name for sources (default: "source_text")
- `include_reasoning`: Include reasoning text (default: False)
- `progress_bar`: Show progress bar (default: True)
- `save_path`: Path to save results as CSV
- `skip_existing`: Skip rows with existing scores (default: True)
- `checkpoint_batch_size`: Save checkpoint every N rows
- `use_batch_processing`: Use batch API for OpenAI (default: True)
- `batch_size`: Batch size for processing

**Returns:** DataFrame with added evaluation columns

## Best Practices

1. **Use GPT-4 for Evaluation**: More reliable scoring than GPT-3.5-turbo
2. **Set Low Temperature**: Use 0.2-0.3 for consistent evaluations
3. **Enable Batch Processing**: Much faster for OpenAI models
4. **Include Reasoning Selectively**: Only when needed (doubles output size)
5. **Use Checkpoints**: Essential for large datasets
6. **Skip Existing Results**: Avoid reprocessing with skip_existing=True
7. **Monitor Costs**: Each evaluation requires 1-2 LLM calls
8. **Calculate Weighted Scores**: Weight dimensions by importance for your use case

## Performance Considerations

- **Single Evaluation**: ~2-3 seconds per item
- **Batch Processing (OpenAI)**: ~0.3-0.5 seconds per item
- **Memory Usage**: Loads entire DataFrame into memory
- **API Costs**: Each evaluation requires 1 LLM call + retries if needed
- **Reasoning Impact**: Including reasoning doesn't increase LLM calls, just output size

## Troubleshooting

### Common Issues

**Issue: All evaluations return None**
- Check LLM API credentials and network
- Verify questions and sources are not empty
- Check max_retries setting
- Ensure LLM model supports structured output

**Issue: Parsing errors despite retries**
- Use GPT-4 instead of GPT-3.5-turbo
- Increase max_retries
- Check custom evaluation_prompt format if using one

**Issue: Slow processing**
- Enable batch processing: `use_batch_processing=True`
- Increase batch_size (try 20-50 for OpenAI)
- Use gpt-3.5-turbo instead of gpt-4

**Issue: Inconsistent scores**
- Lower temperature (0.2-0.3)
- Use more capable model (GPT-4)
- Ensure questions are clear and specific

**Issue: Out of memory on large datasets**
- Process in chunks
- Reduce batch_size
- Don't include reasoning unless necessary

## Integration Examples

### With DatasetCheckAgent

```python
from RAG_supporters.agents.dataset_check import DatasetCheckAgent
from RAG_supporters.agents.source_assesment import SourceEvaluationAgent

# First, check which source is better
check_agent = DatasetCheckAgent(llm=llm)
check_result = check_agent.compare_text_sources(question, source1, source2)

# Then, evaluate the selected source in detail
eval_agent = SourceEvaluationAgent(llm=llm)
if check_result['label'] == 1:
    detailed_eval = eval_agent.evaluate(question, source1)
else:
    detailed_eval = eval_agent.evaluate(question, source2)
```

### With DomainAnalysisAgent

```python
from RAG_supporters.agents.domain_assesment import DomainAnalysisAgent
from RAG_supporters.agents.source_assesment import SourceEvaluationAgent

# Extract domain from source
domain_agent = DomainAnalysisAgent(llm=llm)
domain_result = domain_agent.extract_domains(source)

# Evaluate source quality
eval_agent = SourceEvaluationAgent(llm=llm)
eval_result = eval_agent.evaluate(question, source)

# Combine insights
print(f"Domain: {domain_result['primary_theme']}")
print(f"Quality: {sum(eval_result['scores'].values()) / 6:.1f}/10")
```

## Logging

Enable logging to track evaluation progress:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## See Also

- **DatasetCheckAgent**: For comparing two sources
- **DomainAnalysisAgent**: For domain extraction and assessment
- **QuestionAugmentationAgent**: For generating and rephrasing questions
- **TextAugmentationAgent**: For text augmentation
