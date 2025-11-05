# CSVQuestionAgent - Quick Start Examples

This directory contains examples for using the CSVQuestionAgent.

## Basic Usage Example

```python
from langchain_openai import ChatOpenAI
from RAG_supporters.agents.csv_question_agent import CSVQuestionAgent
import pandas as pd

# Initialize the agent
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
agent = CSVQuestionAgent(llm=llm)

# Example 1: Rephrase a question with source context
question = "What does it do?"
source = "Mitochondria are organelles that produce ATP through cellular respiration."

rephrased = agent.rephrase_question_with_source(question, source)
print(f"Rephrased: {rephrased}")
# Output: "What is the primary function of mitochondria in cellular respiration?"

# Example 2: Generate alternative questions from a source
source = "Photosynthesis is the process by which plants convert light into energy."
questions = agent.generate_alternative_questions(source, n=5)

for i, q in enumerate(questions, 1):
    print(f"{i}. {q}")

# Example 3: Batch process a CSV file
df = pd.read_csv("questions.csv")  # Must have 'question_text' and 'source_text' columns

# Rephrase all questions based on their sources
result_df = agent.process_dataframe_rephrasing(df, rephrase_mode="source")
result_df.to_csv("rephrased_questions.csv", index=False)

# Or generate questions from sources
sources_df = pd.read_csv("sources.csv")  # Must have 'source_text' column
generated_df = agent.process_dataframe_generation(sources_df, n_questions=3)
generated_df.to_csv("generated_questions.csv", index=False)
```

## Expected CSV Format

### For Question Rephrasing (with source)
Your CSV should have columns: `question_text`, `source_text`

```csv
question_text,source_text
"What is it?","Mitochondria are organelles..."
"How does it work?","Neural networks consist of..."
```

### For Alternative Question Generation
Your CSV should have column: `source_text`

```csv
source_text
"Photosynthesis is the process..."
"The water cycle describes..."
```

## See Full Documentation

For complete documentation with all features and advanced usage, see [CSV_QUESTION_AGENT.md](../CSV_QUESTION_AGENT.md)

## Installation

```bash
pip install langchain langchain-core pandas tqdm

# For OpenAI
pip install langchain-openai

# For other LLM providers, install the appropriate langchain integration
```
