# Examples

This directory contains example scripts demonstrating how to use the RAG_supporters library.

## Text Augmentation Example

`text_augmentation_example.py` demonstrates how to use the `TextAugmentationAgent` to generate alternative versions of questions and sources while preserving their meaning.

### Features Demonstrated

1. **Basic Text Rephrasing**: Shows how to rephrase individual questions and sources
2. **CSV Augmentation**: Demonstrates batch processing of CSV files
3. **Custom Column Mapping**: Shows how to work with datasets that use different column names

### Usage

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key"

# Run the examples
python examples/text_augmentation_example.py
```

### Requirements

- langchain-openai
- pandas
- All dependencies from RAG_supporters/requirements.txt and requirements_agents.txt

Install with:
```bash
pip install -r RAG_supporters/requirements.txt
pip install -r RAG_supporters/requirements_agents.txt
pip install langchain-openai
```
