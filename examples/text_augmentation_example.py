"""
Example usage of TextAugmentationAgent for generating alternative questions and sources.

This example demonstrates how to use the TextAugmentationAgent to augment
a dataset by creating rephrased versions of questions and sources while
preserving their original meaning.
"""

import logging
import sys
from pathlib import Path

# Add the RAG_supporters directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "RAG_supporters"))

logging.basicConfig(level=logging.INFO)

try:
    from langchain_openai import ChatOpenAI
    import pandas as pd
    from agents.text_augmentation import TextAugmentationAgent

    def example_basic_rephrasing():
        """Example of basic text rephrasing."""
        print("\n=== Example 1: Basic Text Rephrasing ===\n")

        # Initialize the LLM (you need to set OPENAI_API_KEY environment variable)
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

        # Create the agent
        agent = TextAugmentationAgent(llm=llm, verify_meaning=False)

        # Example question
        question = (
            "What is the primary function of mitochondria in cellular respiration?"
        )

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

    def example_csv_augmentation():
        """Example of augmenting a CSV file."""
        print("\n\n=== Example 2: CSV Augmentation ===\n")

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

    def example_custom_columns():
        """Example with custom column names."""
        print("\n\n=== Example 3: Custom Column Names ===\n")

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

    if __name__ == "__main__":
        print("=" * 70)
        print("TextAugmentationAgent Examples")
        print("=" * 70)
        print("\nNote: Make sure you have set OPENAI_API_KEY environment variable")
        print("or modify the examples to use your preferred LLM.\n")

        try:
            # Run examples
            example_basic_rephrasing()
            example_csv_augmentation()
            example_custom_columns()

            print("\n" + "=" * 70)
            print("Examples completed successfully!")
            print("=" * 70)

        except Exception as e:
            print(f"\nError running examples: {str(e)}")
            print(
                "\nMake sure you have:\n"
                "1. Set OPENAI_API_KEY environment variable\n"
                "2. Installed required packages: pip install langchain-openai pandas\n"
            )

except ImportError as e:
    print(f"Import error: {e}")
    print(
        "\nPlease install required dependencies:\n"
        "pip install langchain-openai pandas\n"
    )
