#!/usr/bin/env python3
"""
Example: LLM-Driven Cluster Steering and Question Rephrasing

This example demonstrates the ClusterSteeringAgent for generating steering texts,
rephrasing questions, and resolving ambiguity in cluster assignments.

Prerequisites:
- Install: pip install langchain langchain-openai
- Set: export OPENAI_API_KEY="your-key"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def example_validation():
    """Example: Validate LLM outputs."""
    print("\n" + "=" * 80)
    print("Example: Validating LLM Outputs")
    print("=" * 80)

    from RAG_supporters.utils.llm_output_validation import (
        validate_steering_text,
        validate_question_rephrase,
        validate_ambiguity_resolution,
    )

    # Example 1: Steering text validation
    steering_result = {
        "cluster_id": 0,
        "steering_text": "Neural networks use backpropagation for learning.",
        "incorporated_descriptors": ["neural networks", "backpropagation"],
        "confidence": 0.85,
    }

    print("\n1. Validating steering text result...")
    is_valid, errors = validate_steering_text(steering_result)
    print(f"   {'✓ Valid' if is_valid else '✗ Invalid: ' + str(errors)}")

    # Example 2: Question rephrase validation
    rephrase_result = {
        "original_question": "What is machine learning?",
        "target_cluster_id": 1,
        "rephrased_question": "Can you explain ML algorithms?",
        "genre_shift": "Formal to conversational",
        "preserved_intent": "Understanding ML basics",
        "confidence": 0.9,
    }

    print("\n2. Validating question rephrase result...")
    is_valid, errors = validate_question_rephrase(rephrase_result)
    print(f"   {'✓ Valid' if is_valid else '✗ Invalid: ' + str(errors)}")

    # Example 3: Ambiguity resolution validation
    ambiguity_result = {
        "question": "Test question",
        "is_ambiguous": True,
        "primary_cluster": {"cluster_id": 0, "reason": "Main", "confidence": 0.7},
        "secondary_clusters": [{"cluster_id": 1, "reason": "Related", "confidence": 0.5}],
        "recommendation": "multi-domain",
        "explanation": "Spans multiple domains",
    }

    print("\n3. Validating ambiguity resolution result...")
    is_valid, errors = validate_ambiguity_resolution(ambiguity_result)
    print(f"   {'✓ Valid' if is_valid else '✗ Invalid: ' + str(errors)}")


def main():
    """Run examples."""
    print("\n" + "=" * 80)
    print("LLM-Driven Cluster Steering Examples")
    print("=" * 80)
    example_validation()
    
    print("\n" + "=" * 80)
    print("Examples completed! See docs/LLM_STEERING_GUIDE.md for more info.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
