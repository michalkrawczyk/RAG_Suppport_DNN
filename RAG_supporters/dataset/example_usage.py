"""
Example usage of the flexible cluster steering dataset.

This script demonstrates how to use the BaseDomainAssignDataset with
different steering modes for cluster/subspace steering.

Note: This is a demonstration script. To run it, you need:
- pandas
- numpy
- torch
- A valid embedding model (e.g., from langchain)
"""

import pandas as pd
import numpy as np

# Mock embedding model for demonstration
class MockEmbeddingModel:
    """Simple mock for demonstration purposes."""
    
    def __init__(self, dim=128):
        self.dim = dim
        np.random.seed(42)
    
    def embed_query(self, text: str) -> list:
        """Generate a deterministic embedding."""
        seed = hash(text) % (2**32)
        np.random.seed(seed)
        return np.random.randn(self.dim).tolist()
    
    def embed_documents(self, texts: list) -> list:
        """Generate embeddings for multiple texts."""
        return [self.embed_query(text) for text in texts]


def example_zero_steering():
    """Example: Zero baseline steering for ablation studies."""
    print("\n" + "="*60)
    print("Example 1: Zero Baseline Steering")
    print("="*60)
    
    from torch_dataset import BaseDomainAssignDataset, SteeringMode
    
    # Sample data
    df = pd.DataFrame({
        "source": ["Source A", "Source B", "Source C"],
        "question": [
            "What is machine learning?",
            "How does deep learning work?",
            "Explain neural networks"
        ],
        "suggestions": [
            '[{"term": "ml", "confidence": 0.9, "type": "keyword"}]',
            '[{"term": "dl", "confidence": 0.8, "type": "keyword"}]',
            '[{"term": "nn", "confidence": 0.85, "type": "keyword"}]',
        ],
    })
    
    # Hard cluster assignments
    cluster_labels = {0: 0, 1: 0, 2: 1}
    
    # Create dataset
    dataset = BaseDomainAssignDataset(
        df=df,
        embedding_model=MockEmbeddingModel(),
        steering_mode=SteeringMode.ZERO,
        cluster_labels=cluster_labels,
        return_triplets=True,
        return_embeddings=True,
    )
    
    # Build (compute embeddings)
    dataset.build(save_to_cache=False)
    
    # Get a sample
    sample = dataset[0]
    
    print(f"\nSample structure:")
    print(f"  - base_embedding shape: {sample['base_embedding'].shape}")
    print(f"  - steering_embedding shape: {sample['steering_embedding'].shape}")
    print(f"  - target: {sample['target'].item()}")
    print(f"  - steering_mode: {sample['metadata']['steering_mode']}")
    print(f"  - Steering is all zeros: {(sample['steering_embedding'] == 0).all()}")


def example_cluster_descriptor_steering():
    """Example: Cluster descriptor steering."""
    print("\n" + "="*60)
    print("Example 2: Cluster Descriptor Steering")
    print("="*60)
    
    from torch_dataset import BaseDomainAssignDataset, SteeringMode
    
    # Sample data
    df = pd.DataFrame({
        "source": ["Source A", "Source B", "Source C"],
        "question": [
            "What is machine learning?",
            "How does deep learning work?",
            "Explain neural networks"
        ],
        "suggestions": [
            '[{"term": "ml", "confidence": 0.9, "type": "keyword"}]',
            '[{"term": "dl", "confidence": 0.8, "type": "keyword"}]',
            '[{"term": "nn", "confidence": 0.85, "type": "keyword"}]',
        ],
    })
    
    # Cluster assignments and descriptors
    cluster_labels = {0: 0, 1: 0, 2: 1}
    cluster_descriptors = {
        0: ["machine learning", "artificial intelligence"],
        1: ["neural networks", "deep learning"]
    }
    
    # Create dataset
    dataset = BaseDomainAssignDataset(
        df=df,
        embedding_model=MockEmbeddingModel(),
        steering_mode=SteeringMode.CLUSTER_DESCRIPTOR,
        cluster_labels=cluster_labels,
        cluster_descriptors=cluster_descriptors,
        return_triplets=True,
        return_embeddings=True,
    )
    
    # Build
    dataset.build(save_to_cache=False)
    
    # Get samples
    for i in range(len(dataset)):
        sample = dataset[i]
        print(f"\nSample {i}:")
        print(f"  - Question: {sample['metadata']['question_text']}")
        print(f"  - Cluster: {sample['target'].item()}")
        print(f"  - Descriptors: {sample['metadata']['cluster_descriptors']}")
        print(f"  - Steering shape: {sample['steering_embedding'].shape}")


def example_soft_labels():
    """Example: Soft (probabilistic) cluster assignments."""
    print("\n" + "="*60)
    print("Example 3: Soft Multi-Label Assignments")
    print("="*60)
    
    from torch_dataset import BaseDomainAssignDataset, SteeringMode
    
    # Sample data
    df = pd.DataFrame({
        "source": ["Source A", "Source B", "Source C"],
        "question": [
            "What is machine learning?",
            "How does deep learning work?",
            "Explain neural networks"
        ],
        "suggestions": [
            '[{"term": "ml", "confidence": 0.9, "type": "keyword"}]',
            '[{"term": "dl", "confidence": 0.8, "type": "keyword"}]',
            '[{"term": "nn", "confidence": 0.85, "type": "keyword"}]',
        ],
    })
    
    # Soft cluster assignments (probabilities)
    cluster_labels = {
        0: [0.8, 0.2],  # 80% cluster 0, 20% cluster 1
        1: [0.7, 0.3],  # 70% cluster 0, 30% cluster 1
        2: [0.3, 0.7],  # 30% cluster 0, 70% cluster 1
    }
    
    # Create dataset
    dataset = BaseDomainAssignDataset(
        df=df,
        embedding_model=MockEmbeddingModel(),
        steering_mode=SteeringMode.ZERO,
        cluster_labels=cluster_labels,
        multi_label_mode="soft",
        return_triplets=True,
        return_embeddings=True,
    )
    
    # Build
    dataset.build(save_to_cache=False)
    
    # Get samples
    for i in range(len(dataset)):
        sample = dataset[i]
        print(f"\nSample {i}:")
        print(f"  - Question: {sample['metadata']['question_text']}")
        print(f"  - Target (soft): {sample['target'].numpy()}")
        print(f"  - Sum of probs: {sample['target'].sum().item():.4f}")


def example_llm_steering():
    """Example: LLM-generated steering texts."""
    print("\n" + "="*60)
    print("Example 4: LLM-Generated Steering")
    print("="*60)
    
    from torch_dataset import BaseDomainAssignDataset, SteeringMode
    
    # Sample data
    df = pd.DataFrame({
        "source": ["Source A", "Source B"],
        "question": [
            "What is machine learning?",
            "How does deep learning work?"
        ],
        "suggestions": [
            '[{"term": "ml", "confidence": 0.9, "type": "keyword"}]',
            '[{"term": "dl", "confidence": 0.8, "type": "keyword"}]',
        ],
    })
    
    # LLM-generated steering texts
    llm_steering_texts = {
        0: "Focus on fundamental ML concepts and algorithms",
        1: "Emphasize deep learning architectures and training"
    }
    
    cluster_labels = {0: 0, 1: 1}
    
    # Create dataset
    dataset = BaseDomainAssignDataset(
        df=df,
        embedding_model=MockEmbeddingModel(),
        steering_mode=SteeringMode.LLM_GENERATED,
        llm_steering_texts=llm_steering_texts,
        cluster_labels=cluster_labels,
        return_triplets=True,
        return_embeddings=True,
    )
    
    # Build
    dataset.build(save_to_cache=False)
    
    # Get samples
    for i in range(len(dataset)):
        sample = dataset[i]
        print(f"\nSample {i}:")
        print(f"  - Question: {sample['metadata']['question_text']}")
        print(f"  - LLM Steering: {sample['metadata']['llm_steering_text']}")
        print(f"  - Cluster: {sample['target'].item()}")


def example_mixed_steering():
    """Example: Mixed/weighted steering."""
    print("\n" + "="*60)
    print("Example 5: Mixed Steering (Weighted Combination)")
    print("="*60)
    
    from torch_dataset import BaseDomainAssignDataset, SteeringMode
    
    # Sample data
    df = pd.DataFrame({
        "source": ["Source A", "Source B"],
        "question": [
            "What is machine learning?",
            "How does deep learning work?"
        ],
        "suggestions": [
            '[{"term": "ml", "confidence": 0.9, "type": "keyword"}]',
            '[{"term": "dl", "confidence": 0.8, "type": "keyword"}]',
        ],
    })
    
    cluster_labels = {0: 0, 1: 1}
    cluster_descriptors = {
        0: ["machine learning", "AI"],
        1: ["deep learning", "neural nets"]
    }
    
    # Create dataset with mixed steering
    dataset = BaseDomainAssignDataset(
        df=df,
        embedding_model=MockEmbeddingModel(),
        steering_mode=SteeringMode.MIXED,
        steering_weights={
            "suggestion": 0.6,
            "cluster_descriptor": 0.4
        },
        cluster_labels=cluster_labels,
        cluster_descriptors=cluster_descriptors,
        return_triplets=True,
        return_embeddings=True,
    )
    
    # Build
    dataset.build(save_to_cache=False)
    
    # Get samples
    sample = dataset[0]
    print(f"\nSample structure:")
    print(f"  - Steering mode: {sample['metadata']['steering_mode']}")
    print(f"  - Steering combines suggestion (60%) + cluster desc (40%)")
    print(f"  - Steering embedding shape: {sample['steering_embedding'].shape}")
    print(f"  - Not zero: {not (sample['steering_embedding'] == 0).all()}")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("FLEXIBLE CLUSTER STEERING DATASET - EXAMPLES")
    print("="*60)
    
    try:
        example_zero_steering()
        example_cluster_descriptor_steering()
        example_soft_labels()
        example_llm_steering()
        example_mixed_steering()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
