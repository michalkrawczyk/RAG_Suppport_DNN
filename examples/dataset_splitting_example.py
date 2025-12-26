"""
Example: Dataset Splitting with Persistent Sample Tracking

This example demonstrates how to:
1. Create a train/val split
2. Save the split configuration
3. Restore the same split later
4. Use with ClusterLabeledDataset (if available)
"""

import numpy as np
from pathlib import Path
import sys
import tempfile

# Add dataset module to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'RAG_supporters' / 'dataset'))

# Import directly from dataset_splitter module to avoid dependencies
from dataset_splitter import DatasetSplitter, create_train_val_split


def example_1_basic_split():
    """Example 1: Basic train/val split"""
    print("\n" + "=" * 60)
    print("Example 1: Basic Train/Val Split")
    print("=" * 60)
    
    # Create a splitter with reproducible random seed
    splitter = DatasetSplitter(random_state=42)
    
    # Split a dataset of 100 samples with 20% validation
    train_indices, val_indices = splitter.split(
        dataset_size=100,
        val_ratio=0.2,
        shuffle=True
    )
    
    print(f"Dataset size: 100")
    print(f"Train samples: {len(train_indices)}")
    print(f"Val samples: {len(val_indices)}")
    print(f"First 5 train indices: {train_indices[:5]}")
    print(f"First 5 val indices: {val_indices[:5]}")
    
    # Verify no overlap
    overlap = set(train_indices) & set(val_indices)
    print(f"Overlap between train and val: {len(overlap)} (should be 0)")


def example_2_save_and_load():
    """Example 2: Save and load split configuration"""
    print("\n" + "=" * 60)
    print("Example 2: Save and Load Split")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        split_path = Path(tmpdir) / "my_split.json"
        
        # Create and save a split
        print("Creating and saving split...")
        splitter1 = DatasetSplitter(random_state=42)
        train_idx1, val_idx1 = splitter1.split(dataset_size=100, val_ratio=0.2)
        
        # Add metadata
        metadata = {
            "experiment": "baseline",
            "model": "bert",
            "date": "2024-01-15"
        }
        splitter1.save_split(split_path, metadata=metadata)
        print(f"Split saved to: {split_path}")
        
        # Load the split
        print("\nLoading split...")
        splitter2 = DatasetSplitter.load_split(split_path)
        train_idx2, val_idx2 = splitter2.get_split()
        
        # Verify they're identical
        print(f"Splits are identical: {train_idx1 == train_idx2 and val_idx1 == val_idx2}")
        print(f"Train indices match: {train_idx1[:5]} == {train_idx2[:5]}")


def example_3_convenience_function():
    """Example 3: Using the convenience function"""
    print("\n" + "=" * 60)
    print("Example 3: Convenience Function")
    print("=" * 60)
    
    # One-liner to create a split
    train_idx, val_idx = create_train_val_split(
        dataset_size=100,
        val_ratio=0.2,
        random_state=42,
        shuffle=True
    )
    
    print(f"Created split with convenience function")
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")


def example_4_different_ratios():
    """Example 4: Different validation ratios"""
    print("\n" + "=" * 60)
    print("Example 4: Different Validation Ratios")
    print("=" * 60)
    
    dataset_size = 1000
    ratios = [0.1, 0.2, 0.3]
    
    for ratio in ratios:
        splitter = DatasetSplitter(random_state=42)
        train_idx, val_idx = splitter.split(
            dataset_size=dataset_size,
            val_ratio=ratio
        )
        print(f"Val ratio {ratio:.1f}: Train={len(train_idx)}, Val={len(val_idx)}")


def example_5_reproducibility():
    """Example 5: Reproducibility with same random seed"""
    print("\n" + "=" * 60)
    print("Example 5: Reproducibility Test")
    print("=" * 60)
    
    # Create two splitters with same seed
    splitter1 = DatasetSplitter(random_state=42)
    train1, val1 = splitter1.split(dataset_size=100, val_ratio=0.2)
    
    splitter2 = DatasetSplitter(random_state=42)
    train2, val2 = splitter2.split(dataset_size=100, val_ratio=0.2)
    
    print(f"Same random seed (42):")
    print(f"  Splits are identical: {train1 == train2 and val1 == val2}")
    
    # Create splitter with different seed
    splitter3 = DatasetSplitter(random_state=123)
    train3, val3 = splitter3.split(dataset_size=100, val_ratio=0.2)
    
    print(f"Different random seed (123):")
    print(f"  Splits are different: {train1 != train3 or val1 != val3}")
    print(f"  But sizes are same: Train={len(train3)}, Val={len(val3)}")


def example_6_validation():
    """Example 6: Split validation"""
    print("\n" + "=" * 60)
    print("Example 6: Split Validation")
    print("=" * 60)
    
    # Create a split for dataset of size 100
    splitter = DatasetSplitter(random_state=42)
    train_idx, val_idx = splitter.split(dataset_size=100, val_ratio=0.2)
    
    # Validate against correct size
    try:
        splitter.validate_split(dataset_size=100)
        print("✓ Split is valid for dataset size 100")
    except ValueError as e:
        print(f"✗ Validation failed: {e}")
    
    # Try to validate against wrong size
    try:
        splitter.validate_split(dataset_size=50)
        print("✓ Split is valid for dataset size 50")
    except ValueError as e:
        print(f"✗ Validation failed for size 50 (expected): {str(e)[:60]}...")


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("Dataset Splitting Examples")
    print("=" * 60)
    
    example_1_basic_split()
    example_2_save_and_load()
    example_3_convenience_function()
    example_4_different_ratios()
    example_5_reproducibility()
    example_6_validation()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
    print("\nFor more examples and documentation, see:")
    print("  docs/DATASET_SPLITTING.md")


if __name__ == "__main__":
    main()
