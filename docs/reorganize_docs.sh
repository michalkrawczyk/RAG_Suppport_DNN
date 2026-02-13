#!/bin/bash
# Script to reorganize module documentation into subdirectories

# Create subdirectories
mkdir -p docs/contrastive_learning
mkdir -p docs/data_preparation
mkdir -p docs/data_validation
mkdir -p docs/jasper_builder
mkdir -p docs/embeddings
mkdir -p docs/utilities

# Move files to appropriate subdirectories
mv docs/CONTRASTIVE_LEARNING.md docs/contrastive_learning/
mv docs/DATA_PREPARATION.md docs/data_preparation/
mv docs/DATA_VALIDATION.md docs/data_validation/
mv docs/JASPER_BUILDER.md docs/jasper_builder/
mv docs/EMBEDDINGS_OPERATIONS.md docs/embeddings/
mv docs/GENERAL_UTILITIES.md docs/utilities/

# CLUSTERING_OPERATIONS.md already moved to docs/clustering/

echo "Module documentation reorganized into subdirectories"
echo "Directory structure:"
echo "  docs/contrastive_learning/CONTRASTIVE_LEARNING.md"
echo "  docs/data_preparation/DATA_PREPARATION.md"
echo "  docs/data_validation/DATA_VALIDATION.md"
echo "  docs/jasper_builder/JASPER_BUILDER.md"
echo "  docs/embeddings/EMBEDDINGS_OPERATIONS.md"
echo "  docs/clustering/CLUSTERING_OPERATIONS.md"
echo "  docs/utilities/GENERAL_UTILITIES.md"
