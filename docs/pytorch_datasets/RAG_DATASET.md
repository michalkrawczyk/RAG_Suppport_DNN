# RAG Dataset

Abstract base class for generating and managing RAG (Retrieval-Augmented Generation) datasets with ChromaDB storage.

## Overview

`BaseRAGDatasetGenerator` provides framework for creating question-answer triplet samples for training and evaluating RAG systems. Not a PyTorch Dataset - used for dataset preparation and labeling.

**Module**: [RAG_supporters/dataset/rag_dataset.py](../../RAG_supporters/dataset/rag_dataset.py)

## Architecture

### Purpose

This is NOT a PyTorch Dataset for training. It's a **dataset generator** for:
1. Loading RAG data into ChromaDB
2. Generating triplet samples (question, source1, source2)
3. Labeling samples with LLMs
4. Exporting to formats consumable by PyTorch Datasets

### Sample Types

Three pairing strategies:

1. **Positive**: Both sources relevant to question
   - Compare two relevant passages
   - Label: which is better? (1, 2, or 0=both equal)

2. **Contrastive**: One relevant, one irrelevant
   - Compare relevant vs random passage
   - Label: usually 1 (relevant is better)

3. **Similar**: One relevant, one similar but less relevant
   - Compare relevant vs embedding-similar passage
   - Label: subtle preference (requires careful scoring)

## API

### Data Classes

#### `SampleTripletRAGChroma`

```python
@dataclass
class SampleTripletRAGChroma:
    question_id: str        # ChromaDB ID for question
    source_id_1: str        # ChromaDB ID for first source
    source_id_2: str        # ChromaDB ID for second source
    label: int = -1         # -1=unlabeled, 0=both bad, 1=first better, 2=second better
```

#### `SamplePairingType`

```python
class SamplePairingType(Enum):
    RELEVANT = "relevant"                      # Both relevant
    ALL_EXISTING = "all_existing"              # Any passage in DB
    EMBEDDING_SIMILARITY = "embedding_similarity"  # Similar by embedding
```

### Abstract Methods

Subclasses must implement:

#### `load_dataset()`

Load data into ChromaDB collections.

```python
def load_dataset(self):
    """Load questions and passages into ChromaDB."""
    self._question_db.add(documents=questions, ids=q_ids)
    self._text_corpus_db.add(documents=passages, ids=p_ids, metadatas=metadata)
```

#### `validate_dataset() -> bool`

Validate dataset structure and completeness.

```python
def validate_dataset(self) -> bool:
    """Check dataset meets requirements."""
    # Verify all questions have at least one relevant passage
    # Verify all passages have metadata
    # Verify no duplicate IDs
    return all_checks_passed
```

#### `generate_samples(sample_type: str, **kwargs) -> List[SampleTripletRAGChroma]`

Generate samples for labeling.

```python
def generate_samples(self, sample_type: str, **kwargs):
    """Generate triplets based on pairing strategy."""
    if sample_type == "positive":
        return self._generate_positive_triplet_samples(...)
    elif sample_type == "contrastive":
        return self._generate_contrastive_triplet_samples(...)
    elif sample_type == "similar":
        return self._generate_similar_triplet_samples(...)
```

## Example Implementation

```python
from RAG_supporters.pytorch_datasets import BaseRAGDatasetGenerator, SampleTripletRAGChroma
from langchain_chroma import Chroma
import pandas as pd

class BioASQDatasetGenerator(BaseRAGDatasetGenerator):
    """Generate RAG samples from BioASQ dataset."""
    
    def __init__(self, embed_function, dataset_dir: str):
        self._embed_function = embed_function
        self._dataset_dir = dataset_dir
        
        # Initialize ChromaDB collections
        self._question_db = Chroma(
            collection_name="bioasq_questions",
            embedding_function=embed_function,
            persist_directory=f"{dataset_dir}/chroma_questions"
        )
        self._text_corpus_db = Chroma(
            collection_name="bioasq_passages",
            embedding_function=embed_function,
            persist_directory=f"{dataset_dir}/chroma_passages"
        )
        
        # Initialize metadata
        self._init_dataset_metadata(
            dataset_names=["BioASQ"],
            dataset_sources=["http://bioasq.org/"],
            embed_function=embed_function
        )
    
    def load_dataset(self):
        """Load BioASQ data into ChromaDB."""
        # Load from source files
        df = pd.read_json("bioasq_training.json")
        
        # Add questions
        questions = df["question"].tolist()
        q_ids = [f"q_{i}" for i in range(len(questions))]
        self._question_db.add(documents=questions, ids=q_ids)
        
        # Add passages with relevance metadata
        passages = []
        p_ids = []
        metadatas = []
        
        for i, row in df.iterrows():
            for j, passage in enumerate(row["relevant_passages"]):
                passages.append(passage)
                p_ids.append(f"p_{i}_{j}")
                metadatas.append({
                    "question_id": f"q_{i}",
                    "relevance": "relevant"
                })
        
        self._text_corpus_db.add(
            documents=passages,
            ids=p_ids,
            metadatas=metadatas
        )
    
    def validate_dataset(self) -> bool:
        """Validate BioASQ dataset."""
        # Check all questions have relevant passages
        q_count = self._question_db._collection.count()
        p_count = self._text_corpus_db._collection.count()
        
        if q_count == 0 or p_count == 0:
            return False
        
        # Check all passages have metadata
        sample = self._text_corpus_db.get(limit=1, include=["metadatas"])
        if not sample["metadatas"]:
            return False
        
        return True
    
    def generate_samples(self, sample_type: str, **kwargs) -> List[SampleTripletRAGChroma]:
        """Generate samples for labeling."""
        # Get all question IDs
        questions = self._question_db.get()
        q_ids = questions["ids"]
        
        all_samples = []
        
        for q_id in q_ids:
            # Get relevant passages for this question
            results = self._text_corpus_db.get(
                where={"question_id": q_id}
            )
            relevant_ids = results["ids"]
            
            if sample_type == "positive":
                samples = self._generate_positive_triplet_samples(q_id, relevant_ids)
            elif sample_type == "contrastive":
                samples = self._generate_contrastive_triplet_samples(
                    q_id, relevant_ids, num_negative_samples=2
                )
            else:
                samples = self._generate_similar_triplet_samples(q_id, relevant_ids)
            
            all_samples.extend(samples)
        
        return all_samples
    
    def _generate_positive_triplet_samples(
        self, question_id: str, relevant_passage_ids: List[str], **kwargs
    ) -> List[SampleTripletRAGChroma]:
        """Generate positive pairs (both relevant)."""
        samples = []
        
        # All pairs of relevant passages
        for i in range(len(relevant_passage_ids)):
            for j in range(i + 1, len(relevant_passage_ids)):
                samples.append(
                    SampleTripletRAGChroma(
                        question_id=question_id,
                        source_id_1=relevant_passage_ids[i],
                        source_id_2=relevant_passage_ids[j],
                        label=-1  # Not yet labeled
                    )
                )
        
        return samples
    
    def _generate_contrastive_triplet_samples(
        self,
        question_id: str,
        relevant_passage_ids: List[str],
        num_negative_samples: int = 2,
        **kwargs
    ) -> List[SampleTripletRAGChroma]:
        """Generate contrastive pairs (relevant vs irrelevant)."""
        samples = []
        
        # Get all passage IDs
        all_passages = self._text_corpus_db.get()
        all_ids = all_passages["ids"]
        
        # Irrelevant passages = all passages - relevant passages
        irrelevant_ids = [p_id for p_id in all_ids if p_id not in relevant_passage_ids]
        
        # Sample negatives
        import random
        negatives = random.sample(irrelevant_ids, min(num_negative_samples, len(irrelevant_ids)))
        
        # Create pairs
        for relevant_id in relevant_passage_ids:
            for negative_id in negatives:
                samples.append(
                    SampleTripletRAGChroma(
                        question_id=question_id,
                        source_id_1=relevant_id,
                        source_id_2=negative_id,
                        label=-1
                    )
                )
        
        return samples
    
    def _generate_similar_triplet_samples(
        self, question_id: str, relevant_passage_ids: List[str], **kwargs
    ) -> List[SampleTripletRAGChroma]:
        """Generate similar pairs (relevant vs embedding-similar)."""
        samples = []
        
        for relevant_id in relevant_passage_ids:
            # Query for similar passages
            results = self._text_corpus_db.similarity_search_by_id(
                relevant_id,
                k=5  # Top 5 similar
            )
            
            similar_ids = [r.id for r in results if r.id not in relevant_passage_ids]
            
            # Pair relevant with most similar irrelevant
            if similar_ids:
                samples.append(
                    SampleTripletRAGChroma(
                        question_id=question_id,
                        source_id_1=relevant_id,
                        source_id_2=similar_ids[0],
                        label=-1
                    )
                )
        
        return samples
```

## Usage Workflow

### 1. Generate Dataset

```python
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize
embed_fn = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
generator = BioASQDatasetGenerator(embed_fn, dataset_dir="output/bioasq")

# Load data
generator.load_dataset()

# Validate
assert generator.validate_dataset(), "Dataset validation failed"
```

### 2. Generate Samples

```python
# Generate positive samples (both relevant)
positive_samples = generator.generate_samples("positive")
print(f"Generated {len(positive_samples)} positive samples")

# Generate contrastive samples (relevant vs irrelevant)
contrastive_samples = generator.generate_samples("contrastive", num_negative_samples=2)
print(f"Generated {len(contrastive_samples)} contrastive samples")
```

### 3. Label Samples with LLM

```python
from langchain_openai import ChatOpenAI
from RAG_supporters.agents import SourceEvaluationAgent

llm = ChatOpenAI(model="gpt-4", temperature=0)
agent = SourceEvaluationAgent(llm=llm)

# Label samples
for sample in positive_samples[:100]:  # Label first 100
    # Get texts from ChromaDB
    question_text = generator._question_db.get(ids=[sample.question_id])["documents"][0]
    source1_text = generator._text_corpus_db.get(ids=[sample.source_id_1])["documents"][0]
    source2_text = generator._text_corpus_db.get(ids=[sample.source_id_2])["documents"][0]
    
    # Get LLM label
    result = agent.compare_text_sources(question_text, source1_text, source2_text)
    sample.label = result["selected_source"]
```

### 4. Export to CSV

```python
import csv

with open("labeled_samples.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["question_id", "source_id_1", "source_id_2", "label"])
    
    for sample in positive_samples:
        if sample.label != -1:  # Only labeled samples
            writer.writerow([
                sample.question_id,
                sample.source_id_1,
                sample.source_id_2,
                sample.label
            ])
```

### 5. Convert to PyTorch Dataset

```python
# Use exported CSV with JASPERSteeringDataset or custom Dataset
# This RAG dataset is just for preparation, not training
```

## Sample Statistics

### Positive Samples

For a dataset with N questions, each having avg R relevant passages:
- **Samples per question**: C(R, 2) = R × (R - 1) / 2
- **Total samples**: N × C(R, 2)

Example: 1000 questions, 5 relevant passages each
- Samples: 1000 × 10 = 10,000 positive samples

### Contrastive Samples

With num_negative_samples=K per relevant passage:
- **Samples per question**: R × K
- **Total samples**: N × R × K

Example: 1000 questions, 5 relevant, 2 negatives each
- Samples: 1000 × 5 × 2 = 10,000 contrastive samples

### Similar Samples

With top-1 similar irrelevant per relevant passage:
- **Samples per question**: R (one similar per relevant)
- **Total samples**: N × R

Example: 1000 questions, 5 relevant each
- Samples: 1000 × 5 = 5,000 similar samples

## ChromaDB Collections

### Question Collection

```python
collection_name: "bioasq_questions"
schema:
  - id: str (e.g., "q_0", "q_1", ...)
  - document: str (question text)
  - embedding: List[float] (computed by embed_function)
```

### Passage Collection

```python
collection_name: "bioasq_passages"
schema:
  - id: str (e.g., "p_0_0", "p_0_1", ...)
  - document: str (passage text)
  - embedding: List[float] (computed by embed_function)
  - metadata: Dict
      - question_id: str (which question this passage is relevant to)
      - relevance: str ("relevant", "irrelevant", "partial")
      - source: str (optional - source URL/document)
```

## Best Practices

### Sample Generation

- **Start with positive samples**: Easiest to label (both are relevant, just compare quality)
- **Add contrastive**: Creates clear decision boundary (relevant vs irrelevant)
- **Use similar sparingly**: Hardest to label, best for fine-tuning

### Labeling Strategy

- **Batch labeling**: Process 100-500 samples at a time
- **LLM consistency**: Use same model/temperature for entire batch
- **Human review**: Sample 10% for quality check
- **Inter-annotator agreement**: Label 100 samples twice, check agreement

### Memory Management

- **ChromaDB persist**: Always set `persist_directory` to save to disk
- **Batch processing**: Don't generate all samples at once for large datasets
- **Incremental labeling**: Label in chunks, save after each batch

## Limitations

- **Not a PyTorch Dataset**: Use for preparation only
- **ChromaDB overhead**: In-memory collections for large datasets may be slow
- **Single-process**: No multi-process generation (ChromaDB limitation)

## See Also

- [JASPER Steering Dataset](JASPER_STEERING_DATASET.md) - PyTorch Dataset for training
- [Source Evaluation Agent](../agents/SOURCE_EVALUATION_AGENT.md) - LLM-based labeling
- [Dataset Check Agent](../agents/DATASET_CHECK_AGENT.md) - Alternative comparison agent
- [ChromaDB Documentation](https://docs.trychroma.com/) - ChromaDB usage
