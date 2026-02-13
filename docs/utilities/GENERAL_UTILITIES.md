# General Utilities

Documentation for general-purpose utility functions.

---

## text_utils.py

**Location**: `RAG_supporters/utils/text_utils.py:1`

**Purpose**: Text validation and processing utilities.

### Functions

#### `is_empty_text(text: Optional[str]) -> bool`

Checks if text is None, empty, or whitespace-only.

**Parameters**:
- `text`: Text to check

**Returns**: True if empty, False otherwise

**Example**:
```python
from RAG_supporters.utils.text_utils import is_empty_text

if not is_empty_text(source_text):
    # Process text
    pass
```

---

#### `normalize_string(text: str) -> str`

Normalizes string by lowercasing, stripping, and normalizing whitespace.

**Parameters**:
- `text`: Text to normalize

**Returns**: Normalized text

**Example**:
```python
normalized = normalize_string("  Hello   World  ")
# Returns: "hello world"
```

---

#### `clean_text(text: str) -> str`

Removes special characters and extra whitespace.

**Parameters**:
- `text`: Text to clean

**Returns**: Cleaned text

**Operations**:
- Removes non-alphanumeric characters (except spaces)
- Collapses multiple spaces to single space
- Strips leading/trailing whitespace

---

#### `truncate_text(text: str, max_len: int, suffix: str = "...") -> str`

Truncates text with ellipsis if longer than max_len.

**Parameters**:
- `text`: Text to truncate
- `max_len`: Maximum length
- `suffix`: Suffix to add (default: "...")

**Returns**: Truncated text

**Example**:
```python
short = truncate_text("Very long text here", max_len=10)
# Returns: "Very lo..."
```

---

## suggestion_processing.py

**Purpose**: Processing utilities for LLM-generated suggestions.

### Functions

#### `filter_by_field_value(suggestions: List[Dict], field: str, min_value: float) -> List[Dict]`

Filters suggestions by field value threshold.

**Parameters**:
- `suggestions`: List of suggestion dicts
- `field`: Field name to filter on (e.g., "confidence")
- `min_value`: Minimum value threshold

**Returns**: Filtered suggestions

**Example**:
```python
from RAG_supporters.utils.suggestion_processing import filter_by_field_value

high_conf = filter_by_field_value(suggestions, "confidence", min_value=0.8)
```

---

#### `aggregate_unique_terms(suggestions: List[Dict], normalize: bool = True) -> List[str]`

Deduplicates and aggregates terms from suggestions.

**Parameters**:
- `suggestions`: List of suggestion dicts
- `normalize`: Whether to normalize terms (default: True)

**Returns**: List of unique terms

**Example**:
```python
from RAG_supporters.utils.suggestion_processing import aggregate_unique_terms

unique = aggregate_unique_terms(suggestions, normalize=True)
```

---

#### `parse_json_suggestions(text: str) -> List[Dict]`

Extracts JSON from LLM output text.

**Parameters**:
- `text`: LLM output text containing JSON

**Returns**: Parsed JSON as list of dicts

**Handles**:
- Multiple JSON objects in text
- JSON within markdown code blocks
- Malformed JSON (returns empty list)

**Example**:
```python
from RAG_supporters.utils.suggestion_processing import parse_json_suggestions

llm_output = '```json\n{"term": "RAG", "confidence": 0.9}\n```'
parsed = parse_json_suggestions(llm_output)
# Returns: [{"term": "RAG", "confidence": 0.9}]
```

---

## text_splitters.py

**Purpose**: Text chunking and segmentation utilities.

### Functions

#### `split_by_sentences(text: str, max_length: int) -> List[str]`

Sentence-aware text splitting.

**Parameters**:
- `text`: Text to split
- `max_length`: Maximum characters per chunk

**Returns**: List of text chunks

**Features**:
- Splits on sentence boundaries (. ! ?)
- Respects max_length constraint
- Handles edge cases (no punctuation, very long sentences)

**Example**:
```python
from RAG_supporters.utils.text_splitters import split_by_sentences

chunks = split_by_sentences(long_text, max_length=512)
```

---

#### `split_by_tokens(text: str, max_tokens: int, tokenizer) -> List[str]`

Token-based text splitting.

**Parameters**:
- `text`: Text to split
- `max_tokens`: Maximum tokens per chunk
- `tokenizer`: Tokenizer to use

**Returns**: List of text chunks

**Features**:
- Uses actual tokenizer to count tokens
- Respects token boundaries
- Handles special tokens

**Example**:
```python
from RAG_supporters.utils.text_splitters import split_by_tokens
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
chunks = split_by_tokens(text, max_tokens=512, tokenizer=tokenizer)
```

---

#### `split_with_overlap(text: str, chunk_size: int, overlap: int) -> List[str]`

Creates overlapping text chunks.

**Parameters**:
- `text`: Text to split
- `chunk_size`: Size of each chunk (characters)
- `overlap`: Overlap between chunks (characters)

**Returns**: List of overlapping text chunks

**Use Case**: Sliding window for text processing

**Example**:
```python
from RAG_supporters.utils.text_splitters import split_with_overlap

chunks = split_with_overlap(text, chunk_size=512, overlap=128)
```

---

## Related Documentation

- [Data Preparation](DATA_PREPARATION.md) - CSV merger uses text_utils
- [Data Validation](DATA_VALIDATION.md) - Validation utilities
- [JASPER Builder](JASPER_BUILDER.md) - Complete pipeline
