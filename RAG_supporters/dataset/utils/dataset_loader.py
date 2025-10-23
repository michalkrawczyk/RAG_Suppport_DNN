import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

LOGGER = logging.getLogger(__name__)



def count_csv_rows_chunked(csv_path: Union[str, Path], chunksize: int = 10000) -> int:
    """Count rows by processing in chunks"""
    total = 0
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        total += len(chunk)
    return total


def parse_suggestions_safe(suggestions_data: Union[str, list]) -> List[Dict[str, Any]]:
    """Safely parse suggestions JSON without using eval()"""
    if isinstance(suggestions_data, list):
        return suggestions_data

    if not isinstance(suggestions_data, str):
        return []

    try:
        # Try standard JSON first
        return json.loads(suggestions_data)
    except json.JSONDecodeError:
        try:
            # Handle single quotes by replacing with double quotes
            return json.loads(suggestions_data.replace("'", '"'))
        except json.JSONDecodeError:
            logging.warning(f"Failed to parse suggestions: {suggestions_data[:100]}...")
            return []


def filter_suggestions(
        suggestions: List[Dict[str, Any]],
        min_confidence: float = 0.0,
        suggestion_types: Optional[List[str]] = None
) -> List[str]:
    """Filter suggestions based on confidence and type, return terms"""
    filtered_terms = []

    for suggestion in suggestions:
        if not isinstance(suggestion, dict):
            continue

        # Check confidence
        confidence = suggestion.get('confidence', 0.0)
        if confidence < min_confidence:
            continue

        # Check type if filter specified
        if suggestion_types is not None:
            suggestion_type = suggestion.get('type', '')
            if suggestion_type not in suggestion_types:
                continue

        # Extract term
        term = suggestion.get('term', '')
        if term:
            filtered_terms.append(term)

    return filtered_terms


def compute_cache_version(
        min_confidence: float,
        suggestion_types: Optional[List[str]],
        embedding_model_name: Optional[str] = None
) -> str:
    """Compute a version hash for cache validation"""
    config = {
        'min_confidence': min_confidence,
        'suggestion_types': sorted(suggestion_types) if suggestion_types else None,
        'embedding_model': embedding_model_name or 'none'
    }
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


