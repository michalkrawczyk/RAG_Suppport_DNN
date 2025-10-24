"""Text splitting utilities for RAG dataset processing."""

import re
from typing import List


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using basic heuristics.

    Parameters
    ----------
    text : str
        Text to split into sentences.

    Returns
    -------
    List[str]
        List of sentences extracted from the text.
    """
    # Simple sentence splitting - handles common cases
    # Splits on . ! ? followed by space and capital letter or end of string
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())
    # Filter out empty sentences
    return [s.strip() for s in sentences if s.strip()]
