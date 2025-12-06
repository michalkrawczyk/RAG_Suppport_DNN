"""
DEPRECATED: Use RAG_supporters.utils.suggestion_processing instead.

This module previously contained suggestion processing functions that have been
moved to RAG_supporters.utils.suggestion_processing for better organization.

Import from the new location:
    from RAG_supporters.utils.suggestion_processing import (
        filter_by_field_value,
        aggregate_unique_terms,
    )
"""

# Re-export for backward compatibility
from ..utils.suggestion_processing import (
    aggregate_unique_terms,
    filter_by_field_value,
)

__all__ = ["filter_by_field_value", "aggregate_unique_terms"]
