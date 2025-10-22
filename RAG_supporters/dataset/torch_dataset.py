from typing import Optional, List, Dict, Any, Union
import json
import logging
import os
# from random import

from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
from langchain_core.embeddings.embeddings import Embeddings

from .rag_dataset import BaseRAGDatasetGenerator

# TODO: Simplify RAGSimpleClassificationDataset( all extras should be done later)
# TODO: DomainAssignDataset should handle RagDataset?
# TODO: DomainAssignDataset should also have option to return embeddings

class DomainAssignDataset(Dataset):
    """
    PyTorch Dataset for Matching Source Texts and Questions with Suggested Terms (for later clustering).

    Args:
        df: pandas DataFrame
        source_col: Column name for source text
        question_col: Column name for questions
        suggestions_col: Column name for suggestions (JSON)
        min_confidence: Minimum confidence threshold for filtering suggestions (default: 0.0)
        suggestion_types: List of suggestion types to include (e.g., ['domain', 'keyword']).
                         If None, includes all types.
    """

    def __init__(
            self,
            df: pd.DataFrame,
            source_col: str = 'source',
            question_col: str = 'question',
            suggestions_col: str = 'suggestions',
            min_confidence: float = 0.0,
            suggestion_types: Optional[List[str]] = None
    ):
        self.df = df.reset_index(drop=True)
        self.source_col = source_col
        self.question_col = question_col
        self.suggestions_col = suggestions_col
        self.min_confidence = min_confidence
        self.suggestion_types = suggestion_types

    def __len__(self):
        return len(self.df)

    def get_unique_suggestions(self) -> List[str]:
        """
        Extract all unique suggestion terms from the entire dataset,
        respecting the confidence and type filters set during initialization.

        Returns:
            List of unique suggestion terms (sorted alphabetically)
        """
        unique_terms = set()

        for idx in range(len(self.df)):
            suggestions = self._parse_suggestions(self.df.iloc[idx][self.suggestions_col])
            unique_terms.update(suggestions)

        return sorted(list(unique_terms))

    def _parse_suggestions(self, suggestions_data: Union[str, list]) -> List[str]:
        """
        Parse and filter suggestions based on confidence and type.

        Returns:
            List of term texts that meet the filtering criteria
        """
        # Parse JSON if it's a string
        if isinstance(suggestions_data, str):
            try:
                suggestions = json.loads(suggestions_data.replace("'", '"'))
            except json.JSONDecodeError:
                # Try eval as fallback for Python dict strings
                try:
                    suggestions = eval(suggestions_data)
                except:
                    return []
        else:
            suggestions = suggestions_data

        if not isinstance(suggestions, list):
            return []

        # Filter and extract terms
        filtered_terms = []
        for suggestion in suggestions:
            if not isinstance(suggestion, dict):
                continue

            # Check confidence threshold
            confidence = suggestion.get('confidence', 0.0)
            if confidence < self.min_confidence:
                continue

            # Check type filter
            if self.suggestion_types is not None:
                suggestion_type = suggestion.get('type', '')
                if suggestion_type not in self.suggestion_types:
                    continue

            # Extract term
            term = suggestion.get('term', '')
            if term:
                filtered_terms.append(term)

        return filtered_terms

    def __getitem__(self, idx):
        """
        Returns:
            dict with keys:
                - 'source': source text
                - 'question': question text
                - 'suggestions': list of filtered suggestion terms
        """
        row = self.df.iloc[idx]

        source = str(row[self.source_col])
        question = str(row[self.question_col])
        suggestions = self._parse_suggestions(row[self.suggestions_col])

        return {
            'source': source,
            'question': question,
            'suggestions': suggestions
        }



# TODO: Dataset for second stage
# class RAGSimpleClassificationDataset(Dataset):
#     logger = logging.getLogger(__name__)
#
#     def __init__(
#             self,
#             dataset: Optional[BaseRAGDatasetGenerator] = None,
#             embedding_model: Optional[Embeddings] = None,
#             sample_limit: Optional[int] = None,
#
#
#             # use_contrastive_samples: bool = True,
#             # use_positive_samples: bool = True,
#             # use_similar_samples: bool = False, # Or that as settings (dict, dataclass, kwargs)?
#     ):
#         self._data_df: Optional[pd.DataFrame] = None
#         self._dataset = dataset
#         self._embedding_model = embedding_model
#         self._sample_limit = sample_limit
#
#         self._validate_dataset()
#         # TODO:
#
#     def __len__(self):
#         return len(self._data_df)
#
#     def _validate_dataset(self):
#         pass #TODO
#
#     def _validate_csv_columns(self, df: pd.DataFrame, base_col: str, target_col: str, value_cols: Optional[List[str]]):
#         if base_col not in df.columns:
#             raise ValueError(f"Base column '{base_col}' not found in CSV.")
#         if target_col not in df.columns:
#             raise ValueError(f"Target column '{target_col}' not found in CSV.")
#         if value_cols:
#             for col in value_cols:
#                 if col not in df.columns:
#                     raise ValueError(f"Value column '{col}' not found in CSV.")
#
#     def samples_from_dataset(self):
#         pass # TODO: samples from BaseRAGDatasetGenerator (Chroma db)
#
#     def samples_from_csv(self,
#                          sample_csv_paths: Union[str, List[str]],
#                          base_col: str = 'question_text',
#                          target_col: str = 'source_text',
#                          value_cols: Optional[List[str]] = None,
#                          overwrite: bool = False
#                          ) -> pd.DataFrame:
#         if not self._embedding_model:
#             raise ValueError("Embedding model must be provided to load samples from CSV.")
#
#         if isinstance(sample_csv_paths, str):
#             sample_csv_paths = [sample_csv_paths]
#
#         # ✅ Keep these outside the file loop to cache across ALL files
#         base_embedd_dict = {}
#         target_embedd_dict = {}
#         samples_list = []  # ✅ Single list for all samples
#         total_samples = 0
#
#         for path in sample_csv_paths:
#             try:
#                 # ✅ For very large CSVs, consider chunked reading
#                 if self._is_large_csv(path):  # You'll need to implement this check
#                     df_chunks = pd.read_csv(path, chunksize=10000)
#                     df_iterator = df_chunks
#                 else:
#                     df_iterator = [pd.read_csv(path)]
#
#                 for df_chunk in df_iterator:
#                     self._validate_csv_columns(df_chunk, base_col, target_col, value_cols)
#
#                     # ✅ Process chunk efficiently
#                     chunk_samples = self._process_chunk(
#                         df_chunk, base_col, target_col, value_cols,
#                         base_embedd_dict, target_embedd_dict,
#                         total_samples
#                     )
#
#                     samples_list.extend(chunk_samples)
#                     total_samples += len(chunk_samples)
#
#                     # ✅ Early termination if limit reached
#                     if self._sample_limit and total_samples >= self._sample_limit:
#                         samples_list = samples_list[:self._sample_limit]
#                         break
#
#             except Exception as e:
#                 self.logger.error(f"Error processing file {path}: {e} (skipping dataset: {path})")
#
#             if self._sample_limit and total_samples >= self._sample_limit:
#                 break
#
#         # ✅ Create DataFrame only ONCE at the end
#         return pd.DataFrame(samples_list) if samples_list else pd.DataFrame()
#
#     def _process_chunk(self, df_chunk, base_col, target_col, value_cols,
#                        base_embedd_dict, target_embedd_dict, current_total):
#         """Process a chunk of data efficiently"""
#         chunk_samples = []
#
#         for row in tqdm(df_chunk.itertuples(), total=len(df_chunk),
#                         desc="Processing chunk", leave=False):
#
#             if self._sample_limit and (current_total + len(chunk_samples)) >= self._sample_limit:
#                 break
#
#             base_text = row[base_col]
#             target_text = row[target_col]
#             values = row[value_cols].tolist() if value_cols else []
#
#             # Get or compute embeddings
#             if base_text not in base_embedd_dict:
#                 base_embedd_dict[base_text] = self._embedding_model.embed_query(base_text)
#
#             if target_text not in target_embedd_dict:
#                 target_embedd_dict[target_text] = self._embedding_model.embed_query(target_text)
#
#             chunk_samples.append({
#                 'base_embedding': base_embedd_dict[base_text],
#                 'target_embedding': target_embedd_dict[target_text],
#                 'values': values,
#             })
#
#         return chunk_samples
#
#     def _is_large_csv(self, path, size_threshold_mb=100):
#         """Check if CSV file is large enough to warrant chunked processing"""
#         return os.path.getsize(path) > (size_threshold_mb * 1024 * 1024)
