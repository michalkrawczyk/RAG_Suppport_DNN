from typing import Optional, List, Dict, Any, Union
import json
import logging
import os
import pickle
from pathlib import Path

from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
from langchain_core.embeddings.embeddings import Embeddings
import torch

# from .rag_dataset import BaseRAGDatasetGenerator
# TODO: DomainAssignDataset should handle RagDataset?

def count_csv_rows_chunked(csv_path: Path, chunksize: int = 10000) -> int:
    """Count rows by processing in chunks"""
    total = 0
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        total += len(chunk)
    return total


class DomainAssignDataset(Dataset):
    """
    PyTorch Dataset for Matching Source Texts and Questions with Suggested Terms.
    Optimized for large CSVs and embedding generation.

    Args:
        df: pandas DataFrame (or path to CSV for lazy loading)
        embedding_model: Embeddings model for generating embeddings
        source_col: Column name for source text
        question_col: Column name for questions
        suggestions_col: Column name for suggestions (JSON)
        min_confidence: Minimum confidence threshold
        suggestion_types: List of suggestion types to include
        cache_dir: Directory to cache embeddings and parsed data
        precompute_embeddings: Whether to precompute all embeddings upfront
        lazy_load: Whether to load data lazily from disk (for large CSVs)
        return_embeddings: Return embeddings instead of raw text
    """

    def __init__(
            self,
            df: Union[pd.DataFrame, str, Path],
            embedding_model: Optional[Embeddings] = None,
            source_col: str = 'source',
            question_col: str = 'question',
            suggestions_col: str = 'suggestions',
            min_confidence: float = 0.0,
            suggestion_types: Optional[List[str]] = None,
            cache_dir: Optional[Union[str, Path]] = None,
            precompute_embeddings: bool = True,
            lazy_load: bool = False,
            return_embeddings: bool = True,
            chunksize: int = 10000,
    ):
        self.source_col = source_col
        self.question_col = question_col
        self.suggestions_col = suggestions_col
        self.min_confidence = min_confidence
        self.suggestion_types = suggestion_types
        self.embedding_model = embedding_model
        self.return_embeddings = return_embeddings and embedding_model is not None
        self.chunksize = chunksize
        self.lazy_load = lazy_load

        # Setup cache directory
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        if isinstance(df, (str, Path)):
            self.csv_path = Path(df)
            if lazy_load:
                # Store only metadata, load on demand
                self.df = None
                self._length = sum(1 for _ in open(self.csv_path)) - 1  # minus header
            else:
                logging.info(f"Loading CSV from {self.csv_path}")
                self.df = pd.read_csv(self.csv_path)
                self._length = len(self.df)
        else:
            self.csv_path = None
            self.df = df.reset_index(drop=True)
            self._length = len(self.df)

        # Initialize caches
        self._parsed_suggestions_cache = {}
        self._text_embeddings_cache = {}
        self._suggestion_embeddings_cache = {}
        self._unique_suggestions = None

        # Load or compute caches
        if self.cache_dir:
            self._load_caches()

        if precompute_embeddings and self.return_embeddings and not self.lazy_load:
            self._precompute_embeddings()

    def _load_caches(self):
        """Load cached data from disk"""
        cache_files = {
            'parsed_suggestions': self.cache_dir / 'parsed_suggestions.pkl',
            'text_embeddings': self.cache_dir / 'text_embeddings.pkl',
            'suggestion_embeddings': self.cache_dir / 'suggestion_embeddings.pkl',
            'unique_suggestions': self.cache_dir / 'unique_suggestions.pkl',
        }

        for cache_name, cache_path in cache_files.items():
            if cache_path.exists():
                logging.info(f"Loading cache: {cache_name}")
                with open(cache_path, 'rb') as f:
                    setattr(self, f'_{cache_name}_cache' if 'cache' not in cache_name else f'_{cache_name}',
                            pickle.load(f))

    def _save_caches(self):
        """Save caches to disk"""
        if not self.cache_dir:
            return

        cache_data = {
            'parsed_suggestions.pkl': self._parsed_suggestions_cache,
            'text_embeddings.pkl': self._text_embeddings_cache,
            'suggestion_embeddings.pkl': self._suggestion_embeddings_cache,
            'unique_suggestions.pkl': self._unique_suggestions,
        }

        for filename, data in cache_data.items():
            if data:
                with open(self.cache_dir / filename, 'wb') as f:
                    pickle.dump(data, f)

    def _get_row(self, idx: int) -> pd.Series:
        """Get row by index, with support for lazy loading"""
        if self.lazy_load:
            # Read only the specific row from CSV
            return pd.read_csv(self.csv_path, skiprows=range(1, idx + 1), nrows=1).iloc[0]
        else:
            return self.df.iloc[idx]

    def _precompute_embeddings(self):
        """Precompute embeddings for all texts and suggestions"""
        if not self.embedding_model:
            return

        logging.info("Precomputing embeddings...")

        # Collect all texts
        all_sources = []
        all_questions = []
        all_suggestions_set = set()

        for idx in tqdm(range(len(self)), desc="Collecting texts"):
            row = self._get_row(idx)
            all_sources.append(str(row[self.source_col]))
            all_questions.append(str(row[self.question_col]))

            suggestions = self._get_parsed_suggestions(idx, row)
            all_suggestions_set.update(suggestions)

        # Batch embed sources and questions
        logging.info("Embedding sources and questions...")
        for idx in tqdm(range(0, len(all_sources), self.chunksize), desc="Embedding texts"):
            end_idx = min(idx + self.chunksize, len(all_sources))

            sources_batch = all_sources[idx:end_idx]
            questions_batch = all_questions[idx:end_idx]

            source_embeds = self.embedding_model.embed_documents(sources_batch)
            question_embeds = self.embedding_model.embed_documents(questions_batch)

            for i, (s_emb, q_emb) in enumerate(zip(source_embeds, question_embeds)):
                self._text_embeddings_cache[idx + i] = {
                    'source': np.array(s_emb, dtype=np.float32),
                    'question': np.array(q_emb, dtype=np.float32)
                }

        # Batch embed unique suggestions
        logging.info("Embedding suggestions...")
        all_suggestions_list = sorted(list(all_suggestions_set))
        self._unique_suggestions = all_suggestions_list

        for idx in tqdm(range(0, len(all_suggestions_list), self.chunksize), desc="Embedding suggestions"):
            end_idx = min(idx + self.chunksize, len(all_suggestions_list))
            batch = all_suggestions_list[idx:end_idx]

            embeds = self.embedding_model.embed_documents(batch)

            for term, emb in zip(batch, embeds):
                self._suggestion_embeddings_cache[term] = np.array(emb, dtype=np.float32)

        # Save caches
        self._save_caches()

    def _get_parsed_suggestions(self, idx: int, row: Optional[pd.Series] = None) -> List[str]:
        """Get parsed suggestions with caching"""
        if idx in self._parsed_suggestions_cache:
            return self._parsed_suggestions_cache[idx]

        if row is None:
            row = self._get_row(idx)

        suggestions = self._parse_suggestions(row[self.suggestions_col])
        self._parsed_suggestions_cache[idx] = suggestions
        return suggestions

    def _parse_suggestions(self, suggestions_data: Union[str, list]) -> List[str]:
        """Parse and filter suggestions based on confidence and type."""
        if isinstance(suggestions_data, str):
            try:
                suggestions = json.loads(suggestions_data.replace("'", '"'))
            except json.JSONDecodeError:
                try:
                    suggestions = eval(suggestions_data)
                except:
                    return []
        else:
            suggestions = suggestions_data

        if not isinstance(suggestions, list):
            return []

        filtered_terms = []
        for suggestion in suggestions:
            if not isinstance(suggestion, dict):
                continue

            confidence = suggestion.get('confidence', 0.0)
            if confidence < self.min_confidence:
                continue

            if self.suggestion_types is not None:
                suggestion_type = suggestion.get('type', '')
                if suggestion_type not in self.suggestion_types:
                    continue

            term = suggestion.get('term', '')
            if term:
                filtered_terms.append(term)

        return filtered_terms

    def get_unique_suggestions(self) -> List[str]:
        """Extract all unique suggestion terms (cached)"""
        if self._unique_suggestions is not None:
            return self._unique_suggestions

        unique_terms = set()
        for idx in tqdm(range(len(self)), desc="Extracting unique suggestions"):
            suggestions = self._get_parsed_suggestions(idx)
            unique_terms.update(suggestions)

        self._unique_suggestions = sorted(list(unique_terms))

        if self.cache_dir:
            with open(self.cache_dir / 'unique_suggestions.pkl', 'wb') as f:
                pickle.dump(self._unique_suggestions, f)

        return self._unique_suggestions

    def get_suggestion_embeddings(self) -> Dict[str, np.ndarray]:
        """Get all suggestion embeddings as a dictionary"""
        if not self._suggestion_embeddings_cache and self.embedding_model:
            unique_suggestions = self.get_unique_suggestions()
            logging.info("Computing suggestion embeddings...")

            for idx in tqdm(range(0, len(unique_suggestions), self.chunksize)):
                end_idx = min(idx + self.chunksize, len(unique_suggestions))
                batch = unique_suggestions[idx:end_idx]
                embeds = self.embedding_model.embed_documents(batch)

                for term, emb in zip(batch, embeds):
                    self._suggestion_embeddings_cache[term] = np.array(emb, dtype=np.float32)

        return self._suggestion_embeddings_cache

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        """
        Returns:
            dict with keys:
                - 'source': source text or embedding
                - 'question': question text or embedding
                - 'suggestions': list of suggestion terms or embeddings
                - 'suggestion_texts': list of suggestion terms (always included)
        """
        row = self._get_row(idx)
        suggestions = self._get_parsed_suggestions(idx, row)

        if self.return_embeddings:
            # Return embeddings
            if idx not in self._text_embeddings_cache:
                # Compute on-the-fly if not cached
                source_text = str(row[self.source_col])
                question_text = str(row[self.question_col])

                source_emb = np.array(self.embedding_model.embed_query(source_text), dtype=np.float32)
                question_emb = np.array(self.embedding_model.embed_query(question_text), dtype=np.float32)
            else:
                source_emb = self._text_embeddings_cache[idx]['source']
                question_emb = self._text_embeddings_cache[idx]['question']

            # Get suggestion embeddings
            suggestion_embeds = []
            for term in suggestions:
                if term not in self._suggestion_embeddings_cache:
                    emb = np.array(self.embedding_model.embed_query(term), dtype=np.float32)
                    self._suggestion_embeddings_cache[term] = emb
                else:
                    emb = self._suggestion_embeddings_cache[term]
                suggestion_embeds.append(emb)

            return {
                'source': torch.from_numpy(source_emb),
                'question': torch.from_numpy(question_emb),
                'suggestions': [torch.from_numpy(emb) for emb in suggestion_embeds],
                'suggestion_texts': suggestions,
                'idx': idx
            }
        else:
            # Return raw text
            return {
                'source': str(row[self.source_col]),
                'question': str(row[self.question_col]),
                'suggestions': suggestions,
                'idx': idx
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
