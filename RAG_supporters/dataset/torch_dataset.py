from typing import Optional, List, Dict, Any, Union
import logging
import os
# from random import

from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
from langchain_core.embeddings.embeddings import Embeddings

from .rag_dataset import BaseRAGDatasetGenerator


class RAGSimpleClassificationDataset(Dataset):
    _data_df: Optional[pd.DataFrame] = None
    logger = logging.getLogger(__name__)

    def __init__(
            self,
            dataset: Optional[BaseRAGDatasetGenerator] = None,
            embedding_model: Optional[Embeddings] = None,
            sample_limit: Optional[int] = None,


            # use_contrastive_samples: bool = True,
            # use_positive_samples: bool = True,
            # use_similar_samples: bool = False, # Or that as settings (dict, dataclass, kwargs)?
    ):
        self._dataset = dataset
        self._embedding_model = embedding_model
        self._sample_limit = sample_limit

        self._validate_dataset()
        # TODO:

    def _validate_dataset(self):
        pass #TODO

    def _validate_csv_columns(self, df: pd.DataFrame, base_col: str, target_col: str, value_cols: Optional[List[str]]):
        if base_col not in df.columns:
            raise ValueError(f"Base column '{base_col}' not found in CSV.")
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in CSV.")
        if value_cols:
            for col in value_cols:
                if col not in df.columns:
                    raise ValueError(f"Value column '{col}' not found in CSV.")

    def samples_from_dataset(self):
        pass # TODO: samples from BaseRAGDatasetGenerator (Chroma db)

    def samples_from_csv(self,
                         sample_csv_paths: Union[str, List[str]],
                         base_col: str = 'question_text',
                         target_col: str = 'source_text',
                         value_cols: Optional[List[str]] = None,
                         overwrite: bool = False
                         ) -> pd.DataFrame:
        if not self._embedding_model:
            raise ValueError("Embedding model must be provided to load samples from CSV.")

        if isinstance(sample_csv_paths, str):
            sample_csv_paths = [sample_csv_paths]

        # ✅ Keep these outside the file loop to cache across ALL files
        base_embedd_dict = {}
        target_embedd_dict = {}
        samples_list = []  # ✅ Single list for all samples
        total_samples = 0

        for path in sample_csv_paths:
            try:
                # ✅ For very large CSVs, consider chunked reading
                if self._is_large_csv(path):  # You'll need to implement this check
                    df_chunks = pd.read_csv(path, chunksize=10000)
                    df_iterator = df_chunks
                else:
                    df_iterator = [pd.read_csv(path)]

                for df_chunk in df_iterator:
                    self._validate_csv_columns(df_chunk, base_col, target_col, value_cols)

                    # ✅ Process chunk efficiently
                    chunk_samples = self._process_chunk(
                        df_chunk, base_col, target_col, value_cols,
                        base_embedd_dict, target_embedd_dict,
                        total_samples
                    )

                    samples_list.extend(chunk_samples)
                    total_samples += len(chunk_samples)

                    # ✅ Early termination if limit reached
                    if self._sample_limit and total_samples >= self._sample_limit:
                        samples_list = samples_list[:self._sample_limit]
                        break

            except Exception as e:
                self.logger.error(f"Error processing file {path}: {e} (skipping dataset: {path})")

            if self._sample_limit and total_samples >= self._sample_limit:
                break

        # ✅ Create DataFrame only ONCE at the end
        return pd.DataFrame(samples_list) if samples_list else pd.DataFrame()

    def _process_chunk(self, df_chunk, base_col, target_col, value_cols,
                       base_embedd_dict, target_embedd_dict, current_total):
        """Process a chunk of data efficiently"""
        chunk_samples = []

        for row in tqdm(df_chunk.itertuples(), total=len(df_chunk),
                        desc="Processing chunk", leave=False):

            if self._sample_limit and (current_total + len(chunk_samples)) >= self._sample_limit:
                break

            base_text = row[base_col]
            target_text = row[target_col]
            values = row[value_cols].tolist() if value_cols else []

            # Get or compute embeddings
            if base_text not in base_embedd_dict:
                base_embedd_dict[base_text] = self._embedding_model.embed_query(base_text)

            if target_text not in target_embedd_dict:
                target_embedd_dict[target_text] = self._embedding_model.embed_query(target_text)

            chunk_samples.append({
                'base_embedding': base_embedd_dict[base_text],
                'target_embedding': target_embedd_dict[target_text],
                'values': values,
            })

        return chunk_samples

    def _is_large_csv(self, path, size_threshold_mb=100):
        """Check if CSV file is large enough to warrant chunked processing"""
        return os.path.getsize(path) > (size_threshold_mb * 1024 * 1024)

    def cache_records(self, df: pd.DataFrame):
        pass # TODO: Make cache file for faster reuse
