import logging
from typing import Optional

import pandas as pd
from torch.utils.data import DataLoader, Dataset

from excluded_drafts.rag_dataset import BaseDatasetRAG

LOGGER = logging.getLogger(__name__)


def load_csv_dataset_pd(
    csv_path: str, sample_limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Load CSV dataset wit triplet samples into a pandas DataFrame.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing triplet samples.
    sample_limit : int, optional
        Maximum number of samples to load. If None, load all samples.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the triplet samples with columns:
        'question_id', 'answer_id_1', 'answer_id_2', 'label'
    """
    df = pd.read_csv(csv_path)

    if sample_limit is not None:
        df = df.head(sample_limit)

    print(f"Loaded {len(df)} triplet samples from {csv_path}")
    return df



