from torch.utils.data import Dataset, DataLoader
import pandas as pd

from excluded_drafts.rag_dataset import BaseDatasetRAG

from logging import Logger
from typing import Optional

LOGGER = Logger(__name__)


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


class TorchDatasetRAG(Dataset):
    # TODO
    def __init__(
        self,
        base_dataset: BaseDatasetRAG,
        transform=None,
        use_contrastive_samples: bool = True,
        use_positive_samples: bool = True,
        use_similar_samples: bool = False,
    ):
        self.base_dataset = base_dataset
        self.transform = transform
        self.question_data = self.base_dataset.get_question_db_data()
        # self.text_corpus_df = self.base_dataset.text_corpus_db
        # self.question_ids = self.question_data['metadatas']['id'] # Assumption: question_ids are unique

        self.sample_holders = []

        # Load samples - skip those without labels
        if use_contrastive_samples:
            self.sample_holders.extend(
                [
                    s
                    for s in self.base_dataset.load_samples_pairs("contrastive")
                    if s.get("label")
                ]
            )
        if use_positive_samples:
            self.sample_holders.extend(
                [
                    s
                    for s in self.base_dataset.load_samples_pairs("positive")
                    if s.get("label")
                ]
            )
        if use_similar_samples:
            self.sample_holders.extend(
                [
                    s
                    for s in self.base_dataset.load_samples_pairs("similar")
                    if s.get("label")
                ]
            )

    def __getitem__(self, idx):
        sample_holder = self.sample_holders[idx]

        # Get question embedding by id

        # Get answer_1 embedding by id

        # Get answer_2 embedding by id

        # Get label

        sample = {
            "question": question_embedding,
            "answer_1": answer_1_embedding,
            "answer_2": answer_2_embedding,
            "label": label,
        }  # TODO: consider storing embeddings as numpy array (3, embedding_size) instead of separate tensors

        return sample

    # def save_csv(self, path):
    #     self.dataset._text_corpus_db.to_csv(path)


class TorchDataLoaderRAG(DataLoader):
    def __init__(
        self, dataset: TorchDatasetRAG, batch_size: int, shuffle: bool, num_workers: int
    ):
        super(TorchDataLoaderRAG, self).__init__(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.sample_holders = self.dataset.sample_holders

    def __len__(self):
        return len(self.sample_holders)

    def __iter__(self):
        return iter(self.sample_holders)

    def __len__(self):
        return len(self.sample_holders)
