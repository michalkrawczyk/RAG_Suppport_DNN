"""RAG Mini BioASQ dataset template implementation."""

import json
import logging
import os
import random
import warnings
from itertools import combinations, product
from typing import List, Optional

import pandas as pd
from datasets import concatenate_datasets, load_dataset
from langchain_chroma import Chroma
from tqdm import tqdm

from RAG_supporters.pytorch_datasets.rag_dataset import (
    BaseRAGDatasetGenerator,
    SamplePairingType,
    SampleTripletRAGChroma,
)

LOGGER = logging.getLogger(__name__)

# TODO: add method to search text corpus subset
# TODO: Consider if "answer" should be required in default BaseRAGDatasetGenerator
# TODO: Review if passage json is still needed
# TODO: Review for logging consistency and needs

try:
    from langchain_openai import OpenAIEmbeddings

except ImportError:

    def OpenAIEmbeddings(*args, **kwargs):
        """Raise ImportError for missing OpenAIEmbeddings."""
        raise ImportError(
            "OpenAIEmbeddings not found. Please install langchain_openai if you want to use OpenAIEmbeddings"
        )


class RagMiniBioASQBase(BaseRAGDatasetGenerator):
    """
    RAG dataset generator for mini BioASQ dataset.

    This class handles the loading, initialization, and generation of samples from the
    BioASQ mini dataset for retrieval-augmented generation (RAG) triplet training.

    Parameters
    ----------
    dataset_dir : str
        Directory where the dataset and Chroma databases will be stored.
    embed_function : callable, optional
        Function to create embeddings. If None, OpenAIEmbeddings will be used.
    **kwargs : dict
        Additional parameters:
        - openai_api_key : str, optional
            OpenAI API key, used if embed_function is None.
        - model : str, optional
            Model to use for embeddings, default is "text-embedding-3-small".
    """

    def __init__(self, dataset_dir: str, embed_function, **kwargs):
        """Initialize the RAG Mini BioASQ dataset generator."""
        super(RagMiniBioASQBase, self).__init__()
        self._dataset_dir = dataset_dir
        self._passage_id_cast_json = os.path.join(self._dataset_dir, "passage_id_to_db_id.json")

        self._embed_function = (
            embed_function
            if embed_function is not None
            else OpenAIEmbeddings(
                openai_api_key=kwargs.get("openai_api_key") or os.getenv("OPENAI_API_KEY"),
                model=kwargs.get("model", "text-embedding-3-small"),
            )
        )
        self.loading_batch_size = kwargs.get("loading_batch_size", 100)  #

        # Initialize dataset metadata
        additional_info = {"loading_batch_size": self.loading_batch_size}
        self._init_dataset_metadata(
            dataset_names=["BioASQ mini"],
            dataset_sources=["enelpol/rag-mini-bioasq"],
            embed_function=embed_function,
            embedding_name=kwargs.get("model") if embed_function is None else None,
            additional_info=additional_info,
        )

        self.load_dataset()
        self._passage_id_to_db_id = {}

    def load_dataset(self):
        """
        Load or initialize the dataset and Chroma databases.

        Creates and initializes two Chroma databases:
        - question_db: containing questions from the BioASQ dataset
        - text_corpus_db: containing passages from the BioASQ text corpus

        If databases exist, it loads them; otherwise, it initializes them
        with data from the BioASQ dataset.
        """
        # Initialize question database Chroma
        self._question_db = Chroma(
            embedding_function=self._embed_function,
            persist_directory=os.path.join(self._dataset_dir, "question_db"),
        )
        # Initialize text corpus database Chroma
        self._text_corpus_db = Chroma(
            embedding_function=self._embed_function,
            persist_directory=os.path.join(self._dataset_dir, "text_corpus_db"),
        )

        # Check if text corpus is initialized, if not - initialize it
        if len(self._text_corpus_db.get()["ids"]) == 0:
            self._init_text_corpus_db(batch_size=self.loading_batch_size)
            self._save_passage_json()  # Save the mapping of passage IDs to Chroma IDs for future use
            # Save dataset metadata when initializing for the first time
            self.save_dataset_metadata()
        else:
            # Text corpus is already initialized - load mapping from passage_id to chroma_id
            if not os.path.exists(os.path.join(self._passage_id_cast_json)):
                # passage_id_cast is only needed for now for question db init so might be skipped
                warnings.warn(f"{self._passage_id_cast_json} not found", RuntimeWarning)
                #  raise FileNotFoundError(f"{self._passage_id_cast_json} not found")
            else:
                with open(self._passage_id_cast_json, "r") as f:
                    self._passage_id_to_db_id = json.load(f)

            # Try to load existing dataset metadata
            try:
                self.load_dataset_metadata()
            except FileNotFoundError:
                LOGGER.warning(
                    "Dataset metadata file not found. Metadata will be created on next save."
                )

            self.validate_dataset()

        # Check if question database is empty and load dataset if necessary
        if len(self._question_db.get()["ids"]) == 0:
            self._init_questions_db(batch_size=self.loading_batch_size)
            # Save metadata after questions are loaded
            self.save_dataset_metadata()

    def validate_dataset(self):
        """
        Validate if dataset is correctly loaded.

        Checks:
        - Databases are not empty
        - All questions have 'relevant_chroma_ids' metadata

        Raises
        ------
        ValueError
            If databases are empty or questions lack required metadata.
        """
        # Check if databases are not empty
        text_corpus_docs = self._text_corpus_db.get(include=["metadatas"])
        if len(text_corpus_docs["ids"]) == 0:
            raise ValueError("Text corpus database is empty")

        question_docs = self._question_db.get(include=["metadatas"])
        if len(question_docs["ids"]) == 0:
            raise ValueError("Question database is empty")

        # if len(self._passage_id_to_db_id) == 0:
        #     raise ValueError(f"{self._passage_id_cast_json}.json is empty")

        # Check if all questions have relevant_chroma_ids metadata
        for i, doc_id in tqdm(enumerate(question_docs["ids"])):
            metadata = question_docs["metadatas"][i]
            if not metadata.get("relevant_chroma_ids", None):
                raise ValueError(
                    f"Question with ID {doc_id} does not have 'relevant_chroma_ids' metadata"
                )
                # TODO: Think about deleting record instead of raising error

    def generate_samples(self, sample_type: str, save_to_csv=True, **kwargs):
        """Generate dataset samples based on sample type."""
        # TODO: This method has inconsistency in returning types (list or DataFrame).
        #  Consider rewrtiing triplets to return DataFrame (if decide to keep old way with triplets)
        valid_types = [
            "positive",
            "contrastive",
            "similar",
            "pairs_relevant",
            "pairs_all_existing",
            "pairs_embedding_similarity",
        ]

        if sample_type not in valid_types:
            raise ValueError(f"Invalid sample_type: {sample_type}. Must be one of {valid_types}")

        # # Validate the dataset before generating samples
        # self.validate_dataset()

        all_samples = []

        # Get all questions from the database
        question_data = self._question_db.get(include=["metadatas"])  # ids are included

        if sample_type in [
            "pairs_relevant",
            "pairs_all_existing",
            "pairs_embedding_similarity",
        ]:
            # Generate pair samples based on the requested type
            sample_df = self._generate_pair_samples_df(
                question_db_ids=question_data["ids"],
                criterion=SamplePairingType(sample_type.replace("pairs_", "")),
                **kwargs,
            )
            if save_to_csv:
                # Save the generated pairs to a CSV file
                pd.DataFrame(sample_df).to_csv(
                    f"{self._dataset_dir}{os.sep}pairs_{sample_type}.csv",
                    index=False,
                    encoding="utf-8",
                )
            return sample_df

        for i, question_id in enumerate(
            tqdm(question_data["ids"], desc=f"Generating {sample_type} samples")
        ):
            # Extract relevant passage IDs from metadata
            metadata = question_data["metadatas"][i]
            relevant_chroma_ids_str = metadata.get("relevant_chroma_ids", "[]")

            try:
                relevant_passage_ids = json.loads(relevant_chroma_ids_str)
            except (json.JSONDecodeError, TypeError, ValueError):
                # Handle malformed data
                LOGGER.warning(
                    f"Malformed relevant_chroma_ids for question ID {question_id} : '{relevant_chroma_ids_str}'"
                )
                relevant_passage_ids = []

            # Skip questions with no relevant passages
            if not relevant_passage_ids:
                continue

            # Generate samples based on the requested type
            if sample_type == "positive":
                samples = self._generate_positive_triplet_samples(
                    question_id, relevant_passage_ids, **kwargs
                )
            elif sample_type == "contrastive":
                samples = self._generate_contrastive_triplet_samples(
                    question_id,
                    relevant_passage_ids,
                    num_negative_samples=kwargs.get("num_negative_samples", 5),
                    keep_same_negatives=kwargs.get("keep_same_negatives", False),
                    assume_relevant_best=kwargs.get("assume_relevant_best", True),
                )
            elif sample_type == "similar":
                samples = self._generate_similar_triplet_samples(
                    question_id,
                    relevant_passage_ids,
                    score_threshold=kwargs.get("score_threshold", 0.3),
                    assume_relevant_best=kwargs.get("assume_relevant_best", True),
                )

            all_samples.extend(samples)

        if save_to_csv:
            self.save_triplets_to_csv(
                all_samples,
                output_file=f"{self._dataset_dir}{os.sep}triplets_{sample_type}.csv",
            )

        return all_samples

    def _init_text_corpus_db(self, batch_size: int = 100):
        """
        Initialize the text corpus database with passages from the BioASQ dataset.

        Loads passages from the BioASQ dataset and adds them to the text corpus
        Chroma database in batches. Also creates a mapping from passage IDs to
        Chroma IDs.

        Parameters
        ----------
        batch_size : int, default=10
            Number of passages to process in a single batch.
        """
        # Load the text corpus from HuggingFace dataset
        dataset = load_dataset("enelpol/rag-mini-bioasq", "text-corpus")["test"]

        self._passage_id_to_db_id = {}
        total = len(dataset["passage"])

        # Process passages in batches
        with tqdm(total=total, desc="Loading text corpus") as pbar:
            for i in range(0, total, batch_size):
                # Extract batch data
                batch_end = min(i + batch_size, total)
                batch_passages = dataset["passage"][i:batch_end]
                batch_ids = dataset["id"][i:batch_end]
                batch_metadata = [{"id": pid} for pid in batch_ids]

                # Add batch to Chroma
                chroma_keys = self._text_corpus_db.add_texts(batch_passages, batch_metadata)

                # Update mapping
                self._passage_id_to_db_id.update(dict(zip(batch_ids, chroma_keys)))

                pbar.update(batch_end - i)

    def _init_questions_db(self, batch_size: int = 100):
        """
        Initialize the question database with questions from the BioASQ dataset.

        Loads questions, their IDs, and relevant passage IDs from the BioASQ
        dataset and adds them to the question Chroma database in batches.

        Parameters
        ----------
        batch_size : int, default=10
            Number of questions to process in a single batch.
        """
        # Load the question-answer-passages dataset
        dataset_full = load_dataset("enelpol/rag-mini-bioasq", "question-answer-passages")

        # Merge train and test splits
        combined_dataset = concatenate_datasets([dataset_full["train"], dataset_full["test"]])

        batch_list = []
        batch_metadata = []

        # Process questions in batches
        for i, (question, qid, relevant_ids_obj, answer) in enumerate(
            tqdm(
                zip(
                    combined_dataset["question"],
                    combined_dataset["id"],
                    combined_dataset["relevant_passage_ids"],
                    combined_dataset["answer"],
                ),
                desc="Loading dataset",
                total=len(combined_dataset),
            )
        ):
            if isinstance(relevant_ids_obj, str):
                # old rag-mini-bioasq version had relevant_ids as string representation of list
                try:
                    relevant_ids = json.loads(relevant_ids_obj)
                    relevant_ids = [int(x) for x in relevant_ids]
                except json.JSONDecodeError as e:
                    LOGGER.warning(f"Failed to decode relevant_passage_ids for question {qid}: {e}")
                    relevant_ids = []
            else:
                # Already as list of int
                relevant_ids = relevant_ids_obj

            metadata = {
                "id": qid,
                "relevant_ids": json.dumps(relevant_ids),
                "answer": answer,
            }

            batch_list.append(question)
            batch_metadata.append(metadata)

            # Convert passage IDs to Chroma IDs for the relevant passages
            metadata["relevant_chroma_ids"] = json.dumps(
                [self._passage_id_to_db_id[pid] for pid in relevant_ids]
            )

            # Process batch when it reaches the desired size
            if i % batch_size == 0 and i > 0:
                self._question_db.add_texts(batch_list, batch_metadata)
                batch_list = []
                batch_metadata = []

        # Process any remaining questions
        if len(batch_list) > 0:
            self._question_db.add_texts(batch_list, batch_metadata)

    def _passage_id_text_to_chroma_id(self, passage_id: str) -> str:
        """
        Convert a passage ID to its corresponding Chroma ID.

        Parameters
        ----------
        passage_id : str
            The passage ID to convert.

        Returns
        -------
        str
            The corresponding Chroma ID.

        Raises
        ------
        FileNotFoundError
            If the mapping file is not found.
        ValueError
            If the passage ID is not found in the mapping.
        """
        # Load mapping if not already loaded
        if not self._passage_id_to_db_id:
            try:
                with open(self._passage_id_cast_json, "r") as f:
                    self._passage_id_to_db_id = json.load(f)
            except FileNotFoundError:
                raise FileNotFoundError(f"{self._passage_id_cast_json} not found")

        # Check if passage ID exists in mapping
        if passage_id not in self._passage_id_to_db_id:
            raise ValueError(f"Passage with ID {passage_id} not found in text corpus database")

        return self._passage_id_to_db_id[passage_id]

    def _generate_positive_triplet_samples(
        self, question_db_id, relevant_passage_db_ids, **kwargs
    ) -> List[SampleTripletRAGChroma]:
        """
        Generate triplets consisting of a question and two relevant passages.

        Creates triplet samples where both passage answers are relevant to the question.
        These are considered positive examples where both passages are equally good.

        Parameters
        ----------
        question_db_id : str
            The Chroma ID of the question embedding.
        relevant_passage_db_ids : list of str
            IDs of passages that are relevant to the question.
        **kwargs : dict
            Additional parameters (not used).

        Returns
        -------
        List[SampleTripletRAGChroma]
            List of triplet samples with question and two relevant passages.
            Label is set to -1 indicating no preference between the two passages.
        """
        sample_triplets = []

        # Generate combinations of relevant passages if there are at least 2
        if len(relevant_passage_db_ids) > 1:
            relevant_combinations = list(combinations(relevant_passage_db_ids, 2))
            for pid_1, pid_2 in relevant_combinations:
                sample_triplets.append(
                    SampleTripletRAGChroma(
                        question_id=question_db_id,
                        source_id_1=pid_1,
                        source_id_2=pid_2,
                        label=-1,  # -1 indicates no preference (both are relevant)
                    )
                )

        return sample_triplets

    def _generate_contrastive_triplet_samples(
        self,
        question_db_id,
        relevant_passage_db_ids,
        num_negative_samples: int = 5,
        keep_same_negatives=False,
        assume_relevant_best=True,
        **kwargs,
    ) -> List[SampleTripletRAGChroma]:
        """
        Generate triplets consisting of a question, one relevant passage, and one non-relevant passage.

        Creates triplet samples where one passage is relevant to the question and the other is not.
        These contrastive samples help the model learn to differentiate between relevant and
        non-relevant passages.

        Parameters
        ----------
        question_db_id : str
            The Chroma ID of the question embedding.
        relevant_passage_db_ids : list of str
            IDs of passages that are relevant to the question.
        num_negative_samples : int, default=5
            Number of negative samples to generate per question.
        keep_same_negatives : bool, default=False
            If True, use the same set of negative passages for all triplets.
        assume_relevant_best : bool, default=True
            If True, label is 1 (relevant passage is better), otherwise -1.
        **kwargs : dict
            Additional parameters.

        Returns
        -------
        List[SampleTripletRAGChroma]
            List of triplet samples with contrastive examples.
        """
        # Find all possible negative passages (not relevant to this question)
        # Get all Chroma IDs from the text corpus
        all_chroma_ids = set(self._text_corpus_db.get()["ids"])

        # Find all possible negative passages (not relevant to this question)
        possible_negatives = list(all_chroma_ids - set(relevant_passage_db_ids))

        sample_triplets = []

        if keep_same_negatives:
            # Use the same set of negative passages for all triplets
            negatives_picked = random.sample(
                possible_negatives, min(num_negative_samples, len(possible_negatives))
            )
            passage_combinations = list(product(relevant_passage_db_ids, negatives_picked))
            for rel_pid, neg_pid in passage_combinations:
                sample_triplets.append(
                    SampleTripletRAGChroma(
                        question_id=question_db_id,
                        source_id_1=rel_pid,  # Already a Chroma ID
                        source_id_2=neg_pid,  # Already a Chroma ID
                        label=1 if assume_relevant_best else -1,
                    )
                )
        else:
            # For each relevant passage, pick random negative passages
            for rel_pid in relevant_passage_db_ids:
                negatives_picked = random.sample(
                    possible_negatives,
                    min(num_negative_samples, len(possible_negatives)),
                )
                for neg_pid in negatives_picked:
                    sample_triplets.append(
                        SampleTripletRAGChroma(
                            question_id=question_db_id,
                            source_id_1=rel_pid,  # Already a Chroma ID
                            source_id_2=neg_pid,  # Already a Chroma ID
                            label=1 if assume_relevant_best else -1,
                        )
                    )

        return sample_triplets

    def _generate_similar_triplet_samples(
        self,
        question_db_id,
        relevant_passage_db_ids,
        score_threshold=0.3,
        assume_relevant_best=True,
        **kwargs,
    ) -> List[SampleTripletRAGChroma]:
        """
        Generate triplets with a question, relevant passage, and a "similar" non-relevant passage.

        Creates challenging triplet samples where the non-relevant passage is semantically
        close to the question in the embedding space, making it harder to distinguish from
        truly relevant passages.

        Parameters
        ----------
        question_db_id : str
            The Chroma ID of the question embedding.
        relevant_passage_db_ids : list of str
            IDs of passages that are relevant to the question.
        score_threshold : float, default=0.3
            Distance threshold for considering a passage as "similar" to the question.
        assume_relevant_best : bool, default=TrueQ
            If True, label is 1 (relevant passage is better), otherwise -1.
        **kwargs : dict
            Additional parameters:
            - top_k : int, optional
                Number of similar passages to retrieve, default is 3.

        Returns
        -------
        List[SampleTripletRAGChroma]
            List of triplet samples with challenging examples.
        """
        sample_triplets = []

        # Find passages that are similar to the question in embedding space
        close_sources = self._raw_similarity_search(
            self._question_db.get(ids=[question_db_id], include=["embeddings"])["embeddings"][0],
            search_db="text",
            k=kwargs.get("top_k", 3),
            include=["distances"],
        )

        # Filter passages that are close but not in the relevant set
        picked_questions = [
            source_id
            for source_id, dist in zip(close_sources["ids"][0], close_sources["distances"][0])
            if dist < score_threshold and source_id not in relevant_passage_db_ids
        ]

        # Generate combinations of relevant and similar non-relevant passages
        passage_combinations = list(product(relevant_passage_db_ids, picked_questions))
        for source_id_1, source_id_2 in passage_combinations:
            sample_triplets.append(
                SampleTripletRAGChroma(
                    question_id=question_db_id,
                    source_id_1=source_id_1,
                    source_id_2=source_id_2,
                    label=1 if assume_relevant_best else -1,
                )
            )

        return sample_triplets

    def _generate_all_existing_pairs(self, question_db_ids, **kwargs):
        """Generate all question-source pairs for ALL_EXISTING criterion.

        This generator processes the text corpus in batches to avoid RAM exhaustion.

        Parameters
        ----------
        question_db_ids : List[str]
            List of question IDs to generate pairs for
        **kwargs : dict
            Additional parameters including optional 'batch_size'

        Yields
        ------
        dict
            Dictionary containing question_id, question_text, source_id, source_text, and answer
        """
        # First, get all source IDs without loading documents (lightweight)
        all_source_ids = self._text_corpus_db.get(include=[])["ids"]
        batch_size = kwargs.get("batch_size", self.loading_batch_size)

        for question_db_id in tqdm(question_db_ids, desc="Generating all-pairs from whole dataset"):
            question_data = self._question_db.get(
                ids=[question_db_id],
                include=["metadatas", "documents"],  # For not overloading memory
            )
            question_text = question_data["documents"][0]
            question_metadata = question_data["metadatas"][0]

            # Process sources in batches to avoid loading all documents at once
            for i in range(0, len(all_source_ids), batch_size):
                batch_ids = all_source_ids[i : i + batch_size]
                sources_batch = self._text_corpus_db.get(ids=batch_ids, include=["documents"])

                for source_id, source_text in zip(sources_batch["ids"], sources_batch["documents"]):
                    if source_text is None or source_text.strip() == "" or source_id == "nan":
                        # Skip empty or invalid passages
                        continue

                    yield {
                        "question_id": question_db_id,
                        "question_text": question_text,
                        "source_id": source_id,
                        "source_text": source_text,
                        "answer": question_metadata.get("answer", ""),
                    }

    def _generate_pair_samples_df(
        self,
        question_db_ids: Optional[List[str]] = None,
        criterion: SamplePairingType = SamplePairingType.EMBEDDING_SIMILARITY,
        **kwargs,
    ) -> List[SampleTripletRAGChroma]:
        result_rows = []

        if question_db_ids is None:
            # Get all questions from the database
            question_db_ids = self._question_db.get(include=["embeddings"])["ids"]
        elif not question_db_ids:
            raise ValueError("question_db_ids cannot be empty")

        if criterion == SamplePairingType.EMBEDDING_SIMILARITY:
            # Find passages that are similar to the question in embedding space
            for question_db_id in tqdm(
                question_db_ids, desc="Generating scored pairs by embedding similarity"
            ):
                # Find close sources based on the question embedding
                question_data = self._question_db.get(
                    ids=[question_db_id],
                    include=["metadatas", "documents"],  # For not overloading memory
                )
                question_text = question_data["documents"][0]
                question_metadata = question_data["metadatas"][0]

                sources = self._raw_similarity_search(
                    self._question_db.get(ids=[question_db_id], include=["embeddings"])[
                        "embeddings"
                    ][0],
                    search_db="text",
                    k=kwargs.get("top_k", 3),
                    include=["distances", "documents"],
                )
                for source_id, source_text in zip(sources["ids"][0], sources["documents"][0]):
                    result_rows.append(
                        {
                            "question_id": question_db_id,
                            "question_text": question_text,
                            "source_id": source_id,
                            "source_text": source_text,
                            "answer": question_metadata.get("answer", ""),
                            # "similarity_score": 1 - distance  # Convert distance to similarity
                        }
                    )

        elif criterion == SamplePairingType.ALL_EXISTING:
            # Use generator to process text corpus in batches and create DataFrame
            return pd.DataFrame(self._generate_all_existing_pairs(question_db_ids, **kwargs))

        elif criterion == SamplePairingType.RELEVANT:
            # Get questions with their relevant passages based on stored metadata
            for question_db_id in tqdm(
                question_db_ids, desc="Generating relevant question-passage pairs"
            ):
                # Get question text and metadata
                question_data = self._question_db.get(
                    ids=[question_db_id], include=["documents", "metadatas"]
                )
                question_text = question_data["documents"][0]
                question_metadata = question_data["metadatas"][0]

                relevant_chroma_ids_str = question_metadata.get("relevant_chroma_ids", "[]")

                try:
                    relevant_passage_ids = json.loads(relevant_chroma_ids_str)
                except (json.JSONDecodeError, TypeError, ValueError):
                    # Handle malformed data
                    LOGGER.warning(
                        f"Malformed relevant_chroma_ids for question ID {question_db_id} : '{relevant_chroma_ids_str}'"
                    )
                    relevant_passage_ids = []

                # Skip questions with no relevant passages
                if not relevant_passage_ids:
                    continue

                # Get the text content of relevant passages
                if relevant_passage_ids:  # Only query if we have IDs
                    relevant_passages_data = self._text_corpus_db.get(
                        ids=relevant_passage_ids, include=["documents"]
                    )

                    # Create pairs for each relevant passage
                    for source_id, source_text in zip(
                        relevant_passage_ids, relevant_passages_data["documents"]
                    ):
                        if source_text is None or source_text.strip() == "" or source_id == "nan":
                            # Skip empty or invalid passages
                            continue

                        result_rows.append(
                            {
                                "question_id": question_db_id,
                                "question_text": question_text,
                                "source_id": source_id,
                                "source_text": source_text,
                                "answer": question_metadata.get("answer", ""),
                            }
                        )

        else:
            raise ValueError(
                f"Unsupported criterion: {criterion}. Only 'ALL_EXISTING', 'RELEVANT' and 'EMBEDDING_SIMILARITY' are supported. (For now)"
            )

        return pd.DataFrame(result_rows)

    def _save_passage_json(self):
        """
        Save the passage database to a JSON file.

        The JSON file will contain the mapping of passage IDs to their corresponding
        Chroma IDs. This is useful for later retrieval and analysis.
        """
        with open(self._passage_id_cast_json, "w") as f:
            json.dump(self._passage_id_to_db_id, f)
