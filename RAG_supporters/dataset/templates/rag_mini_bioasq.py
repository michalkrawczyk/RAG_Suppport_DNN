import json
import os
import random
import warnings
from itertools import combinations, product
from typing import List

from datasets import load_dataset
from langchain_chroma import Chroma
from tqdm import tqdm

from dataset.rag_dataset import BaseRAGDatasetGenerator, SampleTripletRAGChroma

try:
    from langchain_openai import OpenAIEmbeddings

except ImportError:

    def OpenAIEmbeddings(*args, **kwargs):
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
        super(RagMiniBioASQBase, self).__init__()
        self._dataset_dir = dataset_dir
        self._passage_id_cast_json = os.path.join(
            self._dataset_dir, "passage_id_to_db_id.json"
        )

        self._embed_function = (
            embed_function
            if embed_function is not None
            else OpenAIEmbeddings(
                openai_api_key=kwargs.get("openai_api_key")
                or os.getenv("OPENAI_API_KEY"),
                model=kwargs.get("model", "text-embedding-3-small"),
            )
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
            self._init_text_corpus_db()
            self._save_passage_json()  # Save the mapping of passage IDs to Chroma IDs for future use
        else:
            # Text corpus is already initialized - load mapping from passage_id to chroma_id
            if not os.path.exists(os.path.join(self._passage_id_cast_json)):
                # passage_id_cast is only needed for now for question db init so might be skipped
                warnings.warn(f"{self._passage_id_cast_json} not found", RuntimeWarning)
                #  raise FileNotFoundError(f"{self._passage_id_cast_json} not found")
            else:
                with open(self._passage_id_cast_json, "r") as f:
                    self._passage_id_to_db_id = json.load(f)

            self.validate_dataset()

        # Check if question database is empty and load dataset if necessary
        if len(self._question_db.get()["ids"]) == 0:
            self._init_questions_db()

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
        valid_types = ["positive", "contrastive", "similar"]
        if sample_type not in valid_types:
            raise ValueError(
                f"Invalid sample_type: {sample_type}. Must be one of {valid_types}"
            )

        # # Validate the dataset before generating samples
        # self.validate_dataset()

        all_samples = []

        # Get all questions from the database
        question_data = self._question_db.get(include=["metadatas"])  # ids are included

        for i, question_id in enumerate(
            tqdm(question_data["ids"], desc=f"Generating {sample_type} samples")
        ):
            # Extract relevant passage IDs from metadata
            metadata = question_data["metadatas"][i]
            relevant_chroma_ids_str = metadata.get("relevant_chroma_ids", "[]")

            # Parse the string representation of the list
            try:
                relevant_passage_ids = eval(relevant_chroma_ids_str)
            except (SyntaxError, NameError):
                # Handle malformed data
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

    def _init_text_corpus_db(self, batch_size: int = 20):
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
        dataset = load_dataset("rag-datasets/rag-mini-bioasq", "text-corpus")[
            "passages"
        ]

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
                chroma_keys = self._text_corpus_db.add_texts(
                    batch_passages, batch_metadata
                )

                # Update mapping
                self._passage_id_to_db_id.update(dict(zip(batch_ids, chroma_keys)))

                pbar.update(batch_end - i)

    def _init_questions_db(self, batch_size: int = 10):
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
        dataset = load_dataset(
            "rag-datasets/rag-mini-bioasq", "question-answer-passages"
        )["test"]

        batch_list = []
        batch_metadata = []

        # Process questions in batches
        for i, (question, qid, relevant_ids_str) in enumerate(
            tqdm(
                zip(
                    dataset["question"], dataset["id"], dataset["relevant_passage_ids"]
                ),
                desc="Loading dataset",
                total=len(dataset["id"]),
            )
        ):
            metadata = {"id": qid, "relevant_ids": relevant_ids_str}
            batch_list.append(question)
            batch_metadata.append(metadata)
            relevant_ids = relevant_ids_str.strip("[]").split(",")
            relevant_ids = [int(x.strip()) for x in relevant_ids]

            # Convert passage IDs to Chroma IDs for the relevant passages
            # TODO: Think about storing relevant ids in separate keys and method to search them in chroma at once
            metadata["relevant_chroma_ids"] = str(
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
            raise ValueError(
                f"Passage with ID {passage_id} not found in text corpus database"
            )

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
                        answer_id_1=pid_1,
                        answer_id_2=pid_2,
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
            passage_combinations = list(
                product(relevant_passage_db_ids, negatives_picked)
            )
            for rel_pid, neg_pid in passage_combinations:
                sample_triplets.append(
                    SampleTripletRAGChroma(
                        question_id=question_db_id,
                        answer_id_1=rel_pid,  # Already a Chroma ID
                        answer_id_2=neg_pid,  # Already a Chroma ID
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
                            answer_id_1=rel_pid,  # Already a Chroma ID
                            answer_id_2=neg_pid,  # Already a Chroma ID
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
            self._question_db.get(ids=[question_db_id], include=["embeddings"])[
                "embeddings"
            ][0],
            search_db="text",
            k=kwargs.get("top_k", 3),
            include=["distances"],
        )

        # Filter passages that are close but not in the relevant set
        picked_questions = [
            source_id
            for source_id, dist in zip(
                close_sources["ids"][0], close_sources["distances"][0]
            )
            if dist < score_threshold and source_id not in relevant_passage_db_ids
        ]

        # Generate combinations of relevant and similar non-relevant passages
        passage_combinations = list(product(relevant_passage_db_ids, picked_questions))
        for source_id_1, source_id_2 in passage_combinations:
            sample_triplets.append(
                SampleTripletRAGChroma(
                    question_id=question_db_id,
                    answer_id_1=source_id_1,
                    answer_id_2=source_id_2,
                    label=1 if assume_relevant_best else -1,
                )
            )

        return sample_triplets

    def _save_passage_json(self):
        """
        Save the passage database to a JSON file.

        The JSON file will contain the mapping of passage IDs to their corresponding
        Chroma IDs. This is useful for later retrieval and analysis.
        """
        with open(self._passage_id_cast_json, "w") as f:
            json.dump(self._passage_id_to_db_id, f)
