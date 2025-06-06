import csv
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Union

from langchain_chroma import Chroma
from tqdm import tqdm

from agents.dataset_check import DatasetCheckAgent
from prompts_templates import SRC_COMPARE_PROMPT_WITH_SCORES

LOGGER = logging.getLogger(__name__)


@dataclass
class SampleTripletRAGChroma:
    """
    A dataclass representing a triplet of question and two answers for RAG evaluation.

    Attributes
    ----------
    question_id : str
        Unique identifier for the question in ChromaDB
    answer_id_1 : str
        Unique identifier for the first answer in ChromaDB
    answer_id_2 : str
        Unique identifier for the second answer in ChromaDB
    label : int, optional
        Indicates which answer is better:
        -1 means not labeled, 0 means both are irrelevant, 1 means answer_1 is better, 2 means answer_2 is better
    """

    question_id: str
    answer_id_1: str
    answer_id_2: str
    label: int = (
        -1
    )  # -1 means not labeled, 0 means both are irrelevant, 1 means answer_1 is better, 2 means answer_2 is better


# TODO: For later implementation (for efficient storing)
# @dataclass
# class SampleTripletMultiRAGChroma:
#     question_id: str
#     answer_id_1: str
#     invalid_answers: List[str]
#


class BaseRAGDatasetGenerator(ABC):
    """
    Abstract base class for generating RAG dataset samples.

    This class provides the framework for creating different types of
    question-answer triplet samples for evaluating RAG systems.

    Attributes
    ----------
    _question_db : Chroma
        ChromaDB collection containing questions
    _text_corpus_db : Chroma
        ChromaDB collection containing text corpus/passages
    _dataset_dir : str
        Directory path for storing dataset files
    _embed_function : callable, optional
        Function used for embedding text
    """

    _question_db: Chroma
    _text_corpus_db: Chroma
    _dataset_dir: str
    _embed_function = None

    @abstractmethod
    def load_dataset(self):
        """
        Load the RAG dataset from the specified source.

        This method should initialize the ChromaDB collections with
        the questions and text corpus data.

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def validate_dataset(self):
        """
        Validate the loaded dataset for correctness and completeness.

        This method should check that the dataset meets the required format
        and contains all necessary information.

        Returns
        -------
        bool
            True if dataset is valid, False otherwise
        """
        pass

    @abstractmethod
    def generate_samples(self, sample_type: str, **kwargs):
        """
        Generate dataset samples based on the specified type.

        Parameters
        ----------
        sample_type : str
            Type of samples to generate. Valid values are:
            'positive', 'contrastive', 'similar'

        Returns
        -------
        List
            List of generated samples
        """
        pass

    @abstractmethod
    def _generate_positive_triplet_samples(
        self, question_chroma_id, relevant_passage_ids, **kwargs
    ) -> List[SampleTripletRAGChroma]:
        """
        Generate triplets consisting of a question and two relevant passages.

        These samples are intended for comparing two passages that are both
        assigned to the same question in the database.

        Parameters
        ----------
        question_chroma_id : str
            Unique identifier of the question in ChromaDB
        relevant_passage_ids : List[str]
            List of passage IDs that are relevant to the question
        **kwargs : dict
            Additional arguments for customizing sample generation

        Returns
        -------
        List[SampleTripletRAGChroma]
            List of generated triplet samples
        """
        pass

    @abstractmethod
    def _generate_contrastive_triplet_samples(
        self,
        question_db_id,
        relevant_passage_db_ids,
        num_negative_samples: int = 2,
        keep_same_negatives=False,
        **kwargs,
    ) -> List[SampleTripletRAGChroma]:
        """
        Generate triplets with a question, one relevant passage and one irrelevant passage.

        These samples are intended for comparing a relevant passage with a randomly
        chosen passage not assigned to the question.

        Parameters
        ----------
        question_db_id : str
            Unique identifier of the question in ChromaDB
        relevant_passage_db_ids : List[str]
            List of passage IDs that are relevant to the question
        num_negative_samples : int, optional
            Number of negative samples to generate per question-passage pair.
            Default is 2.
        keep_same_negatives : bool, optional
            If True, reuse the same negative samples for different relevant passages.
            Default is False.
        **kwargs : dict
            Additional arguments for customizing sample generation

        Returns
        -------
        List[SampleTripletRAGChroma]
            List of generated triplet samples
        """
        pass

    @abstractmethod
    def _generate_similar_triplet_samples(
        self, question_db_id, relevant_passage_db_ids, **kwargs
    ) -> List[SampleTripletRAGChroma]:
        """
        Generate triplets with a question, a relevant passage, and a similar but less relevant passage.

        These samples are intended for comparing a relevant passage with another
        passage that is semantically similar to the question but may be less relevant.

        Parameters
        ----------
        question_db_id : str
            Unique identifier of the question in ChromaDB
        relevant_passage_db_ids : List[str]
            List of passage IDs that are relevant to the question
        **kwargs : dict
            Additional arguments for customizing sample generation

        Returns
        -------
        List[SampleTripletRAGChroma]
            List of generated triplet samples
        """
        pass

    def chroma_id_to_embedding(self, chroma_ids: Union[List[str], str], search_db: str):
        """
        Retrieve embeddings for the given ChromaDB IDs.

        Parameters
        ----------
        chroma_ids : Union[List[str], str]
            ChromaDB ID or list of IDs to retrieve embeddings for
        search_db : str
            Database to search in, either 'question' or 'text'

        Returns
        -------
        List[List[float]]
            List of embeddings corresponding to the provided IDs

        Raises
        ------
        ValueError
            If search_db is not 'question' or 'text'
        """
        if search_db not in ["question", "text"]:
            raise ValueError(
                f"search_db must be either 'question' or 'text'. Got {search_db}"
            )
        db_to_search = (
            self._question_db if search_db == "question" else self._text_corpus_db
        )

        return db_to_search.get(
            ids=chroma_ids if isinstance(chroma_ids, list) else [chroma_ids],
            include=["embeddings"],
        )["embeddings"]

    def get_question_db_data(
        self, include: Iterable[str] = ("documents", "metadatas", "embeddings")
    ):
        """
        Retrieve data from the question database.

        Parameters
        ----------
        include : Iterable[str], optional
            Data fields to include in the response.
            Default is ("documents", "metadatas", "embeddings").

        Returns
        -------
        Dict
            Dictionary containing the requested data fields
        """
        return self._question_db.get(include=list(include))

    def validate_triplet_samples(
        self,
        llm,
        samples: List[SampleTripletRAGChroma],
        analysis_prompt: str = SRC_COMPARE_PROMPT_WITH_SCORES,
        # answer_extraction_prompt: str = FINAL_VERDICT_PROMPT,
        skip_labeled: bool = True,
        overwrite_mismatched_labels: bool = False,
    ) -> List[SampleTripletRAGChroma]:
        """
        Validate triplet samples using LLM-based verification.

        This method uses a language model to evaluate and label triplet samples
        by determining which of the two passages better answers the question.

        Parameters
        ----------
        llm : object
            Language model to use for validation
        samples : List[SampleTripletRAGChroma]
            List of triplet samples to validate
        analysis_prompt : str, optional
            Prompt template for analysis. Default is SRC_COMPARE_PROMPT_WITH_SCORES.
        skip_labeled : bool, optional
            If True, skip samples that already have a label. Default is True.
        overwrite_mismatched_labels : bool
            If True, overwrite labels other than -1 that have mismatch between ground truth and prediction

        Returns
        -------
        List[SampleTripletRAGChroma]
            List of validated triplet samples with labels
        """
        # Create verification chain using the provided LLM and prompts
        verifier_agent = DatasetCheckAgent(llm, compare_prompt=analysis_prompt)
        samples_verified = []

        for sample in tqdm(samples, desc="Validating samples"):
            try:
                # Skip already labeled samples if requested
                if skip_labeled and sample.label != -1:
                    samples_verified.append(sample)
                    continue

                # Retrieve the question and source texts from ChromaDB
                question_text = self._question_db.get_by_ids(sample.question_id)[
                    "documents"
                ][0]
                sources = self._text_corpus_db.get_by_ids(
                    [sample.answer_id_1, sample.answer_id_2]
                )["documents"]

                # Invoke the verifier chain to determine which source is better
                result = verifier_agent.compare_text_sources(
                    question=question_text,
                    source1=sources[0],
                    source2=sources[1],
                )

                predicted_label = result["label"]
                if sample.label != -1 and predicted_label != sample.label:
                    LOGGER.warning(
                        f"Label mismatch for sample {sample.question_id}: "
                        f"Predicted: {predicted_label}, Actual: {sample.label}"
                    )
                    predicted_label = (
                        predicted_label if overwrite_mismatched_labels else sample.label
                    )

                # Create a new sample with the label assigned by the verifier
                samples_verified.append(
                    SampleTripletRAGChroma(
                        question_id=sample.question_id,
                        answer_id_1=sample.answer_id_1,
                        answer_id_2=sample.answer_id_2,
                        label=predicted_label,
                    )
                )
            except Exception as e:
                LOGGER.info(f"Error validating sample: {e}")

        return samples_verified

    def _raw_similarity_search(
        self,
        embedding_or_text: Union[List[float], str],
        search_db: Literal["question", "text"],
        k: int = 4,
        where: Optional[Dict[str, str]] = None,
        where_document: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Optional[List[Any]]]:
        """
        Perform similarity search by vector and return relevance scores.

        This is a low-level method that directly queries the ChromaDB collection.

        Parameters
        ----------
        search_db : Literal["question", "text"]
            Database to search in, either 'question' or 'text'
        embedding_or_text : List[float]
            Embedding vector to search with
        k : int, optional
            Number of results to return. Default is 4.
        where : Optional[Dict[str, str]], optional
            Dictionary to filter results by metadata fields.
            E.g. {"color" : "red", "price": 4.20}
        where_document : Optional[Dict[str, str]], optional
            Dictionary to filter by document content.
            E.g. {$contains: {"text": "hello"}}
        **kwargs : Any
            Additional keyword arguments to pass to Chroma collection query

        Returns
        -------
        Dict[str, Optional[List[Any]]]
            Dictionary containing query results from ChromaDB
        """
        # Select the appropriate database
        db = self._question_db if search_db == "question" else self._text_corpus_db

        embedding_querry = (
            embedding_or_text
            if not isinstance(embedding_or_text, str)
            else self._embed_function.embed_query(embedding_or_text)
        )
        # Note: Normal search by text won't work as setting embed function in _collection don't work properly

        results = db._collection.query(
            query_embeddings=embedding_querry,
            n_results=k,
            where=where,
            where_document=where_document,
            **kwargs,
        )

        return results

    def save_triplets_to_csv(
        self,
        triplets: List[SampleTripletRAGChroma],
        output_file: str,
        include_embeddings: bool = False,
    ) -> None:
        """
        Save triplet samples to CSV with both IDs and corresponding text content.

        Parameters
        ----------
        triplets : List[SampleTripletRAGChroma]
            The list of triplet samples to save
        output_file : str
            Path where the CSV file will be saved
        include_embeddings : bool, optional
            Whether to include embeddings in the output. Default is False.

        Returns
        -------
        None
        """

        # Create directory if it doesn't exist
        Path(os.path.dirname(output_file)).mkdir(parents=True, exist_ok=True)

        LOGGER.info(f"Saving {len(triplets)} triplets to {output_file}")

        with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "question_id",
                "question_text",
                "answer_id_1",
                "answer_text_1",
                "answer_id_2",
                "answer_text_2",
                "label",
            ]

            if include_embeddings:
                fieldnames.extend(
                    ["question_embedding", "answer_embedding_1", "answer_embedding_2"]
                )

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for triplet in tqdm(triplets, desc="Writing triplets to CSV"):
                # Get texts from ChromaDB by IDs
                question_text = self._question_db.get(ids=[triplet.question_id])[
                    "documents"
                ][0]
                answer_texts = self._text_corpus_db.get(
                    ids=[triplet.answer_id_1, triplet.answer_id_2]
                )["documents"]

                row = {
                    "question_id": triplet.question_id,
                    "question_text": question_text,
                    "answer_id_1": triplet.answer_id_1,
                    "answer_text_1": answer_texts[0],
                    "answer_id_2": triplet.answer_id_2,
                    "answer_text_2": answer_texts[1],
                    "label": triplet.label,
                }

                if include_embeddings:
                    # Get embeddings if requested
                    question_embedding = self._question_db.get(
                        ids=[triplet.question_id], include=["embeddings"]
                    )["embeddings"][0]

                    answer_embeddings = self._text_corpus_db.get(
                        ids=[triplet.answer_id_1, triplet.answer_id_2],
                        include=["embeddings"],
                    )["embeddings"]

                    row.update(
                        {
                            "question_embedding": str(question_embedding),
                            "answer_embedding_1": str(answer_embeddings[0]),
                            "answer_embedding_2": str(answer_embeddings[1]),
                        }
                    )

                writer.writerow(row)

        LOGGER.info(f"Successfully saved triplets to {output_file}")
