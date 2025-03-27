from abc import ABC, abstractmethod
from typing import List, Union, Dict, Iterable, Optional, Any, Tuple, Literal
from dataclasses import dataclass
import logging
import os


from langchain_chroma import Chroma
from tqdm import tqdm

from prompts_templates.rag_verifiers import (
    FINAL_VERDICT_PROMPT,
    SRC_COMPARE_PROMPT_WITH_SCORES,
    create_verifying_chain,
)


LOGGER = logging.getLogger(__name__)


@dataclass
class SampleTripletRAGChroma:
    question_id: str
    answer_id_1: str
    answer_id_2: str
    label: int = (
        -1
    )  # 1 if answer_1 is better, 0 if answer_2 is better, -1 if not labeled


# TODO: For later implementation (for efficient storing)
# @dataclass
# class SampleTripletMultiRAGChroma:
#     question_id: str
#     answer_id_1: str
#     invalid_answers: List[str]
#


class BaseRAGDatasetGenerator(ABC):
    _question_db: Chroma
    _text_corpus_db: Chroma
    _dataset_dir: str
    _embed_function = None

    @abstractmethod
    def load_dataset(self):
        pass

    @abstractmethod
    def validate_dataset(self):
        pass

    @abstractmethod
    def generate_samples(self, sample_type: str):
        pass

    @abstractmethod
    def _generate_positive_triplet_samples(
        self, question_chroma_id, relevant_passage_ids, **kwargs
    ) -> List[SampleTripletRAGChroma]:
        """
        Generate triplets consisting of question and two relevant passages (if both assigned to the same question in database)
        """
        pass

    @abstractmethod
    def _generate_contrastive_triplet_samples(
        self,
        question_chroma_id,
        relevant_passage_ids,
        num_negative_samples: int = 2,
        keep_same_negatives=False,
        **kwargs,
    ) -> List[SampleTripletRAGChroma]:
        """
        Generate triplets consisting of question, its relevant passage and randomly chosen passage not assigned to the question
        """

    @abstractmethod
    def _generate_similar_triplet_samples(
        self, question_chroma_id, relevant_passage_ids, **kwargs
    ) -> List[SampleTripletRAGChroma]:
        """
        Generate triplets consisting of question, its relevant passage and randomly chosen passage assigned to the same question
        close in embedding space to question
        """
        pass

    def _samples_triplet_generator(
        self,
        sample_type: str,
        relevant_ids_field_name: str = "relevant_passage_ids",
        **kwargs,
    ) -> List[Dict[str, Union[str, int]]]:
        sample_triplets = []
        sample_types_dict = {
            "positive": self._generate_positive_triplet_samples,
            "contrastive": self._generate_contrastive_triplet_samples,
            "similar": self._generate_similar_triplet_samples,
        }
        generator_func = sample_types_dict.get(sample_type)

        # TODO: iterate over questions and generate samples
        db_data = self.get_question_db_data(include=["ids", "embeddings", "metadatas"])
        for question_id in tqdm(
            db_data["ids"], desc=f"Generating {sample_type} samples"
        ):
            question_embedding = db_data["embeddings"][question_id]
            relevant_passage_ids = db_data["metadatas"][question_id][
                relevant_ids_field_name
            ]

            sample_triplets.append(
                generator_func(question_embedding, relevant_passage_ids, **kwargs)
            )

            # TODO: Should new ids be added to ChromaDB? - Risk of storing not relevant or poisoned data

        return sample_triplets

    def chroma_id_to_embedding(self, chroma_ids: Union[List[str], str], search_db: str):
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
        return self._question_db.get(include=list(include))

    def validate_triplet_samples(
        self,
        llm,
        samples: List[SampleTripletRAGChroma],
        analysis_prompt: str = SRC_COMPARE_PROMPT_WITH_SCORES,
        answer_extraction_prompt: str = FINAL_VERDICT_PROMPT,
        skip_labeled: bool = True,
    ) -> List[SampleTripletRAGChroma]:

        verifier = create_verifying_chain(
            llm, analysis_prompt, answer_extraction_prompt
        )
        samples_verified = []

        for sample in tqdm(samples, desc="Validating samples"):
            try:
                if skip_labeled and sample.label != -1:
                    samples_verified.append(sample)
                    continue

                question_text = self._question_db.get_by_ids(sample.question_id)[
                    "documents"
                ][0]
                sources = self._text_corpus_db.get_by_ids(
                    [sample.answer_id_1, sample.answer_id_2]
                )["documents"]

                result = verifier.invoke(
                    {
                        "question": question_text,
                        "source1_content": sources[0],
                        "source2_content": sources[1],
                    }
                )

                samples_verified.append(
                    SampleTripletRAGChroma(
                        question_id=sample.question_id,
                        answer_id_1=sample.answer_id_1,
                        answer_id_2=sample.answer_id_2,
                        label=0 if result["answer"] == "Source 1" else 1,
                    )
                )
            except Exception as e:
                LOGGER.info(f"Error validating sample: {e}")

        return samples_verified

    def _raw_similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        where: Optional[Dict[str, str]] = None,
        where_document: Optional[Dict[str, str]] = None,
        search_db: Literal["question", "text"] = "text",
        **kwargs: Any,
    ) -> Dict[str, Optional[List[Any]]]:
        """Perform similarity search by vector and return relevance scores

        Parameters
        ----------
        embedding: Embedding to search
        k: Number of results to return. Defaults to 4.
        where: dict used to filter results by
                    e.g. {"color" : "red", "price": 4.20}.
            where_document: dict used to filter by the documents.
                    E.g. {$contains: {"text": "hello"}}.
            kwargs: Additional keyword arguments to pass to Chroma collection query.

        Returns
        -------

        """
        db = self._question_db if search_db == "question" else self._text_corpus_db

        results = db._collection.query(
            query_embeddings=embedding,
            n_results=k,
            where=where,
            where_document=where_document,
            **kwargs,
        )

        return results
