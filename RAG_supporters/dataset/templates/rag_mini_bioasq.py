import os
import json
from typing import List
from itertools import combinations, product
import random


from dataset.rag_dataset import BaseRAGDatasetGenerator, SampleTripletRAGChroma

from datasets import load_dataset
from tqdm import tqdm
from langchain_chroma import Chroma

try:
    from langchain_openai import OpenAIEmbeddings

except ImportError:

    def OpenAIEmbeddings(*args, **kwargs):
        raise ImportError(
            "OpenAIEmbeddings not found. Please install langchain_openai if you want to use OpenAIEmbeddings"
        )


class RagMiniBioASQBase(BaseRAGDatasetGenerator):

    def __init__(self, dataset_dir: str, embed_function, **kwargs):
        super(RagMiniBioASQBase, self).__init__(dataset_dir)
        self.load_dataset()
        self._embed_function = (
            embed_function
            if embed_function is not None
            else OpenAIEmbeddings(
                openai_api_key=kwargs.get(
                    "openai_api_key", os.getenv("OPENAI_API_KEY")
                ),
                model=kwargs.get("model", "text-embedding-3-small"),
            )
        )
        self._passage_id_to_chroma_id = {}

    def load_dataset(self):
        # init dataset chroma or load existing
        self._question_db = Chroma(
            embedding_function=self._embed_function,
            persist_directory=os.path.join(self._dataset_dir, "question_db"),
        )
        self._text_corpus_db = Chroma(
            embedding_function=self._embed_function,
            persist_directory=os.path.join(self._dataset_dir, "text_corpus_db"),
        )

        if len(self._text_corpus_db.get()["ids"]) == 0:
            self._init_text_corpus_db()
        else:
            # text corpus inited - only load '_passage_id_to_chroma_id'
            if not os.path.exists(
                os.path.join(self._dataset_dir, "passage_id_to_chroma_id.json")
            ):
                raise FileNotFoundError("passage_id_to_chroma_id.json not found")

            # TODO: Check if necessery to store 'passage_id_to_chroma_id'
            with open(
                os.path.join(self._dataset_dir, "passage_id_to_chroma_id.json"), "r"
            ) as f:
                self._passage_id_to_chroma_id = json.load(f)

                if len(self._passage_id_to_chroma_id) == 0:
                    raise ValueError("passage_id_to_chroma_id.json is empty")

            # with open(os.path.join(self._dataset_dir, "additional_data.yaml"), "r") as f:

        # Check if db is empty and load dataset if necessary
        if len(self._question_db.get()["ids"]) == 0:
            self._init_questions_db()

    def validate_dataset(self):
        """Validate if dataset is correctly loaded:
        - Datasets are not empty
        - All questions must have 'relevant_chroma_ids' metadata
        """
        # Check if databases are not empty
        text_corpus_docs = self._text_corpus_db.get(include=["ids"])
        if len(text_corpus_docs["ids"]) == 0:
            raise ValueError("Text corpus database is empty")

        question_docs = self._question_db.get(include=["ids"])
        if len(question_docs["ids"]) == 0:
            raise ValueError("Question database is empty")

        # Check if all questions have relevant_chroma_ids metadata
        for i, doc_id in enumerate(question_docs["ids"]):
            metadata = question_docs["metadatas"][i]
            if not metadata.get("relevant_chroma_ids", None):
                raise ValueError(
                    f"Question with ID {doc_id} does not have 'relevant_chroma_ids' metadata"
                )
                # TODO: Think about deleting record instead of raising error

    def _init_text_corpus_db(self, batch_size: int = 10):
        dataset = load_dataset("rag-datasets/rag-mini-bioasq", "text-corpus")[
            "passages"
        ]
        batch_list = []
        batch_metadata = []

        self._passage_id_to_chroma_id = {}

        for i, (passage, pid) in enumerate(
            tqdm(zip(dataset["passage"], dataset["id"]), desc="Loading text corpus")
        ):
            metadata = {"id": pid}
            batch_list.append(passage)
            batch_metadata.append(metadata)

            if i % batch_size == 0 and i > 0:
                chroma_keys = self._text_corpus_db.add_texts(batch_list, batch_metadata)
                batch_list = []
                batch_metadata = []
                for key, value in zip(chroma_keys, batch_metadata):
                    # TODO: Assure ids are in the same order
                    self._passage_id_to_chroma_id[value["id"]] = key

        if len(batch_list) > 0:
            chroma_keys = self._text_corpus_db.add_texts(batch_list, batch_metadata)
            for key, value in zip(chroma_keys, batch_metadata):
                self._passage_id_to_chroma_id[value["id"]] = key

        # Save passage_id_to_chroma_id mapping to file
        with open(
            os.path.join(self._dataset_dir, "passage_id_to_chroma_id.json"), "w"
        ) as f:
            json.dump(self._passage_id_to_chroma_id, f)

    def _init_questions_db(self, batch_size: int = 10):
        dataset = load_dataset(
            "rag-datasets/rag-mini-bioasq", "question-answer-passages"
        )["test"]

        batch_list = []
        batch_metadata = []

        for i, (question, qid, relevant_ids) in enumerate(
            tqdm(
                zip(
                    dataset["question"], dataset["id"], dataset["relevant_passage_ids"]
                ),
                desc="Loading dataset",
            )
        ):
            metadata = {"id": qid, "relevant_ids": relevant_ids}
            batch_list.append(question)
            batch_metadata.append(metadata)

            # Store text corpus ids for easier processing later
            metadata["relevant_chroma_ids"] = [
                self._passage_id_to_chroma_id[pid] for pid in relevant_ids
            ]

            if i % batch_size == 0 and i > 0:
                self._question_db.add_texts(batch_list, batch_metadata)
                batch_list = []
                batch_metadata = []

        if len(batch_list) > 0:
            self._question_db.add_texts(batch_list, batch_metadata)  # leftovers

    def _passage_id_text_to_chroma_id(self, passage_id: str) -> str:
        if not self._passage_id_to_chroma_id:
            try:
                # TODO: should warn about loading from file?
                with open(
                    os.path.join(self._dataset_dir, "passage_id_to_chroma_id.json"), "r"
                ) as f:
                    self._passage_id_to_chroma_id = json.load(f)
            except FileNotFoundError:
                raise FileNotFoundError("passage_id_to_chroma_id.json not found")

        if passage_id not in self._passage_id_to_chroma_id:
            raise ValueError(
                f"Passage with ID {passage_id} not found in text corpus database"
            )

        return self._passage_id_to_chroma_id[passage_id]

    def _generate_positive_triplet_samples(
        self, question_embedding, relevant_passage_ids, **kwargs
    ) -> List[SampleTripletRAGChroma]:
        """
        Generate triplets consisting of question and two relevant passages (if both assigned to the same question in database)
        """
        sample_triplets = []

        if len(relevant_passage_ids) > 1:
            relevant_combinations = list(combinations(relevant_passage_ids, 2))
            for pid_1, pid_2 in relevant_combinations:
                sample_triplets.append(
                    SampleTripletRAGChroma(
                        question_id=question_embedding,
                        answer_id_1=self._passage_id_text_to_chroma_id(pid_1),
                        answer_id_2=self._passage_id_text_to_chroma_id(pid_2),
                        label=-1,
                    )
                )

        return sample_triplets

    def _generate_contrastive_triplet_samples(
        self,
        question_embedding,
        relevant_passage_ids,
        num_negative_samples: int = 5,
        keep_same_negatives=False,
        assume_relevant_best=True,
        **kwargs,
    ) -> List[SampleTripletRAGChroma]:
        """
        Generate triplets consisting of question, its assigned relevant passage and randomly chosen passage not assigned to the question
        """
        # TODO: move possible negatives to generator or class atribute to avoid recomputing
        possible_negatives = [
            pid
            for pid in self._passage_id_to_chroma_id.keys()
            if pid not in relevant_passage_ids
        ]
        sample_triplets = []

        if keep_same_negatives:
            negatives_picked = random.sample(possible_negatives, num_negative_samples)
            passage_combinations = list(product(relevant_passage_ids, negatives_picked))
            for pid_1, pid_2 in passage_combinations:

                sample_triplets.append(
                    SampleTripletRAGChroma(
                        question_id=question_embedding,
                        answer_id_1=self._passage_id_text_to_chroma_id(pid_1),
                        answer_id_2=self._passage_id_text_to_chroma_id(pid_2),
                        label=1 if assume_relevant_best else -1,
                    )
                )

    def _generate_similar_triplet_samples(
        self,
        question_embedding,
        relevant_passage_ids,
        score_threshold=0.3,
        assume_relevant_best=True,
        **kwargs,
    ) -> List[SampleTripletRAGChroma]:
        """
        Generate triplets consisting of question, its relevant passage and randomly chosen passage assigned to the same question
        close in embedding space to question
        """
        sample_triplets = []

        close_questions = self._raw_similarity_search_by_vector(
            question_embedding, k=kwargs.get("top_k", 3), includes=["ids", "distances"]
        )
        zipped_questions = zip(close_questions["ids"], close_questions["distances"])
        picked_questions = [
            qid
            for qid, dist in zipped_questions
            if dist < score_threshold and qid not in relevant_passage_ids
        ]

        passage_combinations = list(product(relevant_passage_ids, picked_questions))
        for pid_1, pid_2 in passage_combinations:
            sample_triplets.append(
                SampleTripletRAGChroma(
                    question_id=question_embedding,
                    answer_id_1=self._passage_id_text_to_chroma_id(pid_1),
                    answer_id_2=self._passage_id_text_to_chroma_id(pid_2),
                    label=1 if assume_relevant_best else -1,
                )
            )
