"""
Text Augmentation Agent for generating alternative questions and sources.

This agent provides functionality to rephrase questions and source texts while
preserving their original meaning, useful for data augmentation in RAG datasets.
"""

import logging
import random
from typing import List, Optional

LOGGER = logging.getLogger(__name__)

try:
    import pandas as pd
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import HumanMessage
    from tqdm import tqdm

    from prompts_templates.text_augmentation import (
        FULL_TEXT_REPHRASE_PROMPT,
        SENTENCE_REPHRASE_PROMPT,
        VERIFY_MEANING_PRESERVATION_PROMPT,
    )
    from utils.text_splitters import split_into_sentences

    class TextAugmentationAgent:
        """
        Agent for augmenting text data through rephrasing.

        This agent can rephrase entire texts or specific sentences within texts
        while preserving the original meaning. It's designed to work with CSV
        datasets containing questions and sources for RAG training.

        Parameters
        ----------
        llm : BaseChatModel
            Language model instance for performing text rephrasing.
        verify_meaning : bool, optional
            Whether to verify that rephrased text preserves meaning. Default is False.
        max_retries : int, optional
            Maximum number of retries for LLM calls. Default is 3.

        Attributes
        ----------
        _llm : BaseChatModel
            The language model used for rephrasing.
        _verify_meaning : bool
            Whether to verify meaning preservation.
        _max_retries : int
            Maximum retries for LLM operations.
        """

        def __init__(
            self,
            llm: BaseChatModel,
            verify_meaning: bool = False,
            max_retries: int = 3,
        ):
            """
            Initialize the TextAugmentationAgent.

            Parameters
            ----------
            llm : BaseChatModel
                Language model instance for performing text rephrasing.
            verify_meaning : bool, optional
                Whether to verify that rephrased text preserves meaning. Default is False.
            max_retries : int, optional
                Maximum number of retries for LLM calls. Default is 3.
            """
            self._llm = llm
            self._verify_meaning = verify_meaning
            self._max_retries = max_retries

        def _invoke_llm_with_retry(self, prompt: str) -> Optional[str]:
            """
            Invoke the LLM with retry logic.

            Parameters
            ----------
            prompt : str
                The prompt to send to the LLM.

            Returns
            -------
            Optional[str]
                The LLM's response, or None if all retries failed.
            """
            for attempt in range(self._max_retries):
                try:
                    message = HumanMessage(content=prompt)
                    response = self._llm.invoke([message])
                    result = (
                        response.content
                        if hasattr(response, "content")
                        else str(response)
                    )
                    return result.strip()
                except Exception as e:  # pylint: disable=broad-exception-caught
                    LOGGER.warning(
                        "LLM invocation failed (attempt %d/%d): %s",
                        attempt + 1,
                        self._max_retries,
                        str(e),
                    )
                    if attempt == self._max_retries - 1:
                        LOGGER.error(
                            "All %d LLM invocation attempts failed", self._max_retries
                        )
                        return None
            return None

        def _verify_text_equivalence(
            self, original_text: str, rephrased_text: str
        ) -> bool:
            """
            Verify that the rephrased text preserves the original meaning.

            Parameters
            ----------
            original_text : str
                The original text.
            rephrased_text : str
                The rephrased text to verify.

            Returns
            -------
            bool
                True if texts are semantically equivalent, False otherwise.
            """
            if not self._verify_meaning:
                return True

            prompt = VERIFY_MEANING_PRESERVATION_PROMPT.format(
                original_text=original_text, rephrased_text=rephrased_text
            )

            result = self._invoke_llm_with_retry(prompt)
            if result is None:
                LOGGER.warning(
                    "Failed to verify meaning preservation, assuming non-equivalent"
                )
                return False

            return "EQUIVALENT" in result.upper()

        def rephrase_full_text(
            self, text: str, verify: Optional[bool] = None
        ) -> Optional[str]:
            """
            Rephrase an entire text while preserving its meaning.

            Parameters
            ----------
            text : str
                The text to rephrase.
            verify : Optional[bool], optional
                Whether to verify meaning preservation. If None, uses instance default.

            Returns
            -------
            Optional[str]
                The rephrased text, or None if rephrasing failed.
            """
            if not text or not text.strip():
                LOGGER.warning("Empty text provided for rephrasing")
                return None

            prompt = FULL_TEXT_REPHRASE_PROMPT.format(text=text)
            rephrased = self._invoke_llm_with_retry(prompt)

            if rephrased is None:
                return None

            # Verify meaning preservation if requested
            verify_flag = verify if verify is not None else self._verify_meaning
            if verify_flag:
                if not self._verify_text_equivalence(text, rephrased):
                    LOGGER.warning(
                        "Rephrased text failed meaning verification, returning None"
                    )
                    return None

            return rephrased

        def rephrase_random_sentence(
            self, text: str, verify: Optional[bool] = None
        ) -> Optional[str]:
            """
            Rephrase a randomly selected sentence within the text.

            Parameters
            ----------
            text : str
                The text containing sentences to rephrase.
            verify : Optional[bool], optional
                Whether to verify meaning preservation. If None, uses instance default.

            Returns
            -------
            Optional[str]
                The text with one sentence rephrased, or None if rephrasing failed.
            """
            if not text or not text.strip():
                LOGGER.warning("Empty text provided for sentence rephrasing")
                return None

            sentences = split_into_sentences(text)
            if len(sentences) == 0:
                LOGGER.warning("No sentences found in text")
                return None

            # Select a random sentence to rephrase
            selected_sentence = random.choice(sentences)

            prompt = SENTENCE_REPHRASE_PROMPT.format(
                text=text, sentence=selected_sentence
            )
            rephrased = self._invoke_llm_with_retry(prompt)

            if rephrased is None:
                return None

            # Verify meaning preservation if requested
            verify_flag = verify if verify is not None else self._verify_meaning
            if verify_flag:
                if not self._verify_text_equivalence(text, rephrased):
                    LOGGER.warning(
                        "Rephrased text failed meaning verification, returning None"
                    )
                    return None

            return rephrased

        def augment_dataframe(
            self,
            df: pd.DataFrame,
            rephrase_question: bool = True,
            rephrase_source: bool = True,
            rephrase_mode: str = "random",
            probability: float = 0.5,
            columns_mapping: Optional[dict] = None,
        ) -> pd.DataFrame:
            """
            Augment a DataFrame by adding rephrased versions of questions and/or sources.

            Parameters
            ----------
            df : pd.DataFrame
                DataFrame containing question and source columns.
            rephrase_question : bool, optional
                Whether to rephrase questions. Default is True.
            rephrase_source : bool, optional
                Whether to rephrase sources. Default is True.
            rephrase_mode : str, optional
                Rephrasing mode: 'full' (entire text), 'sentence' (random sentence),
                or 'random' (randomly choose between full and sentence). Default is 'random'.
            probability : float, optional
                Probability of applying rephrasing to each row. Default is 0.5.
            columns_mapping : Optional[dict], optional
                Mapping of expected column names. Should contain:
                'question_text' (default: 'question_text'),
                'source_text' (default: 'source_text').

            Returns
            -------
            pd.DataFrame
                New DataFrame with augmented rows added to the original data.
            """
            if df.empty:
                LOGGER.warning("Empty DataFrame provided")
                return df

            # Set up column mapping
            default_mapping = {
                "question_text": "question_text",
                "source_text": "source_text",
            }
            col_map = {**default_mapping, **(columns_mapping or {})}

            # Validate columns exist
            for col_name in col_map.values():
                if col_name not in df.columns:
                    raise ValueError(
                        f"Column '{col_name}' not found in DataFrame. Available columns: {df.columns.tolist()}"
                    )

            valid_modes = ["full", "sentence", "random"]
            if rephrase_mode not in valid_modes:
                raise ValueError(
                    f"Invalid rephrase_mode: {rephrase_mode}. Must be one of {valid_modes}"
                )

            augmented_rows = []

            LOGGER.info("Starting augmentation of %d rows", len(df))

            for idx, row in tqdm(
                df.iterrows(), total=len(df), desc="Augmenting dataset"
            ):
                # Skip augmentation based on probability
                if random.random() > probability:
                    continue

                # Create a copy of the row for augmentation
                augmented_row = row.copy()
                modified = False

                # Determine rephrasing mode for this row
                current_mode = rephrase_mode
                if rephrase_mode == "random":
                    current_mode = random.choice(["full", "sentence"])

                # Rephrase question if requested
                if rephrase_question:
                    question_text = row[col_map["question_text"]]
                    if pd.notna(question_text) and str(question_text).strip():
                        if current_mode == "full":
                            rephrased = self.rephrase_full_text(str(question_text))
                        else:
                            rephrased = self.rephrase_random_sentence(
                                str(question_text)
                            )

                        if rephrased:
                            augmented_row[col_map["question_text"]] = rephrased
                            modified = True
                        else:
                            LOGGER.warning(
                                "Failed to rephrase question at index %s", idx
                            )

                # Rephrase source if requested
                if rephrase_source:
                    source_text = row[col_map["source_text"]]
                    if pd.notna(source_text) and str(source_text).strip():
                        if current_mode == "full":
                            rephrased = self.rephrase_full_text(str(source_text))
                        else:
                            rephrased = self.rephrase_random_sentence(str(source_text))

                        if rephrased:
                            augmented_row[col_map["source_text"]] = rephrased
                            modified = True
                        else:
                            LOGGER.warning("Failed to rephrase source at index %s", idx)

                # Only add row if something was successfully modified
                if modified:
                    augmented_rows.append(augmented_row)

            if augmented_rows:
                augmented_df = pd.DataFrame(augmented_rows)
                result_df = pd.concat([df, augmented_df], ignore_index=True)
                LOGGER.info(
                    "Added %d augmented rows. Total rows: %d",
                    len(augmented_rows),
                    len(result_df),
                )
                return result_df

            LOGGER.warning("No rows were successfully augmented")
            return df

        def process_csv(
            self,
            input_csv_path: str,
            output_csv_path: Optional[str] = None,
            rephrase_question: bool = True,
            rephrase_source: bool = True,
            rephrase_mode: str = "random",
            probability: float = 0.5,
            columns_mapping: Optional[dict] = None,
        ) -> pd.DataFrame:
            """
            Process a CSV file by augmenting it with rephrased versions.

            Parameters
            ----------
            input_csv_path : str
                Path to the input CSV file.
            output_csv_path : Optional[str], optional
                Path to save the augmented CSV. If None, overwrites input file.
            rephrase_question : bool, optional
                Whether to rephrase questions. Default is True.
            rephrase_source : bool, optional
                Whether to rephrase sources. Default is True.
            rephrase_mode : str, optional
                Rephrasing mode: 'full', 'sentence', or 'random'. Default is 'random'.
            probability : float, optional
                Probability of applying rephrasing to each row. Default is 0.5.
            columns_mapping : Optional[dict], optional
                Mapping of expected column names.

            Returns
            -------
            pd.DataFrame
                The augmented DataFrame.
            """
            LOGGER.info("Loading CSV from %s", input_csv_path)
            df = pd.read_csv(input_csv_path)

            augmented_df = self.augment_dataframe(
                df,
                rephrase_question=rephrase_question,
                rephrase_source=rephrase_source,
                rephrase_mode=rephrase_mode,
                probability=probability,
                columns_mapping=columns_mapping,
            )

            output_path = output_csv_path or input_csv_path
            augmented_df.to_csv(output_path, index=False, encoding="utf-8")
            LOGGER.info("Augmented CSV saved to %s", output_path)

            return augmented_df

except ImportError as e:
    _DEPENDENCIES_AVAILABLE = False
    _IMPORT_ERROR = str(e)

    LOGGER.warning(
        "TextAugmentationAgent dependencies not available: %s. "
        "Install with: pip install langchain langchain_core tqdm pandas",
        e,
    )

    class TextAugmentationAgent:
        """
        Placeholder for TextAugmentationAgent when dependencies are missing.

        To use this agent, install required dependencies:
            pip install langchain langchain_core tqdm pandas
        """

        def __init__(self, *args, **kwargs):
            """Raise ImportError for missing dependencies."""
            raise ImportError(
                f"TextAugmentationAgent requires langchain_core and related dependencies to be installed.\n"
                f"Original import error: {_IMPORT_ERROR}\n"
                f"Install with: pip install langchain langchain_core tqdm pandas"
            )

        def __getattr__(self, name):
            """Raise ImportError for missing dependencies."""
            raise ImportError(
                f"TextAugmentationAgent not available due to missing dependencies: {_IMPORT_ERROR}"
            )
