"""
Text Augmentation Agent for generating alternative questions and sources.

This agent provides functionality to rephrase questions and source texts while
preserving their original meaning, useful for data augmentation in RAG datasets.
"""

import logging
import random
from typing import List, Optional, Tuple

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
        batch_size : int, optional
            Default batch size for batch processing. Default is 10.

        Attributes
        ----------
        _llm : BaseChatModel
            The language model used for rephrasing.
        _verify_meaning : bool
            Whether to verify meaning preservation.
        _max_retries : int
            Maximum retries for LLM operations.
        batch_size : int
            Default batch size for processing.
        """

        def __init__(
            self,
            llm: BaseChatModel,
            verify_meaning: bool = False,
            max_retries: int = 3,
            batch_size: int = 10,
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
            batch_size : int, optional
                Default batch size for batch processing. Default is 10.
            """
            self._llm = llm
            self._verify_meaning = verify_meaning
            self._max_retries = max_retries
            self.batch_size = batch_size

            # Check if LLM supports batch processing
            self._is_openai_llm = self._check_openai_llm()

        def _check_openai_llm(self) -> bool:
            """Check if the LLM is from OpenAI for batch processing"""
            try:
                from langchain_openai import AzureChatOpenAI, ChatOpenAI

                return isinstance(self._llm, (ChatOpenAI, AzureChatOpenAI))
            except ImportError:
                LOGGER.debug(
                    "langchain_openai not installed, batch processing unavailable"
                )
                return False
            except Exception as e:
                LOGGER.debug(f"Could not determine if LLM is OpenAI: {e}")
                return False

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

        def _invoke_llm_batch(self, prompts: List[str]) -> List[Optional[str]]:
            """
            Invoke the LLM with a batch of prompts.

            Parameters
            ----------
            prompts : List[str]
                List of prompts to send to the LLM.

            Returns
            -------
            List[Optional[str]]
                List of responses, or None for failed items.
            """
            try:
                # Prepare messages
                messages_batch = [[HumanMessage(content=prompt)] for prompt in prompts]

                # Use LangChain's batch method
                responses = self._llm.batch(messages_batch)

                # Extract content from responses
                results = []
                for response in responses:
                    try:
                        result = (
                            response.content
                            if hasattr(response, "content")
                            else str(response)
                        )
                        results.append(result.strip() if result else None)
                    except Exception as e:
                        LOGGER.warning(f"Error extracting response content: {e}")
                        results.append(None)

                return results

            except Exception as e:
                LOGGER.error(f"Batch LLM invocation failed: {e}")
                return [None] * len(prompts)

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

        def _verify_text_equivalence_batch(
            self, original_texts: List[str], rephrased_texts: List[str]
        ) -> List[bool]:
            """
            Verify meaning preservation for batch of text pairs.

            Parameters
            ----------
            original_texts : List[str]
                List of original texts.
            rephrased_texts : List[str]
                List of rephrased texts.

            Returns
            -------
            List[bool]
                List of verification results.
            """
            if not self._verify_meaning:
                return [True] * len(original_texts)

            prompts = [
                VERIFY_MEANING_PRESERVATION_PROMPT.format(
                    original_text=orig, rephrased_text=reph
                )
                for orig, reph in zip(original_texts, rephrased_texts)
            ]

            results = self._invoke_llm_batch(prompts)

            return [
                "EQUIVALENT" in result.upper() if result else False
                for result in results
            ]

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

        def rephrase_full_text_batch(
            self, texts: List[str], verify: Optional[bool] = None
        ) -> List[Optional[str]]:
            """
            Rephrase multiple texts while preserving their meanings.

            Parameters
            ----------
            texts : List[str]
                List of texts to rephrase.
            verify : Optional[bool], optional
                Whether to verify meaning preservation. If None, uses instance default.

            Returns
            -------
            List[Optional[str]]
                List of rephrased texts, or None for failed items.
            """
            if not self._is_openai_llm:
                LOGGER.info(
                    "Batch processing not available, using sequential processing"
                )
                return [self.rephrase_full_text(text, verify) for text in texts]

            # Filter empty texts
            valid_indices = []
            valid_texts = []
            for i, text in enumerate(texts):
                if text and text.strip():
                    valid_indices.append(i)
                    valid_texts.append(text)

            if not valid_texts:
                LOGGER.warning("No valid texts provided for batch rephrasing")
                return [None] * len(texts)

            # Prepare prompts
            prompts = [
                FULL_TEXT_REPHRASE_PROMPT.format(text=text) for text in valid_texts
            ]

            LOGGER.info(f"Batch rephrasing {len(prompts)} texts")

            try:
                # Get batch responses
                rephrased_texts = self._invoke_llm_batch(prompts)

                # Verify if needed
                verify_flag = verify if verify is not None else self._verify_meaning
                if verify_flag:
                    # Filter out None results for verification
                    texts_to_verify = []
                    rephrased_to_verify = []
                    verify_indices = []

                    for i, rephrased in enumerate(rephrased_texts):
                        if rephrased is not None:
                            texts_to_verify.append(valid_texts[i])
                            rephrased_to_verify.append(rephrased)
                            verify_indices.append(i)

                    if texts_to_verify:
                        verification_results = self._verify_text_equivalence_batch(
                            texts_to_verify, rephrased_to_verify
                        )

                        # Set failed verifications to None
                        for i, verified in zip(verify_indices, verification_results):
                            if not verified:
                                LOGGER.warning(
                                    f"Rephrased text {i} failed meaning verification"
                                )
                                rephrased_texts[i] = None

                # Build final results array
                results = [None] * len(texts)
                for i, idx in enumerate(valid_indices):
                    results[idx] = rephrased_texts[i]

                return results

            except Exception as e:
                LOGGER.error(
                    f"Batch rephrasing failed: {e}, falling back to sequential"
                )
                return [self.rephrase_full_text(text, verify) for text in texts]

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

        def rephrase_random_sentence_batch(
            self, texts: List[str], verify: Optional[bool] = None
        ) -> List[Optional[str]]:
            """
            Rephrase a random sentence in multiple texts.

            Parameters
            ----------
            texts : List[str]
                List of texts to process.
            verify : Optional[bool], optional
                Whether to verify meaning preservation. If None, uses instance default.

            Returns
            -------
            List[Optional[str]]
                List of texts with one sentence rephrased, or None for failed items.
            """
            if not self._is_openai_llm:
                LOGGER.info(
                    "Batch processing not available, using sequential processing"
                )
                return [self.rephrase_random_sentence(text, verify) for text in texts]

            # Filter and prepare texts
            valid_indices = []
            valid_texts = []
            selected_sentences = []

            for i, text in enumerate(texts):
                if not text or not text.strip():
                    continue

                sentences = split_into_sentences(text)
                if len(sentences) == 0:
                    continue

                valid_indices.append(i)
                valid_texts.append(text)
                selected_sentences.append(random.choice(sentences))

            if not valid_texts:
                LOGGER.warning("No valid texts provided for batch sentence rephrasing")
                return [None] * len(texts)

            # Prepare prompts
            prompts = [
                SENTENCE_REPHRASE_PROMPT.format(text=text, sentence=sentence)
                for text, sentence in zip(valid_texts, selected_sentences)
            ]

            LOGGER.info(f"Batch rephrasing sentences in {len(prompts)} texts")

            try:
                # Get batch responses
                rephrased_texts = self._invoke_llm_batch(prompts)

                # Verify if needed
                verify_flag = verify if verify is not None else self._verify_meaning
                if verify_flag:
                    texts_to_verify = []
                    rephrased_to_verify = []
                    verify_indices = []

                    for i, rephrased in enumerate(rephrased_texts):
                        if rephrased is not None:
                            texts_to_verify.append(valid_texts[i])
                            rephrased_to_verify.append(rephrased)
                            verify_indices.append(i)

                    if texts_to_verify:
                        verification_results = self._verify_text_equivalence_batch(
                            texts_to_verify, rephrased_to_verify
                        )

                        for i, verified in zip(verify_indices, verification_results):
                            if not verified:
                                LOGGER.warning(
                                    f"Rephrased text {i} failed meaning verification"
                                )
                                rephrased_texts[i] = None

                # Build final results
                results = [None] * len(texts)
                for i, idx in enumerate(valid_indices):
                    results[idx] = rephrased_texts[i]

                return results

            except Exception as e:
                LOGGER.error(
                    f"Batch sentence rephrasing failed: {e}, falling back to sequential"
                )
                return [self.rephrase_random_sentence(text, verify) for text in texts]

        def augment_dataframe(
            self,
            df: pd.DataFrame,
            rephrase_question: bool = True,
            rephrase_source: bool = True,
            rephrase_mode: str = "random",
            probability: float = 0.5,
            columns_mapping: Optional[dict] = None,
            use_batch_processing: bool = True,
            batch_size: Optional[int] = None,
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
            use_batch_processing : bool, optional
                Use batch processing if available. Default is True.
            batch_size : int, optional
                Batch size for processing. If None, uses instance default.

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

            batch_size = batch_size or self.batch_size
            should_batch = use_batch_processing and self._is_openai_llm

            LOGGER.info("Starting augmentation of %d rows", len(df))

            if should_batch:
                return self._augment_dataframe_batch(
                    df,
                    col_map,
                    rephrase_question,
                    rephrase_source,
                    rephrase_mode,
                    probability,
                    batch_size,
                )
            else:
                return self._augment_dataframe_sequential(
                    df,
                    col_map,
                    rephrase_question,
                    rephrase_source,
                    rephrase_mode,
                    probability,
                )

        def _augment_dataframe_sequential(
            self,
            df: pd.DataFrame,
            col_map: dict,
            rephrase_question: bool,
            rephrase_source: bool,
            rephrase_mode: str,
            probability: float,
        ) -> pd.DataFrame:
            """Process DataFrame sequentially (original implementation)"""
            augmented_rows = []

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

        def _augment_dataframe_batch(
            self,
            df: pd.DataFrame,
            col_map: dict,
            rephrase_question: bool,
            rephrase_source: bool,
            rephrase_mode: str,
            probability: float,
            batch_size: int,
        ) -> pd.DataFrame:
            """Process DataFrame using batch processing"""
            # First, determine which rows to process
            rows_to_augment = []
            row_indices = []
            row_modes = []

            for idx, row in df.iterrows():
                if random.random() <= probability:
                    rows_to_augment.append(row)
                    row_indices.append(idx)
                    # Determine mode for this row
                    if rephrase_mode == "random":
                        row_modes.append(random.choice(["full", "sentence"]))
                    else:
                        row_modes.append(rephrase_mode)

            if not rows_to_augment:
                LOGGER.warning("No rows selected for augmentation")
                return df

            LOGGER.info(f"Augmenting {len(rows_to_augment)} rows in batches")

            augmented_rows = []
            total_batches = (len(rows_to_augment) + batch_size - 1) // batch_size

            for batch_start in tqdm(
                range(0, len(rows_to_augment), batch_size),
                total=total_batches,
                desc="Processing batches",
            ):
                batch_end = min(batch_start + batch_size, len(rows_to_augment))
                batch_rows = rows_to_augment[batch_start:batch_end]
                batch_modes = row_modes[batch_start:batch_end]

                # Separate into full and sentence mode batches
                full_mode_indices = [
                    i for i, mode in enumerate(batch_modes) if mode == "full"
                ]
                sentence_mode_indices = [
                    i for i, mode in enumerate(batch_modes) if mode == "sentence"
                ]

                # Process questions
                question_rephrased = {}
                if rephrase_question:
                    questions = [
                        str(row[col_map["question_text"]])
                        for row in batch_rows
                        if pd.notna(row[col_map["question_text"]])
                        and str(row[col_map["question_text"]]).strip()
                    ]

                    if questions:
                        # Batch for full mode
                        if full_mode_indices:
                            full_questions = [
                                str(batch_rows[i][col_map["question_text"]])
                                for i in full_mode_indices
                                if pd.notna(batch_rows[i][col_map["question_text"]])
                            ]
                            full_results = self.rephrase_full_text_batch(full_questions)
                            for i, result in zip(full_mode_indices, full_results):
                                if result:
                                    question_rephrased[i] = result

                        # Batch for sentence mode
                        if sentence_mode_indices:
                            sentence_questions = [
                                str(batch_rows[i][col_map["question_text"]])
                                for i in sentence_mode_indices
                                if pd.notna(batch_rows[i][col_map["question_text"]])
                            ]
                            sentence_results = self.rephrase_random_sentence_batch(
                                sentence_questions
                            )
                            for i, result in zip(
                                sentence_mode_indices, sentence_results
                            ):
                                if result:
                                    question_rephrased[i] = result

                # Process sources
                source_rephrased = {}
                if rephrase_source:
                    sources = [
                        str(row[col_map["source_text"]])
                        for row in batch_rows
                        if pd.notna(row[col_map["source_text"]])
                        and str(row[col_map["source_text"]]).strip()
                    ]

                    if sources:
                        # Batch for full mode
                        if full_mode_indices:
                            full_sources = [
                                str(batch_rows[i][col_map["source_text"]])
                                for i in full_mode_indices
                                if pd.notna(batch_rows[i][col_map["source_text"]])
                            ]
                            full_results = self.rephrase_full_text_batch(full_sources)
                            for i, result in zip(full_mode_indices, full_results):
                                if result:
                                    source_rephrased[i] = result

                        # Batch for sentence mode
                        if sentence_mode_indices:
                            sentence_sources = [
                                str(batch_rows[i][col_map["source_text"]])
                                for i in sentence_mode_indices
                                if pd.notna(batch_rows[i][col_map["source_text"]])
                            ]
                            sentence_results = self.rephrase_random_sentence_batch(
                                sentence_sources
                            )
                            for i, result in zip(
                                sentence_mode_indices, sentence_results
                            ):
                                if result:
                                    source_rephrased[i] = result

                # Build augmented rows
                for i, row in enumerate(batch_rows):
                    modified = False
                    augmented_row = row.copy()

                    if i in question_rephrased:
                        augmented_row[col_map["question_text"]] = question_rephrased[i]
                        modified = True

                    if i in source_rephrased:
                        augmented_row[col_map["source_text"]] = source_rephrased[i]
                        modified = True

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
            use_batch_processing: bool = True,
            batch_size: Optional[int] = None,
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
            use_batch_processing : bool, optional
                Use batch processing if available. Default is True.
            batch_size : int, optional
                Batch size for processing.

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
                use_batch_processing=use_batch_processing,
                batch_size=batch_size,
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
