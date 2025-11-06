"""
Question Augmentation Agent for question rephrasing and alternative question generation.

This agent provides functionality to:
1. Rephrase questions to align with source context and domain terminology
2. Generate alternative questions based on source content
"""

import json
import logging
from typing import List, Optional

LOGGER = logging.getLogger(__name__)

try:
    import pandas as pd
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import HumanMessage
    from tqdm import tqdm

    from prompts_templates.text_augmentation import (
        ALTERNATIVE_QUESTIONS_GENERATION_PROMPT,
        CONTEXTUAL_QUESTION_PROMPT,
        QUESTION_REPHRASE_WITH_SOURCE_PROMPT,
    )

    class QuestionAugmentationAgent:
        """
        Agent for question generation and rephrasing in CSV/DataFrame contexts.

        This agent can rephrase questions to align with source content and domain,
        and generate alternative questions based on provided sources. It's designed
        to work with CSV datasets containing questions and sources for RAG training.

        Parameters
        ----------
        llm : BaseChatModel
            Language model instance for performing question operations.
        max_retries : int, optional
            Maximum number of retries for LLM calls. Default is 3.
        batch_size : int, optional
            Default batch size for batch processing. Default is 10.

        Attributes
        ----------
        _llm : BaseChatModel
            The language model used for question operations.
        _max_retries : int
            Maximum retries for LLM operations.
        batch_size : int
            Default batch size for processing.
        """

        def __init__(
            self,
            llm: BaseChatModel,
            max_retries: int = 3,
            batch_size: int = 10,
        ):
            """
            Initialize the QuestionAugmentationAgent.

            Parameters
            ----------
            llm : BaseChatModel
                Language model instance for performing question operations.
            max_retries : int, optional
                Maximum number of retries for LLM calls. Default is 3.
            batch_size : int, optional
                Default batch size for batch processing. Default is 10.
            """
            # Validate LLM
            if llm is None:
                raise ValueError("llm parameter cannot be None")
            if not isinstance(llm, BaseChatModel):
                raise TypeError(
                    f"llm must be a BaseChatModel instance, got {type(llm).__name__}"
                )

            self._llm = llm
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

                    # Validate response content
                    if hasattr(response, "content"):
                        result = response.content
                    else:
                        result = str(response) if response else None

                    if result is None:
                        LOGGER.warning(
                            f"LLM returned None content (attempt {attempt + 1}/{self._max_retries})"
                        )
                        continue

                    return result.strip()

                except Exception as e:  # pylint: disable=broad-exception-caught
                    LOGGER.warning(
                        f"LLM invocation failed (attempt {attempt + 1}/{self._max_retries}): {str(e)}"
                    )
                    if attempt == self._max_retries - 1:
                        LOGGER.error(
                            f"All {self._max_retries} LLM invocation attempts failed"
                        )
                        return None
            return None

        def rephrase_question_with_source(
            self, question: str, source: str, allow_vague: bool = False
        ) -> Optional[str]:
            """
            Rephrase a question to align with the terminology and context of a source.

            This method takes a question and a source text, and rephrases the question
            to better fit the domain, terminology, and context found in the source
            while preserving the original question's meaning.

            Parameters
            ----------
            question : str
                The original question to rephrase.
            source : str
                The source text providing context and domain terminology.
            allow_vague : bool, optional
                If True, allows the rephrased question to use less precise language.
                If False (default), ensures the question is clear and specific.

            Returns
            -------
            Optional[str]
                The rephrased question, or None if rephrasing failed.

            Examples
            --------
            >>> agent = QuestionAugmentationAgent(llm=my_llm)
            >>> question = "What does it do?"
            >>> source = "Mitochondria are organelles that generate ATP..."
            >>> rephrased = agent.rephrase_question_with_source(question, source)
            >>> print(rephrased)
            "What is the primary function of mitochondria?"
            """
            if not question or not question.strip():
                LOGGER.warning("Empty question provided for rephrasing")
                return None

            if not source or not source.strip():
                LOGGER.warning("Empty source provided for question rephrasing")
                return None

            clarity_instruction = (
                "The question can use less precise or vague language if it sounds more natural."
                if allow_vague
                else "Ensure the rephrased question is clear, specific, and answerable based on the source."
            )

            prompt = QUESTION_REPHRASE_WITH_SOURCE_PROMPT.format(
                question=question,
                source=source,
                clarity_instruction=clarity_instruction,
            )
            rephrased = self._invoke_llm_with_retry(prompt)

            if rephrased is None:
                LOGGER.error("Failed to rephrase question with source")
                return None

            return rephrased

        def rephrase_question_with_domain(
            self, question: str, domain: str, allow_vague: bool = False
        ) -> Optional[str]:
            """
            Rephrase a question to align with a specific domain or context.

            This method takes a question and a domain description, and rephrases
            the question to better fit the terminology and conventions of that domain.

            Parameters
            ----------
            question : str
                The original question to rephrase.
            domain : str
                The domain or context description (e.g., "biology", "machine learning").
            allow_vague : bool, optional
                If True, allows the rephrased question to use less precise language.
                If False (default), ensures the question is clear and specific.

            Returns
            -------
            Optional[str]
                The rephrased question, or None if rephrasing failed.

            Examples
            --------
            >>> agent = QuestionAugmentationAgent(llm=my_llm)
            >>> question = "How do you make it learn?"
            >>> domain = "machine learning"
            >>> rephrased = agent.rephrase_question_with_domain(question, domain)
            >>> print(rephrased)
            "How do you train a machine learning model?"
            """
            if not question or not question.strip():
                LOGGER.warning("Empty question provided for rephrasing")
                return None

            if not domain or not domain.strip():
                LOGGER.warning("Empty domain provided for question rephrasing")
                return None

            clarity_instruction = (
                "The question can use less precise or vague language if it sounds more natural."
                if allow_vague
                else "Ensure the question remains clear and specific."
            )

            prompt = CONTEXTUAL_QUESTION_PROMPT.format(
                question=question,
                domain=domain,
                clarity_instruction=clarity_instruction,
            )
            rephrased = self._invoke_llm_with_retry(prompt)

            if rephrased is None:
                LOGGER.error("Failed to rephrase question with domain")
                return None

            return rephrased

        def generate_alternative_questions(
            self, source: str, n: int = 5, allow_vague: bool = False
        ) -> Optional[List[str]]:
            """
            Generate n alternative questions that can be answered by the source.

            This method analyzes a source text and generates multiple diverse
            questions that can be answered using the information in the source.

            Parameters
            ----------
            source : str
                The source text to generate questions from.
            n : int, optional
                Number of questions to generate. Default is 5.
            allow_vague : bool, optional
                If True, allows generated questions to use less precise language.
                If False (default), ensures questions are specific and focused.

            Returns
            -------
            Optional[List[str]]
                List of generated questions, or None if generation failed.

            Examples
            --------
            >>> agent = QuestionAugmentationAgent(llm=my_llm)
            >>> source = "Photosynthesis is the process by which plants..."
            >>> questions = agent.generate_alternative_questions(source, n=3)
            >>> for q in questions:
            ...     print(q)
            "What is photosynthesis?"
            "How do plants produce energy?"
            "What role does chlorophyll play in photosynthesis?"
            """
            if not source or not source.strip():
                LOGGER.warning("Empty source provided for question generation")
                return None

            if n < 1:
                LOGGER.warning(
                    f"Invalid n value ({n}) for question generation, must be >= 1"
                )
                return None

            if n > 20:
                LOGGER.warning(
                    f"Large n value ({n}) may result in lower quality questions"
                )

            clarity_instruction = (
                "Questions can be less specific or use vague language if it sounds more natural."
                if allow_vague
                else "Questions should be specific and focused, not overly broad or vague."
            )

            prompt = ALTERNATIVE_QUESTIONS_GENERATION_PROMPT.format(
                source=source, n=n, clarity_instruction=clarity_instruction
            )
            response = self._invoke_llm_with_retry(prompt)

            if response is None:
                LOGGER.error("Failed to generate alternative questions")
                return None

            # Parse JSON response
            try:
                data = json.loads(response)
            except json.JSONDecodeError as e:
                LOGGER.error(f"Failed to parse JSON response: {str(e)}")
                LOGGER.debug(f"Response was: {response}")
                return None

            questions = data.get("questions", [])

            if not isinstance(questions, list):
                LOGGER.error("Response JSON 'questions' field is not a list")
                return None

            if len(questions) != n:
                LOGGER.warning(
                    f"Expected {n} questions but received {len(questions)} from JSON response"
                )

            if not questions:
                LOGGER.error("No questions found in JSON response")
                return None

            return questions

        # Batch processing methods

        def rephrase_question_with_source_batch(
            self,
            questions: List[str],
            sources: List[str],
            allow_vague: bool = False,
        ) -> List[Optional[str]]:
            """
            Rephrase multiple questions with their corresponding sources in batch.

            Parameters
            ----------
            questions : List[str]
                List of questions to rephrase.
            sources : List[str]
                List of source texts (must match length of questions).
            allow_vague : bool, optional
                If True, allows rephrased questions to use less precise language.

            Returns
            -------
            List[Optional[str]]
                List of rephrased questions (None for failures).
            """
            if len(questions) != len(sources):
                raise ValueError(
                    f"Length mismatch: {len(questions)} questions vs {len(sources)} sources"
                )

            if not self._is_openai_llm:
                LOGGER.info(
                    "Batch processing not available, using sequential processing"
                )
                return [
                    self.rephrase_question_with_source(q, s, allow_vague)
                    for q, s in zip(questions, sources)
                ]

            return self._batch_process_rephrase_source(questions, sources, allow_vague)

        def rephrase_question_with_domain_batch(
            self,
            questions: List[str],
            domain: str,
            allow_vague: bool = False,
        ) -> List[Optional[str]]:
            """
            Rephrase multiple questions with a domain in batch.

            Parameters
            ----------
            questions : List[str]
                List of questions to rephrase.
            domain : str
                Domain context (same for all questions).
            allow_vague : bool, optional
                If True, allows rephrased questions to use less precise language.

            Returns
            -------
            List[Optional[str]]
                List of rephrased questions (None for failures).
            """
            if not self._is_openai_llm:
                LOGGER.info(
                    "Batch processing not available, using sequential processing"
                )
                return [
                    self.rephrase_question_with_domain(q, domain, allow_vague)
                    for q in questions
                ]

            return self._batch_process_rephrase_domain(questions, domain, allow_vague)

        def generate_alternative_questions_batch(
            self,
            sources: List[str],
            n: int = 5,
            allow_vague: bool = False,
        ) -> List[Optional[List[str]]]:
            """
            Generate alternative questions for multiple sources in batch.

            Parameters
            ----------
            sources : List[str]
                List of source texts.
            n : int, optional
                Number of questions to generate per source. Default is 5.
            allow_vague : bool, optional
                If True, allows generated questions to use less precise language.

            Returns
            -------
            List[Optional[List[str]]]
                List of question lists (None for failures).
            """
            if not self._is_openai_llm:
                LOGGER.info(
                    "Batch processing not available, using sequential processing"
                )
                return [
                    self.generate_alternative_questions(s, n, allow_vague)
                    for s in sources
                ]

            return self._batch_process_generate(sources, n, allow_vague)

        def _batch_process_rephrase_source(
            self,
            questions: List[str],
            sources: List[str],
            allow_vague: bool,
        ) -> List[Optional[str]]:
            """Internal method for batch rephrasing with sources"""
            clarity_instruction = (
                "The question can use less precise or vague language if it sounds more natural."
                if allow_vague
                else "Ensure the rephrased question is clear, specific, and answerable based on the source."
            )

            # Prepare prompts
            prompts = []
            for question, source in zip(questions, sources):
                prompt = QUESTION_REPHRASE_WITH_SOURCE_PROMPT.format(
                    question=question,
                    source=source,
                    clarity_instruction=clarity_instruction,
                )
                prompts.append([HumanMessage(content=prompt)])

            LOGGER.info(
                f"Batch rephrasing {len(prompts)} questions with sources"
            )

            try:
                # Use LangChain's batch method
                responses = self._llm.batch(prompts)

                # Extract results
                results = []
                for i, response in enumerate(responses):
                    try:
                        if hasattr(response, "content"):
                            content = response.content
                        else:
                            content = str(response)

                        if content:
                            results.append(content.strip())
                        else:
                            LOGGER.warning(f"Empty response for batch item {i}")
                            results.append(None)

                    except Exception as e:
                        LOGGER.error(f"Error processing batch item {i}: {e}")
                        results.append(None)

                return results

            except Exception as e:
                LOGGER.error(f"Batch processing failed: {e}")
                LOGGER.info("Falling back to sequential processing")
                return [
                    self.rephrase_question_with_source(q, s, allow_vague)
                    for q, s in zip(questions, sources)
                ]

        def _batch_process_rephrase_domain(
            self,
            questions: List[str],
            domain: str,
            allow_vague: bool,
        ) -> List[Optional[str]]:
            """Internal method for batch rephrasing with domain"""
            # TODO: Move to prompt templates?
            clarity_instruction = (
                "The question can use less precise or vague language if it sounds more natural."
                if allow_vague
                else "Ensure the question remains clear and specific."
            )

            # Prepare prompts
            prompts = []
            for question in questions:
                prompt = CONTEXTUAL_QUESTION_PROMPT.format(
                    question=question,
                    domain=domain,
                    clarity_instruction=clarity_instruction,
                )
                prompts.append([HumanMessage(content=prompt)])

            LOGGER.info(f"Batch rephrasing {len(prompts)} questions with domain")

            try:
                # Use LangChain's batch method
                responses = self._llm.batch(prompts)

                # Extract results
                results = []
                for i, response in enumerate(responses):
                    try:
                        if hasattr(response, "content"):
                            content = response.content
                        else:
                            content = str(response)

                        if content:
                            results.append(content.strip())
                        else:
                            LOGGER.warning(f"Empty response for batch item {i}")
                            results.append(None)

                    except Exception as e:
                        LOGGER.error(f"Error processing batch item {i}: {e}")
                        results.append(None)

                return results

            except Exception as e:
                LOGGER.error(f"Batch processing failed: {e}")
                LOGGER.info("Falling back to sequential processing")
                return [
                    self.rephrase_question_with_domain(q, domain, allow_vague)
                    for q in questions
                ]

        def _batch_process_generate(
            self,
            sources: List[str],
            n: int,
            allow_vague: bool,
        ) -> List[Optional[List[str]]]:
            """Internal method for batch question generation"""
            clarity_instruction = (
                "Questions can be less specific or use vague language if it sounds more natural."
                if allow_vague
                else "Questions should be specific and focused, not overly broad or vague."
            )

            # Prepare prompts
            prompts = []
            for source in sources:
                prompt = ALTERNATIVE_QUESTIONS_GENERATION_PROMPT.format(
                    source=source,
                    n=n,
                    clarity_instruction=clarity_instruction,
                )
                prompts.append([HumanMessage(content=prompt)])

            LOGGER.info(f"Batch generating questions for {len(prompts)} sources")

            try:
                # Use LangChain's batch method
                responses = self._llm.batch(prompts)

                # Parse results
                results = []
                for i, response in enumerate(responses):
                    try:
                        if hasattr(response, "content"):
                            content = response.content
                        else:
                            content = str(response)

                        # Parse JSON
                        data = json.loads(content)
                        questions = data.get("questions", [])

                        if not isinstance(questions, list):
                            LOGGER.error(
                                f"Response 'questions' field is not a list for item {i}"
                            )
                            results.append(None)
                        elif not questions:
                            LOGGER.error(f"No questions found for item {i}")
                            results.append(None)
                        else:
                            results.append(questions)

                    except json.JSONDecodeError as e:
                        LOGGER.error(f"JSON parse error for batch item {i}: {e}")
                        results.append(None)
                    except Exception as e:
                        LOGGER.error(f"Error processing batch item {i}: {e}")
                        results.append(None)

                return results

            except Exception as e:
                LOGGER.error(f"Batch processing failed: {e}")
                LOGGER.info("Falling back to sequential processing")
                return [
                    self.generate_alternative_questions(s, n, allow_vague)
                    for s in sources
                ]

        def process_dataframe_rephrasing(
            self,
            df: pd.DataFrame,
            rephrase_mode: str = "source",
            domain: Optional[str] = None,
            allow_vague: bool = False,
            columns_mapping: Optional[dict] = None,
            use_batch_processing: bool = True,
            batch_size: Optional[int] = None,
            checkpoint_batch_size: Optional[int] = None,
            save_path: Optional[str] = None,
        ) -> pd.DataFrame:
            """
            Process a DataFrame by rephrasing questions based on source or domain.

            Parameters
            ----------
            df : pd.DataFrame
                DataFrame containing question and optionally source columns.
            rephrase_mode : str, optional
                Mode of rephrasing: 'source' (use source text) or 'domain' (use domain).
                Default is 'source'.
            domain : Optional[str], optional
                Domain context for rephrasing when mode is 'domain'.
            allow_vague : bool, optional
                If True, allows rephrased questions to use less precise language.
                If False (default), ensures questions are clear and specific.
            columns_mapping : Optional[dict], optional
                Mapping of expected column names. Should contain:
                'question_text' (default: 'question_text'),
                'source_text' (default: 'source_text' - required for 'source' mode),
                'rephrased_question' (default: 'rephrased_question' - output column).
            use_batch_processing : bool, optional
                Use batch processing if available. Default is True.
            batch_size : int, optional
                Batch size for processing. If None, uses self.batch_size.
            checkpoint_batch_size : int, optional
                Save checkpoint every N rows.
            save_path : str, optional
                Path to save results and checkpoints.

            Returns
            -------
            pd.DataFrame
                DataFrame with added column for rephrased questions.

            Raises
            ------
            ValueError
                If required columns are missing or invalid mode is specified.
            """
            if df.empty:
                LOGGER.warning("Empty DataFrame provided")
                return df

            batch_size = batch_size or self.batch_size
            should_batch = use_batch_processing and self._is_openai_llm

            # Set up column mapping with validation
            default_mapping = {
                "question_text": "question_text",
                "source_text": "source_text",
                "rephrased_question": "rephrased_question",
            }

            # Validate columns_mapping keys
            if columns_mapping:
                invalid_keys = set(columns_mapping.keys()) - set(default_mapping.keys())
                if invalid_keys:
                    raise ValueError(
                        f"Invalid column mapping keys: {invalid_keys}. "
                        f"Valid keys are: {list(default_mapping.keys())}"
                    )

            col_map = {**default_mapping, **(columns_mapping or {})}

            # Validate mode
            valid_modes = ["source", "domain"]
            if rephrase_mode not in valid_modes:
                raise ValueError(
                    f"Invalid rephrase_mode: {rephrase_mode}. Must be one of {valid_modes}"
                )

            # Validate columns exist
            if col_map["question_text"] not in df.columns:
                raise ValueError(
                    f"Column '{col_map['question_text']}' not found in DataFrame. "
                    f"Available columns: {df.columns.tolist()}"
                )

            if rephrase_mode == "source" and col_map["source_text"] not in df.columns:
                raise ValueError(
                    f"Column '{col_map['source_text']}' not found in DataFrame for 'source' mode. "
                    f"Available columns: {df.columns.tolist()}"
                )

            if rephrase_mode == "domain" and not domain:
                raise ValueError(
                    "domain parameter is required when rephrase_mode is 'domain'"
                )

            # Initialize result column
            result_df = df.copy()
            if col_map["rephrased_question"] not in result_df.columns:
                result_df[col_map["rephrased_question"]] = None

            LOGGER.info(
                f"Starting rephrasing of {len(df)} questions in mode '{rephrase_mode}' "
                f"(batch: {should_batch})"
            )

            if should_batch:
                result_df = self._process_dataframe_rephrasing_batch(
                    result_df,
                    rephrase_mode,
                    domain,
                    allow_vague,
                    col_map,
                    batch_size,
                    checkpoint_batch_size,
                    save_path,
                )
            else:
                result_df = self._process_dataframe_rephrasing_sequential(
                    result_df,
                    rephrase_mode,
                    domain,
                    allow_vague,
                    col_map,
                    checkpoint_batch_size,
                    save_path,
                )

            if save_path:
                result_df.to_csv(save_path, index=False)
                LOGGER.info(f"Final results saved to {save_path}")

            return result_df

        def _process_dataframe_rephrasing_batch(
            self,
            df,
            rephrase_mode,
            domain,
            allow_vague,
            col_map,
            batch_size,
            checkpoint_size,
            save_path,
        ):
            """Process DataFrame rephrasing using batch API"""
            total_batches = (len(df) + batch_size - 1) // batch_size
            processed = 0
            errors = 0

            for batch_start in tqdm(
                range(0, len(df), batch_size),
                total=total_batches,
                desc="Processing batches",
            ):
                batch_end = min(batch_start + batch_size, len(df))
                batch_df = df.iloc[batch_start:batch_end]

                # Collect valid items
                valid_indices = []
                questions_batch = []
                sources_batch = []

                for idx, row in batch_df.iterrows():
                    question = row[col_map["question_text"]]

                    if pd.isna(question) or not str(question).strip():
                        continue

                    if rephrase_mode == "source":
                        source = row[col_map["source_text"]]
                        if pd.isna(source) or not str(source).strip():
                            continue
                        sources_batch.append(str(source))

                    valid_indices.append(idx)
                    questions_batch.append(str(question))

                if not questions_batch:
                    continue

                try:
                    # Batch process
                    if rephrase_mode == "source":
                        batch_results = self.rephrase_question_with_source_batch(
                            questions_batch, sources_batch, allow_vague
                        )
                    else:  # domain
                        batch_results = self.rephrase_question_with_domain_batch(
                            questions_batch, domain, allow_vague
                        )

                    # Update DataFrame
                    for idx, result in zip(valid_indices, batch_results):
                        if result is not None:
                            df.at[idx, col_map["rephrased_question"]] = result
                            processed += 1
                        else:
                            errors += 1

                    # Checkpoint
                    if save_path and checkpoint_size and processed % checkpoint_size == 0:
                        df.to_csv(save_path, index=False)
                        LOGGER.info(f"Checkpoint saved at {processed} rows")

                except Exception as e:
                    LOGGER.error(f"Batch error: {e}")
                    errors += len(valid_indices)

            LOGGER.info(
                f"Batch processing complete: {processed} successful, {errors} errors"
            )
            return df

        def _process_dataframe_rephrasing_sequential(
            self,
            df,
            rephrase_mode,
            domain,
            allow_vague,
            col_map,
            checkpoint_size,
            save_path,
        ):
            """Process DataFrame rephrasing sequentially"""
            processed = 0
            errors = 0

            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Rephrasing questions"):
                question_text = row[col_map["question_text"]]

                if pd.isna(question_text) or not str(question_text).strip():
                    LOGGER.warning(f"Empty question at index {idx}, skipping")
                    errors += 1
                    continue

                try:
                    if rephrase_mode == "source":
                        source_text = row[col_map["source_text"]]
                        if pd.isna(source_text) or not str(source_text).strip():
                            LOGGER.warning(f"Empty source at index {idx}, skipping")
                            errors += 1
                            continue

                        rephrased = self.rephrase_question_with_source(
                            str(question_text), str(source_text), allow_vague=allow_vague
                        )
                    else:  # domain mode
                        rephrased = self.rephrase_question_with_domain(
                            str(question_text), domain, allow_vague=allow_vague
                        )

                    if rephrased is not None:
                        df.at[idx, col_map["rephrased_question"]] = rephrased
                        processed += 1
                    else:
                        LOGGER.warning(f"Failed to rephrase question at index {idx}")
                        errors += 1

                    # Checkpoint
                    if save_path and checkpoint_size and processed % checkpoint_size == 0:
                        df.to_csv(save_path, index=False)
                        LOGGER.info(f"Checkpoint saved at {processed} rows")

                except KeyboardInterrupt:
                    LOGGER.warning("Processing interrupted by user")
                    if save_path:
                        df.to_csv(save_path, index=False)
                        LOGGER.info(f"Progress saved to {save_path}")
                    break
                except Exception as e:
                    LOGGER.error(f"Error processing row {idx}: {e}")
                    errors += 1

            LOGGER.info(
                f"Sequential processing complete: {processed} successful, {errors} errors"
            )
            return df

        def process_dataframe_generation(
            self,
            df: pd.DataFrame,
            n_questions: int = 5,
            allow_vague: bool = False,
            columns_mapping: Optional[dict] = None,
            use_batch_processing: bool = True,
            batch_size: Optional[int] = None,
        ) -> pd.DataFrame:
            """
            Process a DataFrame by generating alternative questions for each source.

            This method creates new rows in the DataFrame, with each row containing
            one of the generated alternative questions paired with its source.

            Parameters
            ----------
            df : pd.DataFrame
                DataFrame containing source text column.
            n_questions : int, optional
                Number of alternative questions to generate per source. Default is 5.
            allow_vague : bool, optional
                If True, allows generated questions to use less precise language.
                If False (default), ensures questions are specific and focused.
            columns_mapping : Optional[dict], optional
                Mapping of expected column names. Should contain:
                'source_text' (default: 'source_text'),
                'question_text' (default: 'question_text' - output column).
            use_batch_processing : bool, optional
                Use batch processing if available. Default is True.
            batch_size : int, optional
                Batch size for processing. If None, uses self.batch_size.

            Returns
            -------
            pd.DataFrame
                New DataFrame with generated question-source pairs.

            Raises
            ------
            ValueError
                If required columns are missing.
            """
            if df.empty:
                LOGGER.warning("Empty DataFrame provided")
                return df

            batch_size = batch_size or self.batch_size
            should_batch = use_batch_processing and self._is_openai_llm

            # Set up column mapping with validation
            default_mapping = {
                "source_text": "source_text",
                "question_text": "question_text",
            }

            # Validate columns_mapping keys
            if columns_mapping:
                invalid_keys = set(columns_mapping.keys()) - set(default_mapping.keys())
                if invalid_keys:
                    raise ValueError(
                        f"Invalid column mapping keys: {invalid_keys}. "
                        f"Valid keys are: {list(default_mapping.keys())}"
                    )

            col_map = {**default_mapping, **(columns_mapping or {})}

            # Validate columns exist
            if col_map["source_text"] not in df.columns:
                raise ValueError(
                    f"Column '{col_map['source_text']}' not found in DataFrame. "
                    f"Available columns: {df.columns.tolist()}"
                )

            LOGGER.info(
                f"Starting generation of {n_questions} questions per source for {len(df)} sources "
                f"(batch: {should_batch})"
            )

            if should_batch:
                return self._process_dataframe_generation_batch(
                    df, n_questions, allow_vague, col_map, batch_size
                )
            else:
                return self._process_dataframe_generation_sequential(
                    df, n_questions, allow_vague, col_map
                )

        def _process_dataframe_generation_batch(
            self, df, n_questions, allow_vague, col_map, batch_size
        ):
            """Process DataFrame generation using batch API"""
            generated_rows = []
            total_batches = (len(df) + batch_size - 1) // batch_size

            for batch_start in tqdm(
                range(0, len(df), batch_size),
                total=total_batches,
                desc="Generating questions",
            ):
                batch_end = min(batch_start + batch_size, len(df))
                batch_df = df.iloc[batch_start:batch_end]

                # Collect valid sources
                valid_rows = []
                sources_batch = []

                for idx, row in batch_df.iterrows():
                    source_text = row[col_map["source_text"]]
                    if pd.isna(source_text) or not str(source_text).strip():
                        continue

                    valid_rows.append(row)
                    sources_batch.append(str(source_text))

                if not sources_batch:
                    continue

                try:
                    # Batch generate
                    batch_results = self.generate_alternative_questions_batch(
                        sources_batch, n_questions, allow_vague
                    )

                    # Create new rows
                    for row, questions in zip(valid_rows, batch_results):
                        if questions is None:
                            continue

                        for question in questions:
                            new_row = row.copy()
                            new_row[col_map["question_text"]] = question
                            generated_rows.append(new_row)

                except Exception as e:
                    LOGGER.error(f"Batch generation error: {e}")

            if generated_rows:
                result_df = pd.DataFrame(generated_rows)
                LOGGER.info(
                    f"Generated {len(result_df)} question-source pairs from {len(df)} sources"
                )
                return result_df

            LOGGER.warning("No questions were successfully generated")
            return pd.DataFrame(columns=df.columns)

        def _process_dataframe_generation_sequential(
            self, df, n_questions, allow_vague, col_map
        ):
            """Process DataFrame generation sequentially"""
            generated_rows = []

            for idx, row in tqdm(
                df.iterrows(), total=len(df), desc="Generating questions"
            ):
                source_text = row[col_map["source_text"]]

                if pd.isna(source_text) or not str(source_text).strip():
                    LOGGER.warning(f"Empty source at index {idx}, skipping")
                    continue

                questions = self.generate_alternative_questions(
                    str(source_text), n=n_questions, allow_vague=allow_vague
                )

                if questions is None:
                    LOGGER.warning(
                        f"Failed to generate questions for source at index {idx}"
                    )
                    continue

                # Create a new row for each generated question
                for question in questions:
                    new_row = row.copy()
                    new_row[col_map["question_text"]] = question
                    generated_rows.append(new_row)

            if generated_rows:
                result_df = pd.DataFrame(generated_rows)
                LOGGER.info(
                    f"Generated {len(result_df)} question-source pairs from {len(df)} sources"
                )
                return result_df

            LOGGER.warning("No questions were successfully generated")
            return pd.DataFrame(columns=df.columns)

        def process_csv_rephrasing(
            self,
            input_csv_path: str,
            output_csv_path: str,
            rephrase_mode: str = "source",
            domain: Optional[str] = None,
            allow_vague: bool = False,
            columns_mapping: Optional[dict] = None,
            use_batch_processing: bool = True,
            batch_size: Optional[int] = None,
            checkpoint_batch_size: Optional[int] = None,
        ) -> pd.DataFrame:
            """
            Process a CSV file by rephrasing questions.

            Parameters
            ----------
            input_csv_path : str
                Path to the input CSV file.
            output_csv_path : str
                Path to save the output CSV with rephrased questions.
            rephrase_mode : str, optional
                Mode of rephrasing: 'source' or 'domain'. Default is 'source'.
            domain : Optional[str], optional
                Domain context for rephrasing when mode is 'domain'.
            allow_vague : bool, optional
                If True, allows rephrased questions to use less precise language.
                If False (default), ensures questions are clear and specific.
            columns_mapping : Optional[dict], optional
                Mapping of expected column names.
            use_batch_processing : bool, optional
                Use batch processing if available. Default is True.
            batch_size : int, optional
                Batch size for processing.
            checkpoint_batch_size : int, optional
                Save checkpoint every N rows.

            Returns
            -------
            pd.DataFrame
                The processed DataFrame with rephrased questions.
            """
            LOGGER.info(f"Loading CSV from {input_csv_path}")
            df = pd.read_csv(input_csv_path, encoding="utf-8")

            result_df = self.process_dataframe_rephrasing(
                df,
                rephrase_mode=rephrase_mode,
                domain=domain,
                allow_vague=allow_vague,
                columns_mapping=columns_mapping,
                use_batch_processing=use_batch_processing,
                batch_size=batch_size,
                checkpoint_batch_size=checkpoint_batch_size,
                save_path=output_csv_path,
            )

            LOGGER.info(f"Processed CSV saved to {output_csv_path}")
            return result_df

        def process_csv_generation(
            self,
            input_csv_path: str,
            output_csv_path: str,
            n_questions: int = 5,
            allow_vague: bool = False,
            columns_mapping: Optional[dict] = None,
            use_batch_processing: bool = True,
            batch_size: Optional[int] = None,
        ) -> pd.DataFrame:
            """
            Process a CSV file by generating alternative questions.

            Parameters
            ----------
            input_csv_path : str
                Path to the input CSV file.
            output_csv_path : str
                Path to save the output CSV with generated questions.
            n_questions : int, optional
                Number of questions to generate per source. Default is 5.
            allow_vague : bool, optional
                If True, allows generated questions to use less precise language.
                If False (default), ensures questions are specific and focused.
            columns_mapping : Optional[dict], optional
                Mapping of expected column names.
            use_batch_processing : bool, optional
                Use batch processing if available. Default is True.
            batch_size : int, optional
                Batch size for processing.

            Returns
            -------
            pd.DataFrame
                The generated DataFrame with question-source pairs.
            """
            LOGGER.info(f"Loading CSV from {input_csv_path}")
            df = pd.read_csv(input_csv_path, encoding="utf-8")

            result_df = self.process_dataframe_generation(
                df,
                n_questions=n_questions,
                allow_vague=allow_vague,
                columns_mapping=columns_mapping,
                use_batch_processing=use_batch_processing,
                batch_size=batch_size,
            )

            result_df.to_csv(output_csv_path, index=False, encoding="utf-8")
            LOGGER.info(f"Generated CSV saved to {output_csv_path}")

            return result_df

except ImportError as e:
    _IMPORT_ERROR = str(e)

    LOGGER.warning(
        f"QuestionAugmentationAgent dependencies not available: {e}. "
        "Install with: pip install langchain langchain_core tqdm pandas"
    )

    class QuestionAugmentationAgent:
        """
        Placeholder for QuestionAugmentationAgent when dependencies are missing.

        To use this agent, install required dependencies:
            pip install langchain langchain_core tqdm pandas
        """

        def __init__(self, *args, **kwargs):
            """Raise ImportError for missing dependencies."""
            raise ImportError(
                f"QuestionAugmentationAgent requires langchain_core and related dependencies to be installed.\n"
                f"Original import error: {_IMPORT_ERROR}\n"
                f"Install with: pip install langchain langchain_core tqdm pandas"
            )

        def __getattr__(self, name):
            """Raise ImportError for missing dependencies."""
            raise ImportError(
                f"QuestionAugmentationAgent not available due to missing dependencies: {_IMPORT_ERROR}"
            )