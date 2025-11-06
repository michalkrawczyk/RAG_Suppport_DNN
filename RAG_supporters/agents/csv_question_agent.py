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

        Attributes
        ----------
        _llm : BaseChatModel
            The language model used for question operations.
        _max_retries : int
            Maximum retries for LLM operations.
        """

        def __init__(
            self,
            llm: BaseChatModel,
            max_retries: int = 3,
        ):
            """
            Initialize the QuestionAugmentationAgent.

            Parameters
            ----------
            llm : BaseChatModel
                Language model instance for performing question operations.
            max_retries : int, optional
                Maximum number of retries for LLM calls. Default is 3.
            """
            self._llm = llm
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
                    "Invalid n value (%d) for question generation, must be >= 1", n
                )
                return None

            if n > 20:
                LOGGER.warning(
                    "Large n value (%d) may result in lower quality questions", n
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
                # Try to extract JSON from the response
                response_cleaned = response.strip()
                # Handle potential markdown code blocks
                if response_cleaned.startswith("```"):
                    lines = response_cleaned.split("\n")
                    response_cleaned = "\n".join(
                        line for line in lines if not line.strip().startswith("```")
                    )

                data = json.loads(response_cleaned)
                questions = data.get("questions", [])

                if not isinstance(questions, list):
                    LOGGER.error("Response JSON 'questions' field is not a list")
                    return None

                if len(questions) != n:
                    LOGGER.warning(
                        "Expected %d questions but received %d from JSON response",
                        n,
                        len(questions),
                    )

                if not questions:
                    LOGGER.error("No questions found in JSON response")
                    return None

                return questions

            except json.JSONDecodeError as e:
                LOGGER.error("Failed to parse JSON response: %s", str(e))
                LOGGER.debug("Response was: %s", response)
                return None

        def process_dataframe_rephrasing(
            self,
            df: pd.DataFrame,
            rephrase_mode: str = "source",
            domain: Optional[str] = None,
            allow_vague: bool = False,
            columns_mapping: Optional[dict] = None,
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

            # Set up column mapping
            default_mapping = {
                "question_text": "question_text",
                "source_text": "source_text",
                "rephrased_question": "rephrased_question",
            }
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

            LOGGER.info(
                "Starting rephrasing of %d questions in mode '%s'",
                len(df),
                rephrase_mode,
            )

            rephrased_questions = []

            for idx, row in tqdm(
                df.iterrows(), total=len(df), desc="Rephrasing questions"
            ):
                question_text = row[col_map["question_text"]]

                if pd.isna(question_text) or not str(question_text).strip():
                    LOGGER.warning("Empty question at index %s, skipping", idx)
                    rephrased_questions.append(None)
                    continue

                if rephrase_mode == "source":
                    source_text = row[col_map["source_text"]]
                    if pd.isna(source_text) or not str(source_text).strip():
                        LOGGER.warning("Empty source at index %s, skipping", idx)
                        rephrased_questions.append(None)
                        continue

                    rephrased = self.rephrase_question_with_source(
                        str(question_text), str(source_text), allow_vague=allow_vague
                    )
                else:  # domain mode
                    rephrased = self.rephrase_question_with_domain(
                        str(question_text), domain, allow_vague=allow_vague
                    )

                if rephrased is None:
                    LOGGER.warning("Failed to rephrase question at index %s", idx)

                rephrased_questions.append(rephrased)

            # Add the rephrased questions as a new column
            result_df = df.copy()
            result_df[col_map["rephrased_question"]] = rephrased_questions

            successful_count = sum(1 for q in rephrased_questions if q is not None)
            LOGGER.info(
                "Successfully rephrased %d out of %d questions",
                successful_count,
                len(df),
            )

            return result_df

        def process_dataframe_generation(
            self,
            df: pd.DataFrame,
            n_questions: int = 5,
            allow_vague: bool = False,
            columns_mapping: Optional[dict] = None,
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

            # Set up column mapping
            default_mapping = {
                "source_text": "source_text",
                "question_text": "question_text",
            }
            col_map = {**default_mapping, **(columns_mapping or {})}

            # Validate columns exist
            if col_map["source_text"] not in df.columns:
                raise ValueError(
                    f"Column '{col_map['source_text']}' not found in DataFrame. "
                    f"Available columns: {df.columns.tolist()}"
                )

            LOGGER.info(
                "Starting generation of %d questions per source for %d sources",
                n_questions,
                len(df),
            )

            generated_rows = []

            for idx, row in tqdm(
                df.iterrows(), total=len(df), desc="Generating questions"
            ):
                source_text = row[col_map["source_text"]]

                if pd.isna(source_text) or not str(source_text).strip():
                    LOGGER.warning("Empty source at index %s, skipping", idx)
                    continue

                questions = self.generate_alternative_questions(
                    str(source_text), n=n_questions, allow_vague=allow_vague
                )

                if questions is None:
                    LOGGER.warning(
                        "Failed to generate questions for source at index %s", idx
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
                    "Generated %d question-source pairs from %d sources",
                    len(result_df),
                    len(df),
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

            Returns
            -------
            pd.DataFrame
                The processed DataFrame with rephrased questions.
            """
            LOGGER.info("Loading CSV from %s", input_csv_path)
            df = pd.read_csv(input_csv_path)

            result_df = self.process_dataframe_rephrasing(
                df,
                rephrase_mode=rephrase_mode,
                domain=domain,
                allow_vague=allow_vague,
                columns_mapping=columns_mapping,
            )

            result_df.to_csv(output_csv_path, index=False, encoding="utf-8")
            LOGGER.info("Processed CSV saved to %s", output_csv_path)

            return result_df

        def process_csv_generation(
            self,
            input_csv_path: str,
            output_csv_path: str,
            n_questions: int = 5,
            allow_vague: bool = False,
            columns_mapping: Optional[dict] = None,
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

            Returns
            -------
            pd.DataFrame
                The generated DataFrame with question-source pairs.
            """
            LOGGER.info("Loading CSV from %s", input_csv_path)
            df = pd.read_csv(input_csv_path)

            result_df = self.process_dataframe_generation(
                df,
                n_questions=n_questions,
                allow_vague=allow_vague,
                columns_mapping=columns_mapping,
            )

            result_df.to_csv(output_csv_path, index=False, encoding="utf-8")
            LOGGER.info("Generated CSV saved to %s", output_csv_path)

            return result_df

except ImportError as e:
    _DEPENDENCIES_AVAILABLE = False
    _IMPORT_ERROR = str(e)

    LOGGER.warning(
        "QuestionAugmentationAgent dependencies not available: %s. "
        "Install with: pip install langchain langchain_core tqdm pandas",
        e,
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
