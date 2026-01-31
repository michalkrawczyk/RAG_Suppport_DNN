"""Source evaluation agent for assessing text source quality."""

import logging

LOGGER = logging.getLogger(__name__)
# TODO: Consider if errors should be reprocessed when skip_existing is True?


try:
    import json
    from enum import Enum
    from typing import Any, Dict, List, Optional, Union

    import pandas as pd
    from langchain_classic.output_parsers import OutputFixingParser
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_core.language_models import BaseChatModel
    from langchain_core.prompts import PromptTemplate
    from langgraph.graph import END, StateGraph
    from langgraph.prebuilt import ToolNode
    from pydantic import BaseModel, Field, field_validator, model_validator
    from tqdm import tqdm

    from RAG_supporters.prompts_templates.rag_verifiers import SINGLE_SRC_SCORE_PROMPT
    from RAG_supporters.utils.text_utils import is_empty_text

    # Pydantic Models for validation (v2.10.3 compatible)
    class ScoreRange(BaseModel):
        """Validates score is within 0-10 range."""

        score: int = Field(..., ge=0, le=10, description="Score between 0 and 10")
        reasoning: Optional[str] = Field(
            default=None, description="Optional reasoning for the score"
        )

    class SourceEvaluation(BaseModel):
        """Model for source evaluation scores and reasoning."""

        inferred_domain: str = Field(
            ..., description="The inferred domain from question and source"
        )
        relevance: ScoreRange = Field(..., description="Relevance score and reasoning")
        expertise_authority: ScoreRange = Field(
            ..., description="Expertise/Authority score and reasoning"
        )
        depth_specificity: ScoreRange = Field(
            ..., description="Depth and Specificity score and reasoning"
        )
        clarity_conciseness: ScoreRange = Field(
            ..., description="Clarity and Conciseness score and reasoning"
        )
        objectivity_bias: ScoreRange = Field(
            ..., description="Objectivity/Bias score and reasoning"
        )
        completeness: ScoreRange = Field(..., description="Completeness score and reasoning")

        @model_validator(mode="after")
        def validate_all_scores_present(self):
            """
            Ensure all required scores are present and valid.

            Validates that all six evaluation dimensions have scores assigned
            and that none of the score values are None.

            Raises
            ------
            ValueError
                If any required field is missing or if any score is None
            """
            required_fields = [
                "relevance",
                "expertise_authority",
                "depth_specificity",
                "clarity_conciseness",
                "objectivity_bias",
                "completeness",
            ]
            for field in required_fields:
                if not hasattr(self, field) or getattr(self, field) is None:
                    raise ValueError(f"Missing required field: {field}")
                if getattr(self, field).score is None:
                    raise ValueError(f"Missing score for field: {field}")
            return self

    class AgentState(BaseModel):
        """State for the LangGraph agent."""

        question: str
        source_content: str
        evaluation: Optional[SourceEvaluation] = None
        error: Optional[str] = None
        retry_count: int = 0
        max_retries: int = 3

    class SourceEvaluationAgent:
        """LangGraph agent for evaluating sources with retry logic."""

        def __init__(
            self,
            llm: BaseChatModel = None,
            max_retries: int = 3,
            evaluation_prompt: str = SINGLE_SRC_SCORE_PROMPT,
            batch_size: int = 10,  # Add default batch size
        ):
            """
            Initialize the agent with an LLM and retry configuration.

            Sets up the evaluation pipeline with output parsers, prompt templates,
            and the LangGraph workflow for reliable source evaluation.

            Parameters
            ----------
            llm : BaseChatModel, optional
                Language model to use for evaluation. Default is None.
            max_retries : int, optional
                Maximum number of retries for getting correct format. Default is 3.
            evaluation_prompt : str, optional
                Prompt template for evaluation. Default is SINGLE_SRC_SCORE_PROMPT.
            batch_size : int, optional
                Default batch size for batch processing. Default is 10.
            """
            self.llm = llm
            self._max_retries = max_retries
            self.eval_prompt = evaluation_prompt
            self.batch_size = batch_size

            # Set up the parser with Pydantic model
            self.parser = PydanticOutputParser(pydantic_object=SourceEvaluation)

            # Wrap with OutputFixingParser for automatic retry/fixing
            self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.llm)

            # Create the prompt template with format instructions
            self.prompt_template = self._create_prompt_template()

            self.graph = self._build_graph()

            # Check if LLM is OpenAI for batch processing
            self._is_openai_llm = self._check_openai_llm()

        def _check_openai_llm(self) -> bool:
            """
            Check if the LLM is from OpenAI (ChatOpenAI).

            Returns
            -------
            bool
                True if LLM is from OpenAI, False otherwise
            """
            try:
                from langchain_openai import AzureChatOpenAI, ChatOpenAI

                return isinstance(self.llm, (ChatOpenAI, AzureChatOpenAI))
            except ImportError:
                LOGGER.debug("langchain_openai not installed, batch processing unavailable")
                return False
            except Exception as e:
                LOGGER.debug(f"Could not determine if LLM is OpenAI: {e}")
                return False

        def _create_prompt_template(self) -> PromptTemplate:
            """Create the prompt template with format instructions from parser."""
            return PromptTemplate(
                template=self.eval_prompt,
                input_variables=["question", "source_content"],
                partial_variables={"format_instructions": self.parser.get_format_instructions()},
            )

        def _build_graph(self) -> StateGraph:
            """Build the LangGraph workflow."""
            workflow = StateGraph(AgentState)

            # Add nodes
            workflow.add_node("evaluate", self._evaluate_source)
            workflow.add_node("validate", self._validate_response)
            workflow.add_node("retry", self._handle_retry)

            # Add edges
            workflow.set_entry_point("evaluate")
            workflow.add_edge("evaluate", "validate")

            # Conditional edges from validate
            workflow.add_conditional_edges(
                "validate", self._should_retry, {"retry": "retry", "end": END}
            )

            # From retry back to evaluate
            workflow.add_edge("retry", "evaluate")

            return workflow.compile()

        def _evaluate_source(self, state: AgentState) -> Dict[str, Any]:
            """Evaluate the source using the LLM with output parser."""
            try:
                # Format the prompt
                prompt = self.prompt_template.format(
                    question=state.question, source_content=state.source_content
                )

                LOGGER.debug(f"Sending prompt to LLM...")

                # Get response from LLM
                response = self.llm.invoke(prompt)

                # Extract content based on response type
                if hasattr(response, "content"):
                    content = response.content
                else:
                    content = str(response)

                LOGGER.debug(f"Raw LLM Response (first 500 chars): {content[:500]}...")

                try:
                    # Use the fixing parser to parse and potentially fix the output
                    evaluation = self.fixing_parser.parse(content)
                    state.evaluation = evaluation
                    state.error = None
                    LOGGER.info("Successfully parsed evaluation with parser")

                except Exception as parse_error:
                    LOGGER.warning(f"Fixing parser failed, trying regular parser: {parse_error}")

                    # Try regular parser as fallback
                    try:
                        evaluation = self.parser.parse(content)
                        state.evaluation = evaluation
                        state.error = None
                        LOGGER.info("Successfully parsed with regular parser")
                    except Exception as e:
                        LOGGER.error(f"All parsing attempts failed: {e}")
                        state.error = f"Parsing error: {str(e)}"
                        state.evaluation = None

            except Exception as e:
                LOGGER.error(f"Evaluation error: {e}")
                state.error = str(e)
                state.evaluation = None

            return {"evaluation": state.evaluation, "error": state.error}

        def _validate_response(self, state: AgentState) -> Dict[str, Any]:
            """Validate the response using Pydantic."""
            if state.evaluation is None:
                state.error = state.error or "No evaluation generated"
                return {"error": state.error}

            try:
                # The parser already validates, but we can double-check
                if isinstance(state.evaluation, SourceEvaluation):
                    # It's already a valid Pydantic model
                    state.error = None
                    LOGGER.info(
                        "Validation successful - evaluation is valid SourceEvaluation model"
                    )
                else:
                    # Try to convert if it's a dict
                    if isinstance(state.evaluation, dict):
                        state.evaluation = SourceEvaluation(**state.evaluation)
                        state.error = None
                        LOGGER.info("Validation successful - converted dict to SourceEvaluation")
                    else:
                        raise ValueError(f"Unexpected evaluation type: {type(state.evaluation)}")

            except Exception as e:
                LOGGER.error(f"Validation error: {e}")
                state.error = f"Validation error: {str(e)}"
                state.evaluation = None

            return {"evaluation": state.evaluation, "error": state.error}

        def _should_retry(self, state: AgentState) -> str:
            """Determine if we should retry or end."""
            if state.error is None and state.evaluation is not None:
                return "end"
            elif state.retry_count < state.max_retries:
                return "retry"
            else:
                LOGGER.error(f"Max retries ({state.max_retries}) reached")
                return "end"

        def _handle_retry(self, state: AgentState) -> Dict[str, Any]:
            """Handle retry logic."""
            state.retry_count += 1
            LOGGER.info(f"Retrying... Attempt {state.retry_count}/{state.max_retries}")
            LOGGER.info(f"Previous error: {state.error}")

            # Clear previous evaluation
            state.evaluation = None
            state.error = None

            return {"retry_count": state.retry_count}

        def _format_output(self, evaluation: Dict[str, Any]) -> Dict[str, Any]:
            """Format the output to match the expected structure."""
            return {
                "inferred_domain": evaluation.get("inferred_domain"),
                "scores": {
                    "relevance": evaluation.get("relevance", {}).get("score"),
                    "expertise_authority": evaluation.get("expertise_authority", {}).get("score"),
                    "depth_specificity": evaluation.get("depth_specificity", {}).get("score"),
                    "clarity_conciseness": evaluation.get("clarity_conciseness", {}).get("score"),
                    "objectivity_bias": evaluation.get("objectivity_bias", {}).get("score"),
                    "completeness": evaluation.get("completeness", {}).get("score"),
                },
                "reasoning": {
                    "relevance": evaluation.get("relevance", {}).get("reasoning"),
                    "expertise_authority": evaluation.get("expertise_authority", {}).get(
                        "reasoning"
                    ),
                    "depth_specificity": evaluation.get("depth_specificity", {}).get("reasoning"),
                    "clarity_conciseness": evaluation.get("clarity_conciseness", {}).get(
                        "reasoning"
                    ),
                    "objectivity_bias": evaluation.get("objectivity_bias", {}).get("reasoning"),
                    "completeness": evaluation.get("completeness", {}).get("reasoning"),
                },
                "score_summary": self._generate_score_summary(evaluation),
            }

        def _generate_score_summary(self, evaluation: Dict[str, Any]) -> str:
            """Generate a formatted score summary."""
            scores = [
                f"- Relevance: {evaluation.get('relevance', {}).get('score')}/10",
                f"- Expertise/Authority: {evaluation.get('expertise_authority', {}).get('score')}/10",
                f"- Depth and Specificity: {evaluation.get('depth_specificity', {}).get('score')}/10",
                f"- Clarity and Conciseness: {evaluation.get('clarity_conciseness', {}).get('score')}/10",
                f"- Objectivity/Bias: {evaluation.get('objectivity_bias', {}).get('score')}/10",
                f"- Completeness: {evaluation.get('completeness', {}).get('score')}/10",
            ]
            return "\n".join(scores)

        def evaluate(self, question: str, source_content: str) -> Optional[Dict[str, Any]]:
            """
            Evaluate a source for a given question using the LangGraph workflow.

            Main interface method that orchestrates the complete evaluation process
            including retry logic and output formatting.

            Parameters
            ----------
            question : str
                The question to evaluate the source against
            source_content : str
                The source content to evaluate for relevance and quality

            Returns
            -------
            Optional[Dict[str, Any]]
                Dictionary with evaluation scores and reasoning, or None if evaluation failed
            """
            initial_state = AgentState(
                question=question,
                source_content=source_content,
                max_retries=self._max_retries,
            )

            # Run the graph
            result = self.graph.invoke(initial_state.model_dump())

            if result.get("evaluation"):
                # Convert to dictionary format for easier use
                evaluation = result["evaluation"]

                # Handle both dict and Pydantic model
                if isinstance(evaluation, SourceEvaluation):
                    return self._format_output(evaluation.model_dump())
                elif isinstance(evaluation, dict):
                    return self._format_output(evaluation)
                else:
                    LOGGER.error(f"Unexpected evaluation type: {type(evaluation)}")
                    return None
            else:
                LOGGER.error(f"Failed to get valid evaluation after {self._max_retries} retries")
                LOGGER.error(f"Final error: {result.get('error')}")
                return None

        def evaluate_batch(
            self, questions: List[str], source_contents: List[str]
        ) -> List[Optional[Dict[str, Any]]]:
            """
            Evaluate multiple question-source pairs in a batch using LangChain's batch processing.

            Only available for OpenAI LLMs. Falls back to sequential processing for other LLMs.

            Parameters
            ----------
            questions : List[str]
                List of questions to evaluate
            source_contents : List[str]
                List of source contents corresponding to each question

            Returns
            -------
            List[Optional[Dict[str, Any]]]
                List of evaluation results, None for failed evaluations
            """
            if not self._is_openai_llm:
                LOGGER.info(
                    "Batch processing not available for non-OpenAI LLMs, using sequential processing"
                )
                return [self.evaluate(q, s) for q, s in zip(questions, source_contents)]

            if len(questions) != len(source_contents):
                raise ValueError("questions and source_contents must have the same length")

            LOGGER.info(f"Processing batch of {len(questions)} evaluations using OpenAI batch API")

            # Prepare batch prompts
            prompts = []
            for question, source in zip(questions, source_contents):
                prompt = self.prompt_template.format(question=question, source_content=source)
                prompts.append(prompt)

            try:
                # Use LangChain's batch method for OpenAI
                # This handles rate limiting and retries internally
                responses = self.llm.batch(prompts)

                # Parse each response
                results = []
                for i, response in enumerate(responses):
                    try:
                        # Extract content based on response type
                        if hasattr(response, "content"):
                            content = response.content
                        else:
                            content = str(response)

                        # Parse the response
                        try:
                            evaluation = self.fixing_parser.parse(content)
                            if isinstance(evaluation, SourceEvaluation):
                                results.append(self._format_output(evaluation.model_dump()))
                            else:
                                results.append(self._format_output(evaluation))
                        except Exception as parse_error:
                            LOGGER.warning(f"Failed to parse batch item {i}: {parse_error}")
                            # Try regular parser as fallback
                            try:
                                evaluation = self.parser.parse(content)
                                if isinstance(evaluation, SourceEvaluation):
                                    results.append(self._format_output(evaluation.model_dump()))
                                else:
                                    results.append(self._format_output(evaluation))
                            except Exception as e:
                                LOGGER.error(f"All parsing attempts failed for batch item {i}: {e}")
                                results.append(None)
                    except Exception as e:
                        LOGGER.error(f"Error processing batch response {i}: {e}")
                        results.append(None)

                return results

            except Exception as e:
                LOGGER.error(f"Batch processing failed: {e}")
                LOGGER.info("Falling back to sequential processing")
                # Fallback to sequential processing
                return [self.evaluate(q, s) for q, s in zip(questions, source_contents)]

        def process_dataframe(
            self,
            df: pd.DataFrame,
            question_col: str = "question_text",
            source_col: str = "source_text",
            include_reasoning: bool = False,
            progress_bar: bool = True,
            save_path: Optional[str] = None,
            skip_existing: bool = True,
            checkpoint_batch_size: Optional[int] = None,
            use_batch_processing: bool = True,
            batch_size: Optional[int] = None,
        ) -> pd.DataFrame:
            """Process a pandas DataFrame with question-source pairs and add evaluation scores.

            Batch processes multiple question-source pairs, adding comprehensive
            evaluation scores across all dimensions with optional reasoning text.

            Parameters
            ----------
            df : pd.DataFrame
                DataFrame containing question and source columns to evaluate
            question_col : str, optional
                Name of the column containing questions. Default is "question_text".
            source_col : str, optional
                Name of the column containing source content. Default is "source_text".
            include_reasoning : bool, optional
                Whether to include reasoning text for each score. Default is False.
            progress_bar : bool, optional
                Whether to display a progress bar during processing. Default is True.
            save_path : str, optional
                Path to save the results as CSV. Default is None.
            skip_existing : bool, optional
                Whether to skip rows that already have evaluation scores. Default is True.
            checkpoint_batch_size : int, optional
                Batch size for saving intermediate checkpoints. Default is None.
            use_batch_processing : bool, optional
                Whether to use batch processing for OpenAI LLMs. Default is True.
            batch_size : int, optional
                Size of batches for batch processing. If None, uses default from init. Default is None.

            Returns
            -------
            pd.DataFrame
                DataFrame with added evaluation score columns and optional reasoning
            """
            # Determine if we should use batch processing
            should_batch = use_batch_processing and self._is_openai_llm
            batch_size = batch_size or self.batch_size

            if should_batch:
                LOGGER.info(f"Using batch processing with batch size {batch_size}")
                return self._process_dataframe_batch(
                    df=df,
                    question_col=question_col,
                    source_col=source_col,
                    include_reasoning=include_reasoning,
                    progress_bar=progress_bar,
                    save_path=save_path,
                    skip_existing=skip_existing,
                    checkpoint_batch_size=checkpoint_batch_size,
                    batch_size=batch_size,
                )
            else:
                if use_batch_processing and not self._is_openai_llm:
                    LOGGER.info(
                        "Batch processing requested but LLM is not OpenAI, using sequential processing"
                    )
                return self._process_dataframe_sequential(
                    df=df,
                    question_col=question_col,
                    source_col=source_col,
                    include_reasoning=include_reasoning,
                    progress_bar=progress_bar,
                    save_path=save_path,
                    skip_existing=skip_existing,
                    checkpoint_batch_size=checkpoint_batch_size,
                )

        def _process_dataframe_batch(
            self,
            df: pd.DataFrame,
            question_col: str,
            source_col: str,
            include_reasoning: bool,
            progress_bar: bool,
            save_path: Optional[str],
            skip_existing: bool,
            checkpoint_batch_size: Optional[int],
            batch_size: int,
        ) -> pd.DataFrame:
            """
            Process DataFrame using batch processing for OpenAI LLMs.

            Internal method that handles the batch processing logic.
            """
            # Create a copy to avoid modifying original
            result_df = df.copy()

            if not question_col in result_df.columns or not source_col in result_df.columns:
                raise ValueError(
                    f"DataFrame must contain columns '{question_col}' and '{source_col}'"
                )

            # Initialize result columns
            result_columns = [
                "inferred_domain",
                "relevance_score",
                "expertise_authority_score",
                "depth_specificity_score",
                "clarity_conciseness_score",
                "objectivity_bias_score",
                "completeness_score",
                "evaluation_error",
            ]
            if include_reasoning:
                result_columns.extend(
                    [
                        "relevance_reasoning",
                        "expertise_authority_reasoning",
                        "depth_specificity_reasoning",
                        "clarity_conciseness_reasoning",
                        "objectivity_bias_reasoning",
                        "completeness_reasoning",
                    ]
                )

            for column in result_columns:
                if column not in result_df.columns:
                    result_df[column] = None

            # Prepare rows for batch processing
            rows_to_process = []
            indices_to_process = []

            score_columns = [c for c in result_columns if "score" in c]

            for idx, row in result_df.iterrows():
                # Skip if requested and has existing scores
                if skip_existing:
                    has_existing_scores = any(
                        col in result_df.columns and pd.notna(row[col]) for col in score_columns
                    )
                    if has_existing_scores:
                        continue

                # If empty question or source, set error
                if (
                    pd.isna(row[question_col])
                    or pd.isna(row[source_col])
                    or is_empty_text(row[question_col])
                    or is_empty_text(row[source_col])
                ):
                    result_df.at[idx, "evaluation_error"] = (
                        "Missing or empty question or source content"
                    )
                    continue

                # Skip rows with missing data
                if pd.isna(row[question_col]) or pd.isna(row[source_col]):
                    LOGGER.warning(f"Skipping row {idx} due to missing question or source content")
                    continue

                rows_to_process.append(row)
                indices_to_process.append(idx)

            if not rows_to_process:
                LOGGER.info("No rows to process")
                return result_df

            # Process in batches
            total_batches = (len(rows_to_process) + batch_size - 1) // batch_size

            iterator = (
                tqdm(
                    range(0, len(rows_to_process), batch_size),
                    total=total_batches,
                    desc="Processing batches",
                )
                if progress_bar
                else range(0, len(rows_to_process), batch_size)
            )

            processed_count = 0
            error_count = 0

            for batch_start in iterator:
                batch_end = min(batch_start + batch_size, len(rows_to_process))
                batch_rows = rows_to_process[batch_start:batch_end]
                batch_indices = indices_to_process[batch_start:batch_end]

                # Extract questions and sources for this batch
                batch_questions = [row[question_col] for row in batch_rows]
                batch_sources = [row[source_col] for row in batch_rows]

                try:
                    # Process batch
                    batch_results = self.evaluate_batch(batch_questions, batch_sources)

                    # Update DataFrame with results
                    for idx, evaluation in zip(batch_indices, batch_results):
                        if evaluation:
                            # Prepare scores
                            update_dict = {
                                "inferred_domain": evaluation["inferred_domain"],
                                "relevance_score": evaluation["scores"]["relevance"],
                                "expertise_authority_score": evaluation["scores"][
                                    "expertise_authority"
                                ],
                                "depth_specificity_score": evaluation["scores"][
                                    "depth_specificity"
                                ],
                                "clarity_conciseness_score": evaluation["scores"][
                                    "clarity_conciseness"
                                ],
                                "objectivity_bias_score": evaluation["scores"]["objectivity_bias"],
                                "completeness_score": evaluation["scores"]["completeness"],
                            }

                            # Add reasoning if requested
                            if include_reasoning:
                                update_dict.update(
                                    {
                                        "relevance_reasoning": evaluation["reasoning"]["relevance"],
                                        "expertise_authority_reasoning": evaluation["reasoning"][
                                            "expertise_authority"
                                        ],
                                        "depth_specificity_reasoning": evaluation["reasoning"][
                                            "depth_specificity"
                                        ],
                                        "clarity_conciseness_reasoning": evaluation["reasoning"][
                                            "clarity_conciseness"
                                        ],
                                        "objectivity_bias_reasoning": evaluation["reasoning"][
                                            "objectivity_bias"
                                        ],
                                        "completeness_reasoning": evaluation["reasoning"][
                                            "completeness"
                                        ],
                                    }
                                )

                            result_df.loc[idx, list(update_dict.keys())] = list(
                                update_dict.values()
                            )

                            processed_count += 1

                            # Save checkpoint if specified
                            if (
                                save_path
                                and checkpoint_batch_size
                                and processed_count > 0
                                and processed_count % checkpoint_batch_size == 0
                            ):
                                result_df.to_csv(save_path, index=False)
                                LOGGER.info(
                                    f"Checkpoint saved at {save_path} after {processed_count} rows"
                                )
                        else:
                            result_df.at[idx, "evaluation_error"] = "Failed to evaluate in batch"
                            error_count += 1
                    if progress_bar:
                        iterator.set_postfix({"Processed": processed_count, "Errors": error_count})

                except KeyboardInterrupt:
                    LOGGER.warning("Batch processing interrupted by user")
                    break

                except Exception as e:
                    LOGGER.error(f"Error processing batch {batch_start}-{batch_end}: {e}")
                    for idx in batch_indices:
                        result_df.at[idx, "evaluation_error"] = f"Batch error: {str(e)}"
                    error_count += len(batch_indices)

            LOGGER.info(
                f"Batch processing complete: {processed_count} successful, {error_count} errors"
            )

            if save_path:
                result_df.to_csv(save_path, index=False)
                LOGGER.info(f"Final results saved to {save_path}")

            return result_df

        def _process_dataframe_sequential(
            self,
            df: pd.DataFrame,
            question_col: str,
            source_col: str,
            include_reasoning: bool,
            progress_bar: bool,
            save_path: Optional[str],
            skip_existing: bool,
            checkpoint_batch_size: Optional[int],
        ) -> pd.DataFrame:
            """
            Original sequential processing method.

            This is the existing process_dataframe logic moved to a separate method.
            """
            # Create a copy to avoid modifying original
            result_df = df.copy()
            checkpoint_batch_size = max(
                0, checkpoint_batch_size or 0
            )  # Ensure it's a positive integer

            if not question_col in result_df.columns or not source_col in result_df.columns:
                raise ValueError(
                    f"DataFrame must contain columns '{question_col}' and '{source_col}'"
                )

            result_columns = [
                "inferred_domain",
                "relevance_score",
                "expertise_authority_score",
                "depth_specificity_score",
                "clarity_conciseness_score",
                "objectivity_bias_score",
                "completeness_score",
                "evaluation_error",
            ]
            if include_reasoning:
                result_columns.extend(
                    [
                        "relevance_reasoning",
                        "expertise_authority_reasoning",
                        "depth_specificity_reasoning",
                        "clarity_conciseness_reasoning",
                        "objectivity_bias_reasoning",
                        "completeness_reasoning",
                    ]
                )

            if skip_existing:
                score_columns = [c for c in result_columns if "score" in c]

            # Initialize new columns
            for column in result_columns:
                if column not in result_df.columns:
                    result_df[column] = None

            # Process each row
            iterator = (
                tqdm(result_df.iterrows(), total=len(result_df))
                if progress_bar
                else result_df.iterrows()
            )
            total_rows = len(result_df)
            processed_rows = 0
            skipped_rows = 0
            error_rows = 0

            for idx, row in iterator:
                try:
                    if (
                        save_path
                        and checkpoint_batch_size
                        and processed_rows > 0
                        and processed_rows % checkpoint_batch_size == 0
                    ):
                        result_df.to_csv(save_path, index=False)
                        LOGGER.info(f"Checkpoint saved at {save_path}")

                    if skip_existing:
                        has_existing_scores = any(
                            col in result_df.columns and pd.notna(row[col]) for col in score_columns
                        )
                        if has_existing_scores:
                            skipped_rows += 1
                            continue

                    if (
                        pd.isna(row[question_col])
                        or pd.isna(row[source_col])
                        or is_empty_text(row[question_col])
                        or is_empty_text(row[source_col])
                    ):
                        result_df.at[idx, "evaluation_error"] = (
                            "Missing or empty question or source content"
                        )

                        LOGGER.warning(
                            f"Skipping row {idx} due to missing question or source content"
                        )
                        skipped_rows += 1
                        continue

                    # Evaluate the source
                    evaluation = self.evaluate(
                        question=row[question_col], source_content=row[source_col]
                    )

                    if evaluation:
                        # Add scores
                        update_dict = {
                            "inferred_domain": evaluation["inferred_domain"],
                            "relevance_score": evaluation["scores"]["relevance"],
                            "expertise_authority_score": evaluation["scores"][
                                "expertise_authority"
                            ],
                            "depth_specificity_score": evaluation["scores"]["depth_specificity"],
                            "clarity_conciseness_score": evaluation["scores"][
                                "clarity_conciseness"
                            ],
                            "objectivity_bias_score": evaluation["scores"]["objectivity_bias"],
                            "completeness_score": evaluation["scores"]["completeness"],
                        }

                        if include_reasoning:
                            update_dict.update(
                                {
                                    "relevance_reasoning": evaluation["reasoning"]["relevance"],
                                    "expertise_authority_reasoning": evaluation["reasoning"][
                                        "expertise_authority"
                                    ],
                                    "depth_specificity_reasoning": evaluation["reasoning"][
                                        "depth_specificity"
                                    ],
                                    "clarity_conciseness_reasoning": evaluation["reasoning"][
                                        "clarity_conciseness"
                                    ],
                                    "objectivity_bias_reasoning": evaluation["reasoning"][
                                        "objectivity_bias"
                                    ],
                                    "completeness_reasoning": evaluation["reasoning"][
                                        "completeness"
                                    ],
                                }
                            )

                        result_df.loc[idx, list(update_dict.keys())] = list(update_dict.values())
                    else:
                        result_df.at[idx, "evaluation_error"] = "Failed to evaluate after retries"

                    processed_rows += 1

                except KeyboardInterrupt:
                    LOGGER.warning("Processing interrupted by user")
                    break

                except Exception as e:
                    LOGGER.error(f"Error processing row {idx}: {e}")
                    result_df.at[idx, "evaluation_error"] = str(e)
                    error_rows += 1

                if progress_bar:
                    iterator.set_postfix(
                        {
                            "Processed": processed_rows,
                            "Errors": error_rows,
                            "Skipped": skipped_rows,
                        }
                    )

            LOGGER.info(
                f"Processing complete: {processed_rows} processed, {skipped_rows} skipped, {error_rows} errors out of {total_rows} total rows"
            )

            if save_path:
                result_df.to_csv(save_path, index=False)
                LOGGER.info(f"Results saved to {save_path}")

            return result_df

except ImportError as e:
    _DEPENDENCIES_AVAILABLE = False
    _IMPORT_ERROR = str(e)

    LOGGER.warning(
        f"SourceEvaluationAgent dependencies not available: {e}. "
        "Install with: pip install langchain langgraph pydantic pandas tqdm"
    )

    class SourceEvaluationAgent:
        """
        Placeholder for SourceEvaluationAgent when dependencies are missing.

        To use this agent, install required dependencies:
            pip install -r requirements_agents.txt
        """

        def __init__(self, *args, **kwargs):
            """Initialize placeholder and raise import error."""
            raise ImportError(
                f"SourceEvaluationAgent requires langgraph, langchain, and pydantic to be installed.\n"
                f"Original import error: {_IMPORT_ERROR}\n"
                f"Install with: pip install -r requirements_agents.txt"
            )

        def __getattr__(self, name):
            """Raise import error for missing dependencies."""
            raise ImportError(
                f"SourceEvaluationAgent not available due to missing dependencies: {_IMPORT_ERROR}"
            )
