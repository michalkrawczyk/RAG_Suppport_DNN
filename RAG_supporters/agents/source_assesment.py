import logging

LOGGER = logging.getLogger(__name__)

# TODO: Consider batch processing for efficiency in dataframe processing (later)
# TODO: Consider partial saves (checkpoint)

try:
    import json
    from enum import Enum
    from typing import Any, Dict, List, Optional

    from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
    from langchain_core.language_models import BaseChatModel
    from langchain_core.prompts import PromptTemplate
    from langgraph.graph import END, StateGraph
    from langgraph.prebuilt import ToolNode
    import pandas as pd
    from pydantic import BaseModel, Field, field_validator, model_validator
    from tqdm import tqdm

    from prompts_templates.rag_verifiers import SINGLE_SRC_SCORE_PROMPT

    # Pydantic Models for validation (v2.10.3 compatible)
    class ScoreRange(BaseModel):
        """Validates score is within 0-10 range"""

        score: int = Field(..., ge=0, le=10, description="Score between 0 and 10")
        reasoning: Optional[str] = Field(
            default=None, description="Optional reasoning for the score"
        )

    class SourceEvaluation(BaseModel):
        """Model for source evaluation scores and reasoning"""

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
        completeness: ScoreRange = Field(
            ..., description="Completeness score and reasoning"
        )

        @model_validator(mode="after")
        def validate_all_scores_present(self):
            """Ensure all required scores are present"""
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
        """State for the LangGraph agent"""

        question: str
        source_content: str
        evaluation: Optional[SourceEvaluation] = None
        error: Optional[str] = None
        retry_count: int = 0
        max_retries: int = 3

    class SourceEvaluationAgent:
        """LangGraph agent for evaluating sources with retry logic"""

        def __init__(
            self,
            llm: BaseChatModel = None,
            max_retries: int = 3,
            evaluation_prompt: str = SINGLE_SRC_SCORE_PROMPT,
        ):
            """
            Initialize the agent with an LLM and retry configuration

            Args:
                llm: Language model to use (defaults to ChatOpenAI)
                max_retries: Maximum number of retries for getting correct format
            """
            self.llm = llm
            self.max_retries = max_retries
            self.eval_prompt = evaluation_prompt

            # Set up the parser with Pydantic model
            self.parser = PydanticOutputParser(pydantic_object=SourceEvaluation)

            # Wrap with OutputFixingParser for automatic retry/fixing
            self.fixing_parser = OutputFixingParser.from_llm(
                parser=self.parser, llm=self.llm
            )

            # Create the prompt template with format instructions
            self.prompt_template = self._create_prompt_template()

            self.graph = self._build_graph()

        def _create_prompt_template(self) -> PromptTemplate:
            """Create the prompt template with format instructions from parser"""

            return PromptTemplate(
                template=self.eval_prompt,
                input_variables=["question", "source_content"],
                partial_variables={
                    "format_instructions": self.parser.get_format_instructions()
                },
            )

        def _build_graph(self) -> StateGraph:
            """Build the LangGraph workflow"""
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
            """Evaluate the source using the LLM with output parser"""
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
                    LOGGER.warning(
                        f"Fixing parser failed, trying regular parser: {parse_error}"
                    )

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
            """Validate the response using Pydantic"""
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
                        LOGGER.info(
                            "Validation successful - converted dict to SourceEvaluation"
                        )
                    else:
                        raise ValueError(
                            f"Unexpected evaluation type: {type(state.evaluation)}"
                        )

            except Exception as e:
                LOGGER.error(f"Validation error: {e}")
                state.error = f"Validation error: {str(e)}"
                state.evaluation = None

            return {"evaluation": state.evaluation, "error": state.error}

        def _should_retry(self, state: AgentState) -> str:
            """Determine if we should retry or end"""
            if state.error is None and state.evaluation is not None:
                return "end"
            elif state.retry_count < state.max_retries:
                return "retry"
            else:
                LOGGER.error(f"Max retries ({state.max_retries}) reached")
                return "end"

        def _handle_retry(self, state: AgentState) -> Dict[str, Any]:
            """Handle retry logic"""
            state.retry_count += 1
            LOGGER.info(f"Retrying... Attempt {state.retry_count}/{state.max_retries}")
            LOGGER.info(f"Previous error: {state.error}")

            # Clear previous evaluation
            state.evaluation = None
            state.error = None

            return {"retry_count": state.retry_count}

        def _format_output(self, evaluation: Dict[str, Any]) -> Dict[str, Any]:
            """Format the output to match the expected structure"""
            return {
                "inferred_domain": evaluation.get("inferred_domain"),
                "scores": {
                    "relevance": evaluation.get("relevance", {}).get("score"),
                    "expertise_authority": evaluation.get(
                        "expertise_authority", {}
                    ).get("score"),
                    "depth_specificity": evaluation.get("depth_specificity", {}).get(
                        "score"
                    ),
                    "clarity_conciseness": evaluation.get(
                        "clarity_conciseness", {}
                    ).get("score"),
                    "objectivity_bias": evaluation.get("objectivity_bias", {}).get(
                        "score"
                    ),
                    "completeness": evaluation.get("completeness", {}).get("score"),
                },
                "reasoning": {
                    "relevance": evaluation.get("relevance", {}).get("reasoning"),
                    "expertise_authority": evaluation.get(
                        "expertise_authority", {}
                    ).get("reasoning"),
                    "depth_specificity": evaluation.get("depth_specificity", {}).get(
                        "reasoning"
                    ),
                    "clarity_conciseness": evaluation.get(
                        "clarity_conciseness", {}
                    ).get("reasoning"),
                    "objectivity_bias": evaluation.get("objectivity_bias", {}).get(
                        "reasoning"
                    ),
                    "completeness": evaluation.get("completeness", {}).get("reasoning"),
                },
                "score_summary": self._generate_score_summary(evaluation),
            }

        def _generate_score_summary(self, evaluation: Dict[str, Any]) -> str:
            """Generate a formatted score summary"""
            scores = [
                f"- Relevance: {evaluation.get('relevance', {}).get('score')}/10",
                f"- Expertise/Authority: {evaluation.get('expertise_authority', {}).get('score')}/10",
                f"- Depth and Specificity: {evaluation.get('depth_specificity', {}).get('score')}/10",
                f"- Clarity and Conciseness: {evaluation.get('clarity_conciseness', {}).get('score')}/10",
                f"- Objectivity/Bias: {evaluation.get('objectivity_bias', {}).get('score')}/10",
                f"- Completeness: {evaluation.get('completeness', {}).get('score')}/10",
            ]
            return "\n".join(scores)

        def evaluate(
            self, question: str, source_content: str
        ) -> Optional[Dict[str, Any]]:
            """
            Main function to evaluate a source for a given question

            Args:
                question: The question to evaluate against
                source_content: The source content to evaluate

            Returns:
                Dictionary with evaluation scores or None if failed after retries
            """
            initial_state = AgentState(
                question=question,
                source_content=source_content,
                max_retries=self.max_retries,
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
                LOGGER.error(
                    f"Failed to get valid evaluation after {self.max_retries} retries"
                )
                LOGGER.error(f"Final error: {result.get('error')}")
                return None

        def process_dataframe(
            self,
            df: pd.DataFrame,
            question_col: str = "question_text",
            source_col: str = "source_text",
            include_reasoning: bool = False,
            progress_bar: bool = True,
            save_path: Optional[str] = None,
            skip_existing: bool = True,
        ) -> pd.DataFrame:
            """
            Process a pandas DataFrame with question-source pairs and add score columns

            Args:
                df: DataFrame with question and source columns
                question_col: Name of the question column
                source_col: Name of the source content column
                include_reasoning: Whether to include reasoning columns
                progress_bar: Whether to show progress bar
                save_path: Optional path to save the results as CSV
                skip_existing: Whether to skip rows that already have scores

            Returns:
                DataFrame with added score columns
            """

            # Create a copy to avoid modifying original
            result_df = df.copy()

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
                "average_score",
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
                score_columns = [
                    c for c in result_columns if "score" in c
                ]  # for skipping existing rows

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
                    if skip_existing:
                        # Skip rows that already have scores
                        has_existing_scores = any(
                            col in result_df.columns and pd.notna(row[col])
                            for col in score_columns
                        )
                        if has_existing_scores:
                            skipped_rows += 1
                            continue

                    if pd.isna(row[question_col]) or pd.isna(row[source_col]):
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
                        result_df.at[idx, "inferred_domain"] = evaluation[
                            "inferred_domain"
                        ]
                        result_df.at[idx, "relevance_score"] = evaluation["scores"][
                            "relevance"
                        ]
                        result_df.at[idx, "expertise_authority_score"] = evaluation[
                            "scores"
                        ]["expertise_authority"]
                        result_df.at[idx, "depth_specificity_score"] = evaluation[
                            "scores"
                        ]["depth_specificity"]
                        result_df.at[idx, "clarity_conciseness_score"] = evaluation[
                            "scores"
                        ]["clarity_conciseness"]
                        result_df.at[idx, "objectivity_bias_score"] = evaluation[
                            "scores"
                        ]["objectivity_bias"]
                        result_df.at[idx, "completeness_score"] = evaluation["scores"][
                            "completeness"
                        ]

                        # Calculate average score
                        scores = [
                            v for v in evaluation["scores"].values() if v is not None
                        ]
                        result_df.at[idx, "average_score"] = (
                            sum(scores) / len(scores) if scores else None
                        )

                        # Add reasoning if requested
                        if include_reasoning:
                            result_df.at[idx, "relevance_reasoning"] = evaluation[
                                "reasoning"
                            ]["relevance"]
                            result_df.at[idx, "expertise_authority_reasoning"] = (
                                evaluation["reasoning"]["expertise_authority"]
                            )
                            result_df.at[idx, "depth_specificity_reasoning"] = (
                                evaluation["reasoning"]["depth_specificity"]
                            )
                            result_df.at[idx, "clarity_conciseness_reasoning"] = (
                                evaluation["reasoning"]["clarity_conciseness"]
                            )
                            result_df.at[idx, "objectivity_bias_reasoning"] = (
                                evaluation["reasoning"]["objectivity_bias"]
                            )
                            result_df.at[idx, "completeness_reasoning"] = evaluation[
                                "reasoning"
                            ]["completeness"]
                    else:
                        result_df.at[idx, "evaluation_error"] = (
                            "Failed to evaluate after retries"
                        )

                    processed_rows += 1

                except Exception as e:
                    LOGGER.error(f"Error processing row {idx}: {e}")
                    result_df.at[idx, "evaluation_error"] = str(e)
                    error_rows += 1

            LOGGER.info(
                f"Processing complete: {processed_rows} processed, {skipped_rows} skipped, {error_rows} errors out of {total_rows} total rows"
            )

            if save_path:
                # Save the results to a CSV file
                result_df.to_csv(save_path, index=False)
                LOGGER.info(f"Results saved to {save_path}")

            return result_df

except ImportError as e:
    LOGGER.warning(
        f"ImportError: {str(e)}. Please ensure all dependencies are installed."
    )

    class SourceEvaluationAgent:
        """Placeholder for SourceEvaluationAgent when dependencies are missing"""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "SourceEvaluationAgent requires langgraph, langchain and pydantic to be installed."
            )
