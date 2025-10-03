"""
domain_analysis_agent.py
Agent for domain extraction, guessing, and assessment tasks.
"""

import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field, field_validator, model_validator
from tqdm import tqdm

from prompts_templates.domain import (
    SRC_DOMAIN_EXTRACTION_PROMPT,
    QUESTION_DOMAIN_GUESS_PROMPT,
    QUESTION_DOMAIN_ASSESS_PROMPT,
)

LOGGER = logging.getLogger(__name__)


def _is_empty_text(text: str) -> bool:
    """Check if the text is empty or only whitespace"""
    if not text or text.strip() == "":
        return True
    if text.lower() == "nan":
        return True
    return False


class OperationMode(str, Enum):
    """Operation modes for domain analysis"""
    EXTRACT = "extract"  # Extract domains from source text
    GUESS = "guess"  # Guess domains needed for question
    ASSESS = "assess"  # Assess question against available terms


# Pydantic Models
class DomainSuggestion(BaseModel):
    """Model for a single domain/subdomain/keyword suggestion"""
    term: str = Field(..., description="The domain, subdomain, or keyword term")
    type: str = Field(..., description="Type: domain, subdomain, or keyword")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")
    reason: str = Field(..., description="Explanation for this suggestion")


class DomainExtractionResult(BaseModel):
    """Result for source domain extraction"""
    suggestions: List[DomainSuggestion] = Field(..., description="List of domain suggestions")
    total_suggestions: int = Field(..., ge=0, le=10, description="Total number of suggestions")
    primary_theme: str = Field(..., description="Main identified theme")

    @model_validator(mode="after")
    def validate_total_matches_length(self):
        """Ensure total_suggestions matches the length of suggestions list"""
        if self.total_suggestions != len(self.suggestions):
            self.total_suggestions = len(self.suggestions)
        return self


class DomainGuessResult(BaseModel):
    """Result for question domain guessing"""
    suggestions: List[DomainSuggestion] = Field(..., description="List of domain suggestions")
    total_suggestions: int = Field(..., ge=0, le=10, description="Total number of suggestions")
    question_category: str = Field(..., description="Identified question type/category")

    @model_validator(mode="after")
    def validate_total_matches_length(self):
        """Ensure total_suggestions matches the length of suggestions list"""
        if self.total_suggestions != len(self.suggestions):
            self.total_suggestions = len(self.suggestions)
        return self


class SelectedTerm(BaseModel):
    """Model for a selected term with relevance score"""
    term: str = Field(..., description="The selected term")
    type: str = Field(..., description="Type: domain, subdomain, or keyword")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score 0-1")
    reason: str = Field(..., description="Explanation of relevance")


class DomainAssessmentResult(BaseModel):
    """Result for domain assessment against available terms"""
    selected_terms: List[SelectedTerm] = Field(..., description="List of selected terms")
    total_selected: int = Field(..., ge=0, le=10, description="Total number of selected terms")
    question_intent: str = Field(..., description="Brief description of question intent")
    primary_topics: List[str] = Field(..., description="Primary topics identified")

    @model_validator(mode="after")
    def validate_total_matches_length(self):
        """Ensure total_selected matches the length of selected_terms list"""
        if self.total_selected != len(self.selected_terms):
            self.total_selected = len(self.selected_terms)
        return self


class AgentState(BaseModel):
    """State for the LangGraph domain analysis agent"""
    mode: OperationMode
    text_source: Optional[str] = None
    question: Optional[str] = None
    available_terms: Optional[str] = None  # JSON string of available terms
    result: Optional[Union[DomainExtractionResult, DomainGuessResult, DomainAssessmentResult]] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


class DomainAnalysisAgent:
    """
    Unified LangGraph agent for domain analysis tasks:
    - Extract domains from text sources
    - Guess domains needed to answer questions
    - Assess questions against available domain terms
    """

    def __init__(
            self,
            llm: BaseChatModel,
            max_retries: int = 3,
            batch_size: int = 10,
    ):
        """
        Initialize the domain analysis agent.

        Parameters
        ----------
        llm : BaseChatModel
            Language model to use for analysis
        max_retries : int, optional
            Maximum number of retries for parsing. Default is 3.
        batch_size : int, optional
            Default batch size for batch processing. Default is 10.
        """
        self.llm = llm
        self.max_retries = max_retries
        self.batch_size = batch_size

        # Set up parsers for each mode
        self.extraction_parser = PydanticOutputParser(pydantic_object=DomainExtractionResult)
        self.guess_parser = PydanticOutputParser(pydantic_object=DomainGuessResult)
        self.assessment_parser = PydanticOutputParser(pydantic_object=DomainAssessmentResult)

        # Set up fixing parsers
        self.extraction_fixing_parser = OutputFixingParser.from_llm(
            parser=self.extraction_parser, llm=self.llm
        )
        self.guess_fixing_parser = OutputFixingParser.from_llm(
            parser=self.guess_parser, llm=self.llm
        )
        self.assessment_fixing_parser = OutputFixingParser.from_llm(
            parser=self.assessment_parser, llm=self.llm
        )

        # Create prompt templates
        self.extraction_template = self._create_prompt_template(
            SRC_DOMAIN_EXTRACTION_PROMPT,
            ["text_source"],
            self.extraction_parser
        )
        self.guess_template = self._create_prompt_template(
            QUESTION_DOMAIN_GUESS_PROMPT,
            ["question"],
            self.guess_parser
        )
        self.assessment_template = self._create_prompt_template(
            QUESTION_DOMAIN_ASSESS_PROMPT,
            ["question", "available_terms"],
            self.assessment_parser
        )

        # Build the workflow graph
        self.graph = self._build_graph()

        # Check if LLM supports batch processing
        self._is_openai_llm = self._check_openai_llm()

    def _check_openai_llm(self) -> bool:
        """Check if the LLM is from OpenAI for batch processing"""
        try:
            from langchain_openai import ChatOpenAI, AzureChatOpenAI
            return isinstance(self.llm, (ChatOpenAI, AzureChatOpenAI))
        except ImportError:
            LOGGER.debug("langchain_openai not installed, batch processing unavailable")
            return False
        except Exception as e:
            LOGGER.debug(f"Could not determine if LLM is OpenAI: {e}")
            return False

    def _create_prompt_template(
            self,
            template: str,
            input_vars: List[str],
            parser: PydanticOutputParser
    ) -> PromptTemplate:
        """Create a prompt template with format instructions"""
        return PromptTemplate(
            template=template,
            input_variables=input_vars,
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

    def _get_parser_for_mode(self, mode: OperationMode) -> tuple:
        """Get the appropriate parser and fixing parser for the operation mode"""
        if mode == OperationMode.EXTRACT:
            return self.extraction_parser, self.extraction_fixing_parser
        elif mode == OperationMode.GUESS:
            return self.guess_parser, self.guess_fixing_parser
        elif mode == OperationMode.ASSESS:
            return self.assessment_parser, self.assessment_fixing_parser
        else:
            raise ValueError(f"Unknown operation mode: {mode}")

    def _get_template_for_mode(self, mode: OperationMode) -> PromptTemplate:
        """Get the appropriate prompt template for the operation mode"""
        if mode == OperationMode.EXTRACT:
            return self.extraction_template
        elif mode == OperationMode.GUESS:
            return self.guess_template
        elif mode == OperationMode.ASSESS:
            return self.assessment_template
        else:
            raise ValueError(f"Unknown operation mode: {mode}")

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("analyze", self._analyze)
        workflow.add_node("validate", self._validate_response)
        workflow.add_node("retry", self._handle_retry)

        # Add edges
        workflow.set_entry_point("analyze")
        workflow.add_edge("analyze", "validate")

        # Conditional edges from validate
        workflow.add_conditional_edges(
            "validate",
            self._should_retry,
            {"retry": "retry", "end": END}
        )

        # From retry back to analyze
        workflow.add_edge("retry", "analyze")

        return workflow.compile()

    def _analyze(self, state: AgentState) -> Dict[str, Any]:
        """Analyze based on the operation mode"""
        try:
            # Get the appropriate template
            template = self._get_template_for_mode(state.mode)
            parser, fixing_parser = self._get_parser_for_mode(state.mode)

            # Format the prompt based on mode
            if state.mode == OperationMode.EXTRACT:
                prompt = template.format(text_source=state.text_source)
            elif state.mode == OperationMode.GUESS:
                prompt = template.format(question=state.question)
            elif state.mode == OperationMode.ASSESS:
                prompt = template.format(
                    question=state.question,
                    available_terms=state.available_terms
                )

            LOGGER.debug(f"Sending prompt to LLM for mode: {state.mode}")

            # Get response from LLM
            response = self.llm.invoke(prompt)

            # Extract content
            if hasattr(response, "content"):
                content = response.content
            else:
                content = str(response)

            LOGGER.debug(f"Raw LLM Response (first 500 chars): {content[:500]}...")

            try:
                # Try fixing parser first
                result = fixing_parser.parse(content)
                state.result = result
                state.error = None
                LOGGER.info(f"Successfully parsed with fixing parser for mode: {state.mode}")

            except Exception as parse_error:
                LOGGER.warning(f"Fixing parser failed, trying regular parser: {parse_error}")
                try:
                    result = parser.parse(content)
                    state.result = result
                    state.error = None
                    LOGGER.info(f"Successfully parsed with regular parser for mode: {state.mode}")
                except Exception as e:
                    LOGGER.error(f"All parsing attempts failed: {e}")
                    state.error = f"Parsing error: {str(e)}"
                    state.result = None

        except Exception as e:
            LOGGER.error(f"Analysis error: {e}")
            state.error = str(e)
            state.result = None

        return {"result": state.result, "error": state.error}

    def _validate_response(self, state: AgentState) -> Dict[str, Any]:
        """Validate the response"""
        if state.result is None:
            state.error = state.error or "No result generated"
            return {"error": state.error}

        try:
            # Check if result is the correct type based on mode
            expected_type = {
                OperationMode.EXTRACT: DomainExtractionResult,
                OperationMode.GUESS: DomainGuessResult,
                OperationMode.ASSESS: DomainAssessmentResult,
            }[state.mode]

            if isinstance(state.result, expected_type):
                state.error = None
                LOGGER.info(f"Validation successful for mode: {state.mode}")
            elif isinstance(state.result, dict):
                # Try to convert dict to appropriate model
                state.result = expected_type(**state.result)
                state.error = None
                LOGGER.info(f"Validation successful - converted dict for mode: {state.mode}")
            else:
                raise ValueError(f"Unexpected result type: {type(state.result)}")

        except Exception as e:
            LOGGER.error(f"Validation error: {e}")
            state.error = f"Validation error: {str(e)}"
            state.result = None

        return {"result": state.result, "error": state.error}

    def _should_retry(self, state: AgentState) -> str:
        """Determine if we should retry or end"""
        if state.error is None and state.result is not None:
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

        # Clear previous result
        state.result = None
        state.error = None

        return {"retry_count": state.retry_count}

    # Public API methods

    def extract_domains(self, text_source: str) -> Optional[Dict[str, Any]]:
        """
        Extract domains, subdomains, and keywords from text source.

        Parameters
        ----------
        text_source : str
            The text to analyze for domain extraction

        Returns
        -------
        Optional[Dict[str, Any]]
            Dictionary with suggestions, total_suggestions, and primary_theme
        """
        initial_state = AgentState(
            mode=OperationMode.EXTRACT,
            text_source=text_source,
            max_retries=self.max_retries,
        )

        result = self.graph.invoke(initial_state.model_dump())

        if result.get("result"):
            result_obj = result["result"]
            if isinstance(result_obj, DomainExtractionResult):
                return result_obj.model_dump()
            elif isinstance(result_obj, dict):
                return result_obj

        LOGGER.error(f"Failed to extract domains after {self.max_retries} retries")
        LOGGER.error(f"Final error: {result.get('error')}")
        return None

    def guess_domains(self, question: str) -> Optional[Dict[str, Any]]:
        """
        Guess which domains are needed to answer a question.

        Parameters
        ----------
        question : str
            The question to analyze

        Returns
        -------
        Optional[Dict[str, Any]]
            Dictionary with suggestions, total_suggestions, and question_category
        """
        initial_state = AgentState(
            mode=OperationMode.GUESS,
            question=question,
            max_retries=self.max_retries,
        )

        result = self.graph.invoke(initial_state.model_dump())

        if result.get("result"):
            result_obj = result["result"]
            if isinstance(result_obj, DomainGuessResult):
                return result_obj.model_dump()
            elif isinstance(result_obj, dict):
                return result_obj

        LOGGER.error(f"Failed to guess domains after {self.max_retries} retries")
        LOGGER.error(f"Final error: {result.get('error')}")
        return None

    def assess_domains(
            self,
            question: str,
            available_terms: Union[List[str], List[Dict], str]
    ) -> Optional[Dict[str, Any]]:
        """
        Assess which available terms are most relevant to a question.

        Parameters
        ----------
        question : str
            The question to analyze
        available_terms : Union[List[str], List[Dict], str]
            List of available terms (as strings, dicts, or JSON string)

        Returns
        -------
        Optional[Dict[str, Any]]
            Dictionary with selected_terms, total_selected, question_intent, and primary_topics
        """
        # Convert available_terms to JSON string if needed
        if isinstance(available_terms, str):
            terms_str = available_terms
        else:
            import json
            terms_str = json.dumps(available_terms, indent=2)

        initial_state = AgentState(
            mode=OperationMode.ASSESS,
            question=question,
            available_terms=terms_str,
            max_retries=self.max_retries,
        )

        result = self.graph.invoke(initial_state.model_dump())

        if result.get("result"):
            result_obj = result["result"]
            if isinstance(result_obj, DomainAssessmentResult):
                return result_obj.model_dump()
            elif isinstance(result_obj, dict):
                return result_obj

        LOGGER.error(f"Failed to assess domains after {self.max_retries} retries")
        LOGGER.error(f"Final error: {result.get('error')}")
        return None

    def extract_domains_batch(
            self,
            text_sources: List[str]
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Extract domains from multiple text sources in batch.

        Parameters
        ----------
        text_sources : List[str]
            List of text sources to analyze

        Returns
        -------
        List[Optional[Dict[str, Any]]]
            List of extraction results
        """
        if not self._is_openai_llm:
            LOGGER.info("Batch processing not available, using sequential processing")
            return [self.extract_domains(text) for text in text_sources]

        return self._batch_process(
            OperationMode.EXTRACT,
            text_sources=text_sources
        )

    def guess_domains_batch(
            self,
            questions: List[str]
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Guess domains for multiple questions in batch.

        Parameters
        ----------
        questions : List[str]
            List of questions to analyze

        Returns
        -------
        List[Optional[Dict[str, Any]]]
            List of guess results
        """
        if not self._is_openai_llm:
            LOGGER.info("Batch processing not available, using sequential processing")
            return [self.guess_domains(q) for q in questions]

        return self._batch_process(
            OperationMode.GUESS,
            questions=questions
        )

    def assess_domains_batch(
            self,
            questions: List[str],
            available_terms: Union[List[str], List[Dict], str]
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Assess domains for multiple questions against the same available terms.

        Parameters
        ----------
        questions : List[str]
            List of questions to analyze
        available_terms : Union[List[str], List[Dict], str]
            Available terms (same for all questions)

        Returns
        -------
        List[Optional[Dict[str, Any]]]
            List of assessment results
        """
        if not self._is_openai_llm:
            LOGGER.info("Batch processing not available, using sequential processing")
            return [self.assess_domains(q, available_terms) for q in questions]

        # Convert available_terms once
        if isinstance(available_terms, str):
            terms_str = available_terms
        else:
            import json
            terms_str = json.dumps(available_terms, indent=2)

        return self._batch_process(
            OperationMode.ASSESS,
            questions=questions,
            available_terms=[terms_str] * len(questions)
        )

    def _batch_process(
            self,
            mode: OperationMode,
            text_sources: Optional[List[str]] = None,
            questions: Optional[List[str]] = None,
            available_terms: Optional[List[str]] = None,
    ) -> List[Optional[Dict[str, Any]]]:
        """Internal method for batch processing"""
        template = self._get_template_for_mode(mode)
        parser, fixing_parser = self._get_parser_for_mode(mode)

        # Prepare prompts
        prompts = []
        if mode == OperationMode.EXTRACT:
            for text in text_sources:
                prompts.append(template.format(text_source=text))
        elif mode == OperationMode.GUESS:
            for question in questions:
                prompts.append(template.format(question=question))
        elif mode == OperationMode.ASSESS:
            for question, terms in zip(questions, available_terms):
                prompts.append(template.format(question=question, available_terms=terms))

        LOGGER.info(f"Processing batch of {len(prompts)} items for mode: {mode}")

        try:
            # Use LangChain's batch method
            responses = self.llm.batch(prompts)

            # Parse each response
            results = []
            for i, response in enumerate(responses):
                try:
                    if hasattr(response, "content"):
                        content = response.content
                    else:
                        content = str(response)

                    # Try fixing parser
                    try:
                        result = fixing_parser.parse(content)
                        if hasattr(result, 'model_dump'):
                            results.append(result.model_dump())
                        else:
                            results.append(result)
                    except Exception:
                        # Fallback to regular parser
                        result = parser.parse(content)
                        if hasattr(result, 'model_dump'):
                            results.append(result.model_dump())
                        else:
                            results.append(result)

                except Exception as e:
                    LOGGER.error(f"Error processing batch item {i}: {e}")
                    results.append(None)

            return results

        except Exception as e:
            LOGGER.error(f"Batch processing failed: {e}")
            LOGGER.info("Falling back to sequential processing")

            # Fallback to sequential
            if mode == OperationMode.EXTRACT:
                return [self.extract_domains(text) for text in text_sources]
            elif mode == OperationMode.GUESS:
                return [self.guess_domains(q) for q in questions]
            elif mode == OperationMode.ASSESS:
                return [self.assess_domains(q, t) for q, t in zip(questions, available_terms)]

    def process_dataframe(
            self,
            df: pd.DataFrame,
            mode: OperationMode,
            text_source_col: Optional[str] = None,
            question_col: Optional[str] = None,
            available_terms: Optional[Union[List[str], List[Dict], str]] = None,
            progress_bar: bool = True,
            save_path: Optional[str] = None,
            skip_existing: bool = True,
            checkpoint_batch_size: Optional[int] = None,
            use_batch_processing: bool = True,
            batch_size: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Process a DataFrame based on the specified operation mode.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to process
        mode : OperationMode
            Operation mode (EXTRACT, GUESS, or ASSESS)
        text_source_col : str, optional
            Column name for text sources (required for EXTRACT mode)
        question_col : str, optional
            Column name for questions (required for GUESS and ASSESS modes)
        available_terms : Union[List[str], List[Dict], str], optional
            Available terms for ASSESS mode
        progress_bar : bool, optional
            Show progress bar. Default is True.
        save_path : str, optional
            Path to save results
        skip_existing : bool, optional
            Skip rows with existing results. Default is True.
        checkpoint_batch_size : int, optional
            Save checkpoint every N rows
        use_batch_processing : bool, optional
            Use batch processing if available. Default is True.
        batch_size : int, optional
            Batch size for processing

        Returns
        -------
        pd.DataFrame
            DataFrame with added result columns
        """
        result_df = df.copy()
        batch_size = batch_size or self.batch_size
        should_batch = use_batch_processing and self._is_openai_llm

        # Validate required columns
        if mode == OperationMode.EXTRACT and not text_source_col:
            raise ValueError("text_source_col required for EXTRACT mode")
        if mode in [OperationMode.GUESS, OperationMode.ASSESS] and not question_col:
            raise ValueError("question_col required for GUESS and ASSESS modes")
        if mode == OperationMode.ASSESS and available_terms is None:
            raise ValueError("available_terms required for ASSESS mode")

        # Add result columns based on mode
        if mode in [OperationMode.EXTRACT, OperationMode.GUESS]:
            result_columns = [
                "suggestions",
                "total_suggestions",
                "primary_theme" if mode == OperationMode.EXTRACT else "question_category",
                "domain_analysis_error",
            ]
        else:  # ASSESS
            result_columns = [
                "selected_terms",
                "total_selected",
                "question_intent",
                "primary_topics",
                "domain_analysis_error",
            ]

        for col in result_columns:
            if col not in result_df.columns:
                result_df[col] = None

        # Collect rows to process
        rows_to_process = []
        indices_to_process = []

        for idx, row in result_df.iterrows():
            # Skip if has existing results
            if skip_existing and pd.notna(row.get("total_suggestions")) or pd.notna(row.get("total_selected")):
                continue

            # Check for empty inputs
            if mode == OperationMode.EXTRACT:
                if pd.isna(row[text_source_col]) or _is_empty_text(row[text_source_col]):
                    result_df.at[idx, "domain_analysis_error"] = "Missing or empty text source"
                    continue
            else:
                if pd.isna(row[question_col]) or _is_empty_text(row[question_col]):
                    result_df.at[idx, "domain_analysis_error"] = "Missing or empty question"
                    continue

            rows_to_process.append(row)
            indices_to_process.append(idx)

        if not rows_to_process:
            LOGGER.info("No rows to process")
            return result_df

        LOGGER.info(f"Processing {len(rows_to_process)} rows with mode: {mode}")

        # Process based on batch mode
        if should_batch:
            results = self._process_dataframe_batch(
                rows_to_process, indices_to_process, result_df, mode,
                text_source_col, question_col, available_terms,
                batch_size, progress_bar, save_path, checkpoint_batch_size
            )
        else:
            results = self._process_dataframe_sequential(
                rows_to_process, indices_to_process, result_df, mode,
                text_source_col, question_col, available_terms,
                progress_bar, save_path, checkpoint_batch_size
            )

        if save_path:
            result_df.to_csv(save_path, index=False)
            LOGGER.info(f"Final results saved to {save_path}")

        return result_df

    def _process_dataframe_batch(
            self, rows, indices, result_df, mode, text_col, question_col,
            available_terms, batch_size, progress_bar, save_path, checkpoint_size
    ):
        """Process DataFrame using batch API"""
        total_batches = (len(rows) + batch_size - 1) // batch_size
        iterator = tqdm(range(0, len(rows), batch_size), total=total_batches) if progress_bar else range(0, len(rows),
                                                                                                         batch_size)

        processed = 0
        errors = 0

        for batch_start in iterator:
            batch_end = min(batch_start + batch_size, len(rows))
            batch_rows = rows[batch_start:batch_end]
            batch_indices = indices[batch_start:batch_end]

            try:
                # Prepare batch inputs
                if mode == OperationMode.EXTRACT:
                    texts = [row[text_col] for row in batch_rows]
                    batch_results = self.extract_domains_batch(texts)
                elif mode == OperationMode.GUESS:
                    questions = [row[question_col] for row in batch_rows]
                    batch_results = self.guess_domains_batch(questions)
                else:  # ASSESS
                    questions = [row[question_col] for row in batch_rows]
                    batch_results = self.assess_domains_batch(questions, available_terms)

                # Update DataFrame
                for idx, result in zip(batch_indices, batch_results):
                    if result:
                        if mode in [OperationMode.EXTRACT, OperationMode.GUESS]:
                            result_df.at[idx, "suggestions"] = str(result["suggestions"])
                            result_df.at[idx, "total_suggestions"] = result["total_suggestions"]
                            result_df.at[
                                idx, "primary_theme" if mode == OperationMode.EXTRACT else "question_category"] = result.get(
                                "primary_theme") or result.get("question_category")
                        else:  # ASSESS
                            result_df.at[idx, "selected_terms"] = str(result["selected_terms"])
                            result_df.at[idx, "total_selected"] = result["total_selected"]
                            result_df.at[idx, "question_intent"] = result["question_intent"]
                            result_df.at[idx, "primary_topics"] = str(result["primary_topics"])
                        processed += 1
                    else:
                        result_df.at[idx, "domain_analysis_error"] = "Analysis failed"
                        errors += 1

                # Checkpoint
                if save_path and checkpoint_size and processed % checkpoint_size == 0:
                    result_df.to_csv(save_path, index=False)
                    LOGGER.info(f"Checkpoint saved at {processed} rows")

            except Exception as e:
                LOGGER.error(f"Batch error: {e}")
                for idx in batch_indices:
                    result_df.at[idx, "domain_analysis_error"] = f"Batch error: {str(e)}"
                errors += len(batch_indices)

            if progress_bar:
                iterator.set_postfix({"Processed": processed, "Errors": errors})

        LOGGER.info(f"Batch processing complete: {processed} successful, {errors} errors")
        return result_df

    def _process_dataframe_sequential(
            self, rows, indices, result_df, mode, text_col, question_col,
            available_terms, progress_bar, save_path, checkpoint_size
    ):
        """Process DataFrame sequentially"""
        iterator = tqdm(zip(indices, rows), total=len(rows)) if progress_bar else zip(indices, rows)

        processed = 0
        errors = 0

        for idx, row in iterator:
            try:
                # Call appropriate method
                if mode == OperationMode.EXTRACT:
                    result = self.extract_domains(row[text_col])
                elif mode == OperationMode.GUESS:
                    result = self.guess_domains(row[question_col])
                else:  # ASSESS
                    result = self.assess_domains(row[question_col], available_terms)

                if result:
                    if mode in [OperationMode.EXTRACT, OperationMode.GUESS]:
                        result_df.at[idx, "suggestions"] = str(result["suggestions"])
                        result_df.at[idx, "total_suggestions"] = result["total_suggestions"]
                        result_df.at[
                            idx, "primary_theme" if mode == OperationMode.EXTRACT else "question_category"] = result.get(
                            "primary_theme") or result.get("question_category")
                    else:  # ASSESS
                        result_df.at[idx, "selected_terms"] = str(result["selected_terms"])
                        result_df.at[idx, "total_selected"] = result["total_selected"]
                        result_df.at[idx, "question_intent"] = result["question_intent"]
                        result_df.at[idx, "primary_topics"] = str(result["primary_topics"])
                    processed += 1
                else:
                    result_df.at[idx, "domain_analysis_error"] = "Analysis failed"
                    errors += 1

                # Checkpoint
                if save_path and checkpoint_size and processed % checkpoint_size == 0:
                    result_df.to_csv(save_path, index=False)
                    LOGGER.info(f"Checkpoint saved at {processed} rows")

            except KeyboardInterrupt:
                LOGGER.warning("Processing interrupted")
                break
            except Exception as e:
                LOGGER.error(f"Error processing row {idx}: {e}")
                result_df.at[idx, "domain_analysis_error"] = str(e)
                errors += 1

            if progress_bar:
                iterator.set_postfix({"Processed": processed, "Errors": errors})

        LOGGER.info(f"Sequential processing complete: {processed} successful, {errors} errors")
        return result_df