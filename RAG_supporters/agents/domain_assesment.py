"""Domain analysis agent for domain extraction, guessing, and assessment tasks.

Agent for domain extraction, guessing, and assessment tasks.
"""

import json
import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Union

LOGGER = logging.getLogger(__name__)

try:
    import pandas as pd
    from langchain_classic.output_parsers import OutputFixingParser
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_core.language_models import BaseChatModel
    from langchain_core.prompts import PromptTemplate
    from langgraph.graph import END, StateGraph
    from pydantic import BaseModel, Field, field_validator, model_validator
    from tqdm import tqdm

    from RAG_supporters.prompts_templates.domain_extraction import (
        QUESTION_TOPIC_RELEVANCE_PROB_PROMPT,
        QUESTION_DOMAIN_ASSESS_PROMPT,
        QUESTION_DOMAIN_GUESS_PROMPT,
        SRC_DOMAIN_EXTRACTION_PROMPT,
    )
    from RAG_supporters.utils.text_utils import is_empty_text

    class OperationMode(str, Enum):
        """Operation modes for domain analysis."""

        EXTRACT = "extract"  # Extract domains from source text
        GUESS = "guess"  # Guess domains needed for question
        ASSESS = "assess"  # Assess question against available terms
        TOPIC_RELEVANCE_PROB = (
            "topic_relevance_prob"  # Assess question-topic relevance probabilities
        )

    # Pydantic Models
    class DomainSuggestion(BaseModel):
        """Model for a single domain/subdomain/keyword suggestion."""

        term: str = Field(..., description="The domain, subdomain, or keyword term")
        type: str = Field(..., description="Type: domain, subdomain, or keyword")
        confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")
        reason: str = Field(..., description="Explanation for this suggestion")

    class DomainExtractionResult(BaseModel):
        """Result for source domain extraction."""

        suggestions: List[DomainSuggestion] = Field(..., description="List of domain suggestions")
        total_suggestions: int = Field(..., ge=0, le=10, description="Total number of suggestions")
        primary_theme: str = Field(..., description="Main identified theme")

        @model_validator(mode="after")
        def validate_suggestions_length(self):
            """Ensure total_suggestions matches the length of suggestions list."""
            if self.total_suggestions != len(self.suggestions):
                LOGGER.warning(
                    f"total_suggestions mismatch: {self.total_suggestions} vs {len(self.suggestions)}, correcting"
                )
                self.total_suggestions = len(self.suggestions)
            return self

    class DomainGuessResult(BaseModel):
        """Result for question domain guessing."""

        suggestions: List[DomainSuggestion] = Field(..., description="List of domain suggestions")
        total_suggestions: int = Field(..., ge=0, le=10, description="Total number of suggestions")
        question_category: str = Field(..., description="Identified question type/category")

        @model_validator(mode="after")
        def validate_suggestions_length(self):
            """Ensure total_suggestions matches the length of suggestions list."""
            if self.total_suggestions != len(self.suggestions):
                LOGGER.warning(
                    f"total_suggestions mismatch: {self.total_suggestions} vs {len(self.suggestions)}, correcting"
                )
                self.total_suggestions = len(self.suggestions)
            return self

    class SelectedTerm(BaseModel):
        """Model for a selected term with relevance score."""

        term: str = Field(..., description="The selected term")
        type: str = Field(..., description="Type: domain, subdomain, or keyword")
        relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score 0-1")
        reason: str = Field(..., description="Explanation of relevance")

    class DomainAssessmentResult(BaseModel):
        """Result for domain assessment against available terms."""

        selected_terms: List[SelectedTerm] = Field(..., description="List of selected terms")
        total_selected: int = Field(..., ge=0, le=10, description="Total number of selected terms")
        question_intent: str = Field(..., description="Brief description of question intent")
        primary_topics: List[str] = Field(..., description="Primary topics identified")

        @model_validator(mode="after")
        def validate_total_selected_length(self):
            """Ensure total_selected matches the length of selected_terms list."""
            if self.total_selected != len(self.selected_terms):
                LOGGER.warning(
                    f"total_selected mismatch: {self.total_selected} vs {len(self.selected_terms)}, correcting"
                )
                self.total_selected = len(self.selected_terms)
            return self

    class TopicRelevanceScore(BaseModel):
        """Model for a single topic descriptor with relevance probability.

        Note: Typically used with cluster descriptors from KeywordClusterer output.
        """

        topic_descriptor: str = Field(
            ..., description="The topic descriptor term (e.g., from cluster analysis)"
        )
        probability: float = Field(
            ..., ge=0.0, le=1.0, description="Probability of semantic connection (0-1)"
        )
        reason: Optional[str] = Field(None, description="Optional explanation for this probability")

    class QuestionTopicRelevanceResult(BaseModel):
        """Result for question-topic relevance assessment.

        Note: Designed to work with topic descriptors from KeywordClusterer
        (RAG_supporters/clustering/keyword_clustering.py). When using KeywordClusterer,
        the topic_descriptors represent the descriptors of the created clusters.
        """

        topic_scores: List[TopicRelevanceScore] = Field(
            ..., description="List of topic descriptors with probabilities"
        )
        total_topics: int = Field(..., ge=0, description="Total number of topics assessed")
        question_summary: Optional[str] = Field(
            None, description="Brief summary of the question's main topic (optional)"
        )

        @model_validator(mode="after")
        def validate_topic_scores_length(self):
            """Ensure total_topics matches the length of topic_scores list."""
            if self.total_topics != len(self.topic_scores):
                LOGGER.warning(
                    f"total_topics mismatch: {self.total_topics} vs {len(self.topic_scores)}, correcting"
                )
                self.total_topics = len(self.topic_scores)
            return self

    class AgentState(BaseModel):
        """State for the LangGraph domain analysis agent."""

        mode: OperationMode
        text_source: Optional[str] = None
        question: Optional[str] = None
        available_terms: Optional[str] = None  # JSON string of available terms
        topic_descriptors: Optional[str] = None  # JSON string of topic descriptors
        result: Optional[
            Union[
                DomainExtractionResult,
                DomainGuessResult,
                DomainAssessmentResult,
                QuestionTopicRelevanceResult,
            ]
        ] = None
        error: Optional[str] = None
        retry_count: int = 0
        max_retries: int = 3

    class DomainAnalysisAgent:
        """Unified LangGraph agent for domain analysis tasks.

        - Extract domains from text sources
        - Guess domains needed to answer questions
        - Assess questions against available domain terms
        """

        def __init__(
            self,
            llm: BaseChatModel,
            max_retries: int = 3,
            batch_size: int = 10,
            include_reason: bool = False,
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
            include_reason : bool, optional
                If True, include reasoning explanations in topic relevance assessments.
                (For now) Only applies to assess_topic_relevance_prob operations (TOPIC_RELEVANCE_PROB mode).
                Default is False.
            """
            self.llm = llm
            self.max_retries = max_retries
            self.batch_size = batch_size
            self.include_reason = include_reason

            # Set up parsers for each mode
            self.extraction_parser = PydanticOutputParser(pydantic_object=DomainExtractionResult)
            self.guess_parser = PydanticOutputParser(pydantic_object=DomainGuessResult)
            self.assessment_parser = PydanticOutputParser(pydantic_object=DomainAssessmentResult)
            self.topic_relevance_prob_parser = PydanticOutputParser(
                pydantic_object=QuestionTopicRelevanceResult
            )

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
            self.topic_relevance_prob_fixing_parser = OutputFixingParser.from_llm(
                parser=self.topic_relevance_prob_parser, llm=self.llm
            )

            # Create prompt templates
            self.extraction_template = self._create_prompt_template(
                SRC_DOMAIN_EXTRACTION_PROMPT, ["text_source"], self.extraction_parser
            )
            self.guess_template = self._create_prompt_template(
                QUESTION_DOMAIN_GUESS_PROMPT, ["question"], self.guess_parser
            )
            self.assessment_template = self._create_prompt_template(
                QUESTION_DOMAIN_ASSESS_PROMPT,
                ["question", "available_terms"],
                self.assessment_parser,
            )
            self.topic_relevance_prob_template = self._create_prompt_template(
                QUESTION_TOPIC_RELEVANCE_PROB_PROMPT(include_reason=self.include_reason),
                ["question", "topic_descriptors"],
                self.topic_relevance_prob_parser,
            )

            # Build the workflow graph
            self.graph = self._build_graph()

            # Check if LLM supports batch processing
            self._is_openai_llm = self._check_openai_llm()

        def _check_openai_llm(self) -> bool:
            """Check if the LLM is from OpenAI for batch processing."""
            try:
                from langchain_openai import AzureChatOpenAI, ChatOpenAI

                return isinstance(self.llm, (ChatOpenAI, AzureChatOpenAI))
            except ImportError:
                LOGGER.debug("langchain_openai not installed, batch processing unavailable")
                return False
            except Exception as e:
                LOGGER.debug(f"Could not determine if LLM is OpenAI: {e}")
                return False

        def _create_prompt_template(
            self, template: str, input_vars: List[str], parser: PydanticOutputParser
        ) -> PromptTemplate:
            """Create a prompt template with format instructions."""
            return PromptTemplate(
                template=template,
                input_variables=input_vars,
                partial_variables={"format_instructions": parser.get_format_instructions()},
            )

        def _get_parser_for_mode(self, mode: OperationMode) -> tuple:
            """Get the appropriate parser and fixing parser for the operation mode."""
            if mode == OperationMode.EXTRACT:
                return self.extraction_parser, self.extraction_fixing_parser
            elif mode == OperationMode.GUESS:
                return self.guess_parser, self.guess_fixing_parser
            elif mode == OperationMode.ASSESS:
                return self.assessment_parser, self.assessment_fixing_parser
            elif mode == OperationMode.TOPIC_RELEVANCE_PROB:
                return (
                    self.topic_relevance_prob_parser,
                    self.topic_relevance_prob_fixing_parser,
                )
            else:
                raise ValueError(f"Unknown operation mode: {mode}")

        def _get_template_for_mode(self, mode: OperationMode) -> PromptTemplate:
            """Get the appropriate prompt template for the operation mode."""
            if mode == OperationMode.EXTRACT:
                return self.extraction_template
            elif mode == OperationMode.GUESS:
                return self.guess_template
            elif mode == OperationMode.ASSESS:
                return self.assessment_template
            elif mode == OperationMode.TOPIC_RELEVANCE_PROB:
                return self.topic_relevance_prob_template
            else:
                raise ValueError(f"Unknown operation mode: {mode}")

        def _get_column_prefix(self, mode: OperationMode) -> str:
            """Get column prefix for mode to avoid overwrites."""
            return mode.value  # Returns 'extract', 'guess', 'assess', or 'topic_relevance_prob'

        def _build_graph(self) -> StateGraph:
            """Build the LangGraph workflow."""
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
                "validate", self._should_retry, {"retry": "retry", "end": END}
            )

            # From retry back to analyze
            workflow.add_edge("retry", "analyze")

            return workflow.compile()

        def _analyze(self, state: AgentState) -> Dict[str, Any]:
            """Analyze based on the operation mode."""
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
                        question=state.question, available_terms=state.available_terms
                    )
                elif state.mode == OperationMode.TOPIC_RELEVANCE_PROB:
                    prompt = template.format(
                        question=state.question,
                        topic_descriptors=state.topic_descriptors,
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
                    LOGGER.info(f"Successfully parsed with fixing parser for mode: {state.mode}")

                except Exception as parse_error:
                    LOGGER.warning(f"Fixing parser failed, trying regular parser: {parse_error}")
                    try:
                        result = parser.parse(content)
                        LOGGER.info(
                            f"Successfully parsed with regular parser for mode: {state.mode}"
                        )
                    except Exception as e:
                        LOGGER.error(
                            f"All parsing attempts failed: {type(e).__name__}: {e!r}", exc_info=True
                        )
                        return {
                            "result": None,
                            "error": f"Parsing error: {type(e).__name__}: {str(e) or repr(e)}",
                        }

                return {"result": result, "error": None}

            except Exception as e:
                LOGGER.error(f"Analysis error: {type(e).__name__}: {e!r}", exc_info=True)
                return {"result": None, "error": f"{type(e).__name__}: {str(e) or repr(e)}"}

        def _validate_response(self, state: AgentState) -> Dict[str, Any]:
            """Validate the response."""
            if state.result is None:
                error = state.error or "No result generated"
                return {"error": error}

            try:
                # Check if result is the correct type based on mode
                expected_type = {
                    OperationMode.EXTRACT: DomainExtractionResult,
                    OperationMode.GUESS: DomainGuessResult,
                    OperationMode.ASSESS: DomainAssessmentResult,
                    OperationMode.TOPIC_RELEVANCE_PROB: QuestionTopicRelevanceResult,
                }[state.mode]

                if isinstance(state.result, expected_type):
                    LOGGER.info(f"Validation successful for mode: {state.mode}")
                    return {"result": state.result, "error": None}
                elif isinstance(state.result, dict):
                    # Try to convert dict to appropriate model
                    result = expected_type(**state.result)
                    LOGGER.info(f"Validation successful - converted dict for mode: {state.mode}")
                    return {"result": result, "error": None}
                else:
                    raise ValueError(f"Unexpected result type: {type(state.result)}")

            except Exception as e:
                LOGGER.error(f"Validation error: {type(e).__name__}: {e!r}", exc_info=True)
                return {
                    "result": None,
                    "error": f"Validation error: {type(e).__name__}: {str(e) or repr(e)}",
                }

        def _should_retry(self, state: AgentState) -> str:
            """Determine if we should retry or end."""
            if state.error is None and state.result is not None:
                return "end"
            elif state.retry_count < state.max_retries:
                return "retry"
            else:
                LOGGER.error(
                    f"Max retries ({state.max_retries}) reached. Last error: {state.error}"
                )
                return "end"

        def _handle_retry(self, state: AgentState) -> Dict[str, Any]:
            """Handle retry logic."""
            new_retry_count = state.retry_count + 1
            LOGGER.info(f"Retrying... Attempt {new_retry_count}/{state.max_retries}")
            LOGGER.info(f"Previous error: {state.error}")

            return {"retry_count": new_retry_count, "result": None, "error": None}

        def _extract_result_dict(self, result: Any) -> Optional[Dict[str, Any]]:
            """Extract result as dictionary.

            Parameters
            ----------
            result : Any
                Result object to extract

            Returns
            -------
            Optional[Dict[str, Any]]
                Result as dictionary or None
            """
            if result is None:
                return None
            if isinstance(result, dict):
                return result
            if hasattr(result, "model_dump"):
                return result.model_dump()
            LOGGER.warning(f"Unexpected result type: {type(result)}")
            return None

        def _create_relevance_json_mapping(self, topic_scores: List[Dict[str, Any]]) -> str:
            """Create JSON mapping of topic descriptors to probabilities.

            Parameters
            ----------
            topic_scores : List[Dict[str, Any]]
                List of topic score dictionaries

            Returns
            -------
            str
                JSON string mapping topic_descriptor to probability
            """
            relevance_json = {
                score["topic_descriptor"]: score["probability"] for score in topic_scores
            }
            return json.dumps(relevance_json)

        def _parse_topic_descriptors(
            self, topic_descriptors: Union[List[str], List[Dict], str, Dict]
        ) -> List[str]:
            """Parse and extract topic descriptors from various input formats.

            Handles:
            - List of strings: ["topic1", "topic2"]
            - JSON string: '["topic1", "topic2"]'
            - File path: "path/to/clusters.json"
            - KeywordClusterer dict: {"cluster_stats": {"0": {"topic_descriptors": [...]}, ...}}

            Parameters
            ----------
            topic_descriptors : Union[List[str], List[Dict], str, Dict]
                Topic descriptors in any supported format

            Returns
            -------
            List[str]
                Flat list of topic descriptor strings

            Raises
            ------
            ValueError
                If the input format is invalid or file not found
            """
            import os

            # Case 1: Already a list of strings
            if isinstance(topic_descriptors, list):
                if all(isinstance(x, str) for x in topic_descriptors):
                    return topic_descriptors
                # List of dicts - try to extract strings
                LOGGER.warning(
                    "List of dicts provided, expected list of strings. "
                    "Converting to JSON string representation."
                )
                return [str(item) for item in topic_descriptors]

            # Case 2: String input - could be JSON string or file path
            if isinstance(topic_descriptors, str):
                # Check if it's a file path
                if os.path.isfile(topic_descriptors):
                    LOGGER.info(f"Loading topic descriptors from file: {topic_descriptors}")
                    try:
                        with open(topic_descriptors, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        # Recursively parse the loaded data
                        return self._parse_topic_descriptors(data)
                    except (
                        OSError,
                        IOError,
                        json.JSONDecodeError,
                        UnicodeDecodeError,
                    ) as e:
                        LOGGER.error(f"Failed to load file {topic_descriptors}: {e}")
                        raise ValueError(f"Failed to load topic descriptors from file: {e}")

                # Try to parse as JSON string
                try:
                    parsed = json.loads(topic_descriptors)
                    # Recursively parse the JSON content
                    return self._parse_topic_descriptors(parsed)
                except json.JSONDecodeError:
                    # Not a valid JSON string, treat as single descriptor
                    LOGGER.warning(
                        f"String '{topic_descriptors[:50]}...' is neither a file nor valid JSON, "
                        "treating as single descriptor"
                    )
                    return [topic_descriptors]

            # Case 3: Dict - could be KeywordClusterer format
            if isinstance(topic_descriptors, dict):
                descriptors = []

                # Try KeywordClusterer format with cluster_stats (preferred)
                # cluster_stats structure: {"0": {"topic_descriptors": [...], "size": ..., ...}}
                if "cluster_stats" in topic_descriptors:
                    LOGGER.info("Detected KeywordClusterer 'cluster_stats' format")
                    for cluster_id, stats in topic_descriptors["cluster_stats"].items():
                        if isinstance(stats, dict) and "topic_descriptors" in stats:
                            topic_descs = stats["topic_descriptors"]
                            if isinstance(topic_descs, list):
                                descriptors.extend(topic_descs)
                            else:
                                LOGGER.warning(
                                    f"Cluster {cluster_id} topic_descriptors not a list, skipping"
                                )
                        else:
                            LOGGER.warning(
                                f"Cluster {cluster_id} has no topic_descriptors, skipping"
                            )

                # If we found descriptors from cluster_stats, return unique ones
                if descriptors:
                    unique_descriptors = list(dict.fromkeys(descriptors))
                    LOGGER.info(
                        f"Extracted {len(unique_descriptors)} unique topic descriptors "
                        f"from KeywordClusterer cluster_stats"
                    )
                    return unique_descriptors

                # Fallback: If no cluster_stats but has clusters, warn user
                if "clusters" in topic_descriptors:
                    LOGGER.warning(
                        "Found 'clusters' key but no 'cluster_stats' with topic_descriptors. "
                        "Please ensure KeywordClusterer JSON includes cluster_stats with topic_descriptors. "
                        "Falling back to all keywords from clusters (may be large)."
                    )
                    for cluster_id, keywords in topic_descriptors["clusters"].items():
                        if isinstance(keywords, list):
                            descriptors.extend(keywords)
                    if descriptors:
                        unique_descriptors = list(dict.fromkeys(descriptors))
                        return unique_descriptors
                    else:
                        # No descriptors found in KeywordClusterer format
                        raise ValueError(
                            "KeywordClusterer dict found but no topic descriptors extracted. "
                            "Ensure 'clusters' contains non-empty keyword lists or "
                            "'cluster_stats' contains 'topic_descriptors'."
                        )

                # If no recognized format, warn and raise error
                raise ValueError(
                    "Dict provided but no 'cluster_stats' or 'clusters' keys found. "
                    "Expected KeywordClusterer format with 'clusters' or 'cluster_stats' keys. "
                    f"Found keys: {list(topic_descriptors.keys())}"
                )

            # Unknown format
            raise ValueError(
                f"Unsupported topic_descriptors format: {type(topic_descriptors)}. "
                f"Expected list of strings, JSON string, file path, or KeywordClusterer dict."
            )

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

            Examples
            --------
            >>> result = agent.extract_domains("Machine learning is a subset of AI...")
            >>> print(result['primary_theme'])
            'Artificial Intelligence'
            """
            initial_state = AgentState(
                mode=OperationMode.EXTRACT,
                text_source=text_source,
                max_retries=self.max_retries,
            )

            final_state = self.graph.invoke(initial_state.model_dump())

            result = self._extract_result_dict(final_state.get("result"))
            if result is not None:
                return result

            LOGGER.error(f"Failed to extract domains after {self.max_retries} retries")
            LOGGER.error(f"Final error: {final_state.get('error')}")
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

            Examples
            --------
            >>> result = agent.guess_domains("What is the capital of France?")
            >>> print(result['question_category'])
            'Geography'
            """
            initial_state = AgentState(
                mode=OperationMode.GUESS,
                question=question,
                max_retries=self.max_retries,
            )

            final_state = self.graph.invoke(initial_state.model_dump())

            result = self._extract_result_dict(final_state.get("result"))
            if result is not None:
                return result

            LOGGER.error(f"Failed to guess domains after {self.max_retries} retries")
            LOGGER.error(f"Final error: {final_state.get('error')}")
            return None

        def assess_domains(
            self, question: str, available_terms: Union[List[str], List[Dict], str]
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

            Examples
            --------
            >>> terms = ["physics", "chemistry", "biology"]
            >>> result = agent.assess_domains("What is photosynthesis?", terms)
            >>> print(result['primary_topics'])
            ['biology']
            """
            # Convert available_terms to JSON string if needed
            if isinstance(available_terms, str):
                # Validate JSON string
                try:
                    json.loads(available_terms)
                    terms_str = available_terms
                except json.JSONDecodeError as e:
                    LOGGER.error(f"Invalid JSON in available_terms: {e}")
                    raise ValueError(f"Invalid JSON in available_terms: {e}")
            else:
                terms_str = json.dumps(available_terms, indent=2)

            initial_state = AgentState(
                mode=OperationMode.ASSESS,
                question=question,
                available_terms=terms_str,
                max_retries=self.max_retries,
            )

            final_state = self.graph.invoke(initial_state.model_dump())

            result = self._extract_result_dict(final_state.get("result"))
            if result is not None:
                return result

            LOGGER.error(f"Failed to assess domains after {self.max_retries} retries")
            LOGGER.error(f"Final error: {final_state.get('error')}")
            return None

        def assess_topic_relevance_prob(
            self,
            question: str,
            topic_descriptors: Union[List[str], List[Dict], str, Dict],
        ) -> Optional[Dict[str, Any]]:
            """
            Assess relevance probabilities between a question and topic descriptors.

            This method is designed to work with cluster descriptors from KeywordClusterer
            (RAG_supporters/clustering/keyword_clustering.py). It automatically handles:
            - List of strings: ["topic1", "topic2"]
            - JSON string: '["topic1", "topic2"]'
            - File path to JSON: "path/to/clusters.json"
            - KeywordClusterer dict format: {"cluster_stats": {"0": {"topic_descriptors": [...]}, ...}}

            Note: Topic descriptors are typically shared across all questions for
            performance and memory efficiency.

            Parameters
            ----------
            question : str
                The question to analyze
            topic_descriptors : Union[List[str], List[Dict], str, Dict]
                Topic descriptors in any supported format:
                - List of strings (direct input)
                - JSON string (will be parsed)
                - File path to JSON file (will be loaded and parsed)
                - KeywordClusterer dict with 'cluster_stats' containing 'topic_descriptors'

            Returns
            -------
            Optional[Dict[str, Any]]
                Dictionary with topic_scores, total_topics, and question_summary (optional).
                Each topic_score contains topic_descriptor, probability, and optionally
                reason (if the agent was initialized with ``include_reason=True``).

            Examples
            --------
            >>> # Example 1: List of strings
            >>> descriptors = ["machine learning", "databases", "web development"]
            >>> result = agent.assess_topic_relevance_prob(
            ...     "What is gradient descent?",
            ...     descriptors
            ... )

            >>> # Example 2: File path
            >>> result = agent.assess_topic_relevance_prob(
            ...     "What is gradient descent?",
            ...     "path/to/keyword_clusters.json"
            ... )

            >>> # Example 3: KeywordClusterer dict
            >>> clustering_data = {
            ...     "cluster_stats": {
            ...         "0": {"topic_descriptors": ["machine learning", "AI"], "size": 50},
            ...         "1": {"topic_descriptors": ["databases", "SQL"], "size": 30}
            ...     }
            ... }
            >>> result = agent.assess_topic_relevance_prob(
            ...     "What is gradient descent?",
            ...     clustering_data
            ... )
            """
            # Parse topic_descriptors using helper method
            descriptors_list = self._parse_topic_descriptors(topic_descriptors)

            # Convert to JSON string for the agent
            descriptors_str = json.dumps(descriptors_list, indent=2)

            initial_state = AgentState(
                mode=OperationMode.TOPIC_RELEVANCE_PROB,
                question=question,
                topic_descriptors=descriptors_str,
                max_retries=self.max_retries,
            )

            final_state = self.graph.invoke(initial_state.model_dump())

            result = self._extract_result_dict(final_state.get("result"))
            if result is not None:
                return result

            LOGGER.error(f"Failed to assess topic relevance after {self.max_retries} retries")
            LOGGER.error(f"Final error: {final_state.get('error')}")
            return None

        def extract_domains_batch(
            self,
            text_sources: List[str],
            batch_size: Optional[int] = None,
            show_progress: bool = True,
        ) -> List[Optional[Dict[str, Any]]]:
            """
            Extract domains from multiple text sources in batch.

            Parameters
            ----------
            text_sources : List[str]
                List of text sources to analyze
            batch_size : int, optional
                Batch size for chunking. If None, uses self.batch_size
            show_progress : bool, optional
                Show progress bar. Default is True.

            Returns
            -------
            List[Optional[Dict[str, Any]]]
                List of extraction results
            """
            if not self._is_openai_llm:
                LOGGER.info("Batch processing not available, using sequential processing")
                return [self.extract_domains(text) for text in text_sources]

            batch_size = batch_size or self.batch_size
            return self._batch_process(
                OperationMode.EXTRACT,
                text_sources=text_sources,
                batch_size=batch_size,
                show_progress=show_progress,
            )

        def guess_domains_batch(
            self,
            questions: List[str],
            batch_size: Optional[int] = None,
            show_progress: bool = True,
        ) -> List[Optional[Dict[str, Any]]]:
            """
            Guess domains for multiple questions in batch.

            Parameters
            ----------
            questions : List[str]
                List of questions to analyze
            batch_size : int, optional
                Batch size for chunking. If None, uses self.batch_size
            show_progress : bool, optional
                Show progress bar. Default is True.

            Returns
            -------
            List[Optional[Dict[str, Any]]]
                List of guess results
            """
            if not self._is_openai_llm:
                LOGGER.info("Batch processing not available, using sequential processing")
                return [self.guess_domains(q) for q in questions]

            batch_size = batch_size or self.batch_size
            return self._batch_process(
                OperationMode.GUESS,
                questions=questions,
                batch_size=batch_size,
                show_progress=show_progress,
            )

        def assess_domains_batch(
            self,
            questions: List[str],
            available_terms: Union[List[str], List[Dict], str],
            batch_size: Optional[int] = None,
            show_progress: bool = True,
        ) -> List[Optional[Dict[str, Any]]]:
            """
            Assess domains for multiple questions against the same available terms.

            Parameters
            ----------
            questions : List[str]
                List of questions to analyze
            available_terms : Union[List[str], List[Dict], str]
                Available terms (same for all questions)
            batch_size : int, optional
                Batch size for chunking. If None, uses self.batch_size
            show_progress : bool, optional
                Show progress bar. Default is True.

            Returns
            -------
            List[Optional[Dict[str, Any]]]
                List of assessment results
            """
            if not self._is_openai_llm:
                LOGGER.info("Batch processing not available, using sequential processing")
                return [self.assess_domains(q, available_terms) for q in questions]

            # Convert and validate available_terms once
            if isinstance(available_terms, str):
                try:
                    json.loads(available_terms)
                    terms_str = available_terms
                except json.JSONDecodeError as e:
                    LOGGER.error(f"Invalid JSON in available_terms: {e}")
                    raise ValueError(f"Invalid JSON in available_terms: {e}")
            else:
                terms_str = json.dumps(available_terms, indent=2)

            batch_size = batch_size or self.batch_size
            return self._batch_process(
                OperationMode.ASSESS,
                questions=questions,
                available_terms=[terms_str] * len(questions),
                batch_size=batch_size,
                show_progress=show_progress,
            )

        def assess_topic_relevance_prob_batch(
            self,
            questions: List[str],
            topic_descriptors: Union[List[str], List[Dict], str, Dict],
            batch_size: Optional[int] = None,
            show_progress: bool = True,
        ) -> List[Optional[Dict[str, Any]]]:
            """
            Assess topic relevance for multiple questions against the same topic descriptors.

            Performance Note: Topic descriptors are shared across all questions for memory
            efficiency. This is the typical use case, as topic descriptors from
            KeywordClusterer are usually consistent across a dataset.

            Automatically handles:
            - List of strings
            - JSON string
            - File path to JSON
            - KeywordClusterer dict with cluster_stats containing topic_descriptors

            Parameters
            ----------
            questions : List[str]
                List of questions to analyze
            topic_descriptors : Union[List[str], List[Dict], str, Dict]
                Topic descriptors (same for all questions). Supports:
                - List of strings
                - JSON string
                - File path to KeywordClusterer JSON
                - KeywordClusterer dict with 'cluster_stats' containing 'topic_descriptors'
            batch_size : int, optional
                Batch size for chunking. If None, uses self.batch_size
            show_progress : bool, optional
                Show progress bar. Default is True.

            Returns
            -------
            List[Optional[Dict[str, Any]]]
                List of topic relevance results
            """
            if not self._is_openai_llm:
                LOGGER.info("Batch processing not available, using sequential processing")
                return [self.assess_topic_relevance_prob(q, topic_descriptors) for q in questions]

            # Parse topic_descriptors once using helper method
            descriptors_list = self._parse_topic_descriptors(topic_descriptors)
            descriptors_str = json.dumps(descriptors_list, indent=2)

            batch_size = batch_size or self.batch_size
            return self._batch_process(
                OperationMode.TOPIC_RELEVANCE_PROB,
                questions=questions,
                topic_descriptors=descriptors_str,
                batch_size=batch_size,
                show_progress=show_progress,
            )

        def _batch_process(
            self,
            mode: OperationMode,
            text_sources: Optional[List[str]] = None,
            questions: Optional[List[str]] = None,
            available_terms: Optional[List[str]] = None,
            topic_descriptors: Optional[str] = None,
            batch_size: Optional[int] = None,
            show_progress: bool = True,
        ) -> List[Optional[Dict[str, Any]]]:
            """Process inputs in batches for efficiency.

            This method properly splits input into chunks and processes each chunk
            separately using the LLM's batch API.

            Handles KeyboardInterrupt gracefully by returning partial results.

            Parameters
            ----------
            mode : OperationMode
                Operation mode (extract, guess, assess, or topic_relevance_prob)
            text_sources : Optional[List[str]]
                List of text sources for extraction mode
            questions : Optional[List[str]]
                List of questions for guess/assess/topic_relevance_prob modes
            available_terms : Optional[List[str]]
                Available terms for assess mode (one per question)
            topic_descriptors : Optional[str]
                Topic descriptors JSON string for topic_relevance_prob mode (shared across all questions)
            batch_size : Optional[int]
                Number of items to process in each batch
            show_progress : bool
                Whether to show progress bar

            Returns
            -------
            List[Optional[Dict[str, Any]]]
                List of results, one per input
            """
            batch_size = batch_size or self.batch_size
            template = self._get_template_for_mode(mode)
            parser, fixing_parser = self._get_parser_for_mode(mode)

            # Determine total count based on mode
            if mode == OperationMode.EXTRACT:
                total_items = len(text_sources)
            elif mode == OperationMode.GUESS:
                total_items = len(questions)
            elif mode == OperationMode.ASSESS:
                total_items = len(questions)
            elif mode == OperationMode.TOPIC_RELEVANCE_PROB:
                total_items = len(questions)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            total_batches = (total_items + batch_size - 1) // batch_size
            LOGGER.info(
                f"Processing {total_items} items in {total_batches} batches "
                f"(batch_size={batch_size}) for mode: {mode}"
            )

            all_results = []
            successful_parses = 0
            failed_parses = 0

            try:
                # Process in chunks with progress bar
                batch_iterator = range(0, total_items, batch_size)

                if show_progress:
                    batch_iterator = tqdm(
                        batch_iterator,
                        total=total_batches,
                        desc=f"Processing {mode.value} batches",
                        unit="batch",
                    )

                for batch_start in batch_iterator:
                    batch_end = min(batch_start + batch_size, total_items)
                    current_batch_size = batch_end - batch_start
                    batch_idx = (batch_start // batch_size) + 1

                    LOGGER.debug(
                        f"Processing batch {batch_idx}/{total_batches}: "
                        f"items {batch_start}-{batch_end}"
                    )

                    # Prepare prompts for this batch
                    prompts = []
                    if mode == OperationMode.EXTRACT:
                        batch_texts = text_sources[batch_start:batch_end]
                        for text in batch_texts:
                            prompts.append(template.format(text_source=text))
                    elif mode == OperationMode.GUESS:
                        batch_questions = questions[batch_start:batch_end]
                        for question in batch_questions:
                            prompts.append(template.format(question=question))
                    elif mode == OperationMode.ASSESS:
                        batch_questions = questions[batch_start:batch_end]
                        batch_terms = available_terms[batch_start:batch_end]
                        for question, terms in zip(batch_questions, batch_terms):
                            prompts.append(
                                template.format(question=question, available_terms=terms)
                            )
                    elif mode == OperationMode.TOPIC_RELEVANCE_PROB:
                        batch_questions = questions[batch_start:batch_end]
                        # topic_descriptors is a single string shared across all questions
                        for question in batch_questions:
                            prompts.append(
                                template.format(
                                    question=question,
                                    topic_descriptors=topic_descriptors,
                                )
                            )

                    try:
                        # Use LangChain's batch method for this chunk
                        responses = self.llm.batch(prompts)

                        # Parse each response in the batch
                        for i, response in enumerate(responses):
                            global_idx = batch_start + i
                            try:
                                if hasattr(response, "content"):
                                    content = response.content
                                else:
                                    content = str(response)

                                # Try fixing parser first
                                try:
                                    result = fixing_parser.parse(content)
                                    all_results.append(self._extract_result_dict(result))
                                    successful_parses += 1
                                except Exception:
                                    # Fallback to regular parser
                                    result = parser.parse(content)
                                    all_results.append(self._extract_result_dict(result))
                                    successful_parses += 1

                            except Exception as e:
                                LOGGER.error(
                                    f"Error parsing item {global_idx} in batch {batch_idx}: {e}"
                                )
                                all_results.append(None)
                                failed_parses += 1

                    except Exception as e:
                        LOGGER.error(
                            f"Batch {batch_idx}/{total_batches} failed: {e}. "
                            f"Adding None for {current_batch_size} items"
                        )
                        # Add None for all items in this failed batch
                        all_results.extend([None] * current_batch_size)
                        failed_parses += current_batch_size

            except KeyboardInterrupt:
                LOGGER.warning(
                    f"Batch processing interrupted by user. "
                    f"Processed {len(all_results)}/{total_items} items "
                    f"({successful_parses} successful, {failed_parses} failed)."
                )
                # Pad remaining items with None to maintain index alignment
                remaining = total_items - len(all_results)
                if remaining > 0:
                    LOGGER.info(f"Padding {remaining} unprocessed items with None")
                    all_results.extend([None] * remaining)
                    failed_parses += remaining

                LOGGER.info(
                    f"Returning partial results: {successful_parses} successful, "
                    f"{failed_parses} failed/unprocessed out of {total_items} total items"
                )
                return all_results

            except Exception as e:
                LOGGER.error(f"Critical error in batch processing: {e}")
                LOGGER.info("Falling back to sequential processing")

                # Fallback to sequential processing
                if mode == OperationMode.EXTRACT:
                    return [self.extract_domains(text) for text in text_sources]
                elif mode == OperationMode.GUESS:
                    return [self.guess_domains(q) for q in questions]
                elif mode == OperationMode.ASSESS:
                    return [self.assess_domains(q, t) for q, t in zip(questions, available_terms)]
                elif mode == OperationMode.TOPIC_RELEVANCE_PROB:
                    return [
                        self.assess_topic_relevance_prob(q, topic_descriptors) for q in questions
                    ]

            LOGGER.info(
                f"Batch processing complete: {successful_parses} successful, "
                f"{failed_parses} failed out of {total_items} total items"
            )
            return all_results

        def process_dataframe(
            self,
            df: pd.DataFrame,
            mode: OperationMode,
            text_source_col: Optional[str] = None,
            question_col: Optional[str] = None,
            available_terms: Optional[Union[List[str], List[Dict], str]] = None,
            topic_descriptors: Optional[Union[List[str], List[Dict], str, Dict]] = None,
            progress_bar: bool = True,
            save_path: Optional[str] = None,
            skip_existing: bool = True,
            checkpoint_batch_size: Optional[int] = None,
            use_batch_processing: bool = True,
            batch_size: Optional[int] = None,
        ) -> pd.DataFrame:
            """
            Process a DataFrame based on the specified operation mode.

            In EXTRACT mode, if 'source_id' column exists, the method will:
            - Group rows by source_id
            - Process each unique source only once (using text from first occurrence)
            - Apply the result to all rows with the same source_id
            This optimization reduces redundant processing when multiple rows reference the same source.

            Parameters
            ----------
            df : pd.DataFrame
                DataFrame to process
            mode : OperationMode
                Operation mode (EXTRACT, GUESS, ASSESS, or TOPIC_RELEVANCE_PROB)
            text_source_col : str, optional
                Column name for text sources (required for EXTRACT mode)
            question_col : str, optional
                Column name for questions (required for GUESS, ASSESS, and TOPIC_RELEVANCE_PROB modes)
            available_terms : Union[List[str], List[Dict], str], optional
                Available terms for ASSESS mode
            topic_descriptors : Union[List[str], List[Dict], str, Dict], optional
                Topic descriptors for TOPIC_RELEVANCE_PROB mode. Supports:
                - List of strings
                - JSON string
                - File path to KeywordClusterer JSON
                - KeywordClusterer dict with 'clusters' or 'cluster_stats'
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
            if (
                mode
                in [
                    OperationMode.GUESS,
                    OperationMode.ASSESS,
                    OperationMode.TOPIC_RELEVANCE_PROB,
                ]
                and not question_col
            ):
                raise ValueError(
                    "question_col required for GUESS, ASSESS, and TOPIC_RELEVANCE_PROB modes"
                )
            if mode == OperationMode.ASSESS and available_terms is None:
                raise ValueError("available_terms required for ASSESS mode")
            if mode == OperationMode.TOPIC_RELEVANCE_PROB and topic_descriptors is None:
                raise ValueError("topic_descriptors required for TOPIC_RELEVANCE_PROB mode")

            # Get column prefix to avoid overwrites
            prefix = self._get_column_prefix(mode)

            # Add result columns based on mode with prefix
            if mode in [OperationMode.EXTRACT, OperationMode.GUESS]:
                result_columns = [
                    f"{prefix}_suggestions",
                    f"{prefix}_total_suggestions",
                    (
                        f"{prefix}_primary_theme"
                        if mode == OperationMode.EXTRACT
                        else f"{prefix}_question_category"
                    ),
                    f"{prefix}_error",
                ]
            elif mode == OperationMode.ASSESS:
                result_columns = [
                    f"{prefix}_selected_terms",
                    f"{prefix}_total_selected",
                    f"{prefix}_question_intent",
                    f"{prefix}_primary_topics",
                    f"{prefix}_error",
                ]
            elif mode == OperationMode.TOPIC_RELEVANCE_PROB:
                # Special column name as per requirements: must be "question_term_relevance_scores"
                # This is different from other modes which use prefix-based naming
                result_columns = [
                    "question_term_relevance_scores",  # Required name from specification
                    f"{prefix}_topic_scores",
                    f"{prefix}_total_topics",
                    f"{prefix}_question_summary",
                    f"{prefix}_error",
                ]

            existing_cols = result_df.columns

            for col in result_columns:
                if col not in existing_cols:
                    result_df[col] = None

            # Check if we should use source_id grouping (EXTRACT mode only)
            use_source_grouping = mode == OperationMode.EXTRACT and "source_id" in existing_cols

            if use_source_grouping:
                LOGGER.info(
                    "source_id column detected in EXTRACT mode - using source grouping optimization"
                )
                return self._process_dataframe_with_source_grouping(
                    result_df,
                    text_source_col,
                    prefix,
                    skip_existing,
                    progress_bar,
                    save_path,
                    checkpoint_batch_size,
                    should_batch,
                    batch_size,
                )

            # Regular processing without source grouping
            # Collect rows to process
            rows_to_process = []
            indices_to_process = []

            for idx, row in result_df.iterrows():
                # Skip if has existing results (check mode-specific columns)
                if skip_existing:
                    if mode in [OperationMode.EXTRACT, OperationMode.GUESS]:
                        if pd.notna(row.get(f"{prefix}_total_suggestions")):
                            continue
                    elif mode == OperationMode.ASSESS:
                        if pd.notna(row.get(f"{prefix}_total_selected")):
                            continue
                    elif mode == OperationMode.TOPIC_RELEVANCE_PROB:
                        if pd.notna(row.get(f"{prefix}_total_topics")):
                            continue

                # Check for empty inputs
                if mode == OperationMode.EXTRACT:
                    if pd.isna(row[text_source_col]) or is_empty_text(row[text_source_col]):
                        result_df.at[idx, f"{prefix}_error"] = "Missing or empty text source"
                        continue
                else:
                    if pd.isna(row[question_col]) or is_empty_text(row[question_col]):
                        result_df.at[idx, f"{prefix}_error"] = "Missing or empty question"
                        continue

                rows_to_process.append(row)
                indices_to_process.append(idx)

            if not rows_to_process:
                LOGGER.info("No rows to process")
                return result_df

            LOGGER.info(f"Processing {len(rows_to_process)} rows with mode: {mode}")

            # Process based on batch mode
            if should_batch:
                result_df = self._process_dataframe_batch(
                    rows_to_process,
                    indices_to_process,
                    result_df,
                    mode,
                    prefix,
                    text_source_col,
                    question_col,
                    available_terms,
                    topic_descriptors,
                    batch_size,
                    progress_bar,
                    save_path,
                    checkpoint_batch_size,
                )
            else:
                result_df = self._process_dataframe_sequential(
                    rows_to_process,
                    indices_to_process,
                    result_df,
                    mode,
                    prefix,
                    text_source_col,
                    question_col,
                    available_terms,
                    topic_descriptors,
                    progress_bar,
                    save_path,
                    checkpoint_batch_size,
                )

            if save_path:
                result_df.to_csv(save_path, index=False)
                LOGGER.info(f"Final results saved to {save_path}")

            return result_df

        def _process_dataframe_batch(
            self,
            rows,
            indices,
            result_df,
            mode,
            prefix,
            text_col,
            question_col,
            available_terms,
            topic_descriptors,
            batch_size,
            progress_bar,
            save_path,
            checkpoint_size,
        ):
            """
            Process DataFrame using batch API.

            Note: Simplified since _batch_process() handles chunking and progress internally.
            """
            LOGGER.info(
                f"Using batch processing for {len(rows)} rows " f"(batch_size={batch_size})"
            )

            # Collect all inputs
            if mode == OperationMode.EXTRACT:
                all_inputs = [row[text_col] for row in rows]
            elif mode == OperationMode.GUESS:
                all_inputs = [row[question_col] for row in rows]
            elif mode == OperationMode.ASSESS:
                all_inputs = [row[question_col] for row in rows]
            elif mode == OperationMode.TOPIC_RELEVANCE_PROB:
                all_inputs = [row[question_col] for row in rows]

            try:
                # Call the appropriate batch method (which handles chunking and progress)
                if mode == OperationMode.EXTRACT:
                    batch_results = self.extract_domains_batch(
                        all_inputs, batch_size=batch_size, show_progress=progress_bar
                    )
                elif mode == OperationMode.GUESS:
                    batch_results = self.guess_domains_batch(
                        all_inputs, batch_size=batch_size, show_progress=progress_bar
                    )
                elif mode == OperationMode.ASSESS:
                    batch_results = self.assess_domains_batch(
                        all_inputs,
                        available_terms,
                        batch_size=batch_size,
                        show_progress=progress_bar,
                    )
                elif mode == OperationMode.TOPIC_RELEVANCE_PROB:
                    batch_results = self.assess_topic_relevance_prob_batch(
                        all_inputs,
                        topic_descriptors,
                        batch_size=batch_size,
                        show_progress=progress_bar,
                    )

                # Update DataFrame with results
                processed = 0
                errors = 0

                iterator = (
                    tqdm(
                        zip(indices, batch_results),
                        total=len(indices),
                        desc="Updating DataFrame",
                    )
                    if progress_bar
                    else zip(indices, batch_results)
                )

                for idx, result in iterator:
                    if result is not None:
                        if mode in [OperationMode.EXTRACT, OperationMode.GUESS]:
                            result_df.at[idx, f"{prefix}_suggestions"] = str(result["suggestions"])
                            result_df.at[idx, f"{prefix}_total_suggestions"] = result[
                                "total_suggestions"
                            ]

                            if mode == OperationMode.EXTRACT:
                                result_df.at[idx, f"{prefix}_primary_theme"] = result.get(
                                    "primary_theme"
                                )
                            else:  # GUESS
                                result_df.at[idx, f"{prefix}_question_category"] = result.get(
                                    "question_category"
                                )
                        elif mode == OperationMode.ASSESS:
                            result_df.at[idx, f"{prefix}_selected_terms"] = str(
                                result["selected_terms"]
                            )
                            result_df.at[idx, f"{prefix}_total_selected"] = result["total_selected"]
                            result_df.at[idx, f"{prefix}_question_intent"] = result[
                                "question_intent"
                            ]
                            result_df.at[idx, f"{prefix}_primary_topics"] = str(
                                result["primary_topics"]
                            )
                        elif mode == OperationMode.TOPIC_RELEVANCE_PROB:
                            # Create JSON mapping using helper method
                            result_df.at[idx, "question_term_relevance_scores"] = (
                                self._create_relevance_json_mapping(result["topic_scores"])
                            )
                            result_df.at[idx, f"{prefix}_topic_scores"] = str(
                                result["topic_scores"]
                            )
                            result_df.at[idx, f"{prefix}_total_topics"] = result["total_topics"]
                            # question_summary is optional
                            if result.get("question_summary"):
                                result_df.at[idx, f"{prefix}_question_summary"] = result[
                                    "question_summary"
                                ]
                        processed += 1
                    else:
                        result_df.at[idx, f"{prefix}_error"] = "Analysis failed"
                        errors += 1

                    # Checkpoint
                    if save_path and checkpoint_size and processed % checkpoint_size == 0:
                        result_df.to_csv(save_path, index=False)
                        LOGGER.info(f"Checkpoint saved at {processed} rows")

            except KeyboardInterrupt:
                LOGGER.warning("Processing interrupted by user")
                if save_path:
                    result_df.to_csv(save_path, index=False)
                    LOGGER.info(f"Progress saved to {save_path}")
            except Exception as e:
                LOGGER.error(f"Batch processing error: {e}")
                for idx in indices:
                    result_df.at[idx, f"{prefix}_error"] = f"Batch error: {str(e)}"
                errors = len(indices)

            LOGGER.info(f"Batch processing complete: {processed} successful, {errors} errors")
            return result_df

        def _process_dataframe_sequential(
            self,
            rows,
            indices,
            result_df,
            mode,
            prefix,
            text_col,
            question_col,
            available_terms,
            topic_descriptors,
            progress_bar,
            save_path,
            checkpoint_size,
        ):
            """Process DataFrame sequentially."""
            # Pre-parse data once to avoid repeated loading/parsing on each iteration
            parsed_data = {}
            if mode == OperationMode.TOPIC_RELEVANCE_PROB and topic_descriptors is not None:
                LOGGER.info("Pre-parsing topic descriptors to avoid repeated file loading")
                parsed_data['topic_descriptors'] = self._parse_topic_descriptors(topic_descriptors)

            iterator = (
                tqdm(zip(indices, rows), total=len(rows), desc="Processing rows")
                if progress_bar
                else zip(indices, rows)
            )

            processed = 0
            errors = 0

            for idx, row in iterator:
                try:
                    # Call appropriate method
                    if mode == OperationMode.EXTRACT:
                        result = self.extract_domains(row[text_col])
                    elif mode == OperationMode.GUESS:
                        result = self.guess_domains(row[question_col])
                    elif mode == OperationMode.ASSESS:
                        result = self.assess_domains(row[question_col], available_terms)
                    elif mode == OperationMode.TOPIC_RELEVANCE_PROB:
                        # Use pre-parsed descriptors to avoid repeated file loading
                        result = self.assess_topic_relevance_prob(
                            row[question_col], parsed_data['topic_descriptors']
                        )

                    if result is not None:
                        if mode in [OperationMode.EXTRACT, OperationMode.GUESS]:
                            result_df.at[idx, f"{prefix}_suggestions"] = str(result["suggestions"])
                            result_df.at[idx, f"{prefix}_total_suggestions"] = result[
                                "total_suggestions"
                            ]

                            if mode == OperationMode.EXTRACT:
                                result_df.at[idx, f"{prefix}_primary_theme"] = result.get(
                                    "primary_theme"
                                )
                            else:  # GUESS
                                result_df.at[idx, f"{prefix}_question_category"] = result.get(
                                    "question_category"
                                )
                        elif mode == OperationMode.ASSESS:
                            result_df.at[idx, f"{prefix}_selected_terms"] = str(
                                result["selected_terms"]
                            )
                            result_df.at[idx, f"{prefix}_total_selected"] = result["total_selected"]
                            result_df.at[idx, f"{prefix}_question_intent"] = result[
                                "question_intent"
                            ]
                            result_df.at[idx, f"{prefix}_primary_topics"] = str(
                                result["primary_topics"]
                            )
                        elif mode == OperationMode.TOPIC_RELEVANCE_PROB:
                            # Create JSON mapping using helper method
                            result_df.at[idx, "question_term_relevance_scores"] = (
                                self._create_relevance_json_mapping(result["topic_scores"])
                            )
                            result_df.at[idx, f"{prefix}_topic_scores"] = str(
                                result["topic_scores"]
                            )
                            result_df.at[idx, f"{prefix}_total_topics"] = result["total_topics"]
                            # question_summary is optional
                            if result.get("question_summary"):
                                result_df.at[idx, f"{prefix}_question_summary"] = result[
                                    "question_summary"
                                ]
                        processed += 1
                    else:
                        result_df.at[idx, f"{prefix}_error"] = "Analysis failed"
                        errors += 1

                    # Checkpoint
                    if save_path and checkpoint_size and processed % checkpoint_size == 0:
                        result_df.to_csv(save_path, index=False)
                        LOGGER.info(f"Checkpoint saved at {processed} rows")

                except KeyboardInterrupt:
                    LOGGER.warning("Processing interrupted by user")
                    if save_path:
                        result_df.to_csv(save_path, index=False)
                        LOGGER.info(f"Progress saved to {save_path}")
                    break
                except Exception as e:
                    LOGGER.error(f"Error processing row {idx}: {e}")
                    result_df.at[idx, f"{prefix}_error"] = str(e)
                    errors += 1

                if progress_bar and hasattr(iterator, "set_postfix"):
                    iterator.set_postfix({"Processed": processed, "Errors": errors})

            LOGGER.info(f"Sequential processing complete: {processed} successful, {errors} errors")
            return result_df

        def _process_dataframe_with_source_grouping(
            self,
            result_df,
            text_source_col,
            prefix,
            skip_existing,
            progress_bar,
            save_path,
            checkpoint_batch_size,
            should_batch,
            batch_size,
        ):
            """Process DataFrame by grouping rows with same source_id.

            Only processes each unique source once and applies results to all rows with that source_id.
            Handles interrupts gracefully at any stage.

            Parameters
            ----------
            result_df : pd.DataFrame
                DataFrame to process (will be modified in place)
            text_source_col : str
                Column name containing source text
            prefix : str
                Column prefix for results (e.g., 'extract')
            skip_existing : bool
                If True, skip sources that already have results
            progress_bar : bool
                Show progress bars
            save_path : str, optional
                Path to save checkpoints and final results
            checkpoint_batch_size : int, optional
                Save checkpoint every N rows
            should_batch : bool
                Use batch processing if True
            batch_size : int
                Batch size for processing

            Returns
            -------
            pd.DataFrame
                Processed DataFrame with results
            """
            LOGGER.info("Starting source grouping optimization for EXTRACT mode")

            # Statistics tracking
            stats = {
                "total_rows": 0,
                "skipped_rows": 0,
                "copied_rows": 0,
                "processed_rows": 0,
                "error_rows": 0,
                "interrupted": False,
            }

            # ============================================================
            # PHASE 1: Build source_id mapping
            # ============================================================
            try:
                LOGGER.info("Phase 1: Building source_id mapping...")

                source_id_mapping = {}  # source_id -> [(row_idx, has_results_bool), ...]

                iterator = (
                    tqdm(
                        result_df.iterrows(),
                        total=len(result_df),
                        desc="Mapping sources",
                    )
                    if progress_bar
                    else result_df.iterrows()
                )

                for idx, row in iterator:
                    source_id = row.get("source_id")

                    # Skip rows without source_id
                    if pd.isna(source_id):
                        result_df.at[idx, f"{prefix}_error"] = "Missing source_id"
                        stats["error_rows"] += 1
                        continue

                    # Skip rows with empty text
                    if pd.isna(row[text_source_col]) or is_empty_text(row[text_source_col]):
                        result_df.at[idx, f"{prefix}_error"] = "Missing or empty text source"
                        stats["error_rows"] += 1
                        continue

                    # Check if this row has results (boolean only)
                    has_results = pd.notna(row.get(f"{prefix}_total_suggestions"))

                    # Add to mapping
                    if source_id not in source_id_mapping:
                        source_id_mapping[source_id] = []
                    source_id_mapping[source_id].append((idx, has_results))
                    stats["total_rows"] += 1

                if not source_id_mapping:
                    LOGGER.info("No valid rows to process")
                    return result_df

                LOGGER.info(
                    f"Found {len(source_id_mapping)} unique sources across "
                    f"{stats['total_rows']} valid rows"
                )

            except KeyboardInterrupt:
                LOGGER.warning("Interrupted during Phase 1 (mapping)")
                stats["interrupted"] = True
                self._save_and_log_stats(result_df, save_path, stats)
                return result_df

            # ============================================================
            # PHASE 2: Categorize sources (skip/copy/process)
            # ============================================================
            try:
                LOGGER.info("Phase 2: Categorizing sources...")

                sources_to_skip = []  # source_ids with all rows already processed
                sources_to_copy = []  # source_ids with some rows processed
                sources_to_process = {}  # source_id -> first_row_idx (for getting text)

                for source_id, row_list in source_id_mapping.items():
                    has_results_list = [has_results for _, has_results in row_list]
                    any_has_results = any(has_results_list)
                    all_have_results = all(has_results_list)

                    if skip_existing and all_have_results:
                        # All rows for this source already have results
                        sources_to_skip.append(source_id)
                        stats["skipped_rows"] += len(row_list)

                    elif skip_existing and any_has_results:
                        # Some rows have results, need to copy to others
                        sources_to_copy.append(source_id)

                    else:
                        # Need to process this source
                        # Store the first row index to get the text from later
                        first_row_idx = row_list[0][0]
                        sources_to_process[source_id] = first_row_idx

                LOGGER.info(
                    f"Categorized: {len(sources_to_skip)} to skip, "
                    f"{len(sources_to_copy)} to copy, "
                    f"{len(sources_to_process)} to process"
                )

            except KeyboardInterrupt:
                LOGGER.warning("Interrupted during Phase 2 (categorization)")
                stats["interrupted"] = True
                self._save_and_log_stats(result_df, save_path, stats)
                return result_df

            # ============================================================
            # PHASE 3: Copy existing results to rows without them
            # ============================================================
            try:
                if sources_to_copy:
                    LOGGER.info(f"Phase 3: Copying results for {len(sources_to_copy)} sources...")

                    iterator = (
                        tqdm(sources_to_copy, desc="Copying results")
                        if progress_bar
                        else sources_to_copy
                    )

                    for source_id in iterator:
                        row_list = source_id_mapping[source_id]

                        # Find first row that has results
                        source_idx = None
                        for idx, has_results in row_list:
                            if has_results:
                                source_idx = idx
                                break

                        if source_idx is None:
                            LOGGER.warning(
                                f"No results found for source_id={source_id}, will process instead"
                            )
                            # Add to processing queue
                            first_idx = row_list[0][0]
                            sources_to_process[source_id] = first_idx
                            continue

                        # Copy results from source_idx row to all rows without results
                        source_row = result_df.loc[source_idx]

                        for idx, has_results in row_list:
                            if not has_results:
                                # Copy result columns
                                result_df.at[idx, f"{prefix}_suggestions"] = source_row[
                                    f"{prefix}_suggestions"
                                ]
                                result_df.at[idx, f"{prefix}_total_suggestions"] = source_row[
                                    f"{prefix}_total_suggestions"
                                ]
                                result_df.at[idx, f"{prefix}_primary_theme"] = source_row[
                                    f"{prefix}_primary_theme"
                                ]

                                # Clear any existing error
                                result_df.at[idx, f"{prefix}_error"] = None

                                stats["copied_rows"] += 1

                    LOGGER.info(f"Copied results to {stats['copied_rows']} rows")

                    # Save checkpoint after copying
                    if save_path:
                        result_df.to_csv(save_path, index=False)
                        LOGGER.info("Checkpoint saved after copying phase")

            except KeyboardInterrupt:
                LOGGER.warning(
                    f"Interrupted during Phase 3 (copying). "
                    f"Copied {stats['copied_rows']} rows so far."
                )
                stats["interrupted"] = True
                self._save_and_log_stats(result_df, save_path, stats)
                return result_df

            # Check if there's anything to process
            if not sources_to_process:
                LOGGER.info(
                    f"No sources to process. "
                    f"Skipped {len(sources_to_skip)} sources, "
                    f"copied results for {len(sources_to_copy)} sources."
                )
                self._save_and_log_stats(result_df, save_path, stats)
                return result_df

            # ============================================================
            # PHASE 4: Process new sources (batch or sequential)
            # ============================================================
            extraction_results = []  # Initialize to handle interrupts

            try:
                LOGGER.info(f"Phase 4: Processing {len(sources_to_process)} unique sources...")

                # Prepare inputs: maintain order for mapping back
                source_ids_ordered = list(sources_to_process.keys())
                texts_to_process = [
                    result_df.at[sources_to_process[sid], text_source_col]
                    for sid in source_ids_ordered
                ]

                # Process with batch or sequential
                if should_batch:
                    LOGGER.info(f"Using batch processing (batch_size={batch_size})")
                    extraction_results = self.extract_domains_batch(
                        texts_to_process,
                        batch_size=batch_size,
                        show_progress=progress_bar,
                    )
                    # Check if batch processing was interrupted
                    if any(r is None for r in extraction_results):
                        stats["interrupted"] = True
                else:
                    LOGGER.info("Using sequential processing")

                    iterator = (
                        tqdm(texts_to_process, desc="Extracting domains")
                        if progress_bar
                        else texts_to_process
                    )

                    for text in iterator:
                        try:
                            result = self.extract_domains(text)
                            extraction_results.append(result)
                        except KeyboardInterrupt:
                            LOGGER.warning(
                                f"Interrupted during sequential processing. "
                                f"Processed {len(extraction_results)}/{len(texts_to_process)} sources."
                            )
                            # Pad remaining with None to maintain alignment
                            remaining = len(texts_to_process) - len(extraction_results)
                            extraction_results.extend([None] * remaining)
                            stats["interrupted"] = True
                            break
                        except Exception as e:
                            LOGGER.error(f"Error extracting domains: {e}")
                            extraction_results.append(None)

                LOGGER.info(
                    f"Extraction complete. Got {len(extraction_results)} results "
                    f"({sum(1 for r in extraction_results if r is not None)} successful)."
                )

            except KeyboardInterrupt:
                LOGGER.warning("Interrupted during Phase 4 (processing)")
                stats["interrupted"] = True
                # Pad extraction_results if incomplete
                if len(extraction_results) < len(sources_to_process):
                    remaining = len(sources_to_process) - len(extraction_results)
                    extraction_results.extend([None] * remaining)

            # ============================================================
            # PHASE 5: Apply extraction results to all rows with same source_id
            # ============================================================
            try:
                LOGGER.info("Phase 5: Applying results to rows...")

                if not extraction_results:
                    LOGGER.warning("No extraction results to apply")
                else:
                    iterator = (
                        tqdm(
                            zip(source_ids_ordered, extraction_results),
                            total=len(source_ids_ordered),
                            desc="Applying results",
                        )
                        if progress_bar
                        else zip(source_ids_ordered, extraction_results)
                    )

                    rows_updated_since_checkpoint = 0

                    for source_id, result in iterator:
                        # Get all row indices for this source_id
                        row_indices = [idx for idx, _ in source_id_mapping[source_id]]

                        if result is not None:
                            # Apply successful result to all rows with this source_id
                            for idx in row_indices:
                                result_df.at[idx, f"{prefix}_suggestions"] = str(
                                    result["suggestions"]
                                )
                                result_df.at[idx, f"{prefix}_total_suggestions"] = result[
                                    "total_suggestions"
                                ]
                                result_df.at[idx, f"{prefix}_primary_theme"] = result.get(
                                    "primary_theme"
                                )

                                # Clear any existing error
                                result_df.at[idx, f"{prefix}_error"] = None

                                stats["processed_rows"] += 1
                                rows_updated_since_checkpoint += 1
                        else:
                            # Mark as failed (either processing failed or interrupted before this source)
                            error_msg = (
                                "Processing interrupted"
                                if stats["interrupted"]
                                else "Analysis failed"
                            )
                            for idx in row_indices:
                                # Only set error if row doesn't already have results
                                if pd.isna(result_df.at[idx, f"{prefix}_total_suggestions"]):
                                    result_df.at[idx, f"{prefix}_error"] = error_msg
                                    stats["error_rows"] += 1

                        # Checkpoint based on rows updated
                        if (
                            save_path
                            and checkpoint_batch_size
                            and rows_updated_since_checkpoint >= checkpoint_batch_size
                        ):
                            result_df.to_csv(save_path, index=False)
                            LOGGER.info(
                                f"Checkpoint saved: {stats['processed_rows']} rows processed"
                            )
                            rows_updated_since_checkpoint = 0

            except KeyboardInterrupt:
                LOGGER.warning(
                    f"Interrupted during Phase 5 (applying results). "
                    f"Applied results to {stats['processed_rows']} rows."
                )
                stats["interrupted"] = True

            # ============================================================
            # PHASE 6: Save and return
            # ============================================================
            self._save_and_log_stats(result_df, save_path, stats)
            return result_df

        def _save_and_log_stats(self, result_df, save_path, stats):
            """Save results and log statistics.

            Parameters
            ----------
            result_df : pd.DataFrame
                DataFrame to save
            save_path : str, optional
                Path to save results
            stats : dict
                Statistics dictionary with processing metrics
            """
            # Log summary
            status = "INTERRUPTED" if stats["interrupted"] else "COMPLETE"
            LOGGER.info("=" * 60)
            LOGGER.info(f"Source grouping processing {status}")
            LOGGER.info(f"  Total valid rows:     {stats['total_rows']}")
            LOGGER.info(f"  Skipped rows:         {stats['skipped_rows']}")
            LOGGER.info(f"  Copied rows:          {stats['copied_rows']}")
            LOGGER.info(f"  Processed rows:       {stats['processed_rows']}")
            LOGGER.info(f"  Error rows:           {stats['error_rows']}")
            LOGGER.info("=" * 60)

            # Save final results
            if save_path:
                result_df.to_csv(save_path, index=False)
                LOGGER.info(f"Results saved to {save_path}")

except ImportError as e:
    _DEPENDENCIES_AVAILABLE = False
    _IMPORT_ERROR = str(e)

    LOGGER.warning(
        f"DomainAnalysisAgent dependencies not available: {e}. "
        "Install with: pip install langchain langgraph pydantic pandas tqdm"
    )

    # Minimal stubs for type checking
    class OperationMode(str, Enum):
        """Operation modes for domain analysis."""

        EXTRACT = "extract"
        GUESS = "guess"
        ASSESS = "assess"
        TOPIC_RELEVANCE_PROB = "topic_relevance_prob"

    class DomainAnalysisAgent:
        """
        Placeholder for DomainAnalysisAgent when dependencies are missing.

        To use this agent, install required dependencies:
            pip install -r requirements_agents.txt
        """

        def __init__(self, *args, **kwargs):
            """Initialize placeholder that raises ImportError."""
            raise ImportError(
                f"DomainAnalysisAgent requires langgraph, langchain, and pydantic to be installed.\n"
                f"Original import error: {_IMPORT_ERROR}\n"
                f"Install with: pip install -r requirements_agents.txt"
            )

        def __getattr__(self, name):
            """Raise ImportError for any attribute access."""
            raise ImportError(
                f"DomainAnalysisAgent not available due to missing dependencies: {_IMPORT_ERROR}"
            )
