"""Cluster steering agent for LLM-driven subspace activation and question rephrasing.

This agent implements advanced LLM features for:
- Generating steering texts to activate specific clusters/subspaces
- Rephrasing questions to emphasize different topics/genres
- Resolving ambiguity in multi-cluster assignments
- Supporting multi-membership steering scenarios
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union

LOGGER = logging.getLogger(__name__)

try:
    from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
    from langchain_core.language_models import BaseChatModel
    from langchain_core.prompts import PromptTemplate
    from pydantic import BaseModel, Field

    from prompts_templates.cluster_steering import (
        AMBIGUITY_RESOLUTION_PROMPT,
        CLUSTER_ACTIVATION_PROMPT,
        MULTI_CLUSTER_REPHRASE_PROMPT,
        QUESTION_REPHRASE_PROMPT,
    )

    # Pydantic Models
    class SteeringTextResult(BaseModel):
        """Result for cluster activation steering text generation."""

        cluster_id: int = Field(..., description="Target cluster ID")
        steering_text: str = Field(
            ..., description="Generated steering text for cluster activation"
        )
        incorporated_descriptors: List[str] = Field(
            ..., description="Descriptors incorporated in the steering text"
        )
        confidence: float = Field(
            ..., ge=0.0, le=1.0, description="Confidence score for the generation"
        )

    class QuestionRephraseResult(BaseModel):
        """Result for question rephrasing toward a specific cluster."""

        original_question: str = Field(..., description="Original question text")
        target_cluster_id: int = Field(..., description="Target cluster ID")
        rephrased_question: str = Field(
            ..., description="Rephrased question emphasizing target cluster"
        )
        genre_shift: str = Field(
            ..., description="Description of genre/topic shift applied"
        )
        preserved_intent: str = Field(
            ..., description="Core information need that was preserved"
        )
        confidence: float = Field(
            ..., ge=0.0, le=1.0, description="Confidence score"
        )

    class QuestionVariation(BaseModel):
        """A single question variation for multi-cluster scenarios."""

        target_cluster_id: int = Field(..., description="Target cluster ID")
        rephrased_question: str = Field(
            ..., description="Question variation emphasizing this cluster"
        )
        emphasized_aspect: str = Field(
            ..., description="Aspect of the question that is emphasized"
        )
        incorporated_descriptors: List[str] = Field(
            ..., description="Descriptors incorporated in the variation"
        )

    class MultiClusterRephraseResult(BaseModel):
        """Result for multi-cluster question rephrasing."""

        original_question: str = Field(..., description="Original question text")
        variations: List[QuestionVariation] = Field(
            ..., description="Question variations for different clusters"
        )
        total_variations: int = Field(..., description="Total number of variations")

    class ClusterRelevance(BaseModel):
        """Relevance information for a cluster."""

        cluster_id: int = Field(..., description="Cluster ID")
        reason: str = Field(..., description="Reason for relevance")
        confidence: float = Field(
            ..., ge=0.0, le=1.0, description="Confidence score"
        )

    class AmbiguityResolutionResult(BaseModel):
        """Result for ambiguity resolution analysis."""

        question: str = Field(..., description="The analyzed question")
        is_ambiguous: bool = Field(
            ..., description="Whether the question is ambiguous across clusters"
        )
        primary_cluster: ClusterRelevance = Field(
            ..., description="Primary cluster information"
        )
        secondary_clusters: List[ClusterRelevance] = Field(
            ..., description="Secondary cluster information"
        )
        recommendation: str = Field(
            ..., description="Recommendation: 'single-domain' or 'multi-domain'"
        )
        explanation: str = Field(
            ..., description="Overall analysis of domain coverage"
        )

    class ClusterSteeringAgent:
        """LLM agent for cluster steering, rephrasing, and ambiguity resolution.

        This agent provides advanced LLM-driven features for:
        - Generating steering texts to activate specific clusters
        - Rephrasing questions for multi-membership scenarios
        - Resolving ambiguity in cluster assignments
        """

        def __init__(
            self,
            llm: BaseChatModel,
            max_retries: int = 3,
        ):
            """
            Initialize the cluster steering agent.

            Parameters
            ----------
            llm : BaseChatModel
                Language model to use
            max_retries : int
                Maximum retries for parsing errors
            """
            self.llm = llm
            self.max_retries = max_retries

            # Set up parsers
            self.steering_parser = PydanticOutputParser(
                pydantic_object=SteeringTextResult
            )
            self.rephrase_parser = PydanticOutputParser(
                pydantic_object=QuestionRephraseResult
            )
            self.multi_rephrase_parser = PydanticOutputParser(
                pydantic_object=MultiClusterRephraseResult
            )
            self.ambiguity_parser = PydanticOutputParser(
                pydantic_object=AmbiguityResolutionResult
            )

            # Set up fixing parsers
            self.steering_fixing_parser = OutputFixingParser.from_llm(
                parser=self.steering_parser, llm=self.llm
            )
            self.rephrase_fixing_parser = OutputFixingParser.from_llm(
                parser=self.rephrase_parser, llm=self.llm
            )
            self.multi_rephrase_fixing_parser = OutputFixingParser.from_llm(
                parser=self.multi_rephrase_parser, llm=self.llm
            )
            self.ambiguity_fixing_parser = OutputFixingParser.from_llm(
                parser=self.ambiguity_parser, llm=self.llm
            )

            # Create prompt templates
            self.steering_template = self._create_prompt_template(
                CLUSTER_ACTIVATION_PROMPT,
                ["question", "cluster_id", "cluster_descriptors"],
                self.steering_parser,
            )
            self.rephrase_template = self._create_prompt_template(
                QUESTION_REPHRASE_PROMPT,
                ["question", "cluster_id", "cluster_descriptors", "alternate_clusters"],
                self.rephrase_parser,
            )
            self.multi_rephrase_template = self._create_prompt_template(
                MULTI_CLUSTER_REPHRASE_PROMPT,
                ["question", "cluster_info", "num_variations"],
                self.multi_rephrase_parser,
            )
            self.ambiguity_template = self._create_prompt_template(
                AMBIGUITY_RESOLUTION_PROMPT,
                ["question", "cluster_assignments"],
                self.ambiguity_parser,
            )

        def _create_prompt_template(
            self,
            template: str,
            input_vars: List[str],
            parser: PydanticOutputParser,
        ) -> PromptTemplate:
            """Create a prompt template with format instructions."""
            return PromptTemplate(
                template=template,
                input_variables=input_vars,
                partial_variables={
                    "format_instructions": parser.get_format_instructions()
                },
            )

        def _invoke_llm_with_retry(
            self,
            prompt: str,
            parser: PydanticOutputParser,
            fixing_parser: OutputFixingParser,
        ) -> Optional[Dict[str, Any]]:
            """Invoke LLM with retry logic for parsing."""
            for attempt in range(self.max_retries):
                try:
                    response = self.llm.invoke(prompt)
                    content = (
                        response.content if hasattr(response, "content") else str(response)
                    )

                    LOGGER.debug(f"LLM response (attempt {attempt + 1}): {content[:200]}...")

                    # Try fixing parser first
                    try:
                        result = fixing_parser.parse(content)
                        if hasattr(result, "model_dump"):
                            return result.model_dump()
                        return result
                    except Exception as parse_error:
                        LOGGER.warning(
                            f"Fixing parser failed (attempt {attempt + 1}): {parse_error}"
                        )
                        # Try regular parser
                        result = parser.parse(content)
                        if hasattr(result, "model_dump"):
                            return result.model_dump()
                        return result

                except Exception as e:
                    LOGGER.error(f"Attempt {attempt + 1} failed: {e}")
                    if attempt == self.max_retries - 1:
                        LOGGER.error(f"All {self.max_retries} attempts failed")
                        return None

            return None

        def generate_steering_text(
            self,
            question: str,
            cluster_id: int,
            cluster_descriptors: List[str],
        ) -> Optional[Dict[str, Any]]:
            """
            Generate steering text to activate a specific cluster.

            Parameters
            ----------
            question : str
                Original question
            cluster_id : int
                Target cluster ID
            cluster_descriptors : List[str]
                List of descriptor keywords for the cluster

            Returns
            -------
            Optional[Dict[str, Any]]
                Steering text generation result with keys:
                - cluster_id: Target cluster
                - steering_text: Generated steering text
                - incorporated_descriptors: Descriptors used
                - confidence: Confidence score
            """
            descriptors_text = ", ".join(cluster_descriptors)
            prompt = self.steering_template.format(
                question=question,
                cluster_id=cluster_id,
                cluster_descriptors=descriptors_text,
            )

            result = self._invoke_llm_with_retry(
                prompt, self.steering_parser, self.steering_fixing_parser
            )

            if result:
                LOGGER.info(
                    f"Generated steering text for cluster {cluster_id}: "
                    f"{result['steering_text'][:100]}..."
                )
            else:
                LOGGER.error(f"Failed to generate steering text for cluster {cluster_id}")

            return result

        def rephrase_question(
            self,
            question: str,
            target_cluster_id: int,
            cluster_descriptors: List[str],
            alternate_clusters: Optional[List[int]] = None,
        ) -> Optional[Dict[str, Any]]:
            """
            Rephrase question to emphasize a specific cluster/topic.

            Parameters
            ----------
            question : str
                Original question
            target_cluster_id : int
                Target cluster ID to emphasize
            cluster_descriptors : List[str]
                Descriptors for the target cluster
            alternate_clusters : Optional[List[int]]
                Other candidate cluster IDs for context

            Returns
            -------
            Optional[Dict[str, Any]]
                Rephrasing result with keys:
                - original_question: Original text
                - target_cluster_id: Target cluster
                - rephrased_question: Rephrased version
                - genre_shift: Description of shift
                - preserved_intent: Preserved core intent
                - confidence: Confidence score
            """
            descriptors_text = ", ".join(cluster_descriptors)
            alternates_text = (
                ", ".join(map(str, alternate_clusters)) if alternate_clusters else "None"
            )

            prompt = self.rephrase_template.format(
                question=question,
                cluster_id=target_cluster_id,
                cluster_descriptors=descriptors_text,
                alternate_clusters=alternates_text,
            )

            result = self._invoke_llm_with_retry(
                prompt, self.rephrase_parser, self.rephrase_fixing_parser
            )

            if result:
                LOGGER.info(
                    f"Rephrased question for cluster {target_cluster_id}: "
                    f"{result['rephrased_question'][:100]}..."
                )
            else:
                LOGGER.error(
                    f"Failed to rephrase question for cluster {target_cluster_id}"
                )

            return result

        def generate_multi_cluster_variations(
            self,
            question: str,
            cluster_info: Dict[int, List[str]],
            num_variations: Optional[int] = None,
        ) -> Optional[Dict[str, Any]]:
            """
            Generate question variations for multiple clusters.

            Parameters
            ----------
            question : str
                Original question
            cluster_info : Dict[int, List[str]]
                Dictionary mapping cluster IDs to their descriptors
            num_variations : Optional[int]
                Number of variations to generate (defaults to number of clusters)

            Returns
            -------
            Optional[Dict[str, Any]]
                Multi-cluster rephrasing result with keys:
                - original_question: Original text
                - variations: List of question variations
                - total_variations: Total count
            """
            if num_variations is None:
                num_variations = len(cluster_info)

            # Format cluster info as JSON string
            cluster_info_formatted = json.dumps(
                {str(cid): descs for cid, descs in cluster_info.items()}, indent=2
            )

            prompt = self.multi_rephrase_template.format(
                question=question,
                cluster_info=cluster_info_formatted,
                num_variations=num_variations,
            )

            result = self._invoke_llm_with_retry(
                prompt, self.multi_rephrase_parser, self.multi_rephrase_fixing_parser
            )

            if result:
                LOGGER.info(
                    f"Generated {len(result['variations'])} question variations"
                )
            else:
                LOGGER.error("Failed to generate multi-cluster variations")

            return result

        def resolve_ambiguity(
            self,
            question: str,
            cluster_assignments: Dict[int, float],
        ) -> Optional[Dict[str, Any]]:
            """
            Analyze ambiguity in cluster assignments and recommend resolution.

            Parameters
            ----------
            question : str
                Question to analyze
            cluster_assignments : Dict[int, float]
                Dictionary mapping cluster IDs to probability scores

            Returns
            -------
            Optional[Dict[str, Any]]
                Ambiguity resolution result with keys:
                - question: The analyzed question
                - is_ambiguous: Whether ambiguous
                - primary_cluster: Primary cluster info
                - secondary_clusters: Secondary clusters info
                - recommendation: 'single-domain' or 'multi-domain'
                - explanation: Overall analysis
            """
            # Format cluster assignments
            assignments_text = json.dumps(cluster_assignments, indent=2)

            prompt = self.ambiguity_template.format(
                question=question, cluster_assignments=assignments_text
            )

            result = self._invoke_llm_with_retry(
                prompt, self.ambiguity_parser, self.ambiguity_fixing_parser
            )

            if result:
                LOGGER.info(
                    f"Ambiguity analysis: is_ambiguous={result['is_ambiguous']}, "
                    f"recommendation={result['recommendation']}"
                )
            else:
                LOGGER.error("Failed to resolve ambiguity")

            return result

except ImportError as e:
    _IMPORT_ERROR = str(e)
    LOGGER.warning(
        f"ClusterSteeringAgent dependencies not available: {e}. "
        "Install with: pip install langchain pydantic"
    )

    # Stub for missing dependencies
    class ClusterSteeringAgent:
        """Placeholder when dependencies are missing."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                f"ClusterSteeringAgent requires langchain and pydantic.\n"
                f"Original import error: {_IMPORT_ERROR}\n"
                f"Install with: pip install -r requirements_agents.txt"
            )
