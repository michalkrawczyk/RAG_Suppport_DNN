"""Prompt templates for domain extraction and analysis tasks. Used for training Knowledge Subspace Classifiers."""

SRC_DOMAIN_EXTRACTION_PROMPT = """Given the following text source, analyze its main topics, themes, and key concepts. Then propose up to 10 relevant domains, subdomains, or keywords that would be suitable for categorizing or tagging this content.

TEXT SOURCE:
{text_source}

Requirements:
- Identify the primary subject matter and key themes
- Suggest specific, relevant domains/subdomains/keywords
- Prioritize accuracy and relevance over quantity
- Include a confidence score (0-1) for each suggestion
- Provide a brief reason for each suggestion

Output the results in the following JSON format:
{{
  "suggestions": [
    {{
      "term": "example-domain",
      "type": "domain|subdomain|keyword",
      "confidence": 0.95,
      "reason": "Brief explanation of why this term is relevant"
    }}
  ],
  "total_suggestions": 5,
  "primary_theme": "Main identified theme"
}}

Provide only the JSON output, no additional text."""


QUESTION_DOMAIN_GUESS_PROMPT = """Given the following question, analyze what domains, subdomains, or keywords would be most relevant to properly answer or address it. Propose up to 10 terms that indicate the expertise areas, knowledge domains, or topics needed.

QUESTION:
{question}

Requirements:
- Identify what expertise/knowledge domains are needed to answer this question
- Suggest relevant domains, subdomains, and specific keywords
- Consider both explicit and implicit topics in the question
- Include a confidence score (0-1) for each suggestion
- Provide a brief reason for each suggestion

Output the results in the following JSON format:
{{
  "suggestions": [
    {{
      "term": "example-domain",
      "type": "domain|subdomain|keyword",
      "confidence": 0.95,
      "reason": "Brief explanation of relevance"
    }}
  ],
  "total_suggestions": 5,
  "question_category": "Identified question type/category"
}}

Provide only the JSON output, no additional text."""


QUESTION_DOMAIN_ASSESS_PROMPT = """Given a user question and a list of available keywords/domains/subdomains, analyze the question's intent and topic, then select up to 10 most relevant terms from the provided list that best match the question's context.

QUESTION:
{question}

AVAILABLE TERMS:
{available_terms}

Requirements:
- Analyze the question's main topic, intent, and context
- Rank the available terms by relevance to the question
- Select up to 10 most relevant terms
- Include a relevance score (0-1) for each selected term
- Provide a brief explanation for why each term is relevant
- Only select terms that have meaningful connection to the question

Output the results in the following JSON format:
{{
  "selected_terms": [
    {{
      "term": "example-term",
      "type": "domain|subdomain|keyword",
      "relevance_score": 0.92,
      "reason": "Brief explanation of relevance to the question"
    }}
  ],
  "total_selected": 5,
  "question_intent": "Brief description of what the question is asking about",
  "primary_topics": ["topic1", "topic2"]
}}

Provide only the JSON output, no additional text."""


def QUESTION_TOPIC_RELEVANCE_PROB_PROMPT(include_reason: bool = False) -> str:
    """Generate prompt for assessing topic relevance probabilities.

    Parameters
    ----------
    include_reason : bool, optional
        If True, includes 'reason' field in each topic assessment. Default is False.

    Returns
    -------
    str
        The formatted prompt template string
    """
    reason_field = (
        ', "reason": "Brief explanation for this probability"' if include_reason else ""
    )

    return f"""Given a user question and a list of topic descriptors, assess the probability of semantic connection between the question and each topic descriptor. Each probability should indicate how likely the question belongs to or is related to that topic.

NOTE: For large numbers of topic descriptors (>50), consider batching to stay within context limits.

QUESTION:
{{question}}

TOPIC DESCRIPTORS:
{{topic_descriptors}}

Requirements:
- Analyze the question's semantic content in relation to each topic descriptor
- Consider that some questions may be ambiguous or context-dependent (e.g., "What about PR" has different meanings in marketing vs IT)
- Use the topic descriptors themselves as context to interpret the question's potential meaning
- For each topic descriptor, determine the probability (0-1) that the question is semantically connected to it
- A probability of 1.0 means the question is highly relevant to the topic
- A probability of 0.0 means no semantic connection
- Provide probabilities for ALL topic descriptors provided
- Base probabilities on semantic similarity, topic matching, and contextual relevance (and provide reasoning for each assessment when requested)

Output the results in the following JSON format:
{{{{
  "topic_scores": [
    {{{{
      "topic_descriptor": "example-descriptor",
      "probability": 0.85{reason_field}
    }}}}
  ],
  "total_topics": 5,
  "question_summary": "Brief summary of the question's main topic (optional)"
}}}}

Provide only the JSON output, no additional text."""
