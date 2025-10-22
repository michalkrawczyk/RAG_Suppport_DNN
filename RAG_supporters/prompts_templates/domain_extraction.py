""" Prompt templates for domain extraction and analysis tasks. Used for training Knowledge Subspace Classifiers."""

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