"""
Prompts for CSV question generation and rephrasing tasks.

These prompts are used by the CSVQuestionAgent to rephrase questions
based on source context and generate alternative questions from sources.
"""

QUESTION_REPHRASE_WITH_SOURCE_PROMPT = """
Task: You are an expert question rewriter. Your goal is to rephrase the given question to better align with the terminology, context, and domain of the provided source text while preserving the original question's meaning and intent.

Source Text:
{source}

Original Question:
{question}

Instructions:
1. **Analyze the Source:** Carefully read the source text to understand its domain, terminology, and context.
2. **Preserve Question Intent:** The rephrased question must ask for the same information as the original question.
3. **Contextualize to Source:** Use terminology, phrasing, and concepts from the source text where appropriate.
4. **Maintain Clarity:** Ensure the rephrased question is clear, specific, and answerable based on the source.
5. **Natural Language:** Use natural, fluent language that fits the source's domain.
6. **Keep Focus:** Don't add information not present in the original question or source.

Output only the rephrased question without any additional explanation or commentary.

Rephrased Question:
"""

ALTERNATIVE_QUESTIONS_GENERATION_PROMPT = """
Task: You are an expert question generator. Your goal is to create {n} diverse and relevant questions that can be answered using the information provided in the source text.

Source Text:
{source}

Instructions:
1. **Analyze the Source:** Carefully read and understand the source text, identifying key facts, concepts, and relationships.
2. **Generate Diverse Questions:** Create {n} different questions that cover various aspects of the source content.
3. **Ensure Answerability:** Each question must be answerable using information from the source text.
4. **Vary Question Types:** Include different types of questions (what, how, why, when, etc.) where appropriate.
5. **Use Source Terminology:** Use the terminology and phrasing found in the source text.
6. **Be Specific:** Questions should be specific and focused, not overly broad or vague.
7. **Natural Language:** Use clear, natural language appropriate for the domain.

Output exactly {n} questions, one per line, numbered from 1 to {n}. Do not include any additional explanation or commentary.

Questions:
"""

CONTEXTUAL_QUESTION_PROMPT = """
Task: You are an expert at adapting questions to specific domains. Rephrase the given question to better fit the context and terminology of the specified domain while maintaining the original intent.

Domain/Context: {domain}

Original Question:
{question}

Instructions:
1. **Understand the Domain:** Consider the terminology, concepts, and common phrasing used in the specified domain.
2. **Preserve Intent:** The rephrased question must ask for the same core information.
3. **Apply Domain Language:** Use terminology and phrasing typical of the specified domain.
4. **Maintain Clarity:** Ensure the question remains clear and specific.
5. **Natural Fit:** The rephrased question should sound natural within the domain context.

Output only the rephrased question without any additional explanation or commentary.

Rephrased Question:
"""
