"""
Prompts for text augmentation and rephrasing tasks.

These prompts are used by the TextAugmentationAgent to generate alternative
versions of questions and sources while preserving their original meaning.
"""

FULL_TEXT_REPHRASE_PROMPT = """
Task: You are a text rewriter. Your goal is to rephrase the given text while preserving its exact meaning, intent, and all key information.

Original Text:
{text}

Instructions:
1. **Preserve Meaning:** The rephrased text must convey the exact same meaning as the original. Do not add, remove, or alter any information.
2. **Maintain Intent:** Keep the same intent, tone, and purpose of the original text.
3. **Natural Language:** Use natural, fluent language that sounds human-written.
4. **Vary Expression:** Use different words, sentence structures, and phrasing while maintaining the same meaning.
5. **Keep Length Similar:** The rephrased text should be roughly the same length as the original, but may be one or two sentences longer if needed for clarity.
6. **Preserve Technical Terms:** Keep domain-specific terminology, proper nouns, and technical terms unchanged or use appropriate synonyms only when they are truly equivalent.

Output only the rephrased text without any additional explanation or commentary.

Rephrased Text:
"""

SENTENCE_REPHRASE_PROMPT = """
Task: You are a text rewriter. Your goal is to rephrase a specific sentence within a text while keeping the rest of the text unchanged and preserving the sentence's exact meaning.

Original Text:
{text}

Sentence to Rephrase:
{sentence}

Instructions:
1. **Identify the Sentence:** Locate the specified sentence in the original text.
2. **Rephrase Only This Sentence:** Modify only this sentence while keeping all other parts of the text exactly the same.
3. **Preserve Meaning:** The rephrased sentence must convey the exact same meaning as the original sentence.
4. **Maintain Context:** Ensure the rephrased sentence fits naturally within the surrounding text.
5. **Natural Language:** Use natural, fluent language that sounds human-written.
6. **Keep Technical Terms:** Preserve domain-specific terminology and proper nouns unless there's a truly equivalent synonym.

Output only the complete text with the one sentence rephrased. Keep all other sentences unchanged.

Modified Text:
"""

VERIFY_MEANING_PRESERVATION_PROMPT = """
Task: You are a semantic analyzer. Your goal is to verify whether two texts have the same meaning.

Original Text:
{original_text}

Rephrased Text:
{rephrased_text}

Instructions:
1. **Compare Meanings:** Carefully compare both texts to determine if they convey the same information and intent.
2. **Check Information:** Verify that no information has been added, removed, or significantly altered.
3. **Assess Equivalence:** Determine if someone reading both texts would understand the same facts and concepts.

Output only one word: "EQUIVALENT" if the texts have the same meaning, or "DIFFERENT" if they differ in meaning.

Answer:
"""

# Question augmentation prompts

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
4. **Language Precision:** {clarity_instruction}
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
6. **Language Precision:** {clarity_instruction}
7. **Natural Language:** Use clear, natural language appropriate for the domain.

Output your response as a valid JSON object with a single key "questions" containing an array of exactly {n} question strings.
Do not include any additional explanation or commentary.

Example format:
{{"questions": ["Question 1?", "Question 2?", "Question 3?"]}}

Response:
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
4. **Language Precision:** {clarity_instruction}
5. **Natural Fit:** The rephrased question should sound natural within the domain context.

Output only the rephrased question without any additional explanation or commentary.

Rephrased Question:
"""
