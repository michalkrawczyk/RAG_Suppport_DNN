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
5. **Keep Length Similar:** The rephrased text should be roughly the same length as the original (within 20%).
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
