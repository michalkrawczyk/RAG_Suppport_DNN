SUB_TEXT_SPLIT_PROMPT = """
Task: You are a text extractor. Given a question and a relevant text source, your goal is to extract two short text snippets from the source.

Question: [QUESTION]

Source Text:
[SOURCE TEXT]

Instructions:

1. **Identify Relevant Section:** Read the question and the source text carefully. Locate a section within the source text that directly answers the question or contains the most important information needed to answer the question. This section should be concise and focused on the question.

2. **Extract Relevant Text (Text 1):**  Extract this relevant section as **Text 1**. Text 1 should be self-contained and provide enough information for someone to understand the answer to the question, based on the source text. Aim for conciseness, but ensure it's informative enough to be useful.

3. **Identify Less Relevant (or Irrelevant) Section:** Now, find another section within the *same* source text that is either:
    * **Less Relevant:**  Related to the general topic of the source text, but not directly related to the question. It might provide background information, tangential details, or discuss related but different aspects.
    * **Irrelevant (if possible):**  A section that is as unrelated to the question as possible while still being part of the same source text. This could be a section that discusses a different subtopic, provides an example that is not directly applicable to the question, or is simply less focused on the core issue of the question.

4. **Extract Less Relevant/Irrelevant Text (Text 2):** Extract this less relevant or irrelevant section as **Text 2**. Text 2 should be noticeably less helpful (or unhelpful) in answering the question compared to Text 1. Aim for a similar length to Text 1, if possible, for comparison.

5. **Format Output:** Present the extracted texts clearly labeled as "Text 1 (Relevant)" and "Text 2 (Less Relevant/Irrelevant)".

Output:

**Text 1 (Relevant):**
[EXTRACTED RELEVANT TEXT]

**Text 2 (Less Relevant/Irrelevant):**
[EXTRACTED LESS RELEVANT OR IRRELEVANT TEXT]
"""

QUESTIONS_FROM_2_SOURCES_PROMPT = """
Task: You are acting as a domain expert. **Infer the relevant domain from Source 1.** Given two text sources, Source 1 and Source 2, your task is to generate questions that Source 1 can answer accurately and completely, but Source 2 cannot answer due to a lack of necessary information.  Act as an expert in the domain **inferred from Source 1**.

Source 1:
[SOURCE TEXT 1]

Source 2:
[SOURCE TEXT 2]

Instructions:

1. **Infer Domain from Source 1:** Carefully read Source 1 and **determine the primary domain or subject area** it covers.  Base your understanding of expertise and relevance on this inferred domain.

2. **Understand Both Sources within Inferred Domain:** Read both Source 1 and Source 2 to understand their content, scope, and level of detail **within the domain you inferred from Source 1.**

3. **Identify Information Gaps in Source 2 (Relative to Source 1):** Analyze Source 2 to identify areas where it lacks specific information, details, or context compared to Source 1, **within the inferred domain**. Consider what questions Source 2 would be unable to answer comprehensively or accurately due to these gaps in this domain.

4. **Generate Questions Answerable by Source 1 but NOT Source 2:**  Formulate questions that:
    *   Are clearly and directly answerable using information present in Source 1.
    *   Require specific details, data, or explanations that are present in Source 1 but demonstrably absent or insufficient in Source 2.
    *   Are relevant and meaningful from the perspective of an expert in the **domain inferred from Source 1**.
    *   Do not rely on making up facts or information not present in either source. Base your questions on the *difference* in information content between the two sources.

5. **Focus on Lack of Information (Not Factual Errors):** Ensure that Source 2's inability to answer is due to a *lack of information* on the specific point, not because Source 2 presents factually incorrect information. Both sources should be assumed to be factually sound within their presented scope, but Source 2 is simply less comprehensive for the specific questions you generate **within the inferred domain**.

6. **Generate [NUMBER] Questions:** Create a list of [NUMBER] such questions.

Output Questions (list each question on a new line):"""
