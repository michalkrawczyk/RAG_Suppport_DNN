from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain_core.output_parsers import StrOutputParser

from excluded_drafts.rag_verifiers import FINAL_VERDICT_PROMPT

SRC_COMPARE_PROMPT= """
Task: You are acting as a domain expert. Given a question and two sources ([SOURCE 1] and [SOURCE 2]), your task is to choose the source that is better suited to answer the question from your expert perspective, inferring the relevant domain from the question and the content of the sources themselves.

Question: {question}

Source 1:
{source1_content}

Source 2:
{source2_content}

Criteria for choosing a better source:
* **Relevance:** The source should be highly relevant to the question and directly address it.
* **Expertise/Authority:** The source should demonstrate expertise in the inferred domain and provide authoritative information.
* **Depth and Specificity:** The source should provide sufficient depth and specific details to answer the question comprehensively from an expert's point of view within the inferred domain.
* **Clarity and Conciseness:** The information in the source should be clear, well-organized, and easy to understand for an expert in the inferred domain.
* **Objectivity/Bias:** The source should present information in an unbiased and objective manner, avoiding strong opinions or promotional content within the inferred domain.

"""

SRC_COMPARE_PROMPT_WITH_SCORES = """
Task: You are acting as a domain expert. Given a question and two sources ([SOURCE 1] and [SOURCE 2]), your task is to analyze which of the source that is better suited to answer the question from your expert perspective, **inferring the relevant domain from the question and the content of the sources themselves.**

Question: {question}

Source 1:
{source1_content}

Source 2:
{source2_content}

Criteria for choosing a better source (rate each criterion on a scale of 1-5, where 5 is the best and 1 is the worst):

* **Relevance:** How relevant is the source to the question and how directly does it address it?
    * Source 1 Relevance Score (1-5):
    * Source 2 Relevance Score (1-5):

* **Expertise/Authority:** How well does the source demonstrate expertise in the **inferred domain** and provide authoritative information?
    * Source 1 Expertise Score (1-5):
    * Source 2 Expertise Score (1-5):

* **Depth and Specificity:** How much depth and specific details does the source provide to answer the question comprehensively from an expert's point of view **within the inferred domain**?
    * Source 1 Depth Score (1-5):
    * Source 2 Depth Score (1-5):

* **Clarity and Conciseness:** How clear, well-organized, and easy to understand is the information in the source for an expert in the **inferred domain**?
    * Source 1 Clarity Score (1-5):
    * Source 2 Clarity Score (1-5):

* **Objectivity/Bias:** How objectively and unbiasedly does the source present information, avoiding strong opinions or promotional content **within the inferred domain**?
    * Source 1 Objectivity Score (1-5):
    * Source 2 Objectivity Score (1-5):


"""

# FINAL_VERDICT_PROMPT = """
# Based on the following detailed analysis, provide a final verdict on which source is better suited to answer the question.
# Respond strictly with "Source 1" or "Source 2" based on the overall evaluation of the sources and their scores.
#
# Analysis:
# {analysis}
# """

FINAL_VERDICT_PROMPT = """
Based on the following detailed analysis, provide a final verdict on which source is better suited to answer the question.

**Decision Logic:**
- If both sources have an average score below 3.0 across all criteria, respond with "Both Insufficient"
- If one source clearly outperforms the other (higher total/average score), choose that source
- If scores are very close (within 0.5 average difference), choose the source with higher relevance score

**Response Options:**
Respond strictly with one of the following:
- "Source 1" - if Source 1 is better suited
- "Source 2" - if Source 2 is better suited  
- "Neither" - if both sources score poorly (average < 3.0) and neither adequately addresses the question

Analysis:
{analysis}
"""

CONTEXT_SUFFICIENCY_PROMPT = """
You are an expert evaluator tasked with assessing whether a given source contains sufficient context information to answer a specific question.

Please rate the sufficiency of the context on a scale of 0-5 based on the following criteria:

**Rating Scale:**
- **5 (Excellent)**: The context contains complete, detailed information that fully answers the question with high confidence
- **4 (Good)**: The context contains most of the necessary information to answer the question adequately
- **3 (Moderate)**: The context contains some relevant information but lacks important details for a complete answer
- **2 (Limited)**: The context contains minimal relevant information, only partially addressing the question
- **1 (Poor)**: The context contains very little relevant information, barely touching on the question topic
- **0 (Insufficient)**: The context contains no relevant information to answer the question

**Question:** {question}

**Context/Source:** {context}

**Instructions:**
1. Carefully read both the question and the provided context
2. Determine how well the context addresses all aspects of the question
3. Consider completeness, accuracy, and relevance of the information
4. Provide only a single number (0-5) as your response

**Rating:**
"""