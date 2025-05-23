from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain_core.output_parsers import StrOutputParser

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

FINAL_VERDICT_PROMPT = """
Based on the following detailed analysis, provide a final verdict on which source is better suited to answer the question.
Respond strictly with "Source 1" or "Source 2" based on the overall evaluation of the sources and their scores.

Analysis:
{analysis}
"""