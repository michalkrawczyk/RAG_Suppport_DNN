from logging import Logger

LOGGER = Logger(__name__)

try:
    from typing import TypedDict, List

    from prompts_templates.rag_verifiers import (
        SRC_COMPARE_PROMPT_WITH_SCORES,
        FINAL_VERDICT_PROMPT,
    )

    from langgraph.graph import END, START, StateGraph
    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
        BaseMessage,
    )
    from tqdm import tqdm

    class CheckAgentState(TypedDict):
        """State object for the agent."""

        messages: List[BaseMessage]
        question: str
        source1: str
        source2: str
        analysis: str
        final_choice: int

    class DatasetCheckAgent:
        _executor = None

        def __init__(self, llm):
            self._llm = llm

            # TODO: Should have websearch tool?

            self._build_graph()

        def _build_graph(self):
            def source_check(state: CheckAgentState):
                """
                Check the sources for duplicates and other issues.
                """
                prompt = SRC_COMPARE_PROMPT_WITH_SCORES.format(
                    question=state["question"],
                    source1_content=state["source1"],
                    source2_content=state["source2"],
                )
                # Create message for LLM
                message = HumanMessage(content=prompt)

                # Get analysis from LLM
                response = self._llm.invoke([message])
                analysis = (
                    response.content if hasattr(response, "content") else str(response)
                )

                # Update state
                state["messages"].append(message)
                state["messages"].append(AIMessage(content=analysis))
                state["analysis"] = analysis

                return state

            def assign_label(state: CheckAgentState):
                """
                Assign a label to the sources based on the comparison.
                """
                # Create final verdict prompt
                verdict_prompt = FINAL_VERDICT_PROMPT.format(analysis=state["analysis"])
                verdict_message = HumanMessage(content=verdict_prompt)

                # Get final decision
                response = self._llm.invoke([verdict_message])
                verdict = (
                    response.content if hasattr(response, "content") else str(response)
                )
                verdict = verdict.lower()

                if verdict == "source 1":
                    label = 0
                elif verdict == "source 2":
                    label = 1
                else:
                    label = -1
                    LOGGER.warning(f"Failed to parse source choice from: {verdict}")

                # Update state
                state["messages"].append(verdict_message)
                state["messages"].append(AIMessage(content=verdict))
                state["final_choice"] = label

                return state

            workflow = StateGraph(CheckAgentState)
            workflow.add_node("source_check", source_check)
            workflow.add_node("assign_label", assign_label)

            workflow.add_edge(START, "source_check")
            workflow.add_edge("source_check", "assign_label")
            workflow.add_edge("assign_label", END)

            self._executor = workflow.compile()

        def compare_text_sources(
            self,
            question: str,
            source1: str,
            source2: str,
            return_analysis=False,
            return_messages=False,
        ):
            state: CheckAgentState = {
                "messages": [],
                "question": question,
                "source1": source1,
                "source2": source2,
                "analysis": "",
                "final_choice": -1,
            }
            result = {}
            analysis = None
            messages = []

            try:
                # Execute the workflow
                llm_result = self._executor.invoke(state)

                result["label"] = llm_result["final_choice"]

                if return_analysis:
                    analysis = llm_result["analysis"]

                if return_messages:
                    messages = llm_result["messages"]

            except Exception as e:
                LOGGER.error(f"Error in source comparison: {str(e)}")
                result["label"] = -1

            result["analysis"] = analysis
            result["messages"] = messages

            return result

        def process_dataframe(self, df, save_path=None, skip_labeled=True):
            """
            Process the dataframe to check for duplicates and other issues.
            """
            results = []

            for idx, row in tqdm(
                df.iterrows(), total=len(df), desc="Processing sources"
            ):
                question = row["question_text"]
                source1 = row["answer_text_1"]
                source2 = row["answer_text_2"]
                current_label = row.get("label", -1)
                try:

                    if current_label != -1 and skip_labeled:
                        results.append(current_label)
                        continue

                    new_label = self.compare_text_sources(question, source1, source2)[
                        "label"
                    ]

                    if current_label != -1:
                        if new_label == -1:
                            new_label = current_label  # Keep same label if not sure
                        elif new_label != current_label:
                            LOGGER.warning(
                                f"Label mismatch for question in row {idx}: {current_label} vs {new_label}"
                            )

                    results.append(new_label)
                except Exception as err:
                    LOGGER.error(f"Error processing row {idx}: {str(err)}")
                    results.append(current_label)  # keep current label

            df["label"] = results
            if save_path:
                df.to_csv(save_path, index=False)
                LOGGER.info(f"Results saved to {save_path}")

            return df

        def process_csv(self, csv_path, skip_labeled=True):
            """
            Perform dataset check on a CSV file (Overwriting the file).
            """
            import pandas as pd

            df = pd.read_csv(csv_path)
            self.process_dataframe(df, save_path=csv_path, skip_labeled=skip_labeled)

except ImportError as e:
    LOGGER.warning(
        f"ImportError: {str(e)}. Please ensure all dependencies are installed."
    )

    class DatasetCheckAgent:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "DatasetCheckAgent requires langgraph and langchain_core. Please install them."
            )

except Exception as e:
    LOGGER.error(f"Unexpected error: {str(e)}")
    raise e
