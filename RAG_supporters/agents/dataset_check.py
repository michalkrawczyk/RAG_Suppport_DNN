import logging

LOGGER = logging.getLogger(__name__)

try:
    from typing import List, TypedDict

    from langchain_core.messages import (AIMessage, BaseMessage, HumanMessage,
                                         SystemMessage)
    from langgraph.graph import END, START, StateGraph
    from tqdm import tqdm

    from prompts_templates.rag_verifiers import (
        FINAL_VERDICT_PROMPT, SRC_COMPARE_PROMPT_WITH_SCORES)

    class CheckAgentState(TypedDict):
        """State object for the agent.

        Attributes
        ----------
        messages : List[BaseMessage]
            List of messages exchanged during the checking process.
        question : str
            The question being analyzed.
        source1 : str
            First source text to compare.
        source2 : str
            Second source text to compare.
        analysis : str
            Analysis result from source comparison.
        final_choice : int
            Final choice label (0 for source1, 1 for source2, -1 for error).
        """

        messages: List[BaseMessage]
        question: str
        source1: str
        source2: str
        analysis: str
        final_choice: int

    class DatasetCheckAgent:
        """Agent for checking and comparing text sources in datasets.

        This agent uses LLM-based analysis to compare two text sources for a given
        question and determine which source is better or if they are duplicates.

        Parameters
        ----------
        llm : object
            Language model instance for performing text analysis.

        Attributes
        ----------
        _llm : object
            The language model used for analysis.
        _executor : object
            Compiled workflow executor for the checking process.
        """

        _executor = None

        def __init__(self, llm, compare_prompt: str = SRC_COMPARE_PROMPT_WITH_SCORES):
            self._llm = llm
            self.compare_prompt = compare_prompt

            # TODO: Should have websearch tool?

            self._build_graph()

        def _build_graph(self):
            """Build the workflow graph for source checking process."""

            def source_check(state: CheckAgentState):
                """Check the sources for duplicates and other issues.

                Parameters
                ----------
                state : CheckAgentState
                    Current state containing question and sources to compare.

                Returns
                -------
                CheckAgentState
                    Updated state with analysis results.
                """
                prompt = self.compare_prompt.format(
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
                """Assign a label to the sources based on the comparison.

                Parameters
                ----------
                state : CheckAgentState
                    Current state containing analysis results.

                Returns
                -------
                CheckAgentState
                    Updated state with final choice label.
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
            """Compare two text sources for a given question.

            Parameters
            ----------
            question : str
                The question to analyze the sources against.
            source1 : str
                First source text to compare.
            source2 : str
                Second source text to compare.
            return_analysis : bool, optional
                Whether to return the analysis text, by default False.
            return_messages : bool, optional
                Whether to return the message history, by default False.

            Returns
            -------
            dict
                Dictionary containing:
                - 'label': int (0 for source1, 1 for source2, -1 for error)
                - 'analysis': str or None (if return_analysis=True)
                - 'messages': list or empty list (if return_messages=True)
            """
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

        def process_dataframe(self, df, save_path=None, skip_labeled=True, start_index=0):
            """Process the dataframe to check for duplicates and other issues.

            Parameters
            ----------
            df : pandas.DataFrame
                DataFrame containing columns: 'question_text', 'answer_text_1',
                'answer_text_2', and optionally 'label'.
            save_path : str, optional
                Path to save the processed DataFrame as CSV, by default None.
            skip_labeled : bool, optional
                Whether to skip rows that already have labels (!= -1), by default True.
            start_index : int, optional
                Index to start processing from, by default 0.

            Returns
            -------
            pandas.DataFrame
                DataFrame with updated 'label' column containing comparison results.
            """
            results = []
            start_index = max(0, start_index)
            interrupted = False

            if start_index >= len(df):
                LOGGER.warning(f"start_index ({start_index}) is beyond DataFrame length ({len(df)}). Processing aborted")
                return df

            sub_df = df.iloc[start_index:]

            for idx, row in tqdm(
                sub_df.iterrows(), total=len(sub_df), desc="Processing sources"
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
                except KeyboardInterrupt:
                    # Update processed partial results
                    LOGGER.info(
                        f"Process interrupted by user. Saving partial results for {len(results)} processed rows.")
                    interrupted = True
                    break

                except Exception as err:
                    LOGGER.error(f"Error processing row {idx}: {str(err)}")
                    results.append(current_label)  # keep current label

            if len(results) > 0:
                end_index = start_index + len(results)
                df.iloc[start_index:end_index, df.columns.get_loc("label")] = results

                if interrupted:
                    LOGGER.info(
                        f"Processed {len(results)} rows before interruption (rows {start_index} to {end_index - 1})")

            if save_path:
                df.to_csv(save_path, index=False)
                LOGGER.info(f"Results saved to {save_path}")

            return df

        def process_csv(self, csv_path, skip_labeled=True, start_index=0):
            """Perform dataset check on a CSV file (Overwriting the file).

            Parameters
            ----------
            csv_path : str
                Path to the CSV file to process. File will be overwritten with results.
            skip_labeled : bool, optional
                Whether to skip rows that already have labels (!= -1), by default True.
            start_index : int, optional
                Index to start processing from, by default 0.
            """
            import pandas as pd

            df = pd.read_csv(csv_path)
            self.process_dataframe(df, save_path=csv_path, skip_labeled=skip_labeled, start_index=start_index)

except ImportError as e:
    LOGGER.warning(
        f"ImportError: {str(e)}. Please ensure all dependencies are installed."
    )

    class DatasetCheckAgent:
        """Fallback class when dependencies are not available."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "DatasetCheckAgent requires langgraph and langchain_core. Please install them."
            )

except Exception as e:
    LOGGER.error(f"Unexpected error: {str(e)}")
    raise e
