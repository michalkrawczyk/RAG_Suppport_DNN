# Agents Overview (`RAG_supporters/agents/`)

## Available Agents

| Agent | Purpose |
|---|---|
| `QuestionAugmentationAgent` | Question generation/rephrasing with source context |
| `TextAugmentationAgent` | Text augmentation preserving semantic meaning |
| `DatasetCheckAgent` | Source comparison via LangGraph `StateGraph` |
| `DomainAnalysisAgent` | Domain extraction/guessing/assessment (3 modes) |
| `SourceEvaluationAgent` | 6-dimensional source quality scoring |

## Agent Rules

See [architecture.md](../coding_guidelines/architecture.md) and [agent_workflow.md](../coding_guidelines/agent_workflow.md).

## Guidelines — Read Only When Needed

| File | Read when |
|------|-----------|
| [coding_guidelines/setup_and_install.md](agents_notes/coding_guidelines/setup_and_install.md) | Installing, adding dependencies, running tests/quality tools |
| [coding_guidelines/architecture.md](agents_notes/coding_guidelines/architecture.md) | Writing or reviewing agent imports or LangChain abstractions |
| [coding_guidelines/agent_workflow.md](agents_notes/coding_guidelines/agent_workflow.md) | Adding or modifying an agent (includes cautions for complex agents) |
| [coding_guidelines/error_handling.md](agents_notes/coding_guidelines/error_handling.md) | Adding error paths or LLM failure handling |
| [coding_guidelines/testing.md](agents_notes/coding_guidelines/testing.md) | Writing agent tests or unsure about mocking patterns |