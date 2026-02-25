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
