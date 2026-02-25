# GitHub Copilot Instructions for RAG_Support_DNN

Read **[AGENTS.md](../AGENTS.md)** for project structure, guidelines index, and the module map tool.

## Non-Negotiable Rules

- **Never bypass LangChain** for LLM interaction — always use `BaseChatModel` abstractions.
- **Every agent requires a test file** with mocked LLM calls in `tests/test_<agent_name>.py`.
- **Agents must never crash** on LLM failures — graceful error handling is mandatory.