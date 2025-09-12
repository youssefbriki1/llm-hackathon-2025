# OntoFlow / MCP RemoteManager + MainAgent — Dev Guide

This README explains how to set up the environment with **uv**, run the **MCP RemoteManager** service, and launch the **MainAgent**. It also tracks what’s done and what remains.

---

## Prerequisites

- Python 3.10+  
- [`pipx`](https://pypa.github.io/pipx/) and [`uv`](https://github.com/astral-sh/uv)
- (Optional) `mcpo` (Model Context Protocol Orchestrator) if you’re exposing tools via MCP

### Install `uv` with `pipx`
```bash
pip install --user pipx
pipx ensurepath
pipx install uv
```

---


## Environment Setup (uv)

> Use a dedicated environment inside each component you run.

### 1)  RemoteManager
```bash
cd 2-aiengine/MCP-remotemanager
uv lock
uv sync
```

### 2) MainAgent (optional but recommended)
```bash
cd 2-aiengine
uv lock
uv sync
```

---


## Run the Main Agent

From the **2-aiengine** root so package imports resolve cleanly:

```bash
cd 2-aiengine
uv run python -m MainAgent.PyDFTAgent
```


## Using the Notebook Retriever (LangChain/LangGraph tool)

1) **Initialize** the retriever once at startup:
```python
from OntoFlow.agent.Onto_wa_rag.retriever_adapter import init_retriever
init_retriever("/absolute/path/to/notebook.ipynb")
```

2) **Pass the tool** to your agent:
```python
from OntoFlow.agent.Onto_wa_rag.retriever_adapter import retriever_tool
tools = [retriever_tool]  # now the agent can call `notebook_retriever`
```

> Make sure `2-aiengine` is on `PYTHONPATH` **or** run with `python -m` from the `2-aiengine` directory.

---

## Status Checklist

- [x] VLLM integration — _to re-check in context_
- [x] Orchestrator + Agent supervisor
- [x] Agent memory
- [x] LangChain tool from MCP

**TODO:**
- [ ] Further testing on the code validator (with LLM agent)
- [ ] RAG pipeline verification with agent (compile & run end-to-end)
- [ ] Test the supervisor with the new architecture
- [ ] Add **Query Transformation** tool (RAG)
- [ ] Add **Reranker** (RAG)
- [ ] Fix coder tool
- [ ] Fix RAG part
- [ ] Link everything together

---

## Troubleshooting

- **ImportError / ModuleNotFoundError**  
  - Run from `2-aiengine` using `python -m MainAgent.PyDFTAgent`, or  
  - Add empty `__init__.py` files in `MainAgent/`, `OntoFlow/`, `OntoFlow/agent/`, `OntoFlow/agent/Onto_wa_rag/`.

- **Tool not firing in the agent**  
  - Ensure `init_retriever(...)` ran before the agent starts.  
  - Prompt the agent to use the `notebook_retriever` tool explicitly in early tests.

Installer aider-install pour build_oci -> pip
requirements.txt est dans agent
fix comment on met les clés API 
Documentation 
ANTHROPIC_SMALL_API_KEY