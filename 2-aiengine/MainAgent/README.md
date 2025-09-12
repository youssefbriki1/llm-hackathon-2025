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

## Project Layout (relevant parts)

```
2-aiengine/
├─ MainAgent/
│  └─ PyDFTAgent.py
├─ MCP-remotemanager/
│  ├─ remoterun/remote.py
│  └─ pyproject.toml
└─ OntoFlow/
   └─ agent/Onto_wa_rag/
      ├─ retriever_adapter.py
      ├─ jupyter_analysis/...
      └─ fortran_analysis/...
```

---

## Environment Setup (uv)

> Use a dedicated environment inside each component you run.

### 1) MCP RemoteManager
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

## Configure API Keys

Set your keys before running agents/tools that require them:
```bash
export OPENAI_API_KEY="sk-..."        # if you use OpenAI
export ANTHROPIC_API_KEY="sk-ant-..." # if you use Anthropic
```

(Windows PowerShell)
```powershell
$env:OPENAI_API_KEY="sk-..."
$env:ANTHROPIC_API_KEY="sk-ant-..."
```

---

## Run the MCP RemoteManager

From the MCP-remotemanager directory:

### Option A — With `mcpo` (recommended during development)
```bash
cd 2-aiengine/MCP-remotemanager
# Start MCP server on port 8000 and run the remote manager
uvx mcpo --port 8000 -- uv run remoterun/remote.py
```

### Option B — Direct (if you don’t need mcpo)
```bash
cd 2-aiengine/MCP-remotemanager
uv run remoterun/remote.py
# Server will bind to 127.0.0.1:8000 unless changed in the code/env
```

---

## Run the Main Agent

From the **2-aiengine** root so package imports resolve cleanly:

```bash
cd 2-aiengine
PYTHONPATH=$(pwd) uv run python -m MainAgent.PyDFTAgent
# or (without uv): PYTHONPATH=$(pwd) python -m MainAgent.PyDFTAgent
```

> If you created `__init__.py` files in `MainAgent/`, `OntoFlow/`, `OntoFlow/agent/`, and `OntoFlow/agent/Onto_wa_rag/`, you can also just run:
>
> ```bash
> cd 2-aiengine
> uv run python -m MainAgent.PyDFTAgent
> ```

---

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

## Quick Commands Reference

```bash
# MCP server (with mcpo)
cd 2-aiengine/MCP-remotemanager
uvx mcpo --port 8000 -- uv run remoterun/remote.py

# Main agent
cd 2-aiengine
PYTHONPATH=$(pwd) uv run python -m MainAgent.PyDFTAgent
```

---

## Troubleshooting

- **ImportError / ModuleNotFoundError**  
  - Run from `2-aiengine` using `python -m MainAgent.PyDFTAgent`, or  
  - Set `PYTHONPATH=$(pwd)` from `2-aiengine`, or  
  - Add empty `__init__.py` files in `MainAgent/`, `OntoFlow/`, `OntoFlow/agent/`, `OntoFlow/agent/Onto_wa_rag/`.

- **Port already in use (8000)**  
  - Change the port in your `mcpo` command or kill the conflicting process.

- **Model downloads fail**  
  - Ensure internet access for `sentence-transformers` on first run, or pre-cache models.

- **Tool not firing in the agent**  
  - Ensure `init_retriever(...)` ran before the agent starts.  
  - Prompt the agent to use the `notebook_retriever` tool explicitly in early tests.

---

## Notes

- Keep secrets out of source control. Prefer environment variables or a secrets manager.
- If you package this repo later, consider removing `sys.path` hacks and rely on proper package imports.
