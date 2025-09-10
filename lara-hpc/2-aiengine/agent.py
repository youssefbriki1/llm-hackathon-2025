from typing import Any, Dict, Optional
import os, requests
from pydantic import BaseModel, Field
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
import httpx
MCP_URL = os.environ.get("MCP_URL", "http://127.0.0.1:8000")

class RunCodeInput(BaseModel):
    """Execute a Python function remotely via MCP RemoteRun."""
    function_source: str = Field(..., description="Full Python function source (def ...).")
    hostname: str = Field("localhost", description="Remote host to run the function on.")
    function_args: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Keyword args passed to your function."
    )

@tool("remote_run_code", args_schema=RunCodeInput)
def remote_run_code(function_source: str, hostname: str = "localhost",
                    function_args: Optional[Dict[str, Any]] = None) -> Any:
    """POST /run_code on the MCP proxy. Returns the tool result or raises on error."""
    url = f"{MCP_URL}/run_code"
    payload = {
        "function_source": function_source,
        "hostname": hostname,
        "function_args": function_args or {},
    }
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    if "Error" in data:
        raise RuntimeError(f"Remote error: {data['Error']}")
    return data.get("Result", data)

class RunCodeInput(BaseModel):
    """Execute a Python function remotely via MCP RemoteRun."""
    function_source: str = Field(..., description="Full Python function source (def ...).")
    hostname: str = Field("localhost", description="Remote host to run the function on.")
    function_args: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Keyword args passed to your function."
    )

@tool("remote_run_code_async", args_schema=RunCodeInput, return_direct=True)
async def remote_run_code_async(
    function_source: str,
    hostname: str = "localhost",
    function_args: Optional[Dict[str, Any]] = None
) -> Any:
    """Async POST /run_code on the MCP proxy. Returns result or raises on error."""
    url = f"{MCP_URL}/run_code"
    payload = {
        "function_source": function_source,
        "hostname": hostname,
        "function_args": function_args or {},
    }
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
    if "Error" in data:
        raise RuntimeError(f"Remote error: {data['Error']}")
    return data.get("Result", data)

@tool("rag_tool")




if __name__ == "__main__":
    llm = ChatOpenAI(model="", temperature=0, base_url="") 
    agent = create_react_agent(llm, tools=[])