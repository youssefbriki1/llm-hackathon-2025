from langchain_core.messages import convert_to_messages
from typing import List, Any, Dict, Optional
from langchain_core.vectorstores.in_memory import InMemoryVectorStore
import getpass
import os
from dotenv import load_dotenv
import ast
import logging
import anyio
from langchain.tools import tool
from langchain_core.tools import ToolException
from pydantic import BaseModel, Field

from remotemanager import Logger, URL, Dataset
from remotemanager.storage.function import Function
from remotemanager.dataset.runner import RunnerFailedError
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents.base import Document

server_instructions = """This server provides functionality to run Python functions remotely on a specified server.
Provide a valid python function as a string, along with the hostname of the server where the function should be executed."""

load_dotenv()

def get_api_key(env_var, prompt):
    if not os.getenv(env_var):
        os.environ[env_var] = getpass.getpass(prompt)

def pretty_print_message(message, indent=False):
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        return

    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)


def pretty_print_messages(update, last_message=False):
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print("\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        update_label = f"Update from node {node_name}:"
        if is_subgraph:
            update_label = "\t" + update_label

        print(update_label)
        print("\n")

        if "messages" in node_update:
            messages = convert_to_messages(node_update["messages"])
            if last_message:
                messages = messages[-1:]
            for m in messages:
                pretty_print_message(m, indent=is_subgraph)
            print("\n")
            continue

        if "structured_response" in node_update:
            sr = node_update["structured_response"]
            try:
                data = sr.model_dump()
            except Exception:
                data = sr if isinstance(sr, dict) else {"response": getattr(sr, "response", str(sr)),
                                                        "sources": getattr(sr, "sources", [])}
            resp = data.get("response")
            sources = data.get("sources", [])
            indent = "\t" if is_subgraph else ""
            if resp:
                print(f"{indent}{resp}\n")
            if sources:
                print(f"{indent}Sources: {', '.join(sources)}\n")
            print("\n")


def create_mock_retriever(docs: List[str]):
    from langchain.schema import Document
    from langchain_openai.embeddings import OpenAIEmbeddings


    documents = [Document(page_content=doc) for doc in docs]
    embeddings = OpenAIEmbeddings()
    vectorstore = InMemoryVectorStore.from_documents(documents, embeddings)
    return vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": 2, "lambda_mult": 0.6}
    )


logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("remote")
file_handler = logging.FileHandler("remoterun.log")
logger.addHandler(file_handler)
Logger.level = "Debug"


def _validate_function(function_source: str) -> None:
    try:
        ast.parse(function_source)
    except SyntaxError as e:
        logger.error(f"Unable to parse function source code: {e}")
        raise ToolException(f"Unable to parse function source code. Ensure valid Python: {e}")
    logger.info("Function source code is valid.")

def _validate_url(hostname: str) -> URL:
    url = URL(hostname, verbose=0)
    try:
        url.cmd("pwd", timeout=1, max_timeouts=1)
    except RuntimeError as e:
        logger.error(f"Invalid hostname or unable to connect: {e}")
        raise ToolException(
            f"Invalid hostname or unable to connect. Check hostname '{hostname}':\n{e}"
        )
    logger.info("Hostname is valid.")
    return url

def _generate_name(fn_name: str, hostname: str) -> str:
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{fn_name}_{hostname}_{ts}"


class RunCodeInput(BaseModel):
    function_source: str = Field(
        ...,
        description="Full Python function source (def ...). Include any required imports inside the function."
    )
    hostname: str = Field(
        ...,
        description="Remote host to execute on."
    )
    function_args: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Keyword args for the function, e.g. {'n': 10}."
    )


@tool("remote_run_code", args_schema=RunCodeInput)
async def remote_run_code(function_source: str,
                          hostname: str,
                          function_args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Execute a Python function on a remote host via remotemanager and return {'Result': ...} or {'Error': ...}.
    """
    function_args = function_args or {}

    logger.info("#### New function execution. ####")
    _validate_function(function_source)
    url = _validate_url(hostname)

    fn = Function(function_source)

    try:
        logger.info("Testing function execution locally (dry-run).")
        # cache_env = os.environ.get("BIGDFT_MPIDRYRUN", "0")
        os.environ["BIGDFT_MPIDRYRUN"] = "1"
        fn(**function_args)
    except Exception as e:
        logger.error(f"Function test execution failed: {e}")
        raise ToolException(f"Function dry-run test execution failed. Ensure the function can be executed as-is: {e}")
    finally:
        os.environ["BIGDFT_MPIDRYRUN"] = "0"

    base_name = _generate_name(fn.name, hostname)
    ds = Dataset(
        fn,
        name=base_name,
        local_dir=f"staging_{base_name}",
        url=url,
        skip=False,
        verbose=False,
    )

    ds.append_run(args=function_args)
    ds.run()

    logger.info("Waiting for function to complete...")
    await anyio.to_thread.run_sync(ds.wait, 1, 300)

    logger.info("Fetching results.")
    ds.fetch_results()

    result = ds.results[0]
    if isinstance(result, RunnerFailedError):
        logger.error(f"Function execution failed: {result}")
        return {"Error": f"Function execution failed: {result}"}

    logger.info(f"Function executed successfully. Result: {result}")
    return {"Result": str(result)}

import asyncio

async def async_test():
    res = await remote_run_code.ainvoke({
        "function_source": """def f(x):
            return x*2""",
        "hostname": "localhost",
        "function_args": {"x": 21}
    })
    print(res)
    
"""
    
import asyncio
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

async def main():
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    agent = create_react_agent(
        model=model,
        tools=[remote_run_code],
        prompt="You can execute remote Python functions using the tool when asked.",
        name="remotemanager_agent",
    )
    res = await agent.ainvoke({
        "messages": [{
            "role": "user",
            "content": (
                "Use the remote_run_code tool to run this on 'localhost': "
                "function_source='def f(x):\\n    return x*2', function_args={'x': 21}"
            )
        }]
    })
    print(res)

    



if __name__ == "__main__":
    print("Async test of remote_run_code tool:")
    asyncio.run(async_test())
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    print("\n\n\n\n\n\n\n\n\n --------------------------------------")
    asyncio.run(main())

"""
