from argparse import OPTIONAL
from dataclasses import dataclass, field
from langgraph.prebuilt import create_react_agent, ToolNode, InjectedState
from langgraph_supervisor import create_supervisor
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, field_validator, ConfigDict, PrivateAttr
from typing import List, Any, Optional, Dict, Annotated
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import get_buffer_string
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.vectorstores.in_memory import InMemoryVectorStore
from langchain_openai.embeddings import OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
import uuid
from .utils import pretty_print_messages, create_mock_retriever,remote_run_code
from langgraph.types import Command
from langgraph.types import Send
import getpass
import os
from dotenv import load_dotenv
from OntoFlow.agent.Onto_wa_rag.retriever_adapter import init_retriever, retriever_tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from .rag_graph_pyd import build_rag_graph, DocSpec
import asyncio
from pathlib import Path


# TODO: Edit this to support embedding of multiple docs 

load_dotenv()

def get_api_key(env_var, prompt):
    if not os.getenv(env_var):
        os.environ[env_var] = getpass.getpass(prompt)

# Memory tool 
@dataclass
class EmbeddingModelWrapper:
    embeddings: Embeddings = field()

    def __post_init__(self):
        if self.embeddings is None:
            raise ValueError("embeddings cannot be empty")

model = EmbeddingModelWrapper(embeddings=OpenAIEmbeddings())

recall_vector_store = InMemoryVectorStore(model.embeddings)



@tool
def save_recall_memory(memory: str, config: RunnableConfig) -> str:
    """Save memory to vectorstore for later semantic retrieval."""
    document = Document(
        page_content=memory, id=str(uuid.uuid4())
    )
    recall_vector_store.add_documents([document])
    return memory


@tool
def search_recall_memories(query: str, config: RunnableConfig) -> List[str]:
    """Search for relevant memories."""


    documents = recall_vector_store.similarity_search(
        query, k=3
    )
    return [document.page_content for document in documents]


# Agent utils

def create_task_description_handoff_tool(*, agent_name, description=None):
    name = f"transfer_to_{agent_name}"
    description = description or f"Ask {agent_name} for help."

    @tool(name, description=description)
    def handoff_tool(
        task_description: Annotated[str, "..."],
        state: Annotated[MessagesState, InjectedState],
    ) -> Command:
        task_description_message = {"role": "user", "content": task_description}
        agent_input = {**state, "messages": [task_description_message]}
        return Command(
            goto=[Send(agent_name, agent_input)],
            graph=Command.PARENT,
        )
    return handoff_tool


# Rag Response format scheme
class ResponseFormat(BaseModel):
    response: str = Field(..., description="LLM's response to the user")
    sources: List[str] = Field(default_factory=list, description="List of sources used to generate the response")


# Remoterun tools 



class PyDFTAgent(BaseModel):
    model: BaseChatModel
    remotemanger_model: Optional[BaseChatModel] = None
    rag_model: Optional[BaseChatModel] = None
    supervisor_model: Optional[BaseChatModel] = None
    #retriever: BaseRetriever
    _remotemanager_agent: Any = PrivateAttr(default=None)
    _rag_agent: Any = PrivateAttr(default=None)
    _supervisor: Any = PrivateAttr(default=None)
    _supervisor_with_description: Any = PrivateAttr(default=None)
    
    path:str # add validators
    

    model_config = ConfigDict(
        extra='forbid',
        arbitrary_types_allowed=True, 
    )

    """
    @field_validator("path")
    @classmethod
    def _check_path(cls, path:str) -> str:
        if os.is
    """    
        
    @field_validator("model")
    @classmethod
    def _check_model(cls, v: BaseChatModel) -> BaseChatModel:
        if v is None:
            raise ValueError("model cannot be empty")
        return v


    def model_post_init(self, __context: Any) -> None:
        
        # Initialize retriever
        rag = OntoRAG(
            storage_dir=STORAGE_DIR,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            ontology_path=ONTOLOGY_PATH_TTL
        )

        await rag.initialize()
        
        
        # Embed docs:
        
        
        
        remotemanager_agent = create_react_agent(
            model=self.remotemanger_model or self.model,
            tools=[save_recall_memory, search_recall_memories, remote_run_code],
            prompt=(
                "You are a Python coding assistant. Use the remote_run_code tool to execute Python functions "
                "remotely. Write complete function definitions and call them with appropriate arguments."
            ),
            name="remotemanager_agent",
        )

        rag_app = build_rag_graph()

        def _latest_human_question(state: Dict[str, Any]) -> Optional[str]:
            for msg in reversed(state.get("messages", [])):
                if isinstance(msg, HumanMessage):
                    c = msg.content if isinstance(msg.content, str) else ""
                    if c.strip():
                        return c.strip()
            return None

        async def rag_graph_node(state: Dict[str, Any]) -> Dict[str, Any]:
            q = _latest_human_question(state) or state.get("query", "")
            docs_raw: List[Dict[str, Any]] = state.get("docs", []) or []
            docs = [d if isinstance(d, DocSpec) else DocSpec(**d) for d in docs_raw]
            rag_in = {"query": q, "docs": [d.model_dump() for d in docs]}
            rag_out: Dict[str, Any] = await rag_app.ainvoke(rag_in)
            ans = (rag_out or {}).get("answer", "") or ""
            srcs = (rag_out or {}).get("sources", []) or []
            return {"rag_answer": ans, "rag_sources": srcs, "messages": [AIMessage(content=ans)]}

        async def submit_rag_to_remotemanager(state: Dict[str, Any], hostname: str = "localhost") -> Dict[str, Any]:
            ans: str = state.get("rag_answer", "") or ""
            srcs: List[Dict[str, Any]] = state.get("rag_sources", []) or []
            function_source = """
    def handle_rag_result(answer: str, sources: list, metadata: dict | None = None):
        return {
            "ok": True,
            "answer_len": len(answer),
            "source_count": len(sources),
            "preview": answer[:200],
            "metadata": (metadata or {})
        }
    """
            args = {
                "function_source": function_source,
                "hostname": hostname,
                "function_args": {"answer": ans, "sources": srcs, "metadata": {"origin": "langgraph", "pipeline": "ontoRAG->remotemanager"}},
            }
            tool_result = await remote_run_code.ainvoke(args)
            tool_msg = ToolMessage(content=str(tool_result), tool_call_id="rag_submit")
            return {"remotemanager_result": tool_result, "messages": [tool_msg]}

        assign_to_research_agent_with_description = create_task_description_handoff_tool(
            agent_name="rag_agent",
            description="Use the OntoRAG research assistant to answer document questions.",
        )

        assign_to_math_agent_with_description = create_task_description_handoff_tool(
            agent_name="remotemanager_agent",
            description="Use RemoteRun to execute Python code on a remote host.",
        )

        supervisor_agent_with_description = create_react_agent(
            model=self.supervisor_model or self.model,
            tools=[assign_to_research_agent_with_description, assign_to_math_agent_with_description, save_recall_memory, search_recall_memories],
            prompt=(
                "You are a supervisor managing two assistants:\n"
                "- 'rag_agent' (OntoRAG) for research over code/docs.\n"
                "- 'remotemanager_agent' for remote code execution via RemoteRun.\n"
                "Assign one assistant at a time. Do not do tasks yourself."
            ),
            name="supervisor",
        )

        self._supervisor_with_description = (
            StateGraph(MessagesState)
            .add_node(supervisor_agent_with_description, destinations=("rag_agent", "remotemanager_agent"))
            .add_node("rag_agent", rag_graph_node)
            .add_node("submit_rag_to_remotemanager", submit_rag_to_remotemanager)
            .add_node("remotemanager_agent", remotemanager_agent)
            .add_edge(START, "supervisor")
            .add_edge("rag_agent", "submit_rag_to_remotemanager")
            .add_edge("submit_rag_to_remotemanager", "supervisor")
            .add_edge("remotemanager_agent", "supervisor")
            .compile()
        )





    @property
    def rag_agent(self):
        return self._rag_agent

    @property
    def supervisor(self):
        return self._supervisor

    @property
    def remote_manager(self):
        return self._remotemanager_agent
    
    def draw_image(self):
        with open("supervisor_graph.png", "wb") as f:
            f.write(self._supervisor_with_description.get_graph().draw_mermaid_png())

    
    async def arun(self, user_input: str):
        async for chunk in self._supervisor_with_description.astream(
            {"messages": [{"role": "user", "content": user_input}]},
            subgraphs=True, verbose=True
        ):
            pretty_print_messages(chunk, last_message=True)

    def run(self, user_input: str):
        asyncio.run(self.arun(user_input))



if __name__ == "__main__":
    agent = PyDFTAgent(
        model=ChatOpenAI(model="gpt-4o", temperature=0),
        path="",
    )
    agent.draw_image()
    agent.run("")
    