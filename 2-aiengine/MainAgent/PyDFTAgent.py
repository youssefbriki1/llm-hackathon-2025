from argparse import OPTIONAL
from dataclasses import dataclass, field
from langgraph.prebuilt import create_react_agent, ToolNode, InjectedState
from langgraph_supervisor import create_supervisor
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, field_validator, ConfigDict, PrivateAttr
from typing import List, Any, Optional, Dict, Annotated
from langchain.tools.retriever import create_retriever_tool
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import get_buffer_string
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai.embeddings import OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
import uuid
from .utils import pretty_print_messages, pretty_print_message
from langgraph.types import Command
from langgraph.types import Send

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
    sources: List[str] = Field(..., description="List of sources used to generate the response")


# Remoterun tools 

class RunCodeInput(BaseModel):
    """Execute a Python function remotely via MCP RemoteRun."""
    function_source: str = Field(..., description="Full Python function source (def ...).")
    #hostname: str = Field("localhost", description="Remote host to run the function on.")
    function_args: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Keyword args passed to your function."
    )
    model_config = ConfigDict(
        extra='allow')


@tool("remote_run_code", args_schema=RunCodeInput)
def remoterun_tool(
    function_source: str,
    function_args: Optional[Dict[str, Any]] = None
) -> Any:
    pass

@tool("remote_run_code_async", args_schema=RunCodeInput)
async def aremoterun_tool(
    function_source: str,
    function_args: Optional[Dict[str, Any]] = None
) -> Any:
    pass



class PyDFTAgent(BaseModel):
    model: BaseChatModel
    remotemanger_model: Optional[BaseChatModel] = None
    rag_model: Optional[BaseChatModel] = None
    supervisor_model: Optional[BaseChatModel] = None
    
    retriever: BaseRetriever
    _remotemanager_agent: Any = PrivateAttr(default=None)
    _rag_agent: Any = PrivateAttr(default=None)
    _supervisor: Any = PrivateAttr(default=None)

    model_config = ConfigDict(
        extra='forbid',
        arbitrary_types_allowed=True, 
    )

    @field_validator("model")
    @classmethod
    def _check_model(cls, v: BaseChatModel) -> BaseChatModel:
        if v is None:
            raise ValueError("model cannot be empty")
        return v

    @field_validator("retriever")
    @classmethod
    def _check_retriever(cls, v: BaseRetriever) -> BaseRetriever:
        if v is None:
            raise ValueError("retriever cannot be empty")
        return v

    def model_post_init(self, __context: Any) -> None:
        """Build runtime agents once the model is fully initialized."""
        retriever_tool = create_retriever_tool(
            retriever=self.retriever,
            name="Document Retriever",
            description="Use this tool to retrieve relevant documents.",
        )
        
        # Rag Agent creation
        self._rag_agent = create_react_agent(
            model=self.model if self.rag_model is None else self.rag_model,
            tools=[retriever_tool],
            # TODO: Structured output
            response_format=PydanticOutputParser(pydantic_object=ResponseFormat),
            prompt=(
                "You are a remote manager agent that oversees the execution of code on remote machines."
            )
        )
        
        
        # Remote Manager Agent creation
        self._remotemanager_agent = create_react_agent(
            model=self.model if self.remotemanger_model is None else self.remotemanger_model,
            tools=[],
            prompt=(
                "You are a remote manager agent that oversees the execution of code on remote machines."
            )
        )   

        # Supervisor Agent creation
        self._supervisor = create_supervisor( 
            model=self.model if self.supervisor_model is None else self.supervisor_model,
            agents=[self._rag_agent, self._remotemanager_agent],
            prompt=(
                "You are a supervisor agent that manages multiple agents "
                "to answer user queries effectively."
            ),
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