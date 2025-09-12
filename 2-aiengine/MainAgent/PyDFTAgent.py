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

    @field_validator("retriever")
    @classmethod
    def _check_retriever(cls, v: BaseRetriever) -> BaseRetriever:
        if v is None:
            raise ValueError("retriever cannot be empty")
        return v

    def model_post_init(self, __context: Any) -> None:
        """Build runtime agents once the model is fully initialized."""
        
        init_retriever(self.path)  
        
        rag_agent = create_react_agent(
            model=self.rag_model or self.model,
            tools=[retriever_tool, save_recall_memory, search_recall_memories],
            prompt=(
                "You are a research assistant. Use the retriever tool to find relevant information "
                "from the provided documents to help answer user queries."
            ),
            response_format=ResponseFormat,
            name="rag_agent",
        ) 
        remotemanager_agent = create_react_agent(
            model=self.remotemanger_model or self.model,
            tools=[save_recall_memory, search_recall_memories, remote_run_code],
            prompt=(
                "You are a Python coding assistant. Use the remote_run_code tool to execute Python functions"
                "remotely. Write complete function definitions and call them with appropriate arguments."
            ),
            name="remotemanager_agent",
        )
        assign_to_research_agent_with_description = create_task_description_handoff_tool(
            agent_name="rag_agent",
            description="",
        )

        assign_to_math_agent_with_description = create_task_description_handoff_tool(
            agent_name="remoterun_agent",
            description="",
        )

        supervisor_agent_with_description = create_react_agent(
            model=self.supervisor_model or self.model,
            tools=[
                assign_to_research_agent_with_description,
                assign_to_math_agent_with_description,
                save_recall_memory, 
                search_recall_memories
            ],
            prompt=(
                "You are a supervisor managing two agents:\n"
                "- a RAG agent. Assign research-related tasks to this assistant\n"
                "- a RemoteRun agent. Assign code execution tasks to this assistant\n"
                "Assign work to one agent at a time, do not call agents in parallel.\n"
                "Do not do any work yourself."
            ),
            name="supervisor",
        )

        self._supervisor_with_description = (
            StateGraph(MessagesState)
            .add_node(
                supervisor_agent_with_description, destinations=("rag_agent", "remotemanager_agent")
            )
            .add_node(rag_agent)
            .add_node(remotemanager_agent)
            .add_edge(START, "supervisor")
            .add_edge("rag_agent", "supervisor")
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

    
    def run(self, user_input: str):
        for chunk in self._supervisor_with_description.stream(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": user_input,
                        }
                    ]
                },
                subgraphs=True, verbose=True):
            pretty_print_messages(chunk, last_message=True)



if __name__ == "__main__":
    
    agent = PyDFTAgent(
        model=ChatOpenAI(model="gpt-4o", temperature=0),
        path="example/D4_02_rag_with_langchain_and_chromadb.ipynb"
    )
    agent.draw_image()
    agent.run("tell me about llama-server in a RAG setting")
    
    