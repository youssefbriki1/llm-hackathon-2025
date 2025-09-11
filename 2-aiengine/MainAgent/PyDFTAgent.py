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
from langchain_core.vectorstores.in_memory import InMemoryVectorStore
from langchain_openai.embeddings import OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
import uuid
from utils import pretty_print_messages, create_mock_retriever
from langgraph.types import Command
from langgraph.types import Send
import getpass
import os
from dotenv import load_dotenv


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
    _supervisor_with_description: Any = PrivateAttr(default=None)

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
            name="document_retriever",
            description="Use this tool to retrieve relevant documents.",
        )
        
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
        """
        self._rag_agent = (
            StateGraph(MessagesState)
            .add_node(rag_agent)
            .add_edge(START, "rag_agent")
            .compile()
        )
        """
        remotemanager_agent = create_react_agent(
            model=self.remotemanger_model or self.model,
            tools=[remoterun_tool, save_recall_memory, search_recall_memories],
            prompt=(
                "You are a Python coding assistant. Use the remote_run_code tool to execute Python functions"
                "remotely. Write complete function definitions and call them with appropriate arguments."
            ),
            name="remotemanager_agent",
        )
        
        """
        self._remotemanager_agent = (
            StateGraph(MessagesState)
            .add_node(remotemanager_agent)
            .add_edge(START, "remoterun_agent")
            .compile()
        )
        """
        
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
    
    
    
    examples = [
        "Python is a high-level, interpreted programming language known for its readability and versatility. It supports multiple programming paradigms, including procedural, object-oriented, and functional programming. Python has a large standard library and a vibrant ecosystem of third-party packages, making it suitable for a wide range of applications such as web development, data analysis, artificial intelligence, scientific computing, and automation.",
        "The capital of France is Paris. It is known for its rich history, culture, art, and architecture. Paris is home to iconic landmarks such as the Eiffel Tower, Louvre Museum ",
        "The Great Wall of China is a series of fortifications that stretches over 13,000 miles across northern China. It was built to protect Chinese states and empires from invasions and raids by nomadic groups from the north. The wall is made of various materials, including stone, brick, tamped earth, and wood, and it is considered one of the most impressive architectural feats in history.",
        "The theory of relativity, developed by Albert Einstein, consists of two main parts: special relativity and general relativity. Special relativity, published in 1905, introduced the concept that the laws of physics are the same for all non-accelerating observers and that the speed of light is constant regardless of the motion of the light source. General relativity, published in 1915, expanded on this by describing gravity as the curvature of spacetime caused by mass and energy.",
        "Photosynthesis is the process by which green plants, algae, and some bacteria convert light energy into chemical energy stored in glucose. During photosynthesis, these organisms use sunlight to convert carbon dioxide and water into glucose and oxygen. The overall equation for photosynthesis can be summarized as: 6CO2 + 6H2O + light energy -> C6H12O6 + 6O2."
    ]
    agent = PyDFTAgent(
        model=ChatOpenAI(model="gpt-4o", temperature=0),
        retriever = create_mock_retriever(examples)
    )
    agent.draw_image()
    agent.run("tell me about python programming language")
    
    