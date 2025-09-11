from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, field_validator, ConfigDict,PrivateAttr
from typing import List, Any
from langchain.tools.retriever import create_retriever_tool
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models.chat_models import BaseChatModel
class ResponseFormat(BaseModel):
    response: str = Field(..., description="LLM's response to the user")
    sources: List[str] = Field(..., description="List of sources used to generate the response")



class PyDFTAgent(BaseModel):
    model: BaseChatModel
    retriever: BaseRetriever

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
        self._rag_agent = create_react_agent(
            model=self.model,
            tools=[retriever_tool],
        )
        self._supervisor = create_supervisor(
            model=self.model,
            agents=[self._rag_agent],
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
