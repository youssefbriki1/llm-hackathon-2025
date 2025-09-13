from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from langgraph.graph import StateGraph, START, END
from OntoFlow.agent.Onto_wa_rag.Integration_fortran_RAG import OntoRAG
from OntoFlow.agent.Onto_wa_rag.CONSTANT import STORAGE_DIR, CHUNK_SIZE, CHUNK_OVERLAP, ONTOLOGY_PATH_TTL, MAX_CONCURRENT

class DocSpec(BaseModel):
    filepath: str
    project_name: str
    version: str

class RAGInput(BaseModel):
    query: str
    docs: List[DocSpec] = Field(default_factory=list)

class SourceHit(BaseModel):
    filename: str
    start_line: int
    end_line: int
    entity_type: Optional[str] = None
    entity_name: Optional[str] = None
    detected_concepts: Optional[List[str]] = None
    relevance_score: Optional[float] = None

class RAGOutput(BaseModel):
    answer: str = ""
    sources: List[SourceHit] = Field(default_factory=list)

class GraphState(BaseModel):
    model_config = ConfigDict(extra="allow")
    query: Optional[str] = None
    docs: Optional[List[DocSpec]] = None
    answer: Optional[str] = None
    sources: Optional[List[SourceHit]] = None

class OntoRAGService:
    def __init__(self):
        self._rag: Optional[OntoRAG] = None
        self._initialized: bool = False

    async def ensure_ready(self):
        if self._rag is None:
            self._rag = OntoRAG(
                storage_dir=STORAGE_DIR,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                ontology_path=ONTOLOGY_PATH_TTL,
            )
        if not self._initialized:
            await self._rag.initialize()
            self._initialized = True

    async def add_docs(self, docs: List[DocSpec]):
        if docs:
            await self._rag.add_documents_batch([d.model_dump() for d in docs], max_concurrent=MAX_CONCURRENT)

    async def ask(self, question: str) -> Dict[str, Any]:
        return await self._rag.query(question, use_ontology=True)

rag_service = OntoRAGService()

async def ingest_node(state: Dict[str, Any]) -> Dict[str, Any]:
    data = RAGInput(**{k: v for k, v in state.items() if k in ("query", "docs")})
    await rag_service.ensure_ready()
    if data.docs:
        await rag_service.add_docs(data.docs)
    return {}

async def query_node(state: Dict[str, Any]) -> Dict[str, Any]:
    data = RAGInput(**{k: v for k, v in state.items() if k in ("query", "docs")})
    raw = await rag_service.ask(data.query)
    hits = []
    for s in raw.get("sources", []) or []:
        hits.append(
            SourceHit(
                filename=s.get("filename", ""),
                start_line=s.get("start_line", 0),
                end_line=s.get("end_line", 0),
                entity_type=s.get("entity_type"),
                entity_name=s.get("entity_name"),
                detected_concepts=s.get("detected_concepts"),
                relevance_score=s.get("relevance_score"),
            )
        )
    out = RAGOutput(answer=raw.get("answer", "") or "", sources=hits)
    return out.model_dump()

def build_rag_graph():
    g = StateGraph(dict)
    g.add_node("ingest", ingest_node)
    g.add_node("ask", query_node)
    g.add_edge(START, "ingest")
    g.add_edge("ingest", "ask")
    g.add_edge("ask", END)
    return g.compile()
