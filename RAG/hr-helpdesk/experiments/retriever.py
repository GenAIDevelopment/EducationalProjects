from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional, Tuple
from langchain_core.documents import Document
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from langchain_postgres import PGVector

@dataclass
class RetrievalConfig:
    connection: str
    collection_name: str
    embedding_model: str

    # primary retrieval parameters
    primary_k: int = 6
    primary_fetch_k: int = 24
    mmr_lambd_mult: float = 0.7

    # thresolding
    use_threshold_retrieval: bool = False
    score_threshold: float = 0.78
    threshold_k: int = 6

    # fallback
    fallback_k:int = 6

    # safety/governance
    max_context_docs: int = 6


@dataclass
class RetrievalResult:
    query: str
    normalized_query: str
    inferred_filters: Dict[str, Any]
    search_strategy: str
    docs: List[Document] = field(default_factory=list)

class HRRetrievalPipeline:
    def __init__(self, config: RetrievalConfig):
        self.config = config
        self.embedding = VertexAIEmbeddings(
            model_name=config.embedding_model
        )
        self.vector_store = PGVector(
            connection=config.connection,
            embeddings=self.embedding,
            collection_name=config.collection_name,
            use_jsonb=True
         )

    def normalize_query(self, query:str) ->str:
        return " ".join(query.strip().split())
    
    def retrieve_mmr(self, query: str, metadata_filter: Optional[Dict[str, Any]] = None) -> list[Document]:
        retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": self.config.primary_k,
                "fetch_k": self.config.primary_fetch_k,
                "lambda_mult": self.config.mmr_lambd_mult,
                "filter": metadata_filter
            }
        )
        return retriever.invoke(query)
    
    def retrieve(self, query:str) -> RetrievalResult:
        normalized_query = self.normalize_query(query)
        
        docs = self.retrieve_mmr(normalized_query)
        # primary: MMR
        if docs:
            return RetrievalResult(
                query=query,
                normalized_query=self.normalize_query,
                inferred_filters={},
                search_strategy="mmr",
                docs=docs
            )
        
        # 2. use threshold search
        pass

    def format_citations(self, docs: List[Document]) -> List[Dict[str,str]]:
        citations = []
        for doc in docs:
            citations.append(
                {
                    "title": doc.metadata.get("title", "Unknown"),
                    "source": doc.metadata.get("source", "#"),

                }
            )
            
        return citations


