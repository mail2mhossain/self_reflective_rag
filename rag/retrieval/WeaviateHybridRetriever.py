from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain.retrievers.weaviate_hybrid_search import WeaviateHybridSearchRetriever


class WeaviateHybridCustomRetriever(BaseRetriever):
    parent_store: WeaviateHybridSearchRetriever = Field(...)
    query_embedding: List[float] = Field(
        ...,
        description="List of floats representing the query embedding.",
        example=[0.1, 0.2, 0.3],
        title="Query Embedding",
    )
    search_kwargs: Dict[str, Any] = Field(
        ...,
        description="Dictionary of search keyword arguments.",
        example={"filter": "some_value", "limit": 10},
        title="Search Keyword Arguments",
    )

    def __init__(
        self,
        parent_store,
        query_embedding,
        search_kwargs,
        *args,  # Additional arguments for BaseRetriever
        **kwargs  # Additional keyword arguments for BaseRetriever
    ):
        super().__init__(*args, **kwargs)
        self.parent_store = parent_store
        self.query_embedding = query_embedding
        self.search_kwargs = search_kwargs

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        results = self.parent_store.get_relevant_documents(
            query,
            score=True,
            hybrid_search_kwargs={"vector": self.query_embedding},
            where_filter=self.search_kwargs,
        )

        return results
