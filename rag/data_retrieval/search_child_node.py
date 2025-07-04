from torch import cuda
import weaviate
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.types import Command
from rag.data_retrieval.retriever_state import RetrieverState

from rag.config import (
    CHILD_INDEX, 
    EMBEDDING_MODEL, 
    WEAVIATE_URL, 
    WEAVIATE_URL_HOST, 
    WEAVIATE_URL_PORT, 
    WEAVIATE_SECURE, 
    file_id_key, 
    parent_id_key
)


device = "cuda" if cuda.is_available() else "cpu"
emb = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True},
)


def get_child_chunks(state: RetrieverState) -> Command:
    print(f"Entering in CHILD SEARCH:\n")
    weaviate_client = weaviate.connect_to_local(
        skip_init_checks=True
    )

    try:
        child_store = WeaviateVectorStore(
            client=weaviate_client,
            index_name=CHILD_INDEX,
            text_key="text",
            embedding=emb,
            attributes=[file_id_key, parent_id_key],
        )

        docs = child_store.similarity_search(state["question"], k=10, alpha=0.5)
        print(f"Total child docs found: {len(docs)}")
         
        return Command(
            update={
                "child_chunks": docs,
            },
            goto="re_ranking"
        )
    finally:
        weaviate_client.close()

if __name__ == "__main__":
    res = get_child_chunks({"question": "Explain Strategic Design of DDD"})

    
