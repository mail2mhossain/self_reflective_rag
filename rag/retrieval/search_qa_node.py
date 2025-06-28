from torch import cuda
import weaviate
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.types import Command
from rag.retrieval.retriever_state import RetrieverState
from rag.config import (
    QA_INDEX, 
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


def get_qa_chunks(state: RetrieverState) -> Command:
    print(f"Entering in QA SEARCH:\n")
    weaviate_client = weaviate.connect_to_local(
        skip_init_checks=True
    )

    try:
        qa_store = WeaviateVectorStore(
            client=weaviate_client,
            index_name=QA_INDEX,
            text_key="text",
            embedding=emb,
            attributes=[file_id_key, parent_id_key],
        )

        docs = qa_store.similarity_search(state["question"], k=10, alpha=0.5)
        print(f"Total qa docs found: {len(docs)}")
        
        return Command(
            update={
                "qa_chunks": docs,
            },
            goto="re_ranking"
        )
    finally:
        weaviate_client.close()

if __name__ == "__main__":
    res = get_qa_chunks({"question": "Explain Strategic Design of DDD"})
    
