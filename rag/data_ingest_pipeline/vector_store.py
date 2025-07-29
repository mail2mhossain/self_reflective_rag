from torch import cuda
import weaviate
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
import warnings

# pip install weaviate-client langchain_weaviate langchain_huggingface
# pip install sentence-transformers
warnings.filterwarnings("ignore")

# docker pull semitechnologies/weaviate:latest

# docker run -d --name weaviate -p 8080:8080 -p 50051:50051 -e QUERY_DEFAULTS_LIMIT=20 -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true -e PERSISTENCE_DATA_PATH=/var/lib/weaviate -v Z:\weaviate-data:/var/lib/weaviate semitechnologies/weaviate:latest



device = f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"
model_kwargs = {"device": device}
encode_kwargs = {"normalize_embeddings": True}


from rag.config import (
    CHILD_INDEX, QA_INDEX,
    EMBEDDING_MODEL, WEAVIATE_URL
)

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

def store_in_vector_db(child_docs, qa_docs):
    msg = f"Storing child and QA documents in vector database"
    print(msg)

    weaviate_client = weaviate.connect_to_local(
        skip_init_checks=True
    )
    
    WeaviateVectorStore.from_documents(
        child_docs, 
        embeddings, 
        client=weaviate_client, 
        index_name=CHILD_INDEX, 
        text_key="text", 
        by_text=False)
    
    WeaviateVectorStore.from_documents(
        qa_docs, 
        embeddings, 
        client=weaviate_client, 
        index_name=QA_INDEX, 
        text_key="text", 
        by_text=False)
    
    msg = f"Successfully saved in vector database"
    print(msg)
    weaviate_client.close()

    

def delete_from_vector_db(file_id):
    msg = f"Deleting file from vector database: {file_id}"
    print(msg)
    pass

