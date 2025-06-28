import uuid
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter

from .file_utils    import generate_file_id
from rag.data_ingest_pipeline.chunking      import suggest_chunk_sizes
from rag.data_ingest_pipeline.loader        import get_dox_from_file
from rag.data_ingest_pipeline.qa_utils      import get_question_answers
from rag.data_ingest_pipeline.vector_store  import store_in_vector_db
from rag.data_ingest_pipeline.mongo_store   import store_parent_docs_in_mongodb
from rag.config import parent_id_key, file_id_key


# python -m rag.data_ingest_pipeline.pipeline

def data_ingest_pipeline(
    file_path: str,
    model_name: str = "gpt-3.5-turbo",
    chunks_per_query: int = 6,
):

    print("Generating file ID...")
    file_id = generate_file_id(file_path)
    print("Getting chunk sizes...")
    parent_size, child_size, parent_ol, child_ol = suggest_chunk_sizes(model_name, chunks_per_query)
    print("Parent size: ", parent_size)
    print("Child size: ", child_size)
    print("Parent overlap: ", parent_ol)
    print("Child overlap: ", child_ol)

    enc = tiktoken.encoding_for_model(model_name)
    def num_tokens(text: str) -> int:
        return len(enc.encode(text))

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_size,
        chunk_overlap=parent_ol,
        length_function=num_tokens,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_size,
        chunk_overlap=child_ol,
        length_function=num_tokens,
    )

    print("Loading document...")
    document   = get_dox_from_file(file_id, file_path)
    print("Splitting document into parent chunks...")
    parent_docs = parent_splitter.split_documents([document])
    print("Parent docs: ", len(parent_docs))
    child_docs, qa_docs = [], []

    print("Splitting document into child chunks and QA pairs...")
    for p in parent_docs:
        p.metadata[parent_id_key] = str(uuid.uuid4())
        p.metadata[file_id_key]   = file_id

        subs = child_splitter.split_documents([p])
        child_docs.extend(subs)
        qa_docs.extend(get_question_answers(p))

    print("Child docs: ", len(child_docs))
    print("QA docs: ", len(qa_docs))
    print("Storing parent docs in MongoDB...")
    store_parent_docs_in_mongodb(parent_docs)
    print("Storing child docs and QA pairs in vector database...")
    store_in_vector_db(child_docs, qa_docs)


if __name__ == "__main__":
    data_ingest_pipeline("rag/data_ingest_pipeline/Architect.pdf")