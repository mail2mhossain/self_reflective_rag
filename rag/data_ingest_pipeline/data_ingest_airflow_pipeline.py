from airflow.sdk import chain, dag, task, Asset
from pendulum import datetime
import os
import glob
import uuid
import tiktoken
from rag.data_ingest_pipeline.file_utils import generate_file_id
from rag.data_ingest_pipeline.loader import get_dox_from_file
from rag.data_ingest_pipeline.chunking import suggest_chunk_sizes
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rag.data_ingest_pipeline.qa_utils import get_question_answers
from rag.data_ingest_pipeline.vector_store import store_in_vector_db
from rag.data_ingest_pipeline.mongo_store import store_parent_docs_in_mongodb

@dag(
    start_date=datetime(2025, 6, 25),
    schedule="@daily",
    catchup=False,
)
def data_ingest_airflow_pipeline():

    @task
    def sensor_task() -> str:
        """Sensor Task: watch an incoming folder for new PDFs."""
        watch_folder = "/data/incoming"
        files = glob.glob(os.path.join(watch_folder, "*.pdf"))
        if not files:
            raise FileNotFoundError("No new PDF found in incoming folder.")
        return files[0]

    _sensor = sensor_task()

    @task
    def metadata_task(file_path: str) -> str:
        """Metadata Task: compute MD5 hash as file_id."""
        return generate_file_id(file_path)

    _file_id = metadata_task(_sensor)

    @task
    def content_extraction_task(file_path: str, file_id: str):
        """Content Extraction Task: load & standardize document."""
        return get_dox_from_file(file_id, file_path)

    _document = content_extraction_task(_sensor, _file_id)

    @task
    def chunking_task(document, file_id: str) -> dict:
        """Chunking Task: split into parent & child chunks."""
        model_name = "gpt-4o"
        chunks_per_query = 6

        parent_size, child_size, parent_ol, child_ol = suggest_chunk_sizes(
            model_name, chunks_per_query
        )

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

        parent_docs = parent_splitter.split_documents([document])
        child_docs = []
        for p in parent_docs:
            p.metadata["parent_id"] = str(uuid.uuid4())
            p.metadata["file_id"] = file_id
            child_docs.extend(child_splitter.split_documents([p]))

        return {"parent_docs": parent_docs, "child_docs": child_docs}

    _chunks = chunking_task(_document, _file_id)

    @task
    def qa_generation_task(parent_docs: list) -> list:
        """QA Generation Task: produce question-answer docs."""
        qa_docs = []
        for p in parent_docs:
            qa_docs.extend(get_question_answers(p))
        return qa_docs

    _qa = qa_generation_task(_chunks["parent_docs"])

    @task
    def storage_task(parent_docs: list, child_docs: list, qa_docs: list) -> None:
        """Storage Task: persist to vector DB and MongoDB."""
        store_in_vector_db(parent_docs, child_docs, qa_docs)
        store_parent_docs_in_mongodb(parent_docs)

    _storage = storage_task(
        _chunks["parent_docs"],
        _chunks["child_docs"],
        _qa,
    )

    chain(
        _sensor,
        _file_id,
        _document,
        _chunks,
        _qa,
        _storage,
    )

# instantiate the DAG
data_ingest_airflow_pipeline()
