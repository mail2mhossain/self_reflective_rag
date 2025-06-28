# python -m rag.data_ingest_pipeline.qa_utils

from langchain.schema import Document
from rag.data_ingest_pipeline.qa_generation import generate_qa

from rag.config import file_id_key, parent_id_key

def get_question_answers(parent_doc: Document) -> list[Document]:
    qa_data = generate_qa(parent_doc.page_content)
    qa_docs = []
    for qa in qa_data.qa_pairs:
        doc = Document(page_content=qa.question + "\n" + qa.answer)
        doc.metadata[file_id_key]   = parent_doc.metadata[file_id_key]
        doc.metadata[parent_id_key] = parent_doc.metadata[parent_id_key]
        doc.metadata["source"]      = parent_doc.metadata["source"]
        qa_docs.append(doc)
    return qa_docs
