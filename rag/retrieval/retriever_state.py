from typing import TypedDict
from langchain_core.documents import Document


class RetrieverState(TypedDict):
    question: str
    child_chunks: list[Document]
    qa_chunks: list[Document]
    parent_ids: list[str]
    parent_docs: list[Document]
    compressed_docs: list[Document]
    
