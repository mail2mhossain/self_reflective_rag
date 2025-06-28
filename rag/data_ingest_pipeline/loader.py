# python -m rag.dox_pipeline.loader

import mimetypes
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from rag.config import encoding, file_id_key

def load_document(file_path: str):
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type == 'application/pdf':
        loader = PyPDFLoader(file_path=file_path)
    elif mime_type == 'text/plain':
        loader = TextLoader(encoding=encoding, file_path=file_path)
    elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        loader = Docx2txtLoader(file_path=file_path)
    else:
        raise ValueError(f"Unsupported document type: {mime_type}")
    return loader.load()

def get_dox_from_file(file_id: str, file_path: str):
    docs = load_document(file_path)
    text = "\n".join(doc.page_content for doc in docs)
    src  = docs[0].metadata["source"]
    
    return Document(page_content=text, metadata={"source": src, file_id_key: file_id})
