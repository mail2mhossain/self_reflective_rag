# 1. Install Marker (requires Python 3.10+ and PyTorch)
# pip install marker-pdf

# 2. For full support (Office, HTML, EPUB)
# pip install "marker-pdf[full]"
# 3. Convert a single file to Markdown
# marker_single report.pdf --output_format markdown --output_dir ./out
# 4. Batch convert with LLM help
# marker ./docs --workers 4 --use_llm

import os
from uuid import uuid4
from io import StringIO
import pandas as pd
import re
import tiktoken
from langchain.schema import Document
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from .file_utils    import generate_file_id
from rag.data_ingest_pipeline.chunking      import suggest_chunk_sizes
from rag.data_ingest_pipeline.qa_utils      import get_question_answers
from rag.data_ingest_pipeline.table_summarization import generate_table_summaries
from rag.data_ingest_pipeline.image_caption_generation import generate_image_caption, clean_up
from rag.data_ingest_pipeline.vector_store  import store_in_vector_db
from rag.data_ingest_pipeline.mongo_store   import store_parent_docs_in_mongodb
from rag.config import parent_id_key, file_id_key, chunk_type_key, CHUNK_TYPE, FILE_PATH


# python -m rag.data_ingest_pipeline.multi_modal_data_ingest_pipeline


def extract_md_tables(md_text: str) -> list[pd.DataFrame]:
    """
    Find all markdown tables in md_text and return them as DataFrames.
    """
    tables = []
    # This regex captures: header row, separator row, then all data rows
    pattern = re.compile(
        r'(\|[^\n]*\|)\s*\n'            # header line
        r'(\|[ \-|]*\|)\s*\n'          # separator line (---|---)
        r'((?:\|[^\n]*\|\s*\n?)+)'     # one or more data lines
    )
    for header, sep, body in pattern.findall(md_text):
        # build a mini-markdown string
        md_table = header + "\n" + sep + "\n" + body
        # Use pandas to read it; drop the outer empty columns
        df = pd.read_csv(
            StringIO(md_table),
            sep="|",
            header=0,
            engine="python"
        ).iloc[:, 1:-1]  # drop the first/last empty columns
        tables.append(df)
    return tables

def remove_md_tables(md_text: str) -> str:
    """
    Remove all markdown tables from the given markdown text.
    """
    pattern = re.compile(
        r'(\|[^\n]*\|)\s*\n'            # header line
        r'(\|[ \-|]*\|)\s*\n'          # separator line (---|---)
        r'((?:\|[^\n]*\|\s*\n?)+)'     # one or more data lines
    )
    # Replace all occurrences of tables with empty string
    cleaned_text = pattern.sub('', md_text)
    return cleaned_text
    
def normalize_text(text: str) -> str:
    # Normalize whitespace, line endings etc.
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def clean_text(text: str, images: dict) -> Document:
    # remove tables
    text = remove_md_tables(text)
    # Remove images
    for key, image in images.items():
        if key in text:
            text = text.replace(key, "")
    # Remove extra whitespace, line endings, etc.
    text = normalize_text(text)
    # Remove extra newlines
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    return Document(page_content=text)
    
def get_absolute_path(file_path: str) -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../.."))
    return os.path.join(project_root, file_path)

def table_to_text(df: pd.DataFrame, max_rows=5) -> str:
    """
    Convert a DataFrame into a concise textual description.
    Limits to max_rows for brevity.
    """
    rows = min(len(df), max_rows)
    lines = []
    
    # Get column names
    columns = df.columns.tolist()
    
    for i in range(rows):
        row = df.iloc[i]
        # Construct a sentence like:
        # "Row 1: Column1 is val1, Column2 is val2, ..."
        parts = []
        for col in columns:
            val = row[col]
            # Convert to string safely, handling NaN
            val_str = str(val) if pd.notna(val) else "N/A"
            parts.append(f"{col} is {val_str}")
        line = f"Row {i+1}: " + ", ".join(parts) + "."
        lines.append(line)
    
    if len(df) > max_rows:
        lines.append(f"... and {len(df) - max_rows} more rows.")
    
    return "\n".join(lines)

converter = PdfConverter(
    artifact_dict=create_model_dict(),
    # device="cpu"  # Force CPU usage
)

def multi_modal_data_ingest_pipeline(
    file_path: str,
    model_name: str = "gpt-3.5-turbo",
    chunks_per_query: int = 6,
):
    FILE_NAME_WITH_EXTENSION = file_path.split("/")[-1]
    FILE_NAME = FILE_NAME_WITH_EXTENSION.split(".")[0]

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

    child_docs, qa_docs, tables, images = [], [], [], []

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

    print("Loading document as markdown format...")
    rendered = converter(file_path)
    md_text, _, images = text_from_rendered(rendered)

    # ❷ split on H1/H2/H3 – choose depth to fit your parent target size
    headers = [("#", "h1"), ("##", "h2"), ("###", "h3")]
    md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers,
            strip_headers=False   # keep the heading text inside the chunk
    )
    documents = md_splitter.split_text(md_text) 
    
    parent_docs = []
    for i, doc in enumerate(documents):
        if num_tokens(doc.page_content) > parent_size:
            subs = parent_splitter.split_documents([doc])
            parent_docs.extend(subs)
        else:
            parent_docs.append(doc)

    for i, parent_doc in enumerate(parent_docs):
        pid = str(uuid4())
        
        parent_doc.metadata[parent_id_key] = pid
        parent_doc.metadata[file_id_key]   = file_id
        parent_doc.metadata["source"] = file_path
        
        print(f"\nParent doc {i}: {pid}")
        extracted_tables = extract_md_tables(parent_doc.page_content)
        if extracted_tables:
            t = 0
            for j, table in enumerate(extracted_tables):
                t += 1
                csv_file_name = f"{FILE_PATH}/table_{pid}_{j}.csv"
                table.to_csv(csv_file_name, index=False)
                table_summary = generate_table_summaries(table.to_csv(index=False))
                table_doc = Document(page_content=table_summary)
                table_doc.metadata[file_id_key] = file_id
                table_doc.metadata[parent_id_key] = pid
                table_doc.metadata[chunk_type_key] = CHUNK_TYPE.TABLE
                table_doc.metadata["source"] = csv_file_name
                if num_tokens(table_doc.page_content) > child_size:
                    child_docs.extend(child_splitter.split_documents([table_doc]))
                else:
                    child_docs.append(table_doc)
            if t > 0:
                print(f"--- Child docs (TABLE): {t}")
        
        if images:
            k = 0
            for key, image in images.items():
                if key in parent_doc.page_content:
                    k += 1
                    image_file_name = f"{FILE_PATH}/image_{pid}_{key}"
                    image.save(image_file_name)
                    image_caption = generate_image_caption(image_file_name)

                    image_doc = Document(page_content=image_caption)
                    image_doc.metadata[file_id_key] = file_id
                    image_doc.metadata[parent_id_key] = pid
                    image_doc.metadata[chunk_type_key] = CHUNK_TYPE.IMAGE
                    image_doc.metadata["source"] = image_file_name
                    if num_tokens(image_doc.page_content) > child_size:
                        child_docs.extend(child_splitter.split_documents([image_doc]))
                    else:
                        child_docs.append(image_doc)
            if k > 0:
                print(f"--- Child docs (IMAGE): {k}")

        clean_p_doc = clean_text(parent_doc.page_content, images)
        clean_p_doc.metadata[parent_id_key] = pid
        clean_p_doc.metadata[file_id_key] = file_id
        clean_p_doc.metadata[chunk_type_key] = CHUNK_TYPE.TEXT
        clean_p_doc.metadata["source"] = file_path
        childs = child_splitter.split_documents([clean_p_doc])
        child_docs.extend(childs)
        q_a = get_question_answers(clean_p_doc)
        qa_docs.extend(q_a)

        print(f"--- Child docs (TEXT): {len(childs)}")
        print(f"--- QA docs: {len(q_a)}")

    print(f"Parent docs: {len(parent_docs)}")
    print(f"Child docs: {len(child_docs)}")
    print(f"QA docs: {len(qa_docs)}")
    clean_up()
    print("Storing parent docs in MongoDB...")
    store_parent_docs_in_mongodb(parent_docs)
    print("Storing child docs and QA pairs in vector database...")
    store_in_vector_db(child_docs, qa_docs)

    print("Data Ingestion completed successfully!")
    

if __name__ == '__main__':
    PDF_FILE_PATH = "rag/data_ingest_pipeline/Architect.pdf"
    
    multi_modal_data_ingest_pipeline(PDF_FILE_PATH)

    


                
        


