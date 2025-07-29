import base64
from pandas import read_csv
from langgraph.types import Command
from rag.data_retrieval.retriever_state import RetrieverState
from rag.config import file_id_key, parent_id_key, chunk_type_key, CHUNK_TYPE


def filter_chunks(chunks, *, parent_id=None, chunk_type=None):
    """
    Return all chunks matching the given parent_id and/or chunk_type.
    If parent_id or chunk_type is None, that filter is ignored.
    """
    filtered = []
    for c in chunks:
        meta = c.metadata
        if parent_id is not None and str(meta.get(parent_id_key)) != str(parent_id):
            continue
        if chunk_type is not None and str(meta.get(chunk_type_key)) != str(chunk_type):
            continue
        filtered.append(c)
    return filtered
    
def enrich_context(state: RetrieverState) -> Command:
    print(f"Entering in CONTEXT ENRICHER:\n")
    parent_docs = state["parent_docs"]
    # compressed_docs = state["parent_docs"]
    child_chunks = state["child_chunks"]

    for doc in parent_docs:
        parent_id = doc.metadata[parent_id_key]
        print(f"\nSearching Table data and Image for parent id: {parent_id}")
        table_chunks = filter_chunks(child_chunks, parent_id=parent_id, chunk_type=CHUNK_TYPE.TABLE)
        image_chunks = filter_chunks(child_chunks, parent_id=parent_id, chunk_type=CHUNK_TYPE.IMAGE)

        for table_chunk in table_chunks:
            csv_file = table_chunk.metadata["source"]
            print(f"Table file: {csv_file}")
            df = read_csv(csv_file)
            sample_size = min(5, len(df))
            sample_df = df.sample(n=sample_size, random_state=42)
            doc.page_content = doc.page_content + "\n" + str(sample_df)
        
        image_file = []
        for image_chunk in image_chunks:
            image_file.append(image_chunk.metadata["source"])
            print(f"Image file: {image_file}")

        doc.metadata["image"] = ",".join(image_file)

    return Command(
        update={
            "enriched_content": parent_docs,
        },
        # goto="compress"
    )
    