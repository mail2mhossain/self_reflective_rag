from langchain_openai import ChatOpenAI
from langchain.retrievers.document_compressors import LLMChainExtractor
from concurrent.futures import ThreadPoolExecutor, as_completed
from langgraph.types import Command
from rag.retrieval.retriever_state import RetrieverState
from rag.config import GPT_MODEL, OPENAI_API_KEY


llm = ChatOpenAI(model_name=GPT_MODEL, temperature=0, openai_api_key=OPENAI_API_KEY, cache=False)

def compress_context(state: RetrieverState) -> Command:
    print(f"Entering in CONTEXT COMPRESSOR:\n")
    query = state["question"]
    parent_docs = state["parent_docs"]

    compressor = LLMChainExtractor.from_llm(llm=llm)

    def compress_document(doc):
        return compressor.compress_documents([doc], query)

    compressed_docs = []

    # Use ThreadPoolExecutor to compress documents in parallel
    with ThreadPoolExecutor() as executor:
        # Create a future for each document compression task
        future_to_doc = {
            executor.submit(compress_document, doc): doc for doc in parent_docs
        }

        for future in as_completed(future_to_doc):
            actual = future.result()
            # print(f"ACTUAL Compressed: {actual}")
            if len(actual) > 0:
                compressed_docs.extend(actual)


    return Command(
        update={
            "compressed_docs": compressed_docs,
        },
    )
