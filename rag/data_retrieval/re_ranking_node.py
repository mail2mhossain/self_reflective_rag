import os
import sys
import numpy as np
from sentence_transformers import CrossEncoder
from langgraph.types import Command
from langgraph.graph import END
from rag.data_retrieval.retriever_state import RetrieverState
from rag.config import parent_id_key


def reciprocal_rank_fusion(state:RetrieverState) -> Command:
    print(f"Entering in RECIPROCAL RE-RANKING:\n")
    context = state["keys"]
    query = context["query"]
    child_dox = context["child_dox"]

    k = 60

    fused_scores = {}

    for rank, doc in enumerate(child_dox):
        doc_id = doc.metadata["doc_id"]

        if doc_id not in fused_scores:
            fused_scores[doc_id] = 0

        fused_scores[doc_id] += 1 / (rank + k)

        reranked_results = [
            (doc_id, score)
            for doc_id, score in sorted(
                fused_scores.items(), key=lambda x: x[1], reverse=True
            )
        ]
    dox = reranked_results[:5]
    parent_ids = [result[0] for result in dox]

    parent_ids = list(set(parent_ids))
    return {
        "keys": {
            "query": query,
            "parent_ids": parent_ids,
        }
    }


def bge_reranker(state:RetrieverState) -> Command:
    # https://python.langchain.com/v0.1/docs/integrations/document_transformers/cross_encoder_reranker/
    from langchain.retrievers.document_compressors import CrossEncoderReranker
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder

    print(f"Entering in CROSS ENCODER RE-RANKING:\n")
    context = state["keys"]
    query = context["query"]
    child_dox = context["child_dox"]

    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=model, top_n=3)

    top_ranked_dox = compressor.compress_documents(child_dox, query)
    parent_ids = [(result.metadata["doc_id"]) for result in top_ranked_dox]

    parent_ids = list(set(parent_ids))

    return {
        "keys": {
            "query": query,
            "parent_ids": parent_ids,
        }
    }


cross_encoder = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device="cuda",        
    # low_cpu_mem_usage=True,   # helps avoid meta tensors on CPU
    max_length=512
)  # cross-encoder/stsb-roberta-base

def cross_encoder_re_rank(state:RetrieverState) -> Command:
    print(f"Entering in CROSS ENCODER RE-RANKING:\n")
    
    query = state["question"]
    all_dox = state["child_chunks"] + state["qa_chunks"]

    if len(all_dox) == 0:
        return Command(
            update={
                "parent_ids": [],
            },
            goto=END
        )
    pairs = [[query, doc.page_content] for doc in all_dox]

    scores = cross_encoder.predict(pairs)

    sorted_indices = np.argsort(-scores)

    top_ranked_dox = []
    for i in sorted_indices[:10]:
        top_ranked_dox.append(all_dox[i])

    parent_ids = [(str(result.metadata[parent_id_key])) for result in top_ranked_dox]

    parent_ids = list(set(parent_ids))
    print(f"Total parent ids found: {len(parent_ids)}")
    return Command(
        update={
            "parent_ids": parent_ids,
        },
        goto="parent"
    )
