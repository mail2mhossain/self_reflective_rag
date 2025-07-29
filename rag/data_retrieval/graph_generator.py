from langgraph.graph import StateGraph
from rag.data_retrieval.retriever_state import RetrieverState
from rag.data_retrieval.search_child_node import get_child_chunks
from rag.data_retrieval.search_qa_node import get_qa_chunks
from rag.data_retrieval.re_ranking_node import cross_encoder_re_rank
from rag.data_retrieval.search_parent_node import get_contents_by_parent_id
from rag.data_retrieval.contextual_compressor_node import compress_context
from rag.data_retrieval.context_enricher_node import enrich_context

def generate_graph():
    workflow = StateGraph(RetrieverState)

    workflow.add_node("child", get_child_chunks)
    workflow.add_node("qa", get_qa_chunks)
    workflow.add_node("re_ranking", cross_encoder_re_rank)
    workflow.add_node("parent", get_contents_by_parent_id)
    workflow.add_node("enrich", enrich_context)
    workflow.add_node("compress", compress_context)

    workflow.set_entry_point("child")
    workflow.set_entry_point("qa")

    workflow.set_finish_point("enrich")
    # memory = SqliteSaver.from_conn_string(":memory:")
    chain = workflow.compile()
    # chain = workflow.compile(checkpointer=memory, interrupt_before=["save"])
    return chain
