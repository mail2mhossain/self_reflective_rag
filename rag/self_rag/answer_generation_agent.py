import pprint
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from rag.self_rag.agent_state import AnswerGenerationState
from rag.self_rag.context_retrieval import retrieve_data
from rag.self_rag.answer_generator import generate
from rag.self_rag.document_grounding_checker import is_document_grounded
from rag.self_rag.relevant_answer_checker import is_answer_relevant
from rag.self_rag.constants import (
    CONTEXT_RETRIEVAL,
    ANSWER_GENERATOR,
    IS_DOCUMENT_GROUNDED,
    IS_ANSWER_RELEVANT,
)


def answer_generation_agent() -> CompiledStateGraph:
    workflow = StateGraph(AnswerGenerationState)

    workflow.add_node(CONTEXT_RETRIEVAL, retrieve_data)
    workflow.add_node(ANSWER_GENERATOR, generate)
    workflow.add_node(IS_DOCUMENT_GROUNDED, is_document_grounded)
    workflow.add_node(IS_ANSWER_RELEVANT, is_answer_relevant)
    
    workflow.set_entry_point(CONTEXT_RETRIEVAL)
    
    graph = workflow.compile()
    # try:
    #     graph.get_graph(xray=1).draw_mermaid_png(output_file_path="total_rag_graph.png")
    # except ValueError as e:
    #     print(f"Failed to generate graph image: {e}")
    #     print("Consider using an alternative visualization method or checking your network connection.")

    return graph


if __name__ == "__main__":
    app = answer_generation_agent()
    query = "Explain Strategic Design of DDD"
    results = app.invoke({"query": query})

    print(f"Results: {results['answer']}")