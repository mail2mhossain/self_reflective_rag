import pprint
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from rag.self_rag.agent_state import DecomposedAnswerState
from rag.self_rag.query_decomposition import decompose_question
from rag.self_rag.answer_summerization import summarize_answers
from rag.self_rag.decompose_answer import decompose_answer, continue_to_decompose_answer
from rag.self_rag.constants import (
    DECOMPOSE_QUERY,
    DECOMPOSED_ANSWER,
    SUMMARIZE_ANSWER
)


def generate_decompose_rag_graph() -> CompiledStateGraph:
    workflow = StateGraph(DecomposedAnswerState)

    workflow.add_node(DECOMPOSE_QUERY, decompose_question)
    workflow.add_node(DECOMPOSED_ANSWER, decompose_answer)
    workflow.add_node(SUMMARIZE_ANSWER, summarize_answers)
    
    workflow.set_entry_point(DECOMPOSE_QUERY)
    workflow.add_conditional_edges(DECOMPOSE_QUERY, continue_to_decompose_answer, [DECOMPOSED_ANSWER])
    workflow.add_edge(DECOMPOSED_ANSWER, SUMMARIZE_ANSWER)
    workflow.add_edge(SUMMARIZE_ANSWER, END)

    graph = workflow.compile()

    # try:
    #     graph.get_graph(xray=1).draw_mermaid_png(output_file_path="decompose_rag_graph.png")
    # except ValueError as e:
    #     print(f"Failed to generate graph image: {e}")
    #     print("Consider using an alternative visualization method or checking your network connection.")  
    
    return graph


if __name__ == "__main__":
    app = generate_decompose_rag_graph()
    query = "Explain Strategic Design of DDD"
    results = app.invoke({"query": query})

    # print(f"Context: {results['decompose_answers']}\n")
    print(f"Results: {results['answer']}")