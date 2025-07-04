from langgraph.graph import START, END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from rag.self_rag.agent_state import MultiQueryAnswerState
from rag.self_rag.multi_query_answer import multi_query_answer, continue_to_multi_query_answer
from rag.self_rag.answer_summerization import summarize_answers
from rag.self_rag.constants import (
    MULTI_QUERY_ANSWER,
    CONTINUE_TO_MULTI_QUERY_ANSWER,
    SUMMARIZE_ANSWER
)


def multi_query_answer_generation_agent() -> CompiledStateGraph:
    workflow = StateGraph(MultiQueryAnswerState)

    workflow.add_node(MULTI_QUERY_ANSWER, multi_query_answer)
    workflow.add_node(CONTINUE_TO_MULTI_QUERY_ANSWER, continue_to_multi_query_answer)
    workflow.add_node(SUMMARIZE_ANSWER, summarize_answers)
    
    workflow.add_conditional_edges(START, continue_to_multi_query_answer, [MULTI_QUERY_ANSWER])
    workflow.add_edge(MULTI_QUERY_ANSWER, SUMMARIZE_ANSWER)
    workflow.add_edge(SUMMARIZE_ANSWER, END)
    

    graph = workflow.compile()
    # try:
    #     graph.get_graph(xray=1).draw_mermaid_png(output_file_path="total_rag_graph.png")
    # except ValueError as e:
    #     print(f"Failed to generate graph image: {e}")
    #     print("Consider using an alternative visualization method or checking your network connection.")

    return graph


if __name__ == "__main__":
    from rag.self_rag.multi_query_generation import multi_query
    from rag.self_rag.answer_generation_agent import answer_generation_agent
    app = multi_query_answer_generation_agent()
    query = "Explain Strategic Design of DDD"
    queries = multi_query({"query": query})
    print(f"Generated Queries: {queries}")
    results = app.invoke({"multi_queries": queries, "answer_generation_graph": answer_generation_agent()})

    print(f"Results: {results['answer']}")