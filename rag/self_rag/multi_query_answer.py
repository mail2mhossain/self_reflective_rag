from operator import itemgetter
from langgraph.constants import Send
from langchain_core.output_parsers import StrOutputParser
from rag.self_rag.agent_state import MultiQueryAnswerState, EachQueryState
from langgraph.types import Command
from rag.self_rag.constants import MULTI_QUERY_ANSWER



def multi_query_answer(state: EachQueryState) -> EachQueryState:
    """
    Generate answer based on sub_questions

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """

    print("---MULTI QUERY ANSWER ---")
   
    query = state["query"]
    config = {"recursion_limit": 50}

    inputs = {"query": query}
    graph = state["answer_generation_graph"]
    response = graph.invoke(inputs, config=config)
   
    return Command(
        update={
            "answers": [response["answer"]],
        },
    ) 


def continue_to_multi_query_answer(state: MultiQueryAnswerState) -> MultiQueryAnswerState:
    return [Send(MULTI_QUERY_ANSWER, {"query": q, "answer_generation_graph": state["answer_generation_graph"]}) for q in state["multi_queries"]]




