from langgraph.constants import Send
from rag.self_rag.agent_state import EachQueryState, MultiQueryAnswerState
from langgraph.types import Command
from rag.self_rag.answer_generation_agent import answer_generation_agent
from rag.self_rag.constants import  DECOMPOSED_ANSWER


def decompose_answer(state: EachQueryState) -> EachQueryState:
    """
    Generate answer based on sub_questions

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """

    print("---ANSWER BASED ON SUB-QUESTIONS ---")
    query = state["query"]
    config = {"recursion_limit": 50}

    inputs = {"query": query}
    graph = answer_generation_agent()
    response = graph.invoke(inputs, config=config)

    return Command(
        update={
            "answers": [response["answer"]],
        },
    ) 


def continue_to_decompose_answer(state: MultiQueryAnswerState) -> MultiQueryAnswerState:
    return [Send(DECOMPOSED_ANSWER, {"query": q}) for q in state["multi_queries"]]


