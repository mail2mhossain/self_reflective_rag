from langgraph.types import Command
from rag.self_rag.agent_state import AnswerGenerationState
from rag.data_retrieval.graph_generator import generate_graph
from rag.self_rag.constants import ANSWER_GENERATOR

# python -m rag.data_retrieval.execute_graph

def retrieve_data(state: AnswerGenerationState) -> Command:
    """
    Retrieves relevant context for the given query.

    Args:
        state (GraphState): The current state of the agent, including all keys.

    Returns:
        Command: A command object containing the retrieved context and the next step to execute.
    """
    print("Retrieving context...")
    query = state["query"]

    inputs = {"question": query}
    config = {"recursion_limit": 50}

    graph = generate_graph()

    output = graph.invoke(inputs, config=config)

    return Command(
        update={
            "context": output["compressed_docs"]
        },
        goto=ANSWER_GENERATOR
    )
