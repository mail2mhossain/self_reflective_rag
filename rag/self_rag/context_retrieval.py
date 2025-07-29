from langgraph.types import Command
from langgraph.graph import END
from rag.self_rag.agent_state import AnswerGenerationState
from rag.data_retrieval.graph_generator import generate_graph
from rag.self_rag.constants import CONTEXT_RETRIEVAL, CONTEXT_PRECISION
from rag.config import MAX_RETRIEVAL_TRY_COUNT
from rag.self_rag.context_precision import compute_context_precision

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
    context = output.get("compressed_docs", [])

    try_count = state.get("try_count", 1)
    print(f"Context Retrieval Try Count: {try_count}")
    if context:
        return Command(
            update={
                "context": context,
                "try_count": try_count + 1,
            },
            goto=CONTEXT_PRECISION
        )
    else:
        if try_count > MAX_RETRIEVAL_TRY_COUNT:
            return Command(
                goto=END
            )
        else:
            return Command(
                update={
                    "try_count": try_count + 1,
                    },
                goto=CONTEXT_RETRIEVAL
            )
    
