from typing import List, Literal, Tuple
from typing import TypedDict
from langchain_core.tools import tool
from rag.llm_config import llm
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from rag.self_rag.query_transformation import transform_user_query
from rag.self_rag.multi_query_generation import multi_query
from rag.self_rag.answer_generation_agent import answer_generation_agent
from rag.self_rag.multi_query_agent import multi_query_answer_generation_agent
from rag.self_rag.decompose_agent import generate_decompose_rag_graph


# ---------------------------------------------------------------------
# ❷  Controller
# ---------------------------------------------------------------------
Technique = Literal[
    "transform_query",
    "handle_multi_query",
    "handle_decomposed_query"
]  # used for type safety only

COMBOS: dict[Tuple[Technique, ...], str] = {
    ("transform_query",):                          "T",
    ("handle_multi_query",):                       "M",
    ("handle_decomposed_query",):                  "D",
    ("transform_query", "handle_multi_query"):     "T→M",
    ("handle_multi_query", "transform_query"):     "M→T",
    ("transform_query", "handle_decomposed_query"):   "T→D",
    ("handle_decomposed_query", "transform_query"):   "D→T",
    ("handle_multi_query", "handle_decomposed_query"): "M→D",
    ("handle_decomposed_query", "handle_multi_query"): "D→M",
    ("transform_query", "handle_multi_query", "handle_decomposed_query"): "T→M→D",
    ("transform_query", "handle_decomposed_query", "handle_multi_query"): "T→D→M",
    ("handle_multi_query", "handle_decomposed_query", "transform_query"): "M→D→T",
    ("handle_decomposed_query", "handle_multi_query", "transform_query"): "D→M→T",
    ("handle_multi_query", "transform_query", "handle_decomposed_query"): "M→T→D",
    ("handle_decomposed_query", "transform_query", "handle_multi_query"): "D→T→M",
    ("transform_query", "handle_decomposed_query", "handle_multi_query"): "T→D→M",
}

class State(TypedDict):
    query: str
    messages:MessagesState
    answer: str


def run_controller(sequence: Tuple[Technique, ...], user_query: str):
    key = COMBOS.get(sequence)

    if key == "T→M" or key == "M→T":
        transformed = transform_user_query(user_query)
        multi_queries = multi_query(transformed)      
        graph = multi_query_answer_generation_agent()
        results = graph.invoke({"multi_queries": multi_queries, "answer_generation_graph": answer_generation_agent()})
        answer = results.get("answer", "No answer found")
        return answer         

    if key == "T→D" or key == "D->T":
        transformed = transform_user_query(user_query)
        graph = generate_decompose_rag_graph()
        results = graph.invoke({"query": transformed})
        answer = results.get("answer", "No answer found")
        return answer

    if key == "M→D" or key == "D→M":
        transformed = transform_user_query(user_query)
        variants = multi_query(transformed)
        graph = multi_query_answer_generation_agent()
        results = graph.invoke({"multi_queries": variants, "answer_generation_graph": generate_decompose_rag_graph()})
        answer = results.get("answer", "No answer found")
        return answer

    if key == "T→M→D" or key == "D→M→T" or key == "M→T→D" or key == "M→D→T" or key == "D→T→M" or key == "T→D→M":
        transformed = transform_user_query(user_query)
        variants    = multi_query(transformed)
        graph = multi_query_answer_generation_agent()
        results = graph.invoke({"multi_queries": variants, "answer_generation_graph": generate_decompose_rag_graph()})
        answer = results.get("answer", "No answer found")
        return answer

    transformed = transform_user_query(user_query)
    return transform_query(transformed)


@tool
def transform_query(transformed_query: str):
    """Rewriting a user’s unclear or casual phrasing into a concise, formal query that preserves its original meaning."""
    print("Has entered transform_query")
    graph = answer_generation_agent()
    results = graph.invoke({"query": transformed_query})
    answer = results.get("answer", "No answer found")
    return answer

@tool
def handle_multi_query(queries: List[str]):
    """Generating several different but related query variants to capture documents that might use different wording or style."""
    print("Has entered handle_multi_query")
    graph = multi_query_answer_generation_agent()
    results = graph.invoke({"multi_queries": queries, "answer_generation_graph": answer_generation_agent()})
    answer = results.get("answer", "No answer found")
    return answer

@tool
def handle_decomposed_query(sub_questions: List[str]):
    """Breaking a complex or multipart query into simpler sub-questions, each of which can be retrieved and answered separately."""
    print("Has entered handle_decomposed_query")
    graph = multi_query_answer_generation_agent()
    results = graph.invoke({"multi_queries": sub_questions, "answer_generation_graph": answer_generation_agent()})
    answer = results.get("answer", "No answer found")
    return answer

@tool
def combine_techniques(sequence: List[str], user_query: str):
    """Combining multiple techniques to apply them in sequence."""
    print("Has entered combine_techniques")
    seq_tuple: Tuple[str, ...] = tuple(sequence)
    return run_controller(seq_tuple, user_query)

tools = [transform_query, handle_multi_query, handle_decomposed_query, combine_techniques]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools, tool_choice="any")


def tool_handler(state: State):
    """Performs the tool call."""
    print("--- Tool Handler ---")
    # Iterate through tool calls
    for tool_call in state["messages"][-1].tool_calls:
        # Get the tool
        tool = tools_by_name[tool_call["name"]]
        print(f"Tool: {tool_call['name']}")
        print(f"Args: {tool_call['args']}")
        # Run it
        observation = tool.invoke(tool_call["args"])
    
    return {"answer": observation}


def query_strategy_router(state: State):
    print("--- Router ---")
    prompt = f"""
    You are an expert in Retrieval-Augmented Generation (RAG). Given the user query below, choose the best technique or sequence of techniques to apply from the options provided, based on the technique comparison table.

    **Available Functions**:
    1. transform_query(transformed_query: str)
    2. handle_multi_query(queries: List[str])
    3. handle_decomposed_query(sub_questions: List[str])
    4. combine_techniques(sequence: List[str], user_query: str)  # Example: ["transform_query", "handle_multi_query"], "What is the capital of France?"

    **Technique Comparison Table**:

    | Technique                  | Use When                                                  | Outcome                                                                                                                                                                          |
    |---------------------------|-----------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
    | Query Transformation       | Query is ambiguous, vague, or uses informal language      | A clearer, more precise version of the original query without changing the intent                                                                                                |
    | Multi-Query Generation     | Query could match many possible phrasings or perspectives | Multiple semantically diverse versions of the original query to cover a broader range of possible relevant documents                                                             |
    | Query Decomposition        | Query has multiple parts or requires multi-hop reasoning  | Split into sub-questions for more focused retrieval. Breaks a complex or multi-part query into smaller sub-questions that can be answered independently and then aggregated      |

    **User Query**: "{state['query']}"

    Return only one function call in valid Python syntax.  
    If multiple techniques should be applied in sequence, use the `combine_techniques` function with a list of technique names in order.  
    Do not include any explanation or markdown.
    """

    response = llm_with_tools.invoke(prompt)

    return {"messages": [response]}


def main_agent() -> CompiledStateGraph:
    overall_workflow = StateGraph(State)

    # Add nodes
    overall_workflow.add_node("router", query_strategy_router)
    overall_workflow.add_node("tool_handler", tool_handler)

    # Add edges
    overall_workflow.add_edge(START, "router")
    overall_workflow.add_edge("router", "tool_handler")
    overall_workflow.add_edge("tool_handler", END)

    # Compile the agent
    agent = overall_workflow.compile()
    return agent

