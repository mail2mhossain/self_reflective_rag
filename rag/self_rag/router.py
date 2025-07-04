from rag.self_rag.tools import tools
from rag.llm_config import llm

def query_strategy_router(state):
    """Pick the best tool (or sequence) based on the user’s query."""
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

    llm_with_tools = llm.bind_tools(tools, tool_choice="any")
    response = llm_with_tools.invoke(prompt)
    return {"messages": [response]}
