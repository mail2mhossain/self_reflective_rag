import re
from typing import List
from langchain_core.tools import tool
from rag.self_rag.constants import TRANSFORM_QUERY
from rag.self_rag.answer_generation_agent import generate_answer_generation_agent


graph = generate_answer_generation_agent()

@tool
def transform_query(transformed_query: str) -> str:
    """Rewrite an informal query into a concise, formal version."""
    result = graph.invoke({"query": transformed_query})
    return result["generation"]

@tool
def handle_multi_query(queries: List[str]) -> str:
    """Generate multiple variants of the original query."""
    results = [graph.invoke({"query": q}) for q in queries]
    answers = "\n".join([r["generation"] for r in results])
    return answers

@tool
def handle_decomposed_query(sub_questions: List[str]) -> str:
    """Split a complex query into simpler sub-questions."""
    return "handle_decomposed_query:\n" + "\n".join(sub_questions)

@tool
def combine_techniques(sequence: List[str], user_query: str):
    from rag.self_rag.controller import run_controller
    return run_controller(tuple(sequence), user_query)


tools=[transform_query, handle_multi_query, handle_decomposed_query, combine_techniques]
tools_by_name={tool.name: tool for tool in tools}