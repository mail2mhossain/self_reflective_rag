from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts.chat import ChatPromptTemplate
from rag.llm_config import llm
from langgraph.types import Command
from rag.self_rag.agent_state import DecomposedAnswerState

DECOMPOSED_QUERY_PROMPT = """You are a helpful assistant that generates multiple sub-questions related to an input question in different context. \n
    The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
    Generate multiple sub-problems related to: {question} in different context\n
    Output (5 queries):"""

def generate_decompose_question(query: str) -> list[str]:
    """
    Decompose the query to produce a better question.

    Args:
        query (str): The current graph state

    Returns:
        list[str]: Sub-questions
    """

    prompt_decomposition = ChatPromptTemplate.from_template(DECOMPOSED_QUERY_PROMPT)

    generate_queries_decomposition = (
        prompt_decomposition | llm | StrOutputParser() | (lambda x: x.split("\n"))
    )
    sub_questions = generate_queries_decomposition.invoke(
        {"question": query}
    )
    # print(f"Sub-Questions: {sub_questions}")
    return sub_questions


def decompose_question(state:DecomposedAnswerState) -> Command:
    print("---DECOMPOSE QUESTIONS ---")
    query = state["query"]
    sub_questions = generate_decompose_question(query)
    return Command(
        update={
            "multi_queries": sub_questions,
        },
    )