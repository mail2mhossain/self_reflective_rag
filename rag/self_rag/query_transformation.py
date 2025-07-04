from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from rag.llm_config import llm


REPHRASED_QUERY_PROMPT = """You are generating questions that is well optimized for retrieval. \n
        Look at the input and try to reason about the underlying sematic intent / meaning. \n
        Here is the initial question:
        \n ------- \n
        {query}
        \n ------- \n
        Rephrased the question: """

def transform_user_query(query: str) -> str:   
    """
    Re-write the query and transforms query to produce a better question.

    Args:
        query (str): The current graph state

    Returns:
        rephrased_query (str): Updates question, query type and rephrased_question keys
    """

    # state (dict): Updates question key with a re-phrased question
    print("---TRANSFORM QUERY---")

    rephrased_query_prompt = PromptTemplate(
        template=REPHRASED_QUERY_PROMPT,
        input_variables=["query"],
    )

    rephrased_chain = rephrased_query_prompt | llm | StrOutputParser()
    rephrased_question = rephrased_chain.invoke({"query": query})

    print(f"Original query: {query}")
    print(f"Rephrased query: {rephrased_question}")

    return rephrased_question
