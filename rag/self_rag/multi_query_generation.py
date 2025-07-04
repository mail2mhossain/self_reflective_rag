from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from rag.llm_config import llm

MULTI_QUERY_SYS_MSG = """
    You are a helpful assistant that generates multiple search queries based on a single input query.
    Your task is to generate different versions of the given user question to retrieve relevant documents from a vector database. 
    By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search.
    Suggest up to {num_of_Query} additional related questions to help them find the information they need, for the provided question.
    Suggest only short questions without compound sentences. Suggest a variety of questions that cover different aspects of the topic.
    Make sure they are complete questions, and that they are related to the original question.
    Output one question per line. Do not number the questions.
    """

def multi_query(query: str) -> list[str]:
    """
    Generate multiple questions based on rephrased question.

    Args:
        query (str): The current graph state

    Returns:
        list[str]: Generated questions
    """

    print("---MULTI QUERY---")

    user_msg = """Content: {question}.
    """
    query_generation_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(MULTI_QUERY_SYS_MSG),
            HumanMessagePromptTemplate.from_template(user_msg),
        ]
    )

    query_generation_chain = (
        query_generation_prompt | llm | StrOutputParser() | (lambda x: x.split("\n"))
    )

    generated_questions = query_generation_chain.invoke(
        {"question": query, "num_of_Query": 5}
    )

    print(f"Generated Question: {generated_questions}")

    return generated_questions
