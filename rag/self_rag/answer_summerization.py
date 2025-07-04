from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langgraph.types import Command
from rag.self_rag.constants import END
from rag.self_rag.agent_state import MultiQueryAnswerState
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from rag.llm_config import llm


SUMMARIZATION_SYS_MSG = """
    Summarize the text in professional tone using plain and simple English language in the following format: 
    - Create a separate section for each topic.
    - For each section, summarize the key points from the answers as bullet points with proper headings.
    - Take your time and do it thoroughly. Be very detailed in your answers and do not skip any key points.
    - Include all provided urls in your writing if any
"""

def summarize_answers(state: MultiQueryAnswerState) -> Command:
    print("---SUMMARIZE ANSWERS ---")
    
    answers = state["answers"]
    # print(f"Answers: {answers}")
    user_msg = """Content: {answers}.
    """
    chain = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(SUMMARIZATION_SYS_MSG),
            HumanMessagePromptTemplate.from_template(user_msg),
        ]
    ) | llm | StrOutputParser()

    answer = chain.invoke(
        {"answers": answers}
    )
    return Command(
        update={
            "answer": answer,
        },
    )