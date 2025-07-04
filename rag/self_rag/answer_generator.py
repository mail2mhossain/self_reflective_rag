from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langgraph.types import Command
from langchain_core.output_parsers import StrOutputParser
from rag.self_rag.agent_state import AnswerGenerationState
from rag.self_rag.constants import IS_DOCUMENT_GROUNDED
from rag.llm_config import llm


ANSWER_SYS_MSG= """
    You are a tabula rasa, a blank slate, with no prior knowledge or information.
    You are also a chatbot in a dialogue with a user
    Always answer based on provided context.
    If unsure of the answer or the context failed to give answer, 
    simply state that 'Sorry, I don't know the answer.' instead of making one up.
    """

ANSWER_USER_MSG = """
    You are a tabula rasa, a blank slate, with no prior knowledge or information.  
    You can only respond based on the exact information or context delimited by triple backticks and 
    chat history delimited by angle bracket provided in this text. 
    Do not draw from any external knowledge or make assumptions beyond what is given here.

    Context:
    ```{context}```

    Previous Conversation:
    <{chat_history}>
    (Note: Utilize the chat history only if it's relevant to the question.)

    Now, based on only the information provided above, answer the following question. 
    Answer should be details as much as possible.
    If possible write results in bullet points with proper headings. 
    {query}
    """

def generate(state: AnswerGenerationState) -> Command:
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATING ANSWER---")

    query = state["query"]
    context = state["context"]
    
    generated_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(ANSWER_SYS_MSG),
            HumanMessagePromptTemplate.from_template(ANSWER_USER_MSG),
        ]
    )
    
    rag_chain = generated_prompt | llm | StrOutputParser()

    answer = rag_chain.invoke(
        {"query": query, "context": context, "chat_history": []}
    )
    # print(f"Context: {context}")
    # print(f"Answer: {answer}")
    return Command(
        update={
            "answer": answer,
        },
        goto=IS_DOCUMENT_GROUNDED
    )
