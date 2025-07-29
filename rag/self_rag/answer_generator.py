import base64
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.messages import HumanMessage
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
    If attached table data is relevant to the question, 
    answer should reflect the table data and include table data as markdown table in appropriate place in the answer.
    If attached image is relevant to the question, 
    answer should reflect the image and include image path from source as markdown link in appropriate place in the answer.
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
    
    # Modified to support multiple images
    image_urls = []
    for doc in context:
        urls = doc.metadata.get("image", None)
        if urls:
            # Split by comma if multiple images are provided in a single metadata field
            url_list = urls.split(",")
            for url in url_list:
                url = url.strip()
                if url not in image_urls:  # Avoid duplicates
                    image_urls.append(url)
   
    if image_urls:
        print(f"Prompt with {len(image_urls)} Images")
        message_content = [
            {"type": "text", "text": ANSWER_USER_MSG.format(context=context, chat_history="", query=query)},
        ]
        
        # Add all images to the message content
        for url in image_urls:
            try:
                with open(url, "rb") as f:
                    image_base64 = base64.b64encode(f.read()).decode("utf-8")
                formatted_image_url = f"data:image/jpeg;base64,{image_base64}"
                message_content.append(
                    {"type": "image_url", "image_url": {"url": formatted_image_url}}
                )
            except Exception as e:
                print(f"Error processing image {url}: {str(e)}")
        
        message = HumanMessage(content=message_content)
        
        generated_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(ANSWER_SYS_MSG),
                message
            ]
        )
    else:
        generated_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(ANSWER_SYS_MSG),
                HumanMessagePromptTemplate.from_template(ANSWER_USER_MSG),
            ]
        )

    rag_chain = generated_prompt | llm | StrOutputParser()

    answer = rag_chain.invoke(
        {"query": query, "context": context, "chat_history": ""}
    )
    # print(f"Context: {context}\n")
    # print(f"Answer: {answer}")
    return Command(
        update={
            "answer": answer,
        },
        goto=IS_DOCUMENT_GROUNDED
    )
