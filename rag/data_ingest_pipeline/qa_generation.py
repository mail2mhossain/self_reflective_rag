from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser
from typing import List, Optional
from pydantic import BaseModel, Field
from rag.config import OPENAI_API_KEY, GPT_MODEL


class QA_Pairs(BaseModel):
    """Extracted data about a question-answer."""

    question: Optional[str]  = Field(..., description="Generated question.")
    answer: Optional[str]  = Field(..., description="Generated answer.")


class QA_Data(BaseModel):
    """Extracted data about question-answers."""
  
    qa_pairs: List[QA_Pairs]



json_example = [
    {
        "Question": "What is the primary goal of cdQA?",
        "Answer": "The mission of cdQA is to allow anyone to ask a question in natural language and get an answer without having to read the internal documents relevant to the question.",
    },
    {
        "Question": "Why is searching in internal document databases often frustrating?",
        "Answer": "Searching in internal document databases is often frustrating because it is not as fast, accurate, and intuitive as the search-engines provided by big tech companies.",
    },
    {
        "Question": "What is the belief of the creators regarding modern search technologies?",
        "Answer": "They believe that everyone should be able to use modern search technologies to find information in their own documents.",
    },
    {
        "Question": "What is the cdQA suite about?",
        "Answer": "Everything you need to build a closed domain question answering system.",
    },
    {
        "Question": "What can the Python Library offered by cdQA help you do?",
        "Answer": "Convert your dataset to a compatible format, create your question answering model in minutes, and deploy your model as a service.",
    },
    {
        "Question": "List some of the features of the cdQA suite.",
        "Answer": "Free, Easy, High level framework, Private, Run it offline, Flexible, Smart, Open source, Works on CPU & GPU, Uses powerful AI, Research by community.",
    },
]

sys_msg = """
You are a teacher who will generate questions and its answers based on provided context.
"""

user_msg = """
    You are a teacher coming up with questions to ask on a quiz. 
    Given the following document delimited by three backticks please generate questions based on that document.
    Based on your assessment, determine the number of questions that would comprehensively cover the document's information. 
    A question should be concise and based explicitly on the document's information. It should be asking about one thing at a time.
    Try to generate a question that can be answered by the whole document not just an individual sentence.
    
    When formulating a question, don't reference the provided document or say "from the provided context", 
    "as described in the document", "according to the given document" or anything similar.
    

    Example of json format is given below delimited by three #.

    json format:
    ###{json_example}###

    ```
    {page_content}
    ```
"""


user_prompt_template = PromptTemplate(
    template=user_msg, input_variables=["page_content"], partial_variables={"json_example": json_example}
)

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(sys_msg),
        HumanMessagePromptTemplate(prompt=user_prompt_template),
    ]
)

# print(prompt.format(page_content="page_content"))

def generate_qa(page_content: str) -> QA_Data:
    llm = ChatOpenAI(model_name=GPT_MODEL, temperature=0, openai_api_key=OPENAI_API_KEY)

    chain = prompt | llm.with_structured_output(
        schema=QA_Data,
        method="function_calling",
        include_raw=False,
    )

    results = chain.invoke({"page_content": page_content})

    return results






# pdf_file_path ="../Data/PLC_mediumArticle.pdf"
# loader = PyPDFLoader(pdf_file_path)
# pdf_text = loader.load()
# pdf_text = [d.page_content for d in pdf_text]

# pdf_text = ' '.join(pdf_text)

# # hash_value = generate_hash(pdf_text)
# # print(f"Hash: {hash_value}")

# results = generate_qa(pdf_text)

# print(f"Total QA: {len(results.qa_pairs)}\n")
# print(f"{results.qa_pairs[0].question}\n")
# print(f"{results.qa_pairs[0].answer}\n")

# json_data = results.json()
# print(f"Data Type: {type(json_data)}\n")
# print(json_data)