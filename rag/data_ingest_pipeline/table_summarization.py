from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from rag.config import OPENAI_API_KEY, GPT_MODEL

prompt_text = """
    You are an expert assistant tasked with analyzing and summarizing tabular data.

    Below is a description of the table extracted from a document:

    {table_text}

    Using this data, please provide:

    1. A concise summary of the key information.
    2. Any notable trends, comparisons, or insights.
    3. Highlights of important values or outliers.

    Be clear, informative, and keep the summary relevant to the table content.
"""


def generate_table_summaries(table_content: str) -> str:
    llm = ChatOpenAI(model_name=GPT_MODEL, temperature=0, openai_api_key=OPENAI_API_KEY)
    prompt = PromptTemplate.from_template(prompt_text)
    chain = prompt | llm | StrOutputParser()
    
    return chain.invoke({"table_text": table_content})
    