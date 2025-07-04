from langchain_openai import ChatOpenAI
from rag.config import GPT_MODEL, OPENAI_API_KEY


llm = ChatOpenAI(
    model_name=GPT_MODEL,
    openai_api_key=OPENAI_API_KEY,
    cache=False
)