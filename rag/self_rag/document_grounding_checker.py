from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langgraph.types import Command
from rag.self_rag.agent_state import AnswerGenerationState
from langgraph.graph import END
from rag.llm_config import llm
from rag.self_rag.constants import IS_ANSWER_RELEVANT

GENERATION_BY_CONTEXT_PROMPT = """You are a grader assessing whether an answer is grounded in / supported by a set of facts. \n 
        Here are the facts:
        \n ------- \n
        {documents} 
        \n ------- \n
        Here is the answer: {generation}
        Give a binary score 'yes' or 'no' to indicate whether the answer is grounded in / supported by a set of facts."""



def is_document_grounded(state: AnswerGenerationState) -> Command:
    """
    Determines whether the generation is grounded in the document.

    Args:
        state (dict): The current state of the agent, including all keys.

    Returns:
        Command: Command object with goto and update
    """

    print("---DOCUMENT GROUNDED CHECKER---")
    documents = state["context"]
    answer = state["answer"]

    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Supported score 'yes' or 'no'")

    llm_with_tool = llm.with_structured_output(grade)

    prompt = PromptTemplate(
        template=GENERATION_BY_CONTEXT_PROMPT,
        input_variables=["generation", "documents"],
    )

    chain = prompt | llm_with_tool 

    sscored_result = chain.invoke({"generation": answer, "documents": documents})
    grade = sscored_result.binary_score

    if grade == "yes":
        print("---DECISION: SUPPORTED, MOVE TO RELEVANT CHECKER---")
        return Command(
            update={
                "grade": grade,
            },
            goto=IS_ANSWER_RELEVANT
        )
    else:
        print("---DECISION: NOT SUPPORTED, MOVE TO END---")
        return Command(
            update={
                "answer": "no_answer",
            },
            goto=END
        )
