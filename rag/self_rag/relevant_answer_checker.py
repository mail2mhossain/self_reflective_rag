from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from langchain.prompts import PromptTemplate
from rag.self_rag.agent_state import AnswerGenerationState
from rag.llm_config import llm
from langgraph.graph import END

GENERATION_BY_QUERY_PROMPT = """You are a grader assessing whether an answer is useful to resolve a question. \n 
        Here is the answer:
        \n ------- \n
        {generation} 
        \n ------- \n
        Here is the question: {question}
        Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question."""



def is_answer_relevant(state: AnswerGenerationState) -> Command:
    """
    Determines whether the generation addresses the question.

    Args:
        state (dict): The current state of the agent, including all keys.

    Returns:
        str: Binary decision
    """

    print("---ANSWER RELEVANT CHECKER---")
    # print_state(state)
   
    question = state["query"]
    answer = state["answer"]


    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Useful score 'yes' or 'no'")

    llm_with_tool = llm.with_structured_output(grade)

    prompt = PromptTemplate(
        template=GENERATION_BY_QUERY_PROMPT,
        input_variables=["generation", "question"],
    )

    chain = prompt | llm_with_tool 

    score = chain.invoke({"generation": answer, "question": question})
    grade = score.binary_score

    if grade == "yes":
        print("---DECISION: USEFUL, MOVE TO END---")
        return Command(
            update={
                "answer": answer,
            },
            goto=END
        )
    else:
        print("---DECISION: NOT USEFUL, MOVE TO END---")
        return Command(
            update={
                "answer": "no_answer",
            },
            goto=END
        )
        
