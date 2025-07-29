import os
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset
from langgraph.types import Command
from langgraph.graph import END
from rag.self_rag.agent_state import AnswerGenerationState
from rag.config import OPENAI_API_KEY

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def validate_answer_quality(state: AnswerGenerationState) -> Command:
    """
    Validates the quality of the generated answer.

    Args:
        state (AnswerGenerationState): The current state of the agent, including all keys.

    Returns:
        Command: A command object containing the validated answer and the next step to execute.
    """
    print("Validating answer quality...")

    scores = evaluate(
        dataset=Dataset.from_dict({
            "question": [state["query"]],
            "contexts": [[c.page_content for c in state["context"]]],
            "answer": [state["answer"]]
        }),
        metrics=[faithfulness, answer_relevancy]
    )

    print(f"Faithfulness: {scores['faithfulness']}")
    print(f"Answer Relevancy: {scores['answer_relevancy']}")

    if scores["faithfulness"][0] < 0.80 or scores["answer_relevancy"][0] < 0.75:
        return Command(
            update={
                "answer": "no_answer",
            },
            goto=END
        )
    return Command(
        update={
            "answer": state["answer"],
        },
        goto=END
    )