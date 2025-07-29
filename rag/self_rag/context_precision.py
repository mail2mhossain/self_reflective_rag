import numpy as np
from sentence_transformers import SentenceTransformer
from langgraph.types import Command
from langgraph.graph import END
from rag.self_rag.agent_state import AnswerGenerationState
from rag.self_rag.constants import CONTEXT_RETRIEVAL, ANSWER_GENERATOR
from rag.config import MAX_RETRIEVAL_TRY_COUNT

_EMBED_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def compute_context_precision(state: AnswerGenerationState) -> Command:
    """
    Quick, reference‑free context‑precision.

    Args:
        state (AnswerGenerationState): The current state of the agent, including all keys.

    Returns:
        Command: A command object containing the computed context precision and the next step to execute.
    """

    print("Computing context precision...")
    context = state["context"]
    context = [c.page_content for c in context]
    try_count = state.get("try_count", 0)

    # 1️⃣  Embed query and contexts
    q_vec = _EMBED_MODEL.encode(state["query"], convert_to_numpy=True, normalize_embeddings=True)
    ctx_vecs = _EMBED_MODEL.encode(context, convert_to_numpy=True, normalize_embeddings=True)

    # 2️⃣  Compute cosine similarities
    sims = np.dot(ctx_vecs, q_vec)
    # 3️⃣  Count how many chunks are 'relevant'
    relevant = (sims >= 0.4).sum()
    # 4️⃣  Precision = relevant / total
    cp = float(relevant) / len(context)
    if cp < 0.50:         
        if try_count > MAX_RETRIEVAL_TRY_COUNT:
            return Command(
                goto=END
            )
        return Command(
            goto=CONTEXT_RETRIEVAL
        )
    else:
        return Command(
            goto=ANSWER_GENERATOR
        )