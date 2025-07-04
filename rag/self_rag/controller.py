from rag.self_rag.types import Technique, COMBO_KEYS
from rag.self_rag.query_transformation import transform_query
from rag.self_rag.multi_query_generation import multi_query
from rag.self_rag.answer_generation_agent import generate_answer_generation_agent

COMBOS: dict[COMBO_KEYS, str] = { ... }

def run_controller(sequence: COMBO_KEYS, user_query: str):
    key = COMBOS.get(sequence)
    if key is None:
        raise ValueError(f"Unsupported technique order: {sequence}")

    if key == "T→M":
        transformed   = transform_query(user_query)
        multi_queries = multi_query(transformed)      
        answers =  [generate_answer_generation_agent(q) for q in multi_queries]    
        return answers     

    if key == "T→D":
        transformed = transform_query(user_query)
        return decompose_answer_generation(transformed)

    if key == "M→D":
        variants = multi_query(user_query)
        return [decompose_answer_generation(q) for q in variants]

    if key == "T→M→D":
        transformed = transform_query(user_query)
        variants    = multi_query(transformed)
        return [decompose_answer_generation(q) for q in variants]
