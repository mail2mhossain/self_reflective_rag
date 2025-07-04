from typing import TypedDict, Tuple, Literal, List
from langgraph.graph import MessagesState

Technique = Literal["transform_query", "handle_multi_query", "handle_decomposed_query"]

class State(TypedDict):
    query: str
    messages: MessagesState
    answer: str

COMBO_KEYS = Tuple[Technique, ...]

# COMBOS: Dict[Tuple[Technique, ...], str] = {
#     ("transform_query",): "T",
#     ("handle_multi_query",): "M",
#     ("handle_decomposed_query",): "D",
#     ("transform_query", "handle_multi_query"): "T→M",
#     # …etc.
# }
