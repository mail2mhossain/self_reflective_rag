import operator
from typing import Annotated, List, TypedDict
from langgraph.graph.state import CompiledStateGraph

class AnswerGenerationState(TypedDict):
    query: str
    context: list[any]
    answer: str

class _BaseAnswerState(TypedDict):
    answers: Annotated[list[str], operator.add]
    answer: str

class MultiQueryAnswerState(_BaseAnswerState):
    multi_queries: list[str]          
    answer_generation_graph: CompiledStateGraph

class DecomposedAnswerState(MultiQueryAnswerState):
    query: str     


class EachQueryState(TypedDict):
    query: str
    answer_generation_graph: CompiledStateGraph
   