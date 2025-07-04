from langgraph.graph import StateGraph, START, END
from rag.self_rag.types import State
from rag.self_rag.router import query_strategy_router
from rag.self_rag.handler import tool_handler
from rag.self_rag.constants import ROUTER, TOOL_HANDLER

overall_workflow = StateGraph(State)

overall_workflow.add_node(ROUTER, query_strategy_router)
overall_workflow.add_node(TOOL_HANDLER, tool_handler)

overall_workflow.add_edge(START, ROUTER)
overall_workflow.add_edge(ROUTER, TOOL_HANDLER)
overall_workflow.add_edge(TOOL_HANDLER, END)

agent = overall_workflow.compile()
