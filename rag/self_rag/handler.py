from rag.self_rag.types import State
from rag.self_rag.tools import tools_by_name

def tool_handler(state: State) -> dict:
    """Execute the tool calls emitted by the LLM."""
    last_msg = state["messages"][-1]
    for call in last_msg.tool_calls:
        tool = tools_by_name[call["name"]]
        observation = tool.invoke(call["args"])
    return {"answer": observation}
