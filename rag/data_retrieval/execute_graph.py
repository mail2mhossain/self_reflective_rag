from rag.data_retrieval.graph_generator import generate_graph
from rag.self_rag.answer_generator import generate

# python -m rag.data_retrieval.execute_graph

query = "What is Combine Outbox + Inbox for end-to-end safety"

inputs = {"question": query}
config = {"recursion_limit": 50}

graph = generate_graph()

output = graph.invoke(inputs, config=config)
context = output.get("enriched_content", [])

# print(f'Context: {context}')
if context:
    inputs = {"query": query, "context": context}
    config = {"recursion_limit": 50}

    state = generate(inputs)
    print (f"Answer:\n{state.update['answer']}")






