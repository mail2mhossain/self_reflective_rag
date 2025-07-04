from rag.data_retrieval.graph_generator import generate_graph

# python -m rag.data_retrieval.execute_graph

query = "Explain Strategic Design of DDD"

inputs = {"question": query}
config = {"recursion_limit": 50}

graph = generate_graph()

output = graph.invoke(inputs, config=config)

print(f'Context: {output["compressed_docs"]}')
