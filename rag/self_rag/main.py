from rag.self_rag.identify_rag_technique import main_agent

if __name__ == "__main__":
    app = main_agent()
    response = app.invoke({"query": "What is CQRS and Event Sourcing? How they differ?"})
    print(response["answer"])
