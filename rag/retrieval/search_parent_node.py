from numpy import concatenate
from pymongo import MongoClient
from typing import List, Optional
from langgraph.types import Command
from rag.retrieval.retriever_state import RetrieverState
from langchain_core.documents import Document
from rag.config import MONGO_URI, MONGO_DB_NAME, MONGO_PARENT_COLLECTION, parent_id_key

def get_contents_by_parent_id(state: RetrieverState) -> Command:
    print(f"Entering in PARENT SEARCH:\n")
    with MongoClient(MONGO_URI) as client:
        coll = client[MONGO_DB_NAME][MONGO_PARENT_COLLECTION]
        cursor = coll.find(
            { parent_id_key: { "$in": state["parent_ids"] } },
            {
                "_id": False,
                parent_id_key: True,
                "content": True,
                "source": True
            }
        )
        # print(f"Total documents found: {cursor.count()}")
        documents = [
            Document(
                page_content=doc["content"],
                metadata={
                    parent_id_key: doc.get(parent_id_key),
                    "source": doc.get("source"),
                }
            )
            for doc in cursor
        ]
        print(f"Total parent docs found: {len(documents)}")
        return Command(
            update={
                "parent_docs": documents,
            },
            goto="compress"
        )

if __name__ == "__main__":
    parent_ids = ['0808f801-6d33-4c12-8522-9d69e7478e58', '0bb761f2-c34e-45a1-ae31-64a43e35305f', '11e995e2-92f2-4f55-a790-3af8ad600b53']
    parent_ids = ['0e2f11f2-424d-4305-9563-0c3c28d851d3', '15d8e8b1-648e-40a1-afc4-b615cbca1f98', '2282134f-ffae-4335-8289-3b4b6997619c']
    # 0e2f11f2-424d-4305-9563-0c3c28d851d3
    # 15d8e8b1-648e-40a1-afc4-b615cbca1f98
    # 2282134f-ffae-4335-8289-3b4b6997619c    
    content = get_contents_by_parent_id({"parent_ids": parent_ids})
    print(content)