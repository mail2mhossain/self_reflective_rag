from pymongo import MongoClient
from typing import List, Optional
from rag.config import MONGO_URI, MONGO_DB_NAME, MONGO_PARENT_COLLECTION, parent_id_key

def list_all_parent_ids() -> list:
    """
    Connects to MongoDB and returns a list of all distinct parent_id values.
    """
    with MongoClient(MONGO_URI) as client:
        coll = client[MONGO_DB_NAME][MONGO_PARENT_COLLECTION]
        # use distinct for efficiency
        parent_ids = coll.distinct(parent_id_key)
    return parent_ids

def get_contents_by_parent_id(parent_id: str) -> str:

    with MongoClient(MONGO_URI) as client:
        coll = client[MONGO_DB_NAME][MONGO_PARENT_COLLECTION]
        doc = coll.find_one({"parent_id": parent_id}, {"_id": False, "content": True})
        content = doc["content"] if doc else None
        return content

if __name__ == "__main__":
    ids = list_all_parent_ids()
    print(f"Found {len(ids)} unique parent_id values:")
    for pid in ids:
        print(pid)
    # 0808f801-6d33-4c12-8522-9d69e7478e58
    # 0bb761f2-c34e-45a1-ae31-64a43e35305f
    # 11e995e2-92f2-4f55-a790-3af8ad600b53

    # content = get_contents_by_parent_id("0bb761f2-c34e-45a1-ae31-64a43e35305f")
    # print(content)


