import weaviate
from .config import (
    CHILD_INDEX, QA_INDEX,
    EMBEDDING_MODEL, WEAVIATE_URL,
    MONGO_URI, MONGO_DB_NAME, MONGO_PARENT_COLLECTION
)

weaviate_client = weaviate.connect_to_local(
    skip_init_checks=True
)

weaviate_client.close()

from pymongo import MongoClient

with MongoClient(MONGO_URI) as client:
    db = client[MONGO_DB_NAME]
    collection = db[MONGO_PARENT_COLLECTION]

    for doc in collection.find():
        print(doc)