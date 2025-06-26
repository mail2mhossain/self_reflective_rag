from pymongo import MongoClient
from .config import MONGO_URI, MONGO_DB_NAME, MONGO_PARENT_COLLECTION, parent_id_key

def ensure_indexes():
    """Run once at application startup to create any needed indexes."""
    with MongoClient(MONGO_URI) as client:
        coll = client[MONGO_DB_NAME][MONGO_PARENT_COLLECTION]
        coll.create_index(
            [(parent_id_key, 1)],
            unique=True,
            name="unique_parent_id"
        )

if __name__ == "__main__":
    ensure_indexes()
