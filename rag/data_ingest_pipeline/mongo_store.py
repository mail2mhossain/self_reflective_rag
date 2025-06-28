# mongo_store.py

# pip install pymongo
from pymongo import MongoClient
import datetime

from rag.config import MONGO_URI, MONGO_DB_NAME, MONGO_PARENT_COLLECTION
from rag.config import file_id_key, parent_id_key
import logging

logger = logging.getLogger(__name__)

def store_parent_docs_in_mongodb(parent_docs) -> None:
    """
    Insert a batch of parent document chunks into MongoDB.

    Each record will include file_id, parent_id, source, content, and a UTC timestamp.
    """
    if not parent_docs:
        return

    records = [
        {
            "file_id":    doc.metadata.get(file_id_key),
            "parent_id":  doc.metadata.get(parent_id_key),
            "source":     doc.metadata.get("source"),
            "content":    doc.page_content,
            "created_at": datetime.datetime.now(datetime.timezone.utc),
        }
        for doc in parent_docs
    ]

    try:
        with MongoClient(MONGO_URI) as client:
            collection = client[MONGO_DB_NAME][MONGO_PARENT_COLLECTION]
            result = collection.insert_many(records)
            logger.info(
                "Inserted %d parent chunks into MongoDB (ids: %s).",
                len(result.inserted_ids),
                result.inserted_ids,
            )
    except Exception:
        logger.exception("Failed to insert parent documents into MongoDB.")
        raise
