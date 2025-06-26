# python -m dox_pipeline.config

from .read_config_file import get_config

cfg = get_config()

file_id_key = cfg['dox_metadata']["file_id_key"]
parent_id_key = cfg['dox_metadata']["parent_id_key"]
encoding = cfg['dox_metadata']["encoding"]

# your OpenAI constants…
OPENAI_API_KEY = cfg['open_ai']["key"]
GPT_MODEL      = cfg['open_ai']["model"]

# vector index names
PARENT_INDEX  = cfg['vector_index']["parent_index"]
CHILD_INDEX   = cfg['vector_index']["child_index"]
QA_INDEX      = cfg['vector_index']["qa_index"]

# embedding model and weaviate URL
EMBEDDING_MODEL = cfg["embedding_model"]
WEAVIATE_URL    = cfg.get("weaviate_url", "http://localhost:8080")


MONGO_URI               = cfg.get("mongodb.url", "mongodb://localhost:27017/")
MONGO_DB_NAME           = cfg.get("mongodb.db_name", "embedding_db")
MONGO_PARENT_COLLECTION = cfg.get("mongodb.parent_collection", "parent_chunks")