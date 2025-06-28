import yaml
from pathlib import Path

# assume this file lives at RAG/rag/config.py
BASE_DIR = Path(__file__).parent.parent  # -> RAG/
CONFIG_PATH = BASE_DIR / "config.yaml"


def get_config() -> dict:
    """
    Load and return the contents of config.yaml
    """
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

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
WEAVIATE_URL    = cfg.get("weaviate_url.url", "http://localhost:8080")
WEAVIATE_URL_HOST    = cfg.get("weaviate_url.url_host", "localhost")
WEAVIATE_URL_PORT    = cfg.get("weaviate_url.url_port", 8080)
WEAVIATE_SECURE      = cfg.get("weaviate_url.secure", False)


MONGO_URI               = cfg.get("mongodb.url", "mongodb://localhost:27017/")
MONGO_DB_NAME           = cfg.get("mongodb.db_name", "embedding_db")
MONGO_PARENT_COLLECTION = cfg.get("mongodb.parent_collection", "parent_chunks")