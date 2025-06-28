

## Weaviate

### Installation

```bash
docker pull semitechnologies/weaviate:latest

docker run -d --name weaviate -p 8080:8080 -p 50051:50051 -e QUERY_DEFAULTS_LIMIT=20 -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true -e PERSISTENCE_DATA_PATH=/var/lib/weaviate -v Z:\weaviate-data:/var/lib/weaviate semitechnologies/weaviate:latest
```


### Connection

```python
WEAVIATE_URL = "http://localhost:8080"

weaviate_client = weaviate.connect_to_local(
    skip_init_checks=True
)
```

## MongoDB

### Installation

```bash
docker pull mongo

docker run -d --name mongodb -p 27017:27017 -v Z:\mongodb-data:/data/db mongo
```

### Connection

```python
from pymongo import MongoClient

MONGODB_URL = "mongodb://localhost:27017"

client = MongoClient(MONGODB_URL)
db = client["mydatabase"]
```
