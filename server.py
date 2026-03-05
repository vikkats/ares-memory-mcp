from fastapi import FastAPI
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import uuid
import os
import requests

app = FastAPI()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

COLLECTION = "ares_memory"

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# create collection if missing
try:
    qdrant.get_collection(COLLECTION)
except:
    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )


def embed(text):

    r = requests.post(
        "https://openrouter.ai/api/v1/embeddings",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "text-embedding-3-small",
            "input": text
        }
    )

    return r.json()["data"][0]["embedding"]


@app.get("/")
def root():
    return {"name": "ares-memory-mcp"}


@app.post("/tools/store_memory")
def store_memory(text: str):

    vector = embed(text)

    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=vector,
        payload={"text": text}
    )

    qdrant.upsert(
        collection_name=COLLECTION,
        points=[point]
    )

    return {"status": "stored", "memory": text}


@app.post("/tools/search_memory")
def search_memory(query: str):

    vector = embed(query)

    results = qdrant.search(
        collection_name=COLLECTION,
        query_vector=vector,
        limit=5
    )

    return [r.payload["text"] for r in results]


@app.get("/tools/list_memories")
def list_memories():

    points, _ = qdrant.scroll(
        collection_name=COLLECTION,
        limit=50
    )

    return [p.payload["text"] for p in points]


@app.post("/tools/delete_memory")
def delete_memory(id: str):

    qdrant.delete(
        collection_name=COLLECTION,
        points_selector=[id]
    )

    return {"status": "deleted"}
