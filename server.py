from fastapi import FastAPI, HTTPException
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

qdrant = None


@app.on_event("startup")
def startup_event():
    global qdrant

    if not QDRANT_URL:
        raise RuntimeError("Missing QDRANT_URL")
    if not QDRANT_API_KEY:
        raise RuntimeError("Missing QDRANT_API_KEY")
    if not OPENROUTER_API_KEY:
        raise RuntimeError("Missing OPENROUTER_API_KEY")

    qdrant = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )

    collections = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION not in collections:
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )


def embed(text: str):
    r = requests.post(
        "https://openrouter.ai/api/v1/embeddings",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "text-embedding-3-small",
            "input": text
        },
        timeout=30
    )

    if r.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Embedding error: {r.text}")

    data = r.json()
    return data["data"][0]["embedding"]


@app.get("/")
def root():
    return {"name": "ares-memory-mcp", "status": "ok"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/tools/store_memory")
def store_memory(text: str):
    vector = embed(text)

    point_id = str(uuid.uuid4())
    point = PointStruct(
        id=point_id,
        vector=vector,
        payload={"text": text}
    )

    qdrant.upsert(
        collection_name=COLLECTION,
        points=[point]
    )

    return {"status": "stored", "id": point_id, "memory": text}


@app.post("/tools/search_memory")
def search_memory(query: str):
    vector = embed(query)

    results = qdrant.search(
        collection_name=COLLECTION,
        query_vector=vector,
        limit=5
    )

    return [{"id": r.id, "text": r.payload.get("text", "")} for r in results]


@app.get("/tools/list_memories")
def list_memories():
    points, _ = qdrant.scroll(
        collection_name=COLLECTION,
        limit=50,
        with_payload=True
    )

    return [{"id": p.id, "text": p.payload.get("text", "")} for p in points]


@app.post("/tools/delete_memory")
def delete_memory(id: str):
    qdrant.delete(
        collection_name=COLLECTION,
        points_selector=[id]
    )

    return {"status": "deleted", "id": id}
