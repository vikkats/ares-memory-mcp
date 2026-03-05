import os
import uuid
import requests

from starlette.applications import Starlette
from starlette.routing import Mount

from mcp.server.fastmcp import FastMCP
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
COLLECTION = "ares_memory"

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

collections = [c.name for c in qdrant.get_collections().collections]
if COLLECTION not in collections:
    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )

def embed(text: str):
    r = requests.post(
        "https://openrouter.ai/api/v1/embeddings",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "text-embedding-3-small",
            "input": text,
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]

mcp = FastMCP("ares-memory")

@mcp.tool()
def store_memory(text: str) -> dict:
    vector = embed(text)
    point_id = str(uuid.uuid4())
    point = PointStruct(id=point_id, vector=vector, payload={"text": text})
    qdrant.upsert(collection_name=COLLECTION, points=[point])
    return {"status": "stored", "id": point_id, "memory": text}

@mcp.tool()
def search_memory(query: str) -> list[dict]:
    vector = embed(query)
    results = qdrant.search(
        collection_name=COLLECTION,
        query_vector=vector,
        limit=5,
    )
    return [{"id": r.id, "text": r.payload.get("text", "")} for r in results]

@mcp.tool()
def list_memories() -> list[dict]:
    points, _ = qdrant.scroll(
        collection_name=COLLECTION,
        limit=50,
        with_payload=True,
    )
    return [{"id": p.id, "text": p.payload.get("text", "")} for p in points]

@mcp.tool()
def delete_memory(id: str) -> dict:
    qdrant.delete(collection_name=COLLECTION, points_selector=[id])
    return {"status": "deleted", "id": id}

# The magic fix: Mounting the SSE app directly to the root
app = Starlette(routes=[
    Mount('/', app=mcp.sse_app())
])
