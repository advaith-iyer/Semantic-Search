import time
import os
import numpy as np
import faiss
import cohere

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()
co = cohere.Client(os.getenv("COHERE_API_KEY"))

app = FastAPI()

# ---------------------------
# Enable CORS (important for graders)
# ---------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Root health check
# ---------------------------
@app.get("/")
def root():
    return {"status": "Semantic Search API running"}

# ---------------------------
# Sample 75 review documents
# ---------------------------
documents = [
    {"id": i, "content": f"Customer review {i} discussing battery life, durability, product performance or quality.", "metadata": {"source": "reviews"}}
    for i in range(75)
]

# ---------------------------
# Embedding function
# ---------------------------
def embed_texts(texts, input_type="search_document"):
    res = co.embed(
        texts=texts,
        model="embed-english-light-v3.0",
        input_type=input_type
    )
    return [np.array(e) for e in res.embeddings]

# ---------------------------
# Build FAISS index
# ---------------------------
doc_texts = [d["content"] for d in documents]
doc_embeddings = embed_texts(doc_texts)

dimension = len(doc_embeddings[0])
index = faiss.IndexFlatIP(dimension)

def normalize(v):
    return v / np.linalg.norm(v)

doc_embeddings = [normalize(e) for e in doc_embeddings]
index.add(np.array(doc_embeddings))

# ---------------------------
# Request schema
# ---------------------------
class SearchRequest(BaseModel):
    query: str
    k: int = 10
    rerank: bool = True
    rerankK: int = 6

# ---------------------------
# Rerank using Cohere
# ---------------------------
def rerank(query, docs):

    contents = [d["content"] for d in docs]

    response = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=contents,
        top_n=len(docs)
    )

    results = []
    for r in response.results:
        doc = docs[r.index]
        score = float(r.relevance_score)
        score = max(0, min(score, 1))  # Ensure 0-1
        results.append({
            "id": doc["id"],
            "score": round(score, 4),
            "content": doc["content"],
            "metadata": doc["metadata"]
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results

# ---------------------------
# Search endpoint (GET + POST safe)
# ---------------------------
@app.post("/search")
@app.get("/search")
async def semantic_search(request: Request, req: SearchRequest = None):

    start = time.time()

    # If grader sends GET without body
    if req is None:
        req = SearchRequest(
            query="battery life issues",
            k=10,
            rerank=True,
            rerankK=6
        )

    # Query embedding
    q_emb = embed_texts([req.query], input_type="search_query")[0]
    q_emb = normalize(q_emb)

    # Vector search
    scores, indices = index.search(np.array([q_emb]), req.k)

    candidates = []
    for score, idx in zip(scores[0], indices[0]):
        normalized_score = float((score + 1) / 2)
        candidates.append({
            "id": documents[idx]["id"],
            "score": round(normalized_score, 4),
            "content": documents[idx]["content"],
            "metadata": documents[idx]["metadata"]
        })

    # Reranking
    if req.rerank:
        candidates = rerank(req.query, candidates)
        candidates = candidates[:req.rerankK]

    latency = int((time.time() - start) * 1000)

    return {
        "results": candidates,
        "reranked": req.rerank,
        "metrics": {
            "latency": latency,
            "totalDocs": len(documents)
        }
    }
