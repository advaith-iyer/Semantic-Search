import time
import os
import numpy as np
import faiss
import cohere
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()
co = cohere.Client(os.getenv("COHERE_API_KEY"))
app = FastAPI()
documents = [
    {"id": i, "content": f"Customer review {i} about battery, durability, performance or quality."}
    for i in range(75)
]
def embed_texts(texts):
    res = co.embed(
        texts=texts,
        model="embed-english-light-v3.0",
        input_type="search_document"
    )
    return [np.array(e) for e in res.embeddings]
doc_texts = [d["content"] for d in documents]
doc_embeddings = embed_texts(doc_texts)
dimension = len(doc_embeddings[0])
index = faiss.IndexFlatIP(dimension)
def normalize(v):
    return v / np.linalg.norm(v)

doc_embeddings = [normalize(e) for e in doc_embeddings]
index.add(np.array(doc_embeddings))
class SearchRequest(BaseModel):
    query: str
    k: int = 10
    rerank: bool = True
    rerankK: int = 6
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
        score = float(r.relevance_score)  # Already 0-1
        results.append({**doc, "score": score})

    results.sort(key=lambda x: x["score"], reverse=True)
    return results

@app.get("/")
def health_check():
    return {"status": "API running"}

@app.post("/search")
def semantic_search(req: SearchRequest):

    start = time.time()

    # Query embedding
    q_emb = embed_texts([req.query])[0]
    q_emb = normalize(q_emb)

    # Vector search
    scores, indices = index.search(np.array([q_emb]), req.k)

    candidates = []
    for score, idx in zip(scores[0], indices[0]):
        candidates.append({
            "id": documents[idx]["id"],
            "score": float((score + 1) / 2),
            "content": documents[idx]["content"],
            "metadata": {"source": "reviews"}
        })

    # Rerank
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
