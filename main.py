from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from retriever import Retriever

app = FastAPI(
    title = "F1 Semantic Search",
    description="Search F1 knowledge using FAISS and sentence-transformer",
    version= "1.0.0"
)

retriever = Retriever()

class SearchRequest(BaseModel):
    query: str
    k: int = 3

class SearchResult(BaseModel):
    rank: int
    text: str
    score: float
    id: int
    
class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    total_results: int

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/index?stats")
def index_stats():
    return {
        "total_vectors": retriever.index.ntotal,
        "model": retriever.MODEL_NAME
    }

@app.post("/search" ,response_model=SearchResponse)
def search(request: SearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

        
    raw_results = retriever.search(request.query, k=request.k)
    
    results = [
        SearchResult(
            rank = rank + 1,
            text = result["text"],
            score = result["score"],
            id = result["id"]
        )
        for rank, result in enumerate(raw_results)
    ]
    
    return SearchResponse(
        query=request.query,
        results=results,
        total_results=len(results)
    )