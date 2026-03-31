import numpy as np
import faiss
import json
import os
from sentence_transformers import SentenceTransformer

MODEL_NAME = 'all-MiniLM-L6-v2'
INDEX_PATH    = 'storage/index.faiss'
METADATA_PATH = 'storage/metadata.json'

class Retriever:
    def __init__(self):
        
        self.model = SentenceTransformer(MODEL_NAME)
        print(f"Loaded Model : {MODEL_NAME}")
        
        print("Loading FAISS Index...")
        self.index = faiss.read_index(INDEX_PATH)
        print(f"Index loaded with {self.index.ntotal} vectors")
        
        print("Loading Metadata...")
        with open(METADATA_PATH,'r') as f:
            self.metadata = json.load(f)
        
        print("Retriever ready\n")
    
    def search(self, query: str, k: int = 3) -> list[dict]:
        query_embedding = self.model.encode([query], convert_to_numpy=True).astype('float32')
        
        query_embedding = query_embedding.reshape(1,-1)
        faiss.normalize_L2(query_embedding)
        
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        
        for idx, score in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            results.append({
                "text": self.metadata[str(idx)["text"]],
                "score": float(score),
                "id": idx
            })
        return results
    
if __name__ == "__main__":
    retriever = Retriever()
    
    queries = [
        "how do pit stops work?",
        "what keeps drivers safe in F1?",
        "how do F1 cars go so fast?",
        ]
    
    for query in queries:
        print(f"Query: {query}")
        results = retriever.search(query, k=3)
        for rank, results in enumerate(results):
            print(f"Rank: {rank} | Score : {results["score"]:.4f} | Results: {results["text"]}")
        print()