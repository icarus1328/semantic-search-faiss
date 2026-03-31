import numpy as np
import faiss
import os
import json
from sentence_transformers import SentenceTransformer

MODEL_NAME = 'all-MiniLM-L6-v2'
INDEX_PATH = 'storage/index.faiss'
METADATA_PATH = 'storage/metadata.json'
CHUNKS_PATH = 'data/chunks.txt'

def load_chunks(filepath: str) -> list[str]:
    with open(filepath, 'r', encoding='utf-8') as f:
        chunks = [line.strip() for line in f.readlines() if line.strip()]
    return chunks

def build_index(chunks: list[str]):
    print(f"Loaded {len(chunks)} chunks from file")
    
    print(f"Loading Model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    
    print("Embedding Chunks")
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True).astype('float32')
    
    print(f"Embeddings shape : {embeddings.shape}")
    
    faiss.normalize_L2(embeddings)
    
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    print(f"Index built with {index.ntotal} vectors")
    
    faiss.write_index(index, INDEX_PATH)
    print(f"Index saved to {INDEX_PATH}")
    
    metadata = {
        str(i): {
            "text" : chunks[i],
            "chunk_id" : i
        }
        for i in range(len(chunks))
    }
    
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to : {METADATA_PATH}")
    
if __name__ == "__main__":
    os.makedirs('storage', exist_ok=True)
    load_chunks(CHUNKS_PATH)
    build_index(chunks)