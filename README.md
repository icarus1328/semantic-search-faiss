# FAISS + Sentence Transformers Retriever

This project demonstrates how to build a semantic search system using **FAISS** and **Sentence Transformers**. It allows you to embed text chunks, create a FAISS index, and perform similarity-based retrieval using natural language queries.

---

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [File Structure](#file-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Example Output](#example-output)
- [Notes](#notes)
- [License](#license)

---

## Overview

This project uses:

- **[FAISS](https://github.com/facebookresearch/faiss)** – for efficient similarity search on vector embeddings.
- **[Sentence Transformers](https://www.sbert.net/)** – for generating embeddings of text chunks.
- **Python** – for scripting the data processing, index building, and querying.

The workflow is:

1. Load text chunks from a file (`data/chunks.txt`).
2. Encode chunks into embeddings using a Sentence Transformer model (`all-MiniLM-L6-v2`).
3. Normalize embeddings and build a FAISS index.
4. Save the index and metadata for later retrieval.
5. Load the index and metadata to answer semantic queries with top-k results.

---

## Requirements

Install the required Python packages:

```bash
pip install numpy faiss-cpu sentence-transformers
```
---

## File Structure
project-root/
│
├─ data/
│   └─ chunks.txt          # Text chunks, one per line
│
├─ storage/
│   ├─ index.faiss         # FAISS index (generated)
│   └─ metadata.json       # Metadata mapping for each chunk
│
├─ indexer.py              # Script to build embeddings and FAISS index
├─ retriever.py            # Semantic search / retrieval script
└─ README.md

---

## Setup
1. Place your text chunks in data/chunks.txt, one chunk per line.
2. Ensure the storage folder exists (it will be automatically created by indexer.py).

---

## Usage

### Building the Index
Run the indexer.py script to generate embeddings and build the FAISS index.

What happens:
1. Loads text chunks from data/chunks.txt.
2. Loads the all-MiniLM-L6-v2 model.
3. Generates embeddings for each chunk.
4. Normalizes embeddings and builds an inner-product FAISS index.
5. Saves the index to storage/index.faiss and metadata to storage/metadata.json.

### Running the Retriever
Run the retriever.py script to query the FAISS index.

It will:
1. Load the model.
2. Load the FAISS index and metadata.
3. Perform top-k semantic searches on example queries.

---

## Notes

- You may see the following warning when loading the Sentence Transformer model:
BertModel LOAD REPORT from: sentence-transformers/all-MiniLM-L6-v2
Key                     | Status
embeddings.position_ids | UNEXPECTED

This can be safely ignored; it does not affect embeddings.
- FAISS uses normalized embeddings (faiss.normalize_L2) to perform cosine similarity search using inner product (IndexFlatIP).
- You can adjust the number of top results by changing the k parameter in the search() method.

# License
This project is open-source and free to use under the MIT License.