"""
rag/retriever.py
Stores chunks in ChromaDB and retrieves relevant ones for a query.
"""

import chromadb
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "hallucination_detector"


def store_chunks(chunks: list[str], embeddings: list) -> chromadb.Collection:
    """
    Store chunks and their embeddings in a local ChromaDB collection.
    Wipes and rebuilds the collection on every run.
    """
    print(f"[Retriever] Storing {len(chunks)} chunks in ChromaDB...")
    client = chromadb.PersistentClient(path="./chroma_db")

    # Fresh rebuild every run
    try:
        client.delete_collection(COLLECTION_NAME)
        print("[Retriever] Existing collection cleared")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(ids=ids, documents=chunks, embeddings=embeddings)

    print(f"[Retriever] ✅ Stored successfully")
    return collection


def retrieve(query: str, model: SentenceTransformer, collection: chromadb.Collection, top_k: int = 5) -> list[str]:
    """
    Embed the query and return the top_k most relevant chunks.
    """
    query_embedding = model.encode([query]).tolist()[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results["documents"][0]
