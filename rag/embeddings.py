"""
rag/embeddings.py
Loads the sentence-transformer model and embeds text chunks.
"""

from sentence_transformers import SentenceTransformer

EMBED_MODEL = "all-MiniLM-L6-v2"


def load_embed_model() -> SentenceTransformer:
    print(f"[Embeddings] Loading model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)
    print(f"[Embeddings] ✅ Model loaded")
    return model


def embed_chunks(model: SentenceTransformer, chunks: list[str]) -> list:
    """
    Convert a list of text chunks into a list of embedding vectors.
    """
    print(f"[Embeddings] Embedding {len(chunks)} chunks...")
    embeddings = model.encode(chunks, show_progress_bar=True).tolist()
    print(f"[Embeddings] ✅ Done — vector size: {len(embeddings[0])}")
    return embeddings
