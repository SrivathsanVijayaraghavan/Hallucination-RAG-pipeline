"""
rag/chunker.py
Splits raw text into overlapping word-based chunks.
"""


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """
    Split `text` into chunks of `chunk_size` words with `overlap` word overlap.
    """
    print(f"[Chunker] Splitting text into chunks ({chunk_size} words, {overlap} overlap)")
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    print(f"[Chunker] ✅ {len(chunks)} chunks created")
    return chunks
