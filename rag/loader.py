"""
rag/loader.py
Loads a PDF file and extracts all text from it.
"""

from pypdf import PdfReader


def load_pdf(path: str) -> str:
    """
    Read a PDF from `path` and return all its text as a single string.
    """
    print(f"[Loader] Loading PDF: {path}")
    reader = PdfReader(path)
    full_text = ""

    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

    word_count = len(full_text.split())
    print(f"[Loader] ✅ {len(reader.pages)} pages loaded — {word_count} words total")
    return full_text
