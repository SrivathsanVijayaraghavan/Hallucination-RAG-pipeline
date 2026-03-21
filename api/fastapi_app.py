"""
api/fastapi_app.py
Phase 7 -- FastAPI Wrapper

Exposes the hallucination detector as a REST API.
Any developer using any LLM can send a POST request and
get back a full hallucination verification report.

Run with:
    uvicorn api.fastapi_app:app --reload

Endpoints:
    POST /verify        -- verify an answer against a document text
    POST /verify-pdf    -- verify an answer against an uploaded PDF file
    GET  /health        -- health check
"""

import os
import sys
import tempfile
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk import sent_tokenize

from rag.loader import load_pdf
from rag.chunker import chunk_text
from rag.embeddings import load_embed_model, embed_chunks
from rag.retriever import store_chunks
from detectors.factual_checker import run_factual_check
from detectors.numerical_checker import run_numerical_check
from detectors.contradiction_checker import run_contradiction_check
from scoring.confidence import compute_confidence_scores, get_verdict


# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────

app = FastAPI(
    title="Hallucination Detector API",
    description="Verify LLM answers against source documents. Detects factual, numerical, and contradiction hallucinations.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# ─────────────────────────────────────────────
# Load models once at startup (not per request)
# ─────────────────────────────────────────────

print("[API] Loading embedding model...")
embed_model = load_embed_model()
print("[API] Embedding model ready.")


# ─────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────

class VerifyRequest(BaseModel):
    answer: str
    question: str = ""
    document_text: str

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "NVIDIA's Q4 FY2024 revenue was $22.1 billion.",
                "question": "What was NVIDIA's revenue in Q4 FY2024?",
                "document_text": "NVIDIA reported revenue of $22.1 billion for Q4 FY2024, up 265% from a year ago."
            }
        }


class SentenceResult(BaseModel):
    sentence: str
    verdict: str
    combined_score: float
    factual_score: float
    numerical_score: float
    contradiction_score: float
    has_numerical: bool


class VerifyResponse(BaseModel):
    overall_score: float
    overall_verdict: str
    sentences: list[SentenceResult]
    processing_time_seconds: float


# ─────────────────────────────────────────────
# Core verification logic
# ─────────────────────────────────────────────

def verify_answer(answer: str, question: str, document_text: str) -> VerifyResponse:
    """
    Core pipeline: chunk document -> embed -> store -> run all detectors -> combine scores.
    """
    start_time = time.time()

    # Chunk and embed the document text
    chunks     = chunk_text(document_text, chunk_size=150, overlap=30)
    embeddings = embed_chunks(embed_model, chunks)
    collection = store_chunks(chunks, embeddings)

    # Run all three detectors
    factual_results       = run_factual_check(answer, embed_model, collection, question)
    numerical_results     = run_numerical_check(answer, embed_model, collection, question)
    contradiction_results = run_contradiction_check(answer, embed_model, collection, question)

    # Combine scores
    sentences = [s.strip() for s in sent_tokenize(answer) if len(s.strip()) > 3]
    scores    = compute_confidence_scores(
        sentences,
        factual_results,
        numerical_results,
        contradiction_results
    )

    if not scores:
        raise HTTPException(status_code=422, detail="No analyzable sentences found in answer.")

    # Build response
    overall_score   = round(sum(s.combined_score for s in scores) / len(scores), 4)
    overall_verdict = get_verdict(overall_score)

    sentence_results = [
        SentenceResult(
            sentence=s.sentence,
            verdict=s.verdict,
            combined_score=s.combined_score,
            factual_score=s.factual_score,
            numerical_score=s.numerical_score,
            contradiction_score=s.contradiction_score,
            has_numerical=s.has_numerical
        )
        for s in scores
    ]

    processing_time = round(time.time() - start_time, 3)

    return VerifyResponse(
        overall_score=overall_score,
        overall_verdict=overall_verdict,
        sentences=sentence_results,
        processing_time_seconds=processing_time
    )


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Check if API is running."""
    return {"status": "ok", "message": "Hallucination Detector API is running."}


@app.post("/verify", response_model=VerifyResponse)
def verify_text(request: VerifyRequest):
    """
    Verify an LLM answer against raw document text.

    - **answer**: The LLM-generated answer to verify
    - **question**: The original question asked (improves retrieval accuracy)
    - **document_text**: The raw source document text to verify against
    """
    if not request.answer.strip():
        raise HTTPException(status_code=400, detail="Answer cannot be empty.")
    if not request.document_text.strip():
        raise HTTPException(status_code=400, detail="Document text cannot be empty.")

    try:
        return verify_answer(request.answer, request.question, request.document_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/verify-pdf", response_model=VerifyResponse)
async def verify_pdf(
    answer: str,
    question: str = "",
    file: UploadFile = File(...)
):
    """
    Verify an LLM answer against an uploaded PDF file.

    - **answer**: The LLM-generated answer to verify
    - **question**: The original question asked
    - **file**: PDF file to verify against
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        # Save uploaded PDF to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Extract text from PDF
        document_text = load_pdf(tmp_path)
        os.unlink(tmp_path)

        return verify_answer(answer, question, document_text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))