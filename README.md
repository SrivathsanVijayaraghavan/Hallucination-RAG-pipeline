---
title: Hallucination RAG Pipeline
emoji: 🔍
colorFrom: blue
colorTo: red
sdk: docker
app_file: app.py
pinned: false
---

# Hallucination RAG-Pipeline

A modular hallucination verification layer for RAG systems. Sits on top of any LLM and verifies every answer against the source document using three independent detection methods, returning a single confidence score per claim.

Any developer building an AI product can plug this into their pipeline via a simple API call — works with Llama, GPT-4, Gemini, or any LLM.

---

## What It Does

Most RAG systems have no way to verify whether the LLM's answer is actually supported by the source document. This system solves that by running three parallel checks on every answer:

- **Factual Checker** — cosine similarity between each claim and source chunks
- **Numerical Checker** — extracts and normalizes all numbers, compares against source
- **Contradiction Checker** — DeBERTa NLI model detects direct contradictions

All three scores are combined into a single hallucination probability (0.0 → 1.0) per sentence.

---

## Architecture

```
PDF → Chunks (150 words, 30 overlap) → Embeddings → ChromaDB
                                                         ↓
User Question → Retrieve top 5 chunks → Llama 3.2 → Answer
                                                         ↓
                                     Split into atomic claims
                                                         ↓
              ┌──────────────────────────────────────────┤
              ↓                  ↓                       ↓
    Factual Checker      Numerical Checker      Contradiction Checker
    (cosine similarity   (SpaCy extraction,     (DeBERTa NLI,
     per claim vs         normalization,         entailment override)
     top 5 chunks)        scale detection)
              └──────────────────────────────────────────┤
                                                         ↓
                               Confidence Score (0.0 → 1.0)
                                                         ↓
                          GREEN / YELLOW / RED Verdict
```

---

## Tech Stack

| Purpose | Tool |
|---|---|
| Local LLM | Ollama + Llama 3.2 3B |
| RAG Pipeline | LangChain |
| Vector Database | ChromaDB |
| Text Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Sentence Splitting | NLTK |
| Number/Entity Extraction | SpaCy |
| Contradiction Detection | HuggingFace Transformers (cross-encoder/nli-deberta-v3-small) |
| Confidence Scoring | Custom weighted combination |
| Demo UI | Streamlit |
| API Layer | FastAPI + Uvicorn |
| PDF Loading | PyPDF |

Everything free. Everything local. No API keys needed.

---

## Detection Methods

### Factual Checker
Splits the LLM answer into atomic claims using rule-based splitting (no LLM). For each claim, retrieves the top 5 most relevant chunks from ChromaDB and calculates cosine similarity. Claims scoring below the threshold are flagged as hallucinations.

Threshold: `0.5` for factual queries, `0.35` for explanation/summarization queries (auto-detected).

### Numerical Checker
Extracts all numbers, percentages, and monetary values using SpaCy NER + regex fallback. Normalizes all formats to float (`$2.4M` = `$2,400,000` = `2400000.0`). Detects "in millions" notation in chunks and scales accordingly. Compares answer numbers against source chunk numbers with 1% relative tolerance.

### Contradiction Checker
Uses `cross-encoder/nli-deberta-v3-small` to check if the answer contradicts the source document. Feeds sentence pairs (chunk sentence as premise, answer sentence as hypothesis) to the NLI model. Entailment overrides contradiction — if any chunk supports the claim, it's grounded.

### Combined Confidence Score
```
Combined = Factual × 0.40 + Numerical × 0.35 + Contradiction × 0.25
```

If contradiction score ≥ 0.9 → combined score forced to 1.0 (definitive hallucination override).

Verdict thresholds:
- `< 0.4` → 🟢 GROUNDED
- `0.4 – 0.7` → 🟡 UNCERTAIN  
- `> 0.7` → 🔴 HALLUCINATION

---

## Installation

**Prerequisites:**
- Python 3.10+
- [Ollama](https://ollama.ai) installed and running
- Llama 3.2 pulled: `ollama pull llama3.2`

**Install dependencies:**
```bash
git clone https://github.com/SrivathsanVijayaraghavan/Hallucination-RAG-pipeline.git
cd Hallucination-RAG-pipeline
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## Usage

### Terminal (Interactive Q&A)
```bash
python main.py
```
Place your PDF in the `data/` folder and update `PDF_PATH` in `main.py`.

### Streamlit UI
```bash
streamlit run ui/streamlit_app.py
```
Upload any PDF, ask questions, get color-coded hallucination verdicts.

### FastAPI
```bash
uvicorn api.fastapi_app:app --reload
```
API docs available at `http://localhost:8000/docs`

**Example request:**
```bash
curl -X POST "http://localhost:8000/verify" \
  -H "Content-Type: application/json" \
  -d '{
    "answer": "NVIDIA revenue was $22.1 billion in Q4 FY2024.",
    "question": "What was NVIDIA revenue in Q4 FY2024?",
    "document_text": "NVIDIA reported revenue of $22.1 billion for Q4 FY2024, up 265% from a year ago."
  }'
```

**Example response:**
```json
{
  "overall_score": 0.03,
  "overall_verdict": "GROUNDED",
  "sentences": [
    {
      "sentence": "NVIDIA revenue was $22.1 billion in Q4 FY2024.",
      "verdict": "GROUNDED",
      "combined_score": 0.03,
      "factual_score": 0.08,
      "numerical_score": 0.0,
      "contradiction_score": 0.0,
      "has_numerical": true
    }
  ],
  "processing_time_seconds": 1.98
}
```

---

## Accuracy

| Detector | Expected Accuracy |
|---|---|
| Factual Checker | 60–70% |
| Numerical Checker | 80–90% |
| Contradiction Checker | 75–85% |
| Combined System | ~75–85% |

**Known limitations:**
- Factual checker uses semantic similarity — cannot catch wrong names or wrong numbers on correct topics (handled by numerical checker)
- Explanation/summarization queries use a lower threshold (0.35) to reduce false positives from paraphrasing
- Sentences under 8 words are skipped by the contradiction checker (NLI requires natural language context)
- Retrieval quality caps detector accuracy — if the wrong chunk is retrieved, correct claims may be flagged

---

## Project Structure

```
hallucination-RAG-pipeline/
├── api/
│   └── fastapi_app.py        # REST API with /verify and /verify-pdf endpoints
├── detectors/
│   ├── factual_checker.py    # Cosine similarity factual verification
│   ├── numerical_checker.py  # Number extraction and comparison
│   └── contradiction_checker.py  # DeBERTa NLI contradiction detection
├── rag/
│   ├── loader.py             # PDF text extraction
│   ├── chunker.py            # Word-based chunking with overlap
│   ├── embeddings.py         # Sentence-transformer embeddings
│   └── retriever.py          # ChromaDB storage and retrieval
├── scoring/
│   └── confidence.py         # Weighted score combination
├── ui/
│   └── streamlit_app.py      # Visual demo interface
├── main.py                   # Terminal interactive mode
└── requirements.txt
```

---


GitHub: [@SrivathsanVijayaraghavan](https://github.com/SrivathsanVijayaraghavan)
