"""
detectors/factual_checker.py
Phase 2 -- Factual Hallucination Detector
"""

import re
import nltk
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

SIMILARITY_THRESHOLD             = 0.5
EXPLANATION_SIMILARITY_THRESHOLD = 0.35
TOP_K    = 5
PRONOUNS = {"it", "this", "that", "they", "he", "she", "its", "these", "those"}

EXPLANATION_KEYWORDS = {
    "explain", "summarize", "summary", "describe", "what is",
    "what are", "how does", "how do", "tell me about", "overview",
    "elaborate", "clarify", "define", "meaning", "purpose",
    "background", "context", "introduction", "outline"
}


def is_explanation_query(question: str) -> bool:
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in EXPLANATION_KEYWORDS)


def split_into_sentences(text: str) -> list[str]:
    sentences = nltk.sent_tokenize(text)
    return [s.strip() for s in sentences if len(s.strip()) > 3]


def split_into_atomic_claims(sentence: str) -> list[str]:
    if len(sentence.split()) <= 15:
        return [sentence]
    parts = re.split(r",\s*(and|but|while|up|down|which|with)\s+", sentence)
    claims = []
    for part in parts:
        part = part.strip()
        if len(part) > 5 and part.lower() not in ("and", "but", "while", "up", "down", "which", "with"):
            claims.append(part)
    return claims if claims else [sentence]


def enrich_claim(claim: str, sentence: str, question: str) -> str:
    first_word = claim.split()[0].lower() if claim.split() else ""
    starts_with_pronoun = first_word in PRONOUNS
    too_short = len(claim.split()) < 10
    if starts_with_pronoun or too_short:
        if question:
            return f"{question} {claim}".strip()
        return sentence
    return claim


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot = np.dot(vec1, vec2)
    norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def check_claim(
    claim: str,
    embed_model: SentenceTransformer,
    collection: chromadb.Collection,
    threshold: float = SIMILARITY_THRESHOLD
) -> dict:
    claim_embedding = embed_model.encode([claim]).tolist()[0]

    results = collection.query(
        query_embeddings=[claim_embedding],
        n_results=TOP_K,
        include=["documents", "embeddings"]
    )

    chunks = results["documents"][0]
    chunk_embeddings = results["embeddings"][0]

    similarities = []
    for chunk_emb in chunk_embeddings:
        sim = cosine_similarity(np.array(claim_embedding), np.array(chunk_emb))
        similarities.append(sim)

    max_similarity = max(similarities)
    best_chunk_idx = similarities.index(max_similarity)
    is_hallucination = max_similarity < threshold

    return {
        "claim": claim,
        "max_similarity": round(max_similarity, 4),
        "verdict": "HALLUCINATION" if is_hallucination else "GROUNDED",
        "best_chunk_preview": chunks[best_chunk_idx][:120] + "...",
        "is_hallucination": is_hallucination
    }


def run_factual_check(
    answer: str,
    embed_model: SentenceTransformer,
    collection: chromadb.Collection,
    question: str = ""
) -> list[dict]:
    print("\n[Factual Checker] Starting factual hallucination check...")

    explanation_mode = is_explanation_query(question)
    threshold = EXPLANATION_SIMILARITY_THRESHOLD if explanation_mode else SIMILARITY_THRESHOLD

    if explanation_mode:
        print(f"[Factual Checker] Explanation query detected -> threshold lowered to {threshold}")
    else:
        print(f"[Factual Checker] Factual query -> threshold: {threshold}")

    sentences = split_into_sentences(answer)
    print(f"[Factual Checker] {len(sentences)} sentence(s) found")

    all_results = []

    for i, sentence in enumerate(sentences):
        print(f"\n  Sentence {i+1}: {sentence[:80]}...")

        claims = split_into_atomic_claims(sentence)
        print(f"  -> {len(claims)} atomic claim(s)")

        for claim in claims:
            claim_to_check = enrich_claim(claim, sentence, question)
            result = check_claim(claim_to_check, embed_model, collection, threshold)
            result["claim"] = claim
            all_results.append(result)

            status = "HALLUCINATION" if result["is_hallucination"] else "GROUNDED"
            print(f"     [{result['max_similarity']:.4f}] {status} -- {claim[:70]}")

    return all_results


def print_factual_report(results: list[dict]):
    print("\n" + "=" * 60)
    print("  FACTUAL HALLUCINATION REPORT")
    print("=" * 60)

    hallucinations = [r for r in results if r["is_hallucination"]]
    grounded = [r for r in results if not r["is_hallucination"]]

    print(f"\n  Total claims checked : {len(results)}")
    print(f"  GROUNDED             : {len(grounded)}")
    print(f"  HALLUCINATIONS       : {len(hallucinations)}")

    print("\n" + "-" * 60)
    for r in results:
        status = "HALLUCINATION" if r["is_hallucination"] else "GROUNDED"
        print(f"\n  {status} (score: {r['max_similarity']})")
        print(f"  Claim : {r['claim']}")
        print(f"  Source: {r['best_chunk_preview']}")

    print("\n" + "=" * 60)