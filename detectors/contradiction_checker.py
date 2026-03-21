"""
detectors/contradiction_checker.py
Phase 4 -- Contradiction Hallucination Detector
"""

import nltk
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import chromadb

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

NLI_MODEL               = "cross-encoder/nli-deberta-v3-small"
CONTRADICTION_THRESHOLD = 0.7
ENTAILMENT_THRESHOLD    = 0.5
MIN_WORDS_FOR_NLI       = 8

print("[Contradiction Checker] Loading DeBERTa NLI model...")
nli_pipeline = pipeline(
    "text-classification",
    model=NLI_MODEL,
    device=-1
)
print("[Contradiction Checker] Model loaded.")


def split_sentences(text: str) -> list[str]:
    from nltk import sent_tokenize
    sentences = sent_tokenize(text)
    return [s.strip() for s in sentences if len(s.strip()) > 3]


def retrieve_chunks(
    sentence: str,
    embed_model: SentenceTransformer,
    collection: chromadb.Collection,
    top_k: int = 3
) -> list[str]:
    embedding = embed_model.encode([sentence]).tolist()[0]
    results = collection.query(query_embeddings=[embedding], n_results=top_k, include=["documents"])
    return results["documents"][0]


def check_nli(premise: str, hypothesis: str) -> dict:
    result = nli_pipeline(
        f"{premise} [SEP] {hypothesis}",
        truncation=True,
        max_length=512
    )
    label = result[0]["label"].upper()
    score = result[0]["score"]
    return {"label": label, "score": score}


def check_sentence_for_contradiction(answer_sentence: str, chunks: list[str]) -> dict:
    best_contradiction_score = 0.0
    best_contradiction_pair  = None
    best_entailment_score    = 0.0

    for chunk in chunks:
        chunk_sentences = split_sentences(chunk)
        for chunk_sentence in chunk_sentences:
            nli_result = check_nli(premise=chunk_sentence, hypothesis=answer_sentence)

            if nli_result["label"] == "CONTRADICTION":
                if nli_result["score"] > best_contradiction_score:
                    best_contradiction_score = nli_result["score"]
                    best_contradiction_pair  = chunk_sentence
            elif nli_result["label"] == "ENTAILMENT":
                if nli_result["score"] > best_entailment_score:
                    best_entailment_score = nli_result["score"]

    is_hallucination = (
        best_contradiction_score >= CONTRADICTION_THRESHOLD and
        best_entailment_score < ENTAILMENT_THRESHOLD
    )

    return {
        "answer_sentence": answer_sentence,
        "verdict": "HALLUCINATION" if is_hallucination else "GROUNDED",
        "is_hallucination": is_hallucination,
        "contradiction_score": round(best_contradiction_score, 4),
        "entailment_score": round(best_entailment_score, 4),
        "contradicting_chunk_sentence": best_contradiction_pair or "None found"
    }


def run_contradiction_check(
    answer: str,
    embed_model: SentenceTransformer,
    collection: chromadb.Collection,
    question: str = ""
) -> list[dict]:
    print("\n[Contradiction Checker] Starting contradiction check...")

    answer_sentences = split_sentences(answer)
    print(f"[Contradiction Checker] {len(answer_sentences)} sentence(s) found")

    all_results = []

    for i, sentence in enumerate(answer_sentences):
        print(f"\n  Sentence {i+1}: {sentence[:80]}...")

        # Skip sentences too short for NLI
        if len(sentence.split()) < MIN_WORDS_FOR_NLI:
            print(f"  -> Skipped (too short for NLI, needs {MIN_WORDS_FOR_NLI}+ words)")
            continue

        enriched = f"{question} {sentence}".strip() if question and len(sentence.split()) < 12 else sentence
        chunks   = retrieve_chunks(enriched, embed_model, collection)
        result   = check_sentence_for_contradiction(enriched, chunks)
        result["answer_sentence"] = sentence
        all_results.append(result)

        c_score = result["contradiction_score"]
        e_score = result["entailment_score"]
        print(f"  -> {result['verdict']} (contradiction: {c_score}, entailment: {e_score})")
        if result["is_hallucination"]:
            print(f"  -> Contradicting source: {result['contradicting_chunk_sentence'][:100]}")

    return all_results


def print_contradiction_report(results: list[dict]):
    print("\n" + "=" * 60)
    print("  CONTRADICTION HALLUCINATION REPORT")
    print("=" * 60)

    if not results:
        print("\n  No sentences long enough for NLI checking.")
        print("=" * 60)
        return

    hallucinations = [r for r in results if r["is_hallucination"]]
    grounded       = [r for r in results if not r["is_hallucination"]]

    print(f"\n  Total sentences checked : {len(results)}")
    print(f"  GROUNDED                : {len(grounded)}")
    print(f"  HALLUCINATIONS          : {len(hallucinations)}")

    print("\n" + "-" * 60)
    for r in results:
        print(f"\n  {r['verdict']} (contradiction: {r['contradiction_score']}, entailment: {r['entailment_score']})")
        print(f"  Sentence    : {r['answer_sentence'][:100]}")
        print(f"  Contradicts : {r['contradicting_chunk_sentence'][:100]}")

    print("\n" + "=" * 60)