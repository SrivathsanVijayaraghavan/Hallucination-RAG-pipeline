"""
detectors/numerical_checker.py
Phase 3 -- Numerical Hallucination Detector
"""

import re
import spacy
import nltk
from sentence_transformers import SentenceTransformer
import chromadb

nlp = spacy.load("en_core_web_sm")

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

MIN_NUMBER_VALUE = 50.0
PRONOUNS = {"it", "this", "that", "they", "he", "she", "its", "these", "those"}


def normalize_number(text: str) -> float | None:
    if not text:
        return None

    text = text.strip().lower()
    text = text.replace("$", "").replace(",", "").strip()

    multiplier = 1.0
    if "trillion" in text:
        multiplier = 1_000_000_000_000
        text = text.replace("trillion", "").strip()
    elif "billion" in text:
        multiplier = 1_000_000_000
        text = text.replace("billion", "").strip()
    elif "million" in text:
        multiplier = 1_000_000
        text = text.replace("million", "").strip()
    elif text.endswith("t"):
        multiplier = 1_000_000_000_000
        text = text[:-1].strip()
    elif text.endswith("b"):
        multiplier = 1_000_000_000
        text = text[:-1].strip()
    elif text.endswith("m"):
        multiplier = 1_000_000
        text = text[:-1].strip()
    elif text.endswith("k"):
        multiplier = 1_000
        text = text[:-1].strip()

    text = text.replace("%", "").strip()
    text = re.sub(r"[^\d.]", "", text)

    try:
        return float(text) * multiplier
    except ValueError:
        return None


def is_percentage(text: str) -> bool:
    return "%" in text


def extract_numbers(text: str, min_value: float = MIN_NUMBER_VALUE) -> list[dict]:
    text_clean = re.sub(r'\bFY\d{2,4}\b', '', text)
    text_clean = re.sub(r'\bQ[1-4]\b', '', text_clean)
    text_clean = re.sub(r'\b(19|20)\d{2}\b', '', text_clean)

    doc = nlp(text_clean)
    found = []

    for ent in doc.ents:
        if ent.label_ in ("MONEY", "PERCENT", "CARDINAL", "QUANTITY"):
            normalized = normalize_number(ent.text)
            if normalized is not None and normalized >= min_value:
                found.append({
                    "text": ent.text,
                    "normalized": normalized,
                    "is_percentage": is_percentage(ent.text)
                })

    pattern = r"""
        \$?
        \d{1,3}(?:,\d{3})*
        (?:\.\d+)?
        \s*(?:billion|million|trillion|[BMKbmk])?
        \s*%?
    """
    for match in re.finditer(pattern, text_clean, re.VERBOSE):
        raw = match.group().strip()
        if not raw or raw in (",", "."):
            continue
        normalized = normalize_number(raw)
        if normalized is not None and normalized >= min_value:
            already_found = any(abs(n["normalized"] - normalized) < 0.01 for n in found)
            if not already_found:
                found.append({
                    "text": raw,
                    "normalized": normalized,
                    "is_percentage": is_percentage(raw)
                })

    return found


def detect_chunk_scale(chunk: str) -> float:
    chunk_lower = chunk.lower()
    if "in millions" in chunk_lower or "$ millions" in chunk_lower:
        return 1_000_000
    elif "in billions" in chunk_lower or "$ billions" in chunk_lower:
        return 1_000_000_000
    return 1.0


def number_exists_in_chunk(answer_num: dict, chunk_nums: list[dict], tolerance: float = 0.01) -> bool:
    for chunk_num in chunk_nums:
        chunk_val = chunk_num["normalized"]
        if chunk_val == 0:
            continue
        relative_diff = abs(answer_num["normalized"] - chunk_val) / max(abs(chunk_val), 1)
        if relative_diff <= tolerance:
            return True
    return False


def get_best_chunk(sentence: str, embed_model: SentenceTransformer, collection: chromadb.Collection) -> str:
    embedding = embed_model.encode([sentence]).tolist()[0]
    results = collection.query(query_embeddings=[embedding], n_results=1, include=["documents"])
    return results["documents"][0][0]


def enrich_sentence(sentence: str, question: str) -> str:
    if not question:
        return sentence
    first_word = sentence.split()[0].lower() if sentence.split() else ""
    if first_word in PRONOUNS or len(sentence.split()) < 8:
        return f"{question} {sentence}".strip()
    return sentence


def run_numerical_check(
    answer: str,
    embed_model: SentenceTransformer,
    collection: chromadb.Collection,
    question: str = ""
) -> list[dict]:
    from nltk import sent_tokenize
    sentences = sent_tokenize(answer)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 3]

    print("\n[Numerical Checker] Starting numerical hallucination check...")
    print(f"[Numerical Checker] {len(sentences)} sentence(s) to check")

    all_results = []

    for i, sentence in enumerate(sentences):
        print(f"\n  Sentence {i+1}: {sentence[:80]}...")

        answer_numbers = extract_numbers(sentence)

        if not answer_numbers:
            print("  -> No meaningful numbers found, skipping")
            continue

        print(f"  -> {len(answer_numbers)} number(s) found: {[n['text'] for n in answer_numbers]}")

        enriched = enrich_sentence(sentence, question)
        best_chunk = get_best_chunk(enriched, embed_model, collection)

        scale = detect_chunk_scale(best_chunk)
        if scale > 1:
            print(f"  -> Chunk uses scaled notation (x{scale:,.0f}), scaling monetary values only")

        chunk_numbers = extract_numbers(best_chunk)
        for n in chunk_numbers:
            if not n["is_percentage"]:
                n["normalized"] *= scale

        print(f"  -> Chunk numbers: {[n['text'] for n in chunk_numbers]}")

        for num in answer_numbers:
            exists = number_exists_in_chunk(num, chunk_numbers)
            is_hallucination = not exists
            verdict = "HALLUCINATION" if is_hallucination else "GROUNDED"

            result = {
                "sentence": sentence,
                "number_text": num["text"],
                "number_normalized": num["normalized"],
                "is_percentage": num["is_percentage"],
                "chunk_preview": best_chunk[:120] + "...",
                "chunk_numbers": [n["text"] for n in chunk_numbers],
                "verdict": verdict,
                "is_hallucination": is_hallucination
            }
            all_results.append(result)
            print(f"     {verdict} -- '{num['text']}' (normalized: {num['normalized']})")

    return all_results


def print_numerical_report(results: list[dict]):
    print("\n" + "=" * 60)
    print("  NUMERICAL HALLUCINATION REPORT")
    print("=" * 60)

    if not results:
        print("\n  No meaningful numbers found in answer to check.")
        print("=" * 60)
        return

    hallucinations = [r for r in results if r["is_hallucination"]]
    grounded = [r for r in results if not r["is_hallucination"]]

    print(f"\n  Total numbers checked : {len(results)}")
    print(f"  GROUNDED              : {len(grounded)}")
    print(f"  HALLUCINATIONS        : {len(hallucinations)}")

    print("\n" + "-" * 60)
    for r in results:
        print(f"\n  {r['verdict']}")
        print(f"  Number  : '{r['number_text']}' -> {r['number_normalized']}")
        print(f"  Sentence: {r['sentence'][:100]}")
        print(f"  In chunk: {r['chunk_numbers']}")
        print(f"  Source  : {r['chunk_preview']}")

    print("\n" + "=" * 60)