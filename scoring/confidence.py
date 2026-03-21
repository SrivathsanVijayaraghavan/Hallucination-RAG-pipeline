"""
scoring/confidence.py
Phase 5 -- Combined Confidence Scorer

Combines Factual, Numerical, and Contradiction detector outputs
into a single hallucination probability score per sentence.

Weights:
  Factual       -> 40%
  Numerical     -> 35%
  Contradiction -> 25%

Thresholds:
  Below 0.4  -> GROUNDED
  0.4 - 0.7  -> UNCERTAIN
  Above 0.7  -> HALLUCINATION

Override:
  Contradiction score >= 0.9 -> always HALLUCINATION
"""

from dataclasses import dataclass


# -------------------------------------------------
# Verdict thresholds
# -------------------------------------------------

GROUNDED_THRESHOLD      = 0.4
HALLUCINATION_THRESHOLD = 0.7
CONTRADICTION_OVERRIDE  = 0.9  # definitive contradiction -> always flag

# Detector weights
FACTUAL_WEIGHT       = 0.40
NUMERICAL_WEIGHT     = 0.35
CONTRADICTION_WEIGHT = 0.25


# -------------------------------------------------
# Data class for a single sentence result
# -------------------------------------------------

@dataclass
class SentenceScore:
    sentence: str
    factual_score: float
    numerical_score: float
    contradiction_score: float
    combined_score: float
    verdict: str
    has_numerical: bool


# -------------------------------------------------
# Normalize factual results per sentence
# -------------------------------------------------

def get_factual_score_for_sentence(sentence: str, factual_results: list[dict]) -> float:
    """
    Find factual results for this sentence and return
    the highest hallucination probability across all its claims.
    Factual score = 1 - max_similarity
    """
    matching = []
    for r in factual_results:
        if r["claim"].strip() in sentence or sentence[:50] in r.get("claim", ""):
            matching.append(r)

    if not matching:
        if len(factual_results) > 0:
            matching = factual_results

    if not matching:
        return 0.5

    worst_similarity = min(r["max_similarity"] for r in matching)
    return round(1.0 - worst_similarity, 4)


# -------------------------------------------------
# Normalize numerical results per sentence
# -------------------------------------------------

def get_numerical_score_for_sentence(sentence: str, numerical_results: list[dict]) -> tuple[float, bool]:
    """
    Find numerical results for this sentence.
    Returns (score, has_numerical)
    """
    matching = [r for r in numerical_results if r["sentence"].strip() == sentence.strip()]

    if not matching:
        return 0.0, False

    any_hallucination = any(r["is_hallucination"] for r in matching)
    return (1.0 if any_hallucination else 0.0), True


# -------------------------------------------------
# Normalize contradiction results per sentence
# -------------------------------------------------

def get_contradiction_score_for_sentence(sentence: str, contradiction_results: list[dict]) -> float:
    """
    Find contradiction result for this sentence.
    Returns contradiction score directly (already 0->1).
    """
    for r in contradiction_results:
        if r["answer_sentence"].strip() == sentence.strip():
            return r["contradiction_score"]
    return 0.0


# -------------------------------------------------
# Compute verdict from combined score
# -------------------------------------------------

def get_verdict(score: float) -> str:
    if score < GROUNDED_THRESHOLD:
        return "GROUNDED"
    elif score < HALLUCINATION_THRESHOLD:
        return "UNCERTAIN"
    else:
        return "HALLUCINATION"


# -------------------------------------------------
# MAIN - Combine all detector results
# -------------------------------------------------

def compute_confidence_scores(
    sentences: list[str],
    factual_results: list[dict],
    numerical_results: list[dict],
    contradiction_results: list[dict]
) -> list[SentenceScore]:
    """
    For each sentence, combine all three detector scores
    into a single hallucination probability.
    """
    scores = []

    for sentence in sentences:
        factual_score = get_factual_score_for_sentence(sentence, factual_results)
        numerical_score, has_numerical = get_numerical_score_for_sentence(sentence, numerical_results)
        contradiction_score = get_contradiction_score_for_sentence(sentence, contradiction_results)

        # Override: very high contradiction = definitive hallucination
        if contradiction_score >= CONTRADICTION_OVERRIDE:
            scores.append(SentenceScore(
                sentence=sentence,
                factual_score=factual_score,
                numerical_score=numerical_score,
                contradiction_score=contradiction_score,
                combined_score=1.0,
                verdict="HALLUCINATION",
                has_numerical=has_numerical
            ))
            continue

        # Adjust weights if numerical didn't apply
        if has_numerical:
            f_weight = FACTUAL_WEIGHT
            n_weight = NUMERICAL_WEIGHT
            c_weight = CONTRADICTION_WEIGHT
        else:
            f_weight = FACTUAL_WEIGHT + NUMERICAL_WEIGHT / 2
            n_weight = 0.0
            c_weight = CONTRADICTION_WEIGHT + NUMERICAL_WEIGHT / 2

        combined = round(
            factual_score       * f_weight +
            numerical_score     * n_weight +
            contradiction_score * c_weight,
            4
        )
        verdict = get_verdict(combined)

        scores.append(SentenceScore(
            sentence=sentence,
            factual_score=factual_score,
            numerical_score=numerical_score,
            contradiction_score=contradiction_score,
            combined_score=combined,
            verdict=verdict,
            has_numerical=has_numerical
        ))

    return scores


# -------------------------------------------------
# Print combined report
# -------------------------------------------------

def print_confidence_report(scores: list[SentenceScore]):
    """Print the final combined hallucination confidence report."""
    print("\n" + "=" * 60)
    print("  HALLUCINATION DETECTION REPORT")
    print("=" * 60)

    hallucinations = [s for s in scores if s.verdict == "HALLUCINATION"]
    uncertain      = [s for s in scores if s.verdict == "UNCERTAIN"]
    grounded       = [s for s in scores if s.verdict == "GROUNDED"]

    print(f"\n  Total sentences : {len(scores)}")
    print(f"  GROUNDED        : {len(grounded)}")
    print(f"  UNCERTAIN       : {len(uncertain)}")
    print(f"  HALLUCINATION   : {len(hallucinations)}")

    print("\n" + "-" * 60)
    for s in scores:
        indicator = {"GROUNDED": "GREEN", "UNCERTAIN": "YELLOW", "HALLUCINATION": "RED"}[s.verdict]

        print(f"\n  [{indicator}] {s.verdict} (combined: {s.combined_score})")
        print(f"  Sentence      : {s.sentence[:100]}")
        print(f"  Factual       : {s.factual_score}  (weight: {FACTUAL_WEIGHT})")
        if s.has_numerical:
            print(f"  Numerical     : {s.numerical_score}  (weight: {NUMERICAL_WEIGHT})")
        else:
            print(f"  Numerical     : N/A")
        print(f"  Contradiction : {s.contradiction_score}  (weight: {CONTRADICTION_WEIGHT})")

    print("\n" + "=" * 60)

    if scores:
        avg_score = round(sum(s.combined_score for s in scores) / len(scores), 4)
        overall_verdict = get_verdict(avg_score)
        print(f"\n  OVERALL SCORE   : {avg_score}")
        print(f"  OVERALL VERDICT : {overall_verdict}")
        print("=" * 60)