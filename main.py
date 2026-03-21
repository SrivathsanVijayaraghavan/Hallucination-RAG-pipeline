"""
main.py
Phase 1 through 5 -- RAG Pipeline + Auto Hallucination Detection

All three detectors run automatically after every answer.
No manual selection needed.
"""

import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

from rag.loader import load_pdf
from rag.chunker import chunk_text
from rag.embeddings import load_embed_model, embed_chunks
from rag.retriever import store_chunks, retrieve
from detectors.factual_checker import run_factual_check
from detectors.numerical_checker import run_numerical_check
from detectors.contradiction_checker import run_contradiction_check
from scoring.confidence import compute_confidence_scores, print_confidence_report
from langchain_community.llms import Ollama
from nltk import sent_tokenize

PDF_PATH  = "data/sample.pdf"
LLM_MODEL = "llama3.2"

# Phrases that indicate Llama couldn't find the answer in the document
NOT_FOUND_PHRASES = [
    "not in the provided document",
    "not mentioned in",
    "not available in",
    "context does not",
    "context only mentions",
    "no information",
    "cannot find",
    "not found in",
    "does not contain",
    "not provided in"
]


def build_prompt(question: str, chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(chunks)
    return f"""You are a precise question-answering assistant.
Answer the question using ONLY the information in the context below.
Write your answer in clear, natural language sentences.
Do NOT copy tables or raw data — summarize the information instead.
If the answer is not in the context, say: "This information is not in the provided document."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

def is_not_found_answer(answer: str) -> bool:
    """Check if Llama indicated the answer wasn't in the document."""
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in NOT_FOUND_PHRASES)


def run_all_checks(answer: str, question: str, model, collection) -> None:
    """Run all three detectors and print the combined confidence report."""

    # Skip checks if Llama said the answer isn't in the document
    if is_not_found_answer(answer):
        print("\n[Skipping hallucination checks — answer not found in document]")
        return

    print("\n[Running hallucination checks...]")

    # Phase 2 - Factual
    factual_results = run_factual_check(answer, model, collection, question)

    # Phase 3 - Numerical
    numerical_results = run_numerical_check(answer, model, collection, question)

    # Phase 4 - Contradiction
    contradiction_results = run_contradiction_check(answer, model, collection, question)

    # Phase 5 - Combined confidence score
    sentences = [s.strip() for s in sent_tokenize(answer) if len(s.strip()) > 3]
    scores = compute_confidence_scores(
        sentences,
        factual_results,
        numerical_results,
        contradiction_results
    )
    print_confidence_report(scores)


def main():
    print("=" * 60)
    print("  Hallucination Detector -- Phase 1 through 5")
    print("=" * 60)

    # Index the PDF
    text       = load_pdf(PDF_PATH)
    chunks     = chunk_text(text, chunk_size=150, overlap=30)
    model      = load_embed_model()
    embeddings = embed_chunks(model, chunks)
    collection = store_chunks(chunks, embeddings)

    # Load LLM
    llm = Ollama(model=LLM_MODEL)

    print("\n" + "=" * 60)
    print("  Ready. Ask questions about your PDF.")
    print("  Type 'quit' to exit.")
    print("=" * 60)

    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            print("\nExiting.")
            break
        if not question:
            continue

        # Phase 1 - Retrieve + Generate
        top_chunks = retrieve(question, model, collection)

        print("\nRetrieved chunks:")
        for i, chunk in enumerate(top_chunks):
            print(f"  Chunk {i+1}: {chunk[:100].replace(chr(10), ' ')}...")

        print("\nThinking...")
        prompt = build_prompt(question, top_chunks)
        answer = llm.invoke(prompt)

        print("\n" + "-" * 60)
        print("ANSWER:")
        print(answer)
        print("-" * 60)

        # Auto run all checks
        run_all_checks(answer, question, model, collection)


if __name__ == "__main__":
    main()