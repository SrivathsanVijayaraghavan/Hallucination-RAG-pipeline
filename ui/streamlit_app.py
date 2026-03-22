"""
ui/streamlit_app.py
Deployment version -- Hugging Face Spaces

Uses HuggingFace Inference API instead of local Ollama.
"""

import streamlit as st
import tempfile
import os
import sys
from huggingface_hub import InferenceClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.loader import load_pdf
from rag.chunker import chunk_text
from rag.embeddings import load_embed_model, embed_chunks
from rag.retriever import store_chunks, retrieve
from detectors.factual_checker import run_factual_check
from detectors.numerical_checker import run_numerical_check
from detectors.contradiction_checker import run_contradiction_check
from scoring.confidence import compute_confidence_scores
from nltk import sent_tokenize
import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Hallucination RAG-Pipeline",
    page_icon="🔍",
    layout="wide"
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    h1, h2, h3 { font-family: 'JetBrains Mono', monospace; }

    .verdict-card {
        border-radius: 8px;
        padding: 16px 20px;
        margin: 10px 0;
        border-left: 4px solid;
    }

    .verdict-green  { background: rgba(34, 197, 94, 0.08);  border-color: #22c55e; }
    .verdict-yellow { background: rgba(234, 179, 8, 0.08);  border-color: #eab308; }
    .verdict-red    { background: rgba(239, 68, 68, 0.08);  border-color: #ef4444; }

    .score-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        padding: 2px 8px;
        border-radius: 4px;
        background: rgba(255,255,255,0.08);
    }

    .sentence-text {
        font-size: 0.95rem;
        margin: 6px 0 10px 0;
        color: #d4d4d4;
        line-height: 1.5;
    }

    .score-row { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 8px; }
    .score-item { font-size: 0.78rem; font-family: 'JetBrains Mono', monospace; color: #888; }
    .score-item span { color: #ccc; }

    .overall-box {
        border-radius: 10px;
        padding: 20px 24px;
        margin: 16px 0;
        text-align: center;
        font-family: 'JetBrains Mono', monospace;
    }

    .overall-green  { background: rgba(34, 197, 94, 0.12);  border: 1px solid #22c55e44; }
    .overall-yellow { background: rgba(234, 179, 8, 0.12);  border: 1px solid #eab30844; }
    .overall-red    { background: rgba(239, 68, 68, 0.12);  border: 1px solid #ef444444; }

    .answer-box {
        background: #111;
        border: 1px solid #2a2a2a;
        border-radius: 8px;
        padding: 16px 20px;
        font-size: 1rem;
        color: #e8e8e8;
        line-height: 1.6;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Not-found phrases
# ─────────────────────────────────────────────

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


def is_not_found_answer(answer: str) -> bool:
    return any(phrase in answer.lower() for phrase in NOT_FOUND_PHRASES)


# ─────────────────────────────────────────────
# LLM via HuggingFace Inference API
# ─────────────────────────────────────────────

def generate_answer(question: str, chunks: list[str], hf_token: str) -> str:
    context = "\n\n---\n\n".join(chunks)
    prompt = f"""You are a precise question-answering assistant.
Answer the question using ONLY the information in the context below.
Write your answer in clear, natural language sentences.
Do NOT copy tables or raw data — summarize the information instead.
If the answer is not in the context, say: "This information is not in the provided document."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

    client = InferenceClient(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        token=hf_token
    )

    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.1
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────

if "indexed" not in st.session_state:
    st.session_state.indexed    = False
    st.session_state.collection = None
    st.session_state.model      = None
    st.session_state.num_chunks = 0


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────

st.markdown("## 🔍 Hallucination RAG-Pipeline")
st.markdown("Upload a PDF, ask a question, get a verified answer with hallucination scores.")
st.markdown("---")


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🔑 HuggingFace Token")
    hf_token = st.text_input(
        "Enter your HF token",
        type="password",
        placeholder="hf_..."
    )
    st.caption("Get a free token at huggingface.co/settings/tokens")

    st.markdown("---")
    st.markdown("### 📄 Document")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file and not st.session_state.indexed:
        if st.button("Index Document"):
            with st.spinner("Indexing PDF..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                text       = load_pdf(tmp_path)
                chunks     = chunk_text(text, chunk_size=150, overlap=30)
                model      = load_embed_model()
                embeddings = embed_chunks(model, chunks)
                collection = store_chunks(chunks, embeddings)

                st.session_state.indexed    = True
                st.session_state.collection = collection
                st.session_state.model      = model
                st.session_state.num_chunks = len(chunks)

                os.unlink(tmp_path)

            st.success(f"Indexed {st.session_state.num_chunks} chunks")

    if st.session_state.indexed:
        st.success(f"✅ Document ready ({st.session_state.num_chunks} chunks)")
        if st.button("Reset"):
            st.session_state.indexed    = False
            st.session_state.collection = None
            st.session_state.model      = None
            st.rerun()

    st.markdown("---")
    st.markdown("### ℹ️ How it works")
    st.markdown("""
- **Factual** — cosine similarity per claim
- **Numerical** — number normalization + comparison
- **Contradiction** — DeBERTa NLI model
- **Combined** — weighted score (40/35/25)
    """)
    st.markdown("---")
    st.markdown("🟢 < 0.4 → Grounded")
    st.markdown("🟡 0.4–0.7 → Uncertain")
    st.markdown("🔴 > 0.7 → Hallucination")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if not hf_token:
    st.info("Enter your HuggingFace token in the sidebar to get started.")
elif not st.session_state.indexed:
    st.info("Upload a PDF and click **Index Document** to get started.")
else:
    question = st.text_input(
        "Ask a question about your document",
        placeholder="e.g. What was NVIDIA's revenue in Q4 FY2024?"
    )

    if st.button("Analyze") and question.strip():
        model      = st.session_state.model
        collection = st.session_state.collection

        with st.spinner("Retrieving and generating answer..."):
            top_chunks = retrieve(question, model, collection)
            answer     = generate_answer(question, top_chunks, hf_token)

        st.markdown("### 💬 Answer")
        st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

        if is_not_found_answer(answer):
            st.info("Answer not found in document — skipping hallucination checks.")
        else:
            with st.spinner("Running hallucination checks..."):
                factual_results       = run_factual_check(answer, model, collection, question)
                numerical_results     = run_numerical_check(answer, model, collection, question)
                contradiction_results = run_contradiction_check(answer, model, collection, question)

                sentences = [s.strip() for s in sent_tokenize(answer) if len(s.strip()) > 3]
                scores    = compute_confidence_scores(
                    sentences,
                    factual_results,
                    numerical_results,
                    contradiction_results
                )

            st.markdown("---")
            st.markdown("### 🧪 Hallucination Analysis")

            if not scores:
                st.warning("No sentences could be analyzed.")
            else:
                for s in scores:
                    if s.verdict == "GROUNDED":
                        card_class = "verdict-green"
                        icon       = "🟢"
                    elif s.verdict == "UNCERTAIN":
                        card_class = "verdict-yellow"
                        icon       = "🟡"
                    else:
                        card_class = "verdict-red"
                        icon       = "🔴"

                    numerical_str = f"Numerical: <span>{s.numerical_score}</span>" if s.has_numerical else "Numerical: <span>N/A</span>"

                    st.markdown(f"""
<div class="verdict-card {card_class}">
    <div>{icon} <strong>{s.verdict}</strong> &nbsp; <span class="score-badge">combined: {s.combined_score}</span></div>
    <div class="sentence-text">{s.sentence}</div>
    <div class="score-row">
        <div class="score-item">Factual: <span>{s.factual_score}</span></div>
        <div class="score-item">{numerical_str}</div>
        <div class="score-item">Contradiction: <span>{s.contradiction_score}</span></div>
    </div>
</div>
""", unsafe_allow_html=True)

                st.markdown("---")
                avg_score = round(sum(s.combined_score for s in scores) / len(scores), 4)

                if avg_score < 0.4:
                    overall_class = "overall-green"
                    overall_icon  = "🟢"
                    overall_label = "GROUNDED"
                elif avg_score < 0.7:
                    overall_class = "overall-yellow"
                    overall_icon  = "🟡"
                    overall_label = "UNCERTAIN"
                else:
                    overall_class = "overall-red"
                    overall_icon  = "🔴"
                    overall_label = "HALLUCINATION"

                st.markdown(f"""
<div class="overall-box {overall_class}">
    <div style="font-size: 2rem">{overall_icon}</div>
    <div style="font-size: 1.2rem; font-weight: 600; margin: 4px 0">{overall_label}</div>
    <div style="font-size: 0.9rem; color: #888">Overall Score: {avg_score}</div>
</div>
""", unsafe_allow_html=True)

                col1, col2, col3 = st.columns(3)
                col1.metric("🟢 Grounded",      sum(1 for s in scores if s.verdict == "GROUNDED"))
                col2.metric("🟡 Uncertain",     sum(1 for s in scores if s.verdict == "UNCERTAIN"))
                col3.metric("🔴 Hallucination", sum(1 for s in scores if s.verdict == "HALLUCINATION"))