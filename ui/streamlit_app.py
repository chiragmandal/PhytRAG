"""
PhytRAG Streamlit UI
Connects to the FastAPI backend at API_URL and renders answers with citations.
"""
import os

import requests
import streamlit as st

API_URL = os.getenv("PHYTRAG_API_URL", "http://localhost:8000")

EXAMPLE_QUESTIONS = [
    "What is the role of GA20ox in Arabidopsis stem elongation?",
    "How does jasmonic acid regulate plant defense against insects?",
    "What mechanisms control flowering time in response to photoperiod?",
    "How do plants respond to drought stress at the molecular level?",
    "What is the role of auxin in lateral root development?",
]

st.set_page_config(
    page_title="PhytRAG",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🌿 PhytRAG")
    st.caption("RAG over open-access plant biology literature")
    st.divider()

    st.subheader("Settings")
    top_k = st.slider("Sources to retrieve (top-k)", min_value=1, max_value=10, value=5)

    st.divider()
    st.subheader("Example questions")
    for eq in EXAMPLE_QUESTIONS:
        if st.button(eq, use_container_width=True, key=eq):
            st.session_state["question"] = eq
            st.session_state.pop("result", None)

    st.divider()
    st.caption("Backend: " + API_URL)

    try:
        h = requests.get(f"{API_URL}/health", timeout=3).json()
        vectors = h.get("collection_vectors", 0)
        if h.get("status") == "ok":
            st.success(f"API healthy · {vectors:,} vectors indexed")
        else:
            st.warning("API degraded")
    except Exception:
        st.error("API unreachable")


# ── Main panel ────────────────────────────────────────────────────────────────

st.title("Plant Biology Research Assistant")
st.caption(
    "Ask questions about plant biology. Answers are grounded in peer-reviewed "
    "literature from PubMed Central Open Access."
)

question = st.text_area(
    "Your question",
    value=st.session_state.get("question", ""),
    placeholder="e.g. How does gibberellin regulate stem elongation in Arabidopsis?",
    height=100,
    key="question_input",
)

col_ask, col_clear = st.columns([1, 5])
with col_ask:
    ask = st.button("Ask", type="primary", use_container_width=True)
with col_clear:
    if st.button("Clear", use_container_width=False):
        st.session_state.pop("question", None)
        st.session_state.pop("result", None)
        st.rerun()

# ── Run query and store result in session_state ───────────────────────────────

if ask:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving sources and generating answer…"):
            try:
                resp = requests.post(
                    f"{API_URL}/query",
                    json={"q": question.strip(), "top_k": top_k},
                    timeout=120,
                )
                resp.raise_for_status()
                st.session_state["result"] = resp.json()
            except requests.exceptions.ConnectionError:
                st.error("Could not reach the API. Is the backend running?")
            except requests.exceptions.HTTPError as e:
                st.error(f"API error {e.response.status_code}: {e.response.text}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

# ── Render result from session_state (persists across reruns) ─────────────────

if "result" in st.session_state:
    data = st.session_state["result"]

    st.divider()

    # Answer
    st.subheader("Answer")
    st.write(data["answer"])

    st.divider()

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("End-to-end latency", f"{data['latency_ms']:,} ms")
    m2.metric("Time to first token", f"{data['ttft_ms']:,} ms")
    m3.metric("Tokens generated", data["tokens_generated"])
    m4.metric("Model", data["model"])

    st.divider()

    # Sources
    st.subheader(f"Sources ({len(data['sources'])} retrieved)")
    for i, src in enumerate(data["sources"], 1):
        score_pct = int(src["score"] * 100)
        with st.expander(f"[{i}] {src['title']}  (score: {score_pct}%)", expanded=(i == 1)):
            st.markdown(f"**PMC{src['pmcid']}** · chunk {src['chunk_index']}")
            pmc_link = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{src['pmcid']}/"
            st.markdown(f"[Open paper on PubMed Central ↗]({pmc_link})")
            st.divider()
            st.write(src["excerpt"])
