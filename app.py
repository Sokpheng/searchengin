"""
app.py
------
Stage 6 of the RAG pipeline: Interface.

A Streamlit UI that ties ingestion -> retrieval -> generation together:
  - Sidebar sliders for top_k and the minimum similarity threshold.
  - A query box + submit button.
  - An Answer panel (LLM response, or the graceful-failure message).
  - A Sources panel listing every retrieved chunk with filename + score.

Run with:
    streamlit run app.py
"""

import os

import streamlit as st

from ingest import IngestError, run_ingestion, load_chunks, CHUNKS_PATH
from retrieve import (
    RetrievalError,
    get_embedding_model,
    build_or_load_index,
    embed_texts,
    INDEX_PATH,
)
from generate import generate_answer, GRACEFUL_FAILURE_MESSAGE

st.set_page_config(page_title="RAG Document Search", page_icon="🔎", layout="wide")


# ---------------------------------------------------------------------------
# Cached resources: loaded once per server process, not once per query.
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading embedding model...")
def cached_embedding_model():
    return get_embedding_model()


@st.cache_resource(show_spinner="Loading vector index...")
def cached_vector_store():
    """Load the saved index, or build it (running ingestion first if needed)."""
    if not os.path.exists(CHUNKS_PATH):
        run_ingestion()
    return build_or_load_index()


def rebuild_index():
    """Force a full re-ingest + re-embed, then clear caches so the new
    index and any dependent state get picked up on the next run."""
    run_ingestion()
    from retrieve import build_index

    build_index()
    cached_vector_store.clear()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🔎 RAG Document Search")
st.caption(
    "Retrieval-Augmented Generation search over your own document collection. "
    "Answers are grounded in retrieved excerpts and cite their sources."
)

# ---------------------------------------------------------------------------
# Sidebar settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Number of chunks to retrieve (top_k)", min_value=1, max_value=10, value=4)
    threshold = st.slider(
        "Minimum similarity threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        step=0.01,
        help="If the best-matching chunk scores below this, the system "
        "reports 'no relevant information' instead of calling the LLM.",
    )
    st.divider()
    st.caption("Index management")
    if st.button("Rebuild index from sample_docs/"):
        try:
            with st.spinner("Re-ingesting and re-embedding documents..."):
                rebuild_index()
            st.success("Index rebuilt.")
        except IngestError as e:
            st.error(f"Rebuild failed: {e}")

# ---------------------------------------------------------------------------
# Load index (with error handling for a missing/empty document collection)
# ---------------------------------------------------------------------------

store = None
load_error = None
try:
    store = cached_vector_store()
except (IngestError, RetrievalError) as e:
    load_error = str(e)

if load_error:
    st.error(
        f"Could not load the document index: {load_error}\n\n"
        f"Add .txt files to `sample_docs/` and click **Rebuild index** in the sidebar."
    )
    st.stop()

st.caption(f"Index ready: {store.embeddings.shape[0]} chunks loaded.")

# ---------------------------------------------------------------------------
# Query input
# ---------------------------------------------------------------------------

query = st.text_input("Ask a question about your documents", placeholder="e.g. What does the warranty cover?")
submitted = st.button("Search", type="primary")

if submitted:
    if not query or not query.strip():
        st.warning("Please enter a question before searching.")
        st.stop()

    try:
        with st.spinner("Retrieving relevant chunks..."):
            model = cached_embedding_model()
            query_vec = embed_texts(model, [query])[0]
            results = store.search(query_vec, top_k)
    except RetrievalError as e:
        st.error(f"Retrieval failed: {e}")
        st.stop()
    except Exception as e:  # noqa: BLE001 - surface unexpected embedding/runtime errors
        st.error(f"Unexpected error during retrieval: {e}")
        st.stop()

    best_score = results[0].score if results else 0.0
    passed_threshold = bool(results) and best_score >= threshold

    st.subheader("Answer")
    if not passed_threshold:
        st.info(GRACEFUL_FAILURE_MESSAGE)
        answer = GRACEFUL_FAILURE_MESSAGE
    else:
        with st.spinner("Generating grounded answer..."):
            answer = generate_answer(query, results)
        st.write(answer)

    st.subheader("Sources")
    if not results:
        st.caption("No chunks were retrieved for this query.")
    else:
        for i, r in enumerate(results, start=1):
            below = r.score < threshold
            label = f"Source {i}: {r.filename} (score: {r.score:.3f})"
            if below:
                label += "  — below threshold"
            with st.expander(label):
                st.write(r.text)
else:
    st.caption("Enter a question above and click Search to get a grounded, cited answer.")
