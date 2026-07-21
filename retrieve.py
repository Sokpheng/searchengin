"""
retrieve.py
-----------
Stages 2-4 of the RAG pipeline: Embed, Vector Store, Retrieve.

Uses a local sentence-transformers model (all-MiniLM-L6-v2) to embed chunks
and queries, and performs cosine similarity search over an in-memory numpy
matrix. No API key or external service is required for this stage.

Run directly to (re)build the vector index from data/chunks.json:
    python retrieve.py
"""

import os
import json
import pickle
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from ingest import load_chunks, IngestError

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = os.path.join("data", "index.pkl")
CHUNKS_PATH = os.path.join("data", "chunks.json")


class RetrievalError(Exception):
    """Raised when retrieval cannot proceed (missing index, bad query, etc.)."""


@dataclass
class SearchResult:
    filename: str
    chunk_index: int
    text: str
    score: float


# --------------------------------------------------------------------------
# Embedding model
# --------------------------------------------------------------------------

_model_singleton = None


def get_embedding_model():
    """Load (and cache in-process) the sentence-transformers model.

    Streamlit's own @st.cache_resource wraps a call to this function in
    app.py so the model is loaded once per server process, not once per
    request. This module-level singleton is a second safety net for any
    non-Streamlit callers (e.g. this file's __main__ block).
    """
    global _model_singleton
    if _model_singleton is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RetrievalError(
                "sentence-transformers is not installed. "
                "Run `pip install -r requirements.txt`."
            ) from e
        _model_singleton = SentenceTransformer(MODEL_NAME)
    return _model_singleton


def embed_texts(model, texts: List[str]) -> np.ndarray:
    """Embed a list of strings and L2-normalize so dot product == cosine similarity."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10  # avoid divide-by-zero for a degenerate empty chunk
    return (vectors / norms).astype(np.float32)


# --------------------------------------------------------------------------
# Vector store
# --------------------------------------------------------------------------

class VectorStore:
    """A minimal in-memory vector store using cosine similarity.

    Embeddings are stored L2-normalized, so cosine similarity between a
    query and every chunk reduces to a single matrix-vector dot product --
    fast enough for the few-thousand-chunk scale this project targets.
    """

    def __init__(self, embeddings: np.ndarray, metadata: List[dict]):
        if embeddings.shape[0] != len(metadata):
            raise ValueError("embeddings and metadata must be the same length")
        self.embeddings = embeddings
        self.metadata = metadata

    def search(self, query_vector: np.ndarray, top_k: int) -> List[SearchResult]:
        if self.embeddings.shape[0] == 0:
            return []
        scores = self.embeddings @ query_vector  # cosine similarity (both normalized)
        top_k = min(top_k, len(scores))
        top_idx = np.argsort(-scores)[:top_k]
        results = []
        for idx in top_idx:
            meta = self.metadata[idx]
            results.append(
                SearchResult(
                    filename=meta["filename"],
                    chunk_index=meta["chunk_index"],
                    text=meta["text"],
                    score=float(scores[idx]),
                )
            )
        return results

    def save(self, path: str = INDEX_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"embeddings": self.embeddings, "metadata": self.metadata}, f)

    @classmethod
    def load(cls, path: str = INDEX_PATH) -> "VectorStore":
        if not os.path.exists(path):
            raise RetrievalError(
                f"No index found at '{path}'. Run `python retrieve.py` (or ingest "
                f"+ build from the app) first."
            )
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(data["embeddings"], data["metadata"])


# --------------------------------------------------------------------------
# Build / load orchestration
# --------------------------------------------------------------------------

def build_index(chunks_path: str = CHUNKS_PATH, index_path: str = INDEX_PATH) -> VectorStore:
    """Embed every chunk in chunks_path and build+save a VectorStore."""
    chunks = load_chunks(chunks_path)
    model = get_embedding_model()
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(model, texts)
    store = VectorStore(embeddings, chunks)
    store.save(index_path)
    return store


def build_or_load_index(
    chunks_path: str = CHUNKS_PATH, index_path: str = INDEX_PATH
) -> VectorStore:
    """Load the saved index if present, otherwise build it from chunks.json.

    This is the entry point the Streamlit app uses so a first-time run
    doesn't require a separate manual `python retrieve.py` step, as long as
    ingestion has already produced data/chunks.json.
    """
    if os.path.exists(index_path):
        return VectorStore.load(index_path)
    return build_index(chunks_path, index_path)


def search(query: str, top_k: int, store: VectorStore = None) -> List[SearchResult]:
    """Embed a query and return its top_k most similar chunks."""
    if not query or not query.strip():
        raise RetrievalError("Query is empty.")
    if store is None:
        store = build_or_load_index()
    model = get_embedding_model()
    query_vec = embed_texts(model, [query])[0]
    return store.search(query_vec, top_k)


if __name__ == "__main__":
    try:
        store = build_index()
    except (IngestError, RetrievalError) as e:
        print(f"Index build failed: {e}")
        raise SystemExit(1)

    print(f"Embedded {store.embeddings.shape[0]} chunk(s) with '{MODEL_NAME}'.")
    print(f"Saved index to {INDEX_PATH}")
