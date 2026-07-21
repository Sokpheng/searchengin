"""
ingest.py
---------
Stage 1 of the RAG pipeline: Ingest & Chunk.

Loads raw .txt documents from a source directory and splits them into
overlapping, fixed-size word chunks. Chunking is done by word count rather
than character count so that chunk boundaries roughly track sentence/
paragraph structure without needing a heavier NLP dependency.

Run directly to (re)build data/chunks.json from sample_docs/:
    python ingest.py
"""

import os
import json
import glob
from dataclasses import dataclass, asdict
from typing import List

SOURCE_DIR = "sample_docs"
OUTPUT_PATH = os.path.join("data", "chunks.json")
CHUNKS_PATH = OUTPUT_PATH  # This fixes the Streamlit import error!

# Chunking strategy: 400-word chunks with a 50-word overlap.
# Overlap prevents ideas that straddle a chunk boundary from being lost
# entirely from retrieval, at the cost of some redundancy in the index.
CHUNK_SIZE_WORDS = 400
CHUNK_OVERLAP_WORDS = 50


@dataclass
class Chunk:
    id: str
    filename: str
    chunk_index: int
    text: str
    word_count: int


class IngestError(Exception):
    """Raised when document ingestion cannot proceed."""


def load_documents(source_dir: str = SOURCE_DIR) -> List[dict]:
    """Load every .txt file in source_dir into memory.

    Returns a list of {"filename": str, "text": str} dicts.
    Raises IngestError if the directory is missing or contains no .txt files.
    """
    if not os.path.isdir(source_dir):
        raise IngestError(
            f"Source directory '{source_dir}' does not exist. "
            f"Create it and add .txt documents before running ingestion."
        )

    paths = sorted(glob.glob(os.path.join(source_dir, "*.txt")))
    if not paths:
        raise IngestError(
            f"No .txt files found in '{source_dir}'. Add your document "
            f"collection (20+ files recommended) and re-run ingestion."
        )

    documents = []
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()
        except OSError as e:
            # Skip unreadable files rather than aborting the whole run.
            print(f"Warning: could not read {path}: {e}")
            continue

        if not text:
            print(f"Warning: {path} is empty, skipping.")
            continue

        documents.append({"filename": os.path.basename(path), "text": text})

    if not documents:
        raise IngestError("All discovered .txt files were empty or unreadable.")

    return documents


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE_WORDS,
    overlap: int = CHUNK_OVERLAP_WORDS,
) -> List[str]:
    """Split text into overlapping word-count chunks.

    A sliding window over the word list: each chunk has `chunk_size` words,
    and consecutive chunks share `overlap` words so context isn't lost at
    the seams.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    if not words:
        return []

    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        window = words[start : start + chunk_size]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
    return chunks


def chunk_documents(documents: List[dict]) -> List[Chunk]:
    """Chunk every document and return a flat list of Chunk objects."""
    all_chunks: List[Chunk] = []
    for doc in documents:
        pieces = chunk_text(doc["text"])
        for i, piece in enumerate(pieces):
            all_chunks.append(
                Chunk(
                    id=f"{doc['filename']}::{i}",
                    filename=doc["filename"],
                    chunk_index=i,
                    text=piece,
                    word_count=len(piece.split()),
                )
            )
    return all_chunks


def save_chunks(chunks: List[Chunk], output_path: str = OUTPUT_PATH) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in chunks], f, ensure_ascii=False, indent=2)


def load_chunks(path: str = OUTPUT_PATH) -> List[dict]:
    if not os.path.exists(path):
        raise IngestError(
            f"No chunk file found at '{path}'. Run `python ingest.py` first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_ingestion(source_dir: str = SOURCE_DIR, output_path: str = OUTPUT_PATH) -> List[Chunk]:
    documents = load_documents(source_dir)
    chunks = chunk_documents(documents)
    if not chunks:
        raise IngestError("Chunking produced zero chunks from the loaded documents.")
    save_chunks(chunks, output_path)
    return chunks


if __name__ == "__main__":
    try:
        chunks = run_ingestion()
    except IngestError as e:
        print(f"Ingestion failed: {e}")
        raise SystemExit(1)

    doc_count = len({c.filename for c in chunks})
    print(f"Loaded {doc_count} document(s).")
    print(f"Produced {len(chunks)} chunk(s) "
          f"({CHUNK_SIZE_WORDS}-word chunks, {CHUNK_OVERLAP_WORDS}-word overlap).")
    print(f"Saved to {OUTPUT_PATH}")
