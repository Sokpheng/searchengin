# RAG Document Search System

A Retrieval-Augmented Generation (RAG) search system. You ask a question,
the system retrieves the most relevant chunks from your document
collection using real embeddings, and an LLM generates an answer grounded
in those chunks with inline citations. If nothing relevant is found, it
says so instead of guessing.

## Architecture

| Stage | Module | What it does |
|---|---|---|
| 1. Ingest & Chunk | `ingest.py` | Loads `.txt` files from `sample_docs/`, splits them into 400-word chunks with 50-word overlap |
| 2. Embed | `retrieve.py` | Encodes chunks with `sentence-transformers` (`all-MiniLM-L6-v2`, local, free, no API key) |
| 3. Vector Store | `retrieve.py` | In-memory cosine-similarity index (normalized vectors, numpy dot product) |
| 4. Retrieve | `retrieve.py` | Embeds the query and returns the top-k most similar chunks |
| 5. Generate | `generate.py` | Sends the query + retrieved chunks to Gemini (or OpenAI) with a strict grounding + citation prompt |
| 6. Interface | `app.py` | Streamlit UI: query box, adjustable top-k / threshold, Answer panel, Sources panel |

Each stage is a separate, independently testable module — ingestion,
retrieval, and generation contain no Streamlit code, and `app.py` only
wires them together.

## 1. Setup

Create and activate your virtual environment:

```bash
# Windows (PowerShell):
py -m venv venv
.\venv\Scripts\activate

# macOS / Linux:
python3 -m venv venv
source venv/bin/activate
Install dependencies:

Bash
# Windows:
py -m pip install -r requirements.txt

# macOS / Linux:
pip install -r requirements.txt
Set an API key for the generation step by creating a .env file in the root folder:

Code snippet
GEMINI_API_KEY="your-gemini-key"
# or
OPENAI_API_KEY="your-openai-key"
No key is needed for embeddings or retrieval — those run entirely locally.

2. Add your documents
Replace the sample files in sample_docs/ with your own .txt document
collection (20+ files, or equivalent chunked volume, per the assignment
brief). Three small sample docs are included so you can run the system
immediately before swapping in your real data.

3. Build the index
Run ingestion, then build the vector index:

Bash
# Windows:
py ingest.py     # loads sample_docs/, writes data/chunks.json
py retrieve.py   # embeds chunks, writes data/index.pkl

# macOS / Linux:
python3 ingest.py
python3 retrieve.py
You can also skip this step — app.py will automatically ingest and build
the index on first launch if data/chunks.json / data/index.pkl don't
exist yet, and the sidebar has a Rebuild index button for whenever you
swap in new documents.

4. Launch the app
Bash
# Windows:
py -m streamlit run app.py

# macOS / Linux:
streamlit run app.py
Open the URL Streamlit prints (usually http://localhost:8501).

Using the interface
top_k slider (sidebar) — how many chunks to retrieve per query (1–10).

Minimum similarity threshold slider (sidebar) — if the best-matching
chunk scores below this, the system returns "No relevant information
found in the document collection." instead of calling the LLM.

Query box + Search — ask a question about your documents.

Answer panel — the LLM's grounded, cited response.

Sources panel — every retrieved chunk in an expandable list, showing
its source filename, similarity score, and full text. Chunks that fell
below the threshold are labeled so you can see why they were excluded.

Known limitations
The in-memory vector store is fine for a few thousand chunks; it isn't
optimized for large-scale corpora (swap in FAISS/Chroma for that).

Chunking is word-count based, not sentence- or section-aware, so a chunk
can occasionally cut a sentence in half.

Generation quality depends on the configured LLM; if no API key is set,
the Answer panel will show a clear configuration error rather than
crashing the app.

Citation accuracy depends on the LLM following the prompt's citation
instructions — it is prompted to cite [Filename.txt] inline, but this
is not mechanically enforced or verified against the retrieved chunks.

Evaluation
Once you have real documents indexed, put your 5–10 test queries and a
short qualitative write-up (what retrieved well, what didn't, why) in a
separate EVALUATION.md — this starter project does not generate that
write-up for you, since it depends on your actual document collection.