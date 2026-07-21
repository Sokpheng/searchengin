"""
generate.py
-----------
Stage 5 of the RAG pipeline: Generation.

Builds a grounding prompt from retrieved chunks and calls an LLM (Gemini by
default, OpenAI as a fallback/alternative) to produce a cited answer. This
module does NOT decide whether retrieval quality was good enough to answer
-- that threshold check happens in app.py before generate_answer() is ever
called, per the "graceful failure" requirement.
"""

import os
from typing import List

from retrieve import SearchResult

GRACEFUL_FAILURE_MESSAGE = "No relevant information found in the document collection."

SYSTEM_INSTRUCTIONS = """You are a document search assistant. You must answer the \
user's question using ONLY the numbered source excerpts provided below. \
Do not use any outside knowledge, and do not make anything up.

Rules:
1. Base your answer strictly on the provided excerpts.
2. Every claim in your answer must be followed by an inline citation to the \
source file it came from, in the format [Filename.txt].
3. If different excerpts disagree, note the disagreement and cite each side.
4. If the excerpts do not actually contain enough information to answer the \
question, say so plainly instead of guessing.
5. Keep the answer concise and directly responsive to the question."""


class GenerationError(Exception):
    """Raised when no LLM backend is configured or the API call fails."""


def build_prompt(query: str, chunks: List[SearchResult]) -> str:
    """Assemble the grounding prompt from the query and retrieved chunks."""
    excerpt_blocks = []
    for i, c in enumerate(chunks, start=1):
        excerpt_blocks.append(
            f"[Source {i}: {c.filename}] (similarity: {c.score:.2f})\n{c.text}"
        )
    excerpts_text = "\n\n".join(excerpt_blocks)

    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"--- SOURCE EXCERPTS ---\n{excerpts_text}\n\n"
        f"--- QUESTION ---\n{query}\n\n"
        f"--- ANSWER (with inline [Filename.txt] citations) ---"
    )


def _call_gemini(prompt: str) -> str:
    import google.generativeai as genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise GenerationError("GEMINI_API_KEY is not set.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-3.5-flash')
    response = model.generate_content(prompt)
    text = getattr(response, "text", None)
    if not text:
        raise GenerationError("Gemini returned an empty response.")
    return text.strip()


def _call_openai(prompt: str) -> str:
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise GenerationError("OPENAI_API_KEY is not set.")
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    text = response.choices[0].message.content
    if not text:
        raise GenerationError("OpenAI returned an empty response.")
    return text.strip()


def call_llm(prompt: str) -> str:
    """Call whichever LLM backend has an API key configured.

    Tries Gemini first, then falls back to OpenAI. Raises GenerationError if
    neither is configured or both calls fail, so the caller can surface a
    clear error in the UI instead of crashing.
    """
    errors = []
    if os.environ.get("GEMINI_API_KEY"):
        try:
            return _call_gemini(prompt)
        except Exception as e:  # noqa: BLE001 - surface any backend failure uniformly
            errors.append(f"Gemini: {e}")

    if os.environ.get("OPENAI_API_KEY"):
        try:
            return _call_openai(prompt)
        except Exception as e:  # noqa: BLE001
            errors.append(f"OpenAI: {e}")

    if not errors:
        raise GenerationError(
            "No LLM API key configured. Set GEMINI_API_KEY or OPENAI_API_KEY "
            "as an environment variable."
        )
    raise GenerationError("All configured LLM backends failed: " + " | ".join(errors))


def generate_answer(query: str, chunks: List[SearchResult]) -> str:
    """Produce a grounded, cited answer from already-retrieved chunks.

    Callers are expected to have already applied the similarity threshold
    check (see app.py) -- this function assumes `chunks` is a non-empty,
    pre-filtered list worth answering from.
    """
    if not chunks:
        return GRACEFUL_FAILURE_MESSAGE
    prompt = build_prompt(query, chunks)
    try:
        return call_llm(prompt)
    except GenerationError as e:
        # Fail gracefully in the UI rather than raising past app.py.
        return f"⚠️ Could not generate an answer: {e}"
