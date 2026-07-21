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
from dotenv import load_dotenv

load_dotenv()  
import os
import time
from functools import lru_cache
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


# Free-tier quota is bucketed per model -- the 429 payload names the quota
# "GenerateRequestsPerMinutePerProjectPerModel" -- so falling through this
# list on a rate limit buys a fresh allowance instead of waiting out one
# model's window. Ordered best-quality first; override via GEMINI_MODELS.
DEFAULT_GEMINI_MODELS = (
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
)
GEMINI_CASCADE_PASSES = 2


def _gemini_models() -> List[str]:
    """Model cascade, from GEMINI_MODELS (comma-separated) or the default."""
    configured = os.environ.get("GEMINI_MODELS", "")
    models = [m.strip() for m in configured.split(",") if m.strip()]
    return models or list(DEFAULT_GEMINI_MODELS)


def _retry_delay_seconds(exc: Exception) -> float:
    """How long the server asked us to wait, or 2s if it didn't say.

    google-genai exposes the error body as raw JSON, so the RetryInfo hint
    arrives as {"@type": ".../RetryInfo", "retryDelay": "2.78s"}.
    """
    details = getattr(exc, "details", None)
    if isinstance(details, dict):
        for item in details.get("error", {}).get("details", []) or []:
            if not isinstance(item, dict) or "RetryInfo" not in item.get("@type", ""):
                continue
            try:
                return float(str(item.get("retryDelay", "")).rstrip("s")) + 1.0
            except ValueError:
                break
    return 2.0


def _is_rate_limit(exc: Exception) -> bool:
    return getattr(exc, "code", None) == 429


@lru_cache(maxsize=1)
def _gemini_client(api_key: str):
    """One client per key -- reused across calls rather than rebuilt each time."""
    from google import genai

    return genai.Client(api_key=api_key)


def _call_gemini(prompt: str) -> str:
    from google.genai import errors as genai_errors

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise GenerationError("GEMINI_API_KEY is not set.")
    client = _gemini_client(api_key)

    models = _gemini_models()
    failures = {}

    for pass_num in range(1, GEMINI_CASCADE_PASSES + 1):
        rate_limited = []
        for name in models:
            try:
                response = client.models.generate_content(model=name, contents=prompt)
            except genai_errors.APIError as e:
                if _is_rate_limit(e):
                    failures[name] = "rate limited"
                    rate_limited.append(e)
                else:
                    failures[name] = f"{getattr(e, 'code', '?')} {getattr(e, 'status', '')}".strip()
                continue
            except Exception as e:  # noqa: BLE001 - a bad model name shouldn't sink the rest
                failures[name] = str(e).splitlines()[0][:120]
                continue
            text = getattr(response, "text", None)
            if not text:
                failures[name] = "empty response"
                continue
            return text.strip()

        # Sleeping and re-running the list only helps if every failure was a
        # quota one -- bad model names and empty responses won't self-heal.
        if len(rate_limited) < len(models) or pass_num == GEMINI_CASCADE_PASSES:
            break
        time.sleep(max(_retry_delay_seconds(e) for e in rate_limited))

    detail = "; ".join(f"{name}: {why}" for name, why in failures.items())
    raise GenerationError(
        f"All Gemini models failed ({detail}). Free-tier quota is per model "
        "per minute -- wait a minute, add more models to GEMINI_MODELS in "
        ".env, or set OPENAI_API_KEY to enable the fallback backend."
    )


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
