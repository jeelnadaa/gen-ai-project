"""
processing.py - Core NLP processing functions using Groq LLM API.

All heavy NLP tasks (simplification, importance detection, summarisation) are
handled by llama-3.3-70b-versatile via the Groq API.
Semantic similarity uses a local sentence-transformer (all-mpnet-base-v2).
"""

import logging
import time

from sklearn.metrics.pairwise import cosine_similarity

from models import ModelBundle

logger = logging.getLogger(__name__)

_INTER_CALL_DELAY_SEC = 0.5
_IMPORTANT = "IMPORTANT"
_NORMAL    = "NORMAL"


def _llm(
    bundle: ModelBundle,
    system: str,
    user: str,
    max_tokens: int = 300,
    temperature: float = 0.1,
) -> str:
    """Send a chat completion request to Groq and return the response text."""
    response = bundle.groq_client.chat.completions.create(
        model=bundle.groq_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    time.sleep(_INTER_CALL_DELAY_SEC)
    return response.choices[0].message.content.strip()


def simplify_clause(clause: str, bundle: ModelBundle) -> str:
    """Rewrite a legal clause in plain English using Llama 3.3 70B via Groq."""
    system = (
        "You are a legal expert who specialises in making complex legal "
        "language easy to understand for ordinary people. "
        "Rewrite legal clauses in plain, simple English. "
        "Keep the same meaning but use everyday words. "
        "Be concise — 1 to 3 sentences maximum. "
        "Output ONLY the simplified text, nothing else."
    )
    user = f"Simplify this legal clause:\n\n{clause}"
    result = _llm(bundle, system, user, max_tokens=200)
    logger.debug("Simplified | in: %s… | out: %s…", clause[:50], result[:50])
    return result


def detect_importance(clause: str, bundle: ModelBundle) -> str:
    """Classify whether a clause is IMPORTANT or NORMAL using Llama 3.3 70B."""
    system = (
        "You are a legal analyst. Determine whether a legal clause is "
        "IMPORTANT or NORMAL.\n\n"
        "IMPORTANT clauses typically involve:\n"
        "  - Liability, indemnification, or damages\n"
        "  - Confidentiality or non-disclosure obligations\n"
        "  - Termination rights or conditions\n"
        "  - Intellectual property ownership or licensing\n"
        "  - Payment terms, penalties, or financial obligations\n"
        "  - Governing law, jurisdiction, or dispute resolution\n"
        "  - Non-compete or non-solicitation restrictions\n"
        "  - Representations, warranties, or guarantees\n\n"
        "NORMAL clauses are routine procedural or administrative text "
        "(e.g. definitions, notice addresses, formatting rules).\n\n"
        "Reply with ONLY one word: IMPORTANT or NORMAL."
    )
    user = f"Classify this clause:\n\n{clause}"
    result = _llm(bundle, system, user, max_tokens=10, temperature=0.0)
    label = result.strip().upper().rstrip(".")
    if "IMPORTANT" in label:
        return _IMPORTANT
    return _NORMAL


def compute_semantic_similarity(
    text_a: str, text_b: str, bundle: ModelBundle
) -> float:
    """Cosine similarity between sentence embeddings of two texts."""
    embeddings = bundle.embedding_model.encode(
        [text_a, text_b], convert_to_numpy=True
    )
    score = float(cosine_similarity(embeddings[0:1], embeddings[1:2])[0][0])
    return round(max(0.0, min(1.0, score)), 4)


def summarize_document(text: str, bundle: ModelBundle) -> str:
    """Generate an abstractive summary using Llama 3.3 70B via Groq."""
    truncated = text[:6_000]
    logger.info("Summarising document (%d chars after truncation).", len(truncated))
    system = (
        "You are a legal expert. Summarise the following legal document in "
        "plain English for a non-lawyer. "
        "Cover the main purpose, key obligations, important rights, and any "
        "significant risks or restrictions. "
        "Write 3 to 5 clear sentences. Output ONLY the summary."
    )
    user = f"Summarise this legal document:\n\n{truncated}"
    summary = _llm(bundle, system, user, max_tokens=400)
    logger.info("Summary generated (%d chars).", len(summary))
    return summary


def process_clauses(
    clauses: list[str],
    bundle: ModelBundle,
    progress_callback=None,
) -> list[dict]:
    """
    Run the per-clause pipeline: simplify → importance → similarity.
    Returns list of dicts: original, simplified, importance, semantic_similarity.

    progress_callback, if provided, should accept (processed_count, total_count).
    """
    results: list[dict] = []
    total = len(clauses)

    for idx, clause in enumerate(clauses, start=1):
        logger.info("Processing clause %d / %d …", idx, total)
        if progress_callback is not None:
            try:
                progress_callback(idx, total)
            except Exception:
                logger.exception("progress_callback raised an exception")

        simplified = simplify_clause(clause, bundle)
        importance = detect_importance(clause, bundle)
        similarity = compute_semantic_similarity(clause, simplified, bundle)
        results.append({
            "original":            clause,
            "simplified":          simplified,
            "importance":          importance,
            "semantic_similarity": similarity,
        })

    return results