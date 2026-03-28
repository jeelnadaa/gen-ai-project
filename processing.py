"""
processing.py - Core NLP processing functions.

Each public function accepts a ModelBundle and raw text, and returns a
structured result.  Functions are stateless so they can be tested in
isolation.
"""

import logging
from typing import Any

import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from models import ModelBundle

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SIMPLIFY_PREFIX = (
    "Simplify the following legal clause into plain, easy-to-understand English: "
)
_EXPLAIN_PREFIX = (
    "Explain in simple terms what the following legal clause means for a "
    "non-lawyer: "
)
_IMPORTANCE_LABELS = ["legally important", "not legally important"]
_IMPORTANCE_THRESHOLD = 0.60  # probability above which a clause is "IMPORTANT"

_MAX_INPUT_TOKENS = 512   # BART's practical limit
_MAX_OUTPUT_TOKENS = 150


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bart_generate(
    tokenizer: Any,
    model: Any,
    prompt: str,
    max_new_tokens: int = _MAX_OUTPUT_TOKENS,
) -> str:
    """Run a single seq2seq generation pass with BART."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=_MAX_INPUT_TOKENS,
        truncation=True,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Public processing functions
# ---------------------------------------------------------------------------

def simplify_clause(clause: str, bundle: ModelBundle) -> str:
    """
    Rewrite *clause* in plain English using facebook/bart-large.

    Args:
        clause: Original legal clause text.
        bundle: Loaded :class:`ModelBundle`.

    Returns:
        Simplified clause string.
    """
    prompt = _SIMPLIFY_PREFIX + clause
    result = _bart_generate(
        bundle.simplifier_tokenizer,
        bundle.simplifier_model,
        prompt,
    )
    logger.debug("Simplified: %s -> %s", clause[:60], result[:60])
    return result


def explain_clause(clause: str, bundle: ModelBundle) -> str:
    """
    Generate a plain-English explanation of *clause*.

    Args:
        clause: Original legal clause text.
        bundle: Loaded :class:`ModelBundle`.

    Returns:
        Explanation string.
    """
    prompt = _EXPLAIN_PREFIX + clause
    result = _bart_generate(
        bundle.simplifier_tokenizer,
        bundle.simplifier_model,
        prompt,
        max_new_tokens=200,
    )
    logger.debug("Explanation generated for clause (first 60 chars): %s", clause[:60])
    return result


def detect_importance(clause: str, bundle: ModelBundle) -> str:
    """
    Classify whether *clause* is legally important using zero-shot classification.

    Uses facebook/bart-large-mnli with the hypothesis labels:
    ``["legally important", "not legally important"]``.

    Args:
        clause: Clause text (original or simplified).
        bundle: Loaded :class:`ModelBundle`.

    Returns:
        ``"IMPORTANT"`` or ``"NORMAL"``.
    """
    result = bundle.classifier_pipeline(
        clause[:512],  # NLI models have token limits
        candidate_labels=_IMPORTANCE_LABELS,
        hypothesis_template="This clause is {}.",
    )
    # result["labels"][0] is the highest-scoring label
    top_label: str = result["labels"][0]
    top_score: float = result["scores"][0]

    logger.debug(
        "Importance - label: %s, score: %.3f", top_label, top_score
    )

    if top_label == "legally important" and top_score >= _IMPORTANCE_THRESHOLD:
        return "IMPORTANT"
    return "NORMAL"


def compute_semantic_similarity(
    text_a: str, text_b: str, bundle: ModelBundle
) -> float:
    """
    Compute cosine similarity between the embeddings of two texts.

    Args:
        text_a: First text (e.g. original clause).
        text_b: Second text (e.g. simplified clause).
        bundle: Loaded :class:`ModelBundle`.

    Returns:
        Cosine similarity score in ``[0.0, 1.0]``.
    """
    embeddings = bundle.embedding_model.encode(
        [text_a, text_b], convert_to_numpy=True
    )
    score: float = float(
        cosine_similarity(embeddings[0:1], embeddings[1:2])[0][0]
    )
    # Clamp to [0, 1] to avoid floating-point edge cases
    return max(0.0, min(1.0, score))


def summarize_document(text: str, bundle: ModelBundle) -> str:
    """
    Generate an abstractive summary of *text* using facebook/bart-large-cnn.

    The text is truncated to 3 000 characters to stay within BART-CNN's
    practical context window.  We call the tokenizer + model directly because
    the ``"summarization"`` pipeline task was removed in recent transformers
    versions.

    Args:
        text: Full document text.
        bundle: Loaded :class:`ModelBundle`.

    Returns:
        Summary string.
    """
    truncated = text[:3_000]
    logger.info("Summarising document (%d characters after truncation).", len(truncated))

    inputs = bundle.summarizer_tokenizer(
        truncated,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
    ).to(bundle.summarizer_model.device)

    with torch.no_grad():
        output_ids = bundle.summarizer_model.generate(
            **inputs,
            max_new_tokens=250,
            min_new_tokens=60,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=2.0,
        )

    summary: str = bundle.summarizer_tokenizer.decode(
        output_ids[0], skip_special_tokens=True
    )
    logger.info("Summary generated (%d characters).", len(summary))
    return summary


# ---------------------------------------------------------------------------
# Batch processor
# ---------------------------------------------------------------------------

def process_clauses(clauses: list[str], bundle: ModelBundle) -> list[dict]:
    """
    Run the full per-clause pipeline (simplify → explain → importance → similarity).

    Args:
        clauses: List of raw clause strings extracted from the document.
        bundle:  Loaded :class:`ModelBundle`.

    Returns:
        List of dicts, one per clause, matching the output JSON schema.
    """
    results: list[dict] = []
    total = len(clauses)

    for idx, clause in enumerate(clauses, start=1):
        logger.info("Processing clause %d / %d …", idx, total)

        simplified   = simplify_clause(clause, bundle)
        explanation  = explain_clause(clause, bundle)
        importance   = detect_importance(clause, bundle)
        similarity   = compute_semantic_similarity(clause, simplified, bundle)

        results.append(
            {
                "original":           clause,
                "simplified":         simplified,
                "explanation":        explanation,
                "importance":         importance,
                "semantic_similarity": round(similarity, 4),
            }
        )

    return results