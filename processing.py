"""
processing.py - Core NLP processing functions.

Explanation field has been removed.
Per-clause output: original, simplified, importance, semantic_similarity.
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

# Prompt prefix for bart-large-xsum:
# The model expects natural prose; prepending a clear instruction
# significantly improves output quality on domain-specific text.
_SIMPLIFY_PREFIX = (
    "Summarize the following legal clause in plain, simple English "
    "so that anyone can understand it: "
)

# DeBERTa NLI labels — must match exactly what the cross-encoder expects
_IMPORTANCE_LABELS = ["legally important", "not legally important"]

# Confidence threshold: score for "legally important" must exceed this
# to be classified as IMPORTANT.  DeBERTa is well-calibrated so 0.65 is
# a good operating point balancing precision and recall.
_IMPORTANCE_THRESHOLD = 0.65

_MAX_INPUT_TOKENS = 512    # safe limit for both BART-xsum and DeBERTa
_MAX_OUTPUT_TOKENS = 120   # concise simplifications — legal clauses are dense


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _generate(
    tokenizer: Any,
    model: Any,
    prompt: str,
    max_new_tokens: int = _MAX_OUTPUT_TOKENS,
    num_beams: int = 5,
) -> str:
    """Run a single seq2seq generation pass (works for BART-xsum & PEGASUS)."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=_MAX_INPUT_TOKENS,
        truncation=True,
        padding=False,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=1.5,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Public processing functions
# ---------------------------------------------------------------------------

def simplify_clause(clause: str, bundle: ModelBundle) -> str:
    """
    Rewrite *clause* in plain English using facebook/bart-large-xsum.

    bart-large-xsum was fine-tuned on extreme summarisation and produces
    noticeably more fluent, concise rewrites than plain bart-large.

    Args:
        clause: Original legal clause text.
        bundle: Loaded :class:`ModelBundle`.

    Returns:
        Simplified clause string.
    """
    prompt = _SIMPLIFY_PREFIX + clause
    result = _generate(
        bundle.simplifier_tokenizer,
        bundle.simplifier_model,
        prompt,
    )
    logger.debug("Simplified | in: %s… | out: %s…", clause[:50], result[:50])
    return result


def detect_importance(clause: str, bundle: ModelBundle) -> str:
    """
    Classify whether *clause* is legally important using
    cross-encoder/nli-deberta-v3-large (zero-shot NLI).

    DeBERTa-v3-large achieves ~91 % accuracy on MNLI vs ~89.9 % for the
    previous bart-large-mnli, giving noticeably better importance detection.

    Args:
        clause: Clause text.
        bundle: Loaded :class:`ModelBundle`.

    Returns:
        ``"IMPORTANT"`` or ``"NORMAL"``.
    """
    result = bundle.classifier_pipeline(
        clause[:512],
        candidate_labels=_IMPORTANCE_LABELS,
        hypothesis_template="This text is {}.",
    )
    top_label: str  = result["labels"][0]
    top_score: float = result["scores"][0]

    logger.debug("Importance | label: %s | score: %.3f", top_label, top_score)

    if top_label == "legally important" and top_score >= _IMPORTANCE_THRESHOLD:
        return "IMPORTANT"
    return "NORMAL"


def compute_semantic_similarity(
    text_a: str, text_b: str, bundle: ModelBundle
) -> float:
    """
    Compute cosine similarity between sentence embeddings of two texts.

    Uses all-mpnet-base-v2 which scores higher on STS benchmarks than
    the previous all-MiniLM-L6-v2.

    Args:
        text_a: First text (original clause).
        text_b: Second text (simplified clause).
        bundle: Loaded :class:`ModelBundle`.

    Returns:
        Cosine similarity in ``[0.0, 1.0]``.
    """
    embeddings = bundle.embedding_model.encode(
        [text_a, text_b], convert_to_numpy=True
    )
    score = float(cosine_similarity(embeddings[0:1], embeddings[1:2])[0][0])
    return round(max(0.0, min(1.0, score)), 4)


def summarize_document(text: str, bundle: ModelBundle) -> str:
    """
    Generate an abstractive summary using google/pegasus-large.

    PEGASUS was pre-trained with the Gap-Sentence Generation objective,
    optimised specifically for abstractive summarisation.

    Args:
        text: Full document text.
        bundle: Loaded :class:`ModelBundle`.

    Returns:
        Summary string.
    """
    # PEGASUS input limit is 1 024 tokens; ~4 000 chars is a safe proxy
    truncated = text[:4_000]
    logger.info("Summarising document (%d chars after truncation).", len(truncated))

    inputs = bundle.summarizer_tokenizer(
        truncated,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
        padding=False,
    ).to(bundle.summarizer_model.device)

    with torch.no_grad():
        output_ids = bundle.summarizer_model.generate(
            **inputs,
            max_new_tokens=280,
            min_new_tokens=60,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=1.8,
        )

    summary = bundle.summarizer_tokenizer.decode(
        output_ids[0], skip_special_tokens=True
    )
    logger.info("Summary generated (%d chars).", len(summary))
    return summary


# ---------------------------------------------------------------------------
# Batch processor
# ---------------------------------------------------------------------------

def process_clauses(clauses: list[str], bundle: ModelBundle) -> list[dict]:
    """
    Run the per-clause pipeline: simplify → importance → similarity.

    The explanation step has been removed.

    Args:
        clauses: Raw clause strings from the document.
        bundle:  Loaded :class:`ModelBundle`.

    Returns:
        List of dicts with keys:
        ``original``, ``simplified``, ``importance``, ``semantic_similarity``.
    """
    results: list[dict] = []
    total = len(clauses)

    for idx, clause in enumerate(clauses, start=1):
        logger.info("Processing clause %d / %d …", idx, total)

        simplified  = simplify_clause(clause, bundle)
        importance  = detect_importance(clause, bundle)
        similarity  = compute_semantic_similarity(clause, simplified, bundle)

        results.append(
            {
                "original":            clause,
                "simplified":          simplified,
                "importance":          importance,
                "semantic_similarity": similarity,
            }
        )

    return results