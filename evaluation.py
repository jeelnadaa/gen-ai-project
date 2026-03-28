"""
evaluation.py - NLP evaluation metrics for the Legal Clause Simplifier.

Metrics computed:
    - BLEU score        : simplification quality (original vs simplified)
    - ROUGE scores      : summarisation quality (generated vs reference summary)
    - Cosine similarity : average semantic similarity across all clauses
    - Accuracy / Precision / Recall / F1 : importance classification quality
"""

import logging
from typing import Any

import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BLEU
# ---------------------------------------------------------------------------

def compute_bleu(
    originals: list[str],
    simplifications: list[str],
) -> float:
    """
    Corpus-level BLEU score treating the *original* clause as the reference
    and the *simplified* clause as the hypothesis.

    A perfect simplification is not expected to be identical to the original,
    so BLEU here measures lexical overlap rather than correctness.  Use it as
    a relative measure across system variations.

    Args:
        originals:       List of original clause strings.
        simplifications: List of simplified clause strings (same order).

    Returns:
        BLEU score in ``[0.0, 1.0]``.
    """
    if len(originals) != len(simplifications):
        raise ValueError("originals and simplifications must have the same length.")

    references = [[ref.lower().split()] for ref in originals]
    hypotheses = [hyp.lower().split() for hyp in simplifications]

    smoother = SmoothingFunction().method4
    score: float = corpus_bleu(references, hypotheses, smoothing_function=smoother)
    logger.info("BLEU score: %.4f", score)
    return round(score, 4)


# ---------------------------------------------------------------------------
# ROUGE
# ---------------------------------------------------------------------------

def compute_rouge(
    generated_summary: str,
    reference_summary: str,
) -> dict[str, float]:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.

    Args:
        generated_summary:  Model-generated document summary.
        reference_summary:  Human-written or gold-standard summary.

    Returns:
        Dict with keys ``rouge1``, ``rouge2``, ``rougeL``.
    """
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    scores = scorer.score(reference_summary, generated_summary)

    result = {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }
    logger.info("ROUGE scores: %s", result)
    return result


# ---------------------------------------------------------------------------
# Semantic similarity (aggregate)
# ---------------------------------------------------------------------------

def aggregate_similarity(clause_results: list[dict]) -> float:
    """
    Compute the mean cosine similarity across all processed clauses.

    Args:
        clause_results: Output from :func:`processing.process_clauses`.

    Returns:
        Mean cosine similarity score.
    """
    scores = [r["semantic_similarity"] for r in clause_results]
    mean_score = float(np.mean(scores)) if scores else 0.0
    logger.info("Mean semantic similarity: %.4f", mean_score)
    return round(mean_score, 4)


# ---------------------------------------------------------------------------
# Classification metrics (importance detection)
# ---------------------------------------------------------------------------

def compute_classification_metrics(
    predicted_labels: list[str],
    ground_truth_labels: list[str],
) -> dict[str, float]:
    """
    Compute accuracy, precision, recall, and F1 for importance classification.

    Both *predicted_labels* and *ground_truth_labels* should contain only
    the strings ``"IMPORTANT"`` or ``"NORMAL"``.

    Args:
        predicted_labels:   Labels output by :func:`processing.detect_importance`.
        ground_truth_labels: Human-annotated ground-truth labels.

    Returns:
        Dict with keys ``accuracy``, ``precision``, ``recall``, ``f1``.
    """
    if len(predicted_labels) != len(ground_truth_labels):
        raise ValueError(
            "predicted_labels and ground_truth_labels must have the same length."
        )

    # Encode as binary (IMPORTANT=1, NORMAL=0)
    label_map = {"IMPORTANT": 1, "NORMAL": 0}
    y_pred = [label_map.get(l, 0) for l in predicted_labels]
    y_true = [label_map.get(l, 0) for l in ground_truth_labels]

    metrics = {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(
            precision_score(y_true, y_pred, zero_division=0), 4
        ),
        "recall":    round(
            recall_score(y_true, y_pred, zero_division=0), 4
        ),
        "f1":        round(
            f1_score(y_true, y_pred, zero_division=0), 4
        ),
    }

    logger.info("Classification metrics: %s", metrics)

    report = classification_report(
        y_true,
        y_pred,
        target_names=["NORMAL", "IMPORTANT"],
        zero_division=0,
    )
    logger.debug("Full classification report:\n%s", report)

    return metrics


# ---------------------------------------------------------------------------
# Top-level evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(
    clause_results: list[dict],
    generated_summary: str,
    reference_summary: str | None = None,
    ground_truth_labels: list[str] | None = None,
) -> dict:
    """
    Run all evaluation metrics and return a combined evaluation dict.

    Args:
        clause_results:      Output of :func:`processing.process_clauses`.
        generated_summary:   Model-generated document summary.
        reference_summary:   Optional gold-standard summary for ROUGE.
        ground_truth_labels: Optional list of true importance labels for
                             classification metrics.

    Returns:
        Evaluation dict matching the output JSON schema.
    """
    originals      = [r["original"]    for r in clause_results]
    simplifications = [r["simplified"] for r in clause_results]
    predicted_labels = [r["importance"] for r in clause_results]

    evaluation: dict[str, Any] = {}

    # BLEU
    evaluation["bleu"] = compute_bleu(originals, simplifications)

    # ROUGE (requires a reference summary)
    if reference_summary:
        evaluation["rouge"] = compute_rouge(generated_summary, reference_summary)
    else:
        # Self-ROUGE: use the first 500 chars of the document as a proxy reference
        proxy_reference = " ".join(originals[:5])[:500]
        evaluation["rouge"] = compute_rouge(generated_summary, proxy_reference)
        logger.warning(
            "No reference summary provided; ROUGE computed against a proxy."
        )

    # Semantic similarity
    evaluation["mean_semantic_similarity"] = aggregate_similarity(clause_results)

    # Classification metrics (requires ground truth)
    if ground_truth_labels:
        evaluation.update(
            compute_classification_metrics(predicted_labels, ground_truth_labels)
        )
    else:
        logger.warning(
            "No ground-truth labels provided; classification metrics skipped."
        )
        evaluation["accuracy"]  = None
        evaluation["precision"] = None
        evaluation["recall"]    = None
        evaluation["f1"]        = None

    return evaluation