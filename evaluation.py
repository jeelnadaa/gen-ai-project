"""
evaluation.py - NLP evaluation metrics for the Legal Clause Simplifier.
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


def compute_bleu(reference: str | list[str], hypothesis: str | list[str]) -> float:
    """
    Corpus-level BLEU score.
    If input is list[str], compute aggregate across clauses.
    If input is str, compute for the summary text.
    """
    if isinstance(reference, str):
        # Comparison of single texts (e.g. Executive Summary)
        refs = [[reference.lower().split()]]
        hyps = [hypothesis.lower().split()]
    else:
        # Comparison of multiple segments (e.g. Clauses)
        if len(reference) != len(hypothesis):
            raise ValueError("Reference and hypothesis lists must have the same length.")
        refs = [[r.lower().split()] for r in reference]
        hyps = [h.lower().split() for h in hypothesis]

    smoother = SmoothingFunction().method4
    score: float = corpus_bleu(refs, hyps, smoothing_function=smoother)
    logger.info("BLEU score: %.4f", score)
    return round(score, 4)


def compute_rouge(generated_summary: str, reference_summary: str) -> dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference_summary, generated_summary)
    result = {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }
    logger.info("ROUGE scores: %s", result)
    return result


def aggregate_similarity(clause_results: list[dict]) -> float:
    """Mean cosine similarity across all processed clauses."""
    scores = [r["semantic_similarity"] for r in clause_results]
    mean_score = float(np.mean(scores)) if scores else 0.0
    logger.info("Mean semantic similarity: %.4f", mean_score)
    return round(mean_score, 4)


def compute_classification_metrics(
    predicted_labels: list[str],
    ground_truth_labels: list[str],
) -> dict[str, float]:
    """Accuracy, precision, recall, F1 for importance classification."""
    if len(predicted_labels) != len(ground_truth_labels):
        raise ValueError("predicted_labels and ground_truth_labels must have the same length.")
    label_map = {"IMPORTANT": 1, "NORMAL": 0}
    y_pred = [label_map.get(l, 0) for l in predicted_labels]
    y_true = [label_map.get(l, 0) for l in ground_truth_labels]
    metrics = {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
    logger.info("Classification metrics: %s", metrics)
    report = classification_report(
        y_true, y_pred, target_names=["NORMAL", "IMPORTANT"], zero_division=0
    )
    logger.debug("Full classification report:\n%s", report)
    return metrics


def run_evaluation(
    clause_results: list[dict],
    generated_summary: str,
    reference_summary: str | None = None,
    ground_truth_labels: list[str] | None = None,
) -> dict:
    """Run all evaluation metrics and return a combined evaluation dict."""
    originals        = [r["original"]   for r in clause_results]
    simplifications  = [r["simplified"] for r in clause_results]
    predicted_labels = [r["importance"] for r in clause_results]

    evaluation: dict[str, Any] = {}

    if reference_summary:
        # Evaluate Executive Summary against Reference Summary
        evaluation["bleu"]  = compute_bleu(reference_summary, generated_summary)
        evaluation["rouge"] = compute_rouge(generated_summary, reference_summary)
    else:
        # Fallback: Clause-level BLEU (sim-to-orig) & proxy ROUGE
        evaluation["bleu"] = compute_bleu(originals, simplifications)
        proxy_reference = " ".join(originals[:5])[:500]
        evaluation["rouge"] = compute_rouge(generated_summary, proxy_reference)
        logger.warning("No reference summary provided; BLEU (clause-level) and ROUGE (proxy) used.")

    evaluation["mean_semantic_similarity"] = aggregate_similarity(clause_results)

    if ground_truth_labels:
        evaluation.update(
            compute_classification_metrics(predicted_labels, ground_truth_labels)
        )
    else:
        logger.warning("No ground-truth labels provided; classification metrics skipped.")
        # Default to 0.0 to avoid 'null' in UI while maintaining data integrity
        evaluation["accuracy"]  = 0.0
        evaluation["precision"] = 0.0
        evaluation["recall"]    = 0.0
        evaluation["f1"]        = 0.0

    return evaluation