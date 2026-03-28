"""
app.py - Main pipeline entry point for the Legal Clause Simplifier.

Usage:
    python app.py --pdf path/to/contract.pdf [options]

Run ``python app.py --help`` for the full argument list.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging setup (call before importing project modules so their loggers
# inherit the root configuration)
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

from utils import extract_text_from_pdf, split_into_clauses
from models import load_models
from processing import process_clauses, summarize_document
from evaluation import run_evaluation


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="legal_clause_simplifier",
        description="Simplify, explain, and analyse clauses in a legal PDF.",
    )
    parser.add_argument(
        "--pdf",
        required=True,
        metavar="PATH",
        help="Path to the input PDF file.",
    )
    parser.add_argument(
        "--output",
        default="output.json",
        metavar="PATH",
        help="Path for the output JSON file (default: output.json).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run Torch models on (default: cpu).",
    )
    parser.add_argument(
        "--max-clauses",
        type=int,
        default=None,
        metavar="N",
        help="Process at most N clauses (useful for quick tests).",
    )
    parser.add_argument(
        "--reference-summary",
        default=None,
        metavar="TEXT",
        help="Optional reference summary text for ROUGE evaluation.",
    )
    parser.add_argument(
        "--ground-truth-labels",
        default=None,
        metavar="LABEL,...",
        help=(
            "Comma-separated ground-truth importance labels "
            "(IMPORTANT or NORMAL), one per clause, for classification eval."
        ),
    )
    parser.add_argument(
        "--min-clause-length",
        type=int,
        default=30,
        metavar="N",
        help="Minimum character length to keep a clause (default: 30).",
    )
    return parser


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> dict:
    """Execute the full Legal Clause Simplifier pipeline."""
    t0 = time.perf_counter()

    # ── 1. Extract text ───────────────────────────────────────────────────
    logger.info("═" * 60)
    logger.info("STEP 1 / 5 — Extracting text from PDF")
    logger.info("═" * 60)
    raw_text = extract_text_from_pdf(args.pdf)

    # ── 2. Split into clauses ─────────────────────────────────────────────
    logger.info("═" * 60)
    logger.info("STEP 2 / 5 — Splitting document into clauses")
    logger.info("═" * 60)
    clauses = split_into_clauses(raw_text, min_length=args.min_clause_length)

    if args.max_clauses:
        logger.info("Limiting to %d clauses as requested.", args.max_clauses)
        clauses = clauses[: args.max_clauses]

    if not clauses:
        logger.error("No clauses extracted — aborting.")
        sys.exit(1)

    # ── 3. Load models ────────────────────────────────────────────────────
    logger.info("═" * 60)
    logger.info("STEP 3 / 5 — Loading NLP models")
    logger.info("═" * 60)
    bundle = load_models(device=args.device)

    # ── 4. Process clauses ────────────────────────────────────────────────
    logger.info("═" * 60)
    logger.info("STEP 4 / 5 — Processing %d clauses", len(clauses))
    logger.info("═" * 60)
    clause_results = process_clauses(clauses, bundle)

    # ── 4b. Summarise document ────────────────────────────────────────────
    logger.info("Generating document summary …")
    summary = summarize_document(raw_text, bundle)

    # ── 5. Evaluate ───────────────────────────────────────────────────────
    logger.info("═" * 60)
    logger.info("STEP 5 / 5 — Evaluating results")
    logger.info("═" * 60)

    # Parse optional ground-truth labels
    ground_truth_labels: list[str] | None = None
    if args.ground_truth_labels:
        ground_truth_labels = [
            lbl.strip().upper() for lbl in args.ground_truth_labels.split(",")
        ]
        if len(ground_truth_labels) != len(clause_results):
            logger.warning(
                "ground-truth-labels count (%d) != clause count (%d). "
                "Classification metrics will be skipped.",
                len(ground_truth_labels),
                len(clause_results),
            )
            ground_truth_labels = None

    evaluation = run_evaluation(
        clause_results=clause_results,
        generated_summary=summary,
        reference_summary=args.reference_summary,
        ground_truth_labels=ground_truth_labels,
    )

    # ── Assemble output ───────────────────────────────────────────────────
    output = {
        "clauses":    clause_results,
        "summary":    summary,
        "evaluation": evaluation,
    }

    elapsed = time.perf_counter() - t0
    logger.info("Pipeline completed in %.1f seconds.", elapsed)
    return output


# ---------------------------------------------------------------------------
# Save output
# ---------------------------------------------------------------------------

def save_output(data: dict, path: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    logger.info("Output saved to: %s", out_path.resolve())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    logger.info("Legal Clause Simplifier — starting")
    logger.info("PDF:    %s", args.pdf)
    logger.info("Output: %s", args.output)
    logger.info("Device: %s", args.device)

    try:
        result = run_pipeline(args)
        save_output(result, args.output)
        # Print a compact preview to stdout
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"  Clauses processed : {len(result['clauses'])}")
        print(f"  Summary length    : {len(result['summary'])} characters")
        print(f"  BLEU              : {result['evaluation'].get('bleu')}")
        print(f"  Mean sim.         : {result['evaluation'].get('mean_semantic_similarity')}")
        print(f"  Output            : {args.output}")
        print("=" * 60 + "\n")
    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()