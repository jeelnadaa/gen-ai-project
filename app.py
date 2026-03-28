"""
app.py - Main pipeline entry point for the Legal Clause Simplifier.

Usage:
    python app.py --pdf contract.pdf [options]

API key setup — create a .env file in the project root:
    GROQ_API_KEY=gsk_your_key_here

Get a free key (no credit card): https://console.groq.com
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from utils import extract_text_from_pdf, split_into_clauses
from models import load_models
from processing import process_clauses, summarize_document
from evaluation import run_evaluation


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="legal_clause_simplifier",
        description=(
            "Simplify and analyse clauses in a legal PDF using "
            "Llama 3.3 70B via Groq. API key is read from .env file."
        ),
    )
    parser.add_argument(
        "--pdf", required=True, metavar="PATH",
        help="Path to the input PDF file.",
    )
    parser.add_argument(
        "--output", default="output.json", metavar="PATH",
        help="Path for the output JSON file (default: output.json).",
    )
    parser.add_argument(
        "--groq-model", default="llama-3.3-70b-versatile", metavar="MODEL",
        help=(
            "Groq model ID (default: llama-3.3-70b-versatile). "
            "Alternatives: llama-3.1-8b-instant, gemma2-9b-it."
        ),
    )
    parser.add_argument(
        "--max-clauses", type=int, default=None, metavar="N",
        help="Process at most N clauses (useful for quick tests).",
    )
    parser.add_argument(
        "--reference-summary", default=None, metavar="TEXT",
        help="Optional reference summary text for ROUGE evaluation.",
    )
    parser.add_argument(
        "--ground-truth-labels", default=None, metavar="LABEL,...",
        help=(
            "Comma-separated ground-truth importance labels "
            "(IMPORTANT or NORMAL), one per clause."
        ),
    )
    parser.add_argument(
        "--min-clause-length", type=int, default=30, metavar="N",
        help="Minimum character length to keep a clause (default: 30).",
    )
    return parser


def run_pipeline(args: argparse.Namespace) -> dict:
    t0 = time.perf_counter()

    logger.info("═" * 60)
    logger.info("STEP 1 / 5 — Extracting text from PDF")
    logger.info("═" * 60)
    raw_text = extract_text_from_pdf(args.pdf)

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

    logger.info("═" * 60)
    logger.info("STEP 3 / 5 — Loading models")
    logger.info("═" * 60)
    bundle = load_models(groq_model=args.groq_model)

    logger.info("═" * 60)
    logger.info("STEP 4 / 5 — Processing %d clauses via Groq API", len(clauses))
    logger.info("═" * 60)
    clause_results = process_clauses(clauses, bundle)

    logger.info("Generating document summary …")
    summary = summarize_document(raw_text, bundle)

    logger.info("═" * 60)
    logger.info("STEP 5 / 5 — Evaluating results")
    logger.info("═" * 60)

    ground_truth_labels: list[str] | None = None
    if args.ground_truth_labels:
        ground_truth_labels = [
            lbl.strip().upper()
            for lbl in args.ground_truth_labels.split(",")
        ]
        if len(ground_truth_labels) != len(clause_results):
            logger.warning(
                "ground-truth-labels count (%d) != clause count (%d). "
                "Classification metrics skipped.",
                len(ground_truth_labels), len(clause_results),
            )
            ground_truth_labels = None

    evaluation = run_evaluation(
        clause_results=clause_results,
        generated_summary=summary,
        reference_summary=args.reference_summary,
        ground_truth_labels=ground_truth_labels,
    )

    output = {
        "clauses":    clause_results,
        "summary":    summary,
        "evaluation": evaluation,
    }

    elapsed = time.perf_counter() - t0
    logger.info("Pipeline completed in %.1f seconds.", elapsed)
    return output


def save_output(data: dict, path: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    logger.info("Output saved to: %s", out_path.resolve())


def process_pdf(
    pdf_path: str,
    groq_model: str | None = None,
    max_clauses: int | None = None,
    reference_summary: str | None = None,
    ground_truth_labels: list[str] | None = None,
    min_clause_length: int = 30,
) -> dict:
    """Process PDF and return the result dict (same as CLI output)."""
    args = argparse.Namespace(
        pdf=pdf_path,
        output="output.json",
        groq_model=groq_model or "llama-3.3-70b-versatile",
        max_clauses=max_clauses,
        reference_summary=reference_summary,
        ground_truth_labels=','.join(ground_truth_labels) if ground_truth_labels else None,
        min_clause_length=min_clause_length,
    )
    return run_pipeline(args)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    logger.info("Legal Clause Simplifier — starting")
    logger.info("PDF    : %s", args.pdf)
    logger.info("Output : %s", args.output)
    logger.info("Model  : %s (via Groq)", args.groq_model)

    try:
        result = run_pipeline(args)
        save_output(result, args.output)
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"  Clauses processed : {len(result['clauses'])}")
        print(f"  Summary length    : {len(result['summary'])} characters")
        print(f"  BLEU              : {result['evaluation'].get('bleu')}")
        print(f"  Mean similarity   : {result['evaluation'].get('mean_semantic_similarity')}")
        print(f"  Output            : {args.output}")
        print("=" * 60 + "\n")
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()