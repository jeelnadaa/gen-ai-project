"""
models.py - Centralised model loading for the Legal Clause Simplifier.

Model choices (upgraded for higher accuracy):
  - Simplification  : facebook/bart-large-xsum
  - Importance      : cross-encoder/nli-deberta-v3-large
  - Summarisation   : google/pegasus-large
  - Similarity      : sentence-transformers/all-mpnet-base-v2
"""

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ModelBundle:
    """Holds every model/tokenizer used by the pipeline."""

    # Simplification  (facebook/bart-large-xsum)
    simplifier_tokenizer: Any = field(default=None, repr=False)
    simplifier_model: Any = field(default=None, repr=False)

    # Zero-shot importance classification  (cross-encoder/nli-deberta-v3-large)
    classifier_pipeline: Any = field(default=None, repr=False)

    # Summarisation  (google/pegasus-large) — tokenizer + model, no pipeline
    summarizer_tokenizer: Any = field(default=None, repr=False)
    summarizer_model: Any = field(default=None, repr=False)

    # Semantic similarity embeddings  (all-mpnet-base-v2)
    embedding_model: Any = field(default=None, repr=False)


_BUNDLE: ModelBundle | None = None


def load_models(device: str = "cpu") -> ModelBundle:
    """
    Load (or return the cached) ModelBundle.

    Args:
        device: ``"cpu"`` or ``"cuda"`` / ``"cuda:0"`` etc.

    Returns:
        A fully populated :class:`ModelBundle`.
    """
    global _BUNDLE
    if _BUNDLE is not None:
        logger.info("Using cached model bundle.")
        return _BUNDLE

    from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        pipeline,
    )
    from sentence_transformers import SentenceTransformer

    bundle = ModelBundle()
    torch_device = 0 if device.startswith("cuda") else -1

    # --- 1. Simplification: facebook/bart-large-xsum -----------------------
    # Fine-tuned on extreme summarisation — produces concise, plain-English
    # rewrites rather than near-verbatim copies of the input.
    logger.info("Loading simplification model : facebook/bart-large-xsum")
    bundle.simplifier_tokenizer = AutoTokenizer.from_pretrained(
        "facebook/bart-large-xsum"
    )
    bundle.simplifier_model = AutoModelForSeq2SeqLM.from_pretrained(
        "facebook/bart-large-xsum"
    ).to(device)
    bundle.simplifier_model.eval()

    # --- 2. Zero-shot classifier: cross-encoder/nli-deberta-v3-large -------
    # DeBERTa-v3-large NLI model; MNLI accuracy ~91 % vs ~89.9 % for
    # bart-large-mnli — meaningfully better importance detection.
    logger.info(
        "Loading zero-shot classifier  : cross-encoder/nli-deberta-v3-large"
    )
    bundle.classifier_pipeline = pipeline(
        "zero-shot-classification",
        model="cross-encoder/nli-deberta-v3-large",
        device=torch_device,
    )

    # --- 3. Summarisation: google/pegasus-large ----------------------------
    # Pre-trained with the Gap-Sentence Generation objective, designed
    # specifically for abstractive summarisation; outperforms BART-CNN on
    # most summarisation benchmarks.
    # Loaded as tokenizer + model to avoid deprecated pipeline task strings.
    logger.info("Loading summarisation model   : google/pegasus-large")
    bundle.summarizer_tokenizer = AutoTokenizer.from_pretrained(
        "google/pegasus-large"
    )
    bundle.summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(
        "google/pegasus-large"
    ).to(device)
    bundle.summarizer_model.eval()

    # --- 4. Sentence-Transformer: all-mpnet-base-v2 ------------------------
    # Higher STS benchmark accuracy than all-MiniLM-L6-v2
    # (avg cosine-sim ~69.6 vs ~68.1 on the SBERT benchmark suite).
    logger.info(
        "Loading sentence-transformer  : sentence-transformers/all-mpnet-base-v2"
    )
    bundle.embedding_model = SentenceTransformer("all-mpnet-base-v2")

    _BUNDLE = bundle
    logger.info("All models loaded successfully.")
    return _BUNDLE