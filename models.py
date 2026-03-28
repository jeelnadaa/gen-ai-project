"""
models.py - Centralised model loading for the Legal Clause Simplifier.

All heavy models are loaded lazily and cached as module-level singletons so
they are initialised at most once per process.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Container for all loaded models
# ---------------------------------------------------------------------------

@dataclass
class ModelBundle:
    """Holds every model/tokenizer used by the pipeline."""

    # Simplification & explanation  (facebook/bart-large)
    simplifier_tokenizer: Any = field(default=None, repr=False)
    simplifier_model: Any = field(default=None, repr=False)

    # Zero-shot importance classification  (facebook/bart-large-mnli)
    classifier_pipeline: Any = field(default=None, repr=False)

    # Summarisation — loaded as tokenizer + model directly to avoid the
    # removed "summarization" pipeline task in newer transformers versions
    summarizer_tokenizer: Any = field(default=None, repr=False)
    summarizer_model: Any = field(default=None, repr=False)

    # Semantic similarity embeddings  (sentence-transformers/all-MiniLM-L6-v2)
    embedding_model: Any = field(default=None, repr=False)


_BUNDLE: ModelBundle | None = None


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

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

    # --- 1. Simplification / Explanation model (BART-large) ----------------
    logger.info("Loading simplification model: facebook/bart-large")
    bundle.simplifier_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    bundle.simplifier_model = AutoModelForSeq2SeqLM.from_pretrained(
        "facebook/bart-large"
    ).to(device)
    bundle.simplifier_model.eval()

    # --- 2. Zero-shot importance classifier (BART-large-MNLI) --------------
    logger.info("Loading zero-shot classifier: facebook/bart-large-mnli")
    bundle.classifier_pipeline = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=0 if device.startswith("cuda") else -1,
    )

    # --- 3. Summarisation model (BART-large-CNN) ----------------------------
    # NOTE: The "summarization" pipeline task was removed in recent versions of
    # transformers.  We load the tokenizer + model directly instead and run
    # inference manually in processing.py — identical results, no task lookup.
    logger.info("Loading summarisation model: facebook/bart-large-cnn")
    bundle.summarizer_tokenizer = AutoTokenizer.from_pretrained(
        "facebook/bart-large-cnn"
    )
    bundle.summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(
        "facebook/bart-large-cnn"
    ).to(device)
    bundle.summarizer_model.eval()

    # --- 4. Sentence-Transformer for semantic similarity ------------------
    logger.info(
        "Loading sentence-transformer: sentence-transformers/all-MiniLM-L6-v2"
    )
    bundle.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    _BUNDLE = bundle
    logger.info("All models loaded successfully.")
    return _BUNDLE