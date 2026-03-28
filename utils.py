"""
utils.py - PDF text extraction and clause splitting using spaCy.
"""

import re
import logging
from pathlib import Path

import PyPDF2
import spacy

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PDF Extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract raw text from every page of a PDF file.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        A single string containing all extracted text.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        ValueError: If no text could be extracted.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info("Extracting text from PDF: %s", path.name)
    pages: list[str] = []

    with open(path, "rb") as fh:
        reader = PyPDF2.PdfReader(fh)
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages.append(text)
            logger.debug("  Page %d: %d characters", page_num, len(text))

    full_text = "\n".join(pages)
    full_text = _clean_text(full_text)

    if not full_text.strip():
        raise ValueError("No text could be extracted from the PDF.")

    logger.info("Total extracted text length: %d characters", len(full_text))
    return full_text


def _clean_text(text: str) -> str:
    """Normalise whitespace and remove common PDF artefacts."""
    # Collapse excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove form-feed characters
    text = text.replace("\x0c", "\n")
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Clause Splitting
# ---------------------------------------------------------------------------

_NLP_MODEL: spacy.language.Language | None = None


def _get_nlp() -> spacy.language.Language:
    """Lazy-load the spaCy model (singleton)."""
    global _NLP_MODEL
    if _NLP_MODEL is None:
        logger.info("Loading spaCy model: en_core_web_sm")
        _NLP_MODEL = spacy.load("en_core_web_sm")
    return _NLP_MODEL


def split_into_clauses(
    text: str,
    min_length: int = 30,
    max_length: int = 1_000,
) -> list[str]:
    """
    Split *text* into individual clauses using spaCy sentence segmentation.

    Very short or very long sentences are filtered / truncated to keep
    downstream model inputs manageable.

    Args:
        text:        Full document text.
        min_length:  Minimum character length for a clause to be kept.
        max_length:  Maximum character length; longer sentences are split on
                     semicolons or truncated.

    Returns:
        A list of clause strings.
    """
    nlp = _get_nlp()

    # spaCy has a default max-doc-length; chunk if necessary
    clauses: list[str] = []
    chunk_size = 100_000
    for start in range(0, len(text), chunk_size):
        chunk = text[start : start + chunk_size]
        doc = nlp(chunk)
        for sent in doc.sents:
            sentence = sent.text.strip()
            if len(sentence) < min_length:
                continue
            if len(sentence) > max_length:
                # Try splitting on semicolons first
                parts = [p.strip() for p in sentence.split(";") if p.strip()]
                for part in parts:
                    if len(part) >= min_length:
                        clauses.append(part[:max_length])
            else:
                clauses.append(sentence)

    logger.info("Extracted %d clauses from the document.", len(clauses))
    return clauses