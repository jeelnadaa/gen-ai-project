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

_NLP_MODEL = None


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract raw text from every page of a PDF file."""
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
    # Specific fix for "Word\n \nWord" pattern reported in stress tests
    text = re.sub(r'(\w)\s*\n\s*\n\s*(\w)', r'\1 \2', text)
    # Join words split by single newlines (common in PDF columnar layouts)
    text = re.sub(r'(\w)\s*\n\s*(\w)', r'\1 \2', text)
    
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.replace("\x0c", "\n")
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _get_nlp():
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
    """Split text into clauses using spaCy sentence segmentation."""
    nlp = _get_nlp()
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
                parts = [p.strip() for p in sentence.split(";") if p.strip()]
                for part in parts:
                    if len(part) >= min_length:
                        clauses.append(part[:max_length])
            else:
                clauses.append(sentence)

    logger.info("Extracted %d clauses from the document.", len(clauses))
    return clauses