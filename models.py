"""
models.py - Model loading for the Legal Clause Simplifier.

Setup:
  1. Sign up free at https://console.groq.com
  2. Create an API key under API Keys
  3. Create a .env file in the project root:
         GROQ_API_KEY=gsk_your_key_here
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

# Load .env into os.environ automatically — silent no-op if file is absent
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class ModelBundle:
    """Holds all clients/models used by the pipeline."""

    groq_client: Any = field(default=None, repr=False)
    groq_model: str = "llama-3.3-70b-versatile"
    embedding_model: Any = field(default=None, repr=False)


_BUNDLE: ModelBundle | None = None


def load_models(groq_model: str = "llama-3.3-70b-versatile") -> ModelBundle:
    """
    Initialise and return the cached ModelBundle.
    API key is read exclusively from GROQ_API_KEY in .env file.
    """
    global _BUNDLE
    if _BUNDLE is not None:
        logger.info("Using cached model bundle.")
        return _BUNDLE

    from groq import Groq
    from sentence_transformers import SentenceTransformer

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found.\n"
            "  Add it to a .env file in the project root:\n"
            "      GROQ_API_KEY=gsk_your_key_here\n"
            "  Get a free key (no credit card) at https://console.groq.com"
        )

    bundle = ModelBundle()

    logger.info("Initialising Groq client  (model: %s)", groq_model)
    bundle.groq_client = Groq(api_key=api_key)
    bundle.groq_model  = groq_model

    logger.info("Loading sentence-transformer : BAAI/bge-large-en-v1.5")
    bundle.embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")

    _BUNDLE = bundle
    logger.info("All models ready.")
    return _BUNDLE