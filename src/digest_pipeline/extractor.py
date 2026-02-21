"""PDF text extraction via PyMuPDF.

EARS coverage
─────────────
- Ubiquitous 2.1-1: use PyMuPDF for raw text extraction.
- Unwanted 2.4-2: flag empty-text documents as "unparseable".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    arxiv_id: str
    text: str
    parseable: bool


def extract_text(pdf_path: Path, arxiv_id: str) -> ExtractionResult:
    """Extract all text from *pdf_path* using PyMuPDF.

    If the resulting text is empty (e.g. scanned images without OCR),
    the document is flagged as unparseable (EARS 2.4-2).
    """
    try:
        doc = fitz.open(pdf_path)
        pages_text = [page.get_text() for page in doc]
        doc.close()
        full_text = "\n".join(pages_text).strip()
    except Exception:
        logger.exception("PyMuPDF failed on %s", pdf_path)
        full_text = ""

    if not full_text:
        logger.warning("Empty text for %s — marking as unparseable.", arxiv_id)
        return ExtractionResult(arxiv_id=arxiv_id, text="", parseable=False)

    return ExtractionResult(arxiv_id=arxiv_id, text=full_text, parseable=True)
