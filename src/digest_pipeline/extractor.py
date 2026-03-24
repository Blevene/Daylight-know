"""PDF text extraction via pypdf.

EARS coverage
─────────────
- Ubiquitous 2.1-1: use pypdf for raw text extraction.
- Unwanted 2.4-2: flag empty-text documents as "unparseable".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    paper_id: str
    text: str
    parseable: bool


def extract_text(pdf_path: Path, paper_id: str) -> ExtractionResult:
    """Extract all text from *pdf_path* using pypdf.

    If the resulting text is empty (e.g. scanned images without OCR),
    the document is flagged as unparseable (EARS 2.4-2).
    """
    try:
        reader = PdfReader(pdf_path)
        pages_text = [page.extract_text() or "" for page in reader.pages]
        # pypdf can produce surrogate characters from math/symbol fonts;
        # strip them so downstream UTF-8 encoding never fails.
        full_text = "\n".join(pages_text).strip()
        full_text = full_text.encode("utf-8", errors="replace").decode("utf-8")
    except Exception:
        logger.exception("pypdf failed on %s", pdf_path)
        full_text = ""

    if not full_text:
        logger.warning("Empty text for %s — marking as unparseable.", paper_id)
        return ExtractionResult(paper_id=paper_id, text="", parseable=False)

    return ExtractionResult(paper_id=paper_id, text=full_text, parseable=True)
