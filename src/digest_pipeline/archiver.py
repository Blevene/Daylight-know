"""PDF archive with DuckDB index and per-day markdown summaries.

Copies fetched PDFs to a date-organized directory and maintains a
searchable DuckDB index across all archived papers.
"""

from __future__ import annotations

import re


def _sanitize_filename(title: str, paper_id: str) -> str:
    """Build a filesystem-safe PDF filename from title and paper ID.

    Format: ``{sanitized_title}_{sanitized_id}.pdf``
    """
    # Sanitize title: strip special chars, collapse whitespace, replace spaces with hyphens
    clean_title = re.sub(r"[^\w\s-]", "", title).strip()
    clean_title = re.sub(r"\s+", "-", clean_title)
    clean_title = clean_title[:80]

    # Sanitize paper ID: replace slashes (old arXiv format like hep-ph/0301001)
    clean_id = paper_id.replace("/", "-")

    return f"{clean_title}_{clean_id}.pdf"
