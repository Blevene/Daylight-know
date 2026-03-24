"""PDF archive with DuckDB index and per-day markdown summaries.

Copies fetched PDFs to a date-organized directory and maintains a
searchable DuckDB index across all archived papers.
"""

from __future__ import annotations

import re

from digest_pipeline.fetcher import Paper


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


def _format_authors(authors: list[str]) -> str:
    """Format author list, truncating to 'First et al.' if more than 2."""
    if len(authors) > 2:
        return f"{authors[0]} et al."
    return ", ".join(authors)


def _generate_markdown_index(
    papers: list[Paper],
    date_str: str,
    filenames: dict[str, str],
) -> str:
    """Generate a markdown index for a day's archived papers.

    *filenames* maps paper_id -> archived filename (only for papers with PDFs).
    """
    lines = [
        f"# Papers \u2014 {date_str}",
        "",
        "| Title | Authors | Categories | Link | File |",
        "|-------|---------|------------|------|------|",
    ]

    for paper in papers:
        authors = _format_authors(paper.authors)
        categories = ", ".join(paper.categories)
        link = f"[arXiv]({paper.url})" if paper.url else "\u2014"
        filename = filenames.get(paper.paper_id)
        file_col = f"[PDF]({filename})" if filename else "\u2014"
        lines.append(f"| {paper.title} | {authors} | {categories} | {link} | {file_col} |")

    lines.append("")
    lines.append(f"{len(papers)} papers archived.")
    lines.append("")
    return "\n".join(lines)
