"""PDF archive with DuckDB index and per-day markdown summaries.

Copies fetched PDFs to a date-organized directory and maintains a
searchable DuckDB index across all archived papers.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import duckdb

from digest_pipeline.fetcher import Paper

logger = logging.getLogger(__name__)


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


def _init_db(db_path: Path) -> duckdb.DuckDBPyConnection:
    """Open (or create) the archive DuckDB and ensure the papers table exists."""
    con = duckdb.connect(str(db_path))
    con.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            paper_id       VARCHAR PRIMARY KEY,
            title          VARCHAR,
            authors        VARCHAR,
            abstract       VARCHAR,
            categories     VARCHAR,
            source         VARCHAR,
            url            VARCHAR,
            pdf_path       VARCHAR,
            published_date TIMESTAMP,
            archived_date  VARCHAR
        )
    """)
    return con


def _upsert_papers(
    con: duckdb.DuckDBPyConnection,
    papers: list[Paper],
    date_str: str,
    filenames: dict[str, str],
) -> None:
    """Insert or update paper records, preserving existing pdf_path if no new PDF."""
    for paper in papers:
        new_pdf_path = filenames.get(paper.paper_id)

        if new_pdf_path:
            # New PDF available - full upsert
            con.execute(
                """INSERT OR REPLACE INTO papers
                   (paper_id, title, authors, abstract, categories, source,
                    url, pdf_path, published_date, archived_date)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    paper.paper_id,
                    paper.title,
                    ", ".join(paper.authors),
                    paper.abstract,
                    ", ".join(paper.categories),
                    paper.source,
                    paper.url,
                    new_pdf_path,
                    paper.published,
                    date_str,
                ],
            )
        else:
            # No new PDF - upsert metadata but preserve existing pdf_path
            existing = con.execute(
                "SELECT pdf_path FROM papers WHERE paper_id = ?",
                [paper.paper_id],
            ).fetchone()
            existing_path = existing[0] if existing else None

            con.execute(
                """INSERT OR REPLACE INTO papers
                   (paper_id, title, authors, abstract, categories, source,
                    url, pdf_path, published_date, archived_date)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    paper.paper_id,
                    paper.title,
                    ", ".join(paper.authors),
                    paper.abstract,
                    ", ".join(paper.categories),
                    paper.source,
                    paper.url,
                    existing_path,
                    paper.published,
                    date_str,
                ],
            )


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
