"""arXiv paper fetching with 24-hour filtering and retry-based PDF download.

EARS coverage
─────────────
- Event 2.2-1: query arXiv API for configured topics on cron trigger.
- Event 2.2-2: filter results to preceding 24-hour window.
- Unwanted 2.4-1: skip paper after 3 failed PDF download attempts.
"""

from __future__ import annotations

import logging
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import arxiv
import requests

from digest_pipeline.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class Paper:
    """Lightweight representation of a fetched research paper.

    Works with any source (arXiv, HuggingFace, OpenAlex, etc.).
    """

    paper_id: str
    title: str
    authors: list[str]
    abstract: str
    url: str
    published: datetime
    source: str = "arxiv"
    pdf_path: Path | None = None
    categories: list[str] = field(default_factory=list)
    upvotes: int = 0
    fields_of_study: list[str] = field(default_factory=list)


def _within_last_24h(dt: datetime) -> bool:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt >= cutoff


def download_pdf(url: str, dest: Path, max_retries: int = 3) -> bool:
    """Download *url* to *dest*, retrying up to *max_retries* times.

    Returns ``True`` on success, ``False`` after all retries exhausted.
    """
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            return True
        except Exception as exc:
            logger.warning("PDF download attempt %d/%d failed for %s: %s", attempt, max_retries, url, exc)
            if attempt < max_retries:
                time.sleep(2 ** attempt)
    return False


def fetch_papers(settings: Settings) -> list[Paper]:
    """Query arXiv for recent papers matching the configured topics.

    Papers outside the 24-hour window are filtered out.  PDFs that fail
    to download after ``settings.pdf_download_max_retries`` attempts are
    logged and skipped (EARS 2.4-1).
    """
    query = " OR ".join(f"cat:{topic}" for topic in settings.arxiv_topics)
    logger.info("Querying arXiv: %s (max %d)", query, settings.arxiv_max_results)

    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=settings.arxiv_max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    papers: list[Paper] = []
    tmp_dir = Path(tempfile.mkdtemp(prefix="arxiv_pdfs_"))

    for result in client.results(search):
        if not _within_last_24h(result.published):
            continue

        paper_id = result.entry_id.split("/")[-1]
        pdf_url = result.pdf_url
        pdf_dest = tmp_dir / f"{paper_id}.pdf"

        if not download_pdf(pdf_url, pdf_dest, max_retries=settings.pdf_download_max_retries):
            logger.error("Skipping paper %s — PDF download failed after %d attempts.", paper_id, settings.pdf_download_max_retries)
            continue

        papers.append(
            Paper(
                paper_id=paper_id,
                title=result.title,
                authors=[a.name for a in result.authors],
                abstract=result.summary,
                url=result.entry_id,
                published=result.published,
                source="arxiv",
                pdf_path=pdf_dest,
                categories=result.categories,
            )
        )

    logger.info("Fetched %d papers within the 24-hour window.", len(papers))
    return papers
