"""HuggingFace Daily Papers fetching with 24-hour filtering.

Fetches recent papers from the HuggingFace Daily Papers API and returns
them as Paper objects compatible with the rest of the pipeline.
Papers from this source do not include PDFs — the pipeline uses their
abstracts directly for chunking and summarization.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import requests

from digest_pipeline.config import Settings
from digest_pipeline.fetcher import Paper

logger = logging.getLogger(__name__)

HF_DAILY_PAPERS_URL = "https://huggingface.co/api/daily_papers"


def _within_last_24h(dt: datetime) -> bool:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt >= cutoff


def fetch_hf_papers(settings: Settings) -> list[Paper]:
    """Fetch recent papers from HuggingFace Daily Papers.

    Returns Paper objects with ``source="huggingface"``.  Papers outside
    the 24-hour window are filtered out.  No PDFs are downloaded — the
    abstract is used directly by the pipeline.
    """
    logger.info(
        "Querying HuggingFace Daily Papers (max %d).",
        settings.huggingface_max_results,
    )

    headers: dict[str, str] = {}
    if settings.huggingface_token:
        headers["Authorization"] = f"Bearer {settings.huggingface_token}"

    try:
        resp = requests.get(HF_DAILY_PAPERS_URL, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        logger.exception("Failed to fetch HuggingFace Daily Papers.")
        return []

    papers: list[Paper] = []
    for entry in data[: settings.huggingface_max_results]:
        paper_data = entry.get("paper", {})
        paper_id = paper_data.get("id", "")
        title = paper_data.get("title", "")
        abstract = paper_data.get("summary", "")

        authors_raw = paper_data.get("authors", [])
        authors = [a.get("name", a.get("user", "")) for a in authors_raw if isinstance(a, dict)]

        published_str = entry.get("publishedAt", "")
        try:
            published = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            published = datetime.now(timezone.utc)

        if not _within_last_24h(published):
            continue

        if not paper_id or not title:
            continue

        papers.append(
            Paper(
                paper_id=f"hf_{paper_id}",
                title=title,
                authors=authors,
                abstract=abstract,
                url=f"https://huggingface.co/papers/{paper_id}",
                published=published,
                source="huggingface",
                pdf_path=None,
            )
        )

    logger.info("Fetched %d papers from HuggingFace within the 24-hour window.", len(papers))
    return papers
