"""arXiv paper fetching via RSS feed with retry-based PDF download.

EARS coverage
─────────────
- Event 2.2-1: fetch arXiv RSS feed for configured topics on cron trigger.
- Event 2.2-2: RSS feed returns current day's papers (no date filtering needed).
- Unwanted 2.4-1: skip paper after 3 failed PDF download attempts.
"""

from __future__ import annotations

import logging
import re
import tempfile
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import requests

from digest_pipeline.config import Settings

logger = logging.getLogger(__name__)

_ARXIV_RSS_URL = "https://rss.arxiv.org/rss/{categories}"
_NS = {
    "dc": "http://purl.org/dc/elements/1.1/",
    "arxiv": "http://arxiv.org/schemas/atom",
}


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
            logger.warning(
                "PDF download attempt %d/%d failed for %s: %s", attempt, max_retries, url, exc
            )
            if attempt < max_retries:
                time.sleep(2**attempt)
    return False


# Regex to extract abstract from description like:
#   "arXiv:2603.12345v1 Announce Type: new \n Abstract: The actual abstract text..."
_ABSTRACT_RE = re.compile(r"Abstract:\s*", re.IGNORECASE)


def _extract_abstract(description: str) -> str:
    """Extract the abstract text from an RSS item description."""
    match = _ABSTRACT_RE.search(description)
    if match:
        return description[match.end():].strip()
    return ""


def _parse_arxiv_id(link: str) -> str:
    """Extract the arXiv paper ID from an abs URL like https://arxiv.org/abs/2603.12345."""
    return link.rstrip("/").split("/abs/")[-1]


def _fetch_rss(settings: Settings, *, max_results: int) -> list[Paper]:
    """Fetch papers from the arXiv RSS feed.

    Returns a list of Paper objects (without PDFs downloaded yet).
    """
    categories = "+".join(settings.arxiv_topics)
    url = _ARXIV_RSS_URL.format(categories=categories)
    logger.info("Fetching arXiv RSS feed: %s", url)

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    root = ET.fromstring(resp.content)

    # RSS 2.0: items are under <channel>
    channel = root.find("channel")
    if channel is None:
        logger.warning("RSS feed has no <channel> element.")
        return []

    items = channel.findall("item")
    logger.info("RSS feed returned %d items.", len(items))

    papers: list[Paper] = []
    now = datetime.now(timezone.utc)

    for item in items:
        if len(papers) >= max_results:
            break

        title_el = item.find("title")
        link_el = item.find("link")
        desc_el = item.find("description")
        creator_el = item.find("dc:creator", _NS)

        if title_el is None or link_el is None:
            continue

        title = (title_el.text or "").strip()
        link = (link_el.text or "").strip()
        description = (desc_el.text or "") if desc_el is not None else ""
        abstract = _extract_abstract(description)

        # Skip items without a real abstract
        if not abstract:
            continue

        paper_id = _parse_arxiv_id(link)

        # Parse authors from dc:creator (comma-separated or newline-separated)
        authors: list[str] = []
        if creator_el is not None and creator_el.text:
            raw = creator_el.text
            for sep in ["\n", ","]:
                if sep in raw:
                    authors = [a.strip() for a in raw.split(sep) if a.strip()]
                    break
            if not authors:
                authors = [raw.strip()]

        # Extract categories
        categories_list: list[str] = [
            cat_el.text.strip()
            for cat_el in item.findall("category")
            if cat_el.text
        ]

        papers.append(
            Paper(
                paper_id=paper_id,
                title=title,
                authors=authors,
                abstract=abstract,
                url=link,
                published=now,  # RSS items are today's papers
                source="arxiv",
                categories=categories_list,
            )
        )

    return papers


def fetch_papers(settings: Settings, *, max_results: int | None = None) -> list[Paper]:
    """Fetch recent arXiv papers via RSS and download their PDFs.

    *max_results* overrides ``settings.arxiv_max_results`` when provided
    (e.g. to fetch a larger pool for ranking).
    """
    effective_max = max_results or settings.arxiv_max_results

    rss_papers = _fetch_rss(settings, max_results=effective_max)

    if not rss_papers:
        logger.warning("No papers found in arXiv RSS feed.")
        return []

    # Download PDFs in parallel
    tmp_dir = Path(tempfile.mkdtemp(prefix="arxiv_pdfs_"))
    max_workers = settings.pdf_download_workers
    max_retries = settings.pdf_download_max_retries

    def _download_one(paper: Paper) -> Paper | None:
        pdf_url = paper.url.replace("/abs/", "/pdf/")
        pdf_dest = tmp_dir / f"{paper.paper_id}.pdf"
        if download_pdf(pdf_url, pdf_dest, max_retries=max_retries):
            paper.pdf_path = pdf_dest
            return paper
        logger.error(
            "Skipping paper %s — PDF download failed after %d attempts.",
            paper.paper_id,
            max_retries,
        )
        return None

    papers: list[Paper] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_download_one, p): p for p in rss_papers}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                papers.append(result)

    logger.info("Fetched %d papers from arXiv RSS feed (%d workers).", len(papers), max_workers)
    return papers
