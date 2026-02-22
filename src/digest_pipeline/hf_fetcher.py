"""HuggingFace Daily Papers fetching with deduplication support.

Fetches recent papers from the HuggingFace Daily Papers API and returns
structured results that the pipeline can reconcile against arXiv papers
already fetched.  Since HuggingFace papers *are* arXiv papers, the pipeline
uses the raw arXiv ID to deduplicate:

- Papers already fetched from arXiv are noted as "trending" rather than
  re-processed.
- Papers *not* in the arXiv set are converted to ``Paper`` objects and
  added to the main pipeline.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import requests

from digest_pipeline.config import Settings
from digest_pipeline.fetcher import Paper

logger = logging.getLogger(__name__)

HF_DAILY_PAPERS_URL = "https://huggingface.co/api/daily_papers"


@dataclass
class HFDailyPaper:
    """A paper from the HuggingFace Daily Papers feed.

    Carries the raw arXiv ID so the pipeline can match it against
    papers already fetched from arXiv.
    """

    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    url: str
    published: datetime
    upvotes: int = 0


def _within_last_24h(dt: datetime) -> bool:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt >= cutoff


def normalize_arxiv_id(raw_id: str) -> str:
    """Strip version suffix from an arXiv ID for comparison.

    ``"2401.00001v2"`` → ``"2401.00001"``
    ``"2401.00001"``   → ``"2401.00001"``
    """
    return re.sub(r"v\d+$", "", raw_id)


def _request_with_retry(url: str, **kwargs) -> requests.Response:
    """GET *url* with up to 2 retries on failure (exponential backoff).

    For HTTP 429 (rate limit), respects the ``Retry-After`` header if
    present, otherwise waits 30s.
    """
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(url, **kwargs)
            resp.raise_for_status()
            return resp
        except requests.exceptions.HTTPError as exc:
            if attempt == max_attempts:
                raise
            status = exc.response.status_code if exc.response is not None else 0
            if status == 429:
                retry_after = exc.response.headers.get("Retry-After")
                wait = int(retry_after) if retry_after and retry_after.isdigit() else 30
                logger.warning(
                    "HF rate-limited (429), attempt %d/%d — waiting %ds…",
                    attempt,
                    max_attempts,
                    wait,
                )
            else:
                wait = 2**attempt
                logger.warning(
                    "HF request attempt %d/%d failed (%s), retrying in %ds…",
                    attempt,
                    max_attempts,
                    status,
                    wait,
                )
            time.sleep(wait)
        except Exception:
            if attempt == max_attempts:
                raise
            wait = 2**attempt
            logger.warning(
                "HF request attempt %d/%d failed, retrying in %ds…", attempt, max_attempts, wait
            )
            time.sleep(wait)
    raise RuntimeError("unreachable")  # pragma: no cover


def fetch_hf_daily(settings: Settings) -> list[HFDailyPaper]:
    """Fetch recent papers from HuggingFace Daily Papers.

    Returns ``HFDailyPaper`` objects (not ``Paper``) so the pipeline can
    reconcile them against arXiv papers before deciding what to process.
    Papers outside the 24-hour window are filtered out.
    """
    logger.info(
        "Querying HuggingFace Daily Papers (max %d).",
        settings.huggingface_max_results,
    )

    headers: dict[str, str] = {}
    if settings.huggingface_token:
        headers["Authorization"] = f"Bearer {settings.huggingface_token}"

    try:
        resp = _request_with_retry(HF_DAILY_PAPERS_URL, headers=headers, timeout=30)
        data = resp.json()
    except Exception:
        logger.exception("Failed to fetch HuggingFace Daily Papers.")
        return []

    results: list[HFDailyPaper] = []
    for entry in data[: settings.huggingface_max_results]:
        paper_data = entry.get("paper", {})
        arxiv_id = paper_data.get("id", "")
        title = paper_data.get("title", "")
        abstract = paper_data.get("summary", "")
        upvotes = paper_data.get("upvotes", 0)

        authors_raw = paper_data.get("authors", [])
        authors = [a.get("name", a.get("user", "")) for a in authors_raw if isinstance(a, dict)]

        published_str = entry.get("publishedAt", "")
        try:
            published = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            published = datetime.now(timezone.utc)

        if not _within_last_24h(published):
            continue

        if not arxiv_id or not title:
            continue

        results.append(
            HFDailyPaper(
                arxiv_id=arxiv_id,
                title=title,
                authors=authors,
                abstract=abstract,
                url=f"https://huggingface.co/papers/{arxiv_id}",
                published=published,
                upvotes=upvotes,
            )
        )

    logger.info("Fetched %d papers from HuggingFace within the 24-hour window.", len(results))
    return results


def reconcile_hf_papers(
    hf_papers: list[HFDailyPaper],
    known_ids: set[str],
) -> tuple[list[Paper], list[HFDailyPaper]]:
    """Split HuggingFace papers into new papers and trending-existing.

    Parameters
    ----------
    hf_papers:
        Raw results from ``fetch_hf_daily``.
    known_ids:
        Set of *normalized* arXiv IDs already fetched (version-stripped).

    Returns
    -------
    (new_papers, trending_existing)
        ``new_papers`` — ``Paper`` objects for papers not in the arXiv set.
        ``trending_existing`` — ``HFDailyPaper`` entries whose arXiv ID
        matches an already-fetched paper (for the trending sidebar).
    """
    new_papers: list[Paper] = []
    trending_existing: list[HFDailyPaper] = []

    for hf in hf_papers:
        normalized = normalize_arxiv_id(hf.arxiv_id)
        if normalized in known_ids:
            trending_existing.append(hf)
        else:
            new_papers.append(
                Paper(
                    paper_id=f"hf_{hf.arxiv_id}",
                    title=hf.title,
                    authors=hf.authors,
                    abstract=hf.abstract,
                    url=hf.url,
                    published=hf.published,
                    source="huggingface",
                    pdf_path=None,
                    upvotes=hf.upvotes,
                )
            )

    logger.info(
        "HuggingFace reconciliation: %d new papers, %d trending (already in arXiv set).",
        len(new_papers),
        len(trending_existing),
    )
    return new_papers, trending_existing
