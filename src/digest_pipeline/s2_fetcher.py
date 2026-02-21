"""Semantic Scholar paper fetching with date filtering.

Queries the Semantic Scholar Academic Graph API for recent papers matching
the configured search query and returns them as Paper objects.
Papers from this source do not include PDFs — the pipeline uses their
abstracts directly for chunking and summarization.

Semantic Scholar provides a flat list of "Fields of Study" that can be
used to narrow results.  Unlike arXiv's hierarchical taxonomy, these are
broad top-level labels:

    Computer Science, Mathematics, Physics, Chemistry, Biology,
    Medicine, Engineering, Environmental Science, Economics,
    Business, Political Science, Psychology, Sociology, Geography,
    History, Art, Philosophy, Linguistics, Materials Science,
    Geology, Agricultural and Food Sciences, Education, Law

Set ``SEMANTICSCHOLAR_FIELDS_OF_STUDY`` in your ``.env`` to filter
(comma-separated, e.g. ``Computer Science,Mathematics``).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import requests

from digest_pipeline.config import Settings
from digest_pipeline.fetcher import Paper

logger = logging.getLogger(__name__)

S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

# All valid Semantic Scholar fields of study (for reference / validation).
S2_FIELDS_OF_STUDY = [
    "Computer Science",
    "Mathematics",
    "Physics",
    "Chemistry",
    "Biology",
    "Medicine",
    "Engineering",
    "Environmental Science",
    "Economics",
    "Business",
    "Political Science",
    "Psychology",
    "Sociology",
    "Geography",
    "History",
    "Art",
    "Philosophy",
    "Linguistics",
    "Materials Science",
    "Geology",
    "Agricultural and Food Sciences",
    "Education",
    "Law",
]


def fetch_s2_papers(settings: Settings) -> list[Paper]:
    """Fetch recent papers from Semantic Scholar.

    Uses the ``semanticscholar_query`` setting as the search term and
    filters results to the last 24 hours based on ``publicationDate``.
    Returns Paper objects with ``source="semanticscholar"``.
    """
    query = settings.semanticscholar_query
    logger.info(
        "Querying Semantic Scholar: %r (max %d).",
        query,
        settings.semanticscholar_max_results,
    )

    headers: dict[str, str] = {}
    if settings.semanticscholar_api_key:
        headers["x-api-key"] = settings.semanticscholar_api_key

    # Use yesterday's date as the publication date lower-bound.
    date_from = (datetime.now(timezone.utc) - timedelta(hours=24)).strftime("%Y-%m-%d")
    date_to = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    params: dict[str, str | int] = {
        "query": query,
        "limit": settings.semanticscholar_max_results,
        "fields": "title,authors,abstract,url,publicationDate,externalIds,openAccessPdf",
        "publicationDateOrYear": f"{date_from}:{date_to}",
    }

    if settings.semanticscholar_fields_of_study:
        params["fieldsOfStudy"] = ",".join(settings.semanticscholar_fields_of_study)

    try:
        resp = requests.get(S2_SEARCH_URL, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        logger.exception("Failed to fetch papers from Semantic Scholar.")
        return []

    results = data.get("data", [])

    papers: list[Paper] = []
    for item in results:
        paper_id = item.get("paperId", "")
        title = item.get("title", "")
        abstract = item.get("abstract", "") or ""

        if not paper_id or not title:
            continue

        authors_raw = item.get("authors", [])
        authors = [a.get("name", "") for a in authors_raw if a.get("name")]

        pub_date_str = item.get("publicationDate")
        if pub_date_str:
            try:
                published = datetime.strptime(pub_date_str, "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                published = datetime.now(timezone.utc)
        else:
            published = datetime.now(timezone.utc)

        # Prefer the Semantic Scholar URL; fall back to an arXiv link if available.
        external_ids = item.get("externalIds", {}) or {}
        s2_url = item.get("url", "")
        if not s2_url and external_ids.get("ArXiv"):
            s2_url = f"https://arxiv.org/abs/{external_ids['ArXiv']}"
        if not s2_url:
            s2_url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"

        # Check for open-access PDF availability.
        pdf_url = None
        oa_pdf = item.get("openAccessPdf")
        if isinstance(oa_pdf, dict):
            pdf_url = oa_pdf.get("url")

        papers.append(
            Paper(
                paper_id=f"s2_{paper_id}",
                title=title,
                authors=authors,
                abstract=abstract,
                url=s2_url,
                published=published,
                source="semanticscholar",
                pdf_path=None,  # PDF download could be added later if pdf_url is available
            )
        )

    logger.info("Fetched %d papers from Semantic Scholar.", len(papers))
    return papers
