"""OpenAlex paper fetching with date filtering and topic categorization.

Queries the OpenAlex Works API for recent papers matching the configured
search query and topic filters.  Returns them as Paper objects with
reconstructed abstracts and topic metadata.

OpenAlex uses a 4-level topic hierarchy:

    Domain (4) → Field (26) → Subfield (~200) → Topic (~4,500)

Configure ``OPENALEX_FIELDS`` with field display names to filter results.
The field names map to numeric IDs used in API queries.

See https://docs.openalex.org/api-entities/works for full documentation.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

import requests

from digest_pipeline.config import Settings
from digest_pipeline.fetcher import Paper

logger = logging.getLogger(__name__)

OPENALEX_WORKS_URL = "https://api.openalex.org/works"

# Mapping from display names to OpenAlex field numeric IDs.
# Full list: https://docs.openalex.org — 26 fields across 4 domains.
OPENALEX_FIELDS: dict[str, int] = {
    # Physical Sciences (domain 3)
    "Chemical Engineering": 15,
    "Chemistry": 16,
    "Computer Science": 17,
    "Earth and Planetary Sciences": 19,
    "Energy": 21,
    "Engineering": 22,
    "Environmental Science": 23,
    "Materials Science": 25,
    "Mathematics": 26,
    "Physics and Astronomy": 31,
    # Life Sciences (domain 1)
    "Agricultural and Biological Sciences": 11,
    "Biochemistry, Genetics and Molecular Biology": 13,
    "Immunology and Microbiology": 24,
    "Neuroscience": 28,
    "Pharmacology, Toxicology and Pharmaceutics": 30,
    "Veterinary": 34,
    # Health Sciences (domain 4)
    "Medicine": 27,
    "Nursing": 29,
    "Dentistry": 35,
    "Health Professions": 36,
    # Social Sciences (domain 2)
    "Arts and Humanities": 12,
    "Business, Management and Accounting": 14,
    "Decision Sciences": 18,
    "Economics, Econometrics and Finance": 20,
    "Psychology": 32,
    "Social Sciences": 33,
}


def reconstruct_abstract(inverted_index: dict | None) -> str:
    """Reconstruct plaintext from an OpenAlex abstract inverted index.

    The inverted index maps each word to a list of integer positions.
    We invert this to produce the original word order.
    """
    if not inverted_index:
        return ""
    positions: dict[int, str] = {}
    for word, idxs in inverted_index.items():
        for idx in idxs:
            positions[idx] = word
    return " ".join(positions[k] for k in sorted(positions.keys()))


def fetch_openalex_papers(
    settings: Settings,
    *,
    known_paper_ids: set[str] | None = None,
) -> list[Paper]:
    """Fetch recent papers from OpenAlex.

    Uses ``openalex_query`` as the search term, filters to the last 24 hours
    by ``publication_date``, and optionally filters by academic field.

    If *known_paper_ids* is provided, papers whose DOI (stripped of the
    ``https://doi.org/`` prefix) matches an ID in the set are skipped.
    """
    query = settings.openalex_query
    logger.info(
        "Querying OpenAlex: %r (max %d).",
        query,
        settings.openalex_max_results,
    )

    date_from = (datetime.now(timezone.utc) - timedelta(hours=24)).strftime("%Y-%m-%d")

    # Build filter string
    filters = [f"from_publication_date:{date_from}", "type:article|preprint"]

    if settings.openalex_fields:
        field_ids = []
        for name in settings.openalex_fields:
            fid = OPENALEX_FIELDS.get(name)
            if fid:
                field_ids.append(str(fid))
            else:
                logger.warning("Unknown OpenAlex field: %r — skipping filter.", name)
        if field_ids:
            filters.append("primary_topic.field.id:" + "|".join(field_ids))

    params: dict[str, str | int] = {
        "search": query,
        "filter": ",".join(filters),
        "per_page": settings.openalex_max_results,
        "sort": "publication_date:desc",
        "select": "id,title,authorships,abstract_inverted_index,publication_date,doi,open_access,topics",
    }

    if settings.openalex_api_key:
        params["api_key"] = settings.openalex_api_key

    headers: dict[str, str] = {}
    if settings.openalex_email:
        headers["User-Agent"] = f"DaylightKnow/1.0 (mailto:{settings.openalex_email})"

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(
                OPENALEX_WORKS_URL,
                params=params,
                headers=headers,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            break
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else 0
            if attempt == max_attempts:
                logger.exception("Failed to fetch papers from OpenAlex after %d attempts.", max_attempts)
                return []
            if status == 429:
                wait = int(exc.response.headers.get("Retry-After", 5))
                logger.warning("OpenAlex rate-limited (429), attempt %d/%d — waiting %ds.", attempt, max_attempts, wait)
            else:
                wait = 2 ** attempt
                logger.warning("OpenAlex request failed (%s), attempt %d/%d — retrying in %ds.", status, attempt, max_attempts, wait)
            time.sleep(wait)
        except Exception:
            if attempt == max_attempts:
                logger.exception("Failed to fetch papers from OpenAlex after %d attempts.", max_attempts)
                return []
            wait = 2 ** attempt
            logger.warning("OpenAlex request failed, attempt %d/%d — retrying in %ds.", attempt, max_attempts, wait)
            time.sleep(wait)

    results = data.get("results", [])
    _known = known_paper_ids or set()

    papers: list[Paper] = []
    for item in results:
        title = item.get("title", "")
        if not title:
            continue

        # Reconstruct abstract from inverted index
        abstract = reconstruct_abstract(item.get("abstract_inverted_index"))
        if not abstract:
            continue

        # Extract DOI for deduplication
        raw_doi = item.get("doi", "") or ""
        doi = raw_doi.removeprefix("https://doi.org/")
        if doi and doi in _known:
            logger.debug("Skipping OpenAlex paper %s — already known via DOI.", doi)
            continue

        # Extract OpenAlex ID
        oa_id = item.get("id", "")
        short_id = oa_id.split("/")[-1] if oa_id else ""
        if not short_id:
            continue

        # Authors
        authorships = item.get("authorships", [])
        authors = [
            a["author"]["display_name"]
            for a in authorships
            if a.get("author", {}).get("display_name")
        ]

        # Publication date
        pub_date_str = item.get("publication_date")
        if pub_date_str:
            try:
                published = datetime.strptime(pub_date_str, "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                published = datetime.now(timezone.utc)
        else:
            published = datetime.now(timezone.utc)

        # Extract topic subfields (most useful granularity)
        topics = item.get("topics", [])
        seen_subfields: set[str] = set()
        fields_of_study: list[str] = []
        for topic in topics:
            subfield = (topic.get("subfield") or {}).get("display_name", "")
            if subfield and subfield not in seen_subfields:
                seen_subfields.add(subfield)
                fields_of_study.append(subfield)

        # URL: prefer DOI, fall back to OpenAlex page
        url = raw_doi if raw_doi else oa_id

        papers.append(
            Paper(
                paper_id=f"oa_{short_id}",
                title=title,
                authors=authors,
                abstract=abstract,
                url=url,
                published=published,
                source="openalex",
                pdf_path=None,
                fields_of_study=fields_of_study,
            )
        )

    logger.info("Fetched %d papers from OpenAlex.", len(papers))
    return papers
