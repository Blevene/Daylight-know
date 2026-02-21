"""Optional GitHub trending repository module.

EARS coverage
─────────────
- Optional 2.5-1: query GitHub Search API for recent repos matching
  configured languages.
- Optional 2.5-2: append top-N repos to the summarization prompt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import requests

from digest_pipeline.config import Settings

logger = logging.getLogger(__name__)

GITHUB_SEARCH_URL = "https://api.github.com/search/repositories"


@dataclass
class TrendingRepo:
    name: str
    description: str
    url: str
    stars: int
    language: str


def fetch_trending(settings: Settings) -> list[TrendingRepo]:
    """Fetch the top-N trending GitHub repos created in the last 24 hours.

    Only called when ``settings.github_enabled`` is ``True`` (EARS 2.5-1).
    """
    if not settings.github_enabled:
        return []

    cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).strftime("%Y-%m-%d")
    lang_filter = " ".join(f"language:{lang}" for lang in settings.github_languages)
    query = f"created:>{cutoff} {lang_filter}"

    logger.info("Querying GitHub Search API: %s", query)
    try:
        resp = requests.get(
            GITHUB_SEARCH_URL,
            params={"q": query, "sort": "stars", "order": "desc", "per_page": settings.github_top_n},
            headers={"Accept": "application/vnd.github+json"},
            timeout=30,
        )
        resp.raise_for_status()
    except requests.RequestException:
        logger.exception("GitHub API request failed.")
        return []

    repos: list[TrendingRepo] = []
    for item in resp.json().get("items", [])[:settings.github_top_n]:
        repos.append(
            TrendingRepo(
                name=item.get("full_name", ""),
                description=item.get("description", "") or "",
                url=item.get("html_url", ""),
                stars=item.get("stargazers_count", 0),
                language=item.get("language", "") or "",
            )
        )
    logger.info("Found %d trending GitHub repos.", len(repos))
    return repos


def format_for_prompt(repos: list[TrendingRepo]) -> str:
    """Format trending repos into a text block for the LLM prompt (EARS 2.5-2)."""
    if not repos:
        return ""
    lines: list[str] = []
    for i, r in enumerate(repos, 1):
        lines.append(f"{i}. **{r.name}** ({r.language}, {r.stars} stars)")
        lines.append(f"   {r.description}")
        lines.append(f"   {r.url}")
    return "\n".join(lines)
