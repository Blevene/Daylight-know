# Replace Semantic Scholar with OpenAlex Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the Semantic Scholar integration with OpenAlex for dramatically better abstract coverage on recent papers (68% vs 9% day-old), no rate limiting issues, and richer topic categorization.

**Architecture:** Drop `s2_fetcher.py` entirely and create `openalex_fetcher.py` that queries the OpenAlex Works API. OpenAlex uses a 4-level topic hierarchy (domain > field > subfield > topic) replacing S2's flat fields-of-study list. Configuration changes from `SEMANTICSCHOLAR_*` env vars to `OPENALEX_*`. Abstracts come as inverted indexes that we reconstruct into plaintext. The Paper dataclass's `fields_of_study` field is reused for OpenAlex topics. Deduplication against arXiv continues via DOI/title matching since OpenAlex doesn't always expose arXiv IDs directly.

**Tech Stack:** Python requests, OpenAlex REST API (no SDK needed), pydantic-settings for config

---

## Reference: OpenAlex API

- **Base URL:** `https://api.openalex.org/works`
- **Auth:** `api_key` query param OR polite pool via `mailto:` in User-Agent header
- **Topic hierarchy:** 4 domains → 26 fields → ~200 subfields → ~4,500 topics
- **Filter syntax:** `?filter=from_publication_date:2026-02-21,primary_topic.subfield.id:1702`
- **Abstract format:** Inverted index `{"word": [positions]}` → reconstruct by sorting positions
- **Rate limits:** Generous free tier ($1/day budget), API key for higher limits

### Key OpenAlex Fields (26 total)

| ID | Field | Domain |
|----|-------|--------|
| 17 | Computer Science | Physical Sciences |
| 26 | Mathematics | Physical Sciences |
| 31 | Physics and Astronomy | Physical Sciences |
| 22 | Engineering | Physical Sciences |
| 13 | Biochemistry, Genetics and Molecular Biology | Life Sciences |
| 27 | Medicine | Health Sciences |
| 33 | Social Sciences | Social Sciences |
| 20 | Economics, Econometrics and Finance | Social Sciences |

### Key Computer Science Subfields

| ID | Subfield |
|----|----------|
| 1702 | Artificial Intelligence |
| 1703 | Computational Theory and Mathematics |
| 1705 | Computer Networks and Communications |
| 1706 | Computer Science Applications |
| 1707 | Computer Vision and Pattern Recognition |
| 1710 | Information Systems |
| 1712 | Software |

---

### Task 1: Create OpenAlex Fetcher Module

**Files:**
- Create: `src/digest_pipeline/openalex_fetcher.py`
- Test: `tests/unit/test_openalex_fetcher.py`

**Step 1: Write the failing tests**

Create `tests/unit/test_openalex_fetcher.py`:

```python
"""Tests for the OpenAlex fetcher module."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from digest_pipeline.openalex_fetcher import (
    fetch_openalex_papers,
    reconstruct_abstract,
    OPENALEX_FIELDS,
)


class TestReconstructAbstract:
    def test_basic_reconstruction(self):
        inv_index = {"Hello": [0], "world": [1]}
        assert reconstruct_abstract(inv_index) == "Hello world"

    def test_word_at_multiple_positions(self):
        inv_index = {"the": [0, 2], "cat": [1], "sat": [3]}
        assert reconstruct_abstract(inv_index) == "the cat the sat"

    def test_empty_index(self):
        assert reconstruct_abstract({}) == ""

    def test_none_input(self):
        assert reconstruct_abstract(None) == ""


def _make_openalex_work(
    openalex_id="W123",
    title="OA Paper",
    abstract_words=None,
    pub_date=None,
    doi=None,
    topics=None,
    oa_url=None,
):
    """Build a fake OpenAlex work object."""
    if pub_date is None:
        pub_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if abstract_words is None:
        abstract_words = {"A": [0], "study": [1], "on": [2], "testing.": [3]}

    work = {
        "id": f"https://openalex.org/{openalex_id}",
        "title": title,
        "authorships": [
            {"author": {"display_name": "Alice"}},
            {"author": {"display_name": "Bob"}},
        ],
        "abstract_inverted_index": abstract_words,
        "publication_date": pub_date,
        "doi": f"https://doi.org/{doi}" if doi else None,
        "ids": {"openalex": f"https://openalex.org/{openalex_id}"},
        "open_access": {"oa_url": oa_url},
        "primary_location": {
            "source": {"display_name": "Test Journal"},
        },
        "topics": topics or [],
    }
    return work


@patch("digest_pipeline.openalex_fetcher.requests.get")
def test_fetch_openalex_papers_success(mock_get, make_settings):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "meta": {"count": 1},
        "results": [_make_openalex_work()],
    }
    mock_get.return_value = mock_resp

    settings = make_settings(
        openalex_enabled=True,
        openalex_query="deep learning",
        openalex_max_results=10,
    )
    papers = fetch_openalex_papers(settings)

    assert len(papers) == 1
    assert papers[0].source == "openalex"
    assert papers[0].paper_id.startswith("oa_")
    assert papers[0].title == "OA Paper"
    assert papers[0].authors == ["Alice", "Bob"]
    assert papers[0].abstract == "A study on testing."
    assert papers[0].pdf_path is None


@patch("digest_pipeline.openalex_fetcher.requests.get")
def test_fetch_openalex_papers_uses_query_and_filters(mock_get, make_settings):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"meta": {"count": 0}, "results": []}
    mock_get.return_value = mock_resp

    settings = make_settings(
        openalex_enabled=True,
        openalex_query="reinforcement learning",
        openalex_max_results=5,
        openalex_fields=["Computer Science"],
    )
    fetch_openalex_papers(settings)

    call_args = mock_get.call_args
    params = call_args.kwargs.get("params") or call_args[1].get("params", {})
    assert "reinforcement learning" in params.get("search", "")
    assert "from_publication_date" in params.get("filter", "")


@patch("digest_pipeline.openalex_fetcher.requests.get")
def test_fetch_openalex_papers_sends_api_key(mock_get, make_settings):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"meta": {"count": 0}, "results": []}
    mock_get.return_value = mock_resp

    settings = make_settings(
        openalex_enabled=True,
        openalex_api_key="test-oa-key",
    )
    fetch_openalex_papers(settings)

    call_args = mock_get.call_args
    params = call_args.kwargs.get("params") or call_args[1].get("params", {})
    assert params.get("api_key") == "test-oa-key"


@patch("digest_pipeline.openalex_fetcher.requests.get")
def test_fetch_openalex_papers_sends_email_in_user_agent(mock_get, make_settings):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"meta": {"count": 0}, "results": []}
    mock_get.return_value = mock_resp

    settings = make_settings(
        openalex_enabled=True,
        openalex_email="user@example.com",
    )
    fetch_openalex_papers(settings)

    call_args = mock_get.call_args
    headers = call_args.kwargs.get("headers") or call_args[1].get("headers", {})
    assert "user@example.com" in headers.get("User-Agent", "")


@patch("digest_pipeline.openalex_fetcher.requests.get")
def test_fetch_openalex_papers_handles_error(mock_get, make_settings):
    mock_get.side_effect = Exception("network error")

    settings = make_settings(openalex_enabled=True)
    papers = fetch_openalex_papers(settings)

    assert papers == []


@patch("digest_pipeline.openalex_fetcher.requests.get")
def test_fetch_openalex_papers_skips_no_abstract(mock_get, make_settings):
    """Papers without abstracts are skipped."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "meta": {"count": 2},
        "results": [
            _make_openalex_work(openalex_id="W1", abstract_words=None),
            _make_openalex_work(openalex_id="W2", abstract_words={}),
            _make_openalex_work(openalex_id="W3", title="Good Paper"),
        ],
    }
    mock_get.return_value = mock_resp

    settings = make_settings(openalex_enabled=True)
    papers = fetch_openalex_papers(settings)

    assert len(papers) == 1
    assert papers[0].title == "Good Paper"


@patch("digest_pipeline.openalex_fetcher.requests.get")
def test_fetch_openalex_papers_deduplicates_known_ids(mock_get, make_settings):
    """Papers with DOIs already in known set are skipped."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "meta": {"count": 2},
        "results": [
            _make_openalex_work(openalex_id="W1", doi="10.1234/dup"),
            _make_openalex_work(openalex_id="W2", title="New Paper", doi="10.1234/new"),
        ],
    }
    mock_get.return_value = mock_resp

    settings = make_settings(openalex_enabled=True)
    papers = fetch_openalex_papers(
        settings,
        known_paper_ids={"10.1234/dup"},
    )

    assert len(papers) == 1
    assert papers[0].title == "New Paper"


@patch("digest_pipeline.openalex_fetcher.requests.get")
def test_fetch_openalex_papers_extracts_topics(mock_get, make_settings):
    """Topic hierarchy from OpenAlex is stored in fields_of_study."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "meta": {"count": 1},
        "results": [
            _make_openalex_work(
                topics=[
                    {
                        "display_name": "Neural Networks",
                        "subfield": {"display_name": "Artificial Intelligence"},
                        "field": {"display_name": "Computer Science"},
                        "score": 0.95,
                    },
                    {
                        "display_name": "Deep Learning",
                        "subfield": {"display_name": "Artificial Intelligence"},
                        "field": {"display_name": "Computer Science"},
                        "score": 0.80,
                    },
                ]
            ),
        ],
    }
    mock_get.return_value = mock_resp

    settings = make_settings(openalex_enabled=True)
    papers = fetch_openalex_papers(settings)

    assert len(papers) == 1
    # fields_of_study should contain the subfield names (most useful granularity)
    assert "Artificial Intelligence" in papers[0].fields_of_study


@patch("digest_pipeline.openalex_fetcher.requests.get")
def test_fetch_openalex_papers_field_filter(mock_get, make_settings):
    """When openalex_fields is set, the filter includes field IDs."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"meta": {"count": 0}, "results": []}
    mock_get.return_value = mock_resp

    settings = make_settings(
        openalex_enabled=True,
        openalex_fields=["Computer Science", "Mathematics"],
    )
    fetch_openalex_papers(settings)

    call_args = mock_get.call_args
    params = call_args.kwargs.get("params") or call_args[1].get("params", {})
    filter_str = params.get("filter", "")
    # Should contain field IDs for CS (17) and Math (26)
    assert "primary_topic.field.id" in filter_str
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/angel/Documents/code/Daylight-know && python -m pytest tests/unit/test_openalex_fetcher.py -v 2>&1 | head -30`
Expected: FAIL with `ModuleNotFoundError: No module named 'digest_pipeline.openalex_fetcher'`

**Step 3: Write the implementation**

Create `src/digest_pipeline/openalex_fetcher.py`:

```python
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
    filters = [f"from_publication_date:{date_from}", "type:article"]

    if settings.openalex_fields:
        field_ids = []
        for name in settings.openalex_fields:
            fid = OPENALEX_FIELDS.get(name)
            if fid:
                field_ids.append(str(fid))
            else:
                logger.warning("Unknown OpenAlex field: %r — skipping filter.", name)
        if field_ids:
            filters.append(f"primary_topic.field.id:{"|".join(field_ids)}")

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

    try:
        resp = requests.get(
            OPENALEX_WORKS_URL,
            params=params,
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        logger.exception("Failed to fetch papers from OpenAlex.")
        return []

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

        # OA URL (for potential future PDF download)
        oa_url = (item.get("open_access") or {}).get("oa_url", "")

        papers.append(
            Paper(
                paper_id=f"oa_{short_id}",
                title=title,
                authors=authors,
                abstract=abstract,
                url=url or oa_id,
                published=published,
                source="openalex",
                pdf_path=None,
                fields_of_study=fields_of_study,
            )
        )

    logger.info("Fetched %d papers from OpenAlex.", len(papers))
    return papers
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/angel/Documents/code/Daylight-know && python -m pytest tests/unit/test_openalex_fetcher.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/digest_pipeline/openalex_fetcher.py tests/unit/test_openalex_fetcher.py
git commit -m "feat: add OpenAlex fetcher module with topic filtering"
```

---

### Task 2: Update Config — Replace S2 Settings with OpenAlex

**Files:**
- Modify: `src/digest_pipeline/config.py:54-59` (replace S2 block)
- Test: existing `make_settings` fixture should still work

**Step 1: Write the failing test**

No new test file needed — we verify by importing the new fields. Add a quick check at the end of `test_openalex_fetcher.py` tests (already covered by `make_settings(openalex_enabled=True, ...)` calls above).

**Step 2: Modify config.py**

Replace the Semantic Scholar settings block (lines 54-59) with:

```python
    # ── OpenAlex (optional) ────────────────────────────────────────
    openalex_enabled: bool = Field(default=False)
    openalex_api_key: str = Field(default="")
    openalex_email: str = Field(default="")
    openalex_max_results: int = Field(default=20)
    openalex_query: str = Field(default="machine learning")
    openalex_fields: list[str] = Field(default_factory=list)
```

**Step 3: Run tests**

Run: `cd /home/angel/Documents/code/Daylight-know && python -m pytest tests/unit/test_openalex_fetcher.py tests/unit/test_fetcher.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/digest_pipeline/config.py
git commit -m "feat: replace Semantic Scholar config with OpenAlex settings"
```

---

### Task 3: Update Pipeline to Use OpenAlex Instead of S2

**Files:**
- Modify: `src/digest_pipeline/pipeline.py:40,120-125` (swap import and usage)
- Test: `tests/unit/test_pipeline.py`

**Step 1: Read current test_pipeline.py to understand existing S2 pipeline test coverage**

**Step 2: Update pipeline.py**

In imports, replace:
```python
from digest_pipeline.s2_fetcher import fetch_s2_papers
```
with:
```python
from digest_pipeline.openalex_fetcher import fetch_openalex_papers
```

Replace the S2 block (lines 120-125):
```python
    if settings.semanticscholar_enabled:
        # Rebuild known IDs to include HF new papers so S2 doesn't duplicate them.
        all_known_ids = {normalize_arxiv_id(p.paper_id) for p in papers}
        s2_papers = fetch_s2_papers(settings, known_arxiv_ids=all_known_ids)
        logger.info("Fetched %d papers from Semantic Scholar.", len(s2_papers))
        papers.extend(s2_papers)
```
with:
```python
    if settings.openalex_enabled:
        # Build set of known DOIs to avoid duplicating arXiv/HF papers.
        all_known_dois: set[str] = set()
        for p in papers:
            # arXiv papers have DOIs like 10.48550/arXiv.XXXX.XXXXX
            if p.source == "arxiv":
                raw_id = p.paper_id.split("v")[0]  # strip version
                all_known_dois.add(f"10.48550/arXiv.{raw_id}")
        oa_papers = fetch_openalex_papers(settings, known_paper_ids=all_known_dois)
        logger.info("Fetched %d papers from OpenAlex.", len(oa_papers))
        papers.extend(oa_papers)
```

Also update the docstring at the top of pipeline.py: change `1c. (Optional) Fetch papers from Semantic Scholar` to `1c. (Optional) Fetch papers from OpenAlex`.

And update the comment on line 147:
```python
            # Papers without PDFs (e.g. HuggingFace, OpenAlex) —
```

**Step 3: Update test_pipeline.py**

Replace any `semanticscholar` references with `openalex` equivalents. Patch `digest_pipeline.pipeline.fetch_openalex_papers` instead of `fetch_s2_papers`.

**Step 4: Run tests**

Run: `cd /home/angel/Documents/code/Daylight-know && python -m pytest tests/unit/test_pipeline.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/digest_pipeline/pipeline.py tests/unit/test_pipeline.py
git commit -m "feat: wire OpenAlex into pipeline, replacing Semantic Scholar"
```

---

### Task 4: Update Setup Wizard

**Files:**
- Modify: `src/digest_pipeline/setup.py` (replace S2 section with OpenAlex)

**Step 1: Replace `_collect_s2_fields_of_study()` with `_collect_openalex_fields()`**

```python
def _collect_openalex_fields() -> list[str]:
    """Interactive multi-select for OpenAlex academic fields."""
    from digest_pipeline.openalex_fetcher import OPENALEX_FIELDS

    field_names = list(OPENALEX_FIELDS.keys())

    console.print("\n  [bold]Select OpenAlex fields to filter by:[/]")
    console.print("  [dim]Enter numbers (comma-separated), 'all', or press Enter to skip.[/]\n")

    table = Table(show_header=False, box=None, padding=(0, 2))
    half = (len(field_names) + 1) // 2
    for i in range(half):
        left = f"  {i + 1:>2}. {field_names[i]}"
        right_idx = i + half
        if right_idx < len(field_names):
            right = f"{right_idx + 1:>2}. {field_names[right_idx]}"
        else:
            right = ""
        table.add_row(left, right)

    console.print(table)

    try:
        picks = console.input("\n  Fields (numbers, 'all', or Enter to skip): ").strip()
    except EOFError:
        return []

    if not picks:
        return []

    if picks.lower() == "all":
        return field_names

    selected: list[str] = []
    for p in picks.split(","):
        p = p.strip()
        try:
            idx = int(p) - 1
            if 0 <= idx < len(field_names):
                name = field_names[idx]
                if name not in selected:
                    selected.append(name)
                    console.print(f"    [green]✓[/] {name}")
            else:
                console.print(f"    [yellow]⚠ Invalid number: {p}[/]")
        except ValueError:
            match = [f for f in field_names if f.lower() == p.lower()]
            if match and match[0] not in selected:
                selected.append(match[0])
                console.print(f"    [green]✓[/] {match[0]}")
            else:
                console.print(f"    [yellow]⚠ Unknown field: {p}[/]")

    return selected
```

**Step 2: Update `_collect_optional_settings()`**

Replace the S2 block (lines 395-407) with:

```python
    if _prompt_bool("Enable OpenAlex integration?", default=False):
        config["OPENALEX_ENABLED"] = "true"
        config["OPENALEX_API_KEY"] = _prompt(
            "OpenAlex API key (optional, press Enter to skip)"
        )
        config["OPENALEX_EMAIL"] = _prompt(
            "Email for OpenAlex polite pool (optional)"
        )
        config["OPENALEX_MAX_RESULTS"] = _prompt("OpenAlex max results", "20")
        config["OPENALEX_QUERY"] = _prompt("OpenAlex search query", "machine learning")
        selected_fields = _collect_openalex_fields()
        if selected_fields:
            import json as _json
            config["OPENALEX_FIELDS"] = _json.dumps(selected_fields)
    else:
        config["OPENALEX_ENABLED"] = "false"
```

**Step 3: Update `_write_env_file()` sections**

Replace the "Optional: Semantic Scholar" section (lines 527-535) with:

```python
        (
            "Optional: OpenAlex",
            [
                "OPENALEX_ENABLED",
                "OPENALEX_API_KEY",
                "OPENALEX_EMAIL",
                "OPENALEX_MAX_RESULTS",
                "OPENALEX_QUERY",
                "OPENALEX_FIELDS",
            ],
        ),
```

**Step 4: Update the welcome panel text**

Replace `"research digest pipeline (arXiv, HuggingFace, Semantic Scholar)."` with `"research digest pipeline (arXiv, HuggingFace, OpenAlex)."`.

**Step 5: Remove the `_collect_s2_fields_of_study` import of `S2_FIELDS_OF_STUDY`**

Delete the function `_collect_s2_fields_of_study()` entirely.

**Step 6: Run tests**

Run: `cd /home/angel/Documents/code/Daylight-know && python -m pytest tests/unit/ -v -k "not test_s2" 2>&1 | tail -20`
Expected: PASS (no S2 references remain in active code)

**Step 7: Commit**

```bash
git add src/digest_pipeline/setup.py
git commit -m "feat: replace Semantic Scholar with OpenAlex in setup wizard"
```

---

### Task 5: Update .env.example and Emailer Tests

**Files:**
- Modify: `.env.example` (replace S2 section with OpenAlex)
- Modify: `tests/unit/test_emailer.py` (update source references)
- Modify: `tests/unit/test_llm_utils.py` (update source references)

**Step 1: Update .env.example**

Replace lines 43-52 (Semantic Scholar section) with:

```
# ── Optional: OpenAlex ──────────────────────────────────────────
OPENALEX_ENABLED="false"
# Optional: OpenAlex API key (for higher rate limits)
# OPENALEX_API_KEY="..."
# Optional: email for OpenAlex polite pool (recommended)
# OPENALEX_EMAIL="you@example.com"
OPENALEX_MAX_RESULTS="20"
OPENALEX_QUERY="machine learning"
# Optional: filter by academic fields (JSON array)
# Valid fields: Computer Science, Mathematics, Physics and Astronomy,
#   Chemistry, Engineering, Medicine, Biology, Economics, etc.
# OPENALEX_FIELDS=["Computer Science","Mathematics"]
```

**Step 2: Update test_emailer.py**

Change `source="semanticscholar"` to `source="openalex"` and update the URL from `https://semanticscholar.org/paper/abc` to `https://openalex.org/W123` in the `test_build_email_with_fields_of_study` test.

**Step 3: Update test_llm_utils.py**

Change `source="semanticscholar"` to `source="openalex"` in the `test_includes_source` test, and update the assertion from `"**Source:** semanticscholar"` to `"**Source:** openalex"`.

**Step 4: Run tests**

Run: `cd /home/angel/Documents/code/Daylight-know && python -m pytest tests/unit/test_emailer.py tests/unit/test_llm_utils.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add .env.example tests/unit/test_emailer.py tests/unit/test_llm_utils.py
git commit -m "chore: update .env.example and tests for OpenAlex migration"
```

---

### Task 6: Update Network Smoke Tests and Add OpenAlex Smoke Test

**Files:**
- Modify: `tests/integration/test_network_smoke.py`

**Step 1: Add OpenAlex smoke test**

Add to `TestNetworkSmoke` class:

```python
    def test_fetch_openalex_real(self):
        """Hit the live OpenAlex API and verify Paper objects."""
        from digest_pipeline.openalex_fetcher import fetch_openalex_papers

        settings = _make_settings(
            openalex_enabled=True,
            openalex_query="machine learning",
            openalex_max_results=3,
        )
        papers = fetch_openalex_papers(settings)

        # OpenAlex should always return papers for "machine learning"
        assert isinstance(papers, list)
        for p in papers:
            assert isinstance(p, Paper)
            assert p.source == "openalex"
            assert p.paper_id.startswith("oa_")
            assert p.abstract  # OpenAlex papers we return always have abstracts
            assert p.title
```

**Step 2: Run tests**

Run: `cd /home/angel/Documents/code/Daylight-know && python -m pytest tests/integration/test_network_smoke.py -v -k openalex -m network`
Expected: PASS (if network available)

**Step 3: Commit**

```bash
git add tests/integration/test_network_smoke.py
git commit -m "test: add OpenAlex network smoke test"
```

---

### Task 7: Delete S2 Fetcher and Its Tests

**Files:**
- Delete: `src/digest_pipeline/s2_fetcher.py`
- Delete: `tests/unit/test_s2_fetcher.py`

**Step 1: Verify no remaining imports of s2_fetcher**

Run: `grep -r "s2_fetcher\|fetch_s2_papers\|S2_FIELDS_OF_STUDY\|S2_SEARCH_URL" src/ tests/ --include="*.py"`
Expected: Only hits in the files we're about to delete

**Step 2: Delete the files**

```bash
git rm src/digest_pipeline/s2_fetcher.py tests/unit/test_s2_fetcher.py
```

**Step 3: Run full test suite**

Run: `cd /home/angel/Documents/code/Daylight-know && python -m pytest tests/unit/ -v 2>&1 | tail -30`
Expected: All PASS, no import errors

**Step 4: Commit**

```bash
git commit -m "chore: remove Semantic Scholar fetcher (replaced by OpenAlex)"
```

---

### Task 8: Final Verification

**Step 1: Run full unit test suite**

Run: `cd /home/angel/Documents/code/Daylight-know && python -m pytest tests/unit/ -v`
Expected: All PASS

**Step 2: Verify no remaining S2 references in source code**

Run: `grep -ri "semantic.scholar\|semanticscholar\|s2_fetcher" src/ tests/ .env.example --include="*.py" --include="*.example"`
Expected: No matches

**Step 3: Verify imports work**

Run: `python -c "from digest_pipeline.openalex_fetcher import fetch_openalex_papers, OPENALEX_FIELDS; print(f'{len(OPENALEX_FIELDS)} fields loaded')"`
Expected: `26 fields loaded`

**Step 4: Commit (if any stragglers)**

```bash
git add -A && git status
# Only commit if there are changes
```
