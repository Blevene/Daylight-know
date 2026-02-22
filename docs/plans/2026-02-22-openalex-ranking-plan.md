# OpenAlex Hybrid Ranking Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add LLM-based relevance ranking to filter OpenAlex papers against a user interest profile before including them in the digest.

**Architecture:** Fetch a wide pool (100 papers) from OpenAlex using subfield filters only (no search term). A new `ranker.py` module scores each paper via keyword boosting + LLM batch scoring against a natural language interest profile. Only the top N papers pass through to the existing summarize/implications/critiques pipeline.

**Tech Stack:** litellm (existing), structured JSON output, pytest

---

### Task 1: Add config fields for ranking

**Files:**
- Modify: `src/digest_pipeline/config.py:54-60`
- Modify: `.env.example:43-54`
- Test: `tests/unit/test_config.py` (if exists, otherwise verify via make_settings)

**Step 1: Write the failing test**

Create `tests/unit/test_ranker_config.py`:

```python
"""Tests for ranking-related config fields."""

from digest_pipeline.config import Settings


def test_ranking_config_defaults():
    s = Settings(_env_file=None, llm_api_key="k", smtp_user="u",
                 smtp_password="p", email_from="a@b", email_to="c@d")
    assert s.openalex_interest_profile == ""
    assert s.openalex_interest_keywords == []
    assert s.openalex_fetch_pool == 100


def test_ranking_config_from_values():
    s = Settings(
        _env_file=None, llm_api_key="k", smtp_user="u",
        smtp_password="p", email_from="a@b", email_to="c@d",
        openalex_interest_profile="I study LLMs for drug discovery",
        openalex_interest_keywords=["LLM", "drug discovery"],
        openalex_fetch_pool=50,
    )
    assert s.openalex_interest_profile == "I study LLMs for drug discovery"
    assert s.openalex_interest_keywords == ["LLM", "drug discovery"]
    assert s.openalex_fetch_pool == 50
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_ranker_config.py -v`
Expected: FAIL — `openalex_interest_profile` not found on Settings

**Step 3: Write minimal implementation**

Add to `src/digest_pipeline/config.py` in the OpenAlex section (after line 60):

```python
    openalex_interest_profile: str = Field(default="")
    openalex_interest_keywords: list[str] = Field(default_factory=list)
    openalex_fetch_pool: int = Field(default=100)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_ranker_config.py -v`
Expected: PASS

**Step 5: Update .env.example**

Add after the existing OpenAlex section:

```
# Optional: Interest-based ranking (filters OpenAlex papers by relevance)
# Natural language description of your research interests
# OPENALEX_INTEREST_PROFILE="I study how large language models can be applied to scientific discovery"
# Comma-separated keywords for boosting (JSON array)
# OPENALEX_INTEREST_KEYWORDS=["LLM","drug discovery","protein folding"]
# How many papers to fetch before ranking (ranked down to OPENALEX_MAX_RESULTS)
OPENALEX_FETCH_POOL="100"
```

**Step 6: Commit**

```bash
git add src/digest_pipeline/config.py .env.example tests/unit/test_ranker_config.py
git commit -m "feat: add config fields for OpenAlex interest-based ranking"
```

---

### Task 2: Create ranking prompt file

**Files:**
- Create: `src/digest_pipeline/prompts/ranker.md`

**Step 1: Create the prompt file**

```markdown
You are a research relevance scorer. Given a researcher's interest profile and a batch of paper titles and abstracts, rate each paper's relevance on a scale of 1-10.

Scoring guide:
- 10: Directly addresses the researcher's core interests
- 7-9: Strongly related, covers adjacent topics or methods
- 4-6: Tangentially related, shares some themes
- 1-3: Minimally related or unrelated

Researcher's interest profile:
{interest_profile}

Rate each paper below. Return ONLY a JSON object mapping paper keys to integer scores.
```

**Step 2: Verify prompt loads**

Run: `python -c "from digest_pipeline.prompts import load_prompt; print(load_prompt('ranker')[:50])"`
Expected: prints first 50 chars of the prompt

**Step 3: Commit**

```bash
git add src/digest_pipeline/prompts/ranker.md
git commit -m "feat: add LLM scoring prompt for paper ranking"
```

---

### Task 3: Implement keyword scoring

**Files:**
- Create: `src/digest_pipeline/ranker.py`
- Create: `tests/unit/test_ranker.py`

**Step 1: Write the failing test**

```python
"""Tests for the paper ranker module."""

from digest_pipeline.ranker import compute_keyword_scores


def test_keyword_scoring_basic(make_paper):
    papers = [
        make_paper(title="LLM for Drug Discovery", abstract="We apply large language models to find drugs."),
        make_paper(title="Image Segmentation", abstract="A new method for segmenting images."),
    ]
    scores = compute_keyword_scores(papers, ["LLM", "drug discovery"])
    assert scores[0] > scores[1]


def test_keyword_scoring_case_insensitive(make_paper):
    papers = [make_paper(title="llm research", abstract="about llm")]
    scores = compute_keyword_scores(papers, ["LLM"])
    assert scores[0] == 4  # +2 for title match, +2 for abstract match


def test_keyword_scoring_empty_keywords(make_paper):
    papers = [make_paper()]
    scores = compute_keyword_scores(papers, [])
    assert scores == [0]


def test_keyword_scoring_no_match(make_paper):
    papers = [make_paper(title="Unrelated", abstract="Nothing here")]
    scores = compute_keyword_scores(papers, ["quantum"])
    assert scores == [0]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_ranker.py::test_keyword_scoring_basic -v`
Expected: FAIL — `cannot import name 'compute_keyword_scores'`

**Step 3: Write minimal implementation**

Create `src/digest_pipeline/ranker.py`:

```python
"""Interest-based paper ranking with keyword boosting and LLM scoring.

Scores papers against a user-defined interest profile to select the most
relevant papers from a larger fetch pool. Used between OpenAlex fetching
and the main digest pipeline.
"""

from __future__ import annotations

import logging

from digest_pipeline.config import Settings
from digest_pipeline.fetcher import Paper

logger = logging.getLogger(__name__)

KEYWORD_BOOST = 2


def compute_keyword_scores(papers: list[Paper], keywords: list[str]) -> list[int]:
    """Score each paper by keyword matches in title and abstract.

    Returns a list of integer scores (one per paper). Each keyword match
    in the title adds KEYWORD_BOOST points, and each match in the
    abstract adds KEYWORD_BOOST points.
    """
    if not keywords:
        return [0] * len(papers)

    lower_keywords = [kw.lower() for kw in keywords]
    scores: list[int] = []
    for paper in papers:
        score = 0
        title_lower = paper.title.lower()
        abstract_lower = paper.abstract.lower()
        for kw in lower_keywords:
            if kw in title_lower:
                score += KEYWORD_BOOST
            if kw in abstract_lower:
                score += KEYWORD_BOOST
        scores.append(score)
    return scores
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_ranker.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add src/digest_pipeline/ranker.py tests/unit/test_ranker.py
git commit -m "feat: add keyword scoring for paper ranking"
```

---

### Task 4: Implement LLM batch scoring

**Files:**
- Modify: `src/digest_pipeline/ranker.py`
- Modify: `tests/unit/test_ranker.py`

**Step 1: Write the failing tests**

Add to `tests/unit/test_ranker.py`:

```python
from unittest.mock import patch, MagicMock
from digest_pipeline.ranker import score_batch_with_llm


@patch("digest_pipeline.ranker.litellm.completion")
def test_llm_batch_scoring(mock_completion, make_paper, make_settings):
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = '{"paper_1": 8, "paper_2": 3}'
    mock_completion.return_value = mock_resp

    papers = [
        make_paper(title="LLM Drug Discovery"),
        make_paper(title="Image Segmentation"),
    ]
    settings = make_settings(
        openalex_interest_profile="I study LLMs for drug discovery",
    )
    scores = score_batch_with_llm(papers, settings)
    assert scores == [8, 3]


@patch("digest_pipeline.ranker.litellm.completion")
def test_llm_batch_scoring_handles_failure(mock_completion, make_paper, make_settings):
    mock_completion.side_effect = Exception("API error")

    papers = [make_paper(), make_paper()]
    settings = make_settings(openalex_interest_profile="anything")
    scores = score_batch_with_llm(papers, settings)
    assert scores == [0, 0]  # graceful degradation


@patch("digest_pipeline.ranker.litellm.completion")
def test_llm_batch_scoring_handles_invalid_json(mock_completion, make_paper, make_settings):
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = "not json"
    mock_completion.return_value = mock_resp

    papers = [make_paper()]
    settings = make_settings(openalex_interest_profile="anything")
    scores = score_batch_with_llm(papers, settings)
    assert scores == [0]
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_ranker.py::test_llm_batch_scoring -v`
Expected: FAIL — `cannot import name 'score_batch_with_llm'`

**Step 3: Write implementation**

Add to `src/digest_pipeline/ranker.py`:

```python
import json

import litellm

from digest_pipeline.prompts import load_prompt

BATCH_SIZE = 20


def score_batch_with_llm(papers: list[Paper], settings: Settings) -> list[int]:
    """Score a batch of papers using the LLM.

    Sends paper titles and abstracts to the configured LLM with the
    interest profile. Returns a list of integer scores (1-10) per paper.
    On failure, returns zeros for graceful degradation.
    """
    if not papers:
        return []

    system_prompt = load_prompt("ranker").replace(
        "{interest_profile}", settings.openalex_interest_profile
    )

    # Build user message with paper titles and abstracts
    parts: list[str] = []
    for i, p in enumerate(papers, 1):
        parts.append(f"paper_{i}: {p.title}\n{p.abstract[:500]}")
    user_prompt = "\n---\n".join(parts)

    # Build response format expecting integer scores
    properties = {
        f"paper_{i}": {"type": "integer"} for i in range(1, len(papers) + 1)
    }
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "relevance_scores",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": properties,
                "required": list(properties.keys()),
                "additionalProperties": False,
            },
        },
    }

    try:
        response = litellm.completion(
            model=settings.llm_model,
            max_tokens=256,
            api_key=settings.llm_api_key,
            api_base=settings.llm_api_base,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=response_format,
        )
        raw = response.choices[0].message.content or ""
        parsed = json.loads(raw)
        return [
            int(parsed.get(f"paper_{i}", 0))
            for i in range(1, len(papers) + 1)
        ]
    except Exception:
        logger.warning("LLM scoring failed — falling back to zero scores.", exc_info=True)
        return [0] * len(papers)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_ranker.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add src/digest_pipeline/ranker.py tests/unit/test_ranker.py
git commit -m "feat: add LLM batch scoring for paper relevance"
```

---

### Task 5: Implement the main rank_papers orchestrator

**Files:**
- Modify: `src/digest_pipeline/ranker.py`
- Modify: `tests/unit/test_ranker.py`

**Step 1: Write the failing tests**

Add to `tests/unit/test_ranker.py`:

```python
from digest_pipeline.ranker import rank_papers


@patch("digest_pipeline.ranker.score_batch_with_llm")
def test_rank_papers_combines_scores(mock_llm_score, make_paper, make_settings):
    """Papers are ranked by combined keyword + LLM score."""
    papers = [
        make_paper(paper_id="low", title="Unrelated Topic", abstract="Nothing relevant here"),
        make_paper(paper_id="high", title="LLM for Drug Discovery", abstract="We use LLM to find drugs"),
    ]
    # LLM gives both a 5, but keywords will differentiate
    mock_llm_score.return_value = [5, 5]

    settings = make_settings(
        openalex_interest_profile="LLMs for drug discovery",
        openalex_interest_keywords=["LLM", "drug"],
        openalex_max_results=1,
    )
    ranked = rank_papers(papers, settings)
    assert len(ranked) == 1
    assert ranked[0].paper_id == "high"


def test_rank_papers_no_profile_returns_unchanged(make_paper, make_settings):
    """When no interest profile or keywords, return papers unchanged."""
    papers = [make_paper(paper_id="a"), make_paper(paper_id="b")]
    settings = make_settings(openalex_max_results=20)
    ranked = rank_papers(papers, settings)
    assert len(ranked) == 2
    assert ranked[0].paper_id == "a"


@patch("digest_pipeline.ranker.score_batch_with_llm")
def test_rank_papers_batches_correctly(mock_llm_score, make_paper, make_settings):
    """Papers are scored in batches of BATCH_SIZE."""
    papers = [make_paper(paper_id=f"p{i}") for i in range(25)]
    mock_llm_score.side_effect = [
        [5] * 20,  # first batch
        [5] * 5,   # second batch
    ]
    settings = make_settings(
        openalex_interest_profile="test",
        openalex_max_results=10,
    )
    ranked = rank_papers(papers, settings)
    assert len(ranked) == 10
    assert mock_llm_score.call_count == 2
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_ranker.py::test_rank_papers_combines_scores -v`
Expected: FAIL — `cannot import name 'rank_papers'`

**Step 3: Write implementation**

Add to `src/digest_pipeline/ranker.py`:

```python
def rank_papers(papers: list[Paper], settings: Settings) -> list[Paper]:
    """Rank papers by relevance and return the top N.

    Combines keyword boost scores with LLM-based relevance scoring.
    If no interest profile or keywords are configured, returns papers
    unchanged (backward compatible).
    """
    if not settings.openalex_interest_profile and not settings.openalex_interest_keywords:
        return papers

    max_results = settings.openalex_max_results
    if len(papers) <= max_results:
        return papers

    # 1. Keyword scores
    keyword_scores = compute_keyword_scores(papers, settings.openalex_interest_keywords)

    # 2. LLM scores (in batches)
    llm_scores = [0] * len(papers)
    if settings.openalex_interest_profile:
        for start in range(0, len(papers), BATCH_SIZE):
            batch = papers[start : start + BATCH_SIZE]
            batch_scores = score_batch_with_llm(batch, settings)
            for j, score in enumerate(batch_scores):
                llm_scores[start + j] = score

    # 3. Combine and sort
    combined = [
        (kw + llm, i, paper)
        for i, (paper, kw, llm) in enumerate(zip(papers, keyword_scores, llm_scores))
    ]
    combined.sort(key=lambda x: (-x[0], x[1]))  # highest score first, stable order

    ranked = [paper for _, _, paper in combined[:max_results]]
    logger.info(
        "Ranked %d papers → top %d (score range: %d–%d).",
        len(papers),
        len(ranked),
        combined[-1][0],
        combined[0][0],
    )
    return ranked
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_ranker.py -v`
Expected: All 10 tests PASS

**Step 5: Commit**

```bash
git add src/digest_pipeline/ranker.py tests/unit/test_ranker.py
git commit -m "feat: add rank_papers orchestrator with batch LLM scoring"
```

---

### Task 6: Update OpenAlex fetcher to use fetch pool

**Files:**
- Modify: `src/digest_pipeline/openalex_fetcher.py:96-125`
- Modify: `tests/unit/test_openalex_fetcher.py`

**Step 1: Write the failing test**

Add to `tests/unit/test_openalex_fetcher.py`:

```python
@patch("digest_pipeline.openalex_fetcher.requests.get")
def test_fetch_openalex_uses_fetch_pool_size(mock_get, make_settings):
    """When fetch_pool is set, per_page uses fetch_pool instead of max_results."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"meta": {"count": 0}, "results": []}
    mock_get.return_value = mock_resp

    settings = make_settings(
        openalex_enabled=True,
        openalex_fetch_pool=100,
        openalex_max_results=20,
    )
    fetch_openalex_papers(settings)

    call_args = mock_get.call_args
    params = call_args.kwargs.get("params") or call_args[1].get("params", {})
    assert params["per_page"] == 100


@patch("digest_pipeline.openalex_fetcher.requests.get")
def test_fetch_openalex_no_search_when_profile_set(mock_get, make_settings):
    """When interest_profile is configured, search param is omitted."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"meta": {"count": 0}, "results": []}
    mock_get.return_value = mock_resp

    settings = make_settings(
        openalex_enabled=True,
        openalex_interest_profile="I study LLMs",
        openalex_query="machine learning",
    )
    fetch_openalex_papers(settings)

    call_args = mock_get.call_args
    params = call_args.kwargs.get("params") or call_args[1].get("params", {})
    assert "search" not in params
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_openalex_fetcher.py::test_fetch_openalex_uses_fetch_pool_size -v`
Expected: FAIL — `openalex_fetch_pool` not recognized or `per_page` is still `max_results`

**Step 3: Modify the fetcher**

In `src/digest_pipeline/openalex_fetcher.py`, update `fetch_openalex_papers`:

Replace the params construction (lines 119-125):

```python
    # Use fetch_pool size when ranking is configured, otherwise max_results
    pool_size = settings.openalex_fetch_pool if settings.openalex_interest_profile or settings.openalex_interest_keywords else settings.openalex_max_results

    params: dict[str, str | int] = {
        "filter": ",".join(filters),
        "per_page": pool_size,
        "sort": "publication_date:desc",
        "select": "id,title,authorships,abstract_inverted_index,publication_date,doi,open_access,topics",
    }

    # Only use search term when not using interest-based ranking
    if query and not (settings.openalex_interest_profile or settings.openalex_interest_keywords):
        params["search"] = query
```

**Step 4: Run all OpenAlex fetcher tests**

Run: `pytest tests/unit/test_openalex_fetcher.py -v`
Expected: All tests PASS (existing test for `search` param may need updating — see step 5)

**Step 5: Update existing test that asserts search param**

The test `test_fetch_openalex_papers_uses_query_and_filters` currently asserts `"reinforcement learning" in params.get("search", "")`. Update it to verify search IS present when no interest profile is set (which is the default in that test — so it should still pass). Verify this.

**Step 6: Commit**

```bash
git add src/digest_pipeline/openalex_fetcher.py tests/unit/test_openalex_fetcher.py
git commit -m "feat: use fetch pool size and skip search when ranking enabled"
```

---

### Task 7: Wire ranking into the pipeline

**Files:**
- Modify: `src/digest_pipeline/pipeline.py:120-132`

**Step 1: Write the failing test**

Add to `tests/unit/test_pipeline.py` (or create if needed):

```python
from unittest.mock import patch, MagicMock
from digest_pipeline.pipeline import run
from digest_pipeline.config import Settings


@patch("digest_pipeline.pipeline.send_digest")
@patch("digest_pipeline.pipeline.summarize", return_value={})
@patch("digest_pipeline.pipeline.store_chunks")
@patch("digest_pipeline.pipeline.rank_papers")
@patch("digest_pipeline.pipeline.fetch_openalex_papers")
@patch("digest_pipeline.pipeline.fetch_papers", return_value=[])
def test_pipeline_calls_ranker_for_openalex(
    mock_fetch, mock_oa_fetch, mock_rank, mock_store, mock_summarize, mock_send,
    make_paper, make_settings
):
    """rank_papers is called on OpenAlex results before adding to pipeline."""
    oa_paper = make_paper(paper_id="oa_W1", source="openalex")
    mock_oa_fetch.return_value = [oa_paper]
    mock_rank.return_value = [oa_paper]

    settings = make_settings(
        openalex_enabled=True,
        openalex_interest_profile="test",
    )
    run(settings)

    mock_rank.assert_called_once()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_pipeline.py::test_pipeline_calls_ranker_for_openalex -v`
Expected: FAIL — `rank_papers` not imported in pipeline

**Step 3: Modify pipeline.py**

Add import at top of `src/digest_pipeline/pipeline.py`:

```python
from digest_pipeline.ranker import rank_papers
```

Update the OpenAlex section (around line 120-132):

```python
    if settings.openalex_enabled:
        # Build set of known DOIs to avoid duplicating arXiv/HF papers.
        all_known_dois: set[str] = set()
        for p in papers:
            if p.source == "arxiv":
                raw_id = p.paper_id.split("v")[0]  # strip version
                all_known_dois.add(f"10.48550/arXiv.{raw_id}")
            elif p.source == "huggingface":
                raw_id = p.paper_id.removeprefix("hf_").split("v")[0]
                all_known_dois.add(f"10.48550/arXiv.{raw_id}")
        oa_papers = fetch_openalex_papers(settings, known_paper_ids=all_known_dois)
        oa_papers = rank_papers(oa_papers, settings)
        logger.info("Ranked OpenAlex papers: %d selected.", len(oa_papers))
        papers.extend(oa_papers)
```

**Step 4: Run tests**

Run: `pytest tests/unit/ -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/digest_pipeline/pipeline.py tests/unit/test_pipeline.py
git commit -m "feat: wire paper ranker into pipeline after OpenAlex fetch"
```

---

### Task 8: Update setup wizard and .env.example

**Files:**
- Modify: `src/digest_pipeline/setup.py:394-408`
- Modify: `.env.example`

**Step 1: Update setup wizard**

In `_collect_optional_settings()` in `setup.py`, after the existing OpenAlex config block (around line 403), add the ranking prompts inside the `if _prompt_bool("Enable OpenAlex integration?")` block:

```python
        # Interest-based ranking
        if _prompt_bool("Enable interest-based paper ranking?", default=False):
            config["OPENALEX_INTEREST_PROFILE"] = _prompt(
                "Describe your research interests (natural language)"
            )
            keywords = _prompt(
                "Boost keywords (comma-separated, optional)"
            )
            if keywords:
                import json as _json
                kw_list = [k.strip() for k in keywords.split(",") if k.strip()]
                config["OPENALEX_INTEREST_KEYWORDS"] = _json.dumps(kw_list)
            config["OPENALEX_FETCH_POOL"] = _prompt("Papers to fetch before ranking", "100")
```

**Step 2: Update _write_env_file sections**

In the `_write_env_file` function's sections list, add the new keys to the OpenAlex section:

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
                "OPENALEX_INTEREST_PROFILE",
                "OPENALEX_INTEREST_KEYWORDS",
                "OPENALEX_FETCH_POOL",
            ],
        ),
```

**Step 3: Run full test suite**

Run: `pytest tests/unit/ -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add src/digest_pipeline/setup.py .env.example
git commit -m "feat: add ranking config to setup wizard and .env.example"
```

---

### Task 9: Run full test suite and verify

**Step 1: Run all unit tests**

Run: `pytest tests/unit/ -v`
Expected: All tests PASS (should be ~150+ tests)

**Step 2: Run a dry-run pipeline test (if API keys available)**

Run: `cd /home/angel/Documents/code/Daylight-know && python -m digest_pipeline.pipeline --dry-run -v`
Expected: Pipeline runs, OpenAlex papers are fetched, ranked, and output printed

**Step 3: Final commit if any fixups needed**

---

### Task 10: Update production .env with interest profile

**Step 1: Add ranking config to .env**

Add the interest profile and keywords to the production `.env` file based on the user's actual research interests. This is a manual step — ask the user what their interests are.

**Step 2: Verify with dry run**

Run: `cd /home/angel/Documents/code/Daylight-know && python -m digest_pipeline.pipeline --dry-run -v`
Expected: Papers are ranked and only top 20 appear in output
