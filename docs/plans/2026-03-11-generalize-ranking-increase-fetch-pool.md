# Generalize Interest-Based Ranking & Increase Fetch Pools

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generalize the ranking system to work across all paper sources (arXiv + OpenAlex), fetch larger pools, and rank/filter down to the best papers before summarization.

**Architecture:** Promote `openalex_interest_profile` and `openalex_interest_keywords` to pipeline-wide settings (`interest_profile`, `interest_keywords`). Add `arxiv_fetch_pool` to fetch a large arXiv pool then rank down to `arxiv_max_results`. The existing `rank_papers()` function already handles keyword + LLM scoring and batching — it just needs to accept the new config field names. Backward-compatible: old `OPENALEX_*` env vars still work via aliases; `.env` migration handled in one task.

**Tech Stack:** Python 3.11+, pydantic-settings, litellm, pytest, arxiv API

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/digest_pipeline/config.py` | Modify | Add pipeline-wide interest settings, `arxiv_fetch_pool`, bump `llm_max_tokens` default |
| `src/digest_pipeline/ranker.py` | Modify | Use pipeline-wide settings instead of `openalex_*` |
| `src/digest_pipeline/pipeline.py` | Modify | Apply ranking to arXiv papers, use new config fields |
| `src/digest_pipeline/fetcher.py` | Modify | Accept optional `max_results` override for fetch pool |
| `src/digest_pipeline/openalex_fetcher.py` | Modify | Fall back to pipeline-wide interest settings |
| `src/digest_pipeline/setup.py` | Modify | Update wizard to use pipeline-wide interest settings |
| `.env` | Modify | Rename `OPENALEX_INTEREST_*` → `INTEREST_*`, bump limits |
| `tests/unit/test_ranker_config.py` | Modify | Test new config field names + backward compat |
| `tests/unit/test_ranker.py` | Modify | Update to use pipeline-wide settings |
| `tests/unit/test_pipeline.py` | Modify | Add test for arXiv ranking integration |
| `tests/unit/test_openalex_fetcher.py` | Modify | Update to use pipeline-wide interest settings |
| `tests/unit/test_fetcher.py` | Modify | Add test for `max_results` override parameter |

---

## Chunk 1: Config & Ranker Generalization

### Task 1: Add pipeline-wide interest settings to config

**Files:**
- Modify: `src/digest_pipeline/config.py`

The key change: add `interest_profile`, `interest_keywords`, and `arxiv_fetch_pool` as pipeline-wide fields. Keep `openalex_interest_profile` and `openalex_interest_keywords` as deprecated aliases that feed into the pipeline-wide fields. Bump default `llm_max_tokens` to 32768.

- [ ] **Step 1: Write failing tests for new config fields**

Add to `tests/unit/test_ranker_config.py`:

```python
def test_pipeline_wide_interest_config_defaults():
    s = Settings(
        _env_file=None,
        llm_api_key="k",
        smtp_user="u",
        smtp_password="p",
        email_from="a@b",
        email_to="c@d",
    )
    assert s.interest_profile == ""
    assert s.interest_keywords == []
    assert s.arxiv_fetch_pool == 200


def test_pipeline_wide_interest_config_from_values():
    s = Settings(
        _env_file=None,
        llm_api_key="k",
        smtp_user="u",
        smtp_password="p",
        email_from="a@b",
        email_to="c@d",
        interest_profile="AI safety and alignment research",
        interest_keywords=["alignment", "RLHF", "interpretability"],
        arxiv_fetch_pool=150,
    )
    assert s.interest_profile == "AI safety and alignment research"
    assert s.interest_keywords == ["alignment", "RLHF", "interpretability"]
    assert s.arxiv_fetch_pool == 150


def test_openalex_interest_fields_still_exist():
    """Backward compat: openalex-prefixed fields still work for OpenAlex-specific overrides."""
    s = Settings(
        _env_file=None,
        llm_api_key="k",
        smtp_user="u",
        smtp_password="p",
        email_from="a@b",
        email_to="c@d",
        openalex_interest_profile="OpenAlex-specific profile",
        openalex_interest_keywords=["openalex-kw"],
    )
    assert s.openalex_interest_profile == "OpenAlex-specific profile"
    assert s.openalex_interest_keywords == ["openalex-kw"]


def test_llm_max_tokens_default_increased():
    s = Settings(
        _env_file=None,
        llm_api_key="k",
        smtp_user="u",
        smtp_password="p",
        email_from="a@b",
        email_to="c@d",
    )
    assert s.llm_max_tokens == 32768
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_ranker_config.py -v`
Expected: FAIL — `interest_profile`, `interest_keywords`, `arxiv_fetch_pool` don't exist yet.

- [ ] **Step 3: Add new fields to config.py**

In `src/digest_pipeline/config.py`, add under the arXiv section:

```python
    arxiv_fetch_pool: int = Field(default=200)
```

Add a new section after the post-processing block:

```python
    # ── Interest-based ranking (pipeline-wide) ───────────────
    interest_profile: str = Field(default="")
    interest_keywords: list[str] = Field(default_factory=list)
```

Change the `llm_max_tokens` default:

```python
    llm_max_tokens: int = Field(default=32768)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_ranker_config.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/digest_pipeline/config.py tests/unit/test_ranker_config.py
git commit -m "feat: add pipeline-wide interest settings and arxiv_fetch_pool to config"
```

---

### Task 2: Generalize ranker to use pipeline-wide settings

**Files:**
- Modify: `src/digest_pipeline/ranker.py`
- Modify: `tests/unit/test_ranker.py`

The ranker currently reads `settings.openalex_interest_profile`, `settings.openalex_interest_keywords`, and `settings.openalex_max_results`. Change it to accept explicit parameters for profile, keywords, and max_results so it's source-agnostic. The `rank_papers()` function signature changes to accept these directly.

- [ ] **Step 1: Update test fixtures to use pipeline-wide settings**

In `tests/unit/test_ranker.py`, update all tests that use `openalex_interest_profile` / `openalex_interest_keywords` / `openalex_max_results` to pass these as explicit arguments to `rank_papers()`. The new signature will be:

```python
def rank_papers(
    papers: list[Paper],
    settings: Settings,
    *,
    interest_profile: str = "",
    interest_keywords: list[str] | None = None,
    max_results: int = 20,
) -> list[Paper]:
```

Update tests:

```python
@patch("digest_pipeline.ranker.score_batch_with_llm")
def test_rank_papers_combines_scores(mock_llm_score, make_paper, make_settings):
    """Papers are ranked by combined keyword + LLM score."""
    papers = [
        make_paper(paper_id="low", title="Unrelated Topic", abstract="Nothing relevant here"),
        make_paper(
            paper_id="high", title="LLM for Drug Discovery", abstract="We use LLM to find drugs"
        ),
    ]
    mock_llm_score.return_value = [5, 5]

    settings = make_settings(
        interest_profile="LLMs for drug discovery",
        interest_keywords=["LLM", "drug"],
    )
    ranked = rank_papers(papers, settings, max_results=1)
    assert len(ranked) == 1
    assert ranked[0].paper_id == "high"


def test_rank_papers_no_profile_returns_unchanged(make_paper, make_settings):
    """When no interest profile or keywords, return papers unchanged."""
    papers = [make_paper(paper_id="a"), make_paper(paper_id="b")]
    settings = make_settings()
    ranked = rank_papers(papers, settings, max_results=20)
    assert len(ranked) == 2
    assert ranked[0].paper_id == "a"


@patch("digest_pipeline.ranker.score_batch_with_llm")
def test_rank_papers_batches_correctly(mock_llm_score, make_paper, make_settings):
    """Papers are scored in batches of BATCH_SIZE."""
    papers = [make_paper(paper_id=f"p{i}") for i in range(25)]
    mock_llm_score.side_effect = [
        [5] * 20,
        [5] * 5,
    ]
    settings = make_settings(
        interest_profile="test",
    )
    ranked = rank_papers(papers, settings, max_results=10)
    assert len(ranked) == 10
    assert mock_llm_score.call_count == 2
```

Also update `score_batch_with_llm` tests to pass `interest_profile` on settings (the pipeline-wide field) since that function still reads from settings:

```python
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
    settings = make_settings(interest_profile="I study LLMs for drug discovery")
    scores = score_batch_with_llm(papers, settings)
    assert scores == [8, 3]


@patch("digest_pipeline.ranker.litellm.completion")
def test_llm_batch_scoring_handles_failure(mock_completion, make_paper, make_settings):
    mock_completion.side_effect = Exception("API error")
    papers = [make_paper(), make_paper()]
    settings = make_settings(interest_profile="anything")
    scores = score_batch_with_llm(papers, settings)
    assert scores == [0, 0]


@patch("digest_pipeline.ranker.litellm.completion")
def test_llm_batch_scoring_handles_invalid_json(mock_completion, make_paper, make_settings):
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = "not json"
    mock_completion.return_value = mock_resp
    papers = [make_paper()]
    settings = make_settings(interest_profile="anything")
    scores = score_batch_with_llm(papers, settings)
    assert scores == [0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_ranker.py -v`
Expected: FAIL — `rank_papers()` doesn't accept `max_results` kwarg yet.

- [ ] **Step 3: Update ranker.py to use pipeline-wide settings**

Replace the `rank_papers` function in `src/digest_pipeline/ranker.py`:

```python
def rank_papers(
    papers: list[Paper],
    settings: Settings,
    *,
    interest_profile: str = "",
    interest_keywords: list[str] | None = None,
    max_results: int = 20,
) -> list[Paper]:
    """Rank papers by relevance and return the top N.

    Combines keyword boost scores with LLM-based relevance scoring.
    Uses explicit parameters if provided, otherwise falls back to
    pipeline-wide settings. If no interest profile or keywords are
    configured, returns papers unchanged.
    """
    profile = interest_profile or settings.interest_profile
    keywords = interest_keywords if interest_keywords is not None else settings.interest_keywords

    if not profile and not keywords:
        return papers

    if len(papers) <= max_results:
        return papers

    # 1. Keyword scores
    keyword_scores = compute_keyword_scores(papers, keywords)

    # 2. LLM scores (in batches)
    llm_scores = [0] * len(papers)
    if profile:
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
    combined.sort(key=lambda x: (-x[0], x[1]))

    ranked = [paper for _, _, paper in combined[:max_results]]
    logger.info(
        "Ranked %d papers -> top %d (score range: %d-%d).",
        len(papers),
        len(ranked),
        combined[-1][0],
        combined[0][0],
    )
    return ranked
```

Also update `score_batch_with_llm` to accept an explicit `interest_profile` parameter so the resolved profile is forwarded from `rank_papers`:

```python
def score_batch_with_llm(papers: list[Paper], settings: Settings, *, interest_profile: str = "") -> list[int]:
    """Score a batch of papers using the LLM.

    Sends paper titles and abstracts to the configured LLM with the
    interest profile. Returns a list of integer scores (1-10) per paper.
    On failure, returns zeros for graceful degradation.
    """
    if not papers:
        return []

    profile = interest_profile or settings.interest_profile or settings.openalex_interest_profile
    system_prompt = load_prompt("ranker").replace(
        "{interest_profile}", profile
    )
    # ... rest of function unchanged
```

And update the call site in `rank_papers` to forward the resolved profile:

```python
    if profile:
        for start in range(0, len(papers), BATCH_SIZE):
            batch = papers[start : start + BATCH_SIZE]
            batch_scores = score_batch_with_llm(batch, settings, interest_profile=profile)
            for j, score in enumerate(batch_scores):
                llm_scores[start + j] = score
```

Update the `score_batch_with_llm` tests to pass `interest_profile` as a kwarg:

```python
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
    settings = make_settings()
    scores = score_batch_with_llm(papers, settings, interest_profile="I study LLMs for drug discovery")
    assert scores == [8, 3]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_ranker.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/digest_pipeline/ranker.py tests/unit/test_ranker.py
git commit -m "refactor: generalize ranker to use pipeline-wide interest settings"
```

---

## Chunk 2: Pipeline Integration & .env Migration

### Task 3: Apply ranking to arXiv papers in pipeline

**Files:**
- Modify: `src/digest_pipeline/pipeline.py`
- Modify: `tests/unit/test_pipeline.py`

The pipeline currently only ranks OpenAlex papers. Apply `rank_papers()` to arXiv papers when `interest_profile` or `interest_keywords` are set and `arxiv_fetch_pool > arxiv_max_results`.

- [ ] **Step 1: Write failing test for arXiv ranking in pipeline**

Add to `tests/unit/test_pipeline.py`:

```python
@patch("digest_pipeline.pipeline.rank_papers")
@patch("digest_pipeline.pipeline.fetch_papers")
@patch("digest_pipeline.pipeline.summarize", return_value={"paper_1": "summary", "paper_2": "summary"})
@patch("digest_pipeline.pipeline.extract_implications", return_value={})
@patch("digest_pipeline.pipeline.generate_critiques", return_value={})
@patch("digest_pipeline.pipeline.chunk_text", return_value=["chunk1"])
@patch("digest_pipeline.pipeline.store_chunks")
@patch("digest_pipeline.pipeline.send_digest")
@patch("digest_pipeline.pipeline.load_seen", return_value={})
@patch("digest_pipeline.pipeline.filter_unseen", side_effect=lambda papers, seen: papers)
@patch("digest_pipeline.pipeline.record_papers")
@patch("digest_pipeline.pipeline.save_seen")
def test_arxiv_papers_ranked_when_interest_configured(
    mock_save, mock_record, mock_filter, mock_load_seen,
    mock_send, mock_store, mock_chunk, mock_critiques, mock_impl,
    mock_summarize, mock_fetch, mock_rank, make_paper, make_settings
):
    """arXiv papers should be ranked when interest profile is set."""
    arxiv_papers = [make_paper(paper_id=f"p{i}", abstract=f"abstract {i}") for i in range(5)]
    mock_fetch.return_value = arxiv_papers
    mock_rank.return_value = arxiv_papers[:2]  # ranked down to 2

    settings = make_settings(
        interest_profile="AI safety research",
        interest_keywords=["alignment"],
        arxiv_max_results=2,
        arxiv_fetch_pool=50,
        postprocessing_implications=False,
        postprocessing_critiques=False,
    )
    run(settings)

    mock_rank.assert_called_once()
    call_args = mock_rank.call_args
    assert len(call_args[0][0]) == 5  # all papers passed to ranker
    assert call_args[1]["max_results"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_pipeline.py::test_arxiv_papers_ranked_when_interest_configured -v`
Expected: FAIL — pipeline doesn't call rank_papers for arXiv yet.

- [ ] **Step 3: Update pipeline.py to rank arXiv papers**

In `src/digest_pipeline/pipeline.py`, after `papers = fetch_papers(settings)` (line 107) and before the HuggingFace section, add:

```python
    # ── Rank arXiv papers if interest-based ranking is configured ──
    if (settings.interest_profile or settings.interest_keywords) and len(papers) > settings.arxiv_max_results:
        papers = rank_papers(
            papers,
            settings,
            max_results=settings.arxiv_max_results,
        )
        logger.info("arXiv: %d papers after ranking.", len(papers))
```

Also update the OpenAlex ranking call to pass `max_results` explicitly:

```python
        oa_papers = rank_papers(
            oa_papers,
            settings,
            interest_profile=settings.openalex_interest_profile or settings.interest_profile,
            interest_keywords=settings.openalex_interest_keywords or settings.interest_keywords,
            max_results=settings.openalex_max_results,
        )
```

And update the `arxiv_max_results` in fetcher usage — the fetcher should now use `arxiv_fetch_pool` when ranking is enabled. In `pipeline.py`, before calling `fetch_papers`, temporarily override the setting:

```python
    # If ranking is enabled, fetch the larger pool
    if settings.interest_profile or settings.interest_keywords:
        original_max = settings.arxiv_max_results
        settings.arxiv_max_results = settings.arxiv_fetch_pool
```

Wait — pydantic Settings are frozen. Instead, update `fetch_papers` in `fetcher.py` to accept an optional override, OR just read `arxiv_fetch_pool` directly. Simpler approach: have the pipeline pass the fetch pool size. Update `fetch_papers()` to accept an optional `max_results` override:

In `src/digest_pipeline/fetcher.py`, change the signature and update both usage sites:

```python
def fetch_papers(settings: Settings, *, max_results: int | None = None) -> list[Paper]:
    """Query arXiv for recent papers matching the configured topics."""
    effective_max = max_results or settings.arxiv_max_results
    query = " OR ".join(f"cat:{topic}" for topic in settings.arxiv_topics)
    logger.info("Querying arXiv: %s (max %d)", query, effective_max)

    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=effective_max,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    # ... rest of function unchanged
```

Add a test for the new parameter in `tests/unit/test_fetcher.py`:

```python
def test_fetch_papers_respects_max_results_override(make_settings):
    """fetch_papers should use max_results kwarg when provided."""
    from unittest.mock import patch, MagicMock
    settings = make_settings(arxiv_max_results=10)
    with patch("digest_pipeline.fetcher.arxiv.Client") as mock_client:
        mock_client.return_value.results.return_value = iter([])
        from digest_pipeline.fetcher import fetch_papers
        fetch_papers(settings, max_results=200)
        # Verify the Search was created with 200, not 10
        search_call = mock_client.return_value.results.call_args[0][0]
        assert search_call.max_results == 200
```

In `pipeline.py`:

```python
    # Determine arXiv fetch size: use larger pool if ranking is enabled
    arxiv_fetch_size = settings.arxiv_max_results
    if settings.interest_profile or settings.interest_keywords:
        arxiv_fetch_size = settings.arxiv_fetch_pool

    papers = fetch_papers(settings, max_results=arxiv_fetch_size)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_pipeline.py -v && pytest tests/unit/test_fetcher.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/digest_pipeline/pipeline.py src/digest_pipeline/fetcher.py tests/unit/test_pipeline.py
git commit -m "feat: apply interest-based ranking to arXiv papers"
```

---

### Task 4: Update openalex_fetcher to use pipeline-wide settings

**Files:**
- Modify: `src/digest_pipeline/openalex_fetcher.py:119-121`
- Modify: `tests/unit/test_openalex_fetcher.py`

The `openalex_fetcher.py` checks `settings.openalex_interest_profile` and `settings.openalex_interest_keywords` to decide whether to use `fetch_pool` or `max_results`. After the `.env` migration removes the `OPENALEX_INTEREST_*` vars, this breaks silently. Update to fall back to pipeline-wide settings.

- [ ] **Step 1: Update openalex_fetcher.py to fall back to pipeline-wide settings**

In `src/digest_pipeline/openalex_fetcher.py`, change lines 119-121:

```python
    # Use fetch_pool size when ranking is configured, otherwise max_results
    has_ranking = bool(
        settings.openalex_interest_profile or settings.openalex_interest_keywords
        or settings.interest_profile or settings.interest_keywords
    )
    pool_size = settings.openalex_fetch_pool if has_ranking else settings.openalex_max_results
```

- [ ] **Step 2: Update test_openalex_fetcher.py tests that set interest_profile**

Any test passing `openalex_interest_profile` to `make_settings` should also work with `interest_profile`. Verify existing tests still pass — the fallback chain means old field names still work.

- [ ] **Step 3: Run tests**

Run: `pytest tests/unit/test_openalex_fetcher.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add src/digest_pipeline/openalex_fetcher.py tests/unit/test_openalex_fetcher.py
git commit -m "fix: openalex_fetcher falls back to pipeline-wide interest settings"
```

---

### Task 5: Update .env (do NOT commit) and setup wizard

**Files:**
- Modify: `.env`
- Modify: `src/digest_pipeline/setup.py`

- [ ] **Step 1: Update .env to use pipeline-wide interest settings**

Add new pipeline-wide fields and increase limits. Keep `OPENALEX_INTEREST_*` for backward compat but add the pipeline-wide versions:

```env
# ── Interest-Based Ranking (pipeline-wide) ───────────────
INTEREST_PROFILE="AI applications including world models, frontier AI methods, memory and retrieval systems, RAG, graph-based reasoning, and knowledge graphs. Also highly interested in cybersecurity research, especially AI-driven security, threat detection, and adversarial machine learning."
INTEREST_KEYWORDS=["world model","frontier AI","memory","retrieval","RAG","graph neural network","knowledge graph","cybersecurity","LLM","reasoning","agent"]

# ── arXiv Settings ──────────────────────────────────────────
ARXIV_TOPICS=["cs.AI","cs.LG","cs.MA","cs.CL","stat.ML","cs.NE","cs.RO"]
ARXIV_MAX_RESULTS=20
ARXIV_FETCH_POOL=200

# ── LLM Settings (via litellm) ──────────────────────────────
LLM_MAX_TOKENS=32768
```

Remove `OPENALEX_INTEREST_PROFILE`, `OPENALEX_INTEREST_KEYWORDS`, and `OPENALEX_FETCH_POOL` from `.env` since they're now pipeline-wide. Keep `OPENALEX_MAX_RESULTS=20` as the per-source output cap.

- [ ] **Step 2: Update setup wizard to use pipeline-wide settings**

In `src/digest_pipeline/setup.py`, find the interest-based ranking section (around line 405) and change:
- `OPENALEX_INTEREST_PROFILE` → `INTEREST_PROFILE`
- `OPENALEX_INTEREST_KEYWORDS` → `INTEREST_KEYWORDS`
- `OPENALEX_FETCH_POOL` → remove (fetch pool is now per-source in config defaults)
- Add `ARXIV_FETCH_POOL` prompt: "arXiv papers to fetch before ranking (default 200)"

Also update the env keys list (around line 544) to include the new field names.

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/unit/ -v`
Expected: All PASS

- [ ] **Step 4: Commit (setup.py only — .env contains secrets, do NOT commit)**

```bash
git add src/digest_pipeline/setup.py
git commit -m "chore: migrate setup wizard to pipeline-wide interest settings"
```

---

### Task 6: Run lint and dry-run validation

- [ ] **Step 1: Lint**

Run: `ruff check src/ tests/ --fix && ruff format src/ tests/`

- [ ] **Step 2: Run full test suite**

Run: `pytest tests/unit/ -v`
Expected: All PASS

- [ ] **Step 3: Dry-run pipeline**

Run: `python -m digest_pipeline --dry-run -v`
Expected: Pipeline fetches ~200 arXiv papers, ranks to 20, fetches ~100 OpenAlex, ranks to 20. Total ~40 papers through summarization in batches of 7.

- [ ] **Step 4: Final commit if any lint fixes**

```bash
git add -u
git commit -m "chore: lint and format after ranking generalization"
```

---

## Summary of Config Changes

| Setting | Before | After |
|---------|--------|-------|
| `INTEREST_PROFILE` | _(didn't exist)_ | Pipeline-wide interest profile |
| `INTEREST_KEYWORDS` | _(didn't exist)_ | Pipeline-wide keyword list |
| `ARXIV_FETCH_POOL` | _(didn't exist)_ | 200 (fetch pool before ranking) |
| `ARXIV_MAX_RESULTS` | 50 default / 20 in .env (also fetch limit) | 20 in .env (output cap after ranking) |
| `LLM_MAX_TOKENS` | 16384 | 32768 |
| `OPENALEX_INTEREST_PROFILE` | Primary config | Kept for backward compat / per-source override |
| `OPENALEX_INTEREST_KEYWORDS` | Primary config | Kept for backward compat / per-source override |
| `OPENALEX_FETCH_POOL` | 100 | 100 (unchanged, per-source) |
